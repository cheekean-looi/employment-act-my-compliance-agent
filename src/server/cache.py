#!/usr/bin/env python3
"""
Two-tier caching system: in-memory LRU + Redis backend.
Production-grade caching with deterministic cache keys and observability.
"""

import json
import hashlib
import time
import logging
from typing import Any, Optional, Dict, List, Protocol
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
from cachetools import TTLCache
import redis.asyncio as redis
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""
    value: Any
    created_at: float
    ttl: int
    hit_count: int = 0
    size_bytes: Optional[int] = None


class CacheBackend(Protocol):
    """Cache backend interface."""
    
    async def get(self, key: str) -> Optional[Any]:
        """Retrieve value by key."""
        ...
    
    async def set(self, key: str, value: Any, ttl: int = 3600) -> bool:
        """Store value with TTL."""
        ...
    
    async def delete(self, key: str) -> bool:
        """Delete key."""
        ...
    
    async def exists(self, key: str) -> bool:
        """Check if key exists."""
        ...
    
    async def clear(self) -> bool:
        """Clear all cached data."""
        ...
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        ...


class InMemoryCache(CacheBackend):
    """In-memory LRU cache with per-key TTL support."""
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 1800):
        """Initialize in-memory cache."""
        # Use regular LRU cache and manage TTL manually for per-key support
        from cachetools import LRUCache
        self.cache = LRUCache(maxsize=max_size)
        self.expiry_times = {}  # key -> expiry timestamp
        self.default_ttl = default_ttl
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        
        logger.info(f"Initialized in-memory cache with max_size={max_size}, ttl={default_ttl}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from memory cache with TTL check."""
        try:
            # First check if key has expired
            if key in self.expiry_times:
                if time.time() > self.expiry_times[key]:
                    # Expired, remove from both caches
                    self.cache.pop(key, None)
                    self.expiry_times.pop(key, None)
                    self.misses += 1
                    return None
            
            entry = self.cache.get(key)
            if entry is not None:
                entry.hit_count += 1
                self.hits += 1
                return entry.value
            else:
                self.misses += 1
                return None
        except Exception as e:
            logger.warning(f"Memory cache get error: {e}")
            self.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: int = None) -> bool:
        """Set value in memory cache with per-key TTL."""
        try:
            if ttl is None:
                ttl = self.default_ttl
                
            entry = CacheEntry(
                value=value,
                created_at=time.time(),
                ttl=ttl,
                size_bytes=len(json.dumps(value, default=str)) if value else 0
            )
            
            # Store in cache and set expiry time
            self.cache[key] = entry
            self.expiry_times[key] = time.time() + ttl
            self.sets += 1
            return True
        except Exception as e:
            logger.warning(f"Memory cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete from memory cache."""
        try:
            if key in self.cache:
                del self.cache[key]
                self.deletes += 1
                return True
            return False
        except Exception as e:
            logger.warning(f"Memory cache delete error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in memory cache."""
        return key in self.cache
    
    async def clear(self) -> bool:
        """Clear memory cache."""
        try:
            self.cache.clear()
            return True
        except Exception as e:
            logger.error(f"Memory cache clear error: {e}")
            return False
    
    def _cleanup_expired(self):
        """Remove expired keys from cache."""
        current_time = time.time()
        expired_keys = [
            key for key, expiry_time in self.expiry_times.items()
            if current_time > expiry_time
        ]
        for key in expired_keys:
            self.cache.pop(key, None)
            self.expiry_times.pop(key, None)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get memory cache statistics."""
        # Clean up expired keys before reporting stats
        self._cleanup_expired()
        
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "type": "memory",
            "size": len(self.cache),
            "max_size": self.cache.maxsize,
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "sets": self.sets,
            "deletes": self.deletes,
            "expired_keys_count": len([
                key for key, expiry_time in self.expiry_times.items()
                if time.time() > expiry_time
            ])
        }


class RedisCache(CacheBackend):
    """Redis-backed cache for distributed caching."""
    
    def __init__(self, redis_url: str, key_prefix: str = "employment_act"):
        """Initialize Redis cache."""
        self.redis_url = redis_url
        self.key_prefix = key_prefix
        self.redis_client = None
        self.hits = 0
        self.misses = 0
        self.sets = 0
        self.deletes = 0
        
        logger.info(f"Initialized Redis cache with URL: {redis_url}")
    
    async def _get_client(self) -> redis.Redis:
        """Get Redis client, creating if needed."""
        if self.redis_client is None:
            self.redis_client = redis.from_url(
                self.redis_url,
                encoding="utf-8",
                decode_responses=True,
                socket_timeout=5.0,
                socket_connect_timeout=5.0
            )
        return self.redis_client
    
    def _make_key(self, key: str) -> str:
        """Create prefixed key."""
        return f"{self.key_prefix}:{key}"
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from Redis."""
        try:
            client = await self._get_client()
            raw_value = await client.get(self._make_key(key))
            
            if raw_value is not None:
                self.hits += 1
                # Deserialize JSON
                return json.loads(raw_value)
            else:
                self.misses += 1
                return None
                
        except Exception as e:
            logger.warning(f"Redis cache get error: {e}")
            self.misses += 1
            return None
    
    async def set(self, key: str, value: Any, ttl: int = 1800) -> bool:
        """Set value in Redis with TTL."""
        try:
            client = await self._get_client()
            serialized = json.dumps(value, default=str)
            
            await client.setex(
                self._make_key(key),
                ttl,
                serialized
            )
            
            self.sets += 1
            return True
            
        except Exception as e:
            logger.warning(f"Redis cache set error: {e}")
            return False
    
    async def delete(self, key: str) -> bool:
        """Delete from Redis."""
        try:
            client = await self._get_client()
            result = await client.delete(self._make_key(key))
            
            if result > 0:
                self.deletes += 1
                return True
            return False
            
        except Exception as e:
            logger.warning(f"Redis cache delete error: {e}")
            return False
    
    async def exists(self, key: str) -> bool:
        """Check if key exists in Redis."""
        try:
            client = await self._get_client()
            return await client.exists(self._make_key(key)) > 0
        except Exception as e:
            logger.warning(f"Redis cache exists error: {e}")
            return False
    
    async def clear(self) -> bool:
        """Clear all keys with prefix."""
        try:
            client = await self._get_client()
            pattern = f"{self.key_prefix}:*"
            
            # Use scan to avoid blocking
            keys = []
            async for key in client.scan_iter(match=pattern):
                keys.append(key)
            
            if keys:
                await client.delete(*keys)
            
            return True
            
        except Exception as e:
            logger.error(f"Redis cache clear error: {e}")
            return False
    
    async def health_check(self) -> bool:
        """Check Redis connectivity."""
        try:
            client = await self._get_client()
            await client.ping()
            return True
        except Exception as e:
            logger.warning(f"Redis health check failed: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get Redis cache statistics."""
        total_requests = self.hits + self.misses
        hit_rate = self.hits / total_requests if total_requests > 0 else 0
        
        return {
            "type": "redis",
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "sets": self.sets,
            "deletes": self.deletes,
            "url": self.redis_url
        }
    
    async def close(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()


class TieredCache:
    """
    Two-tier cache: L1 (memory) + L2 (Redis).
    Provides high-performance caching with fallback to distributed cache.
    """
    
    def __init__(self, 
                 memory_cache: InMemoryCache,
                 redis_cache: Optional[RedisCache] = None):
        """Initialize tiered cache."""
        self.l1_cache = memory_cache
        self.l2_cache = redis_cache
        self.total_hits = 0
        self.total_misses = 0
        
        cache_tiers = "memory"
        if redis_cache:
            cache_tiers += " + redis"
        
        logger.info(f"Initialized tiered cache: {cache_tiers}")
    
    async def get(self, key: str) -> Optional[Any]:
        """Get from L1, then L2 if miss."""
        # Try L1 first
        value = await self.l1_cache.get(key)
        if value is not None:
            self.total_hits += 1
            return value
        
        # Try L2 if available
        if self.l2_cache:
            value = await self.l2_cache.get(key)
            if value is not None:
                # Backfill L1 cache
                await self.l1_cache.set(key, value)
                self.total_hits += 1
                return value
        
        self.total_misses += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: int = 1800) -> bool:
        """Set in both L1 and L2."""
        results = []
        
        # Set in L1
        results.append(await self.l1_cache.set(key, value, ttl))
        
        # Set in L2 if available
        if self.l2_cache:
            results.append(await self.l2_cache.set(key, value, ttl))
        
        return any(results)
    
    async def delete(self, key: str) -> bool:
        """Delete from both tiers."""
        results = []
        
        results.append(await self.l1_cache.delete(key))
        
        if self.l2_cache:
            results.append(await self.l2_cache.delete(key))
        
        return any(results)
    
    async def exists(self, key: str) -> bool:
        """Check existence in either tier."""
        if await self.l1_cache.exists(key):
            return True
        
        if self.l2_cache:
            return await self.l2_cache.exists(key)
        
        return False
    
    async def clear(self) -> bool:
        """Clear both tiers."""
        results = []
        
        results.append(await self.l1_cache.clear())
        
        if self.l2_cache:
            results.append(await self.l2_cache.clear())
        
        return all(results)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get combined statistics."""
        total_requests = self.total_hits + self.total_misses
        overall_hit_rate = self.total_hits / total_requests if total_requests > 0 else 0
        
        stats = {
            "overall_hit_rate": overall_hit_rate,
            "total_hits": self.total_hits,
            "total_misses": self.total_misses,
            "l1_stats": self.l1_cache.get_stats()
        }
        
        if self.l2_cache:
            stats["l2_stats"] = self.l2_cache.get_stats()
        
        return stats
    
    async def health_check(self) -> Dict[str, bool]:
        """Check health of both cache tiers."""
        health = {
            "l1_memory": True  # Memory cache is always healthy
        }
        
        if self.l2_cache:
            health["l2_redis"] = await self.l2_cache.health_check()
        
        return health


class CacheKeyBuilder:
    """Builds deterministic cache keys for RAG responses."""
    
    @staticmethod
    def build_query_key(query: str, 
                       retrieval_context_ids: List[str],
                       config_hash: str,
                       guardrails_version: str) -> str:
        """
        Build cache key for query + context + config.
        
        Args:
            query: User query (normalized)
            retrieval_context_ids: Sorted list of retrieved document IDs
            config_hash: Hash of RAG configuration
            guardrails_version: Guardrails config version
            
        Returns:
            Deterministic cache key
        """
        # Normalize query
        normalized_query = query.lower().strip()
        
        # Sort context IDs for deterministic key
        sorted_context = sorted(retrieval_context_ids)
        
        # Create deterministic input
        key_input = {
            "query": normalized_query,
            "context_ids": sorted_context,
            "config_hash": config_hash,
            "guardrails_version": guardrails_version
        }
        
        # Generate stable hash
        key_str = json.dumps(key_input, sort_keys=True)
        key_hash = hashlib.sha256(key_str.encode()).hexdigest()
        
        return f"query:{key_hash[:16]}"
    
    @staticmethod
    def build_section_key(section_id: str) -> str:
        """Build cache key for section lookup."""
        return f"section:{section_id}"
    
    @staticmethod
    def build_severance_key(monthly_wage: float,
                          years_of_service: float,
                          termination_reason: str,
                          annual_leave_days: Optional[int] = None) -> str:
        """Build cache key for severance calculation."""
        key_input = {
            "monthly_wage": monthly_wage,
            "years_of_service": years_of_service,
            "termination_reason": termination_reason,
            "annual_leave_days": annual_leave_days
        }
        
        key_str = json.dumps(key_input, sort_keys=True)
        key_hash = hashlib.sha256(key_str.encode()).hexdigest()
        
        return f"severance:{key_hash[:16]}"


# Factory functions for dependency injection
def create_memory_cache(max_size: int = 1000, ttl: int = 1800) -> InMemoryCache:
    """Create in-memory cache."""
    return InMemoryCache(max_size=max_size, default_ttl=ttl)


def create_redis_cache(redis_url: str, key_prefix: str = "employment_act") -> RedisCache:
    """Create Redis cache."""
    return RedisCache(redis_url=redis_url, key_prefix=key_prefix)


def create_tiered_cache(memory_max_size: int = 1000,
                       memory_ttl: int = 1800,
                       redis_url: Optional[str] = None) -> TieredCache:
    """Create tiered cache with optional Redis backend."""
    memory_cache = create_memory_cache(memory_max_size, memory_ttl)
    
    redis_cache = None
    if redis_url:
        redis_cache = create_redis_cache(redis_url)
    
    return TieredCache(memory_cache, redis_cache)


# Test function
async def test_cache_system():
    """Test cache system functionality."""
    print("Testing cache system...")
    
    # Test memory cache
    memory_cache = create_memory_cache(max_size=10, ttl=5)
    
    await memory_cache.set("test_key", {"message": "hello"})
    value = await memory_cache.get("test_key")
    print(f"Memory cache: {value}")
    
    # Test cache key builder
    key = CacheKeyBuilder.build_query_key(
        "What is sick leave?",
        ["doc1", "doc2"],
        "config123",
        "v1.0.0"
    )
    print(f"Generated key: {key}")
    
    # Test tiered cache
    tiered = create_tiered_cache(memory_max_size=5)
    await tiered.set("tiered_key", {"data": "test"})
    result = await tiered.get("tiered_key")
    print(f"Tiered cache: {result}")
    
    # Print stats
    stats = tiered.get_stats()
    print(f"Cache stats: {stats}")


if __name__ == "__main__":
    import asyncio
    asyncio.run(test_cache_system())