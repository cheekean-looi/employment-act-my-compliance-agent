#!/usr/bin/env python3
"""
Embedding and reranker score caching for Employment Act Malaysia compliance agent.
Implements L1 memory + L2 Redis caching for embeddings and cross-encoder scores.
"""

import hashlib
import json
import logging
import time
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

# Import existing cache infrastructure
from .cache import InMemoryCache, RedisCache, TieredCache

logger = logging.getLogger(__name__)


@dataclass
class EmbeddingCacheEntry:
    """Cache entry for embeddings."""
    embedding: np.ndarray
    model_name: str
    created_at: float
    hit_count: int = 0


@dataclass
class RerankerCacheEntry:
    """Cache entry for reranker scores."""
    score: float
    model_name: str
    created_at: float
    hit_count: int = 0


class EmbeddingCache:
    """
    Cache for query embeddings to avoid recomputation.
    Uses L1 memory + optional L2 Redis with 2h TTL.
    """
    
    def __init__(self, 
                 memory_max_size: int = 1000,
                 redis_url: Optional[str] = None,
                 ttl_seconds: int = 7200):  # 2 hours
        """
        Initialize embedding cache.
        
        Args:
            memory_max_size: Maximum entries in L1 memory cache
            redis_url: Redis URL for L2 cache (optional)
            ttl_seconds: Time-to-live in seconds (default 2 hours)
        """
        self.ttl_seconds = ttl_seconds
        
        # L1 memory cache
        self.l1_cache = InMemoryCache(max_size=memory_max_size, default_ttl=ttl_seconds)
        
        # L2 Redis cache (optional)
        self.l2_cache = None
        if redis_url:
            self.l2_cache = RedisCache(redis_url, key_prefix="embeddings")
        
        # Statistics
        self.l1_hits = 0
        self.l2_hits = 0
        self.misses = 0
        
        logger.info(f"Embedding cache initialized: L1({memory_max_size}) + L2({'Redis' if redis_url else 'None'})")
    
    def _normalize_query(self, query: str) -> str:
        """Normalize query for consistent caching."""
        return query.lower().strip()
    
    def _create_cache_key(self, query: str, model_name: str) -> str:
        """Create deterministic cache key for query embedding."""
        normalized_query = self._normalize_query(query)
        key_input = f"{model_name}:{normalized_query}"
        key_hash = hashlib.sha256(key_input.encode()).hexdigest()[:16]
        return f"embed:{key_hash}"
    
    async def get_embedding(self, query: str, model_name: str) -> Optional[np.ndarray]:
        """
        Get cached embedding for query.
        
        Args:
            query: Query text
            model_name: Embedding model name
            
        Returns:
            Cached embedding array or None if not found
        """
        cache_key = self._create_cache_key(query, model_name)
        
        # Try L1 first
        entry = await self.l1_cache.get(cache_key)
        if entry and isinstance(entry, dict):
            try:
                embedding = np.array(entry['embedding'])
                self.l1_hits += 1
                logger.debug(f"L1 embedding cache hit for query: {query[:30]}...")
                return embedding
            except (KeyError, TypeError) as e:
                logger.warning(f"Invalid L1 embedding entry: {e}")
        
        # Try L2 if available
        if self.l2_cache:
            entry_data = await self.l2_cache.get(cache_key)
            if entry_data and isinstance(entry_data, dict):
                try:
                    embedding = np.array(entry_data['embedding'])
                    
                    # Backfill L1
                    await self.l1_cache.set(cache_key, entry_data, ttl=self.ttl_seconds)
                    
                    self.l2_hits += 1
                    logger.debug(f"L2 embedding cache hit for query: {query[:30]}...")
                    return embedding
                except (KeyError, TypeError) as e:
                    logger.warning(f"Invalid L2 embedding entry: {e}")
        
        self.misses += 1
        return None
    
    async def set_embedding(self, query: str, model_name: str, embedding: np.ndarray) -> bool:
        """
        Cache embedding for query.
        
        Args:
            query: Query text
            model_name: Embedding model name
            embedding: Embedding array
            
        Returns:
            True if successfully cached
        """
        cache_key = self._create_cache_key(query, model_name)
        
        # Prepare cache entry
        entry_data = {
            'embedding': embedding.tolist(),  # Convert to JSON-serializable
            'model_name': model_name,
            'created_at': time.time(),
            'hit_count': 0
        }
        
        results = []
        
        # Store in L1
        results.append(await self.l1_cache.set(cache_key, entry_data, ttl=self.ttl_seconds))
        
        # Store in L2 if available
        if self.l2_cache:
            results.append(await self.l2_cache.set(cache_key, entry_data, ttl=self.ttl_seconds))
        
        success = any(results)
        if success:
            logger.debug(f"Cached embedding for query: {query[:30]}...")
        
        return success
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.l1_hits + self.l2_hits + self.misses
        hit_rate = (self.l1_hits + self.l2_hits) / total_requests if total_requests > 0 else 0
        
        stats = {
            "type": "embedding_cache",
            "l1_hits": self.l1_hits,
            "l2_hits": self.l2_hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "l1_stats": self.l1_cache.get_stats()
        }
        
        if self.l2_cache:
            stats["l2_stats"] = self.l2_cache.get_stats()
        
        return stats


class RerankerScoreCache:
    """
    Cache for cross-encoder scores to avoid recomputation.
    Uses L1 memory + optional L2 Redis with 2h TTL.
    """
    
    def __init__(self, 
                 memory_max_size: int = 5000,  # More entries for query-chunk pairs
                 redis_url: Optional[str] = None,
                 ttl_seconds: int = 7200):  # 2 hours
        """
        Initialize reranker score cache.
        
        Args:
            memory_max_size: Maximum entries in L1 memory cache
            redis_url: Redis URL for L2 cache (optional)
            ttl_seconds: Time-to-live in seconds (default 2 hours)
        """
        self.ttl_seconds = ttl_seconds
        
        # L1 memory cache
        self.l1_cache = InMemoryCache(max_size=memory_max_size, default_ttl=ttl_seconds)
        
        # L2 Redis cache (optional)
        self.l2_cache = None
        if redis_url:
            self.l2_cache = RedisCache(redis_url, key_prefix="reranker_scores")
        
        # Statistics
        self.l1_hits = 0
        self.l2_hits = 0
        self.misses = 0
        
        logger.info(f"Reranker cache initialized: L1({memory_max_size}) + L2({'Redis' if redis_url else 'None'})")
    
    def _create_cache_key(self, query: str, chunk_id: str, model_name: str) -> str:
        """Create deterministic cache key for reranker score."""
        # Normalize inputs
        normalized_query = query.lower().strip()
        key_input = f"{model_name}:{normalized_query}:{chunk_id}"
        key_hash = hashlib.sha256(key_input.encode()).hexdigest()[:16]
        return f"rerank:{key_hash}"
    
    async def get_score(self, query: str, chunk_id: str, model_name: str) -> Optional[float]:
        """
        Get cached reranker score.
        
        Args:
            query: Query text
            chunk_id: Chunk identifier
            model_name: Reranker model name
            
        Returns:
            Cached score or None if not found
        """
        cache_key = self._create_cache_key(query, chunk_id, model_name)
        
        # Try L1 first
        entry = await self.l1_cache.get(cache_key)
        if entry and isinstance(entry, dict):
            try:
                score = float(entry['score'])
                self.l1_hits += 1
                logger.debug(f"L1 reranker cache hit for {chunk_id}")
                return score
            except (KeyError, TypeError, ValueError) as e:
                logger.warning(f"Invalid L1 reranker entry: {e}")
        
        # Try L2 if available
        if self.l2_cache:
            entry_data = await self.l2_cache.get(cache_key)
            if entry_data and isinstance(entry_data, dict):
                try:
                    score = float(entry_data['score'])
                    
                    # Backfill L1
                    await self.l1_cache.set(cache_key, entry_data, ttl=self.ttl_seconds)
                    
                    self.l2_hits += 1
                    logger.debug(f"L2 reranker cache hit for {chunk_id}")
                    return score
                except (KeyError, TypeError, ValueError) as e:
                    logger.warning(f"Invalid L2 reranker entry: {e}")
        
        self.misses += 1
        return None
    
    async def set_score(self, query: str, chunk_id: str, model_name: str, score: float) -> bool:
        """
        Cache reranker score.
        
        Args:
            query: Query text
            chunk_id: Chunk identifier
            model_name: Reranker model name
            score: Reranker score
            
        Returns:
            True if successfully cached
        """
        cache_key = self._create_cache_key(query, chunk_id, model_name)
        
        # Prepare cache entry
        entry_data = {
            'score': float(score),
            'model_name': model_name,
            'created_at': time.time(),
            'hit_count': 0
        }
        
        results = []
        
        # Store in L1
        results.append(await self.l1_cache.set(cache_key, entry_data, ttl=self.ttl_seconds))
        
        # Store in L2 if available
        if self.l2_cache:
            results.append(await self.l2_cache.set(cache_key, entry_data, ttl=self.ttl_seconds))
        
        success = any(results)
        if success:
            logger.debug(f"Cached reranker score for {chunk_id}: {score:.4f}")
        
        return success
    
    async def set_batch_scores(self, 
                             query: str, 
                             chunk_scores: List[Tuple[str, float]], 
                             model_name: str) -> int:
        """
        Cache multiple reranker scores efficiently.
        
        Args:
            query: Query text
            chunk_scores: List of (chunk_id, score) tuples
            model_name: Reranker model name
            
        Returns:
            Number of scores successfully cached
        """
        cached_count = 0
        
        for chunk_id, score in chunk_scores:
            success = await self.set_score(query, chunk_id, model_name, score)
            if success:
                cached_count += 1
        
        logger.debug(f"Batch cached {cached_count}/{len(chunk_scores)} reranker scores")
        return cached_count
    
    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.l1_hits + self.l2_hits + self.misses
        hit_rate = (self.l1_hits + self.l2_hits) / total_requests if total_requests > 0 else 0
        
        stats = {
            "type": "reranker_cache",
            "l1_hits": self.l1_hits,
            "l2_hits": self.l2_hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "l1_stats": self.l1_cache.get_stats()
        }
        
        if self.l2_cache:
            stats["l2_stats"] = self.l2_cache.get_stats()
        
        return stats


# Factory functions
def create_embedding_cache(memory_max_size: int = 1000, 
                         redis_url: Optional[str] = None) -> EmbeddingCache:
    """Create embedding cache instance."""
    return EmbeddingCache(memory_max_size=memory_max_size, redis_url=redis_url)


def create_reranker_cache(memory_max_size: int = 5000, 
                        redis_url: Optional[str] = None) -> RerankerScoreCache:
    """Create reranker score cache instance."""
    return RerankerScoreCache(memory_max_size=memory_max_size, redis_url=redis_url)


# Test function
async def test_embedding_cache():
    """Test embedding cache functionality."""
    print("Testing embedding cache...")
    
    cache = create_embedding_cache(memory_max_size=10)
    
    # Test data
    query = "What is annual leave?"
    model_name = "BAAI/bge-m3"
    embedding = np.random.rand(1024)  # Mock embedding
    
    # Test set
    success = await cache.set_embedding(query, model_name, embedding)
    print(f"Set embedding: {success}")
    
    # Test get
    cached_embedding = await cache.get_embedding(query, model_name)
    print(f"Retrieved embedding: {cached_embedding is not None}")
    
    if cached_embedding is not None:
        print(f"Embedding match: {np.allclose(embedding, cached_embedding)}")
    
    # Test stats
    stats = cache.get_stats()
    print(f"Cache stats: {stats}")


async def test_reranker_cache():
    """Test reranker cache functionality."""
    print("Testing reranker cache...")
    
    cache = create_reranker_cache(memory_max_size=10)
    
    # Test data
    query = "What is annual leave?"
    chunk_id = "chunk_123"
    model_name = "cross-encoder/ms-marco-MiniLM-L-2-v2"
    score = 0.85
    
    # Test set
    success = await cache.set_score(query, chunk_id, model_name, score)
    print(f"Set score: {success}")
    
    # Test get
    cached_score = await cache.get_score(query, chunk_id, model_name)
    print(f"Retrieved score: {cached_score}")
    print(f"Score match: {abs(score - (cached_score or 0)) < 1e-6}")
    
    # Test batch set
    batch_scores = [(f"chunk_{i}", 0.5 + i * 0.1) for i in range(5)]
    cached_count = await cache.set_batch_scores(query, batch_scores, model_name)
    print(f"Batch cached: {cached_count}/{len(batch_scores)}")
    
    # Test stats
    stats = cache.get_stats()
    print(f"Cache stats: {stats}")


if __name__ == "__main__":
    import asyncio
    
    async def run_tests():
        await test_embedding_cache()
        print("\n" + "="*50 + "\n")
        await test_reranker_cache()
    
    asyncio.run(run_tests())