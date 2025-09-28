#!/usr/bin/env python3
"""
Dependency injection providers for FastAPI.
Manages singletons for RAG pipeline, guardrails, cache, and vLLM client.
"""

import os
import logging
import hashlib
import json
from typing import Optional, Dict, Any
from pathlib import Path
from functools import lru_cache
from contextlib import asynccontextmanager

# Import core components
import sys
sys.path.append(str(Path(__file__).parent.parent))

from src.generation.rag_pipeline import EmploymentActRAG, RAGConfig
from src.generation.guardrails import ProductionGuardrailsEngine
from src.retriever.hybrid_retriever import HybridRetriever
from src.tools.severance_calculator import SeveranceCalculator
from src.server.vllm_client import VLLMClient, GenerationConfig
from src.server.cache import TieredCache, create_tiered_cache, CacheKeyBuilder
from src.server.model_router import ModelRouter, create_model_router
from src.server.embedding_cache import EmbeddingCache, RerankerScoreCache, create_embedding_cache, create_reranker_cache

logger = logging.getLogger(__name__)


class Dependencies:
    """Container for application dependencies."""
    
    def __init__(self):
        """Initialize dependency container."""
        self._rag_pipeline: Optional[EmploymentActRAG] = None
        self._guardrails_engine: Optional[ProductionGuardrailsEngine] = None
        self._retriever: Optional[HybridRetriever] = None
        self._severance_calculator: Optional[SeveranceCalculator] = None
        self._vllm_client: Optional[VLLMClient] = None
        self._cache: Optional[TieredCache] = None
        self._cache_key_builder: Optional[CacheKeyBuilder] = None
        self._model_router: Optional[ModelRouter] = None
        self._embedding_cache: Optional[EmbeddingCache] = None
        self._reranker_cache: Optional[RerankerScoreCache] = None
        
        # Configuration from environment
        self.config = self._load_config()
        
        # Compute config hash once at startup
        self._config_hash: Optional[str] = None
        
        logger.info("Initialized dependency container")
    
    def _load_config(self) -> dict:
        """Load configuration from environment variables."""
        return {
            # vLLM settings
            "vllm_base_url": os.getenv("VLLM_BASE_URL", "http://localhost:8000"),
            # Default to a small instruct model; override via env
            "model_name": os.getenv("MODEL_NAME", "Qwen/Qwen2.5-1.5B-Instruct"),
            "adapter_path": os.getenv("ADAPTER_PATH"),
            
            # Data paths
            "faiss_index_path": os.getenv("FAISS_INDEX_PATH", "data/indices/faiss.index"),
            "store_path": os.getenv("STORE_PATH", "data/indices/store.pkl"),
            "guardrails_config": os.getenv("GUARDRAILS_CONFIG"),
            
            # Models for retrieval
            "embedding_model": os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3"),
            "reranker_model": os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L-2-v2"),
            
            # Cache settings
            "redis_url": os.getenv("REDIS_URL"),
            "cache_memory_size": int(os.getenv("CACHE_MEMORY_SIZE", "1000")),
            "cache_ttl": int(os.getenv("CACHE_TTL", "1800")),  # 30 minutes
            
            # Generation settings
            "max_tokens": int(os.getenv("MAX_TOKENS", "512")),
            "temperature": float(os.getenv("TEMPERATURE", "0.1")),
            "top_k": int(os.getenv("TOP_K", "8")),
            "min_context_score": float(os.getenv("MIN_CONTEXT_SCORE", "0.2")),
            
            # Performance settings
            "vllm_timeout": float(os.getenv("VLLM_TIMEOUT", "60.0")),
            "circuit_breaker_threshold": int(os.getenv("CIRCUIT_BREAKER_THRESHOLD", "5")),
        }
    
    def _compute_config_hash(self) -> str:
        """
        Compute a deterministic hash of all configuration that affects cache correctness.
        
        Includes:
        - Model name and adapter path
        - Prompt template fingerprints
        - Retrieval parameters (top_k, min_context_score, models)
        - Guardrails configuration version
        - Generation settings (temperature, max_tokens)
        
        Returns:
            8-character hex hash of configuration fingerprint
        """
        try:
            config_fingerprint = {
                # Model configuration
                "model_name": self.config["model_name"],
                "adapter_path": self.config.get("adapter_path"),
                
                # Retrieval configuration
                "embedding_model": self.config["embedding_model"],
                "reranker_model": self.config["reranker_model"],
                "retrieval_top_k": self.config["top_k"],
                "min_context_score": self.config["min_context_score"],
                
                # Generation configuration
                "max_tokens": self.config["max_tokens"],
                "temperature": self.config["temperature"],
                
                # Prompt template fingerprint (if available)
                "prompt_template_version": self._get_prompt_template_fingerprint(),
                
                # Guardrails version
                "guardrails_version": self._get_guardrails_version(),
            }
            
            # Create deterministic JSON and hash
            config_json = json.dumps(config_fingerprint, sort_keys=True)
            config_hash = hashlib.sha256(config_json.encode()).hexdigest()[:8]
            
            logger.info(f"Computed config hash: {config_hash}")
            return config_hash
            
        except Exception as e:
            logger.warning(f"Failed to compute config hash, using fallback: {e}")
            # Fallback to timestamp-based hash for development
            import time
            fallback = f"dev_{int(time.time()) % 10000:04d}"
            return fallback
    
    def _get_prompt_template_fingerprint(self) -> str:
        """Get fingerprint of prompt templates used in RAG pipeline."""
        try:
            # Try to read prompt templates from the generation module
            from src.generation.rag_pipeline import RAGConfig
            
            # For now, use a version string that should be updated when templates change
            # TODO: Read actual template files and compute hash
            return "v1.2.0"  # Update this when prompt templates change
            
        except Exception:
            return "unknown"
    
    def _get_guardrails_version(self) -> str:
        """Get guardrails configuration version."""
        try:
            guardrails_config_path = self.config.get("guardrails_config")
            if not guardrails_config_path:
                guardrails_config_path = "config/guardrails.yaml"
            
            config_path = Path(guardrails_config_path)
            if config_path.exists():
                # Read the metadata version from guardrails config
                import yaml
                with open(config_path, 'r') as f:
                    yaml_config = yaml.safe_load(f)
                    metadata = yaml_config.get("metadata", {})
                    return metadata.get("version", "1.0.0")
            else:
                return "1.0.0"  # Default version
                
        except Exception as e:
            logger.warning(f"Could not read guardrails version: {e}")
            return "unknown"
    
    async def get_rag_pipeline(self) -> EmploymentActRAG:
        """Get or create RAG pipeline singleton."""
        if self._rag_pipeline is None:
            try:
                # Get cache instances for performance optimization
                embedding_cache = await self.get_embedding_cache()
                reranker_cache = await self.get_reranker_cache()
                
                rag_config = RAGConfig(
                    faiss_index_path=Path(self.config["faiss_index_path"]),
                    store_path=Path(self.config["store_path"]),
                    embedding_model=self.config["embedding_model"],
                    reranker_model=self.config["reranker_model"],
                    retrieval_top_k=self.config["top_k"],
                    min_context_score=self.config["min_context_score"],
                    enable_guardrails=True,
                    embedding_cache=embedding_cache,
                    reranker_cache=reranker_cache
                )
                
                self._rag_pipeline = EmploymentActRAG(rag_config)
                logger.info("Created RAG pipeline singleton")
                
            except Exception as e:
                logger.error(f"Failed to initialize RAG pipeline: {e}")
                raise
        
        return self._rag_pipeline
    
    async def get_guardrails_engine(self) -> ProductionGuardrailsEngine:
        """Get or create guardrails engine singleton."""
        if self._guardrails_engine is None:
            try:
                config_path = None
                if self.config["guardrails_config"]:
                    config_path = Path(self.config["guardrails_config"])
                
                self._guardrails_engine = ProductionGuardrailsEngine(config_path)
                logger.info("Created guardrails engine singleton")
                
            except Exception as e:
                logger.error(f"Failed to initialize guardrails engine: {e}")
                raise
        
        return self._guardrails_engine
    
    async def get_retriever(self) -> HybridRetriever:
        """Get or create retriever singleton."""
        if self._retriever is None:
            try:
                # Get cache instances for performance optimization
                embedding_cache = await self.get_embedding_cache()
                reranker_cache = await self.get_reranker_cache()
                
                self._retriever = HybridRetriever(
                    faiss_index_path=Path(self.config["faiss_index_path"]),
                    store_path=Path(self.config["store_path"]),
                    embedding_model=self.config["embedding_model"],
                    reranker_model=self.config["reranker_model"],
                    embedding_cache=embedding_cache,
                    reranker_cache=reranker_cache
                )
                logger.info("Created retriever singleton")
                
            except Exception as e:
                logger.error(f"Failed to initialize retriever: {e}")
                raise
        
        return self._retriever
    
    async def get_severance_calculator(self) -> SeveranceCalculator:
        """Get or create severance calculator singleton."""
        if self._severance_calculator is None:
            try:
                self._severance_calculator = SeveranceCalculator()
                logger.info("Created severance calculator singleton")
                
            except Exception as e:
                logger.error(f"Failed to initialize severance calculator: {e}")
                raise
        
        return self._severance_calculator
    
    async def get_vllm_client(self) -> VLLMClient:
        """Get or create vLLM client singleton."""
        if self._vllm_client is None:
            try:
                self._vllm_client = VLLMClient(
                    base_url=self.config["vllm_base_url"],
                    model_name=self.config["model_name"],
                    timeout=self.config["vllm_timeout"],
                    circuit_breaker_threshold=self.config["circuit_breaker_threshold"]
                )
                logger.info("Created vLLM client singleton")
                
            except Exception as e:
                logger.error(f"Failed to initialize vLLM client: {e}")
                raise
        
        return self._vllm_client
    
    async def get_cache(self) -> TieredCache:
        """Get or create cache singleton."""
        if self._cache is None:
            try:
                self._cache = create_tiered_cache(
                    memory_max_size=self.config["cache_memory_size"],
                    memory_ttl=self.config["cache_ttl"],
                    redis_url=self.config["redis_url"]
                )
                logger.info("Created cache singleton")
                
            except Exception as e:
                logger.error(f"Failed to initialize cache: {e}")
                raise
        
        return self._cache
    
    def get_cache_key_builder(self) -> CacheKeyBuilder:
        """Get cache key builder (stateless)."""
        if self._cache_key_builder is None:
            self._cache_key_builder = CacheKeyBuilder()
        
        return self._cache_key_builder
    
    def get_config_hash(self) -> str:
        """Get computed configuration hash."""
        if self._config_hash is None:
            self._config_hash = self._compute_config_hash()
        return self._config_hash
    
    def get_model_router(self) -> ModelRouter:
        """Get model router (stateless)."""
        if self._model_router is None:
            self._model_router = create_model_router()
        return self._model_router
    
    async def get_embedding_cache(self) -> EmbeddingCache:
        """Get embedding cache singleton."""
        if self._embedding_cache is None:
            try:
                self._embedding_cache = create_embedding_cache(
                    memory_max_size=int(self.config.get("embedding_cache_size", "1000")),
                    redis_url=self.config.get("redis_url")
                )
                logger.info("Created embedding cache singleton")
            except Exception as e:
                logger.error(f"Failed to initialize embedding cache: {e}")
                raise
        return self._embedding_cache
    
    async def get_reranker_cache(self) -> RerankerScoreCache:
        """Get reranker score cache singleton."""
        if self._reranker_cache is None:
            try:
                self._reranker_cache = create_reranker_cache(
                    memory_max_size=int(self.config.get("reranker_cache_size", "5000")),
                    redis_url=self.config.get("redis_url")
                )
                logger.info("Created reranker cache singleton")
            except Exception as e:
                logger.error(f"Failed to initialize reranker cache: {e}")
                raise
        return self._reranker_cache
    
    def get_generation_config(self, 
                            max_tokens: Optional[int] = None,
                            temperature: Optional[float] = None) -> GenerationConfig:
        """Get generation configuration with optional overrides."""
        return GenerationConfig(
            max_tokens=max_tokens or self.config["max_tokens"],
            temperature=temperature or self.config["temperature"],
            top_k=50  # For generation, not retrieval
        )
    
    async def health_check(self) -> dict:
        """Check health of all dependencies."""
        health = {}
        
        try:
            # Check RAG pipeline
            rag = await self.get_rag_pipeline()
            health["rag_ready"] = rag is not None
        except Exception as e:
            health["rag_ready"] = False
            health["rag_error"] = str(e)
        
        try:
            # Check vLLM
            vllm = await self.get_vllm_client()
            health["vllm_ready"] = await vllm.health_check()
        except Exception as e:
            health["vllm_ready"] = False
            health["vllm_error"] = str(e)
        
        try:
            # Check indices
            retriever = await self.get_retriever()
            # Try a simple test query
            test_results = retriever.retrieve("test", top_k=1)
            health["indices_ok"] = len(test_results) >= 0  # Even empty results are OK
        except Exception as e:
            health["indices_ok"] = False
            health["indices_error"] = str(e)
        
        try:
            # Check cache
            cache = await self.get_cache()
            cache_health = await cache.health_check()
            health["cache_status"] = "ok" if all(cache_health.values()) else "degraded"
            health["cache_details"] = cache_health
        except Exception as e:
            health["cache_status"] = "error"
            health["cache_error"] = str(e)
        
        try:
            # Check guardrails
            guardrails = await self.get_guardrails_engine()
            health["guardrails_version"] = guardrails.config_version
        except Exception as e:
            health["guardrails_version"] = "error"
            health["guardrails_error"] = str(e)
        
        return health
    
    async def cleanup(self):
        """Cleanup resources."""
        try:
            if self._vllm_client:
                await self._vllm_client.close()
            
            if self._cache and hasattr(self._cache, 'l2_cache') and self._cache.l2_cache:
                await self._cache.l2_cache.close()
                
            logger.info("Cleaned up dependencies")
            
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")


# Global dependencies instance
_deps: Optional[Dependencies] = None


def get_dependencies() -> Dependencies:
    """Get global dependencies instance."""
    global _deps
    if _deps is None:
        _deps = Dependencies()
    return _deps


# FastAPI dependency functions
async def get_rag_pipeline() -> EmploymentActRAG:
    """FastAPI dependency for RAG pipeline."""
    deps = get_dependencies()
    return await deps.get_rag_pipeline()


async def get_guardrails_engine() -> ProductionGuardrailsEngine:
    """FastAPI dependency for guardrails engine."""
    deps = get_dependencies()
    return await deps.get_guardrails_engine()


async def get_retriever() -> HybridRetriever:
    """FastAPI dependency for retriever."""
    deps = get_dependencies()
    return await deps.get_retriever()


async def get_severance_calculator() -> SeveranceCalculator:
    """FastAPI dependency for severance calculator."""
    deps = get_dependencies()
    return await deps.get_severance_calculator()


async def get_vllm_client() -> VLLMClient:
    """FastAPI dependency for vLLM client."""
    deps = get_dependencies()
    return await deps.get_vllm_client()


async def get_cache() -> TieredCache:
    """FastAPI dependency for cache."""
    deps = get_dependencies()
    return await deps.get_cache()


def get_cache_key_builder() -> CacheKeyBuilder:
    """FastAPI dependency for cache key builder."""
    deps = get_dependencies()
    return deps.get_cache_key_builder()


def get_config_hash() -> str:
    """FastAPI dependency for configuration hash."""
    deps = get_dependencies()
    return deps.get_config_hash()


def get_model_router() -> ModelRouter:
    """FastAPI dependency for model router."""
    deps = get_dependencies()
    return deps.get_model_router()


async def get_embedding_cache() -> EmbeddingCache:
    """FastAPI dependency for embedding cache."""
    deps = get_dependencies()
    return await deps.get_embedding_cache()


async def get_reranker_cache() -> RerankerScoreCache:
    """FastAPI dependency for reranker cache."""
    deps = get_dependencies()
    return await deps.get_reranker_cache()


# Lifespan management for FastAPI
@asynccontextmanager
async def lifespan_manager(app):
    """Manage application lifespan for FastAPI."""
    logger.info("Starting application...")
    
    # Initialize dependencies
    deps = get_dependencies()
    
    try:
        # Pre-load critical dependencies and compute config hash
        await deps.get_guardrails_engine()
        await deps.get_cache()
        
        # Compute config hash for caching
        config_hash = deps.get_config_hash()
        logger.info(f"Application config hash: {config_hash}")
        
        # Check vLLM connectivity (non-blocking)
        try:
            vllm = await deps.get_vllm_client()
            healthy = await vllm.health_check()
            if healthy:
                logger.info("vLLM service is ready")
            else:
                logger.warning("vLLM service not ready, will retry on first request")
        except Exception as e:
            logger.warning(f"vLLM not available at startup: {e}")
        
        logger.info("Application startup complete")
        yield
        
    finally:
        logger.info("Shutting down application...")
        await deps.cleanup()
        logger.info("Application shutdown complete")


# Configuration helpers
@lru_cache()
def get_config() -> dict:
    """Get cached configuration."""
    deps = get_dependencies()
    return deps.config


def validate_config() -> dict:
    """Validate configuration and return status."""
    config = get_config()
    status = {"valid": True, "errors": [], "warnings": []}
    
    # Check required paths
    faiss_path = Path(config["faiss_index_path"])
    if not faiss_path.exists():
        status["errors"].append(f"FAISS index not found: {faiss_path}")
        status["valid"] = False
    
    store_path = Path(config["store_path"])
    if not store_path.exists():
        status["errors"].append(f"Store file not found: {store_path}")
        status["valid"] = False
    
    # Check vLLM URL format
    vllm_url = config["vllm_base_url"]
    if not vllm_url.startswith(("http://", "https://")):
        status["errors"].append(f"Invalid vLLM URL format: {vllm_url}")
        status["valid"] = False
    
    # Check optional Redis URL
    redis_url = config["redis_url"]
    if redis_url and not redis_url.startswith("redis://"):
        status["warnings"].append(f"Redis URL format may be invalid: {redis_url}")
    
    return status


if __name__ == "__main__":
    # Test dependencies
    import asyncio
    
    async def test_deps():
        deps = get_dependencies()
        
        print("Configuration:")
        for key, value in deps.config.items():
            print(f"  {key}: {value}")
        
        print("\nValidation:")
        validation = validate_config()
        print(f"  Valid: {validation['valid']}")
        if validation["errors"]:
            print(f"  Errors: {validation['errors']}")
        if validation["warnings"]:
            print(f"  Warnings: {validation['warnings']}")
        
        print("\nHealth check:")
        health = await deps.health_check()
        for key, value in health.items():
            print(f"  {key}: {value}")
        
        await deps.cleanup()
    
    asyncio.run(test_deps())
