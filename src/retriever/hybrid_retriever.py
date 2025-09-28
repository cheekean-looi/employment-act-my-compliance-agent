#!/usr/bin/env python3
"""
Hybrid retriever implementing BM25 + dense + cross-encoder re-ranking.
BM25 top-100 ∪ dense top-50 → cross-encoder re-rank → top-k (k=8).
"""

import pickle
import os
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi
from .advanced_query_rewrite import AdvancedQueryRewriter


class HybridRetriever:
    """Hybrid retriever combining BM25, dense embeddings, and cross-encoder re-ranking."""
    
    def __init__(
        self,
        faiss_index_path: Path,
        store_path: Path,
        embedding_model: str = None,
        reranker_model: str = None,
        bm25_topk: Optional[int] = None,
        dense_topk: Optional[int] = None,
        ce_max_pairs: Optional[int] = None,
        final_topk: int = 8,
        min_chunks: int = 6,
        embedding_cache=None,
        reranker_cache=None
    ):
        """Initialize the hybrid retriever.
        
        Args:
            faiss_index_path: Path to FAISS index file
            store_path: Path to pickle store with chunks and BM25 index
            embedding_model: Sentence transformer model for dense retrieval
            reranker_model: Cross-encoder model for re-ranking
            bm25_topk: Number of BM25 candidates (env: BM25_TOPK, default: 30 optimized, 100 spec)
            dense_topk: Number of dense candidates (env: DENSE_TOPK, default: 20 optimized, 50 spec)
            ce_max_pairs: Max candidates for cross-encoder (env: CE_MAX_PAIRS, default: 40 optimized, 150 spec)
            final_topk: Final number of results to return (env: FINAL_TOPK, default: 8)
            min_chunks: Minimum chunks to guarantee (env: MIN_CHUNKS, default: 6)
            embedding_cache: Cache for query embeddings (L1+L2 tiered cache)
            reranker_cache: Cache for cross-encoder scores (L1+L2 tiered cache)
        """
        # Use environment variables first, then parameters, then defaults
        self.embedding_model_name = embedding_model or os.getenv('EMBEDDING_MODEL', 'BAAI/bge-m3')
        self.reranker_model_name = reranker_model or os.getenv('RERANKER_MODEL', 'cross-encoder/ms-marco-MiniLM-L-2-v2')
        
        # Configure candidate sizes from environment or parameters
        # Hour 2 spec: BM25 top-100 ∪ dense top-50 → CE → top-8
        # Optimized defaults: BM25 30, dense 20, CE max 40 for performance
        self.bm25_topk = bm25_topk or int(os.getenv('BM25_TOPK', '30'))  # Spec: 100
        self.dense_topk = dense_topk or int(os.getenv('DENSE_TOPK', '20'))  # Spec: 50  
        self.ce_max_pairs = ce_max_pairs or int(os.getenv('CE_MAX_PAIRS', '40'))  # Spec: 150
        self.final_topk = int(os.getenv('FINAL_TOPK', str(final_topk)))
        self.min_chunks = int(os.getenv('MIN_CHUNKS', str(min_chunks)))
        
        # Store cache references for embedding and reranker optimization
        self.embedding_cache = embedding_cache
        self.reranker_cache = reranker_cache
        
        # Load indices and data with FAISS fallback
        self.dense_enabled = True
        try:
            self.faiss_index = faiss.read_index(str(faiss_index_path))
            print(f"Successfully loaded FAISS index: {faiss_index_path}")
        except Exception as e:
            print(f"⚠️ Failed to load FAISS index: {e}")
            print("Falling back to BM25-only mode")
            self.faiss_index = None
            self.dense_enabled = False
        
        try:
            with open(store_path, 'rb') as f:
                store_data = pickle.load(f)
            
            self.chunks = store_data['chunks']
            self.bm25_index = store_data['bm25_index']
            print(f"Successfully loaded store data: {len(self.chunks)} chunks")
        except Exception as e:
            raise RuntimeError(f"Failed to load store data from {store_path}: {e}")
        
        # Configure device for models
        import torch
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load models (only if dense retrieval is enabled)
        if self.dense_enabled:
            print(f"Loading embedding model: {self.embedding_model_name}")
            self.embedding_model = SentenceTransformer(self.embedding_model_name, device=str(self.device))
                
            # Store reference to embedding vector cache for query embeddings only
            if hasattr(embedding_cache, 'get_embedding') and hasattr(embedding_cache, 'set_embedding'):
                self.embedding_vector_cache = embedding_cache
            else:
                self.embedding_vector_cache = None
        else:
            print("Skipping embedding model loading (FAISS unavailable)")
            self.embedding_model = None
            self.embedding_vector_cache = None
        
        # Load cross-encoder model (models are singletons via dependency injection)
        print(f"Loading cross-encoder model: {self.reranker_model_name}")
        self.reranker = CrossEncoder(self.reranker_model_name, device=str(self.device))
        
        # Configure cross-encoder batching
        self.ce_batch_size = int(os.getenv('CE_BATCH_SIZE', '16'))
        
        # Initialize advanced query rewriter
        self.advanced_rewriter = AdvancedQueryRewriter()
        
        # Log effective configuration
        print(f"Retriever configuration:")
        print(f"  Chunks: {len(self.chunks)}")
        print(f"  BM25 top-k: {self.bm25_topk}")
        print(f"  Dense top-k: {self.dense_topk}")
        print(f"  CE max pairs: {self.ce_max_pairs}")
        print(f"  CE batch size: {self.ce_batch_size}")
        print(f"  Final top-k: {self.final_topk}")
        print(f"  Min chunks: {self.min_chunks}")
        
        # Validate configuration ranges
        if not 1 <= self.final_topk <= 20:
            raise ValueError(f"FINAL_TOPK must be 1-20, got {self.final_topk}")
        if not 1 <= self.ce_max_pairs <= 200:
            raise ValueError(f"CE_MAX_PAIRS must be 1-200, got {self.ce_max_pairs}")
        if not 1 <= self.min_chunks <= self.final_topk:
            raise ValueError(f"MIN_CHUNKS must be 1-{self.final_topk}, got {self.min_chunks}")
    
    def _get_bm25_candidates(self, query: str, top_k: int = 100) -> List[Tuple[int, float]]:
        """Get top-k candidates using BM25 sparse retrieval.
        
        Args:
            query: Search query
            top_k: Number of candidates to retrieve
            
        Returns:
            List of (chunk_index, score) tuples
        """
        # Tokenize query (same way as corpus was tokenized)
        query_tokens = query.lower().split()
        
        # Get BM25 scores for all documents
        scores = self.bm25_index.get_scores(query_tokens)
        
        # Get top-k indices
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        # Return (index, score) tuples
        candidates = [(int(idx), float(scores[idx])) for idx in top_indices if scores[idx] > 0]
        
        return candidates
    
    def _get_dense_candidates(self, query: str, top_k: int = 50) -> List[Tuple[int, float]]:
        """Get top-k candidates using dense vector retrieval.
        
        Args:
            query: Search query
            top_k: Number of candidates to retrieve
            
        Returns:
            List of (chunk_index, score) tuples (empty if dense retrieval unavailable)
        """
        # Skip if dense retrieval is disabled
        if not self.dense_enabled or self.faiss_index is None or self.embedding_model is None:
            print("Dense retrieval unavailable, skipping")
            return []
        
        try:
            # Note: Embedding cache access is handled at API level via asyncio.to_thread
            # In worker thread context, we skip caching to avoid async complexity
            # and rely on API-level caching for performance
            
            # Compute embedding directly (cache handled at higher level)
            query_embedding = self.embedding_model.encode([f"query: {query}"])
            
            # Normalize for cosine similarity
            faiss.normalize_L2(query_embedding)
            
            # Search FAISS index
            scores, indices = self.faiss_index.search(query_embedding, top_k)
            
            # Return (index, score) tuples
            candidates = [(int(indices[0][i]), float(scores[0][i])) for i in range(len(indices[0]))]
            
            return candidates
            
        except Exception as e:
            print(f"⚠️ Dense retrieval failed: {e}")
            print("Continuing with BM25-only results")
            return []
    
    def _merge_candidates(
        self, 
        bm25_candidates: List[Tuple[int, float]], 
        dense_candidates: List[Tuple[int, float]]
    ) -> List[int]:
        """Merge BM25 and dense candidates (union).
        
        Args:
            bm25_candidates: BM25 results as (index, score) tuples
            dense_candidates: Dense results as (index, score) tuples
            
        Returns:
            List of unique chunk indices
        """
        # Get all unique indices
        all_indices = set()
        
        # Add BM25 candidates
        for idx, _ in bm25_candidates:
            all_indices.add(idx)
        
        # Add dense candidates
        for idx, _ in dense_candidates:
            all_indices.add(idx)
        
        return list(all_indices)
    
    def _rerank_candidates(self, query: str, candidate_indices: List[int], top_k: int = 8) -> List[Dict[str, Any]]:
        """Re-rank candidates using cross-encoder and return top-k.
        
        Args:
            query: Search query
            candidate_indices: List of chunk indices to re-rank
            top_k: Number of final results to return
            
        Returns:
            List of ranked chunks with scores
        """
        if not candidate_indices:
            return []
        
        # Prepare query-document pairs for cross-encoder
        query_doc_pairs = []
        valid_indices = []
        
        for idx in candidate_indices:
            if 0 <= idx < len(self.chunks):
                chunk = self.chunks[idx]
                query_doc_pairs.append([query, chunk['text']])
                valid_indices.append(idx)
        
        if not query_doc_pairs:
            return []
        
        # Get cross-encoder scores with caching and batching
        rerank_scores = []
        cache_keys = []
        uncached_pairs = []
        uncached_indices = []
        
        # Note: Reranker cache access is handled at API level via asyncio.to_thread
        # In worker thread context, we skip caching to avoid async complexity
        # and rely on API-level caching for performance
        
        # Process all pairs without cache lookup (cache handled at higher level)
        for i, (q, doc) in enumerate(query_doc_pairs):
            chunk_id = self.chunks[valid_indices[i]].get('chunk_id', f"chunk_{valid_indices[i]}")
            cache_keys.append(chunk_id)
        
        uncached_pairs = query_doc_pairs
        uncached_indices = list(range(len(query_doc_pairs)))
        rerank_scores = [None] * len(query_doc_pairs)
        
        # Process uncached pairs with batching
        if uncached_pairs:
            if len(uncached_pairs) <= self.ce_batch_size:
                uncached_scores = self.reranker.predict(uncached_pairs)
            else:
                # Process in batches to manage memory and improve throughput
                uncached_scores = []
                for i in range(0, len(uncached_pairs), self.ce_batch_size):
                    batch = uncached_pairs[i:i + self.ce_batch_size]
                    batch_scores = self.reranker.predict(batch)
                    uncached_scores.extend(batch_scores)
            
            # Fill in uncached scores (caching handled at higher level)
            for i, score in enumerate(uncached_scores):
                original_idx = uncached_indices[i]
                rerank_scores[original_idx] = score
            
            print(f"Cross-encoder: processed {len(uncached_pairs)} pairs directly (cache handled at API level)")
        
        print(f"Cross-encoder processed {len(query_doc_pairs)} pairs in batches of {self.ce_batch_size}")
        
        # Combine indices with scores
        scored_results = []
        for i, idx in enumerate(valid_indices):
            chunk = self.chunks[idx]
            scored_results.append({
                'chunk_index': idx,
                'chunk_id': chunk.get('chunk_id', f"chunk_{idx}"),
                'text': chunk['text'],
                'section_id': chunk.get('section_id'),
                'url': chunk.get('url', ''),
                'score': float(rerank_scores[i])
            })
        
        # Sort by cross-encoder score (descending)
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        
        # Return top-k
        return scored_results[:top_k]
    
    def get_retrieval_status(self) -> Dict[str, Any]:
        """Get current retrieval configuration and status.
        
        Returns:
            Dictionary with retrieval status information
        """
        return {
            "dense_enabled": self.dense_enabled,
            "retrieval_mode": "hybrid" if self.dense_enabled else "bm25_only",
            "bm25_topk": self.bm25_topk,
            "dense_topk": self.dense_topk if self.dense_enabled else 0,
            "ce_max_pairs": self.ce_max_pairs,
            "final_topk": self.final_topk,
            "min_chunks": self.min_chunks
        }
    
    def retrieve(self, query: str, top_k: int = 8, use_advanced_rewrite: bool = True) -> List[Dict[str, Any]]:
        """Perform hybrid retrieval with re-ranking.
        
        Args:
            query: Search query
            top_k: Number of final results to return
            use_advanced_rewrite: Whether to use advanced query rewriting
            
        Returns:
            List of top-k ranked chunks with metadata
        """
        print(f"Retrieving for query: '{query[:50]}...'")
        
        # Allow env toggle to disable advanced rewrite globally
        import os
        env_toggle = os.getenv("USE_ADVANCED_REWRITE")
        if env_toggle is not None:
            use_advanced_rewrite = env_toggle.lower() in ("1", "true", "yes")

        if use_advanced_rewrite:
            # Use advanced query rewriting with statutory terms
            return self._retrieve_with_advanced_rewrite(query, top_k)
        else:
            # Original single-query retrieval
            return self._retrieve_single(query, top_k)
    
    def _retrieve_single(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """Single-query retrieval with configurable candidate sizes."""
        # Step 1: Get BM25 candidates (configurable)
        print("Getting BM25 candidates...")
        bm25_candidates = self._get_bm25_candidates(query, top_k=self.bm25_topk)
        print(f"BM25 found {len(bm25_candidates)} candidates")
        
        # Step 2: Get dense candidates (configurable)
        print("Getting dense candidates...")
        dense_candidates = self._get_dense_candidates(query, top_k=self.dense_topk)
        print(f"Dense found {len(dense_candidates)} candidates")
        
        # Step 3: Merge candidates (union), cap at configured max for cross-encoder
        candidate_indices = self._merge_candidates(bm25_candidates, dense_candidates)
        if len(candidate_indices) > self.ce_max_pairs:
            candidate_indices = candidate_indices[:self.ce_max_pairs]
        print(f"Merged to {len(candidate_indices)} unique candidates (max {self.ce_max_pairs} for CE)")
        
        # Step 4: Always keep minimum chunks, never drop all
        min_chunks = max(self.min_chunks, min(top_k, len(candidate_indices)))
        actual_top_k = max(top_k, min_chunks)
        
        # Step 5: Re-rank with cross-encoder
        print("Re-ranking with cross-encoder...")
        final_results = self._rerank_candidates(query, candidate_indices, top_k=actual_top_k)
        print(f"Returning top-{len(final_results)} results (min {min_chunks} guaranteed)")
        
        return final_results
    
    
    def _retrieve_with_advanced_rewrite(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """Enhanced retrieval using advanced query rewriting with statutory terms."""
        # Check for out-of-scope queries first
        normalized, priority_sections, is_out_of_scope = self.advanced_rewriter.normalize_query(query)
        
        if is_out_of_scope:
            print("Query marked as out-of-scope (no statutory basis)")
            return []
        
        # Generate enhanced query variants
        query_variants = self.advanced_rewriter.expand_query(query)
        print(f"Generated {len(query_variants)} enhanced query variants")
        
        all_candidates = set()
        
        # Collect candidates from all query variants
        for i, variant in enumerate(query_variants):
            print(f"Processing variant {i+1}/{len(query_variants)}: '{variant[:50]}...'")
            
            # Get BM25 candidates for this variant (configurable)
            bm25_candidates = self._get_bm25_candidates(variant, top_k=self.bm25_topk)
            
            # Get dense candidates for this variant (configurable)
            dense_candidates = self._get_dense_candidates(variant, top_k=self.dense_topk)
            
            # Merge and add to global candidate set
            variant_candidates = self._merge_candidates(bm25_candidates, dense_candidates)
            all_candidates.update(variant_candidates)
        
        print(f"Total unique candidates from all variants: {len(all_candidates)}")
        
        # Cap at configured max candidates for cross-encoder efficiency
        candidate_list = list(all_candidates)
        if len(candidate_list) > self.ce_max_pairs:
            candidate_list = candidate_list[:self.ce_max_pairs]
        print(f"Capped to {len(candidate_list)} candidates for cross-encoder (max {self.ce_max_pairs})")
        
        # Always keep minimum chunks, never drop all
        min_chunks = max(self.min_chunks, min(top_k, len(candidate_list)))
        actual_top_k = max(top_k, min_chunks)
        
        # Re-rank all candidates using original query
        print("Re-ranking with cross-encoder using original query...")
        enhanced_results = self._rerank_candidates(query, candidate_list, top_k=actual_top_k)
        
        # Boost priority sections if they exist
        if priority_sections and enhanced_results:
            boosted_results = []
            non_priority = []
            
            for result in enhanced_results:
                section_id = result.get('section_id', '')
                if section_id in priority_sections:
                    result['score'] = result.get('score', 0) + 2.0  # Strong boost for priority sections
                    boosted_results.append(result)
                else:
                    non_priority.append(result)
            
            # Re-sort by boosted scores
            all_results = boosted_results + non_priority
            all_results.sort(key=lambda x: x.get('score', 0), reverse=True)
            enhanced_results = all_results[:top_k]
        
        print(f"Returning top-{len(enhanced_results)} results with priority boosting")
        return enhanced_results
    
    def get_chunk_by_id(self, chunk_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific chunk by its ID.
        
        Args:
            chunk_id: The chunk ID to retrieve
            
        Returns:
            Chunk dictionary or None if not found
        """
        for chunk in self.chunks:
            if chunk.get('chunk_id') == chunk_id:
                return chunk
        return None


def main():
    """Test the hybrid retriever."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test hybrid retriever")
    parser.add_argument('--faiss', required=True, help='Path to FAISS index')
    parser.add_argument('--store', required=True, help='Path to store pickle file')
    parser.add_argument('--query', required=True, help='Search query')
    parser.add_argument('--top-k', type=int, default=8, help='Number of results')
    parser.add_argument('--embedding-model', default="BAAI/bge-m3", 
                        help='Embedding model')
    parser.add_argument('--reranker-model', default="cross-encoder/ms-marco-MiniLM-L-2-v2",
                        help='Cross-encoder model')
    
    args = parser.parse_args()
    
    # Initialize retriever
    retriever = HybridRetriever(
        faiss_index_path=Path(args.faiss),
        store_path=Path(args.store),
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model
    )
    
    # Display retrieval status
    status = retriever.get_retrieval_status()
    print(f"\n{'='*60}")
    print("RETRIEVAL CONFIGURATION")
    print(f"{'='*60}")
    print(f"Mode: {status['retrieval_mode'].upper()}")
    if status['retrieval_mode'] == 'bm25_only':
        print("⚠️  FAISS DENSE RETRIEVAL UNAVAILABLE - BM25-ONLY MODE")
    print(f"BM25 top-k: {status['bm25_topk']}")
    print(f"Dense top-k: {status['dense_topk']}")
    print(f"Cross-encoder max pairs: {status['ce_max_pairs']}")
    print(f"Final top-k: {status['final_topk']}")
    print(f"Minimum chunks: {status['min_chunks']}")
    
    # Perform retrieval
    results = retriever.retrieve(args.query, top_k=args.top_k)
    
    # Print results
    print(f"\n{'='*60}")
    print(f"QUERY: {args.query}")
    print(f"{'='*60}")
    
    for i, result in enumerate(results, 1):
        print(f"\n[{i}] Score: {result['score']:.4f}")
        print(f"Section: {result['section_id'] or 'N/A'}")
        print(f"Text: {result['text'][:200]}...")
        if len(result['text']) > 200:
            print("...")


if __name__ == "__main__":
    main()
