#!/usr/bin/env python3
"""
Hybrid retriever implementing BM25 + dense + cross-encoder re-ranking.
BM25 top-100 ∪ dense top-50 → cross-encoder re-rank → top-k (k=8).
"""

import pickle
from pathlib import Path
from typing import Dict, List, Any, Tuple, Optional
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
from rank_bm25 import BM25Okapi


class HybridRetriever:
    """Hybrid retriever combining BM25, dense embeddings, and cross-encoder re-ranking."""
    
    def __init__(
        self,
        faiss_index_path: Path,
        store_path: Path,
        embedding_model: str = "intfloat/e5-large-v2",
        reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    ):
        """Initialize the hybrid retriever.
        
        Args:
            faiss_index_path: Path to FAISS index file
            store_path: Path to pickle store with chunks and BM25 index
            embedding_model: Sentence transformer model for dense retrieval
            reranker_model: Cross-encoder model for re-ranking
        """
        self.embedding_model_name = embedding_model
        self.reranker_model_name = reranker_model
        
        # Load indices and data
        self.faiss_index = faiss.read_index(str(faiss_index_path))
        
        with open(store_path, 'rb') as f:
            store_data = pickle.load(f)
        
        self.chunks = store_data['chunks']
        self.bm25_index = store_data['bm25_index']
        
        # Load models
        print(f"Loading embedding model: {embedding_model}")
        self.embedding_model = SentenceTransformer(embedding_model)
        
        print(f"Loading cross-encoder model: {reranker_model}")
        self.reranker = CrossEncoder(reranker_model)
        
        print(f"Loaded retriever with {len(self.chunks)} chunks")
    
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
            List of (chunk_index, score) tuples
        """
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        
        # Normalize for cosine similarity
        faiss.normalize_L2(query_embedding)
        
        # Search FAISS index
        scores, indices = self.faiss_index.search(query_embedding, top_k)
        
        # Return (index, score) tuples
        candidates = [(int(indices[0][i]), float(scores[0][i])) for i in range(len(indices[0]))]
        
        return candidates
    
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
        
        # Get cross-encoder scores
        rerank_scores = self.reranker.predict(query_doc_pairs)
        
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
    
    def retrieve(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        """Perform hybrid retrieval with re-ranking.
        
        Args:
            query: Search query
            top_k: Number of final results to return
            
        Returns:
            List of top-k ranked chunks with metadata
        """
        print(f"Retrieving for query: '{query[:50]}...'")
        
        # Step 1: Get BM25 candidates (top-100)
        print("Getting BM25 candidates...")
        bm25_candidates = self._get_bm25_candidates(query, top_k=100)
        print(f"BM25 found {len(bm25_candidates)} candidates")
        
        # Step 2: Get dense candidates (top-50)
        print("Getting dense candidates...")
        dense_candidates = self._get_dense_candidates(query, top_k=50)
        print(f"Dense found {len(dense_candidates)} candidates")
        
        # Step 3: Merge candidates (union)
        candidate_indices = self._merge_candidates(bm25_candidates, dense_candidates)
        print(f"Merged to {len(candidate_indices)} unique candidates")
        
        # Step 4: Re-rank with cross-encoder
        print("Re-ranking with cross-encoder...")
        final_results = self._rerank_candidates(query, candidate_indices, top_k=top_k)
        print(f"Returning top-{len(final_results)} results")
        
        return final_results
    
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
    parser.add_argument('--embedding-model', default="intfloat/e5-large-v2", 
                        help='Embedding model')
    parser.add_argument('--reranker-model', default="cross-encoder/ms-marco-MiniLM-L-12-v2",
                        help='Cross-encoder model')
    
    args = parser.parse_args()
    
    # Initialize retriever
    retriever = HybridRetriever(
        faiss_index_path=Path(args.faiss),
        store_path=Path(args.store),
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model
    )
    
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