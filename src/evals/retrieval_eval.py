#!/usr/bin/env python3
# python -m src.evals.retrieval_eval --gold data/eval/retrieval_gold.jsonl --faiss data/indices/faiss.index --store data/processed/store.pkl
"""
Retrieval evaluation for Employment Act Malaysia compliance agent.
Computes Recall@k, MRR (Mean Reciprocal Rank), and nDCG metrics.
"""

import argparse
import json
import math
from pathlib import Path
from typing import List, Dict, Any, Tuple, Set
import sys
import os

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.retriever.hybrid_retriever import HybridRetriever


class RetrievalEvaluator:
    """Evaluates retrieval performance using standard IR metrics."""
    
    def __init__(self, retriever: HybridRetriever):
        """Initialize evaluator with a retriever instance.
        
        Args:
            retriever: HybridRetriever instance
        """
        self.retriever = retriever
    
    def load_gold_dataset(self, gold_file: Path) -> List[Dict[str, Any]]:
        """Load gold standard dataset.
        
        Args:
            gold_file: Path to JSONL file with gold standard queries
            
        Returns:
            List of gold standard entries
        """
        gold_data = []
        with open(gold_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    entry = json.loads(line.strip())
                    gold_data.append(entry)
                except json.JSONDecodeError as e:
                    print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
        
        print(f"Loaded {len(gold_data)} gold standard queries")
        return gold_data
    
    def check_relevance(self, retrieved_chunk: Dict[str, Any], relevant_section_ids: List[str]) -> bool:
        """Check if a retrieved chunk is relevant based on section IDs.
        
        Args:
            retrieved_chunk: Retrieved chunk with metadata
            relevant_section_ids: List of relevant section IDs from gold standard
            
        Returns:
            True if chunk is relevant
        """
        chunk_section = retrieved_chunk.get('section_id', '').strip()
        chunk_text = retrieved_chunk.get('text', '').lower()
        
        if not chunk_section and not chunk_text:
            return False
        
        # Check direct section ID match
        for relevant_id in relevant_section_ids:
            relevant_id_clean = relevant_id.strip().lower()
            
            # Direct section ID match
            if chunk_section.lower() == relevant_id_clean:
                return True
            
            # Section ID contained in chunk section
            if relevant_id_clean in chunk_section.lower():
                return True
            
            # Section ID mentioned in chunk text (for flexible matching)
            if relevant_id_clean in chunk_text:
                return True
        
        return False
    
    def compute_recall_at_k(self, retrieved_chunks: List[Dict[str, Any]], 
                          relevant_section_ids: List[str], k: int) -> float:
        """Compute Recall@k metric.
        
        Args:
            retrieved_chunks: List of retrieved chunks (top-k)
            relevant_section_ids: List of relevant section IDs
            k: Cut-off rank
            
        Returns:
            Recall@k score (0-1)
        """
        if not relevant_section_ids:
            return 0.0
        
        # Take only top-k results
        top_k_chunks = retrieved_chunks[:k]
        
        # Count how many relevant items were retrieved
        relevant_retrieved = 0
        for chunk in top_k_chunks:
            if self.check_relevance(chunk, relevant_section_ids):
                relevant_retrieved += 1
        
        # Recall = relevant retrieved / total relevant
        # For our case, we assume each query has at least 1 relevant section
        # We'll use the number of unique relevant sections as denominator
        unique_relevant = len(set(relevant_section_ids))
        recall = min(relevant_retrieved / unique_relevant, 1.0)  # Cap at 1.0
        
        return recall
    
    def compute_mrr(self, retrieved_chunks: List[Dict[str, Any]], 
                   relevant_section_ids: List[str]) -> float:
        """Compute Mean Reciprocal Rank for a single query.
        
        Args:
            retrieved_chunks: List of retrieved chunks
            relevant_section_ids: List of relevant section IDs
            
        Returns:
            Reciprocal rank (0-1)
        """
        for rank, chunk in enumerate(retrieved_chunks, 1):
            if self.check_relevance(chunk, relevant_section_ids):
                return 1.0 / rank
        
        return 0.0  # No relevant documents found
    
    def compute_ndcg_at_k(self, retrieved_chunks: List[Dict[str, Any]], 
                         relevant_section_ids: List[str], k: int) -> float:
        """Compute Normalized Discounted Cumulative Gain at k.
        
        Args:
            retrieved_chunks: List of retrieved chunks
            relevant_section_ids: List of relevant section IDs
            k: Cut-off rank
            
        Returns:
            nDCG@k score (0-1)
        """
        def dcg_at_k(relevances: List[int], k: int) -> float:
            """Compute DCG@k given relevance scores."""
            dcg = 0.0
            for i, rel in enumerate(relevances[:k]):
                if i == 0:
                    dcg += rel
                else:
                    dcg += rel / math.log2(i + 1)
            return dcg
        
        # Get relevance scores for retrieved chunks
        retrieved_relevances = []
        for chunk in retrieved_chunks[:k]:
            if self.check_relevance(chunk, relevant_section_ids):
                retrieved_relevances.append(1)  # Binary relevance
            else:
                retrieved_relevances.append(0)
        
        # Compute DCG
        dcg = dcg_at_k(retrieved_relevances, k)
        
        # Compute ideal DCG (all relevant items at top)
        num_relevant = len(relevant_section_ids)
        ideal_relevances = [1] * min(num_relevant, k) + [0] * max(0, k - num_relevant)
        idcg = dcg_at_k(ideal_relevances, k)
        
        if idcg == 0:
            return 0.0
        
        return dcg / idcg
    
    def evaluate_query(self, query: str, relevant_section_ids: List[str], 
                      top_k: int = 20) -> Dict[str, float]:
        """Evaluate a single query.
        
        Args:
            query: Query string
            relevant_section_ids: List of relevant section IDs
            top_k: Number of results to retrieve
            
        Returns:
            Dictionary with evaluation metrics
        """
        # Retrieve chunks
        retrieved_chunks = self.retriever.retrieve(query, top_k=top_k)
        
        # Compute metrics
        metrics = {}
        
        # Recall@k for different k values
        for k in [1, 3, 5, 8, 10]:
            if k <= top_k:
                metrics[f'recall@{k}'] = self.compute_recall_at_k(retrieved_chunks, relevant_section_ids, k)
        
        # MRR
        metrics['mrr'] = self.compute_mrr(retrieved_chunks, relevant_section_ids)
        
        # nDCG@k for different k values
        for k in [5, 8, 10]:
            if k <= top_k:
                metrics[f'ndcg@{k}'] = self.compute_ndcg_at_k(retrieved_chunks, relevant_section_ids, k)
        
        return metrics
    
    def evaluate_dataset(self, gold_data: List[Dict[str, Any]], 
                        top_k: int = 20) -> Dict[str, Any]:
        """Evaluate entire gold dataset.
        
        Args:
            gold_data: List of gold standard entries
            top_k: Number of results to retrieve per query
            
        Returns:
            Aggregated evaluation results
        """
        print(f"Evaluating {len(gold_data)} queries...")
        
        all_metrics = []
        query_results = []
        
        for i, entry in enumerate(gold_data, 1):
            query = entry['query']
            relevant_section_ids = entry['relevant_section_ids']
            query_id = entry.get('query_id', f'Q{i:03d}')
            
            print(f"[{i}/{len(gold_data)}] Evaluating: {query[:50]}...")
            
            try:
                metrics = self.evaluate_query(query, relevant_section_ids, top_k)
                all_metrics.append(metrics)
                
                query_results.append({
                    'query_id': query_id,
                    'query': query,
                    'relevant_sections': relevant_section_ids,
                    'metrics': metrics
                })
                
            except Exception as e:
                print(f"Error evaluating query {query_id}: {e}")
                continue
        
        # Aggregate metrics
        if not all_metrics:
            return {'error': 'No successful evaluations'}
        
        aggregated = {}
        metric_names = all_metrics[0].keys()
        
        for metric_name in metric_names:
            values = [m[metric_name] for m in all_metrics if metric_name in m]
            if values:
                aggregated[f'mean_{metric_name}'] = sum(values) / len(values)
                aggregated[f'min_{metric_name}'] = min(values)
                aggregated[f'max_{metric_name}'] = max(values)
        
        # Overall statistics
        results = {
            'num_queries': len(gold_data),
            'num_successful': len(all_metrics),
            'aggregated_metrics': aggregated,
            'query_results': query_results
        }
        
        return results
    
    def print_evaluation_report(self, results: Dict[str, Any]):
        """Print formatted evaluation report.
        
        Args:
            results: Evaluation results from evaluate_dataset
        """
        print("\n" + "="*80)
        print("RETRIEVAL EVALUATION REPORT")
        print("="*80)
        
        print(f"Queries evaluated: {results['num_successful']}/{results['num_queries']}")
        print()
        
        # Key metrics
        agg = results['aggregated_metrics']
        
        print("KEY METRICS:")
        print("-" * 40)
        key_metrics = ['recall@1', 'recall@3', 'recall@5', 'recall@8', 'mrr', 'ndcg@8']
        
        for metric in key_metrics:
            mean_key = f'mean_{metric}'
            if mean_key in agg:
                mean_val = agg[mean_key]
                print(f"{metric.upper():>12}: {mean_val:.4f}")
        
        print()
        
        # Quality gates
        print("QUALITY GATES:")
        print("-" * 40)
        recall_8 = agg.get('mean_recall@8', 0.0)
        target_recall_8 = 0.9
        
        if recall_8 >= target_recall_8:
            print(f"✅ Recall@8: {recall_8:.4f} >= {target_recall_8} (PASS)")
        else:
            print(f"❌ Recall@8: {recall_8:.4f} < {target_recall_8} (FAIL)")
            print(f"   Need to improve by {target_recall_8 - recall_8:.4f}")
        
        print()
        
        # Per-query breakdown for failed cases
        print("FAILED QUERIES (Recall@8 = 0):")
        print("-" * 40)
        failed_count = 0
        
        for query_result in results['query_results']:
            recall_8_query = query_result['metrics'].get('recall@8', 0.0)
            if recall_8_query == 0.0:
                failed_count += 1
                query_id = query_result['query_id']
                query = query_result['query']
                print(f"{query_id}: {query[:60]}...")
        
        if failed_count == 0:
            print("None! All queries have at least some relevant results.")
        else:
            print(f"\nTotal failed queries: {failed_count}/{results['num_successful']}")
        
        print("\n" + "="*80)


def main():
    """Main evaluation script."""
    parser = argparse.ArgumentParser(description="Evaluate retrieval performance")
    parser.add_argument('--gold', required=True, help='Path to gold standard JSONL file')
    parser.add_argument('--faiss', required=True, help='Path to FAISS index')
    parser.add_argument('--store', required=True, help='Path to store pickle file')
    parser.add_argument('--embedding-model', default="intfloat/e5-large-v2")
    parser.add_argument('--reranker-model', default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    parser.add_argument('--top-k', type=int, default=20, help='Number of results to retrieve')
    parser.add_argument('--output', help='Path to save detailed results JSON')
    
    args = parser.parse_args()
    
    # Initialize retriever
    print("Initializing retriever...")
    retriever = HybridRetriever(
        faiss_index_path=Path(args.faiss),
        store_path=Path(args.store),
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model
    )
    
    # Initialize evaluator
    evaluator = RetrievalEvaluator(retriever)
    
    # Load gold dataset
    gold_data = evaluator.load_gold_dataset(Path(args.gold))
    
    # Run evaluation
    results = evaluator.evaluate_dataset(gold_data, top_k=args.top_k)
    
    # Print report
    evaluator.print_evaluation_report(results)
    
    # Save detailed results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nDetailed results saved to: {args.output}")


if __name__ == "__main__":
    main()