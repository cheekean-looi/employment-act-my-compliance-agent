#!/usr/bin/env python3
# python -m src.generation.rag_pipeline --faiss data/indices/faiss.index --store data/indices/store.pkl --query "your query here"
"""
RAG pipeline for Employment Act Malaysia compliance agent.
Integrates hybrid retrieval, prompt templates, guardrails, and LLM generation.
"""

import json
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ..retriever.hybrid_retriever import HybridRetriever
from .prompt_templates import PromptTemplates, Response
from .guardrails import ProductionGuardrailsEngine, GuardrailResult


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""
    faiss_index_path: Path
    store_path: Path
    embedding_model: str = "intfloat/e5-large-v2"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-12-v2"
    retrieval_top_k: int = 8
    min_context_score: float = 0.3
    enable_guardrails: bool = True


class EmploymentActRAG:
    """Complete RAG pipeline for Employment Act Malaysia compliance."""
    
    def __init__(self, config: RAGConfig):
        """Initialize RAG pipeline.
        
        Args:
            config: RAG configuration
        """
        self.config = config
        
        # Initialize components
        print("Initializing Employment Act RAG pipeline...")
        
        self.retriever = HybridRetriever(
            faiss_index_path=config.faiss_index_path,
            store_path=config.store_path,
            embedding_model=config.embedding_model,
            reranker_model=config.reranker_model
        )
        
        self.prompt_templates = PromptTemplates()
        
        if config.enable_guardrails:
            self.guardrails = ProductionGuardrailsEngine()
        else:
            self.guardrails = None
        
        print("RAG pipeline initialized successfully")
    
    def _filter_low_quality_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out low-quality retrieved chunks.
        
        Args:
            chunks: Retrieved chunks with scores
            
        Returns:
            Filtered chunks above quality threshold
        """
        return [
            chunk for chunk in chunks 
            if chunk.get('score', 0.0) >= self.config.min_context_score
        ]
    
    def _estimate_confidence(self, chunks: List[Dict[str, Any]], guardrails_result: Optional[GuardrailResult]) -> float:
        """Estimate confidence score based on retrieval quality and guardrails.
        
        Args:
            chunks: Retrieved context chunks
            guardrails_result: Guardrails evaluation result
            
        Returns:
            Confidence score (0-1)
        """
        if not chunks:
            return 0.1
        
        # Base confidence on top retrieval score
        top_score = chunks[0].get('score', 0.0)
        
        # Adjust based on number of good results
        good_chunks = len([c for c in chunks if c.get('score', 0.0) > 0.5])
        diversity_bonus = min(good_chunks / 5.0, 0.2)  # Up to 20% bonus
        
        # Adjust based on section coverage
        has_sections = any(c.get('section_id') for c in chunks)
        section_bonus = 0.1 if has_sections else 0.0
        
        base_confidence = min(top_score + diversity_bonus + section_bonus, 0.95)
        
        # Apply guardrails penalties
        if guardrails_result:
            if guardrails_result.should_escalate:
                base_confidence *= 0.8  # Reduce confidence for escalation cases
            
            if 'insufficient_context' in guardrails_result.safety_flags:
                base_confidence *= 0.6
            
            if 'complex_legal' in guardrails_result.safety_flags:
                base_confidence *= 0.7
        
        return max(base_confidence, 0.1)  # Minimum confidence
    
    def retrieve_and_evaluate(self, query: str) -> Dict[str, Any]:
        """Retrieve context and apply guardrails.
        
        Args:
            query: User query
            
        Returns:
            Dictionary with retrieval results and guardrails evaluation
        """
        # Retrieve context
        retrieved_chunks = self.retriever.retrieve(query, top_k=self.config.retrieval_top_k)
        
        # Filter low-quality chunks
        filtered_chunks = self._filter_low_quality_chunks(retrieved_chunks)
        
        # Apply guardrails if enabled
        guardrails_result = None
        if self.guardrails:
            guardrails_result = self.guardrails.apply_guardrails(query, filtered_chunks)
        
        # Estimate confidence
        confidence = self._estimate_confidence(filtered_chunks, guardrails_result)
        
        return {
            'query': query,
            'retrieved_chunks': retrieved_chunks,
            'filtered_chunks': filtered_chunks,
            'guardrails_result': guardrails_result,
            'estimated_confidence': confidence
        }
    
    def generate_prompt(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """Generate prompt for LLM.
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            
        Returns:
            Formatted prompt string
        """
        system_prompt = self.prompt_templates.get_system_prompt()
        user_prompt = self.prompt_templates.get_rag_prompt(query, context_chunks)
        
        # Combine system and user prompts
        full_prompt = f"{system_prompt}\n\n{user_prompt}"
        
        return full_prompt
    
    def handle_refusal(self, query: str, guardrails_result: GuardrailResult) -> Dict[str, Any]:
        """Handle refusal cases based on guardrails.
        
        Args:
            query: User query
            guardrails_result: Guardrails evaluation
            
        Returns:
            Refusal response
        """
        if guardrails_result.refusal_reason:
            refusal_message = self.guardrails.get_refusal_message(
                guardrails_result.refusal_reason, 
                guardrails_result.safety_flags
            )
        else:
            refusal_message = "I cannot provide guidance on this matter. Please consult with a qualified legal professional."
        
        response = Response(
            answer=refusal_message,
            citations=[],
            confidence=1.0,
            should_escalate=True,
            safety_flags=guardrails_result.safety_flags
        )
        
        return response.to_dict()
    
    def handle_insufficient_context(self, query: str, context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Handle cases with insufficient context.
        
        Args:
            query: User query
            context_chunks: Available context (may be poor quality)
            
        Returns:
            Response acknowledging limitations
        """
        # Use template for insufficient context
        response_json = self.prompt_templates.get_insufficient_context_prompt(query, context_chunks)
        return json.loads(response_json)
    
    def process_query(self, query: str, llm_generate_func=None) -> Dict[str, Any]:
        """Process a complete query through the RAG pipeline.
        
        Args:
            query: User query
            llm_generate_func: Function to call LLM (optional, returns mock response if None)
            
        Returns:
            Complete response with answer, citations, confidence, etc.
        """
        print(f"Processing query: '{query[:50]}...'")
        
        # Step 1: Retrieve and evaluate
        retrieval_result = self.retrieve_and_evaluate(query)
        
        # Step 2: Check for refusal cases
        if retrieval_result['guardrails_result'] and retrieval_result['guardrails_result'].should_refuse:
            print("Query refused by guardrails")
            return self.handle_refusal(query, retrieval_result['guardrails_result'])
        
        # Step 3: Check for insufficient context
        if not retrieval_result['filtered_chunks']:
            print("Insufficient context for query")
            return self.handle_insufficient_context(query, retrieval_result['retrieved_chunks'])
        
        # Step 4: Generate prompt
        prompt = self.generate_prompt(query, retrieval_result['filtered_chunks'])
        
        # Step 5: Generate response (mock if no LLM function provided)
        if llm_generate_func:
            llm_response = llm_generate_func(prompt)
            
            try:
                # Parse and validate LLM response
                response_data = self.prompt_templates.extract_response_data(llm_response)
                
                # Override confidence with our estimate if LLM confidence seems off
                if abs(response_data['confidence'] - retrieval_result['estimated_confidence']) > 0.3:
                    response_data['confidence'] = retrieval_result['estimated_confidence']
                
                # Add guardrails flags
                if retrieval_result['guardrails_result']:
                    existing_flags = set(response_data.get('safety_flags', []))
                    guardrails_flags = set(retrieval_result['guardrails_result'].safety_flags)
                    response_data['safety_flags'] = list(existing_flags.union(guardrails_flags))
                    
                    # Add structured guardrails report for audit
                    if hasattr(retrieval_result['guardrails_result'], 'report') and retrieval_result['guardrails_result'].report:
                        response_data['guardrails_report'] = {
                            'timestamp': retrieval_result['guardrails_result'].report.timestamp,
                            'decision': retrieval_result['guardrails_result'].report.decision,
                            'confidence': retrieval_result['guardrails_result'].report.confidence,
                            'processing_time_ms': retrieval_result['guardrails_result'].report.processing_time_ms,
                            'input_flags': retrieval_result['guardrails_result'].report.input_flags,
                            'citations_valid': retrieval_result['guardrails_result'].report.citations_valid,
                            'invalid_citations': retrieval_result['guardrails_result'].report.invalid_citations,
                            'numeric_out_of_bounds': retrieval_result['guardrails_result'].report.numeric_out_of_bounds,
                            'config_version': retrieval_result['guardrails_result'].report.config_version
                        }
                
                return response_data
                
            except ValueError as e:
                print(f"Invalid LLM response format: {e}")
                # Fall back to insufficient context response
                return self.handle_insufficient_context(query, retrieval_result['filtered_chunks'])
        
        else:
            # Mock response for testing
            print("Generating mock response (no LLM provided)")
            
            # Create mock citations from context
            citations = []
            for chunk in retrieval_result['filtered_chunks'][:3]:  # Top 3 chunks
                if chunk.get('section_id'):
                    citations.append({
                        'section_id': chunk['section_id'],
                        'snippet': chunk['text'][:150] + "..." if len(chunk['text']) > 150 else chunk['text']
                    })
            
            mock_response = {
                'answer': f"Based on the retrieved Employment Act provisions, here is the guidance for your query about: {query[:100]}...",
                'citations': citations,
                'confidence': retrieval_result['estimated_confidence'],
                'should_escalate': retrieval_result['guardrails_result'].should_escalate if retrieval_result['guardrails_result'] else False,
                'safety_flags': retrieval_result['guardrails_result'].safety_flags if retrieval_result['guardrails_result'] else []
            }
            
            # Add guardrails report to mock response too
            if retrieval_result['guardrails_result'] and hasattr(retrieval_result['guardrails_result'], 'report') and retrieval_result['guardrails_result'].report:
                mock_response['guardrails_report'] = {
                    'timestamp': retrieval_result['guardrails_result'].report.timestamp,
                    'decision': retrieval_result['guardrails_result'].report.decision,
                    'confidence': retrieval_result['guardrails_result'].report.confidence,
                    'processing_time_ms': retrieval_result['guardrails_result'].report.processing_time_ms,
                    'input_flags': retrieval_result['guardrails_result'].report.input_flags,
                    'config_version': retrieval_result['guardrails_result'].report.config_version
                }
            
            return mock_response


def main():
    """Test the RAG pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test Employment Act RAG pipeline")
    parser.add_argument('--faiss', required=True, help='Path to FAISS index')
    parser.add_argument('--store', required=True, help='Path to store pickle file')
    parser.add_argument('--query', required=True, help='Test query')
    parser.add_argument('--embedding-model', default="intfloat/e5-large-v2")
    parser.add_argument('--reranker-model', default="cross-encoder/ms-marco-MiniLM-L-12-v2")
    
    args = parser.parse_args()
    
    # Initialize RAG pipeline
    config = RAGConfig(
        faiss_index_path=Path(args.faiss),
        store_path=Path(args.store),
        embedding_model=args.embedding_model,
        reranker_model=args.reranker_model
    )
    
    rag = EmploymentActRAG(config)
    
    # Process query
    result = rag.process_query(args.query)
    
    # Display result
    print(f"\n{'='*60}")
    print("RAG PIPELINE RESULT")
    print(f"{'='*60}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()