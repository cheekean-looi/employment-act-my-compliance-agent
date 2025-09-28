#!/usr/bin/env python3
# Mock mode: python -m src.generation.rag_pipeline --faiss data/indices/faiss.index --store data/indices/store.pkl --query "your query here"
# Real model: python -m src.generation.rag_pipeline --faiss data/indices/faiss.index --store data/indices/store.pkl --query "your query here" --json
"""
RAG pipeline for Employment Act Malaysia compliance agent.
Integrates hybrid retrieval, prompt templates, guardrails, and LLM generation.
"""

import json
import os
import sys
import asyncio
from pathlib import Path
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ..retriever.hybrid_retriever import HybridRetriever
from .prompt_templates import PromptTemplates, Response
from .guardrails import ProductionGuardrailsEngine, GuardrailResult
from .json_schemas import JSONGenerator, EmploymentActResponse
from .citation_backfill import backfill_empty_citations, has_valid_section_ids
from ..server.vllm_client import VLLMClient, GenerationConfig


@dataclass
class RAGConfig:
    """Configuration for RAG pipeline."""
    faiss_index_path: Path
    store_path: Path
    embedding_model: str = "BAAI/bge-m3"
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-2-v2"
    retrieval_top_k: int = 8
    min_context_score: float = 0.3
    enable_guardrails: bool = True
    embedding_cache: Optional[Any] = None
    reranker_cache: Optional[Any] = None


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
            reranker_model=config.reranker_model,
            embedding_cache=config.embedding_cache,
            reranker_cache=config.reranker_cache
        )
        
        self.prompt_templates = PromptTemplates()
        self.json_generator = JSONGenerator()
        
        if config.enable_guardrails:
            self.guardrails = ProductionGuardrailsEngine()
        else:
            self.guardrails = None
        
        print("RAG pipeline initialized successfully")
    
    async def process_query_with_json(self, query: str, llm_generate_func=None) -> Dict[str, Any]:
        """
        Process query with constrained JSON generation and auto-repair.
        Implements the new best practice pipeline with guaranteed citations.
        
        Args:
            query: User query
            llm_generate_func: Function to call LLM with JSON constraints
            
        Returns:
            Structured response with validated JSON format
        """
        print(f"Processing query with JSON constraints: '{query[:50]}...'")
        
        # Step 1: Retrieve and evaluate
        retrieval_result = self.retrieve_and_evaluate(query)
        
        return await self.process_query_with_json_from_retrieval(query, retrieval_result, llm_generate_func)
    
    async def process_query_with_json_from_retrieval(self, query: str, retrieval_result: Dict[str, Any], llm_generate_func=None) -> Dict[str, Any]:
        """
        Process query with pre-computed retrieval result to avoid double retrieval.
        Optimized version that reuses retrieval_result for better performance.
        
        Args:
            query: User query
            retrieval_result: Pre-computed retrieval result from retrieve_and_evaluate
            llm_generate_func: Function to call LLM with JSON constraints
            
        Returns:
            Structured response with validated JSON format
        """
        print(f"Processing query with JSON constraints from pre-computed retrieval: '{query[:50]}...'")
        
        # Skip Step 1: Use provided retrieval_result
        
        # Step 2: Check for refusal cases
        if retrieval_result['guardrails_result'] and retrieval_result['guardrails_result'].should_refuse:
            print("Query refused by guardrails")
            return self.handle_refusal(query, retrieval_result['guardrails_result'])
        
        # Step 3: Always keep minimum 6-8 chunks, never drop all
        context_chunks = retrieval_result['retrieved_chunks']
        if len(context_chunks) < 6:
            # If we have fewer than 6 chunks, that's all we have
            filtered_chunks = context_chunks
        else:
            # Apply quality filter but keep at least 6-8
            filtered_chunks = self._filter_low_quality_chunks(context_chunks)
            if len(filtered_chunks) < 6:
                # Fall back to top 8 chunks regardless of score
                filtered_chunks = context_chunks[:8]
        
        print(f"Using {len(filtered_chunks)} context chunks (minimum 6-8 guaranteed)")
        
        # Step 4: Generate JSON prompt with few-shot examples
        json_prompt = self.json_generator.create_prompt_with_examples(query, filtered_chunks)
        
        # Step 5: Generate response with JSON constraints
        if llm_generate_func:
            # First attempt with JSON constraints
            try:
                llm_response = await llm_generate_func(json_prompt, json_constrained=True)
                
                # Validate response against schema
                validated_response, validation_errors = self.json_generator.validate_response(llm_response)
                
                if validated_response:
                    # Check if citations are empty and backfill if needed (Hour 2 integration)
                    final_response = self._format_final_response(validated_response, retrieval_result)
                    
                    # Apply citation backfill if citations are empty
                    if not final_response.get("citations") and has_valid_section_ids(filtered_chunks):
                        print("🔄 Backfilling empty citations from context")
                        final_response_json = json.dumps(final_response)
                        enhanced_response_json = backfill_empty_citations(final_response_json, filtered_chunks, max_citations=3)
                        enhanced_response = json.loads(enhanced_response_json)
                        
                        # Update guardrails report if present
                        if "guardrails_report" in final_response and "guardrails_report" in enhanced_response:
                            enhanced_response["guardrails_report"] = final_response["guardrails_report"]
                        
                        print("✅ Valid JSON response generated with citation backfill")
                        return enhanced_response
                    else:
                        print("✅ Valid JSON response generated")
                        return final_response
                else:
                    print(f"❌ JSON validation failed: {validation_errors}")
                    
                    # Attempt auto-repair
                    repaired_response = self.json_generator.auto_repair_response(
                        llm_response, filtered_chunks, validation_errors
                    )
                    
                    if repaired_response:
                        print("🔧 Successfully auto-repaired response")
                        return self._format_final_response(repaired_response, retrieval_result)
                    else:
                        print("🚨 Auto-repair failed, creating fallback response")
                        return self._create_fallback_response(query, filtered_chunks, retrieval_result)
                        
            except Exception as e:
                print(f"🚨 LLM generation failed: {e}")
                return self._create_fallback_response(query, filtered_chunks, retrieval_result)
        else:
            # Mock response for testing
            return self._create_mock_json_response(query, filtered_chunks, retrieval_result)
            return self.handle_refusal(query, retrieval_result['guardrails_result'])
        
        # Step 3: Always keep minimum 6-8 chunks, never drop all
        context_chunks = retrieval_result['retrieved_chunks']
        if len(context_chunks) < 6:
            # If we have fewer than 6 chunks, that's all we have
            filtered_chunks = context_chunks
        else:
            # Apply quality filter but keep at least 6-8
            filtered_chunks = self._filter_low_quality_chunks(context_chunks)
            if len(filtered_chunks) < 6:
                # Fall back to top 8 chunks regardless of score
                filtered_chunks = context_chunks[:8]
        
        print(f"Using {len(filtered_chunks)} context chunks (minimum 6-8 guaranteed)")
        
        # Step 4: Generate JSON prompt with few-shot examples
        json_prompt = self.json_generator.create_prompt_with_examples(query, filtered_chunks)
        
        # Step 5: Generate response with JSON constraints
        if llm_generate_func:
            # First attempt with JSON constraints
            try:
                llm_response = await llm_generate_func(json_prompt, json_constrained=True)
                
                # Validate response against schema
                validated_response, validation_errors = self.json_generator.validate_response(llm_response)
                
                if validated_response:
                    # Check if citations are empty and backfill if needed (Hour 2 integration)
                    final_response = self._format_final_response(validated_response, retrieval_result)
                    
                    # Apply citation backfill if citations are empty
                    if not final_response.get("citations") and has_valid_section_ids(filtered_chunks):
                        print("🔄 Backfilling empty citations from context")
                        final_response_json = json.dumps(final_response)
                        enhanced_response_json = backfill_empty_citations(final_response_json, filtered_chunks, max_citations=3)
                        enhanced_response = json.loads(enhanced_response_json)
                        
                        # Update guardrails report if present
                        if "guardrails_report" in final_response and "guardrails_report" in enhanced_response:
                            enhanced_response["guardrails_report"] = final_response["guardrails_report"]
                        
                        print("✅ Valid JSON response generated with citation backfill")
                        return enhanced_response
                    else:
                        print("✅ Valid JSON response generated")
                        return final_response
                else:
                    print(f"❌ JSON validation failed: {validation_errors}")
                    
                    # Attempt auto-repair
                    repaired_response = self.json_generator.auto_repair_response(
                        llm_response, filtered_chunks, validation_errors
                    )
                    
                    if repaired_response:
                        print("🔧 Successfully auto-repaired response")
                        return self._format_final_response(repaired_response, retrieval_result)
                    else:
                        print("🚨 Auto-repair failed, creating fallback response")
                        return self._create_fallback_response(query, filtered_chunks, retrieval_result)
                        
            except Exception as e:
                print(f"🚨 LLM generation failed: {e}")
                return self._create_fallback_response(query, filtered_chunks, retrieval_result)
        else:
            # Mock response for testing
            return self._create_mock_json_response(query, filtered_chunks, retrieval_result)
    
    def _format_final_response(self, validated_response: EmploymentActResponse, retrieval_result: Dict[str, Any]) -> Dict[str, Any]:
        """Format validated JSON response into final API response format."""
        response_dict = {
            "answer": validated_response.answer,
            "citations": [
                {
                    "section_id": citation.section_id,
                    "snippet": citation.snippet,
                    "relevance_score": citation.relevance_score
                }
                for citation in validated_response.citations
            ],
            "confidence": validated_response.confidence.value if hasattr(validated_response.confidence, 'value') else validated_response.confidence,
            "should_escalate": validated_response.should_escalate,
            "safety_flags": validated_response.safety_flags,
            "numerical_claims": validated_response.numerical_claims or {},
        }
        
        # Add guardrails metadata
        if retrieval_result['guardrails_result']:
            guardrails_result = retrieval_result['guardrails_result']
            response_dict["safety_flags"].extend(guardrails_result.safety_flags)
            
            if hasattr(guardrails_result, 'report') and guardrails_result.report:
                response_dict['guardrails_report'] = {
                    'timestamp': guardrails_result.report.timestamp,
                    'decision': guardrails_result.report.decision,
                    'confidence': guardrails_result.report.confidence,
                    'processing_time_ms': guardrails_result.report.processing_time_ms,
                    'config_version': guardrails_result.report.config_version
                }
        
        # Add retrieval metadata
        response_dict.update({
            "retrieved_chunks": len(retrieval_result['retrieved_chunks']),
            "filtered_chunks": len(retrieval_result['filtered_chunks']),
            "top_retrieval_score": max([chunk.get('score', 0) for chunk in retrieval_result['retrieved_chunks']], default=0.0),
            "processing_method": "json_constrained"
        })
        
        return response_dict
    
    def _create_fallback_response(self, query: str, chunks: List[Dict[str, Any]], retrieval_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create fallback response when JSON generation fails."""
        fallback_response = self.json_generator._create_fallback_response("", chunks)
        fallback_response.answer = f"I found relevant Employment Act provisions for your query, but need additional clarification. Based on the retrieved sections, please consult the specific provisions or seek legal guidance for your situation."
        fallback_response.should_escalate = True
        fallback_response.safety_flags.append("generation_failed")
        
        return self._format_final_response(fallback_response, retrieval_result)
    
    def _create_mock_json_response(self, query: str, chunks: List[Dict[str, Any]], retrieval_result: Dict[str, Any]) -> Dict[str, Any]:
        """Create mock JSON response for testing."""
        from .json_schemas import ConfidenceLevel, Citation
        
        mock_response = EmploymentActResponse(
            answer=f"Mock response for query: '{query}'. This is a test response from the Employment Act RAG pipeline with JSON schema validation.",
            citations=[
                Citation(
                    section_id=chunk.get('section_id', 'EA-TEST'),
                    snippet=chunk.get('text', 'Test snippet')[:150] + '...',
                    relevance_score=chunk.get('score', 0.5)
                )
                for chunk in chunks[:3]
            ],
            confidence=ConfidenceLevel.MEDIUM,
            should_escalate=False,
            safety_flags=["mock_response"],
            numerical_claims=None
        )
        
        return self._format_final_response(mock_response, retrieval_result)
    
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
                
                # Apply citation backfill if citations are empty (consistency with JSON path)
                if not response_data.get("citations") and has_valid_section_ids(retrieval_result['filtered_chunks']):
                    print("🔄 Backfilling empty citations in non-JSON LLM response")
                    response_json = json.dumps(response_data)
                    enhanced_response_json = backfill_empty_citations(response_json, retrieval_result['filtered_chunks'], max_citations=3)
                    response_data = json.loads(enhanced_response_json)
                
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
            
            # Apply citation backfill if citations are empty (Hour 2 integration)
            if not mock_response.get("citations") and has_valid_section_ids(retrieval_result['filtered_chunks']):
                print("🔄 Backfilling empty citations in mock response")
                mock_response_json = json.dumps(mock_response)
                enhanced_response_json = backfill_empty_citations(mock_response_json, retrieval_result['filtered_chunks'], max_citations=3)
                mock_response = json.loads(enhanced_response_json)
            
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
    parser.add_argument('--embedding-model', default="BAAI/bge-m3")
    parser.add_argument('--reranker-model', default="cross-encoder/ms-marco-MiniLM-L-2-v2")
    parser.add_argument('--json', action='store_true', 
                        help='Use real model with JSON pipeline (requires running vLLM server)')
    parser.add_argument('--vllm-base-url', 
                        default=os.getenv('VLLM_BASE_URL', 'http://localhost:8000'),
                        help='vLLM server base URL')
    parser.add_argument('--model-name',
                        default=os.getenv('MODEL_NAME', 'meta-llama/Llama-3.1-8B-Instruct'),
                        help='Model name for vLLM generation')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug mode with detailed retrieval metadata')
    
    args = parser.parse_args()
    
    # Initialize RAG pipeline
    try:
        config = RAGConfig(
            faiss_index_path=Path(args.faiss),
            store_path=Path(args.store),
            embedding_model=args.embedding_model,
            reranker_model=args.reranker_model
        )
        
        rag = EmploymentActRAG(config)
        
    except FileNotFoundError as e:
        print(f"❌ ERROR: Required index file not found")
        print(f"Details: {e}")
        print("\n💡 Please ensure indices are built:")
        print("   python src/ingest/build_index.py --in data/processed/chunks.jsonl --faiss data/indices/faiss.index --store data/indices/store.pkl")
        sys.exit(1)
    except Exception as e:
        print(f"❌ ERROR: Failed to initialize RAG pipeline")
        print(f"Details: {e}")
        sys.exit(1)
    
    # Get retrieval status for CLI output
    retrieval_status = rag.retriever.get_retrieval_status()
    
    # Display retrieval status
    print(f"\n{'='*60}")
    print("RETRIEVAL CONFIGURATION")
    print(f"{'='*60}")
    print(f"Mode: {retrieval_status['retrieval_mode'].upper()}")
    if retrieval_status['retrieval_mode'] == 'bm25_only':
        print("⚠️  FAISS DENSE RETRIEVAL UNAVAILABLE - BM25-ONLY MODE")
    print(f"BM25 top-k: {retrieval_status['bm25_topk']}")
    print(f"Dense top-k: {retrieval_status['dense_topk']}")
    print(f"Cross-encoder max pairs: {retrieval_status['ce_max_pairs']}")
    print(f"Final top-k: {retrieval_status['final_topk']}")
    print(f"Minimum chunks: {retrieval_status['min_chunks']}")
    
    # Debug mode: show detailed retrieval metadata
    if args.debug:
        print(f"\n{'='*60}")
        print("DEBUG: RETRIEVAL METADATA")
        print(f"{'='*60}")
        
        # Quick retrieval test to show debug info
        debug_results = rag.retriever.retrieve(args.query, top_k=8)
        print(f"Retrieved {len(debug_results)} chunks:")
        
        for i, chunk in enumerate(debug_results[:5], 1):  # Show top 5
            section_id = chunk.get('section_id', 'N/A')
            score = chunk.get('score', 0.0)
            text_preview = chunk.get('text', '')[:100] + '...' if len(chunk.get('text', '')) > 100 else chunk.get('text', '')
            print(f"  [{i}] Section: {section_id} | Score: {score:.4f}")
            print(f"      Text: {text_preview}")
        
        if len(debug_results) > 5:
            print(f"  ... and {len(debug_results) - 5} more chunks")
        print()
    
    # Process query using appropriate path
    if args.json:
        print(f"\n{'='*60}")
        print("USING REAL MODEL WITH JSON-CONSTRAINED GENERATION")
        print(f"{'='*60}")
        print(f"vLLM URL: {args.vllm_base_url}")
        print(f"Model: {args.model_name}")
        
        try:
            # Create VLLMClient
            vllm_client = VLLMClient(
                base_url=args.vllm_base_url,
                model_name=args.model_name
            )
            
            # Define async LLM function with error handling
            async def llm_generate_func(prompt: str, json_constrained: bool = False) -> str:
                """Generate response using vLLM with optional JSON constraints."""
                try:
                    config = GenerationConfig(
                        json_only=json_constrained,
                        response_format={"type": "json_object"} if json_constrained else None,
                        max_tokens=512,
                        temperature=0.0
                    )
                    result = await vllm_client.generate(prompt, config)
                    return result.text
                except Exception as e:
                    print(f"❌ vLLM generation failed: {e}")
                    raise
            
            # Run async query processing
            result = asyncio.run(rag.process_query_with_json(args.query, llm_generate_func=llm_generate_func))
            print("✅ Real model generation completed successfully")
            
        except Exception as e:
            print(f"\n❌ ERROR: Cannot reach vLLM server at {args.vllm_base_url}")
            print(f"Details: {e}")
            print("\n💡 Troubleshooting:")
            print(f"   1. Ensure vLLM server is running: python src/server/serve_vllm.py")
            print(f"   2. Check server URL: {args.vllm_base_url}")
            print(f"   3. Verify model is loaded: {args.model_name}")
            print(f"   4. Use mock mode for testing: remove --json flag")
            sys.exit(1)
    else:
        print(f"\n{'='*60}")  
        print("MOCK GENERATION MODE")
        print(f"{'='*60}")
        print("💡 Use --json flag for real model generation with running vLLM server")
        print("   Example: python -m src.generation.rag_pipeline --faiss ... --query '...' --json")
        result = rag.process_query(args.query)
    
    # Display result
    print(f"\n{'='*60}")
    print("RAG PIPELINE RESULT") 
    print(f"{'='*60}")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()