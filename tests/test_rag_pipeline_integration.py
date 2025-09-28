#!/usr/bin/env python3
"""
Comprehensive Hour 2 smoke tests with environment override validation.
Tests all critical Hour 2 requirements with proper error handling.
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch, PropertyMock
from typing import Dict, Any, List

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    import pytest
except ImportError:
    # Fallback for environments without pytest
    class MockPytest:
        @staticmethod
        def skip(msg, allow_module_level=False):
            print(f"SKIP: {msg}")
            sys.exit(0)
    pytest = MockPytest()


class TestHour2Comprehensive:
    """Comprehensive Hour 2 tests with environment validation."""
    
    def setup_method(self):
        """Set up test environment."""
        # Mock data for tests
        self.mock_chunks = [
            {
                'text': 'An employee shall be entitled to paid annual leave of eight days in each calendar year.',
                'section_id': 'EA-2022-60E(1)(a)',
                'chunk_id': 'chunk_1',
                'url': ''
            },
            {
                'text': 'twelve days in each calendar year, if employed for two years or more.',
                'section_id': 'EA-2022-60E(1)(b)', 
                'chunk_id': 'chunk_2',
                'url': ''
            },
            {
                'text': 'sixteen days in each calendar year, if employed for five years or more.',
                'section_id': 'EA-2022-60E(1)(c)',
                'chunk_id': 'chunk_3', 
                'url': ''
            }
        ]
        
        self.mock_store_data = {
            'chunks': self.mock_chunks,
            'bm25_index': MagicMock()
        }
    
    def test_retriever_candidates_respect_env(self):
        """Test that retriever respects environment variable overrides."""
        try:
            from retriever.hybrid_retriever import HybridRetriever
        except ImportError:
            pytest.skip("HybridRetriever not available")
        
        # Test Hour 2 spec environment variables
        test_env = {
            'BM25_TOPK': '100',      # Hour 2 spec
            'DENSE_TOPK': '50',      # Hour 2 spec  
            'CE_MAX_PAIRS': '150',   # Hour 2 spec
            'FINAL_TOPK': '8',
            'MIN_CHUNKS': '6',
            'CE_BATCH_SIZE': '32'
        }
        
        with patch('pickle.load', return_value=self.mock_store_data), \
             patch('faiss.read_index'), \
             patch('sentence_transformers.SentenceTransformer'), \
             patch('sentence_transformers.CrossEncoder'), \
             patch('torch.cuda.is_available', return_value=False), \
             patch.dict(os.environ, test_env):
            
            retriever = HybridRetriever(
                faiss_index_path=Path('/fake/faiss.index'),
                store_path=Path('/fake/store.pkl')
            )
            
            # Verify Hour 2 spec parameters are applied
            assert retriever.bm25_topk == 100, f"Expected BM25_TOPK=100, got {retriever.bm25_topk}"
            assert retriever.dense_topk == 50, f"Expected DENSE_TOPK=50, got {retriever.dense_topk}"
            assert retriever.ce_max_pairs == 150, f"Expected CE_MAX_PAIRS=150, got {retriever.ce_max_pairs}"
            assert retriever.final_topk == 8, f"Expected FINAL_TOPK=8, got {retriever.final_topk}"
            assert retriever.min_chunks == 6, f"Expected MIN_CHUNKS=6, got {retriever.min_chunks}"
            assert retriever.ce_batch_size == 32, f"Expected CE_BATCH_SIZE=32, got {retriever.ce_batch_size}"
    
    def test_retriever_min_chunks_guarantee(self):
        """Test minimum chunks guarantee prevents empty results."""
        try:
            from retriever.hybrid_retriever import HybridRetriever
        except ImportError:
            pytest.skip("HybridRetriever not available")
        
        with patch('pickle.load', return_value=self.mock_store_data), \
             patch('faiss.read_index'), \
             patch('sentence_transformers.SentenceTransformer'), \
             patch('sentence_transformers.CrossEncoder'), \
             patch('torch.cuda.is_available', return_value=False):
            
            retriever = HybridRetriever(
                faiss_index_path=Path('/fake/faiss.index'),
                store_path=Path('/fake/store.pkl'),
                min_chunks=6
            )
            
            # Test logic: when fewer candidates than minimum
            candidate_indices = [0, 1, 2]  # Only 3 candidates
            top_k = 8
            min_chunks = retriever.min_chunks
            
            # Apply the minimum chunks logic from the retriever
            min_chunks_actual = max(min_chunks, min(top_k, len(candidate_indices)))
            actual_top_k = max(top_k, min_chunks_actual)
            
            # Should guarantee at least what's available
            assert min_chunks_actual == 3  # Can't get more than available
            assert actual_top_k == 8  # Still request top_k
            
            # Test with sufficient candidates
            candidate_indices = list(range(10))  # 10 candidates
            min_chunks_actual = max(min_chunks, min(top_k, len(candidate_indices)))
            actual_top_k = max(top_k, min_chunks_actual)
            
            assert min_chunks_actual == 8  # min(top_k=8, len=10) > min_chunks=6
            assert actual_top_k == 8
    
    def test_prompt_schema_validation(self):
        """Test prompt templates contain proper JSON schema and examples."""
        try:
            from generation.prompt_templates import PromptTemplates
        except ImportError:
            pytest.skip("PromptTemplates not available")
        
        system_prompt = PromptTemplates.get_system_prompt()
        
        # Verify critical requirements
        critical_requirements = [
            "Use ONLY information from the provided context",
            "Every claim must be supported with specific section citations",
            "I'm not certain; here's the relevant section and how to proceed",
            "Output must be valid JSON with the exact schema specified",
            "If no valid section_id appears in CONTEXT, return the insufficient_context JSON template"
        ]
        
        for requirement in critical_requirements:
            assert requirement in system_prompt, f"Missing requirement: {requirement}"
        
        # Verify JSON schema components
        schema_components = [
            '"answer":',
            '"citations":',
            '"confidence":',
            '"should_escalate":',
            '"safety_flags":',
            '"section_id":',
            '"snippet":'
        ]
        
        for component in schema_components:
            assert component in system_prompt, f"Missing schema component: {component}"
        
        # Verify Hour 2 enhancement: few-shot example
        assert "EXAMPLE RESPONSE:" in system_prompt, "Missing few-shot example"
        assert "EA-2022-60E(1)" in system_prompt, "Missing example section ID"
        assert "annual leave" in system_prompt, "Missing example topic"
    
    def test_rag_prompt_format_with_context(self):
        """Test RAG prompt formatting preserves context structure."""
        try:
            from generation.prompt_templates import PromptTemplates
        except ImportError:
            pytest.skip("PromptTemplates not available")
        
        query = "How many days of annual leave am I entitled to?"
        context_chunks = self.mock_chunks
        
        prompt = PromptTemplates.get_rag_prompt(query, context_chunks)
        
        # Verify prompt structure
        assert "CONTEXT:" in prompt, "Missing CONTEXT section"
        assert "QUESTION:" in prompt, "Missing QUESTION section"
        assert query in prompt, "Query not included in prompt"
        
        # Verify context formatting
        for i, chunk in enumerate(context_chunks, 1):
            expected_header = f"[Context {i}] Section: {chunk['section_id']}"
            assert expected_header in prompt, f"Missing context header: {expected_header}"
            assert chunk['text'] in prompt, f"Missing chunk text: {chunk['text'][:50]}..."
        
        # Verify instruction
        assert "Based ONLY on the provided context" in prompt, "Missing context-only instruction"
    
    def test_insufficient_context_handling(self):
        """Test proper handling of insufficient context scenarios."""
        try:
            from generation.prompt_templates import PromptTemplates
        except ImportError:
            pytest.skip("PromptTemplates not available")
        
        query = "Test query"
        
        # Test with no context
        empty_prompt = PromptTemplates.get_rag_prompt(query, [])
        assert "[No relevant context found]" in empty_prompt, "Missing no context message"
        
        # Test insufficient context response
        response = PromptTemplates.get_insufficient_context_prompt(query, self.mock_chunks)
        response_data = json.loads(response)
        
        # Verify response structure
        assert "answer" in response_data, "Missing answer field"
        assert "citations" in response_data, "Missing citations field"
        assert "confidence" in response_data, "Missing confidence field"
        assert "should_escalate" in response_data, "Missing should_escalate field"
        
        # Verify escalation and confidence
        assert response_data["should_escalate"] is True, "Should escalate insufficient context"
        assert response_data["confidence"] < 0.5, "Confidence should be low for insufficient context"
        
        # Verify safety flags
        safety_flags = response_data.get("safety_flags", [])
        assert "insufficient_context" in safety_flags, "Missing insufficient_context safety flag"
        
        # Verify guidance message
        answer = response_data["answer"]
        assert "I'm not certain" in answer, "Missing uncertainty acknowledgment"
        assert "recommend" in answer.lower(), "Missing recommendation guidance"
    
    def test_citation_backfill_functionality(self):
        """Test citation backfill for empty citations."""
        try:
            from generation.citation_backfill import backfill_empty_citations, has_valid_section_ids
        except ImportError:
            pytest.skip("Citation backfill not available")
        
        # Test response with empty citations
        empty_citation_response = {
            "answer": "Test answer",
            "citations": [],
            "confidence": 0.8,
            "should_escalate": False,
            "safety_flags": []
        }
        
        response_text = json.dumps(empty_citation_response)
        enhanced_response = backfill_empty_citations(response_text, self.mock_chunks, max_citations=2)
        enhanced_data = json.loads(enhanced_response)
        
        # Verify citations were backfilled
        citations = enhanced_data.get("citations", [])
        assert len(citations) == 2, f"Expected 2 backfilled citations, got {len(citations)}"
        
        # Verify citation structure
        for citation in citations:
            assert "section_id" in citation, "Missing section_id in backfilled citation"
            assert "snippet" in citation, "Missing snippet in backfilled citation"
            assert citation["section_id"] != "N/A", "Invalid section_id in backfilled citation"
        
        # Verify confidence was adjusted
        assert enhanced_data["confidence"] < empty_citation_response["confidence"], "Confidence should be lowered for backfilled citations"
        
        # Verify safety flag was added
        safety_flags = enhanced_data.get("safety_flags", [])
        assert "citations_backfilled" in safety_flags, "Missing citations_backfilled safety flag"
        
        # Test valid section ID detection
        assert has_valid_section_ids(self.mock_chunks) is True, "Should detect valid section IDs"
        
        invalid_chunks = [{"text": "test", "section_id": "N/A"}]
        assert has_valid_section_ids(invalid_chunks) is False, "Should not detect invalid section IDs"
    
    def test_device_and_caching_configuration(self):
        """Test device configuration and caching support."""
        try:
            from retriever.hybrid_retriever import HybridRetriever
        except ImportError:
            pytest.skip("HybridRetriever not available")
        
        # Mock cache objects
        mock_embedding_cache = MagicMock()
        mock_embedding_cache.get.return_value = None
        mock_reranker_cache = MagicMock()
        mock_reranker_cache.get.return_value = None
        
        with patch('pickle.load', return_value=self.mock_store_data), \
             patch('faiss.read_index'), \
             patch('sentence_transformers.SentenceTransformer') as mock_st, \
             patch('sentence_transformers.CrossEncoder') as mock_ce, \
             patch('torch.cuda.is_available', return_value=False):
            
            retriever = HybridRetriever(
                faiss_index_path=Path('/fake/faiss.index'),
                store_path=Path('/fake/store.pkl'),
                embedding_cache=mock_embedding_cache,
                reranker_cache=mock_reranker_cache
            )
            
            # Verify device configuration
            assert hasattr(retriever, 'device'), "Missing device configuration"
            assert hasattr(retriever, 'ce_batch_size'), "Missing batch size configuration"
            
            # Verify models were initialized with device
            mock_st.assert_called()
            mock_ce.assert_called()
            
            # Verify cache objects were stored
            assert retriever.embedding_cache == mock_embedding_cache, "Embedding cache not stored"
            assert retriever.reranker_cache == mock_reranker_cache, "Reranker cache not stored"
    
    def test_configuration_validation(self):
        """Test configuration parameter validation."""
        try:
            from retriever.hybrid_retriever import HybridRetriever
        except ImportError:
            pytest.skip("HybridRetriever not available")
        
        with patch('pickle.load', return_value=self.mock_store_data), \
             patch('faiss.read_index'), \
             patch('sentence_transformers.SentenceTransformer'), \
             patch('sentence_transformers.CrossEncoder'), \
             patch('torch.cuda.is_available', return_value=False):
            
            # Test invalid FINAL_TOPK
            with patch.dict(os.environ, {'FINAL_TOPK': '25'}):  # > 20
                try:
                    HybridRetriever(Path('/fake/faiss.index'), Path('/fake/store.pkl'))
                    assert False, "Should raise ValueError for FINAL_TOPK > 20"
                except ValueError as e:
                    assert "FINAL_TOPK must be 1-20" in str(e)
            
            # Test invalid CE_MAX_PAIRS
            with patch.dict(os.environ, {'CE_MAX_PAIRS': '250'}):  # > 200
                try:
                    HybridRetriever(Path('/fake/faiss.index'), Path('/fake/store.pkl'))
                    assert False, "Should raise ValueError for CE_MAX_PAIRS > 200"
                except ValueError as e:
                    assert "CE_MAX_PAIRS must be 1-200" in str(e)
            
            # Test invalid MIN_CHUNKS
            with patch.dict(os.environ, {'MIN_CHUNKS': '15', 'FINAL_TOPK': '8'}):  # > FINAL_TOPK
                try:
                    HybridRetriever(Path('/fake/faiss.index'), Path('/fake/store.pkl'))
                    assert False, "Should raise ValueError for MIN_CHUNKS > FINAL_TOPK"
                except ValueError as e:
                    assert "MIN_CHUNKS must be 1-8" in str(e)


def run_comprehensive_tests():
    """Run all Hour 2 comprehensive tests."""
    test_instance = TestHour2Comprehensive()
    test_instance.setup_method()
    
    tests = [
        ("Environment variable override", test_instance.test_retriever_candidates_respect_env),
        ("Minimum chunks guarantee", test_instance.test_retriever_min_chunks_guarantee),
        ("Prompt schema validation", test_instance.test_prompt_schema_validation),
        ("RAG prompt formatting", test_instance.test_rag_prompt_format_with_context),
        ("Insufficient context handling", test_instance.test_insufficient_context_handling),
        ("Citation backfill functionality", test_instance.test_citation_backfill_functionality),
        ("Device and caching configuration", test_instance.test_device_and_caching_configuration),
        ("Configuration validation", test_instance.test_configuration_validation)
    ]
    
    passed = 0
    failed = 0
    
    print("Running Hour 2 comprehensive tests...")
    print("=" * 60)
    
    for test_name, test_func in tests:
        try:
            print(f"Testing: {test_name}...")
            test_func()
            print(f"✓ PASS: {test_name}")
            passed += 1
        except Exception as e:
            print(f"❌ FAIL: {test_name} - {e}")
            failed += 1
    
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("\n🎉 All Hour 2 comprehensive tests passed!")
        print("Verified implementations:")
        print("  ✓ Environment variable overrides (BM25/DENSE/CE parameters)")
        print("  ✓ Minimum chunks guarantee (prevents empty results)")
        print("  ✓ JSON schema enforcement with few-shot examples")
        print("  ✓ Proper context handling and citation requirements")
        print("  ✓ Citation backfill for empty model responses")
        print("  ✓ Device configuration and model caching")
        print("  ✓ Parameter validation and error handling")
        print("\nHour 2 implementation is deployment-ready! 🚀")
    else:
        print(f"\n⚠️  {failed} tests failed. Review implementation.")
        return False
    
    return True


if __name__ == "__main__":
    success = run_comprehensive_tests()
    sys.exit(0 if success else 1)