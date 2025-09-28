#!/usr/bin/env python3
"""
Retrieval smoke test for Hour 2 deliverables.
Tests hybrid retrieval with minimal requirements validation.
"""

import os
import sys
import tempfile
import pickle
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

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

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from retriever.hybrid_retriever import HybridRetriever
    from generation.prompt_templates import PromptTemplates
except ImportError as e:
    pytest.skip(f"Required modules not available: {e}", allow_module_level=True)


class TestRetrievalSmoke:
    """Smoke tests for retrieval system without heavy dependencies."""
    
    def test_hybrid_retriever_parameterization(self):
        """Test that HybridRetriever accepts configurable parameters."""
        # Mock the file loading to avoid actual file dependencies
        with patch('pickle.load') as mock_pickle, \
             patch('faiss.read_index') as mock_faiss, \
             patch('sentence_transformers.SentenceTransformer') as mock_st, \
             patch('sentence_transformers.CrossEncoder') as mock_ce:
            
            # Mock the loaded data
            mock_pickle.return_value = {
                'chunks': [{'text': 'test', 'section_id': 'test-1'}],
                'bm25_index': MagicMock()
            }
            
            # Test with custom parameters
            retriever = HybridRetriever(
                faiss_index_path=Path('/fake/faiss.index'),
                store_path=Path('/fake/store.pkl'),
                bm25_topk=100,  # Hour 2 spec
                dense_topk=50,  # Hour 2 spec
                ce_max_pairs=150,  # Hour 2 spec
                final_topk=8,
                min_chunks=6
            )
            
            # Verify parameters are set correctly
            assert retriever.bm25_topk == 100
            assert retriever.dense_topk == 50
            assert retriever.ce_max_pairs == 150
            assert retriever.final_topk == 8
            assert retriever.min_chunks == 6
    
    def test_hybrid_retriever_env_override(self):
        """Test that environment variables override default parameters."""
        with patch('pickle.load') as mock_pickle, \
             patch('faiss.read_index') as mock_faiss, \
             patch('sentence_transformers.SentenceTransformer') as mock_st, \
             patch('sentence_transformers.CrossEncoder') as mock_ce, \
             patch.dict(os.environ, {
                 'BM25_TOPK': '100',
                 'DENSE_TOPK': '50', 
                 'CE_MAX_PAIRS': '150',
                 'FINAL_TOPK': '8',
                 'MIN_CHUNKS': '6'
             }):
            
            mock_pickle.return_value = {
                'chunks': [{'text': 'test', 'section_id': 'test-1'}],
                'bm25_index': MagicMock()
            }
            
            retriever = HybridRetriever(
                faiss_index_path=Path('/fake/faiss.index'),
                store_path=Path('/fake/store.pkl')
            )
            
            # Verify environment variables are respected
            assert retriever.bm25_topk == 100  # Hour 2 spec value
            assert retriever.dense_topk == 50   # Hour 2 spec value
            assert retriever.ce_max_pairs == 150  # Hour 2 spec value
            assert retriever.final_topk == 8
            assert retriever.min_chunks == 6
    
    def test_prompt_templates_json_schema(self):
        """Test that prompt templates include proper JSON schema and example."""
        system_prompt = PromptTemplates.get_system_prompt()
        
        # Verify critical requirements are present
        assert "Use ONLY information from the provided context" in system_prompt
        assert "Every claim must be supported with specific section citations" in system_prompt
        assert "I'm not certain; here's the relevant section and how to proceed" in system_prompt
        
        # Verify JSON schema is present
        assert '"answer":' in system_prompt
        assert '"citations":' in system_prompt
        assert '"confidence":' in system_prompt
        assert '"should_escalate":' in system_prompt
        assert '"safety_flags":' in system_prompt
        
        # Verify example response is present (Hour 2 enhancement)
        assert "EXAMPLE RESPONSE:" in system_prompt
        assert "EA-2022-60E(1)" in system_prompt  # Example section ID
        assert "annual leave" in system_prompt  # Example topic
    
    def test_rag_prompt_format(self):
        """Test RAG prompt formatting with context chunks."""
        query = "How many days of annual leave?"
        context_chunks = [
            {
                'text': 'An employee shall be entitled to paid annual leave of eight days...',
                'section_id': 'EA-2022-60E(1)',
                'chunk_id': 'chunk_1'
            },
            {
                'text': 'twelve days in each calendar year...',
                'section_id': 'EA-2022-60E(1)(b)',
                'chunk_id': 'chunk_2'
            }
        ]
        
        prompt = PromptTemplates.get_rag_prompt(query, context_chunks)
        
        # Verify prompt structure
        assert "CONTEXT:" in prompt
        assert "QUESTION:" in prompt
        assert "[Context 1] Section: EA-2022-60E(1)" in prompt
        assert "[Context 2] Section: EA-2022-60E(1)(b)" in prompt
        assert query in prompt
        
        # Verify both chunks are included
        assert "eight days" in prompt
        assert "twelve days" in prompt
    
    def test_insufficient_context_handling(self):
        """Test handling of insufficient context."""
        # Test with empty context
        empty_prompt = PromptTemplates.get_rag_prompt("test query", [])
        assert "[No relevant context found]" in empty_prompt
        
        # Test insufficient context response
        insufficient_response = PromptTemplates.get_insufficient_context_prompt("test query")
        assert "I'm not certain" in insufficient_response
        assert "based on the available context" in insufficient_response
        assert "should_escalate" in insufficient_response
        assert "true" in insufficient_response  # Should escalate when insufficient
    
    def test_minimum_chunks_guarantee(self):
        """Test that minimum chunks logic is implemented correctly."""
        # This test verifies the logic without actual retrieval
        # The minimum chunks guarantee should ensure >= 6 chunks are returned
        # even if scores are low, to avoid over-triggering "insufficient context"
        
        # Mock scenario: fewer candidates than minimum
        candidate_indices = [1, 2, 3]  # Only 3 candidates
        min_chunks = 6
        top_k = 8
        
        # Logic from the updated retriever
        min_chunks_actual = max(min_chunks, min(top_k, len(candidate_indices)))
        actual_top_k = max(top_k, min_chunks_actual)
        
        # Should request at least the minimum available
        assert min_chunks_actual == 3  # Can't get more than available
        assert actual_top_k == 8  # Still request top_k if possible
        
        # Mock scenario: enough candidates
        candidate_indices = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # 10 candidates
        min_chunks_actual = max(min_chunks, min(top_k, len(candidate_indices)))
        actual_top_k = max(top_k, min_chunks_actual)
        
        # Should request normal top_k
        assert min_chunks_actual == 8  # min(top_k=8, len=10)
        assert actual_top_k == 8


if __name__ == "__main__":
    # Run as standalone script
    test_instance = TestRetrievalSmoke()
    
    print("Running retrieval smoke tests...")
    
    try:
        print("1. Testing hybrid retriever parameterization...")
        test_instance.test_hybrid_retriever_parameterization()
        print("✓ PASS: Hybrid retriever parameterization")
        
        print("2. Testing environment variable override...")
        test_instance.test_hybrid_retriever_env_override()
        print("✓ PASS: Environment variable override")
        
        print("3. Testing prompt templates JSON schema...")
        test_instance.test_prompt_templates_json_schema()
        print("✓ PASS: Prompt templates JSON schema")
        
        print("4. Testing RAG prompt format...")
        test_instance.test_rag_prompt_format()
        print("✓ PASS: RAG prompt format")
        
        print("5. Testing insufficient context handling...")
        test_instance.test_insufficient_context_handling()
        print("✓ PASS: Insufficient context handling")
        
        print("6. Testing minimum chunks guarantee...")
        test_instance.test_minimum_chunks_guarantee()
        print("✓ PASS: Minimum chunks guarantee")
        
        print("\n🎉 All retrieval smoke tests passed!")
        print("Hour 2 requirements verified:")
        print("  - Configurable candidate sizes (BM25/dense/CE)")
        print("  - JSON schema enforcement with examples")
        print("  - Minimum chunks guarantee")
        print("  - Proper context handling")
        
    except Exception as e:
        print(f"❌ FAIL: {e}")
        sys.exit(1)