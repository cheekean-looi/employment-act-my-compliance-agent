#!/usr/bin/env python3
"""
Generation Evaluation Smoke Test
Quick test to verify attention_mask fix and shared validator integration.
"""

import unittest
import sys
import tempfile
import warnings
from pathlib import Path
from unittest.mock import patch, MagicMock
import pytest
# Require torch for this optional smoke test; skip cleanly if missing
torch = pytest.importorskip("torch", reason="Requires PyTorch")

# Project imports are resolved via pytest.ini (pythonpath = src)

from evals.generation_eval import LLMJudge, GenerationEvaluator
from evals.validation_utils import CitationValidator, NumericValidator

class TestGenerationEvalSmoke(unittest.TestCase):
    """Smoke tests for generation evaluation system."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Mock the transformers components to avoid downloading models
        self.mock_tokenizer = MagicMock()
        self.mock_model = MagicMock()
        
        # Configure mock tokenizer
        self.mock_tokenizer.pad_token = "[PAD]"
        self.mock_tokenizer.eos_token = "[EOS]"
        self.mock_tokenizer.eos_token_id = 2
        
        # Mock tokenizer call to return proper format
        def mock_tokenizer_call(*args, **kwargs):
            if isinstance(args[0], str):
                # Return mock tokenizer output with attention_mask
                mock_output = MagicMock()
                mock_output.__getitem__ = lambda self, key: torch.tensor([[1, 2, 3, 4, 5]]) if key == "input_ids" else torch.tensor([[1, 1, 1, 1, 1]])
                return mock_output
            return args[0]
        
        self.mock_tokenizer.side_effect = mock_tokenizer_call
        self.mock_tokenizer.apply_chat_template.return_value = "Formatted prompt"
        
        # Mock model generation
        self.mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])
        self.mock_tokenizer.decode.return_value = "GROUNDED: YES\nSCORE: 0.9\nREASONING: Valid response"
    
    @patch('evals.generation_eval.AutoModelForCausalLM')
    @patch('evals.generation_eval.AutoTokenizer')
    def test_attention_mask_fix(self, mock_tokenizer_class, mock_model_class):
        """Test that attention_mask is properly passed to model.generate()."""
        
        # Setup mocks
        mock_tokenizer_class.from_pretrained.return_value = self.mock_tokenizer
        mock_model_class.from_pretrained.return_value = self.mock_model
        
        # Initialize judge
        judge = LLMJudge("mock-model")
        
        # Verify shared validators are initialized
        self.assertIsNotNone(judge.citation_validator)
        self.assertIsNotNone(judge.numeric_validator)
        
        # Test groundedness evaluation
        result = judge.evaluate_groundedness(
            question="What is sick leave?",
            response="You are entitled to 14 days sick leave according to Section EA-60F.",
            reference_context="Employment Act context"
        )
        
        # Verify model.generate was called with attention_mask
        self.mock_model.generate.assert_called_once()
        call_kwargs = self.mock_model.generate.call_args[1]
        
        # Check that attention_mask was passed
        self.assertIn('attention_mask', call_kwargs)
        self.assertIsNotNone(call_kwargs['attention_mask'])
        
        # Verify result structure
        self.assertIsNotNone(result.is_grounded)
        self.assertIsInstance(result.groundedness_score, float)
        self.assertIsInstance(result.evidence_citations, list)
        self.assertIsInstance(result.hallucination_detected, bool)
        
        print("✅ Attention mask properly passed to model.generate()")
        print(f"✅ Groundedness result: {result.is_grounded}, score: {result.groundedness_score}")
    
    @patch('evals.generation_eval.AutoModelForCausalLM')
    @patch('evals.generation_eval.AutoTokenizer')
    def test_shared_validators_integration(self, mock_tokenizer_class, mock_model_class):
        """Test that shared validators are properly integrated."""
        
        # Setup mocks
        mock_tokenizer_class.from_pretrained.return_value = self.mock_tokenizer
        mock_model_class.from_pretrained.return_value = self.mock_model
        
        # Initialize judge
        judge = LLMJudge("mock-model")
        
        # Test with response containing invalid citations
        result = judge.evaluate_groundedness(
            question="What compensation do I get?",
            response="According to Section EA-999, you get triple compensation for wrongful termination.",
            reference_context=""
        )
        
        # Should detect hallucination due to invalid citation
        self.assertTrue(result.hallucination_detected)
        
        # Test with response containing out-of-bounds numbers
        result = judge.evaluate_groundedness(
            question="How much leave do I get?",
            response="You are entitled to 365 days of annual leave per year.",
            reference_context=""
        )
        
        # Should detect hallucination due to out-of-bounds claim
        self.assertTrue(result.hallucination_detected)
        
        print("✅ Shared validators detect invalid citations and numeric claims")
        print(f"✅ Citation validator has {len(judge.citation_validator.known_sections)} known sections")
        print(f"✅ Numeric validator has {len(judge.numeric_validator.bounds)} bound categories")
    
    def test_citation_validator_consistency(self):
        """Test citation validator works consistently."""
        
        # Test citation validator directly
        validator = CitationValidator()
        
        test_text = "Section EA-60F is valid but EA-999 is not valid."
        result = validator.validate_citations(test_text)
        
        # Should find both valid and invalid citations
        self.assertEqual(result.valid_citations, ["EA-60F"])
        self.assertEqual(result.invalid_citations, ["EA-999"])
        self.assertFalse(result.validation_passed)
        
        print(f"✅ Citation validator: {len(result.valid_citations)} valid, {len(result.invalid_citations)} invalid")
    
    def test_numeric_validator_bounds(self):
        """Test numeric validator detects out-of-bounds claims."""
        
        validator = NumericValidator()
        
        # Test reasonable claim
        reasonable_text = "You get 15 days of annual leave per year."
        result = validator.validate_numeric_claims(reasonable_text)
        self.assertTrue(result.validation_passed)
        self.assertEqual(len(result.out_of_bounds), 0)
        
        # Test unreasonable claim
        unreasonable_text = "You get 999 days of annual leave per year."
        result = validator.validate_numeric_claims(unreasonable_text)
        self.assertFalse(result.validation_passed)
        self.assertGreater(len(result.out_of_bounds), 0)
        
        print(f"✅ Numeric validator detects out-of-bounds claims: {len(result.out_of_bounds)}")
    
    @patch('evals.generation_eval.AutoModelForCausalLM')
    @patch('evals.generation_eval.AutoTokenizer')
    def test_no_attention_mask_warnings(self, mock_tokenizer_class, mock_model_class):
        """Test that no attention_mask warnings are generated."""
        
        # Setup mocks
        mock_tokenizer_class.from_pretrained.return_value = self.mock_tokenizer
        mock_model_class.from_pretrained.return_value = self.mock_model
        
        # Capture warnings
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            
            # Initialize and use judge
            judge = LLMJudge("mock-model")
            result = judge.evaluate_groundedness(
                question="Test question",
                response="Test response with Section EA-60F",
                reference_context=""
            )
            
            # Check for attention_mask related warnings
            attention_warnings = [warning for warning in w 
                                if 'attention_mask' in str(warning.message).lower()]
            
            self.assertEqual(len(attention_warnings), 0, 
                           f"Found attention_mask warnings: {[str(w.message) for w in attention_warnings]}")
        
        print("✅ No attention_mask warnings generated")
    
    @patch('evals.generation_eval.AutoModelForCausalLM')
    @patch('evals.generation_eval.AutoTokenizer')
    def test_generation_evaluator_batch_processing(self, mock_tokenizer_class, mock_model_class):
        """Test GenerationEvaluator can process multiple samples."""
        
        # Setup mocks
        mock_tokenizer_class.from_pretrained.return_value = self.mock_tokenizer
        mock_model_class.from_pretrained.return_value = self.mock_model
        
        # Initialize evaluator
        evaluator = GenerationEvaluator("mock-model")
        
        # Test sample dataset
        test_samples = [
            {
                "question": "What is sick leave?",
                "expected_answer": "14 days per year",
                "actual_answer": "You get 14 days of sick leave according to Section EA-60F.",
                "context": "Employment Act provisions"
            },
            {
                "question": "What about overtime?",
                "expected_answer": "1.5x rate",
                "actual_answer": "Overtime is paid at 1.5 times normal rate per Section EA-77.",
                "context": "Overtime provisions"
            }
        ]
        
        # Evaluate samples
        results = []
        for sample in test_samples:
            result = evaluator.judge.evaluate_groundedness(
                question=sample["question"],
                response=sample["actual_answer"],
                reference_context=sample["context"]
            )
            results.append(result)
        
        # Verify all samples processed
        self.assertEqual(len(results), 2)
        
        # Check that all results have expected structure
        for result in results:
            self.assertIsInstance(result.is_grounded, bool)
            self.assertIsInstance(result.groundedness_score, float)
            self.assertIsInstance(result.evidence_citations, list)
        
        print(f"✅ Batch processing: {len(results)} samples evaluated")
    
    def test_performance_tracking(self):
        """Test that evaluation can track basic performance metrics."""
        
        # Test citation validation performance
        validator = CitationValidator()
        
        import time
        start_time = time.time()
        
        # Validate multiple texts
        test_texts = [
            "Section EA-60F provides guidance",
            "Section EA-999 is invalid",
            "No citations here",
            "Multiple: EA-12, EA-77, EA-999"
        ]
        
        results = []
        for text in test_texts:
            result = validator.validate_citations(text)
            results.append(result)
        
        end_time = time.time()
        processing_time = (end_time - start_time) * 1000  # ms
        
        # Verify performance is reasonable (should be very fast)
        self.assertLess(processing_time, 100)  # Less than 100ms for 4 texts
        
        # Verify all results processed
        self.assertEqual(len(results), 4)
        
        print(f"✅ Performance: {len(test_texts)} texts validated in {processing_time:.2f}ms")

def run_smoke_tests():
    """Run generation evaluation smoke tests."""
    print("💨 Running Generation Evaluation Smoke Tests")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestGenerationEvalSmoke))
    
    # Run tests with warnings captured
    runner = unittest.TextTestRunner(verbosity=2, stream=sys.stdout)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Smoke Tests Summary:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    
    success_rate = ((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100)
    print(f"  Success rate: {success_rate:.1f}%")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback.split(chr(10))[-2]}")
    
    if result.errors:
        print(f"\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback.split(chr(10))[-2]}")
    
    if len(result.failures) + len(result.errors) == 0:
        print(f"\n🎉 All smoke tests passed! generation_eval.py is ready for production.")
        print(f"✅ Attention mask fix verified")
        print(f"✅ Shared validators integration confirmed")
        print(f"✅ No warnings about missing attention_mask")
    
    return len(result.failures) + len(result.errors) == 0

if __name__ == '__main__':
    success = run_smoke_tests()
    exit(0 if success else 1)
