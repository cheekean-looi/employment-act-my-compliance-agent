#!/usr/bin/env python3
"""
End-to-End Integration Test for YAML → ProductionGuardrailsEngine → Validators
Verifies the complete production guardrails pipeline works correctly.
"""

import unittest
import sys
import os
import tempfile
import yaml
import json
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from generation.guardrails import ProductionGuardrailsEngine
from evals.refusal_eval import RefusalEvaluator
from evals.validation_utils import CitationValidator, NumericValidator

class TestProductionGuardrailsIntegration(unittest.TestCase):
    """End-to-end integration tests for production guardrails system."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.engine = ProductionGuardrailsEngine()
    
    def test_yaml_config_to_validators_flow(self):
        """Test YAML → ProductionGuardrailsEngine → Validators integration."""
        
        # Test 1: Invalid citation in output should trigger validators
        invalid_citation_text = "According to Section EA-999, you get triple compensation for wrongful termination."
        
        output_result = self.engine.process_output(invalid_citation_text)
        
        # Should fail output validation
        self.assertFalse(output_result.passed)
        self.assertFalse(output_result.report.citations_valid)
        self.assertIn("EA-999", output_result.report.invalid_citations)
        self.assertIn("invalid_citations", output_result.report.output_flags)
        
        print(f"✅ Invalid citation detection: {output_result.report.invalid_citations}")
    
    def test_numeric_bounds_validation_flow(self):
        """Test numeric validation through output processing."""
        
        # Test 2: Out-of-bounds numeric claim
        out_of_bounds_text = "You are entitled to 999 days of annual leave per year."
        
        output_result = self.engine.process_output(out_of_bounds_text)
        
        # Should fail numeric validation
        self.assertFalse(output_result.passed)
        self.assertTrue(len(output_result.report.numeric_out_of_bounds) > 0)
        self.assertIn("numeric_out_of_bounds", output_result.report.output_flags)
        
        # Check the specific out-of-bounds item
        oob_item = output_result.report.numeric_out_of_bounds[0]
        self.assertEqual(999.0, oob_item["number"])
        self.assertIn("annual_leave", oob_item["category"])
        
        print(f"✅ Numeric bounds detection: {len(output_result.report.numeric_out_of_bounds)} violations")
    
    def test_hallucination_pattern_detection(self):
        """Test YAML-driven hallucination pattern detection."""
        
        # Test 3: Hallucination pattern from YAML config
        hallucination_text = "Section EA-999 provides triple compensation for all employees."
        
        output_result = self.engine.process_output(hallucination_text)
        
        # Should fail due to both invalid citation AND hallucination pattern
        self.assertFalse(output_result.passed)
        self.assertTrue(output_result.report.hallucination_detected)
        self.assertIn("hallucination_pattern", output_result.report.output_flags)
        
        print(f"✅ Hallucination pattern detection: {output_result.report.hallucination_detected}")
    
    def test_policy_classification_integration(self):
        """Test policy classification with structured reporting."""
        
        # Test 4: Policy violation should be caught
        legal_advice_query = "Can you give me legal advice about suing my employer for discrimination?"
        
        input_result = self.engine.process_input(legal_advice_query)
        
        # Should be refused due to policy violation
        self.assertFalse(input_result.passed)
        self.assertIn("policy_violation", input_result.report.input_flags)
        self.assertTrue(len(input_result.report.triggered_rules) > 0)
        self.assertIn("legal_advice", input_result.report.triggered_rules[0])
        
        print(f"✅ Policy violation detection: {input_result.report.triggered_rules}")
    
    def test_pii_detection_and_masking(self):
        """Test PII detection and masking integration."""
        
        # Test 5: PII should be detected and masked
        pii_query = "My email is john.doe@company.com and my phone number is 012-345-6789. What are my rights?"
        
        input_result = self.engine.process_input(pii_query)
        
        # Should pass but mask PII
        self.assertTrue(input_result.passed)
        self.assertIn("[EMAIL_REDACTED]", input_result.processed_text)
        self.assertIn("[PHONE_REDACTED]", input_result.processed_text)
        self.assertIn("pii_masked", input_result.report.input_flags)
        self.assertIn("email", input_result.report.pii_detected)
        # Phone pattern name might vary based on config
        phone_detected = any("phone" in pii_type for pii_type in input_result.report.pii_detected)
        self.assertTrue(phone_detected, f"Phone not detected in {input_result.report.pii_detected}")
        
        print(f"✅ PII masking: {input_result.report.pii_detected}")
    
    def test_escalation_logic(self):
        """Test escalation logic based on validator results."""
        
        # Test 6: Multiple violations should trigger escalation
        multiple_issues_text = "Section EA-999 guarantees 365 days of sick leave with triple compensation."
        
        output_result = self.engine.process_output(multiple_issues_text)
        
        # Should trigger escalation due to multiple issues
        self.assertFalse(output_result.passed)
        self.assertTrue(output_result.report.should_escalate)
        self.assertEqual("escalated", output_result.report.decision)
        self.assertTrue(len(output_result.report.escalation_reasons) > 0)
        
        print(f"✅ Escalation triggered: {output_result.report.escalation_reasons}")
    
    def test_structured_reporting_completeness(self):
        """Test that structured reports contain all required fields."""
        
        # Test 7: Comprehensive report structure
        test_query = "What is my annual leave entitlement?"
        
        input_result = self.engine.process_input(test_query)
        output_result = self.engine.process_output("You are entitled to 15 days of annual leave per year according to Section EA-60F.")
        
        # Verify input report structure
        input_report = input_result.report
        self.assertIsNotNone(input_report.timestamp)
        self.assertIsNotNone(input_report.prompt_hash)
        self.assertIn(input_report.decision, ["allowed", "refused", "escalated"])
        self.assertIsInstance(input_report.processing_time_ms, float)
        self.assertIsInstance(input_report.input_flags, list)
        self.assertIsInstance(input_report.pii_detected, list)
        self.assertIsInstance(input_report.triggered_rules, list)
        self.assertIsNotNone(input_report.config_version)
        
        # Verify output report structure
        output_report = output_result.report
        self.assertIsInstance(output_report.citations_valid, bool)
        self.assertIsInstance(output_report.invalid_citations, list)
        self.assertIsInstance(output_report.numeric_out_of_bounds, list)
        self.assertIsInstance(output_report.hallucination_detected, bool)
        self.assertIsInstance(output_report.inappropriate_advice_detected, bool)
        self.assertIsInstance(output_report.should_escalate, bool)
        
        print(f"✅ Report structure validation passed")
    
    def test_shared_validators_consistency(self):
        """Test that shared validators work consistently across components."""
        
        # Test 8: Direct validator vs. guardrails integration consistency
        citation_validator = CitationValidator()
        numeric_validator = NumericValidator()
        
        test_text = "Section EA-60F provides 15 days annual leave, but Section EA-999 gives unlimited leave."
        
        # Test citation validation directly
        direct_citation_result = citation_validator.validate_citations(test_text)
        
        # Test through guardrails
        guardrails_result = self.engine.process_output(test_text)
        
        # Results should be consistent
        self.assertEqual(direct_citation_result.valid_citations, ["EA-60F"])
        self.assertEqual(direct_citation_result.invalid_citations, ["EA-999"])
        self.assertEqual(guardrails_result.report.invalid_citations, ["EA-999"])
        self.assertFalse(guardrails_result.report.citations_valid)
        
        print(f"✅ Validator consistency: Direct={direct_citation_result.invalid_citations}, Guardrails={guardrails_result.report.invalid_citations}")
    
    def test_feature_toggles_work(self):
        """Test that YAML config toggles are respected."""
        
        # Test 9: Create config with citation validation disabled
        disabled_config = {
            'metadata': {'version': '1.0.0-test'},
            'policy_classifier': {'enabled': True, 'categories': {}},
            'pii_scrubber': {'enabled': True, 'mode': 'mask', 'patterns': {}},
            'output_validation': {
                'enabled': True,
                'citation_validation': {'enabled': False},  # Disabled
                'numeric_validation': {'enabled': True}
            },
            'logging': {'enabled': True}
        }
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(disabled_config, f)
            temp_config_path = Path(f.name)
        
        try:
            # Create engine with disabled citation validation
            disabled_engine = ProductionGuardrailsEngine(config_path=temp_config_path)
            
            # Citation validation should be disabled
            self.assertIsNone(disabled_engine.citation_validator)
            
            # Invalid citation should pass through (only fail on other issues)
            result = disabled_engine.process_output("Section EA-999 provides information.")
            # Should pass citation validation (disabled) but might fail on hallucination patterns
            self.assertTrue(result.report.citations_valid)  # Not validated, so reports as valid
            
            print(f"✅ Feature toggle works: citation_validator={disabled_engine.citation_validator}")
            
        finally:
            os.unlink(temp_config_path)
    
    def test_config_path_fallbacks(self):
        """Test robust config path handling."""
        
        # Test 10: Config path fallback behavior
        
        # Test with non-existent explicit path
        nonexistent_path = Path("/tmp/nonexistent_guardrails.yaml")
        fallback_engine = ProductionGuardrailsEngine(config_path=nonexistent_path)
        
        # Should fall back to existing config
        self.assertIsNotNone(fallback_engine.config)
        self.assertIsNotNone(fallback_engine.config_version)
        
        # Should work normally
        result = fallback_engine.process_input("What is sick leave?")
        self.assertTrue(result.passed)
        
        print(f"✅ Config fallback works: version={fallback_engine.config_version}")
    
    def test_refusal_evaluator_integration(self):
        """Test RefusalEvaluator works with ProductionGuardrailsEngine."""
        
        # Test 11: RefusalEvaluator integration
        evaluator = RefusalEvaluator(use_production=True)
        
        # Test refusal decision
        was_refused, triggered_rules, processed_text = evaluator.evaluate_refusal_decision(
            "Can you give me legal advice about my case?"
        )
        
        self.assertTrue(was_refused)
        self.assertTrue(len(triggered_rules) > 0)
        # Check that the refusal message is appropriate for legal advice requests
        self.assertTrue(
            "legal" in processed_text.lower() or "qualified" in processed_text.lower(),
            f"Refusal message doesn't indicate legal nature: {processed_text}"
        )
        
        print(f"✅ RefusalEvaluator integration: refused={was_refused}, rules={len(triggered_rules)}")
    
    def test_performance_and_logging(self):
        """Test performance tracking and privacy-safe logging."""
        
        # Test 12: Performance and logging
        test_input = "What are my overtime rights?"
        
        # Process input and output
        input_result = self.engine.process_input(test_input)
        output_result = self.engine.process_output("You have rights to overtime pay as per Employment Act.")
        
        # Check performance tracking
        self.assertGreater(input_result.report.processing_time_ms, 0)
        self.assertGreater(output_result.report.processing_time_ms, 0)
        
        # Check privacy-safe reporting
        self.assertEqual(len(input_result.report.prompt_hash), 16)  # SHA256 truncated
        self.assertNotEqual(input_result.report.prompt_hash, output_result.report.prompt_hash)
        
        # Final text should be safe for logging (no raw PII)
        self.assertIsInstance(input_result.report.final_text, str)
        self.assertIsInstance(output_result.report.final_text, str)
        
        print(f"✅ Performance tracking: input={input_result.report.processing_time_ms:.2f}ms, output={output_result.report.processing_time_ms:.2f}ms")

def run_integration_tests():
    """Run all integration tests."""
    print("🔗 Running End-to-End Integration Tests")
    print("=" * 50)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    test_suite.addTests(unittest.TestLoader().loadTestsFromTestCase(TestProductionGuardrailsIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*50}")
    print(f"Integration Tests Summary:")
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
        print(f"\n🎉 All integration tests passed! Production guardrails are ready.")
    
    return len(result.failures) + len(result.errors) == 0

if __name__ == '__main__':
    success = run_integration_tests()
    exit(0 if success else 1)