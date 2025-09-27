#!/usr/bin/env python3
"""
Integration Tests for Guardrails → Validators → YAML Pipeline
Tests the complete production-grade guardrails system end-to-end.
"""

import unittest
import sys
import os
import tempfile
import yaml
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from generation.guardrails import ProductionGuardrailsEngine, GuardrailsEngine
from evals.validation_utils import CitationValidator, NumericValidator

class TestGuardrailsYAMLIntegration(unittest.TestCase):
    """Integration tests for YAML config-driven guardrails system."""
    
    def setUp(self):
        """Set up test fixtures."""
        # Create temporary YAML config for testing
        self.test_config = {
            'metadata': {
                'version': '1.0.0-test',
                'description': 'Test configuration for integration tests'
            },
            'policy_classifier': {
                'enabled': True,
                'min_confidence': 0.8,
                'categories': {
                    'legal_advice': {
                        'enabled': True,
                        'keywords': ['legal advice', 'sue', 'lawsuit'],
                        'refusal_template': 'I cannot provide legal advice for testing.'
                    },
                    'out_of_scope': {
                        'enabled': True,
                        'keywords': ['visa', 'passport'],
                        'refusal_template': 'This is outside my scope for testing.'
                    }
                }
            },
            'pii_scrubber': {
                'enabled': True,
                'mode': 'mask',
                'patterns': {
                    'email': {
                        'regex': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                        'placeholder': '[EMAIL_REDACTED]'
                    },
                    'phone_test': {
                        'regex': r'\b\d{3}-\d{3}-\d{4}\b',
                        'placeholder': '[PHONE_REDACTED]'
                    }
                },
                'refusal_message': 'Please avoid sharing personal information in tests.'
            },
            'output_validation': {
                'enabled': True,
                'citation_validation': {
                    'enabled': True,
                    'corpus_store_path': ''  # Will use defaults
                },
                'numeric_validation': {
                    'enabled': True,
                    'bounds': {
                        'annual_leave': {'min': 8, 'max': 30, 'unit': 'days'},
                        'sick_leave': {'min': 14, 'max': 60, 'unit': 'days'}
                    }
                },
                'hallucination_patterns': {
                    'enabled': True,
                    'patterns': [
                        r'section ea-999',
                        r'triple compensation'
                    ]
                },
                'inappropriate_advice': {
                    'enabled': True,
                    'patterns': [
                        r'you should sue',
                        r'i guarantee'
                    ]
                }
            },
            'escalation': {
                'enabled': True,
                'triggers': [
                    'invalid_citations_detected',
                    'numeric_claims_out_of_bounds'
                ]
            },
            'logging': {
                'enabled': True,
                'log_level': 'INFO'
            }
        }
        
        # Create temporary config file
        self.temp_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(self.test_config, self.temp_config)
        self.temp_config.close()
        self.config_path = Path(self.temp_config.name)
    
    def tearDown(self):
        """Clean up test fixtures."""
        if self.config_path.exists():
            os.unlink(self.config_path)
    
    def test_yaml_config_loading(self):
        """Test that YAML config is properly loaded."""
        engine = ProductionGuardrailsEngine(config_path=self.config_path)
        
        self.assertEqual(engine.config_version, '1.0.0-test')
        self.assertTrue(engine.policy_classifier.enabled)
        self.assertTrue(engine.pii_scrubber.enabled)
        self.assertIsNotNone(engine.citation_validator)
        self.assertIsNotNone(engine.numeric_validator)
    
    def test_policy_classification_from_yaml(self):
        """Test policy classification using YAML-driven keywords."""
        engine = ProductionGuardrailsEngine(config_path=self.config_path)
        
        # Test legal advice detection
        result = engine.process_input("Can you give me legal advice about my case?")
        self.assertFalse(result.passed)
        self.assertIn("legal advice", result.processed_text.lower())
        self.assertIn("policy_violation", result.report.input_flags)
        
        # Test out-of-scope detection
        result = engine.process_input("Help me with my visa application")
        self.assertFalse(result.passed)
        self.assertIn("scope", result.processed_text.lower())
        
        # Test valid question
        result = engine.process_input("What is my sick leave entitlement?")
        self.assertTrue(result.passed)
        self.assertEqual([], result.report.input_flags)
    
    def test_pii_scrubbing_from_yaml(self):
        """Test PII scrubbing using YAML-driven patterns."""
        engine = ProductionGuardrailsEngine(config_path=self.config_path)
        
        # Test email masking
        result = engine.process_input("My email is test@example.com")
        self.assertTrue(result.passed)
        self.assertIn("[EMAIL_REDACTED]", result.processed_text)
        self.assertIn("pii_masked", result.report.input_flags)
        self.assertIn("email", result.report.pii_detected)
        
        # Test phone masking
        result = engine.process_input("Call me at 123-456-7890")
        self.assertTrue(result.passed)
        self.assertIn("[PHONE_REDACTED]", result.processed_text)
        self.assertIn("phone_test", result.report.pii_detected)
    
    def test_citation_validation_integration(self):
        """Test citation validation through output processing."""
        engine = ProductionGuardrailsEngine(config_path=self.config_path)
        
        # Test valid citation
        result = engine.process_output("According to Section EA-60F, you are entitled to leave.")
        self.assertTrue(result.passed)
        self.assertTrue(result.report.citations_valid)
        self.assertEqual([], result.report.invalid_citations)
        
        # Test invalid citation
        result = engine.process_output("Section EA-999 provides triple compensation.")
        self.assertFalse(result.passed)
        self.assertFalse(result.report.citations_valid)
        self.assertIn("EA-999", result.report.invalid_citations)
        self.assertIn("invalid_citations", result.report.output_flags)
    
    def test_numeric_validation_integration(self):
        """Test numeric validation through output processing."""
        engine = ProductionGuardrailsEngine(config_path=self.config_path)
        
        # Test valid numeric claim
        result = engine.process_output("You are entitled to 15 days of annual leave.")
        self.assertTrue(result.passed)
        self.assertEqual([], result.report.numeric_out_of_bounds)
        
        # Test out-of-bounds claim (using custom test bounds)
        result = engine.process_output("You get 100 days of annual leave.")
        self.assertFalse(result.passed)
        self.assertTrue(len(result.report.numeric_out_of_bounds) > 0)
        self.assertIn("numeric_out_of_bounds", result.report.output_flags)
    
    def test_hallucination_pattern_detection(self):
        """Test hallucination detection from YAML patterns."""
        engine = ProductionGuardrailsEngine(config_path=self.config_path)
        
        # Test hallucination pattern
        result = engine.process_output("Section EA-999 guarantees triple compensation.")
        self.assertFalse(result.passed)
        self.assertTrue(result.report.hallucination_detected)
        self.assertIn("hallucination_pattern", result.report.output_flags)
    
    def test_inappropriate_advice_detection(self):
        """Test inappropriate advice detection from YAML patterns."""
        engine = ProductionGuardrailsEngine(config_path=self.config_path)
        
        # Test inappropriate advice pattern
        result = engine.process_output("You should sue your employer immediately.")
        self.assertFalse(result.passed)
        self.assertTrue(result.report.inappropriate_advice_detected)
        self.assertIn("inappropriate_advice", result.report.output_flags)
    
    def test_escalation_logic(self):
        """Test escalation logic based on YAML config."""
        engine = ProductionGuardrailsEngine(config_path=self.config_path)
        
        # Test escalation trigger
        result = engine.process_output("Section EA-999 provides guidance.")
        self.assertFalse(result.passed)
        self.assertTrue(result.report.should_escalate)
        self.assertIn("invalid_citations", result.report.escalation_reasons)
        self.assertEqual("escalated", result.report.decision)
    
    def test_structured_reporting(self):
        """Test structured guardrails reporting."""
        engine = ProductionGuardrailsEngine(config_path=self.config_path)
        
        # Test comprehensive report structure
        result = engine.process_input("Can you give me legal advice? My email is test@example.com")
        report = result.report
        
        # Verify all report fields are present
        self.assertIsNotNone(report.timestamp)
        self.assertIsNotNone(report.prompt_hash)
        self.assertEqual("refused", report.decision)
        self.assertIsInstance(report.processing_time_ms, float)
        self.assertIsInstance(report.input_flags, list)
        self.assertIsInstance(report.pii_detected, list)
        self.assertIsInstance(report.triggered_rules, list)
        self.assertEqual("1.0.0-test", report.config_version)
        
        # Test JSON serialization
        report_json = engine.get_report_json(report)
        self.assertIsInstance(report_json, str)
        # Should be valid JSON
        import json
        parsed = json.loads(report_json)
        self.assertEqual(parsed["config_version"], "1.0.0-test")
    
    def test_legacy_compatibility(self):
        """Test that legacy GuardrailsEngine still works alongside new system."""
        # Test legacy engine
        legacy_engine = GuardrailsEngine(pii_mode="mask")
        result = legacy_engine.process_input("Can you give me legal advice?")
        self.assertFalse(result.passed)
        self.assertIsNone(result.report)  # Legacy doesn't have structured reports
        
        # Test production engine
        prod_engine = ProductionGuardrailsEngine(config_path=self.config_path)
        result = prod_engine.process_input("Can you give me legal advice?")
        self.assertFalse(result.passed)
        self.assertIsNotNone(result.report)  # Production has structured reports
    
    def test_fallback_behavior(self):
        """Test fallback behavior when YAML config is missing or invalid."""
        # Test with non-existent config path
        nonexistent_path = Path("/tmp/nonexistent_config.yaml")
        engine = ProductionGuardrailsEngine(config_path=nonexistent_path)
        
        # Should still work with default config
        result = engine.process_input("What is sick leave?")
        self.assertTrue(result.passed)
        
        # Test with invalid YAML
        invalid_config = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        invalid_config.write("invalid: yaml: content: ][")
        invalid_config.close()
        
        engine = ProductionGuardrailsEngine(config_path=Path(invalid_config.name))
        result = engine.process_input("What is sick leave?")
        self.assertTrue(result.passed)
        
        os.unlink(invalid_config.name)
    
    def test_shared_validators_consistency(self):
        """Test that shared validators work consistently across components."""
        # Test citation validator directly
        citation_validator = CitationValidator()
        citation_result = citation_validator.validate_citations("Section EA-60F is valid but EA-999 is not.")
        
        # Test through guardrails engine
        engine = ProductionGuardrailsEngine(config_path=self.config_path)
        output_result = engine.process_output("Section EA-60F is valid but EA-999 is not.")
        
        # Results should be consistent
        self.assertEqual(citation_result.valid_citations, ["EA-60F"])
        self.assertEqual(citation_result.invalid_citations, ["EA-999"])
        self.assertEqual(output_result.report.invalid_citations, ["EA-999"])
        self.assertFalse(output_result.report.citations_valid)
    
    def test_feature_toggles(self):
        """Test that YAML feature toggles work correctly."""
        # Create config with citation validation disabled
        disabled_config = self.test_config.copy()
        disabled_config['output_validation']['citation_validation']['enabled'] = False
        
        disabled_temp = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False)
        yaml.dump(disabled_config, disabled_temp)
        disabled_temp.close()
        
        engine = ProductionGuardrailsEngine(config_path=Path(disabled_temp.name))
        
        # Citation validation should be disabled
        self.assertIsNone(engine.citation_validator)
        
        # Invalid citation should pass through
        result = engine.process_output("Section EA-999 provides guidance.")
        # Should only fail on hallucination pattern, not citation validation
        self.assertFalse(result.passed)  # Still fails due to hallucination pattern
        self.assertTrue(result.report.citations_valid)  # But citation validation wasn't performed
        
        os.unlink(disabled_temp.name)

def run_integration_tests():
    """Run all integration tests."""
    print("🧪 Running Integration Tests: Guardrails → Validators → YAML")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    test_suite.addTest(unittest.makeSuite(TestGuardrailsYAMLIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\n{'='*60}")
    print(f"Integration Tests Summary:")
    print(f"  Tests run: {result.testsRun}")
    print(f"  Failures: {len(result.failures)}")
    print(f"  Errors: {len(result.errors)}")
    print(f"  Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for test, traceback in result.failures:
            print(f"  {test}: {traceback.split(chr(10))[-2]}")
    
    if result.errors:
        print(f"\n💥 Errors:")
        for test, traceback in result.errors:
            print(f"  {test}: {traceback.split(chr(10))[-2]}")
    
    return len(result.failures) + len(result.errors) == 0

if __name__ == '__main__':
    success = run_integration_tests()
    exit(0 if success else 1)