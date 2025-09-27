#!/usr/bin/env python3
"""
Production-Grade Guardrails System for Employment Act Malaysia Compliance Agent
Config-driven, validator-integrated, audit-ready implementation.
"""

import re
import json
import yaml
import time
import hashlib
import os
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
import sys

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

# Import shared validators
from evals.validation_utils import CitationValidator, NumericValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class GuardrailsReport:
    """Structured guardrails report for audit and observability."""
    timestamp: str
    prompt_hash: str
    decision: str  # "allowed" | "refused" | "escalated"
    confidence: float
    processing_time_ms: float
    
    # Input processing
    input_flags: List[str]
    pii_detected: List[str]
    pii_masked: bool
    
    # Output validation
    output_flags: List[str]
    citations_valid: bool
    invalid_citations: List[str]
    numeric_out_of_bounds: List[Dict[str, Any]]
    hallucination_detected: bool
    inappropriate_advice_detected: bool
    
    # Escalation
    should_escalate: bool
    escalation_reasons: List[str]
    
    # Metadata
    config_version: str
    triggered_rules: List[str]
    final_text: str  # Safe to log (PII-free)

@dataclass
class GuardrailResult:
    """Result from guardrail evaluation with enhanced reporting."""
    passed: bool
    processed_text: str
    confidence: float
    reason: str
    report: GuardrailsReport

class ConfigurablePolicyClassifier:
    """Policy classifier driven by YAML configuration."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize from config."""
        self.enabled = config.get('policy_classifier', {}).get('enabled', True)
        self.min_confidence = config.get('policy_classifier', {}).get('min_confidence', 0.8)
        
        # Load categories from config
        self.categories = {}
        categories_config = config.get('policy_classifier', {}).get('categories', {})
        
        for category, settings in categories_config.items():
            if settings.get('enabled', True):
                self.categories[category] = {
                    'keywords': settings.get('keywords', []),
                    'refusal_template': settings.get('refusal_template', self._default_refusal_template())
                }
    
    def _default_refusal_template(self) -> str:
        """Default refusal template."""
        return ("I can only provide general information about Malaysia's Employment Act provisions. "
                "For specific legal matters, please consult with qualified professionals.")
    
    def classify(self, text: str) -> Tuple[bool, List[str], str]:
        """
        Classify text for policy violations.
        
        Returns:
            (should_refuse, triggered_rules, refusal_message)
        """
        if not self.enabled:
            return False, [], ""
        
        text_lower = text.lower()
        triggered_rules = []
        first_triggered_category: Optional[str] = None
        
        # Iterate in insertion order; remember the first category that triggers
        for category, settings in self.categories.items():
            for keyword in settings['keywords']:
                if keyword.lower() in text_lower:
                    rule_name = f"policy_{category}_{keyword.replace(' ', '_')}"
                    triggered_rules.append(rule_name)
                    if first_triggered_category is None:
                        first_triggered_category = category
        
        if triggered_rules:
            # Use the first triggered category's configured template
            template = self.categories.get(first_triggered_category, {}).get('refusal_template', self._default_refusal_template())
            return True, triggered_rules, template
        
        return False, [], ""

# Legacy PolicyClassifier for backward compatibility
class PolicyClassifier:
    """Legacy policy classifier - use ConfigurablePolicyClassifier for new code."""
    
    def __init__(self):
        # Default configuration for backward compatibility
        default_config = {
            'policy_classifier': {
                'enabled': True,
                'categories': {
                    'legal_advice': {
                        'keywords': [
                            'give me legal advice', 'give legal advice', 'provide legal advice', 'legal counsel',
                            'sue', 'lawsuit', 'court case', 'litigation', 'attorney',
                            'lawyer recommendation', 'legal representation', 'hire a lawyer',
                            'should i hire', 'fight my case'
                        ],
                        'refusal_template': "I understand you're asking about legal advice, but I can only provide general information about Malaysia's Employment Act provisions."
                    },
                    'guarantees': {
                        'keywords': ['guarantee', 'promise', 'ensure', 'definitely will', 'absolutely'],
                        'refusal_template': "I understand you're asking about guarantees, but I can only provide general information about Malaysia's Employment Act provisions."
                    },
                    'out_of_scope': {
                        'keywords': ['criminal law', 'family law', 'property law', 'visa', 'passport', 'company registration'],
                        'refusal_template': "I understand you're asking about this matter, but I can only provide general information about Malaysia's Employment Act provisions."
                    }
                }
            }
        }
        
        self.classifier = ConfigurablePolicyClassifier(default_config)
    
    def classify(self, text: str) -> GuardrailResult:
        """Legacy interface - returns GuardrailResult."""
        should_refuse, triggered_rules, refusal_message = self.classifier.classify(text)
        
        if should_refuse:
            return GuardrailResult(
                passed=False,
                processed_text=refusal_message,
                confidence=0.9,
                reason=f"Policy violation detected: {', '.join(triggered_rules[:2])}",
                report=None  # Legacy mode doesn't have full report
            )
        
        return GuardrailResult(
            passed=True,
            processed_text=text,
            confidence=1.0,
            reason="No policy violations detected",
            report=None
        )

class ConfigurablePIIScrubber:
    """PII scrubber driven by YAML configuration."""
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize from config."""
        pii_config = config.get('pii_scrubber', {})
        self.enabled = pii_config.get('enabled', True)
        self.mode = pii_config.get('mode', 'mask')
        
        # Compile patterns from config
        self.patterns = {}
        patterns_config = pii_config.get('patterns', {})
        
        for pattern_name, pattern_settings in patterns_config.items():
            try:
                regex = pattern_settings.get('regex', '')
                placeholder = pattern_settings.get('placeholder', '[REDACTED]')
                self.patterns[pattern_name] = {
                    'compiled': re.compile(regex),
                    'placeholder': placeholder
                }
            except re.error as e:
                logger.warning(f"Invalid regex for {pattern_name}: {e}")
        
        self.refusal_message = pii_config.get('refusal_message', 
            "Please avoid sharing personal information. I can help with general Employment Act questions.")
    
    def scrub(self, text: str) -> Tuple[bool, str, List[str]]:
        """
        Scrub PII from text.
        
        Returns:
            (should_refuse, processed_text, detected_pii_types)
        """
        if not self.enabled:
            return False, text, []
        
        detected_pii = []
        processed_text = text
        
        for pii_type, pattern_info in self.patterns.items():
            matches = pattern_info['compiled'].findall(text)
            if matches:
                detected_pii.append(pii_type)
                
                if self.mode == "refuse":
                    return True, self.refusal_message, detected_pii
                elif self.mode == "mask":
                    processed_text = pattern_info['compiled'].sub(pattern_info['placeholder'], processed_text)
        
        return False, processed_text, detected_pii

# Legacy PIIScrubber for backward compatibility  
class PIIScrubber:
    """Legacy PII scrubber - use ConfigurablePIIScrubber for new code."""
    
    def __init__(self, mode: str = "mask"):
        # Default configuration for backward compatibility
        default_config = {
            'pii_scrubber': {
                'enabled': True,
                'mode': mode,
                'patterns': {
                    'email': {
                        'regex': r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                        'placeholder': '[EMAIL_REDACTED]'
                    },
                    'phone_my': {
                        'regex': r'\b(?:\+?60|0)(?:1[0-46-9]|[3-9])\d{7,8}\b|(?:01[0-9])[- ]?\d{3,4}[- ]?\d{4}\b',
                        'placeholder': '[PHONE_REDACTED]'
                    },
                    'ic_number': {
                        'regex': r'\b\d{6}-\d{2}-\d{4}\b',
                        'placeholder': '[IC_NUMBER_REDACTED]'
                    }
                },
                'refusal_message': (
                    "I notice your message contains personal information. For your privacy and security, "
                    "please avoid sharing personal details. I can help with general Employment Act questions."
                )
            }
        }
        
        self.scrubber = ConfigurablePIIScrubber(default_config)
    
    def scrub(self, text: str) -> GuardrailResult:
        """Legacy interface - returns GuardrailResult."""
        should_refuse, processed_text, detected_pii = self.scrubber.scrub(text)
        
        if should_refuse:
            return GuardrailResult(
                passed=False,
                processed_text=processed_text,
                confidence=0.95,
                reason=f"PII detected: {', '.join(detected_pii)}",
                report=None
            )
        elif detected_pii:
            return GuardrailResult(
                passed=True,
                processed_text=processed_text,
                confidence=0.9,
                reason=f"PII masked: {', '.join(detected_pii)}",
                report=None
            )
        
        return GuardrailResult(
            passed=True,
            processed_text=processed_text,
            confidence=1.0,
            reason="No PII detected",
            report=None
        )

class ProductionGuardrailsEngine:
    """
    Production-grade guardrails engine with YAML config, shared validators, and structured reporting.
    """
    
    def __init__(self, config_path: Optional[Path] = None):
        """Initialize guardrails engine with config."""
        self.start_time = time.time()
        
        # Load configuration with robust path handling
        self.config_path = self._resolve_config_path(config_path)
        self.config = self._load_config(self.config_path)
        self.config_version = self.config.get('metadata', {}).get('version', '1.0.0')
        
        # Initialize components
        self.policy_classifier = ConfigurablePolicyClassifier(self.config)
        self.pii_scrubber = ConfigurablePIIScrubber(self.config)
        
        # Initialize validators
        self._init_validators()
        
        # Compile output validation patterns
        self._init_output_validation()
        
        # Setup logging
        self._setup_logging()
        
        logger.info(f"Guardrails engine initialized with config version {self.config_version}")
    
    def _resolve_config_path(self, config_path: Optional[Path] = None) -> Path:
        """
        Resolve config path with multiple fallback options.
        
        Priority order:
        1. Explicit config_path parameter
        2. GUARDRAILS_CONFIG environment variable
        3. ./config/guardrails.yaml (relative to CWD)
        4. config/guardrails.yaml (relative to project root)
        5. Default config (in-memory fallback)
        
        Args:
            config_path: Explicit path override
            
        Returns:
            Resolved config path or None for fallback to defaults
        """
        # 1. Explicit parameter
        if config_path is not None:
            if config_path.exists():
                logger.info(f"Using explicit config path: {config_path}")
                return config_path
            else:
                logger.warning(f"Explicit config path not found: {config_path}, trying fallbacks")
        
        # 2. Environment variable
        env_config = os.getenv('GUARDRAILS_CONFIG')
        if env_config:
            env_path = Path(env_config)
            if env_path.exists():
                logger.info(f"Using config from GUARDRAILS_CONFIG: {env_path}")
                return env_path
            else:
                logger.warning(f"GUARDRAILS_CONFIG path not found: {env_path}, trying fallbacks")
        
        # 3. Relative to current working directory
        cwd_config = Path.cwd() / "config" / "guardrails.yaml"
        if cwd_config.exists():
            logger.info(f"Using config relative to CWD: {cwd_config}")
            return cwd_config
        
        # 4. Relative to project root (where this file is located)
        project_root = Path(__file__).parent.parent.parent
        project_config = project_root / "config" / "guardrails.yaml"
        if project_config.exists():
            logger.info(f"Using config relative to project root: {project_config}")
            return project_config
        
        # 5. Fallback to defaults (no config file)
        logger.warning("No config file found, using built-in defaults")
        return None
    
    def _load_config(self, config_path: Optional[Path]) -> Dict[str, Any]:
        """Load YAML configuration."""
        if config_path is None:
            logger.info("Using built-in default configuration")
            return self._get_default_config()
        
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            logger.info(f"Loaded config from {config_path}")
            return config
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            logger.info("Falling back to built-in default configuration")
            return self._get_default_config()
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get comprehensive default configuration."""
        return {
            'metadata': {
                'version': '1.0.0-default',
                'description': 'Built-in default guardrails configuration'
            },
            'policy_classifier': {
                'enabled': True,
                'min_confidence': 0.8,
                'categories': {
                    'legal_advice': {
                        'enabled': True,
                        'keywords': [
                            'give me legal advice', 'provide legal advice', 'legal counsel',
                            'sue', 'lawsuit', 'court case', 'litigation', 'attorney',
                            'lawyer recommendation', 'legal representation', 'hire a lawyer'
                        ],
                        'refusal_template': "I can only provide general information about Malaysia's Employment Act provisions. For specific legal advice, please consult with qualified professionals."
                    },
                    'guarantees': {
                        'enabled': True,
                        'keywords': ['guarantee', 'promise', 'ensure', 'definitely will', 'absolutely'],
                        'refusal_template': "I cannot provide guarantees about legal outcomes. I can only provide general information about Employment Act provisions."
                    },
                    'out_of_scope': {
                        'enabled': True,
                        'keywords': ['visa', 'passport', 'company registration', 'criminal law', 'family law'],
                        'refusal_template': "This is outside my scope. I can only provide information about Malaysia's Employment Act provisions."
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
                    'phone_my': {
                        'regex': r'\b(?:\+?60|0)(?:1[0-46-9]|[3-9])\d{7,8}\b',
                        'placeholder': '[PHONE_REDACTED]'
                    },
                    'ic_number': {
                        'regex': r'\b\d{6}-\d{2}-\d{4}\b',
                        'placeholder': '[IC_NUMBER_REDACTED]'
                    }
                },
                'refusal_message': 'Please avoid sharing personal information. I can help with general Employment Act questions.'
            },
            'output_validation': {
                'enabled': True,
                'citation_validation': {
                    'enabled': True,
                    'corpus_store_path': ''
                },
                'numeric_validation': {
                    'enabled': True,
                    'bounds': {
                        'annual_leave': {'min': 8, 'max': 60, 'unit': 'days'},
                        'sick_leave': {'min': 14, 'max': 365, 'unit': 'days'},
                        'notice_period': {'min': 1, 'max': 16, 'unit': 'weeks'}
                    }
                },
                'hallucination_patterns': {
                    'enabled': True,
                    'patterns': [
                        r'section ea-999',
                        r'triple compensation',
                        r'unlimited overtime'
                    ]
                },
                'inappropriate_advice': {
                    'enabled': True,
                    'patterns': [
                        r'you should sue',
                        r'i guarantee',
                        r'definitely will win'
                    ]
                }
            },
            'escalation': {
                'enabled': False,
                'triggers': []
            },
            'logging': {
                'enabled': True,
                'log_level': 'INFO'
            }
        }
    
    def _init_validators(self):
        """Initialize citation and numeric validators."""
        output_config = self.config.get('output_validation', {})
        
        # Citation validator
        citation_config = output_config.get('citation_validation', {})
        self.citation_validator = None
        
        if citation_config.get('enabled', False):
            corpus_store_path = citation_config.get('corpus_store_path')
            known_sections = None
            
            if corpus_store_path and Path(corpus_store_path).exists():
                self.citation_validator = CitationValidator()
                known_sections = self.citation_validator.load_sections_from_corpus(Path(corpus_store_path))
                self.citation_validator.known_sections = known_sections
            else:
                self.citation_validator = CitationValidator()  # Use defaults
            
            logger.info(f"Citation validator initialized with {len(self.citation_validator.known_sections)} sections")
        
        # Numeric validator
        numeric_config = output_config.get('numeric_validation', {})
        self.numeric_validator = None
        
        if numeric_config.get('enabled', False):
            self.numeric_validator = NumericValidator()
            
            # Override bounds from config if provided
            config_bounds = numeric_config.get('bounds', {})
            if config_bounds:
                self.numeric_validator.bounds.update(config_bounds)
            
            logger.info(f"Numeric validator initialized with {len(self.numeric_validator.bounds)} bound categories")
    
    def _init_output_validation(self):
        """Initialize output validation patterns from config."""
        output_config = self.config.get('output_validation', {})
        
        # Hallucination patterns
        hallucination_config = output_config.get('hallucination_patterns', {})
        self.hallucination_patterns = []
        
        if hallucination_config.get('enabled', False):
            patterns = hallucination_config.get('patterns', [])
            for pattern in patterns:
                try:
                    self.hallucination_patterns.append(re.compile(pattern, re.IGNORECASE))
                except re.error as e:
                    logger.warning(f"Invalid hallucination pattern '{pattern}': {e}")
        
        # Inappropriate advice patterns
        advice_config = output_config.get('inappropriate_advice', {})
        self.inappropriate_patterns = []
        
        if advice_config.get('enabled', False):
            patterns = advice_config.get('patterns', [])
            for pattern in patterns:
                try:
                    self.inappropriate_patterns.append(re.compile(pattern, re.IGNORECASE))
                except re.error as e:
                    logger.warning(f"Invalid inappropriate advice pattern '{pattern}': {e}")
    
    def _setup_logging(self):
        """Setup logging based on config."""
        logging_config = self.config.get('logging', {})
        self.logging_enabled = logging_config.get('enabled', True)
        
        if self.logging_enabled:
            log_level = logging_config.get('log_level', 'INFO')
            logger.setLevel(getattr(logging, log_level))
    
    def _create_prompt_hash(self, text: str) -> str:
        """Create privacy-safe hash of prompt."""
        return hashlib.sha256(text.encode()).hexdigest()[:16]
    
    def process_input(self, user_input: str) -> GuardrailResult:
        """Process user input through input guardrails."""
        start_time = time.time()
        prompt_hash = self._create_prompt_hash(user_input)
        
        input_flags = []
        triggered_rules = []
        
        # Step 1: Policy classification
        should_refuse_policy, policy_rules, refusal_message = self.policy_classifier.classify(user_input)
        if should_refuse_policy:
            input_flags.extend(["policy_violation"])
            triggered_rules.extend(policy_rules)
        
        # Step 2: PII scrubbing
        should_refuse_pii, processed_text, detected_pii = self.pii_scrubber.scrub(user_input)
        if should_refuse_pii:
            input_flags.extend(["pii_refusal"])
            triggered_rules.extend([f"pii_{pii_type}_refused" for pii_type in detected_pii])
        elif detected_pii:
            input_flags.extend(["pii_masked"])
            triggered_rules.extend([f"pii_{pii_type}_masked" for pii_type in detected_pii])
        
        # Determine overall decision
        should_refuse = should_refuse_policy or should_refuse_pii
        final_text = refusal_message if should_refuse_policy else processed_text
        
        # Create report
        processing_time = (time.time() - start_time) * 1000
        
        report = GuardrailsReport(
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            prompt_hash=prompt_hash,
            decision="refused" if should_refuse else "allowed",
            confidence=0.9 if should_refuse else 1.0,
            processing_time_ms=processing_time,
            
            input_flags=input_flags,
            pii_detected=detected_pii,
            pii_masked=bool(detected_pii and not should_refuse_pii),
            
            output_flags=[],  # No output processing in input phase
            citations_valid=True,  # Not applicable for input
            invalid_citations=[],
            numeric_out_of_bounds=[],
            hallucination_detected=False,
            inappropriate_advice_detected=False,
            
            should_escalate=False,  # Could add escalation logic
            escalation_reasons=[],
            
            config_version=self.config_version,
            triggered_rules=triggered_rules,
            final_text=final_text if not should_refuse else "[REFUSAL_MESSAGE]"  # Safe for logging
        )
        
        # Log if enabled
        if self.logging_enabled:
            self._log_processing(report, is_input=True)
        
        return GuardrailResult(
            passed=not should_refuse,
            processed_text=final_text,
            confidence=report.confidence,
            reason=f"Input processing: {', '.join(input_flags) if input_flags else 'passed'}",
            report=report
        )
    
    def process_output(self, model_output: str, context: str = "") -> GuardrailResult:
        """Process model output through output guardrails with full validation."""
        start_time = time.time()
        output_hash = self._create_prompt_hash(model_output)
        
        output_flags = []
        triggered_rules = []
        
        # Step 1: Citation validation
        citations_valid = True
        invalid_citations = []
        
        if self.citation_validator:
            citation_result = self.citation_validator.validate_citations(model_output)
            citations_valid = citation_result.validation_passed
            invalid_citations = citation_result.invalid_citations
            
            if invalid_citations:
                output_flags.append("invalid_citations")
                triggered_rules.extend([f"citation_invalid_{cite}" for cite in invalid_citations])
        
        # Step 2: Numeric validation
        numeric_out_of_bounds = []
        
        if self.numeric_validator:
            numeric_result = self.numeric_validator.validate_numeric_claims(model_output)
            numeric_out_of_bounds = numeric_result.out_of_bounds
            
            if numeric_out_of_bounds:
                output_flags.append("numeric_out_of_bounds")
                triggered_rules.extend([f"numeric_invalid_{item['category']}" for item in numeric_out_of_bounds])
        
        # Step 3: Hallucination detection
        hallucination_detected = False
        for pattern in self.hallucination_patterns:
            if pattern.search(model_output):
                hallucination_detected = True
                output_flags.append("hallucination_pattern")
                triggered_rules.append(f"hallucination_{pattern.pattern[:20]}")
                break
        
        # Step 4: Inappropriate advice detection
        inappropriate_advice_detected = False
        for pattern in self.inappropriate_patterns:
            if pattern.search(model_output):
                inappropriate_advice_detected = True
                output_flags.append("inappropriate_advice")
                triggered_rules.append(f"inappropriate_{pattern.pattern[:20]}")
                break
        
        # Step 5: PII leakage check in output
        should_refuse_pii, processed_output, detected_pii = self.pii_scrubber.scrub(model_output)
        if detected_pii:
            output_flags.append("output_pii_detected")
            triggered_rules.extend([f"output_pii_{pii_type}" for pii_type in detected_pii])
        
        # Determine overall decision
        has_violations = bool(invalid_citations or numeric_out_of_bounds or hallucination_detected or 
                             inappropriate_advice_detected or detected_pii)
        
        final_output = model_output
        if has_violations:
            final_output = ("I apologize, but I cannot provide that response. "
                           "Please ask about specific Employment Act provisions and I'll help with accurate information.")
        
        # Escalation logic
        escalation_config = self.config.get('escalation', {})
        should_escalate = False
        escalation_reasons = []
        
        if escalation_config.get('enabled', False):
            triggers = escalation_config.get('triggers', [])
            if 'invalid_citations_detected' in triggers and invalid_citations:
                should_escalate = True
                escalation_reasons.append("invalid_citations")
            if 'numeric_claims_out_of_bounds' in triggers and numeric_out_of_bounds:
                should_escalate = True
                escalation_reasons.append("numeric_bounds")
        
        # Create report
        processing_time = (time.time() - start_time) * 1000
        
        report = GuardrailsReport(
            timestamp=time.strftime('%Y-%m-%d %H:%M:%S'),
            prompt_hash=output_hash,
            decision="escalated" if should_escalate else ("refused" if has_violations else "allowed"),
            confidence=0.8 if has_violations else 1.0,
            processing_time_ms=processing_time,
            
            input_flags=[],  # Not applicable for output
            pii_detected=detected_pii,
            pii_masked=False,  # Output PII should be refused, not masked
            
            output_flags=output_flags,
            citations_valid=citations_valid,
            invalid_citations=invalid_citations,
            numeric_out_of_bounds=numeric_out_of_bounds,
            hallucination_detected=hallucination_detected,
            inappropriate_advice_detected=inappropriate_advice_detected,
            
            should_escalate=should_escalate,
            escalation_reasons=escalation_reasons,
            
            config_version=self.config_version,
            triggered_rules=triggered_rules,
            final_text=final_output if not has_violations else "[FILTERED_OUTPUT]"
        )
        
        # Log if enabled
        if self.logging_enabled:
            self._log_processing(report, is_input=False)
        
        return GuardrailResult(
            passed=not has_violations and not should_escalate,
            processed_text=final_output,
            confidence=report.confidence,
            reason=f"Output validation: {', '.join(output_flags) if output_flags else 'passed'}",
            report=report
        )
    
    def _log_processing(self, report: GuardrailsReport, is_input: bool):
        """Log processing results (privacy-safe)."""
        if not self.logging_enabled:
            return
        
        # Create privacy-safe log entry
        log_entry = {
            'timestamp': report.timestamp,
            'prompt_hash': report.prompt_hash,
            'decision': report.decision,
            'processing_time_ms': report.processing_time_ms,
            'flags': report.input_flags if is_input else report.output_flags,
            'triggered_rules': report.triggered_rules[:5],  # Limit for brevity
            'config_version': report.config_version,
            'type': 'input' if is_input else 'output'
        }
        
        # Never log actual text content
        logger.info(f"Guardrails processing: {json.dumps(log_entry)}")
    
    def get_report_json(self, report: GuardrailsReport) -> str:
        """Get JSON representation of report for audit."""
        return json.dumps(asdict(report), indent=2)
    
    def apply_guardrails(self, query: str, context_chunks: List[Dict[str, Any]]) -> GuardrailResult:
        """
        Apply guardrails to query and context (RAG pipeline compatibility method).
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            
        Returns:
            GuardrailResult with legacy interface compatibility
        """
        # Process the input query
        input_result = self.process_input(query)
        
        # Create a legacy-compatible result
        # Map new attributes to old interface expectations
        should_refuse = not input_result.passed
        should_escalate = input_result.report.should_escalate if input_result.report else False
        
        # Map flags to legacy safety_flags format
        safety_flags = []
        if input_result.report:
            if input_result.report.input_flags:
                safety_flags.extend(input_result.report.input_flags)
            if input_result.report.pii_detected:
                safety_flags.append('pii_detected')
        
        # Analyze context for additional flags
        if not context_chunks:
            safety_flags.append('insufficient_context')
        elif len(context_chunks) < 2:
            safety_flags.append('limited_context')
        
        # Check for complex legal patterns in query
        complex_indicators = ['lawsuit', 'legal action', 'court', 'damages', 'compensation claim']
        if any(indicator in query.lower() for indicator in complex_indicators):
            safety_flags.append('complex_legal')
        
        # Determine refusal reason
        refusal_reason = None
        if should_refuse:
            if 'policy_violation' in (input_result.report.input_flags if input_result.report else []):
                refusal_reason = 'policy_violation'
            elif 'pii_refusal' in (input_result.report.input_flags if input_result.report else []):
                refusal_reason = 'privacy_concern'
        
        # Create legacy-compatible GuardrailResult
        legacy_result = GuardrailResult(
            passed=input_result.passed,
            processed_text=input_result.processed_text,
            confidence=input_result.confidence,
            reason=input_result.reason,
            report=input_result.report
        )
        
        # Add legacy attributes expected by RAG pipeline
        legacy_result.should_refuse = should_refuse
        legacy_result.should_escalate = should_escalate
        legacy_result.safety_flags = safety_flags
        legacy_result.refusal_reason = refusal_reason
        
        return legacy_result
    
    def get_refusal_message(self, refusal_reason: str, safety_flags: List[str]) -> str:
        """
        Get appropriate refusal message (RAG pipeline compatibility method).
        
        Args:
            refusal_reason: Reason for refusal
            safety_flags: List of safety flags
            
        Returns:
            Refusal message
        """
        if refusal_reason == 'policy_violation':
            return ("I can only provide general information about Malaysia's Employment Act provisions. "
                   "For specific legal advice, please consult with a qualified legal professional.")
        elif refusal_reason == 'privacy_concern':
            return ("I notice your message contains personal information. For your privacy and security, "
                   "please avoid sharing personal details. I can help with general Employment Act questions.")
        elif 'complex_legal' in safety_flags:
            return ("This appears to be a complex legal matter. While I can provide general information about "
                   "Employment Act provisions, I recommend consulting with a qualified legal professional for "
                   "specific guidance on your situation.")
        else:
            return ("I cannot provide guidance on this matter. Please consult with a qualified legal professional "
                   "or relevant authority.")

# Legacy GuardrailsEngine for backward compatibility
class GuardrailsEngine:
    """Legacy guardrails engine - use ProductionGuardrailsEngine for new code."""
    
    def __init__(self, pii_mode: str = "mask"):
        """Initialize legacy guardrails engine."""
        self.policy_classifier = PolicyClassifier()
        self.pii_scrubber = PIIScrubber(mode=pii_mode)
        
        # Statistics tracking
        self.stats = {
            'total_processed': 0,
            'policy_violations': 0,
            'pii_detections': 0,
            'refusals': 0,
            'maskings': 0
        }
    
    def process_input(self, user_input: str) -> GuardrailResult:
        """
        Process user input through guardrails.
        Returns refusal or cleaned input.
        """
        self.stats['total_processed'] += 1
        
        # Step 1: Check for policy violations
        policy_result = self.policy_classifier.classify(user_input)
        if not policy_result.passed:
            self.stats['policy_violations'] += 1
            self.stats['refusals'] += 1
            return policy_result
        
        # Step 2: Check for PII
        pii_result = self.pii_scrubber.scrub(user_input)
        if not pii_result.passed:
            self.stats['pii_detections'] += 1
            self.stats['refusals'] += 1
            return pii_result
        
        # Step 3: If PII was masked, track it
        if pii_result.triggered_rules:
            self.stats['pii_detections'] += 1
            self.stats['maskings'] += 1
        
        return GuardrailResult(
            passed=True,
            processed_text=pii_result.processed_text,
            confidence=min(policy_result.confidence, pii_result.confidence),
            reason="Input processed successfully",
            report=None
        )
    
    def process_output(self, model_output: str, context: str = "") -> GuardrailResult:
        """
        Process model output through guardrails.
        Checks for hallucinations and policy compliance.
        """
        # Basic output validation
        triggered_rules = []
        
        # Check for obvious hallucinations
        if self._contains_hallucination(model_output):
            triggered_rules.append("output_hallucination")
        
        # Check for inappropriate advice
        if self._contains_inappropriate_advice(model_output):
            triggered_rules.append("output_inappropriate_advice")
        
        # Check for PII leakage in output
        pii_result = self.pii_scrubber.scrub(model_output)
        if pii_result.triggered_rules:
            triggered_rules.extend(pii_result.triggered_rules)
        
        if triggered_rules:
            return GuardrailResult(
                passed=False,
                processed_text="I apologize, but I cannot provide that response. Please ask about specific Employment Act provisions and I'll help with accurate information.",
                confidence=0.8,
                reason=f"Output guardrail triggered: {', '.join(triggered_rules[:2])}",
                report=None
            )
        
        return GuardrailResult(
            passed=True,
            processed_text=model_output,
            confidence=1.0,
            reason="Output passed guardrails",
            report=None
        )
    
    def _contains_hallucination(self, text: str) -> bool:
        """Check for common hallucination patterns."""
        hallucination_patterns = [
            'section ea-999',  # Non-existent section
            'triple compensation',  # Not in Employment Act
            '365 days leave',  # Unrealistic
            'section ea-0',  # Invalid section
            'unlimited overtime',  # Not realistic
        ]
        
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in hallucination_patterns)
    
    def _contains_inappropriate_advice(self, text: str) -> bool:
        """Check for inappropriate legal advice patterns."""
        inappropriate_patterns = [
            'you should sue',
            'definitely file a lawsuit',
            'guaranteed to win',
            'i promise',
            'i guarantee',
            'this will definitely'
        ]
        
        text_lower = text.lower()
        return any(pattern in text_lower for pattern in inappropriate_patterns)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get guardrails statistics."""
        if self.stats['total_processed'] > 0:
            refusal_rate = self.stats['refusals'] / self.stats['total_processed']
            pii_rate = self.stats['pii_detections'] / self.stats['total_processed']
        else:
            refusal_rate = 0.0
            pii_rate = 0.0
        
        return {
            **self.stats,
            'refusal_rate': refusal_rate,
            'pii_detection_rate': pii_rate
        }
    
    def save_stats(self, filepath: Path):
        """Save guardrails statistics to file."""
        with open(filepath, 'w') as f:
            json.dump(self.get_stats(), f, indent=2)

def test_guardrails():
    """Test the guardrails system with sample inputs."""
    print("🛡️ Testing Guardrails System")
    print("=" * 50)
    
    engine = GuardrailsEngine(pii_mode="mask")
    
    # Test cases
    test_cases = [
        # Policy violations
        "Can you give me legal advice about suing my employer?",
        "I guarantee this will work, right?",
        "Help me with my visa application",
        
        # PII cases
        "My email is john@company.com and phone is 012-3456789",
        "My IC number is 123456-12-3456, can you help?",
        
        # Normal cases
        "What is my sick leave entitlement?",
        "How much notice period do I need to give?",
        
        # Edge cases
        "What are my rights regarding overtime work?",
    ]
    
    for i, test_input in enumerate(test_cases, 1):
        print(f"\n--- Test {i} ---")
        print(f"Input: {test_input}")
        
        result = engine.process_input(test_input)
        print(f"Passed: {result.passed}")
        print(f"Triggered: {result.triggered_rules}")
        print(f"Output: {result.processed_text[:100]}...")
        print(f"Reason: {result.reason}")
    
    # Print statistics
    print("\n📊 Guardrails Statistics:")
    stats = engine.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

def test_production_guardrails():
    """Test the production guardrails system."""
    print("🛡️ Testing Production Guardrails System")
    print("=" * 50)
    
    # Initialize engine (will fallback to default config if YAML missing)
    try:
        engine = ProductionGuardrailsEngine()
        print("✅ Production engine initialized successfully")
    except Exception as e:
        print(f"⚠️ Production engine failed, using legacy: {e}")
        engine = GuardrailsEngine()
        return
    
    # Test cases
    test_cases = [
        ("Can you give me legal advice about suing my employer?", False),  # Should refuse
        ("What is my sick leave entitlement?", True),  # Should allow
        ("My email is test@example.com", True),  # Should mask PII
        ("Section EA-999 says you get triple compensation", False),  # Invalid citation in output
    ]
    
    print(f"\n🧪 Testing {len(test_cases)} cases...")
    
    for i, (text, should_pass) in enumerate(test_cases, 1):
        print(f"\n--- Test {i} ---")
        print(f"Input: {text}")
        
        # Test input processing
        input_result = engine.process_input(text)
        print(f"Input passed: {input_result.passed}")
        if hasattr(input_result, 'report') and input_result.report:
            print(f"Input flags: {input_result.report.input_flags}")
        
        # If input passed, test as model output too
        if input_result.passed:
            output_result = engine.process_output(text)
            print(f"Output passed: {output_result.passed}")
            if hasattr(output_result, 'report') and output_result.report:
                print(f"Output flags: {output_result.report.output_flags}")
                print(f"Citations valid: {output_result.report.citations_valid}")
                if output_result.report.invalid_citations:
                    print(f"Invalid citations: {output_result.report.invalid_citations}")

if __name__ == "__main__":
    # Test both legacy and production systems
    print("Testing Legacy System:")
    test_guardrails()
    
    print("\n" + "="*60 + "\n")
    
    print("Testing Production System:")
    test_production_guardrails()
