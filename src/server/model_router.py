#!/usr/bin/env python3
"""
Model routing strategy for Employment Act Malaysia compliance agent.
Routes between Llama-3.1-8B-Instruct (default) and Llama-3.1-70B-Instruct (escalation).
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class ModelTier(Enum):
    """Model tier enumeration."""
    SMALL = "8B"  # Llama-3.1-8B-Instruct (default)
    LARGE = "70B"  # Llama-3.1-70B-Instruct (escalation)


@dataclass
class RoutingDecision:
    """Model routing decision with reasoning."""
    model_tier: ModelTier
    model_name: str
    confidence_threshold: float
    reasoning: List[str]
    escalation_signals: List[str]


class ModelRouter:
    """
    Routes queries between 8B (default) and 70B (escalation) models based on:
    - Retrieval confidence (top_score < 0.25)
    - Multi-section synthesis requirements
    - Policy/safety hits from guardrails
    - Explicit user escalation requests
    """
    
    def __init__(self):
        """Initialize model router with configuration."""
        # Model configuration
        self.small_model = os.getenv("SMALL_MODEL", "meta-llama/Llama-3.1-8B-Instruct")
        self.large_model = os.getenv("LARGE_MODEL", "meta-llama/Llama-3.1-70B-Instruct")
        
        # Routing thresholds
        self.low_confidence_threshold = float(os.getenv("LOW_CONFIDENCE_THRESHOLD", "0.25"))
        self.multi_section_threshold = int(os.getenv("MULTI_SECTION_THRESHOLD", "3"))
        
        # Escalation keywords
        self.escalation_keywords = [
            "review", "escalate", "complex", "detailed analysis",
            "legal opinion", "comprehensive", "thorough review",
            "expert analysis", "detailed explanation"
        ]
        
        logger.info(f"Model router initialized: {self.small_model} → {self.large_model}")
        logger.info(f"Escalation thresholds: confidence < {self.low_confidence_threshold}, sections >= {self.multi_section_threshold}")
    
    def should_escalate(
        self,
        query: str,
        retrieval_results: List[Dict[str, Any]],
        guardrails_flags: List[str],
        user_escalation: bool = False
    ) -> RoutingDecision:
        """
        Determine if query should be escalated to larger model.
        
        Args:
            query: User query text
            retrieval_results: Retrieved context chunks with scores
            guardrails_flags: Flags from guardrails processing
            user_escalation: Explicit user escalation request
            
        Returns:
            RoutingDecision with model choice and reasoning
        """
        escalation_signals = []
        reasoning = []
        
        # 1. Explicit user escalation request
        if user_escalation:
            escalation_signals.append("explicit_user_request")
            reasoning.append("User explicitly requested review/escalation")
        
        # 2. Check for escalation keywords in query
        query_lower = query.lower()
        found_keywords = [kw for kw in self.escalation_keywords if kw in query_lower]
        if found_keywords:
            escalation_signals.append("escalation_keywords")
            reasoning.append(f"Query contains escalation keywords: {', '.join(found_keywords)}")
        
        # 3. Low retrieval confidence (top score < 0.25)
        top_score = 0.0
        if retrieval_results:
            top_score = max(result.get('score', 0.0) for result in retrieval_results)
        
        if top_score < self.low_confidence_threshold:
            escalation_signals.append("low_retrieval_confidence")
            reasoning.append(f"Low retrieval confidence: top_score={top_score:.3f} < {self.low_confidence_threshold}")
        
        # 4. Multi-section synthesis requirement
        unique_sections = set()
        for result in retrieval_results:
            section_id = result.get('section_id')
            if section_id:
                unique_sections.add(section_id)
        
        if len(unique_sections) >= self.multi_section_threshold:
            escalation_signals.append("multi_section_synthesis")
            reasoning.append(f"Multi-section synthesis required: {len(unique_sections)} sections >= {self.multi_section_threshold}")
        
        # 5. Policy/safety hits from guardrails
        safety_flags = [flag for flag in guardrails_flags if flag in [
            "legal_advice", "guarantees", "inappropriate_advice", "escalation_required"
        ]]
        if safety_flags:
            escalation_signals.append("safety_policy_hits")
            reasoning.append(f"Safety/policy flags triggered: {', '.join(safety_flags)}")
        
        # 6. Complex legal terms or cross-references
        complex_patterns = [
            "interpretation", "cross-reference", "exception", "provision",
            "notwithstanding", "subject to", "in accordance with",
            "comparative analysis", "precedent", "jurisprudence"
        ]
        found_complex = [pattern for pattern in complex_patterns if pattern in query_lower]
        if found_complex:
            escalation_signals.append("complex_legal_language")
            reasoning.append(f"Complex legal language detected: {', '.join(found_complex)}")
        
        # Decision logic
        should_use_large_model = bool(escalation_signals)
        
        if should_use_large_model:
            model_tier = ModelTier.LARGE
            model_name = self.large_model
            confidence_threshold = 0.8  # Higher threshold for large model
            reasoning.insert(0, f"ESCALATED to {ModelTier.LARGE.value} model")
        else:
            model_tier = ModelTier.SMALL
            model_name = self.small_model
            confidence_threshold = 0.7  # Standard threshold for small model
            reasoning.insert(0, f"Using default {ModelTier.SMALL.value} model")
        
        decision = RoutingDecision(
            model_tier=model_tier,
            model_name=model_name,
            confidence_threshold=confidence_threshold,
            reasoning=reasoning,
            escalation_signals=escalation_signals
        )
        
        logger.info(f"Routing decision: {model_tier.value} model ({len(escalation_signals)} escalation signals)")
        if escalation_signals:
            logger.info(f"Escalation signals: {', '.join(escalation_signals)}")
        
        return decision
    
    def get_model_config(self, model_tier: ModelTier) -> Dict[str, Any]:
        """
        Get model-specific configuration.
        
        Args:
            model_tier: Model tier (SMALL or LARGE)
            
        Returns:
            Configuration dictionary for the model
        """
        if model_tier == ModelTier.LARGE:
            return {
                "model_name": self.large_model,
                "max_tokens": int(os.getenv("LARGE_MODEL_MAX_TOKENS", "1024")),
                "temperature": float(os.getenv("LARGE_MODEL_TEMPERATURE", "0.05")),  # Lower temp for accuracy
                "top_p": float(os.getenv("LARGE_MODEL_TOP_P", "0.9")),
                "timeout": float(os.getenv("LARGE_MODEL_TIMEOUT", "120.0")),  # Longer timeout
                "quantization": os.getenv("LARGE_MODEL_QUANTIZATION", "4bit"),
            }
        else:
            return {
                "model_name": self.small_model,
                "max_tokens": int(os.getenv("SMALL_MODEL_MAX_TOKENS", "512")),
                "temperature": float(os.getenv("SMALL_MODEL_TEMPERATURE", "0.1")),
                "top_p": float(os.getenv("SMALL_MODEL_TOP_P", "0.95")),
                "timeout": float(os.getenv("SMALL_MODEL_TIMEOUT", "60.0")),
                "quantization": os.getenv("SMALL_MODEL_QUANTIZATION", "4bit"),
            }
    
    def format_routing_metadata(self, decision: RoutingDecision) -> Dict[str, Any]:
        """
        Format routing decision for response metadata.
        
        Args:
            decision: Routing decision
            
        Returns:
            Metadata dictionary for API response
        """
        return {
            "model_tier": decision.model_tier.value,
            "model_name": decision.model_name,
            "escalation_signals": decision.escalation_signals,
            "escalation_reason": "; ".join(decision.reasoning),
            "confidence_threshold": decision.confidence_threshold
        }


# Factory function for dependency injection
def create_model_router() -> ModelRouter:
    """Create model router instance."""
    return ModelRouter()


# Test function
def test_model_routing():
    """Test model routing logic."""
    router = ModelRouter()
    
    # Test cases
    test_cases = [
        {
            "name": "Simple query - should use 8B",
            "query": "What is annual leave?",
            "retrieval_results": [{"score": 0.8, "section_id": "EA-60E"}],
            "guardrails_flags": [],
            "user_escalation": False
        },
        {
            "name": "Low confidence - should escalate to 70B",
            "query": "Complex employment issue",
            "retrieval_results": [{"score": 0.15, "section_id": "EA-60E"}],
            "guardrails_flags": [],
            "user_escalation": False
        },
        {
            "name": "Multi-section synthesis - should escalate to 70B",
            "query": "Overtime and leave interaction",
            "retrieval_results": [
                {"score": 0.7, "section_id": "EA-60A"},
                {"score": 0.6, "section_id": "EA-60E"},
                {"score": 0.5, "section_id": "EA-102"}
            ],
            "guardrails_flags": [],
            "user_escalation": False
        },
        {
            "name": "User escalation request - should escalate to 70B",
            "query": "Please review this complex case",
            "retrieval_results": [{"score": 0.8, "section_id": "EA-60E"}],
            "guardrails_flags": [],
            "user_escalation": True
        },
        {
            "name": "Safety flags - should escalate to 70B",
            "query": "Employment law question",
            "retrieval_results": [{"score": 0.8, "section_id": "EA-60E"}],
            "guardrails_flags": ["legal_advice", "escalation_required"],
            "user_escalation": False
        }
    ]
    
    print("Testing model routing decisions...\n")
    
    for test_case in test_cases:
        print(f"Test: {test_case['name']}")
        decision = router.should_escalate(
            test_case["query"],
            test_case["retrieval_results"],
            test_case["guardrails_flags"],
            test_case["user_escalation"]
        )
        
        print(f"  → Model: {decision.model_tier.value}")
        print(f"  → Signals: {decision.escalation_signals}")
        print(f"  → Reasoning: {decision.reasoning[0]}")
        print()


if __name__ == "__main__":
    test_model_routing()