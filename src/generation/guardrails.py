#!/usr/bin/env python3
# python src/generation/guardrails.py
"""
Guardrails for Employment Act Malaysia compliance agent.
Handles refusal cases, safety filtering, and escalation logic.
"""

import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class SafetyFlag(Enum):
    """Safety flags for different types of issues."""
    OUT_OF_SCOPE = "out_of_scope"
    INSUFFICIENT_CONTEXT = "insufficient_context"
    LOW_CONFIDENCE = "low_confidence"
    PII_DETECTED = "pii_detected"
    HARMFUL_CONTENT = "harmful_content"
    COMPLEX_LEGAL = "complex_legal"


@dataclass
class GuardrailsResult:
    """Result of guardrails check."""
    should_refuse: bool
    should_escalate: bool
    safety_flags: List[str]
    refusal_reason: Optional[str] = None


class EmploymentActGuardrails:
    """Guardrails system for Employment Act Malaysia compliance agent."""
    
    def __init__(self):
        """Initialize guardrails with patterns and thresholds."""
        
        # PII patterns (Malaysian context)
        self.pii_patterns = [
            r'\b\d{6}-\d{2}-\d{4}\b',  # Malaysian IC format
            r'\b\d{12}\b',  # 12-digit IC without dashes
            r'\b[A-Z]{1,2}\d{1,4}[A-Z]?\b',  # Malaysian car plate
            r'\b\+?60\d{8,10}\b',  # Malaysian phone numbers
            r'\b\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\b',  # Credit card numbers
            r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',  # Email addresses
        ]
        
        # Employment Act scope keywords
        self.employment_keywords = [
            'employment', 'employee', 'employer', 'workplace', 'salary', 'wages',
            'overtime', 'leave', 'annual leave', 'sick leave', 'maternity',
            'termination', 'notice', 'severance', 'contract', 'working hours',
            'rest day', 'public holiday', 'bonus', 'allowance', 'increment',
            'promotion', 'disciplinary', 'misconduct', 'resignation', 'retirement'
        ]
        
        # Out-of-scope indicators
        self.out_of_scope_patterns = [
            # Other legal areas
            r'\b(criminal|civil|family|divorce|custody|inheritance|property|contract(?!\s+of\s+employment)|tort|defamation)\s+law\b',
            r'\b(traffic|driving|vehicle|motor|accident|insurance|medical|immigration|tax|income\s+tax)\b',
            
            # Other countries' laws
            r'\b(singapore|thailand|indonesia|philippines|vietnam|brunei|uk|usa|australia)\s+(employment|law|act)\b',
            
            # Non-legal topics
            r'\b(recipe|cooking|travel|sports|entertainment|weather|technology|programming|software)\b',
            
            # Personal advice (non-legal)
            r'\b(relationship|dating|marriage|health|medical|financial\s+planning|investment)\b'
        ]
        
        # Complex legal indicators that should trigger escalation
        self.complex_legal_patterns = [
            r'\b(constitutional|federal|state|jurisdiction|appeal|judicial\s+review)\b',
            r'\b(class\s+action|representative|collective|union|strike|industrial\s+action)\b',
            r'\b(discrimination|harassment|whistleblowing|retaliation)\b',
            r'\b(workers?\s+compensation|occupational\s+safety|osha|socso)\b'
        ]
        
        # Confidence thresholds
        self.min_confidence_threshold = 0.7
        self.escalation_confidence_threshold = 0.8
        
    def check_pii(self, text: str) -> bool:
        """Check if text contains personally identifiable information.
        
        Args:
            text: Input text to check
            
        Returns:
            True if PII detected
        """
        text_upper = text.upper()
        
        for pattern in self.pii_patterns:
            if re.search(pattern, text_upper):
                return True
        
        return False
    
    def check_scope(self, query: str) -> Tuple[bool, str]:
        """Check if query is within Employment Act scope.
        
        Args:
            query: User query
            
        Returns:
            (is_in_scope, reason_if_out_of_scope)
        """
        query_lower = query.lower()
        
        # Check for out-of-scope patterns
        for pattern in self.out_of_scope_patterns:
            if re.search(pattern, query_lower):
                return False, f"Query appears to be about {pattern} which is outside Employment Act scope"
        
        # Check for employment-related keywords
        has_employment_keywords = any(keyword in query_lower for keyword in self.employment_keywords)
        
        # If no employment keywords and query is about legal matters, likely out of scope
        legal_terms = ['law', 'legal', 'rights', 'regulation', 'act', 'section', 'clause']
        has_legal_terms = any(term in query_lower for term in legal_terms)
        
        if has_legal_terms and not has_employment_keywords:
            return False, "Legal query does not appear to be employment-related"
        
        # If very short query without clear employment context
        if len(query.split()) < 3 and not has_employment_keywords:
            return False, "Query too vague to determine if employment-related"
        
        return True, ""
    
    def check_complexity(self, query: str) -> bool:
        """Check if query indicates complex legal matter requiring escalation.
        
        Args:
            query: User query
            
        Returns:
            True if complex legal matter detected
        """
        query_lower = query.lower()
        
        for pattern in self.complex_legal_patterns:
            if re.search(pattern, query_lower):
                return True
        
        # Check for multiple legal concepts in one query
        legal_concept_count = 0
        employment_concepts = [
            'termination', 'severance', 'discrimination', 'harassment', 
            'overtime', 'salary', 'bonus', 'leave', 'contract'
        ]
        
        for concept in employment_concepts:
            if concept in query_lower:
                legal_concept_count += 1
        
        # If query involves multiple complex concepts, escalate
        if legal_concept_count >= 3:
            return True
        
        return False
    
    def evaluate_context_quality(self, retrieved_chunks: List[Dict[str, Any]], query: str) -> float:
        """Evaluate quality of retrieved context for the query.
        
        Args:
            retrieved_chunks: Retrieved chunks from hybrid retriever
            query: User query
            
        Returns:
            Context quality score (0-1)
        """
        if not retrieved_chunks:
            return 0.0
        
        # Check if top result has reasonable score
        top_score = retrieved_chunks[0].get('score', 0.0)
        
        # Check if multiple results have good scores (diversity)
        good_results = sum(1 for chunk in retrieved_chunks if chunk.get('score', 0.0) > 0.5)
        diversity_score = min(good_results / 3.0, 1.0)  # Normalize to max 3 good results
        
        # Check if results have section IDs (structured content)
        has_sections = sum(1 for chunk in retrieved_chunks if chunk.get('section_id'))
        section_score = has_sections / len(retrieved_chunks)
        
        # Overall quality score
        quality_score = (top_score * 0.5) + (diversity_score * 0.3) + (section_score * 0.2)
        
        return min(quality_score, 1.0)
    
    def apply_guardrails(
        self, 
        query: str, 
        retrieved_chunks: List[Dict[str, Any]], 
        confidence_score: Optional[float] = None
    ) -> GuardrailsResult:
        """Apply all guardrails and return decision.
        
        Args:
            query: User query
            retrieved_chunks: Retrieved context chunks
            confidence_score: Model confidence score (if available)
            
        Returns:
            GuardrailsResult with decision and flags
        """
        safety_flags = []
        should_refuse = False
        should_escalate = False
        refusal_reason = None
        
        # Check for PII in query
        if self.check_pii(query):
            safety_flags.append(SafetyFlag.PII_DETECTED.value)
            should_refuse = True
            refusal_reason = "Cannot process queries containing personal identification information"
        
        # Check scope
        is_in_scope, scope_reason = self.check_scope(query)
        if not is_in_scope:
            safety_flags.append(SafetyFlag.OUT_OF_SCOPE.value)
            should_refuse = True
            refusal_reason = scope_reason
        
        # Check complexity
        if self.check_complexity(query):
            safety_flags.append(SafetyFlag.COMPLEX_LEGAL.value)
            should_escalate = True
        
        # Evaluate context quality
        context_quality = self.evaluate_context_quality(retrieved_chunks, query)
        if context_quality < 0.4:
            safety_flags.append(SafetyFlag.INSUFFICIENT_CONTEXT.value)
            should_escalate = True
        
        # Check confidence score
        if confidence_score is not None:
            if confidence_score < self.min_confidence_threshold:
                safety_flags.append(SafetyFlag.LOW_CONFIDENCE.value)
                should_escalate = True
            elif confidence_score < self.escalation_confidence_threshold:
                should_escalate = True
        
        return GuardrailsResult(
            should_refuse=should_refuse,
            should_escalate=should_escalate,
            safety_flags=safety_flags,
            refusal_reason=refusal_reason
        )
    
    def get_refusal_message(self, reason: str, safety_flags: List[str]) -> str:
        """Generate appropriate refusal message.
        
        Args:
            reason: Specific refusal reason
            safety_flags: List of safety flags
            
        Returns:
            Formatted refusal message
        """
        if SafetyFlag.PII_DETECTED.value in safety_flags:
            return "I cannot process queries that contain personal identification information. Please rephrase your question without including IC numbers, phone numbers, or other personal details."
        
        if SafetyFlag.OUT_OF_SCOPE.value in safety_flags:
            return f"I can only provide guidance on Employment Act Malaysia matters. {reason}. Please consult with the appropriate legal professional or authority for questions outside employment law."
        
        return "I cannot provide guidance on this matter. Please consult with a qualified legal professional."


def main():
    """Test guardrails system."""
    guardrails = EmploymentActGuardrails()
    
    # Test cases
    test_cases = [
        "How many days of annual leave am I entitled to?",  # Good
        "What's the speed limit in KL?",  # Out of scope
        "My IC is 123456-78-9012, can I get overtime pay?",  # PII
        "Complex union strike action with federal jurisdiction appeal",  # Complex
        "Recipe for nasi lemak",  # Out of scope
    ]
    
    for query in test_cases:
        print(f"\nQuery: {query}")
        
        # Mock retrieved chunks
        chunks = [{"score": 0.8, "section_id": "EA-2022-15", "text": "sample text"}]
        
        result = guardrails.apply_guardrails(query, chunks, confidence_score=0.75)
        
        print(f"Should refuse: {result.should_refuse}")
        print(f"Should escalate: {result.should_escalate}")
        print(f"Safety flags: {result.safety_flags}")
        if result.refusal_reason:
            print(f"Refusal reason: {result.refusal_reason}")


if __name__ == "__main__":
    main()