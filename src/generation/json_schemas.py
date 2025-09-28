#!/usr/bin/env python3
"""
JSON schemas and validation for Employment Act Malaysia compliance agent.
Implements constrained JSON generation with schema validation and auto-repair.
"""

import json
import re
import logging
from typing import Dict, Any, List, Optional, Union
from pydantic import BaseModel, Field, validator
from enum import Enum

logger = logging.getLogger(__name__)


class ConfidenceLevel(str, Enum):
    """Confidence level enumeration."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class Citation(BaseModel):
    """Citation structure for Employment Act references."""
    section_id: str = Field(..., description="Employment Act section ID (e.g., 'EA-60E(1)')")
    snippet: str = Field(..., description="Relevant text snippet from the section", min_length=10, max_length=200)
    relevance_score: Optional[float] = Field(None, description="Relevance score 0.0-1.0")
    
    @validator('section_id')
    def validate_section_id(cls, v):
        """Validate section ID format."""
        # Basic pattern: EA-\d+[A-Z]?(\(\d+\))?
        if not re.match(r'^EA-\d+[A-Z]?(\(\d+\))?.*$', v):
            logger.warning(f"Invalid section ID format: {v}")
        return v


class EmploymentActResponse(BaseModel):
    """Structured response schema for Employment Act queries."""
    answer: str = Field(..., description="Clear, factual answer to the user's question", min_length=20)
    citations: List[Citation] = Field(..., description="Supporting Employment Act citations", min_items=1, max_items=5)
    confidence: ConfidenceLevel = Field(..., description="Confidence level in the response")
    should_escalate: bool = Field(False, description="Whether this query requires human review")
    safety_flags: List[str] = Field(default_factory=list, description="Safety or policy flags")
    numerical_claims: Optional[Dict[str, Union[int, float]]] = Field(None, description="Any numerical claims made")
    
    @validator('answer')
    def validate_answer_quality(cls, v):
        """Validate answer quality."""
        if len(v.strip()) < 20:
            raise ValueError("Answer too short")
        if "I don't know" in v or "I cannot" in v:
            logger.warning("Response contains uncertainty language")
        return v
    
    @validator('citations')
    def validate_citations_not_empty(cls, v):
        """Ensure citations are provided."""
        if not v:
            raise ValueError("At least one citation is required")
        return v


# Few-shot examples for prompt engineering
FEW_SHOT_EXAMPLES = [
    {
        "query": "How many days of annual leave am I entitled to?",
        "response": {
            "answer": "Under the Employment Act 1955, employees are entitled to annual leave based on their length of service. Employees who have worked for less than 2 years receive 8 days of paid annual leave per year. Those who have worked for 2-5 years receive 12 days, and employees with more than 5 years of service receive 16 days of paid annual leave annually.",
            "citations": [
                {
                    "section_id": "EA-60E(1)",
                    "snippet": "An employee shall be entitled to paid annual leave of— (a) eight days for every twelve months of continuous service with the same employer if his period of employment is less than two years",
                    "relevance_score": 0.95
                },
                {
                    "section_id": "EA-60E(1)(b)",
                    "snippet": "twelve days for every twelve months of continuous service with the same employer if his period of employment is two years or more but less than five years",
                    "relevance_score": 0.90
                }
            ],
            "confidence": "high",
            "should_escalate": False,
            "safety_flags": [],
            "numerical_claims": {
                "annual_leave_under_2_years": 8,
                "annual_leave_2_to_5_years": 12,
                "annual_leave_over_5_years": 16
            }
        }
    },
    {
        "query": "What happens if my employer terminates me without notice?",
        "response": {
            "answer": "If your employer terminates your employment without notice and without just cause, they must pay you termination benefits in lieu of notice. The amount depends on your length of service: less than 2 years gets 4 weeks' wages, 2-5 years gets 6 weeks' wages, and 5+ years gets 8 weeks' wages. Additionally, you may be entitled to severance pay.",
            "citations": [
                {
                    "section_id": "EA-12(2)",
                    "snippet": "If either party fails to give such notice, such party shall be liable to pay to the other party an indemnity equal to the amount of wages which would have been earned during the period of such notice",
                    "relevance_score": 0.88
                }
            ],
            "confidence": "medium",
            "should_escalate": False,
            "safety_flags": [],
            "numerical_claims": {
                "notice_period_under_2_years_weeks": 4,
                "notice_period_2_to_5_years_weeks": 6,
                "notice_period_over_5_years_weeks": 8
            }
        }
    }
]


class JSONGenerator:
    """JSON response generator with schema validation and repair."""
    
    def __init__(self):
        """Initialize JSON generator."""
        self.schema = EmploymentActResponse.schema()
        self.examples = FEW_SHOT_EXAMPLES
    
    def create_prompt_with_examples(self, query: str, context_chunks: List[Dict[str, Any]]) -> str:
        """
        Create prompt with few-shot examples and strict JSON formatting.
        
        Args:
            query: User query
            context_chunks: Retrieved context chunks
            
        Returns:
            Formatted prompt with examples
        """
        # Format context
        context_text = "\n\n".join([
            f"Section {chunk.get('section_id', 'Unknown')}: {chunk.get('text', '')}"
            for chunk in context_chunks[:8]  # Limit context
        ])
        
        # Create prompt with examples
        prompt = f"""You are an expert on Malaysia's Employment Act 1955. Provide accurate, factual answers with proper citations.

IMPORTANT: You must respond with valid JSON matching this exact format:

Example 1:
Query: "{self.examples[0]['query']}"
Response: {json.dumps(self.examples[0]['response'], indent=2)}

Example 2:
Query: "{self.examples[1]['query']}"
Response: {json.dumps(self.examples[1]['response'], indent=2)}

Context from Employment Act:
{context_text}

Query: "{query}"
Response:"""
        
        return prompt
    
    def validate_response(self, response_text: str) -> tuple[Optional[EmploymentActResponse], List[str]]:
        """
        Validate JSON response against schema.
        
        Args:
            response_text: Raw response text
            
        Returns:
            Tuple of (validated_response, validation_errors)
        """
        errors = []
        
        try:
            # Extract JSON from response
            json_text = self._extract_json(response_text)
            if not json_text:
                errors.append("No valid JSON found in response")
                return None, errors
            
            # Parse JSON
            response_data = json.loads(json_text)
            
            # Validate against schema
            validated_response = EmploymentActResponse(**response_data)
            return validated_response, []
            
        except json.JSONDecodeError as e:
            errors.append(f"JSON parsing error: {e}")
        except ValueError as e:
            errors.append(f"Validation error: {e}")
        except Exception as e:
            errors.append(f"Unexpected error: {e}")
        
        return None, errors
    
    def auto_repair_response(
        self, 
        response_text: str, 
        context_chunks: List[Dict[str, Any]], 
        validation_errors: List[str]
    ) -> Optional[EmploymentActResponse]:
        """
        Attempt to auto-repair invalid JSON response.
        
        Args:
            response_text: Original response text
            context_chunks: Retrieved context for citation backfill
            validation_errors: List of validation errors
            
        Returns:
            Repaired response or None if repair failed
        """
        try:
            # Extract JSON and parse with lenient parsing
            json_text = self._extract_json(response_text)
            if not json_text:
                return self._create_fallback_response(response_text, context_chunks)
            
            response_data = json.loads(json_text)
            
            # Auto-repair common issues
            response_data = self._repair_response_data(response_data, context_chunks)
            
            # Try validation again
            validated_response = EmploymentActResponse(**response_data)
            logger.info("Successfully auto-repaired JSON response")
            return validated_response
            
        except Exception as e:
            logger.warning(f"Auto-repair failed: {e}")
            return self._create_fallback_response(response_text, context_chunks)
    
    def _extract_json(self, text: str) -> Optional[str]:
        """Extract JSON from response text."""
        # Try to find JSON in curly braces
        json_pattern = r'\{.*\}'
        match = re.search(json_pattern, text, re.DOTALL)
        if match:
            return match.group(0)
        
        # Try to find JSON between markers
        if '```json' in text:
            start = text.find('```json') + 7
            end = text.find('```', start)
            if end > start:
                return text[start:end].strip()
        
        return None
    
    def _repair_response_data(self, data: Dict[str, Any], context_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Repair common issues in response data."""
        # Ensure required fields exist
        if 'answer' not in data or not data['answer']:
            data['answer'] = "Based on the Employment Act provisions, further clarification is needed."
        
        # Backfill citations if empty or missing
        if 'citations' not in data or not data['citations']:
            data['citations'] = self._backfill_citations(context_chunks)
        
        # Fix citation format
        if 'citations' in data:
            data['citations'] = self._fix_citations(data['citations'], context_chunks)
        
        # Set default values for optional fields
        data.setdefault('confidence', 'low')
        data.setdefault('should_escalate', True)  # Conservative default
        data.setdefault('safety_flags', [])
        
        # Normalize confidence
        if data['confidence'] not in ['high', 'medium', 'low']:
            data['confidence'] = 'low'
        
        return data
    
    def _backfill_citations(self, context_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Backfill citations from context chunks."""
        citations = []
        for chunk in context_chunks[:3]:  # Top 3 chunks
            if chunk.get('section_id') and chunk.get('text'):
                citations.append({
                    'section_id': chunk['section_id'],
                    'snippet': chunk['text'][:150] + '...' if len(chunk['text']) > 150 else chunk['text'],
                    'relevance_score': chunk.get('score', 0.5)
                })
        
        return citations if citations else [
            {
                'section_id': 'EA-General',
                'snippet': 'Employment Act 1955 provisions apply.',
                'relevance_score': 0.3
            }
        ]
    
    def _fix_citations(self, citations: List[Any], context_chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Fix citation format issues."""
        fixed_citations = []
        
        for citation in citations:
            if isinstance(citation, dict):
                # Ensure required fields
                if 'section_id' not in citation or not citation['section_id']:
                    citation['section_id'] = 'EA-General'
                
                if 'snippet' not in citation or not citation['snippet']:
                    # Try to find snippet from context
                    for chunk in context_chunks:
                        if chunk.get('section_id') == citation.get('section_id'):
                            citation['snippet'] = chunk['text'][:150] + '...'
                            break
                    else:
                        citation['snippet'] = 'Employment Act provision applies.'
                
                # Limit snippet length
                if len(citation['snippet']) > 200:
                    citation['snippet'] = citation['snippet'][:197] + '...'
                
                fixed_citations.append(citation)
        
        return fixed_citations if fixed_citations else self._backfill_citations(context_chunks)
    
    def _create_fallback_response(self, original_text: str, context_chunks: List[Dict[str, Any]]) -> EmploymentActResponse:
        """Create fallback response when parsing fails."""
        # Extract answer from original text if possible
        answer = original_text[:500] if original_text else "Unable to process request. Please consult the Employment Act or seek legal advice."
        
        return EmploymentActResponse(
            answer=answer,
            citations=self._backfill_citations(context_chunks),
            confidence=ConfidenceLevel.LOW,
            should_escalate=True,
            safety_flags=["parsing_failed"],
            numerical_claims=None
        )


# Test function
def test_json_generation():
    """Test JSON generation and validation."""
    generator = JSONGenerator()
    
    # Test prompt creation
    mock_chunks = [
        {
            'section_id': 'EA-60E',
            'text': 'An employee shall be entitled to paid annual leave...',
            'score': 0.8
        }
    ]
    
    prompt = generator.create_prompt_with_examples("How much annual leave?", mock_chunks)
    print("Generated prompt:")
    print(prompt[:500] + "...")
    
    # Test validation
    valid_response = json.dumps(FEW_SHOT_EXAMPLES[0]['response'])
    validated, errors = generator.validate_response(valid_response)
    print(f"\nValidation result: {validated is not None}, Errors: {errors}")
    
    # Test auto-repair
    broken_response = '{"answer": "Some answer", "citations": []}'
    repaired = generator.auto_repair_response(broken_response, mock_chunks, ["missing citations"])
    print(f"\nAuto-repair result: {repaired is not None}")


if __name__ == "__main__":
    test_json_generation()