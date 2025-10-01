#!/usr/bin/env python3
"""
Pydantic schemas for SFT dataset validation.
"""

from typing import List, Optional
from pydantic import BaseModel, Field, field_validator
import re


class SFTExample(BaseModel):
    """Schema for a single SFT training example."""
    
    instruction: str = Field(..., min_length=10, max_length=2000, description="The user instruction/question")
    input: str = Field(default="", max_length=500, description="Additional input context (can be empty)")
    output: str = Field(..., min_length=20, max_length=1000, description="The assistant response")
    citations: List[str] = Field(..., min_items=1, max_items=10, description="Section ID citations")
    category: Optional[str] = Field(None, description="Question category")
    source_chunk_id: Optional[str] = Field(None, description="Source chunk identifier")
    has_numeric_claims: bool = Field(default=False, description="Whether response contains numeric claims")
    
    @field_validator('citations')
    @classmethod
    def validate_citations(cls, v):
        """Validate citation format."""
        if not v:
            raise ValueError("Citations cannot be empty")
        
        citation_pattern = r'^EA-\d{4}-\d+[A-Z]*(?:\(\d+\))?$'
        for citation in v:
            if not re.match(citation_pattern, citation):
                raise ValueError(f"Invalid citation format: {citation}")
        return v
    
    @field_validator('instruction')
    @classmethod
    def validate_instruction(cls, v):
        """Validate instruction quality."""
        if len(v.strip()) < 10:
            raise ValueError("Instruction too short")
        
        # Check for reasonable question patterns
        question_indicators = ['?', 'how', 'what', 'when', 'where', 'why', 'can', 'should', 'is', 'are']
        if not any(indicator in v.lower() for indicator in question_indicators):
            raise ValueError("Instruction should be a question or request")
        
        return v.strip()
    
    @field_validator('output')
    @classmethod
    def validate_output(cls, v):
        """Validate output quality."""
        if len(v.strip()) < 20:
            raise ValueError("Output too short")
        
        # Should contain some legal terminology
        legal_terms = ['employment act', 'section', 'according', 'entitled', 'shall', 'must']
        if not any(term in v.lower() for term in legal_terms):
            raise ValueError("Output should contain legal terminology")
        
        return v.strip()