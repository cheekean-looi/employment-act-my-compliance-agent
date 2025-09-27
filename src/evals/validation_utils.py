#!/usr/bin/env python3
"""
Validation Utilities
Shared utilities for citation validation, numeric sanity checking, and content validation.
"""

import re
import json
from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass
from pathlib import Path
import pickle

@dataclass
class CitationValidationResult:
    """Result of citation validation."""
    valid_citations: List[str]
    invalid_citations: List[str]
    missing_citations: List[str]
    validation_passed: bool
    details: str

@dataclass
class NumericValidationResult:
    """Result of numeric validation."""
    extracted_numbers: List[Dict[str, Any]]
    out_of_bounds: List[Dict[str, Any]]
    validation_passed: bool
    details: str

class CitationValidator:
    """
    Validates Employment Act citations against known corpus.
    Shared by guardrails and generation evaluation.
    """
    
    def __init__(self, known_sections: Set[str] = None):
        """
        Initialize citation validator.
        
        Args:
            known_sections: Set of valid section IDs (e.g., from corpus)
        """
        self.known_sections = known_sections or self._get_default_sections()
        self.citation_patterns = [
            r'Section EA-(\d+[A-Z]*)',  # Section EA-60F
            r'Employment Act Section (\d+[A-Z]*)',  # Employment Act Section 60
            r'Section (\d+[A-Z]*)',  # Section 12
            r'EA-(\d+[A-Z]*)',  # EA-60F
        ]
    
    def _get_default_sections(self) -> Set[str]:
        """Get default set of known Employment Act sections."""
        # Common Employment Act sections - in production, load from corpus
        return {
            "EA-2", "EA-3", "EA-4", "EA-5", "EA-6", "EA-7", "EA-8", "EA-9", "EA-10",
            "EA-11", "EA-12", "EA-13", "EA-14", "EA-15", "EA-16", "EA-17", "EA-18", "EA-19", "EA-20",
            "EA-25", "EA-30", "EA-35", "EA-37", "EA-40", "EA-42", "EA-44", "EA-45", "EA-47",
            "EA-60", "EA-60A", "EA-60B", "EA-60C", "EA-60D", "EA-60E", "EA-60F",
            "EA-77", "EA-78", "EA-80", "EA-81", "EA-82", "EA-83", "EA-84", "EA-90",
            "EA-100", "EA-101", "EA-102", "EA-103", "EA-104", "EA-105"
        }
    
    def load_sections_from_corpus(self, store_path: Path) -> Set[str]:
        """
        Load valid sections from corpus store file.
        
        Args:
            store_path: Path to store.pkl or chunks file
        """
        if not store_path.exists():
            print(f"⚠️ Store file not found: {store_path}, using default sections")
            return self.known_sections
        
        try:
            if store_path.suffix == '.pkl':
                with open(store_path, 'rb') as f:
                    store = pickle.load(f)
                sections = set()
                for chunk in store:
                    if 'section_id' in chunk:
                        sections.add(chunk['section_id'])
            elif store_path.suffix in ['.json', '.jsonl']:
                sections = set()
                with open(store_path, 'r') as f:
                    if store_path.suffix == '.jsonl':
                        for line in f:
                            chunk = json.loads(line)
                            if 'section_id' in chunk:
                                sections.add(chunk['section_id'])
                    else:
                        data = json.load(f)
                        for chunk in data:
                            if 'section_id' in chunk:
                                sections.add(chunk['section_id'])
            else:
                print(f"⚠️ Unsupported store format: {store_path}")
                return self.known_sections
            
            print(f"📚 Loaded {len(sections)} valid sections from corpus")
            return sections
            
        except Exception as e:
            print(f"⚠️ Error loading sections from {store_path}: {e}")
            return self.known_sections
    
    def extract_citations(self, text: str) -> List[str]:
        """Extract all citations from text."""
        citations = []
        for pattern in self.citation_patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                # Normalize to EA-XX format
                if not match.startswith('EA-'):
                    citation = f"EA-{match}"
                else:
                    citation = match
                citations.append(citation)
        
        # Remove duplicates while preserving order
        seen = set()
        unique_citations = []
        for citation in citations:
            if citation not in seen:
                seen.add(citation)
                unique_citations.append(citation)
        
        return unique_citations
    
    def validate_citations(self, text: str) -> CitationValidationResult:
        """
        Validate all citations in text against known sections.
        
        Returns:
            CitationValidationResult with valid/invalid citations
        """
        extracted = self.extract_citations(text)
        
        if not extracted:
            return CitationValidationResult(
                valid_citations=[],
                invalid_citations=[],
                missing_citations=[],
                validation_passed=True,  # No citations to validate
                details="No citations found"
            )
        
        valid_citations = []
        invalid_citations = []
        
        for citation in extracted:
            if citation in self.known_sections:
                valid_citations.append(citation)
            else:
                invalid_citations.append(citation)
        
        validation_passed = len(invalid_citations) == 0
        
        details = []
        if valid_citations:
            details.append(f"Valid: {', '.join(valid_citations)}")
        if invalid_citations:
            details.append(f"Invalid: {', '.join(invalid_citations)}")
        
        return CitationValidationResult(
            valid_citations=valid_citations,
            invalid_citations=invalid_citations,
            missing_citations=[],  # Could implement claim detection
            validation_passed=validation_passed,
            details="; ".join(details)
        )

class NumericValidator:
    """
    Validates numeric claims in Employment Act context.
    Checks for reasonable bounds on common employment metrics.
    """
    
    def __init__(self):
        """Initialize numeric validator with Employment Act bounds."""
        
        # Define reasonable bounds for Employment Act context
        self.bounds = {
            # Leave entitlements (days)
            "annual_leave": {"min": 8, "max": 60, "unit": "days"},
            "sick_leave": {"min": 14, "max": 365, "unit": "days"},
            "maternity_leave": {"min": 60, "max": 98, "unit": "days"},
            
            # Notice periods (weeks/days)
            "notice_period": {"min": 1, "max": 16, "unit": "weeks"},
            "notice_days": {"min": 7, "max": 112, "unit": "days"},
            
            # Working time
            "daily_hours": {"min": 1, "max": 12, "unit": "hours"},
            "weekly_hours": {"min": 1, "max": 72, "unit": "hours"},
            "overtime_rate": {"min": 1.0, "max": 3.0, "unit": "multiplier"},
            
            # Service years
            "service_years": {"min": 0, "max": 60, "unit": "years"},
            
            # Compensation
            "severance_multiplier": {"min": 10, "max": 30, "unit": "days"}
        }
        
        # Patterns to extract numbers with context
        self.numeric_patterns = [
            # X days of leave
            r'(\d+)\s+(days?)\s+(?:of\s+)?(annual\s+leave|sick\s+leave|maternity\s+leave|leave)',
            # X weeks notice
            r'(\d+)\s+(weeks?)\s+(?:of\s+)?notice',
            # X hours per day/week
            r'(\d+)\s+(hours?)\s+per\s+(day|week)',
            # X years of service
            r'(\d+)\s+(years?)\s+(?:of\s+)?service',
            # X times/multiplier
            r'(\d+(?:\.\d+)?)\s+times?(?:\s+(?:the\s+)?(?:normal\s+)?(?:hourly\s+)?rate)?',
            # Standalone numbers with units
            r'(\d+(?:\.\d+)?)\s+(days?|weeks?|months?|years?|hours?)',
        ]
    
    def extract_numbers_with_context(self, text: str) -> List[Dict[str, Any]]:
        """Extract numbers with their context from text."""
        
        extracted = []
        text_lower = text.lower()
        
        for pattern in self.numeric_patterns:
            matches = re.finditer(pattern, text_lower)
            for match in matches:
                groups = match.groups()
                
                if len(groups) >= 2:
                    try:
                        number = float(groups[0])
                        unit = groups[1].rstrip('s')  # Remove plural
                        context = groups[2] if len(groups) > 2 else ""
                        
                        # Determine category based on context
                        category = self._categorize_numeric_claim(text_lower, match.start(), match.end())
                        
                        extracted.append({
                            "number": number,
                            "unit": unit,
                            "context": context,
                            "category": category,
                            "text_span": match.group(),
                            "position": (match.start(), match.end())
                        })
                    except ValueError:
                        continue
        
        return extracted
    
    def _categorize_numeric_claim(self, text: str, start: int, end: int) -> str:
        """Categorize a numeric claim based on surrounding context."""
        
        # Look at text around the number for context clues
        context_window = 50
        context_start = max(0, start - context_window)
        context_end = min(len(text), end + context_window)
        context = text[context_start:context_end]
        
        # Category keywords
        if any(word in context for word in ['annual', 'vacation']):
            return "annual_leave"
        elif any(word in context for word in ['sick', 'medical']):
            return "sick_leave"
        elif any(word in context for word in ['maternity', 'pregnancy']):
            return "maternity_leave"
        elif any(word in context for word in ['notice', 'resignation', 'termination']):
            return "notice_period"
        elif any(word in context for word in ['overtime', 'extra']):
            return "overtime_rate"
        elif any(word in context for word in ['service', 'employment', 'work']):
            return "service_years"
        elif any(word in context for word in ['hour', 'daily', 'per day']):
            return "daily_hours"
        elif any(word in context for word in ['week', 'weekly']):
            return "weekly_hours"
        elif any(word in context for word in ['severance', 'retrenchment']):
            return "severance_multiplier"
        else:
            return "general"
    
    def validate_numeric_claims(self, text: str) -> NumericValidationResult:
        """
        Validate all numeric claims in text against reasonable bounds.
        
        Returns:
            NumericValidationResult with validation details
        """
        extracted = self.extract_numbers_with_context(text)
        
        if not extracted:
            return NumericValidationResult(
                extracted_numbers=[],
                out_of_bounds=[],
                validation_passed=True,
                details="No numeric claims found"
            )
        
        out_of_bounds = []
        
        for item in extracted:
            category = item["category"]
            number = item["number"]
            unit = item["unit"]
            
            # Check against bounds if we have them for this category
            if category in self.bounds:
                bounds = self.bounds[category]
                expected_unit = bounds["unit"]
                
                # Unit conversion if needed
                converted_number = self._convert_units(number, unit, expected_unit)
                
                if converted_number is not None:
                    if converted_number < bounds["min"] or converted_number > bounds["max"]:
                        out_of_bounds.append({
                            **item,
                            "converted_number": converted_number,
                            "expected_range": f"{bounds['min']}-{bounds['max']} {expected_unit}",
                            "reason": f"Outside reasonable range for {category}"
                        })
            # Check for obviously unreasonable numbers
            elif number > 1000 and unit in ["days", "weeks", "hours"]:
                out_of_bounds.append({
                    **item,
                    "reason": f"Unreasonably large: {number} {unit}"
                })
        
        validation_passed = len(out_of_bounds) == 0
        
        details = []
        if extracted:
            details.append(f"Found {len(extracted)} numeric claims")
        if out_of_bounds:
            details.append(f"{len(out_of_bounds)} out of bounds")
        
        return NumericValidationResult(
            extracted_numbers=extracted,
            out_of_bounds=out_of_bounds,
            validation_passed=validation_passed,
            details="; ".join(details)
        )
    
    def _convert_units(self, number: float, from_unit: str, to_unit: str) -> Optional[float]:
        """Convert between time units."""
        
        # Conversion factors to days
        to_days = {
            "day": 1,
            "week": 7,
            "month": 30,
            "year": 365
        }
        
        # Normalize units
        from_unit = from_unit.rstrip('s')
        to_unit = to_unit.rstrip('s')
        
        if from_unit == to_unit:
            return number
        
        # Convert via days
        if from_unit in to_days and to_unit in to_days:
            days = number * to_days[from_unit]
            return days / to_days[to_unit]
        
        # Special cases
        if from_unit == "multiplier" and to_unit == "multiplier":
            return number
        
        return None  # Can't convert

def test_validation_utils():
    """Test the validation utilities."""
    
    print("🧪 Testing Validation Utilities")
    print("=" * 35)
    
    # Test citation validation
    print("\n📋 Citation Validation Test:")
    citation_validator = CitationValidator()
    
    test_text = "According to Section EA-60F and Employment Act Section 12, you are entitled to leave. Invalid reference to Section EA-999."
    
    result = citation_validator.validate_citations(test_text)
    print(f"  Valid: {result.valid_citations}")
    print(f"  Invalid: {result.invalid_citations}")
    print(f"  Passed: {result.validation_passed}")
    
    # Test numeric validation
    print("\n🔢 Numeric Validation Test:")
    numeric_validator = NumericValidator()
    
    test_text = "You are entitled to 15 days of annual leave and 8 weeks notice period. Overtime is paid at 1.5 times the normal rate. Invalid claim: 999 days of sick leave."
    
    result = numeric_validator.validate_numeric_claims(test_text)
    print(f"  Extracted: {len(result.extracted_numbers)} numbers")
    print(f"  Out of bounds: {len(result.out_of_bounds)}")
    print(f"  Passed: {result.validation_passed}")
    
    if result.out_of_bounds:
        for item in result.out_of_bounds:
            print(f"    ❌ {item['text_span']}: {item['reason']}")

if __name__ == "__main__":
    test_validation_utils()