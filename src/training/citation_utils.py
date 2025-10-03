#!/usr/bin/env python3
"""
Canonical Citation Utilities for Employment Act Malaysia Compliance Agent

Provides unified citation regex patterns and validation functions
to ensure consistency across preference pairs, DPO, and PPO components.

Canonical Pattern: EA-YYYY-NNN[L]*[(N)] where:
- YYYY: 4-digit year (e.g., 2022)
- NNN: section number (e.g., 60)
- L: optional letter suffix (e.g., E)
- (N): optional parenthesis number (e.g., (1))

Examples: EA-2022-60E(1), EA-2022-37, EA-2022-13(2)
"""

import re
from typing import Set, List, Optional, Dict, Tuple
from pathlib import Path
import json


class CanonicalCitationValidator:
    """Unified validator for Employment Act section IDs with canonical patterns."""
    
    # Canonical regex pattern matching EA-YYYY-NNN[L]*[(N)]
    CANONICAL_PATTERN = re.compile(r'\b(EA-\d{4}-\d+[A-Z]*(?:\(\d+\))?)\b', re.IGNORECASE)
    
    # Simplified pattern for extracting numeric/letter parts
    SECTION_PARTS_PATTERN = re.compile(r'EA-(\d{4})-(\d+)([A-Z]*)(?:\((\d+)\))?', re.IGNORECASE)
    
    def __init__(self, valid_sections: Optional[Set[str]] = None):
        """Initialize validator with optional known valid sections."""
        self.valid_sections = valid_sections or set()
        
        # If no valid sections provided, initialize with common ones
        if not self.valid_sections:
            self.valid_sections = self._get_default_valid_sections()
    
    @classmethod
    def _get_default_valid_sections(cls) -> Set[str]:
        """Get default set of valid Employment Act sections."""
        return {
            # Maternity/pregnancy protection
            "EA-2022-37", "EA-2022-40", "EA-2022-41", "EA-2022-42",
            # Working hours and leave
            "EA-2022-60A", "EA-2022-60B", "EA-2022-60C", "EA-2022-60D", 
            "EA-2022-60E", "EA-2022-60F", "EA-2022-60G",
            "EA-2022-60E(1)", "EA-2022-60F(1)", "EA-2022-60G(1)",
            # Termination
            "EA-2022-13", "EA-2022-14", "EA-2022-20",
            "EA-2022-13(1)", "EA-2022-14(1)", "EA-2022-20(1)",
            # Complaints
            "EA-2022-69", "EA-2022-69(1)"
        }
    
    @classmethod
    def load_valid_sections_from_chunks(cls, chunks_file: Path) -> Set[str]:
        """Load valid sections from chunks file."""
        valid_sections = set()
        
        try:
            with open(chunks_file, 'r', encoding='utf-8') as f:
                for line in f:
                    chunk = json.loads(line.strip())
                    section_id = chunk.get('section_id')
                    if section_id:
                        # Normalize to canonical format
                        normalized = cls.normalize_section_id(section_id)
                        if normalized:
                            valid_sections.add(normalized)
        except Exception as e:
            print(f"Warning: Could not load sections from {chunks_file}: {e}")
            return cls._get_default_valid_sections()
        
        return valid_sections if valid_sections else cls._get_default_valid_sections()
    
    @classmethod
    def normalize_section_id(cls, section_id: str) -> Optional[str]:
        """Normalize section ID to canonical format."""
        if not section_id:
            return None

        s = section_id.strip().upper()

        # 1) If it's already canonical, return as-is
        canonical_match = cls.CANONICAL_PATTERN.fullmatch(s) or cls.CANONICAL_PATTERN.match(s)
        if canonical_match:
            return canonical_match.group(1)

        # 2) Handle legacy forms like "EA-60E" or "EA-60E(1)" → "EA-2022-60E(1)"
        legacy = re.match(r'^EA-(\d+)([A-Z]*)(?:\((\d+)\))?$', s, re.IGNORECASE)
        if legacy:
            number = legacy.group(1)
            letter = legacy.group(2) or ''
            paren = legacy.group(3)
            base = f"EA-2022-{number}{letter}"
            return f"{base}({paren})" if paren else base

        return None
    
    def extract_section_ids(self, text: str) -> Set[str]:
        """Extract all canonical section IDs from text."""
        matches = self.CANONICAL_PATTERN.findall(text.upper())
        return set(matches)
    
    def validate_section_ids(self, section_ids: Set[str]) -> Set[str]:
        """Return only valid section IDs from the given set."""
        return section_ids.intersection(self.valid_sections)
    
    def is_valid_section(self, section_id: str) -> bool:
        """Check if a section ID is valid."""
        normalized = self.normalize_section_id(section_id)
        return normalized in self.valid_sections if normalized else False
    
    def get_section_family(self, section_id: str) -> Optional[str]:
        """Get section family (e.g., EA-2022-60E(1) -> 60E)."""
        match = self.SECTION_PARTS_PATTERN.match(section_id)
        if match:
            number = match.group(2)
            letter = match.group(3)
            return f"{number}{letter}"
        return None
    
    def get_sections_by_family(self, family: str) -> Set[str]:
        """Get all valid sections matching a family (e.g., 60E -> {EA-2022-60E, EA-2022-60E(1)})."""
        matching_sections = set()
        for section in self.valid_sections:
            if self.get_section_family(section) == family:
                matching_sections.add(section)
        return matching_sections
    
    def get_different_valid_section(self, exclude_section: str) -> Optional[str]:
        """Get a different valid section ID (for generating wrong-section negatives).
        
        Prefers sections from different families for harder negatives.
        """
        import random
        
        available_sections = self.valid_sections - {exclude_section}
        if not available_sections:
            return None
        
        # Try to get a section from a different family for harder negatives
        exclude_family = self.get_section_family(exclude_section)
        if exclude_family:
            different_family_sections = []
            for section in available_sections:
                if self.get_section_family(section) != exclude_family:
                    different_family_sections.append(section)
            
            # If we found sections from different families, prefer those
            if different_family_sections:
                return random.choice(different_family_sections)
        
        # Fallback: any available section (same family is OK)
        return random.choice(list(available_sections))
    
    def compute_citation_metrics(self, predicted_sections: Set[str], 
                                gold_sections: Set[str]) -> Tuple[float, float]:
        """Compute citation exact match and IoU."""
        # Normalize all sections
        pred_normalized = {self.normalize_section_id(s) for s in predicted_sections}
        pred_normalized = {s for s in pred_normalized if s}
        
        gold_normalized = {self.normalize_section_id(s) for s in gold_sections}
        gold_normalized = {s for s in gold_normalized if s}
        
        # Only consider valid sections
        pred_valid = self.validate_section_ids(pred_normalized)
        gold_valid = self.validate_section_ids(gold_normalized)
        
        # Exact match: any gold section found
        exact_match = 1.0 if pred_valid.intersection(gold_valid) else 0.0
        
        # IoU: intersection over union
        if pred_valid or gold_valid:
            intersection = len(pred_valid.intersection(gold_valid))
            union = len(pred_valid.union(gold_valid))
            iou = intersection / union if union > 0 else 0.0
        else:
            iou = 0.0
        
        return exact_match, iou
    
    def compute_groundedness_score(self, response: str, gold_sections: Set[str]) -> Dict[str, float]:
        """Compute comprehensive groundedness score."""
        predicted_sections = self.extract_section_ids(response)
        citation_em, citation_iou = self.compute_citation_metrics(predicted_sections, gold_sections)
        
        # Check for legal anchoring terms
        response_lower = response.lower()
        legal_terms = [
            "employment act", "according to section", "under section",
            "the law", "provision", "legislation", "malaysia employment act"
        ]
        legal_anchoring = any(term in response_lower for term in legal_terms)
        
        return {
            "citation_em": citation_em,
            "citation_iou": citation_iou,
            "legal_anchoring": 1.0 if legal_anchoring else 0.0,
            "predicted_sections": list(predicted_sections),
            "valid_predicted": list(self.validate_section_ids(predicted_sections))
        }


class KeywordSectionMapper:
    """Maps keywords to section families for better chunk selection."""
    
    KEYWORD_TO_FAMILY = {
        # Leave entitlements
        "annual leave": ["60E", "60F", "60G"],
        "vacation": ["60E", "60F", "60G"],
        "sick leave": ["60F"],
        "medical leave": ["60F"],
        "maternity": ["37", "40", "41", "42"],
        "pregnancy": ["37", "40", "41", "42"],
        "female employee": ["37", "40", "41", "42"],
        
        # Working hours
        "overtime": ["60A", "13"],
        "working hours": ["60A", "13"],
        "hours of work": ["60A", "13"],
        "rest day": ["60C", "60D"],
        "public holiday": ["60C", "60D"],
        "holiday": ["60C", "60D"],
        
        # Termination
        "termination": ["13", "14", "20"],
        "dismiss": ["13", "14", "20"],
        "dismissal": ["13", "14", "20"],
        "terminate": ["13", "14", "20"],
        "notice": ["13", "14", "20"],
        
        # Complaints
        "complaint": ["69"],
        "file complaint": ["69"],
        "grievance": ["69"],
    }
    
    @classmethod
    def get_relevant_families(cls, prompt: str) -> List[str]:
        """Get relevant section families based on prompt keywords."""
        prompt_lower = prompt.lower()
        relevant_families = []
        
        for keyword, families in cls.KEYWORD_TO_FAMILY.items():
            if keyword in prompt_lower:
                relevant_families.extend(families)
        
        return list(set(relevant_families))  # Remove duplicates


def compute_lexical_f1(text1: str, text2: str) -> float:
    """Compute word-level F1 score between two texts."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 and not words2:
        return 1.0
    if not words1 or not words2:
        return 0.0
    
    intersection = len(words1.intersection(words2))
    precision = intersection / len(words1)
    recall = intersection / len(words2)
    
    if precision + recall == 0:
        return 0.0
    
    return 2 * precision * recall / (precision + recall)


def compute_enhanced_similarity(response: str, target: str, 
                               validator: CanonicalCitationValidator,
                               gold_sections: Set[str]) -> float:
    """
    Compute enhanced similarity with groundedness-aware scoring.
    
    Score = 0.5 × citation EM + 0.3 × citation IoU + 0.2 × lexical F1
    """
    # Get groundedness metrics
    groundedness = validator.compute_groundedness_score(response, gold_sections)
    
    # Compute lexical F1
    lexical_f1 = compute_lexical_f1(response, target)
    
    # Weighted combination
    score = (
        0.5 * groundedness["citation_em"] +
        0.3 * groundedness["citation_iou"] +
        0.2 * lexical_f1
    )
    
    return score
