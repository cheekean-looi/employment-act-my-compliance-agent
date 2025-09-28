#!/usr/bin/env python3
"""
Citation backfill utility for Hour 2 prompt templates.
Ensures responses always have proper citations from retrieved chunks.
"""

from typing import List, Dict, Any, Optional
import json
import re
from .prompt_templates import Citation


def backfill_empty_citations(
    response_text: str, 
    context_chunks: List[Dict[str, Any]], 
    max_citations: int = 3
) -> str:
    """
    Backfill empty citations in JSON response with top retrieved chunks.
    
    Args:
        response_text: JSON response string from model
        context_chunks: Retrieved chunks with section_id and text
        max_citations: Maximum citations to backfill
        
    Returns:
        Enhanced JSON response with backfilled citations
    """
    try:
        # Parse the response
        response_data = json.loads(response_text)
        
        # Check if citations are empty or missing
        citations = response_data.get('citations', [])
        if citations:
            return response_text  # Already has citations
        
        # Backfill from top chunks with valid section_ids
        backfilled_citations = []
        for chunk in context_chunks[:max_citations]:
            section_id = chunk.get('section_id')
            if section_id and section_id != 'N/A':
                # Extract snippet (first 150 chars for readability)
                text = chunk.get('text', '')
                snippet = text[:150] + "..." if len(text) > 150 else text
                
                backfilled_citations.append({
                    "section_id": section_id,
                    "snippet": snippet
                })
        
        # Update response with backfilled citations
        response_data['citations'] = backfilled_citations
        
        # Lower confidence if we had to backfill
        if 'confidence' in response_data and backfilled_citations:
            response_data['confidence'] = max(0.1, response_data['confidence'] - 0.2)
        
        # Add safety flag for backfilled citations
        safety_flags = response_data.get('safety_flags', [])
        if backfilled_citations and 'citations_backfilled' not in safety_flags:
            safety_flags.append('citations_backfilled')
            response_data['safety_flags'] = safety_flags
        
        return json.dumps(response_data, indent=2)
        
    except (json.JSONDecodeError, KeyError) as e:
        # If parsing fails, return original response
        return response_text


def validate_citations_against_context(
    response_text: str,
    context_chunks: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Validate that citations in response match available context.
    
    Args:
        response_text: JSON response string
        context_chunks: Retrieved chunks with metadata
        
    Returns:
        Validation report with valid/invalid citations
    """
    try:
        response_data = json.loads(response_text)
        citations = response_data.get('citations', [])
        
        # Build lookup of available section_ids
        available_sections = set()
        for chunk in context_chunks:
            section_id = chunk.get('section_id')
            if section_id and section_id != 'N/A':
                available_sections.add(section_id)
        
        # Validate each citation
        valid_citations = []
        invalid_citations = []
        
        for citation in citations:
            section_id = citation.get('section_id', '')
            if section_id in available_sections:
                valid_citations.append(citation)
            else:
                invalid_citations.append(citation)
        
        return {
            'valid_citations': valid_citations,
            'invalid_citations': invalid_citations,
            'available_sections': list(available_sections),
            'citation_accuracy': len(valid_citations) / len(citations) if citations else 0.0
        }
        
    except (json.JSONDecodeError, KeyError):
        return {
            'valid_citations': [],
            'invalid_citations': [],
            'available_sections': [],
            'citation_accuracy': 0.0,
            'error': 'Failed to parse response'
        }


def extract_section_ids_from_context(context_chunks: List[Dict[str, Any]]) -> List[str]:
    """
    Extract all valid section IDs from context chunks.
    
    Args:
        context_chunks: Retrieved chunks with metadata
        
    Returns:
        List of unique section IDs
    """
    section_ids = []
    for chunk in context_chunks:
        section_id = chunk.get('section_id')
        if section_id and section_id != 'N/A':
            section_ids.append(section_id)
    
    # Return unique section IDs in order of appearance
    seen = set()
    unique_sections = []
    for section_id in section_ids:
        if section_id not in seen:
            unique_sections.append(section_id)
            seen.add(section_id)
    
    return unique_sections


def format_citation_for_display(citation: Dict[str, Any]) -> str:
    """
    Format citation for user-friendly display.
    
    Args:
        citation: Citation dictionary with section_id and snippet
        
    Returns:
        Formatted citation string
    """
    section_id = citation.get('section_id', 'Unknown')
    snippet = citation.get('snippet', '')
    
    # Truncate snippet if too long
    if len(snippet) > 100:
        snippet = snippet[:97] + "..."
    
    return f"[{section_id}] {snippet}"


def has_valid_section_ids(context_chunks: List[Dict[str, Any]]) -> bool:
    """
    Check if any context chunks have valid section IDs.
    
    Args:
        context_chunks: Retrieved chunks with metadata
        
    Returns:
        True if at least one chunk has a valid section_id
    """
    for chunk in context_chunks:
        section_id = chunk.get('section_id')
        if section_id and section_id != 'N/A':
            return True
    return False