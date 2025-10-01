#!/usr/bin/env python3
"""
Utilities for stable evaluation subset management.
"""

import hashlib
import json
from typing import Dict, Any, List


def generate_stable_example_id(example: Dict[str, Any]) -> str:
    """
    Generate a stable, deterministic ID for an example.
    
    Uses instruction + primary section ID + source chunk ID to create
    a consistent identifier that persists across dataset generation runs.
    """
    # Extract key components for ID generation
    instruction = example.get('instruction', '').strip()
    citations = example.get('citations', [])
    primary_citation = citations[0] if citations else 'unknown'
    source_chunk_id = example.get('source_chunk_id', 'unknown')
    
    # Create deterministic string for hashing
    id_string = f"{instruction}|{primary_citation}|{source_chunk_id}"
    
    # Generate short hash (first 12 chars of SHA256)
    hash_object = hashlib.sha256(id_string.encode('utf-8'))
    stable_id = hash_object.hexdigest()[:12]
    
    return f"eval_{stable_id}"


def create_stable_eval_subset(examples: List[Dict[str, Any]], max_size: int = 30) -> List[Dict[str, Any]]:
    """
    Create a stable evaluation subset with deterministic IDs.
    
    Args:
        examples: List of examples to select from
        max_size: Maximum number of examples (default 30 for Hour 4)
    
    Returns:
        List of examples with stable_id field added
    """
    # Add stable IDs to all examples
    examples_with_ids = []
    for example in examples:
        example_copy = example.copy()
        example_copy['stable_id'] = generate_stable_example_id(example)
        examples_with_ids.append(example_copy)
    
    # Sort by stable_id for deterministic ordering
    examples_with_ids.sort(key=lambda x: x['stable_id'])
    
    # Take first max_size examples
    stable_subset = examples_with_ids[:max_size]
    
    return stable_subset


def load_stable_eval_subset(eval_subset_path, fallback_examples: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Load stable evaluation subset, with fallback to generating from examples.
    
    Args:
        eval_subset_path: Path to eval_subset.jsonl file
        fallback_examples: Examples to use if eval_subset.jsonl doesn't exist
    
    Returns:
        List of evaluation examples with stable_id field
    """
    try:
        if eval_subset_path.exists():
            # Load existing stable subset
            eval_subset = []
            with open(eval_subset_path, 'r') as f:
                for line in f:
                    example = json.loads(line.strip())
                    # Ensure stable_id exists (backward compatibility)
                    if 'stable_id' not in example:
                        example['stable_id'] = generate_stable_example_id(example)
                    eval_subset.append(example)
            return eval_subset
        elif fallback_examples:
            # Generate stable subset from fallback examples
            return create_stable_eval_subset(fallback_examples)
        else:
            return []
    except Exception as e:
        print(f"⚠️ Error loading stable eval subset: {e}")
        if fallback_examples:
            return create_stable_eval_subset(fallback_examples)
        return []