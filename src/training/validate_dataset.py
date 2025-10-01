#!/usr/bin/env python3
# python src/training/validate_dataset.py --train-data outputs/sft_dataset_train.jsonl --eval-data outputs/sft_dataset_eval.jsonl
"""
Dataset Validator for SFT datasets.
Performs comprehensive validation of dataset quality before training.
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Any, Set, Optional, Tuple
from collections import Counter
import argparse
from dataclasses import dataclass
from pydantic import ValidationError
import sys

try:
    from .sft_schemas import SFTExample
    from .citation_utils import CanonicalCitationValidator
except ImportError:
    # For standalone execution
    sys.path.append(str(Path(__file__).parent))
    from sft_schemas import SFTExample
    from citation_utils import CanonicalCitationValidator


@dataclass
class ValidationResult:
    """Results from dataset validation."""
    is_valid: bool
    errors: List[str]
    warnings: List[str]
    stats: Dict[str, Any]


class SFTDatasetValidator:
    """Validates SFT datasets for quality and consistency."""
    
    def __init__(self, valid_section_ids: Optional[Set[str]] = None):
        self.required_fields = ['instruction', 'input', 'output', 'citations']
        self.valid_section_ids = valid_section_ids or set()
        # Use canonical citation validator for consistent pattern matching
        self.citation_validator = CanonicalCitationValidator(self.valid_section_ids)
    
    @classmethod
    def from_chunks_file(cls, chunks_path: Path) -> 'SFTDatasetValidator':
        """Create validator with section IDs from chunks file."""
        valid_section_ids = set()
        
        if chunks_path.exists():
            with open(chunks_path, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        chunk = json.loads(line.strip())
                        section_id = chunk.get('section_id')
                        if section_id and section_id != 'unknown':
                            valid_section_ids.add(section_id)
                    except json.JSONDecodeError:
                        continue
        
        print(f"📚 Loaded {len(valid_section_ids)} valid section IDs from chunks")
        return cls(valid_section_ids)
    
    def validate_with_schema(self, examples: List[Dict[str, Any]]) -> Tuple[int, List[str]]:
        """Validate examples using Pydantic schema."""
        valid_count = 0
        schema_errors = []
        
        for i, example in enumerate(examples):
            try:
                SFTExample(**example)
                valid_count += 1
            except ValidationError as e:
                schema_errors.append(f"Example {i}: Schema validation failed - {str(e)}")
            except Exception as e:
                schema_errors.append(f"Example {i}: Unexpected validation error - {str(e)}")
        
        return valid_count, schema_errors
    
    def validate_citations_against_chunks(self, examples: List[Dict[str, Any]]) -> Tuple[int, List[str]]:
        """Validate that all citations exist in the known section universe."""
        if not self.valid_section_ids:
            return len(examples), []  # Skip if no section IDs available
        
        valid_count = 0
        citation_errors = []
        
        for i, example in enumerate(examples):
            citations = example.get('citations', [])
            invalid_citations = []
            
            for citation in citations:
                if citation not in self.valid_section_ids:
                    invalid_citations.append(citation)
            
            if invalid_citations:
                citation_errors.append(
                    f"Example {i}: Invalid citations not found in chunks: {invalid_citations}"
                )
            else:
                valid_count += 1
        
        return valid_count, citation_errors
    
    def validate_citation_presence_in_output(self, examples: List[Dict[str, Any]]) -> Tuple[int, List[str], float]:
        """Validate that gold citations are actually present in the output text.
        
        Uses canonical citation normalization to handle legacy formats.
        
        Returns:
            Tuple of (examples_with_any_citations_present, presence_warnings, avg_citation_presence_rate)
        """
        citations_present_count = 0
        presence_warnings = []
        total_citations = 0
        present_citations_total = 0
        
        for i, example in enumerate(examples):
            output = example.get('output', '')
            citations = example.get('citations', [])
            
            if not citations:
                continue
                
            present_citations = []
            missing_citations = []
            
            # Extract all citation IDs found in the output text using canonical validator
            output_citations = self.citation_validator.extract_section_ids(output)
            
            for citation in citations:
                total_citations += 1
                
                # Normalize the gold citation to canonical format
                normalized_citation = self.citation_validator.normalize_section_id(citation)
                
                # Check if any form of this citation is present in output
                citation_found = False
                
                # Direct string match (current behavior)
                if citation in output:
                    citation_found = True
                # Normalized match (improved behavior for legacy formats)
                elif normalized_citation and normalized_citation in output_citations:
                    citation_found = True
                
                if citation_found:
                    present_citations.append(citation)
                    present_citations_total += 1
                else:
                    missing_citations.append(citation)
            
            if present_citations:  # At least one citation is present
                citations_present_count += 1
            
            if missing_citations:  # Some citations are missing from output
                presence_warnings.append(
                    f"Example {i}: Citations not found in output text: {missing_citations}"
                )
        
        # Calculate percentage of individual citations present
        avg_citation_presence_rate = (present_citations_total / total_citations * 100) if total_citations > 0 else 0.0
        
        return citations_present_count, presence_warnings, avg_citation_presence_rate
    
    def validate_dataset(self, dataset_path: Path) -> ValidationResult:
        """Validate a single dataset file."""
        errors = []
        warnings = []
        
        # Load dataset
        try:
            examples = self._load_jsonl(dataset_path)
        except Exception as e:
            return ValidationResult(
                is_valid=False,
                errors=[f"Failed to load dataset: {e}"],
                warnings=[],
                stats={}
            )
        
        if not examples:
            return ValidationResult(
                is_valid=False,
                errors=["Dataset is empty"],
                warnings=[],
                stats={}
            )
        
        # Validate each example
        valid_examples = 0
        field_errors = Counter()
        citation_error_count = 0
        length_issues = 0
        
        all_citations = set()
        instruction_lengths = []
        output_lengths = []
        
        for i, example in enumerate(examples):
            example_errors = []
            
            # Check required fields
            for field in self.required_fields:
                if field not in example:
                    example_errors.append(f"Missing field: {field}")
                    field_errors[field] += 1
                elif not example[field] and field != 'input':  # input can be empty
                    example_errors.append(f"Empty field: {field}")
                    field_errors[f"empty_{field}"] += 1
            
            if example_errors:
                errors.extend([f"Example {i}: {err}" for err in example_errors])
                continue
            
            # Validate citations
            citations = example.get('citations', [])
            if not citations:
                citation_error_count += 1
            else:
                all_citations.update(citations)
                # Check citation format and existence using canonical validator
                for citation in citations:
                    normalized = self.citation_validator.normalize_section_id(citation)
                    if not normalized:
                        warnings.append(f"Example {i}: Invalid citation format: {citation}")
                    # Check if citation exists in known section IDs
                    elif self.valid_section_ids and not self.citation_validator.is_valid_section(citation):
                        errors.append(f"Example {i}: Citation {citation} not found in chunks")
            
            # Check lengths
            instruction = example.get('instruction', '')
            output = example.get('output', '')
            
            instruction_lengths.append(len(instruction))
            output_lengths.append(len(output))
            
            if len(instruction) < 10:
                length_issues += 1
                warnings.append(f"Example {i}: Very short instruction ({len(instruction)} chars)")
            
            if len(output) < 20:
                length_issues += 1
                warnings.append(f"Example {i}: Very short output ({len(output)} chars)")
            
            if len(output) > 1000:
                warnings.append(f"Example {i}: Very long output ({len(output)} chars)")
            
            valid_examples += 1
        
        # Initialize counts
        schema_valid_count = 0
        citation_valid_count = 0
        citation_present_count = 0
        avg_citation_presence_rate = 0.0
        
        # Schema validation (only if examples were processed)
        if examples:
            schema_valid_count, schema_errors = self.validate_with_schema(examples)
            errors.extend(schema_errors)
            
            # Citation validation against chunks
            citation_valid_count, citation_errors = self.validate_citations_against_chunks(examples)
            errors.extend(citation_errors)
            
            # Citation presence validation in output text
            citation_present_count, presence_warnings, avg_citation_presence_rate = self.validate_citation_presence_in_output(examples)
            warnings.extend(presence_warnings)
        
        # Calculate statistics
        stats = {
            "total_examples": len(examples),
            "valid_examples": valid_examples,
            "schema_valid_examples": schema_valid_count,
            "citation_valid_examples": citation_valid_count,
            "citation_present_in_output": citation_present_count if examples else 0,
            "citation_presence_rate": citation_present_count / len(examples) * 100 if examples else 0,
            "individual_citation_presence_rate": avg_citation_presence_rate if examples else 0,
            "unique_citations": len(all_citations),
            "citation_coverage": list(all_citations)[:10],  # Show first 10
            "avg_instruction_length": sum(instruction_lengths) / len(instruction_lengths) if instruction_lengths else 0,
            "avg_output_length": sum(output_lengths) / len(output_lengths) if output_lengths else 0,
            "examples_without_citations": citation_error_count,
            "examples_with_length_issues": length_issues,
            "known_section_ids_count": len(self.valid_section_ids),
        }
        
        # Overall validation
        is_valid = (len(errors) == 0 and valid_examples > 0 and 
                   schema_valid_count == len(examples) and 
                   citation_valid_count == len(examples))
        
        if citation_error_count > len(examples) * 0.1:  # More than 10% without citations
            warnings.append(f"High percentage of examples without citations: {citation_error_count}/{len(examples)}")
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings,
            stats=stats
        )
    
    def _load_jsonl(self, path: Path) -> List[Dict[str, Any]]:
        """Load JSONL file."""
        examples = []
        with open(path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    example = json.loads(line.strip())
                    examples.append(example)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON at line {line_num}: {e}")
        return examples
    
    def validate_train_eval_split(self, train_path: Path, eval_path: Path) -> ValidationResult:
        """Validate train/eval split for consistency."""
        errors = []
        warnings = []
        
        # Validate individual datasets
        train_result = self.validate_dataset(train_path)
        eval_result = self.validate_dataset(eval_path)
        
        if not train_result.is_valid:
            errors.extend([f"Training set: {err}" for err in train_result.errors])
        
        if not eval_result.is_valid:
            errors.extend([f"Evaluation set: {err}" for err in eval_result.errors])
        
        if errors:
            return ValidationResult(
                is_valid=False,
                errors=errors,
                warnings=warnings,
                stats={}
            )
        
        # Check split ratios
        train_size = train_result.stats['total_examples']
        eval_size = eval_result.stats['total_examples']
        total_size = train_size + eval_size
        
        eval_ratio = eval_size / total_size
        if eval_ratio < 0.1:
            warnings.append(f"Evaluation set is very small: {eval_ratio:.1%} of total")
        elif eval_ratio > 0.3:
            warnings.append(f"Evaluation set is large: {eval_ratio:.1%} of total")
        
        # Check for potential data leakage (same instructions and sections)
        train_examples = self._load_jsonl(train_path)
        eval_examples = self._load_jsonl(eval_path)
        
        train_instructions = set(ex['instruction'] for ex in train_examples)
        eval_instructions = set(ex['instruction'] for ex in eval_examples)
        
        instruction_overlap = train_instructions & eval_instructions
        if instruction_overlap:
            errors.append(f"Data leakage detected: {len(instruction_overlap)} identical instructions in train and eval")
        
        # Check section-level leakage (should be split by section)
        train_sections = set()
        eval_sections = set()
        
        for ex in train_examples:
            train_sections.update(ex.get('citations', []))
        for ex in eval_examples:
            eval_sections.update(ex.get('citations', []))
        
        section_overlap = train_sections & eval_sections
        if section_overlap:
            # This is an error for Hour 4 requirements, not just a warning
            errors.append(f"Section-level leakage detected: {len(section_overlap)} sections appear in both train and eval splits: {list(section_overlap)[:5]}")
        
        # Combined statistics
        combined_stats = {
            "train_examples": train_size,
            "eval_examples": eval_size,
            "total_examples": total_size,
            "eval_ratio": eval_ratio,
            "train_citations": train_result.stats['unique_citations'],
            "eval_citations": eval_result.stats['unique_citations'],
            "train_sections": len(train_sections),
            "eval_sections": len(eval_sections),
            "instruction_overlap": len(instruction_overlap),
            "section_overlap": len(section_overlap),
            "section_overlap_list": list(section_overlap)[:10] if section_overlap else [],
            "train_schema_valid": train_result.stats.get('schema_valid_examples', 0),
            "eval_schema_valid": eval_result.stats.get('schema_valid_examples', 0),
            "train_citation_valid": train_result.stats.get('citation_valid_examples', 0),
            "eval_citation_valid": eval_result.stats.get('citation_valid_examples', 0),
            "train_citation_present_rate": train_result.stats.get('citation_presence_rate', 0),
            "eval_citation_present_rate": eval_result.stats.get('citation_presence_rate', 0),
        }
        
        is_valid = len(errors) == 0
        
        return ValidationResult(
            is_valid=is_valid,
            errors=errors,
            warnings=warnings + train_result.warnings + eval_result.warnings,
            stats=combined_stats
        )


def main():
    parser = argparse.ArgumentParser(description="Validate SFT dataset")
    parser.add_argument('--train-data', type=Path, required=True, help='Training JSONL file')
    parser.add_argument('--eval-data', type=Path, required=True, help='Evaluation JSONL file')
    parser.add_argument('--chunks', type=Path, help='Chunks file to validate citations against')
    parser.add_argument('--output', type=Path, help='Output validation report')
    
    args = parser.parse_args()
    
    # Create validator with section IDs if chunks provided
    if args.chunks and args.chunks.exists():
        validator = SFTDatasetValidator.from_chunks_file(args.chunks)
    else:
        validator = SFTDatasetValidator()
        if args.chunks:
            print(f"⚠️ Chunks file not found: {args.chunks}")
    
    print("🔍 Validating SFT dataset...")
    result = validator.validate_train_eval_split(args.train_data, args.eval_data)
    
    # Print results
    print(f"\n📊 Validation Results")
    print(f"{'='*50}")
    print(f"Status: {'✅ VALID' if result.is_valid else '❌ INVALID'}")
    
    print(f"\n📈 Statistics:")
    for key, value in result.stats.items():
        print(f"  {key}: {value}")
    
    if result.errors:
        print(f"\n❌ Errors ({len(result.errors)}):")
        for error in result.errors:
            print(f"  - {error}")
    
    if result.warnings:
        print(f"\n⚠️  Warnings ({len(result.warnings)}):")
        for warning in result.warnings:
            print(f"  - {warning}")
    
    # Save report if requested
    if args.output:
        report = {
            "validation_result": result.is_valid,
            "errors": result.errors,
            "warnings": result.warnings,
            "statistics": result.stats,
        }
        
        with open(args.output, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📁 Report saved to: {args.output}")
    
    # Exit with error code if validation failed
    if not result.is_valid:
        exit(1)
    else:
        print("\n✅ Dataset validation passed!")


if __name__ == "__main__":
    main()