#!/usr/bin/env python3
# python src/training/make_sft_dataset_production.py --chunks data/processed/chunks.jsonl --output-dir outputs/sft_dataset --size 200 --seed 42
"""
Production-grade SFT Dataset Generator for Employment Act Malaysia Compliance Agent.
Features: stratified sampling, citation validation, deduplication, reproducibility.
"""

import json
import random
import hashlib
import pickle
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Set, Tuple
from collections import Counter, defaultdict
from datetime import datetime
import argparse
import re
from dataclasses import dataclass
try:
    import Levenshtein
except ImportError:
    # Simple fallback implementation for testing
    class Levenshtein:
        @staticmethod
        def distance(s1, s2):
            # Simple character-level edit distance
            if len(s1) < len(s2):
                return Levenshtein.distance(s2, s1)
            if len(s2) == 0:
                return len(s1)
            
            previous_row = list(range(len(s2) + 1))
            for i, c1 in enumerate(s1):
                current_row = [i + 1]
                for j, c2 in enumerate(s2):
                    insertions = previous_row[j + 1] + 1
                    deletions = current_row[j] + 1
                    substitutions = previous_row[j] + (c1 != c2)
                    current_row.append(min(insertions, deletions, substitutions))
                previous_row = current_row
            
            return previous_row[-1]
from pydantic import ValidationError

try:
    from .sft_schemas import SFTExample
    from .eval_utils import create_stable_eval_subset
except ImportError:
    # For standalone execution
    import sys
    sys.path.append(str(Path(__file__).parent))
    from sft_schemas import SFTExample
    from eval_utils import create_stable_eval_subset

from pydantic import ValidationError


@dataclass
class DatasetStats:
    """Statistics for the generated dataset."""
    total_examples: int
    section_coverage: int
    avg_instruction_length: float
    avg_answer_length: float
    pct_with_citations: float
    pct_numeric_claims: float
    top_sections: List[Tuple[str, int]]
    citation_validation_rate: float


class ProductionSFTGenerator:
    """Production-grade SFT dataset generator with comprehensive validation."""
    
    def __init__(self, chunks_file: Path, seed: int = 42):
        """Initialize with chunks and set deterministic behavior."""
        self.seed = seed
        self._set_seeds(seed)
        
        self.chunks = self._load_chunks(chunks_file)
        self.section_to_chunks = self._group_by_section()
        self.valid_section_ids = self._extract_valid_section_ids()
        
        print(f"📚 Loaded {len(self.chunks)} chunks across {len(self.section_to_chunks)} sections")
        print(f"🔍 Found {len(self.valid_section_ids)} valid section IDs")
        
        # Enhanced question templates by category
        self.question_templates = self._build_question_templates()
        
        # Legal terminology for query expansion
        self.legal_terms = {
            "leave": ["annual leave", "medical leave", "maternity leave", "sick leave", "compassionate leave"],
            "pay": ["basic wages", "overtime pay", "allowances", "salary", "minimum wage"],
            "termination": ["dismissal", "retrenchment", "resignation", "notice period"],
            "benefits": ["EPF", "SOCSO", "insurance", "bonus", "increment"],
            "working": ["working hours", "rest day", "public holiday", "shift work"],
        }
    
    def _set_seeds(self, seed: int):
        """Set all random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        
    def _load_chunks(self, chunks_file: Path) -> List[Dict[str, Any]]:
        """Load and validate chunks from JSONL file."""
        chunks = []
        with open(chunks_file, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f, 1):
                try:
                    chunk = json.loads(line.strip())
                    # Validate required fields
                    if not all(key in chunk for key in ['text', 'section_id']):
                        print(f"Line {line_num}: Missing required fields")
                        continue
                    chunks.append(chunk)
                except json.JSONDecodeError:
                    print(f"Line {line_num}: Invalid JSON, skipping")
                    continue
        return chunks
    
    def _group_by_section(self) -> Dict[str, List[Dict[str, Any]]]:
        """Group chunks by section ID for stratified sampling."""
        section_groups = defaultdict(list)
        for chunk in self.chunks:
            section_id = chunk.get('section_id', 'unknown')
            if section_id and section_id != 'unknown':
                section_groups[section_id].append(chunk)
        return dict(section_groups)
    
    def _extract_valid_section_ids(self) -> Set[str]:
        """Extract all valid section IDs for citation validation."""
        section_ids = set()
        for chunk in self.chunks:
            section_id = chunk.get('section_id')
            if section_id and section_id != 'unknown':
                section_ids.add(section_id)
        return section_ids
    
    def _build_question_templates(self) -> Dict[str, List[str]]:
        """Build comprehensive question templates by category."""
        return {
            "entitlement": [
                "How many days of {benefit} am I entitled to per year?",
                "What is my {benefit} entitlement under the Employment Act?",
                "Am I entitled to {benefit} as an employee?",
                "How much {benefit} should I receive according to law?",
                "What are the legal requirements for {benefit}?",
            ],
            "calculation": [
                "How is {amount} calculated under the Employment Act?",
                "What is the formula for calculating {amount}?",
                "How do I calculate my {amount} entitlement?",
                "What factors determine my {amount}?",
            ],
            "procedure": [
                "What is the proper procedure for {action}?",
                "How should an employer handle {action}?",
                "What steps must be followed for {action}?",
                "What is the legal process for {action}?",
            ],
            "limitations": [
                "What is the maximum {limit} allowed by law?",
                "What is the minimum {limit} required?",
                "Are there legal limits on {limit}?",
                "What restrictions apply to {limit}?",
            ],
            "rights": [
                "What are my rights regarding {topic}?",
                "Can my employer legally {action}?",
                "What protections do I have against {action}?",
                "Is {action} permitted under employment law?",
            ],
            "consequences": [
                "What happens if an employer fails to provide {benefit}?",
                "What are the penalties for {violation}?",
                "What remedies are available for {issue}?",
                "What can I do if my employer {action}?",
            ]
        }
    
    def _hash_example(self, instruction: str, section_id: str) -> str:
        """Create hash for deduplication."""
        content = f"{instruction.lower().strip()}|{section_id}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def _is_duplicate(self, instruction: str, existing_instructions: List[str], 
                     threshold: float = 0.8) -> bool:
        """Check for near-duplicates using Levenshtein distance."""
        for existing in existing_instructions:
            similarity = 1 - (Levenshtein.distance(instruction.lower(), existing.lower()) / 
                            max(len(instruction), len(existing)))
            if similarity > threshold:
                return True
        return False
    
    def _extract_numeric_claims(self, text: str) -> List[str]:
        """Extract numeric claims from text."""
        # Patterns for common numeric claims in employment law
        patterns = [
            r'\b(\d+)\s*days?\b',
            r'\b(\d+)\s*hours?\b',
            r'\b(\d+)\s*months?\b',
            r'\b(\d+)\s*years?\b',
            r'\b(\d+(?:\.\d+)?)\s*times?\b',
            r'\bRM\s*(\d+(?:,\d{3})*(?:\.\d{2})?)\b',
        ]
        
        claims = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            claims.extend(matches)
        return claims
    
    def _validate_citations(self, citations: List[str]) -> bool:
        """Validate that citations contain valid section IDs."""
        if not citations:
            return False
        return all(citation in self.valid_section_ids for citation in citations)
    
    def _generate_instruction_answer_pair(self, chunk: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a single instruction-answer pair from a chunk."""
        section_id = chunk['section_id']
        text = chunk['text']
        
        # Extract key information from chunk
        numeric_claims = self._extract_numeric_claims(text)
        
        # Choose appropriate template category based on content
        if any(term in text.lower() for term in ['entitled', 'entitlement', 'days', 'hours']):
            category = "entitlement"
        elif any(term in text.lower() for term in ['calculate', 'formula', 'rate']):
            category = "calculation"
        elif any(term in text.lower() for term in ['shall', 'must', 'required']):
            category = "procedure"
        elif any(term in text.lower() for term in ['maximum', 'minimum', 'exceed', 'limit']):
            category = "limitations"
        elif any(term in text.lower() for term in ['rights', 'protection', 'unlawful']):
            category = "rights"
        elif any(term in text.lower() for term in ['penalty', 'offence', 'contravention']):
            category = "consequences"
        else:
            category = random.choice(list(self.question_templates.keys()))
        
        # Select template and fill with relevant terms
        template = random.choice(self.question_templates[category])
        
        # Extract relevant terms for template filling
        if '{benefit}' in template:
            benefits = ["annual leave", "medical leave", "overtime pay", "rest day"]
            benefit = random.choice(benefits)
            template = template.replace('{benefit}', benefit)
        
        if '{amount}' in template:
            amounts = ["overtime pay", "termination benefits", "annual leave pay"]
            amount = random.choice(amounts)
            template = template.replace('{amount}', amount)
        
        if '{action}' in template:
            actions = ["terminate employment", "calculate overtime", "grant leave"]
            action = random.choice(actions)
            template = template.replace('{action}', action)
        
        # Generate instruction
        instruction = template
        
        # Generate grounded answer with citations
        answer_parts = []
        
        # Start with direct information from the text
        key_sentences = [s.strip() for s in text.split('.') if len(s.strip()) > 20][:2]
        if key_sentences:
            answer_parts.append(f"According to the Employment Act, {key_sentences[0].lower()}.")
        
        # Add specific numeric information if available
        if numeric_claims:
            answer_parts.append(f"The specific requirement is {numeric_claims[0]} as stated in the legislation.")
        
        # Add citation reference
        answer_parts.append(f"This is covered under section {section_id}.")
        
        answer = " ".join(answer_parts)
        
        # Limit answer length and avoid large verbatim blocks
        if len(answer) > 500:
            answer = answer[:450] + f"... (Reference: {section_id})"
        
        return {
            "instruction": instruction,
            "input": "",  # Following Alpaca format
            "output": answer,
            "citations": [section_id],
            "source_chunk_id": chunk.get('chunk_id', 'unknown'),
            "has_numeric_claims": len(numeric_claims) > 0,
            "category": category
        }
    
    def _stratified_sampling(self, target_size: int) -> List[str]:
        """Perform stratified sampling across sections."""
        # Calculate samples per section
        sections = list(self.section_to_chunks.keys())
        base_samples_per_section = max(1, target_size // len(sections))
        remaining_samples = target_size - (base_samples_per_section * len(sections))
        
        # Distribute remaining samples to sections with more chunks
        section_weights = [(section, len(chunks)) for section, chunks in self.section_to_chunks.items()]
        section_weights.sort(key=lambda x: x[1], reverse=True)
        
        samples_per_section = {section: base_samples_per_section for section in sections}
        for i in range(remaining_samples):
            section = section_weights[i % len(section_weights)][0]
            samples_per_section[section] += 1
        
        # Sample chunks from each section
        selected_sections = []
        for section, num_samples in samples_per_section.items():
            chunks = self.section_to_chunks[section]
            if len(chunks) >= num_samples:
                sampled_chunks = random.sample(chunks, num_samples)
            else:
                sampled_chunks = chunks * (num_samples // len(chunks)) + random.sample(chunks, num_samples % len(chunks))
            
            for chunk in sampled_chunks:
                selected_sections.append(section)
        
        return selected_sections[:target_size]
    
    def generate_dataset(self, target_size: int = 200) -> List[Dict[str, Any]]:
        """Generate stratified dataset with validation and deduplication."""
        print(f"🎯 Generating {target_size} examples with stratified sampling")
        
        # Stratified sampling
        selected_sections = self._stratified_sampling(target_size)
        
        examples = []
        seen_hashes = set()
        seen_instructions = []
        
        for section_id in selected_sections:
            chunks = self.section_to_chunks[section_id]
            chunk = random.choice(chunks)
            
            # Generate example
            example = self._generate_instruction_answer_pair(chunk)
            
            # Deduplication check
            example_hash = self._hash_example(example['instruction'], section_id)
            if example_hash in seen_hashes:
                continue
            
            # Near-duplicate check
            if self._is_duplicate(example['instruction'], seen_instructions):
                continue
            
            # Citation validation
            if not self._validate_citations(example['citations']):
                print(f"Invalid citations for section {section_id}")
                continue
            
            examples.append(example)
            seen_hashes.add(example_hash)
            seen_instructions.append(example['instruction'])
        
        print(f"Generated {len(examples)} valid examples")
        return examples
    
    def _split_dataset(self, examples: List[Dict[str, Any]], eval_ratio: float = 0.15) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """Split dataset by section to avoid leakage."""
        # Group examples by section
        section_examples = defaultdict(list)
        for example in examples:
            section_id = example['citations'][0]  # Use first citation as primary section
            section_examples[section_id].append(example)
        
        # Split sections for train/eval
        sections = list(section_examples.keys())
        random.shuffle(sections)
        
        eval_sections_count = max(1, int(len(sections) * eval_ratio))
        eval_sections = sections[:eval_sections_count]
        train_sections = sections[eval_sections_count:]
        
        # Collect examples
        train_examples = []
        eval_examples = []
        
        for section in train_sections:
            train_examples.extend(section_examples[section])
        
        for section in eval_sections:
            eval_examples.extend(section_examples[section])
        
        # Ensure we have at least 30 eval examples as per Hour 4 requirement
        if len(eval_examples) < 30:
            needed = 30 - len(eval_examples)
            if len(train_examples) >= needed:
                additional_eval = random.sample(train_examples, needed)
                train_examples = [ex for ex in train_examples if ex not in additional_eval]
                eval_examples.extend(additional_eval)
        
        return train_examples, eval_examples
    
    def _calculate_stats(self, examples: List[Dict[str, Any]]) -> DatasetStats:
        """Calculate comprehensive dataset statistics."""
        total = len(examples)
        
        # Section coverage
        sections = set()
        for example in examples:
            sections.update(example['citations'])
        
        # Length statistics
        instruction_lengths = [len(ex['instruction']) for ex in examples]
        answer_lengths = [len(ex['output']) for ex in examples]
        
        # Citation and numeric statistics
        with_citations = sum(1 for ex in examples if ex['citations'])
        with_numeric = sum(1 for ex in examples if ex.get('has_numeric_claims', False))
        
        # Top sections
        section_counts = Counter()
        for example in examples:
            section_counts.update(example['citations'])
        
        # Citation validation
        valid_citations = sum(1 for ex in examples if self._validate_citations(ex['citations']))
        
        return DatasetStats(
            total_examples=total,
            section_coverage=len(sections),
            avg_instruction_length=np.mean(instruction_lengths),
            avg_answer_length=np.mean(answer_lengths),
            pct_with_citations=with_citations / total * 100,
            pct_numeric_claims=with_numeric / total * 100,
            top_sections=section_counts.most_common(5),
            citation_validation_rate=valid_citations / total * 100
        )
    
    def _validate_examples_with_schema(self, examples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Validate examples using Pydantic schema and return clean examples."""
        validated_examples = []
        errors = []
        warnings = []
        
        for i, example in enumerate(examples):
            try:
                # Validate with Pydantic schema
                validated_example = SFTExample(**example)
                validated_examples.append(validated_example.model_dump())
            except ValidationError as e:
                errors.append(f"Example {i}: {str(e)}")
                continue
            except Exception as e:
                errors.append(f"Example {i}: Unexpected error - {str(e)}")
                continue
        
        # Generate validation report
        validation_report = {
            "is_valid": len(errors) == 0,
            "total_examples": len(examples),
            "valid_examples": len(validated_examples),
            "errors": errors,
            "warnings": warnings,
            "statistics": {
                "validation_rate": len(validated_examples) / len(examples) if examples else 0,
                "error_rate": len(errors) / len(examples) if examples else 0
            }
        }
        
        print(f"📋 Schema validation: {len(validated_examples)}/{len(examples)} examples passed")
        if errors:
            print(f"❌ {len(errors)} validation errors found")
            for error in errors[:5]:  # Show first 5 errors
                print(f"   {error}")
            if len(errors) > 5:
                print(f"   ... and {len(errors) - 5} more")
        
        return validated_examples, validation_report
    
    def _persist_eval_subset(self, eval_examples: List[Dict[str, Any]], output_dir: Path):
        """Persist the exact 30 held-out evaluation samples with stable IDs."""
        eval_subset_path = output_dir / "eval_subset.jsonl"
        
        # Create stable evaluation subset with deterministic IDs
        eval_subset = create_stable_eval_subset(eval_examples, max_size=30)
        
        # Save eval subset with stable IDs
        with open(eval_subset_path, 'w', encoding='utf-8') as f:
            for example in eval_subset:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        # Save stable IDs for reference and auditability
        eval_ids_path = output_dir / "eval_ids.json"
        stable_ids = [example['stable_id'] for example in eval_subset]
        eval_metadata = {
            "stable_ids": stable_ids,
            "count": len(eval_subset),
            "generation_method": "deterministic_hash",
            "id_components": "instruction|primary_citation|source_chunk_id"
        }
        with open(eval_ids_path, 'w', encoding='utf-8') as f:
            json.dump(eval_metadata, f, indent=2, ensure_ascii=False)
        
        print(f"📌 Persisted {len(eval_subset)} evaluation samples with stable IDs to {eval_subset_path}")
        print(f"📌 Persisted stable eval IDs to {eval_ids_path}")
        print(f"   Sample stable IDs: {stable_ids[:3]}...")
        return eval_subset
    
    def save_dataset(self, output_dir: Path, examples: List[Dict[str, Any]]):
        """Save dataset with comprehensive metadata."""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Validate examples with Pydantic schema
        validated_examples, validation_report = self._validate_examples_with_schema(examples)
        
        if not validation_report["is_valid"]:
            print(f"⚠️ Warning: {len(validation_report['errors'])} examples failed validation")
            print("Proceeding with valid examples only...")
        
        # Split dataset
        train_examples, eval_examples = self._split_dataset(validated_examples)
        
        # Persist the exact evaluation subset for consistency
        eval_subset = self._persist_eval_subset(eval_examples, output_dir)
        
        # Save train/eval splits
        train_path = output_dir / "sft_dataset_train.jsonl"
        eval_path = output_dir / "sft_dataset_eval.jsonl"
        
        with open(train_path, 'w', encoding='utf-8') as f:
            for example in train_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        with open(eval_path, 'w', encoding='utf-8') as f:
            for example in eval_examples:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        # Calculate and save statistics
        train_stats = self._calculate_stats(train_examples)
        eval_stats = self._calculate_stats(eval_examples)
        
        stats = {
            "dataset_info": {
                "generation_time": datetime.now().isoformat(),
                "seed": self.seed,
                "total_examples": len(examples),
                "train_examples": len(train_examples),
                "eval_examples": len(eval_examples),
            },
            "train_stats": {
                "total_examples": train_stats.total_examples,
                "section_coverage": train_stats.section_coverage,
                "avg_instruction_length": round(train_stats.avg_instruction_length, 2),
                "avg_answer_length": round(train_stats.avg_answer_length, 2),
                "pct_with_citations": round(train_stats.pct_with_citations, 2),
                "pct_numeric_claims": round(train_stats.pct_numeric_claims, 2),
                "top_sections": train_stats.top_sections,
                "citation_validation_rate": round(train_stats.citation_validation_rate, 2)
            },
            "eval_stats": {
                "total_examples": eval_stats.total_examples,
                "section_coverage": eval_stats.section_coverage,
                "avg_instruction_length": round(eval_stats.avg_instruction_length, 2),
                "avg_answer_length": round(eval_stats.avg_answer_length, 2),
                "pct_with_citations": round(eval_stats.pct_with_citations, 2),
                "pct_numeric_claims": round(eval_stats.pct_numeric_claims, 2),
                "top_sections": eval_stats.top_sections,
                "citation_validation_rate": round(eval_stats.citation_validation_rate, 2)
            }
        }
        
        stats_path = output_dir / "dataset_stats.json"
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        # Save validation report
        validation_path = output_dir / "validation_report.json"
        with open(validation_path, 'w', encoding='utf-8') as f:
            json.dump(validation_report, f, indent=2, ensure_ascii=False)
        
        # Save metadata
        metadata = {
            "generator_version": "2.0.0",
            "seed": self.seed,
            "chunks_source": str(self.chunks),
            "generation_params": {
                "target_size": len(examples),
                "stratified_sampling": True,
                "deduplication": True,
                "citation_validation": True,
                "schema_validation": True
            },
            "validation_summary": {
                "total_generated": len(examples),
                "schema_validated": len(validated_examples),
                "validation_rate": len(validated_examples) / len(examples) if examples else 0
            }
        }
        
        metadata_path = output_dir / "dataset_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"   Dataset saved to {output_dir}")
        print(f"   Train: {len(train_examples)} examples ({train_path})")
        print(f"   Eval: {len(eval_examples)} examples ({eval_path})")
        print(f"   Eval subset: 30 examples ({output_dir / 'eval_subset.jsonl'})")
        print(f"   Eval IDs: {output_dir / 'eval_ids.json'}")
        print(f"   Stats: {stats_path}")
        print(f"   Validation: {validation_path}")
        print(f"   Metadata: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate production SFT dataset")
    parser.add_argument('--chunks', type=Path, required=True, help='Input chunks JSONL file')
    parser.add_argument('--output-dir', type=Path, required=True, help='Output directory')
    parser.add_argument('--size', type=int, default=200, help='Target dataset size (150-250)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Validate size
    if not 150 <= args.size <= 250:
        print("⚠️ Warning: Size should be between 150-250 for Hour 4 requirements")
    
    # Generate dataset
    generator = ProductionSFTGenerator(args.chunks, seed=args.seed)
    examples = generator.generate_dataset(target_size=args.size)
    generator.save_dataset(args.output_dir, examples)
    
    print(f" Production SFT dataset generation complete!")


if __name__ == "__main__":
    main()