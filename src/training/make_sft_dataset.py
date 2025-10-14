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
    from .citation_utils import CanonicalCitationValidator
except ImportError:
    # For standalone execution
    import sys
    sys.path.append(str(Path(__file__).parent))
    from sft_schemas import SFTExample
    from eval_utils import create_stable_eval_subset
    from citation_utils import CanonicalCitationValidator

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
        # Initialize canonical citation validator using chunks
        self.validator = CanonicalCitationValidator(
            valid_sections=CanonicalCitationValidator.load_valid_sections_from_chunks(chunks_file)
        )
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
            raw_id = chunk.get('section_id', 'unknown')
            canonical_id = self.validator.normalize_section_id(raw_id) if raw_id else None
            if canonical_id:
                section_groups[canonical_id].append(chunk)
        return dict(section_groups)
    
    def _extract_valid_section_ids(self) -> Set[str]:
        """Extract all valid (canonical) section IDs for citation validation."""
        return set(self.validator.valid_sections)
    
    def _build_question_templates(self) -> Dict[str, List[str]]:
        """Build comprehensive question templates by category."""
        return {
            "entitlement": [
                "How many days of {benefit} am I entitled to per year?",
                "What is my {benefit} entitlement under the Employment Act?",
                "Am I entitled to {benefit} as an employee?",
                "How much {benefit} should I receive according to law?",
                "What are the legal requirements for {benefit}?",
                "Does the Employment Act require my employer to provide {benefit}?",
                "If I am part-time, what is my {benefit} entitlement?",
            ],
            "calculation": [
                "How is {amount} calculated under the Employment Act?",
                "What is the formula for calculating {amount}?",
                "How do I calculate my {amount} entitlement?",
                "What factors determine my {amount}?",
                "Is {amount} prorated and how is it computed?",
                "How do public holidays affect {amount}?",
            ],
            "procedure": [
                "What is the proper procedure for {action}?",
                "How should an employer handle {action}?",
                "What steps must be followed for {action}?",
                "What is the legal process for {action}?",
                "Who should I contact to {action}?",
                "What documents are required to {action}?",
            ],
            "limitations": [
                "What is the maximum {limit} allowed by law?",
                "What is the minimum {limit} required?",
                "Are there legal limits on {limit}?",
                "What restrictions apply to {limit}?",
                "Can my employer increase {limit} beyond the legal maximum?",
                "Does the law allow exceptions to the {limit}?",
            ],
            "rights": [
                "What are my rights regarding {topic}?",
                "Can my employer legally {action}?",
                "What protections do I have against {action}?",
                "Is {action} permitted under employment law?",
                "Under what circumstances can my employer {action}?",
                "Do I have the right to refuse {action}?",
            ],
            "consequences": [
                "What happens if an employer fails to provide {benefit}?",
                "What are the penalties for {violation}?",
                "What remedies are available for {issue}?",
                "What can I do if my employer {action}?",
                "Can I claim compensation if my employer {action}?",
                "What enforcement actions apply for {violation}?",
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
        raw_section_id = chunk['section_id']
        section_id = self.validator.normalize_section_id(raw_section_id) or raw_section_id
        text = chunk['text']
        title = chunk.get('title', '') or chunk.get('heading', '') or ''
        
        # Extract key information from chunk
        numeric_claims = self._extract_numeric_claims(text)
        
        # Choose appropriate template category based on content (title + text)
        combined_text = f"{title} {text}".lower()
        if any(term in combined_text for term in ['entitled', 'entitlement', 'days', 'hours', 'leave', 'overtime']):
            category = "entitlement"
        elif any(term in combined_text for term in ['calculate', 'calculation', 'formula', 'rate', 'compute']):
            category = "calculation"
        elif any(term in combined_text for term in ['shall', 'must', 'required', 'procedure', 'steps']):
            category = "procedure"
        elif any(term in combined_text for term in ['maximum', 'minimum', 'exceed', 'limit', 'restriction']):
            category = "limitations"
        elif any(term in combined_text for term in ['rights', 'protection', 'unlawful', 'prohibit', 'may not']):
            category = "rights"
        elif any(term in combined_text for term in ['penalty', 'offence', 'contravention', 'fine']):
            category = "consequences"
        else:
            category = random.choice(list(self.question_templates.keys()))
        
        # Select template and fill with relevant terms
        template = random.choice(self.question_templates[category])
        
        # Extract relevant terms for template filling
        if '{benefit}' in template:
            benefits = [
                "annual leave", "medical leave", "sick leave", "maternity leave",
                "public holiday pay", "rest day", "overtime pay", "termination benefits"
            ]
            benefit = random.choice(benefits)
            template = template.replace('{benefit}', benefit)
        
        if '{amount}' in template:
            amounts = [
                "overtime pay", "termination benefits", "annual leave pay",
                "rest day compensation", "public holiday pay"
            ]
            amount = random.choice(amounts)
            template = template.replace('{amount}', amount)
        
        if '{action}' in template:
            actions = [
                "terminate employment", "calculate overtime", "grant leave",
                "resign without notice", "file a complaint", "reduce working hours"
            ]
            action = random.choice(actions)
            template = template.replace('{action}', action)

        if '{limit}' in template:
            limits = [
                "working hours", "overtime", "salary deductions", "rest days", "probation period"
            ]
            limit = random.choice(limits)
            template = template.replace('{limit}', limit)

        if '{topic}' in template:
            topics = [
                "working hours", "termination", "leave entitlements", "overtime pay",
                "salary deductions", "pregnancy protections"
            ]
            topic = random.choice(topics)
            template = template.replace('{topic}', topic)

        if '{violation}' in template:
            violations = [
                "failing to pay overtime", "unlawful dismissal", "not granting rest day",
                "unauthorised salary deduction"
            ]
            violation = random.choice(violations)
            template = template.replace('{violation}', violation)

        if '{issue}' in template:
            issues = [
                "unpaid wages", "overtime disputes", "wrongful termination", "leave rejection"
            ]
            issue = random.choice(issues)
            template = template.replace('{issue}', issue)
        
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
        
        # Helper to process a batch of sections
        def _ingest_sections(section_ids: List[str],
                             examples: List[Dict[str, Any]],
                             seen_hashes: Set[str],
                             seen_by_section: Dict[str, List[str]]):
            for section_id in section_ids:
                chunks = self.section_to_chunks[section_id]
                chunk = random.choice(chunks)

                # Generate example
                example = self._generate_instruction_answer_pair(chunk)

                # Hash-level deduplication
                example_hash = self._hash_example(example['instruction'], section_id)
                if example_hash in seen_hashes:
                    continue

                # Per-section near-duplicate check with stricter threshold
                section_seen = seen_by_section.setdefault(section_id, [])
                if self._is_duplicate(example['instruction'], section_seen, threshold=0.95):
                    continue

                # Citation validation
                if not self._validate_citations(example['citations']):
                    print(f"Invalid citations for section {section_id}")
                    continue

                examples.append(example)
                seen_hashes.add(example_hash)
                section_seen.append(example['instruction'])

        # First pass
        examples: List[Dict[str, Any]] = []
        seen_hashes: Set[str] = set()
        from collections import defaultdict as _dd
        seen_instructions_by_section: Dict[str, List[str]] = _dd(list)

        selected_sections = self._stratified_sampling(target_size)
        _ingest_sections(selected_sections, examples, seen_hashes, seen_instructions_by_section)

        # Retry rounds if under target
        max_rounds = 5
        rounds = 0
        while len(examples) < target_size and rounds < max_rounds:
            missing = target_size - len(examples)
            # Oversample to offset filters; bound by available sections
            request = min(max(missing * 2, 10), max(10 * len(self.section_to_chunks), missing * 3))
            extra_sections = self._stratified_sampling(request)
            _ingest_sections(extra_sections, examples, seen_hashes, seen_instructions_by_section)
            rounds += 1

        if len(examples) < target_size:
            print(f"⚠️ Could not reach target size {target_size}. Generated {len(examples)} after retries.")
        else:
            print(f"Generated {len(examples)} valid examples")
        
        # Return exactly target_size examples if we exceeded due to oversampling
        return examples[:target_size]
    
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
        
        # Collect examples by current section split
        def collect_from_sections(sec_list: List[str]) -> List[Dict[str, Any]]:
            items: List[Dict[str, Any]] = []
            for s in sec_list:
                items.extend(section_examples[s])
            return items

        train_examples = collect_from_sections(train_sections)
        eval_examples = collect_from_sections(eval_sections)

        # Ensure we have at least 30 eval examples while preserving section isolation
        # If eval is short, move whole sections (prefer sections with more examples first)
        if len(eval_examples) < 30:
            # Sort remaining train sections by descending size
            remaining = sorted(train_sections, key=lambda s: len(section_examples[s]), reverse=True)
            while len(eval_examples) < 30 and remaining:
                move_sec = remaining.pop(0)
                # Move this section from train to eval
                train_sections.remove(move_sec)
                eval_sections.append(move_sec)
                # Rebuild example lists
                train_examples = collect_from_sections(train_sections)
                eval_examples = collect_from_sections(eval_sections)

        return train_examples, eval_examples
    
    def _calculate_stats(self, examples: List[Dict[str, Any]]) -> DatasetStats:
        """Calculate comprehensive dataset statistics."""
        total = len(examples)
        if total == 0:
            # Graceful handling when no valid examples exist
            return DatasetStats(
                total_examples=0,
                section_coverage=0,
                avg_instruction_length=0.0,
                avg_answer_length=0.0,
                pct_with_citations=0.0,
                pct_numeric_claims=0.0,
                top_sections=[],
                citation_validation_rate=0.0,
            )
        
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
            avg_instruction_length=float(np.mean(instruction_lengths)) if instruction_lengths else 0.0,
            avg_answer_length=float(np.mean(answer_lengths)) if answer_lengths else 0.0,
            pct_with_citations=(with_citations / total * 100) if total > 0 else 0.0,
            pct_numeric_claims=(with_numeric / total * 100) if total > 0 else 0.0,
            top_sections=section_counts.most_common(5),
            citation_validation_rate=(valid_citations / total * 100) if total > 0 else 0.0,
        )
    
    def _validate_examples_with_schema(self, examples: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """Validate examples using Pydantic schema and return clean examples."""
        validated_examples = []
        errors = []
        warnings = []

        for i, example in enumerate(examples):
            # Normalize citations to canonical form defensively before schema validation
            try:
                raw_citations = example.get('citations', []) or []
                canonical = []
                for c in raw_citations:
                    norm = self.validator.normalize_section_id(c)
                    if norm:
                        canonical.append(norm)
                # If normalization produced at least one citation, replace
                if canonical:
                    example['citations'] = canonical
            except Exception:
                # Leave as-is; schema may catch invalids
                pass
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
