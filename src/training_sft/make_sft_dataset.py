#!/usr/bin/env python3
# python3 src/training/make_sft_dataset.py --chunks data/chunks.jsonl --output data/sft_dataset.jsonl --size 200 --seed 42
"""
SFT Dataset Generator for Employment Act Malaysia Compliance Agent
Synthesizes 150-250 instruction-answer pairs grounded in the text chunks.
Each example stores citations: [section_ids].
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any
import argparse
from datetime import datetime

class SFTDatasetGenerator:
    def __init__(self, chunks_file: Path):
        """Initialize with chunks from the Employment Act."""
        self.chunks = self._load_chunks(chunks_file)
        self.section_to_chunks = self._group_by_section()
        
        # Question templates for different types of queries
        self.question_templates = {
            "entitlement": [
                "How many {benefit} am I entitled to?",
                "What is my {benefit} entitlement?",
                "Am I entitled to {benefit}?",
                "How much {benefit} should I receive?",
                "What are the rules for {benefit}?",
            ],
            "procedure": [
                "How do I {action}?",
                "What is the process for {action}?",
                "What steps should I take to {action}?",
                "How can I {action}?",
            ],
            "limitation": [
                "What is the maximum {limit}?",
                "What is the minimum {limit}?",
                "Are there limits on {limit}?",
                "What are the restrictions on {limit}?",
            ],
            "rights": [
                "What are my rights regarding {topic}?",
                "Can my employer {action}?",
                "Is it legal for my employer to {action}?",
                "What protections do I have against {action}?",
            ],
            "consequences": [
                "What happens if I {action}?",
                "What are the consequences of {action}?",
                "What penalties apply if {action}?",
                "Can I be {consequence} for {action}?",
            ]
        }
        
        # Topic-specific variables for templates
        self.template_variables = {
            "benefits": ["annual leave", "sick leave", "maternity leave", "paternity leave", 
                        "overtime pay", "public holidays", "rest days"],
            "actions": ["resign without notice", "file a complaint", "work overtime", 
                       "take emergency leave", "negotiate salary"],
            "limits": ["working hours", "probationary period", "notice period", 
                      "retirement age", "overtime hours"],
            "topics": ["pregnancy", "termination", "salary deductions", "working conditions", 
                      "female employee rights"],
            "employer_actions": ["terminate me during pregnancy", "deduct from my salary", 
                               "make me work on rest days", "refuse annual leave"],
            "consequences": ["terminated", "penalized", "sued", "fined"],
            "employee_actions": ["resign without notice", "refuse overtime", "take sick leave", 
                               "file a complaint", "work part-time"]
        }
    
    def _load_chunks(self, chunks_file: Path) -> List[Dict]:
        """Load text chunks from JSONL file."""
        chunks = []
        with open(chunks_file, 'r', encoding='utf-8') as f:
            for line in f:
                chunks.append(json.loads(line.strip()))
        return chunks
    
    def _group_by_section(self) -> Dict[str, List[Dict]]:
        """Group chunks by section ID."""
        section_groups = {}
        for chunk in self.chunks:
            section_id = chunk.get('section_id')
            if section_id:
                if section_id not in section_groups:
                    section_groups[section_id] = []
                section_groups[section_id].append(chunk)
        return section_groups
    
    def _extract_key_info(self, chunk: Dict) -> Dict[str, Any]:
        """Extract key information from a chunk for answer generation."""
        text = chunk.get('original_text', chunk.get('text', ''))
        section_id = chunk.get('section_id')
        
        # Identify key concepts in the text
        key_concepts = {
            'numbers': self._extract_numbers(text),
            'rights': self._extract_rights(text),
            'procedures': self._extract_procedures(text),
            'restrictions': self._extract_restrictions(text),
            'definitions': self._extract_definitions(text)
        }
        
        return {
            'section_id': section_id,
            'text': text[:500],  # Limit text length
            'concepts': key_concepts
        }
    
    def _extract_numbers(self, text: str) -> List[str]:
        """Extract numerical information (days, hours, percentages)."""
        import re
        patterns = [
            r'\b(\d+)\s*days?\b',
            r'\b(\d+)\s*hours?\b', 
            r'\b(\d+)\s*months?\b',
            r'\b(\d+)\s*years?\b',
            r'\b(\d+)\s*percent\b',
            r'\b(\d+)%\b'
        ]
        numbers = []
        for pattern in patterns:
            matches = re.findall(pattern, text.lower())
            numbers.extend(matches)
        return numbers[:3]  # Limit to first 3
    
    def _extract_rights(self, text: str) -> List[str]:
        """Extract rights and entitlements."""
        rights_keywords = ['entitled', 'right', 'shall', 'may', 'benefit', 'allowance']
        rights = []
        sentences = text.split('.')
        for sentence in sentences[:3]:  # First 3 sentences
            if any(keyword in sentence.lower() for keyword in rights_keywords):
                rights.append(sentence.strip())
        return rights
    
    def _extract_procedures(self, text: str) -> List[str]:
        """Extract procedural information."""
        procedure_keywords = ['shall', 'must', 'apply', 'notify', 'submit', 'process']
        procedures = []
        sentences = text.split('.')
        for sentence in sentences[:3]:
            if any(keyword in sentence.lower() for keyword in procedure_keywords):
                procedures.append(sentence.strip())
        return procedures
    
    def _extract_restrictions(self, text: str) -> List[str]:
        """Extract restrictions and limitations."""
        restriction_keywords = ['not', 'shall not', 'maximum', 'minimum', 'limit', 'except']
        restrictions = []
        sentences = text.split('.')
        for sentence in sentences[:3]:
            if any(keyword in sentence.lower() for keyword in restriction_keywords):
                restrictions.append(sentence.strip())
        return restrictions
    
    def _extract_definitions(self, text: str) -> List[str]:
        """Extract definitions and explanations."""
        definition_keywords = ['means', 'includes', 'refers to', 'defined as']
        definitions = []
        sentences = text.split('.')
        for sentence in sentences[:3]:
            if any(keyword in sentence.lower() for keyword in definition_keywords):
                definitions.append(sentence.strip())
        return definitions
    
    def _generate_question(self, chunk_info: Dict) -> str:
        """Generate a question based on chunk content."""
        concepts = chunk_info['concepts']
        section_id = chunk_info['section_id']
        
        # Choose question type based on available concepts
        if concepts['numbers']:
            # Numerical questions
            if 'day' in chunk_info['text'].lower():
                return random.choice([
                    "How many days of leave am I entitled to?",
                    "What is the duration of this benefit?",
                    f"How many days are specified in section {section_id}?"
                ])
            elif 'hour' in chunk_info['text'].lower():
                return random.choice([
                    "What are the working hour limits?",
                    "How many hours can I work per day?",
                    f"What are the hour restrictions in section {section_id}?"
                ])
        
        if concepts['rights']:
            return random.choice([
                "What are my rights in this situation?",
                "Am I entitled to this benefit?",
                f"What rights does section {section_id} provide?"
            ])
        
        if concepts['procedures']:
            return random.choice([
                "What is the required procedure?",
                "How should I proceed in this case?",
                f"What process is outlined in section {section_id}?"
            ])
        
        if concepts['restrictions']:
            return random.choice([
                "What are the limitations?",
                "Are there restrictions on this?",
                f"What restrictions does section {section_id} impose?"
            ])
        
        # Fallback general questions
        return random.choice([
            f"What does section {section_id} cover?",
            "Can you explain this provision?",
            "What should I know about this topic?"
        ])
    
    def _generate_answer(self, question: str, chunk_info: Dict) -> str:
        """Generate a grounded answer with citations."""
        section_id = chunk_info['section_id']
        text = chunk_info['text']
        concepts = chunk_info['concepts']
        
        # Start with a direct answer based on the content
        answer_parts = []
        
        # Add specific information based on concepts
        if concepts['numbers'] and any(num in text for num in concepts['numbers']):
            if 'day' in question.lower():
                answer_parts.append(f"According to the Employment Act, you are entitled to specific leave days as outlined in the legislation.")
            elif 'hour' in question.lower():
                answer_parts.append(f"The working hour provisions specify clear limits on daily and weekly working time.")
        
        # Add rights information
        if concepts['rights'] and 'right' in question.lower():
            answer_parts.append("You have specific rights under the Employment Act that protect your interests as an employee.")
        
        # Add procedural information
        if concepts['procedures'] and ('how' in question.lower() or 'process' in question.lower()):
            answer_parts.append("The Act outlines specific procedures that must be followed in this situation.")
        
        # Add the key content (first 200 chars of relevant text)
        key_content = text[:200].strip()
        if key_content.endswith('.'):
            answer_parts.append(f"The relevant provision states: {key_content}")
        else:
            answer_parts.append(f"The relevant provision states: {key_content}...")
        
        # Combine answer parts
        if answer_parts:
            answer = " ".join(answer_parts)
        else:
            answer = f"Based on Section {section_id} of the Employment Act: {key_content}"
        
        return answer
    
    def generate_dataset(self, target_size: int = 200) -> List[Dict]:
        """Generate the SFT dataset with instruction-answer pairs."""
        dataset = []
        sections_used = list(self.section_to_chunks.keys())
        
        # Ensure we have enough sections
        if len(sections_used) < target_size // 3:
            print(f"Warning: Only {len(sections_used)} sections available for {target_size} examples")
        
        while len(dataset) < target_size and sections_used:
            # Select a random section
            section_id = random.choice(sections_used)
            chunks = self.section_to_chunks[section_id]
            
            # Select a random chunk from this section
            chunk = random.choice(chunks)
            chunk_info = self._extract_key_info(chunk)
            
            # Generate question and answer
            question = self._generate_question(chunk_info)
            answer = self._generate_answer(question, chunk_info)
            
            # Create the training example
            example = {
                "instruction": question,
                "input": "",  # Empty for single-turn conversations
                "output": answer,
                "citations": [section_id],  # Store section ID for evaluation
                "chunk_id": chunk.get('chunk_id'),
                "metadata": {
                    "section_id": section_id,
                    "generated_at": datetime.now().isoformat(),
                    "concepts": chunk_info['concepts']
                }
            }
            
            dataset.append(example)
            
            # Avoid too many examples from the same section
            if len([ex for ex in dataset if ex['citations'][0] == section_id]) >= 5:
                sections_used.remove(section_id)
        
        return dataset
    
    def save_dataset(self, dataset: List[Dict], output_file: Path, 
                    train_split: float = 0.8) -> None:
        """Save dataset split into train/eval sets."""
        
        # Shuffle dataset
        random.shuffle(dataset)
        
        # Split into train and eval
        split_idx = int(len(dataset) * train_split)
        train_data = dataset[:split_idx]
        eval_data = dataset[split_idx:]
        
        # Save train set
        train_file = output_file.parent / f"{output_file.stem}_train.jsonl"
        with open(train_file, 'w', encoding='utf-8') as f:
            for example in train_data:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        # Save eval set
        eval_file = output_file.parent / f"{output_file.stem}_eval.jsonl"
        with open(eval_file, 'w', encoding='utf-8') as f:
            for example in eval_data:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        # Save combined dataset
        with open(output_file, 'w', encoding='utf-8') as f:
            for example in dataset:
                f.write(json.dumps(example, ensure_ascii=False) + '\n')
        
        print(f"Dataset saved:")
        print(f"  Total: {len(dataset)} examples -> {output_file}")
        print(f"  Train: {len(train_data)} examples -> {train_file}")
        print(f"  Eval:  {len(eval_data)} examples -> {eval_file}")
        
        # Print dataset statistics
        print(f"\nDataset Statistics:")
        print(f"  Sections covered: {len(set(ex['citations'][0] for ex in dataset))}")
        print(f"  Avg question length: {sum(len(ex['instruction']) for ex in dataset) / len(dataset):.1f} chars")
        print(f"  Avg answer length: {sum(len(ex['output']) for ex in dataset) / len(dataset):.1f} chars")


def main():
    parser = argparse.ArgumentParser(description="Generate SFT dataset for Employment Act compliance")
    parser.add_argument('--chunks', required=True, help='Path to chunks.jsonl file')
    parser.add_argument('--output', required=True, help='Output JSONL file path')
    parser.add_argument('--size', type=int, default=200, 
                       help='Target dataset size (default: 200)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate dataset
    generator = SFTDatasetGenerator(Path(args.chunks))
    dataset = generator.generate_dataset(args.size)
    generator.save_dataset(dataset, output_path)
    
    print(f"\n✅ SFT dataset generation complete!")
    print(f"📊 Generated {len(dataset)} instruction-answer pairs")


if __name__ == "__main__":
    main()