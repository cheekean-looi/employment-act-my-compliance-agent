#!/usr/bin/env python3
# python src/training/make_pref_pairs.py --chunks data/processed/chunks.jsonl --output outputs/sft_dataset.jsonl --size 60 --seed 42
"""
Preference Pairs Generator for Employment Act Malaysia Compliance Agent
Creates ~60 prompts with chosen vs rejected answers for DPO training.

Chosen: grounded, cites correctly
Rejected: subtly hallucinates or omits citations
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any
import argparse
from datetime import datetime

class PreferencePairGenerator:
    def __init__(self, chunks_file: Path, sft_model_path: Path = None):
        """Initialize with chunks and optional SFT model for drafting."""
        self.chunks = self._load_chunks(chunks_file)
        self.section_to_chunks = self._group_by_section()
        self.sft_model_path = sft_model_path
        
        # Employment Act specific prompts for preference generation
        self.prompt_templates = {
            "entitlement": [
                "How many days of {benefit} am I entitled to under the Employment Act?",
                "What is my {benefit} entitlement according to Malaysian employment law?",
                "Am I entitled to {benefit} and how much?",
                "What does the Employment Act say about {benefit}?",
            ],
            "procedure": [
                "How do I {action} according to the Employment Act?",
                "What is the proper procedure for {action}?",
                "What steps must I follow to {action}?",
                "How should I handle {action} under Malaysian employment law?",
            ],
            "rights": [
                "What are my rights regarding {topic} under the Employment Act?",
                "Can my employer {action} legally?",
                "Is it legal for my employer to {action}?",
                "What protections do I have against {action}?",
            ],
            "consequences": [
                "What happens if I {action}?",
                "What are the legal consequences of {action}?",
                "Can I be penalized for {action}?",
                "What penalties apply if {action}?",
            ]
        }
        
        # Template variables for generating diverse prompts
        self.template_variables = {
            "benefits": ["annual leave", "sick leave", "maternity leave", "overtime pay", 
                        "public holiday pay", "rest day compensation", "termination benefits"],
            "actions": ["resign without notice", "work overtime", "take emergency leave", 
                       "file a complaint", "refuse unsafe work", "take maternity leave"],
            "topics": ["pregnancy", "termination", "salary deductions", "working hours", 
                      "overtime work", "female employee rights", "rest days"],
            "employer_actions": ["terminate me during pregnancy", "deduct my salary", 
                               "make me work on rest days", "refuse my leave application", 
                               "dismiss me without notice", "reduce my pay"]
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
    
    def _generate_prompt(self, template_type: str) -> str:
        """Generate a random prompt from templates."""
        templates = self.prompt_templates[template_type]
        template = random.choice(templates)
        
        # Fill in template variables
        if "{benefit}" in template:
            benefit = random.choice(self.template_variables["benefits"])
            return template.format(benefit=benefit)
        elif "{action}" in template:
            action = random.choice(self.template_variables["actions"])
            return template.format(action=action)
        elif "{topic}" in template:
            topic = random.choice(self.template_variables["topics"])
            return template.format(topic=topic)
        elif "{action}" in template and template_type == "rights":
            action = random.choice(self.template_variables["employer_actions"])
            return template.format(action=action)
        else:
            return template
    
    def _get_relevant_chunk(self, prompt: str) -> Dict:
        """Get a relevant chunk for the prompt."""
        # Simple keyword matching to find relevant sections
        prompt_lower = prompt.lower()
        
        # Priority sections for common topics
        if any(word in prompt_lower for word in ["annual leave", "vacation"]):
            target_sections = ["EA-60E", "EA-60F", "EA-60G"]
        elif any(word in prompt_lower for word in ["sick leave", "medical"]):
            target_sections = ["EA-60F"]
        elif any(word in prompt_lower for word in ["maternity", "pregnancy"]):
            target_sections = ["EA-37", "EA-40", "EA-41", "EA-42"]
        elif any(word in prompt_lower for word in ["overtime", "hours"]):
            target_sections = ["EA-60A", "EA-13"]
        elif any(word in prompt_lower for word in ["termination", "dismiss"]):
            target_sections = ["EA-13", "EA-14", "EA-20"]
        elif any(word in prompt_lower for word in ["rest day", "holiday"]):
            target_sections = ["EA-60D", "EA-60C"]
        else:
            # Random section if no specific match
            target_sections = list(self.section_to_chunks.keys())
        
        # Find available sections
        available_sections = [s for s in target_sections if s in self.section_to_chunks]
        if not available_sections:
            available_sections = list(self.section_to_chunks.keys())
        
        section_id = random.choice(available_sections)
        chunks = self.section_to_chunks[section_id]
        return random.choice(chunks)
    
    def _generate_chosen_answer(self, prompt: str, chunk: Dict) -> str:
        """Generate a high-quality chosen answer with proper citations."""
        section_id = chunk.get('section_id')
        text = chunk.get('original_text', chunk.get('text', ''))[:300]
        
        # Create a well-grounded answer
        if "entitled" in prompt.lower() or "how many" in prompt.lower():
            answer = f"According to Section {section_id} of the Employment Act Malaysia, you have specific entitlements outlined in the legislation. "
        elif "procedure" in prompt.lower() or "how do i" in prompt.lower():
            answer = f"Under Section {section_id} of the Employment Act, the proper procedure requires following specific steps. "
        elif "rights" in prompt.lower():
            answer = f"Your rights under Section {section_id} of the Employment Act provide important protections. "
        else:
            answer = f"Based on Section {section_id} of the Employment Act Malaysia, "
        
        # Add the relevant content
        answer += f"The relevant provision states: \"{text.strip()}\""
        
        # Add citation
        answer += f" (Source: Employment Act Section {section_id})"
        
        return answer
    
    def _generate_rejected_answer(self, prompt: str, chunk: Dict) -> str:
        """Generate a subtly flawed rejected answer (hallucination or missing citations)."""
        section_id = chunk.get('section_id')
        text = chunk.get('original_text', chunk.get('text', ''))[:300]
        
        rejection_type = random.choice([
            "missing_citation", 
            "wrong_section", 
            "hallucinated_details", 
            "vague_response"
        ])
        
        if rejection_type == "missing_citation":
            # Good content but no citation
            answer = f"You are entitled to specific benefits under the Employment Act. "
            answer += f"The law provides: \"{text.strip()}\""
            # No citation provided - this is the flaw
            
        elif rejection_type == "wrong_section":
            # Wrong section cited
            wrong_sections = ["EA-99", "EA-88", "EA-77"]
            wrong_section = random.choice(wrong_sections)
            answer = f"According to Section {wrong_section} of the Employment Act, you have certain entitlements. "
            answer += f"The provision states: \"{text[:200].strip()}...\""
            answer += f" (Source: Employment Act Section {wrong_section})"
            
        elif rejection_type == "hallucinated_details":
            # Add fake specific details
            fake_details = ["14 days", "30 calendar days", "6 months notice", "triple compensation"]
            fake_detail = random.choice(fake_details)
            answer = f"Under the Employment Act, you are entitled to {fake_detail} as specified in the legislation. "
            answer += f"The relevant section mentions: \"{text[:150].strip()}...\""
            answer += f" (Source: Employment Act Section {section_id})"
            
        else:  # vague_response
            # Too vague, unhelpful
            answer = "The Employment Act contains various provisions related to your question. "
            answer += "You should check the relevant sections or consult with HR for specific details. "
            answer += "Different situations may have different requirements."
        
        return answer
    
    def generate_preference_pairs(self, target_size: int = 60) -> List[Dict]:
        """Generate preference pairs for DPO training."""
        pairs = []
        
        # Ensure we cover all prompt types
        template_types = list(self.prompt_templates.keys())
        pairs_per_type = target_size // len(template_types)
        
        for template_type in template_types:
            for _ in range(pairs_per_type):
                # Generate prompt
                prompt = self._generate_prompt(template_type)
                
                # Get relevant chunk
                chunk = self._get_relevant_chunk(prompt)
                
                # Generate chosen and rejected answers
                chosen = self._generate_chosen_answer(prompt, chunk)
                rejected = self._generate_rejected_answer(prompt, chunk)
                
                # Create preference pair
                pair = {
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "source_section": chunk.get('section_id'),
                    "template_type": template_type,
                    "metadata": {
                        "chunk_id": chunk.get('chunk_id'),
                        "generated_at": datetime.now().isoformat()
                    }
                }
                
                pairs.append(pair)
        
        # Fill remaining slots
        remaining = target_size - len(pairs)
        for _ in range(remaining):
            template_type = random.choice(template_types)
            prompt = self._generate_prompt(template_type)
            chunk = self._get_relevant_chunk(prompt)
            chosen = self._generate_chosen_answer(prompt, chunk)
            rejected = self._generate_rejected_answer(prompt, chunk)
            
            pair = {
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "source_section": chunk.get('section_id'),
                "template_type": template_type,
                "metadata": {
                    "chunk_id": chunk.get('chunk_id'),
                    "generated_at": datetime.now().isoformat()
                }
            }
            pairs.append(pair)
        
        # Shuffle for diversity
        random.shuffle(pairs)
        return pairs[:target_size]
    
    def save_preference_pairs(self, pairs: List[Dict], output_file: Path, 
                            train_split: float = 0.8) -> None:
        """Save preference pairs split into train/eval sets."""
        
        # Shuffle pairs
        random.shuffle(pairs)
        
        # Split into train and eval
        split_idx = int(len(pairs) * train_split)
        train_pairs = pairs[:split_idx]
        eval_pairs = pairs[split_idx:]
        
        # Save train set
        train_file = output_file.parent / f"{output_file.stem}_train.jsonl"
        with open(train_file, 'w', encoding='utf-8') as f:
            for pair in train_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')
        
        # Save eval set
        eval_file = output_file.parent / f"{output_file.stem}_eval.jsonl"
        with open(eval_file, 'w', encoding='utf-8') as f:
            for pair in eval_pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')
        
        # Save combined dataset
        with open(output_file, 'w', encoding='utf-8') as f:
            for pair in pairs:
                f.write(json.dumps(pair, ensure_ascii=False) + '\n')
        
        print(f"Preference pairs saved:")
        print(f"  Total: {len(pairs)} pairs -> {output_file}")
        print(f"  Train: {len(train_pairs)} pairs -> {train_file}")
        print(f"  Eval:  {len(eval_pairs)} pairs -> {eval_file}")
        
        # Print dataset statistics
        print(f"\nDataset Statistics:")
        template_counts = {}
        for pair in pairs:
            template_type = pair['template_type']
            template_counts[template_type] = template_counts.get(template_type, 0) + 1
        
        for template_type, count in template_counts.items():
            print(f"  {template_type}: {count} pairs")
        
        sections_covered = len(set(pair['source_section'] for pair in pairs))
        print(f"  Sections covered: {sections_covered}")
        
        avg_prompt_len = sum(len(pair['prompt']) for pair in pairs) / len(pairs)
        avg_chosen_len = sum(len(pair['chosen']) for pair in pairs) / len(pairs)
        avg_rejected_len = sum(len(pair['rejected']) for pair in pairs) / len(pairs)
        
        print(f"  Avg prompt length: {avg_prompt_len:.1f} chars")
        print(f"  Avg chosen length: {avg_chosen_len:.1f} chars")
        print(f"  Avg rejected length: {avg_rejected_len:.1f} chars")


def main():
    parser = argparse.ArgumentParser(description="Generate preference pairs for DPO training")
    parser.add_argument('--chunks', required=True, help='Path to chunks.jsonl file')
    parser.add_argument('--output', required=True, help='Output JSONL file path')
    parser.add_argument('--size', type=int, default=60, 
                       help='Number of preference pairs to generate (default: 60)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    
    args = parser.parse_args()
    
    # Set random seed
    random.seed(args.seed)
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Generate preference pairs
    generator = PreferencePairGenerator(Path(args.chunks))
    pairs = generator.generate_preference_pairs(args.size)
    generator.save_preference_pairs(pairs, output_path)
    
    print(f"\n✅ Preference pair generation complete!")
    print(f"📊 Generated {len(pairs)} chosen vs rejected pairs")


if __name__ == "__main__":
    main()