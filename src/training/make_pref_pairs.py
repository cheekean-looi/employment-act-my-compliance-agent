#!/usr/bin/env python3
# python src/training/make_pref_pairs.py --chunks data/processed/chunks.jsonl --output outputs/dpo_pairs.jsonl --size 60 --seed 42 --sft-model outputs/lora_sft
"""
Fixed Preference Pairs Generator for Employment Act Malaysia Compliance Agent
Creates ~60 prompts with chosen vs rejected answers for DPO training.

FIXES APPLIED:
- Canonical citation regex: EA-YYYY-NNN[L]*[(N)] pattern everywhere
- Improved chunk selection using section families and keyword mapping
- Wrong-section negatives from valid ID universe (not synthetic)
- Labeling CLI moved to outputs/tools/ with dry-run/strict modes
- Enhanced grounding validation with unified patterns

Features:
- Bold SFT drafting: Optional --sft-model to draft both chosen/rejected via chat template
- Bold subtle negatives: Plausible wrong sections from valid EA universe
- Bold grounding check: Validates section IDs using canonical patterns
- Bold labeling workflow: CSV/JSONL for fast human pass with enhanced CLI
- Bold split hygiene: Split by source section to avoid leakage

Chosen: grounded, cites correctly
Rejected: subtly hallucinates or omits citations
"""

import json
import random
import csv
from pathlib import Path
from typing import List, Dict, Any, Set, Optional
import argparse
from datetime import datetime
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import sys
import os

# Add the training directory to the path to import citation_utils
sys.path.append(str(Path(__file__).parent))
from citation_utils import (
    CanonicalCitationValidator, 
    KeywordSectionMapper, 
    compute_enhanced_similarity
)


class FixedPreferencePairGenerator:
    def __init__(self, chunks_file: Path, sft_model_path: Optional[Path] = None, 
                 base_model: str = "meta-llama/Llama-3.2-1B-Instruct"):
        """Initialize with chunks and optional SFT model for drafting."""
        self.chunks = self._load_chunks(chunks_file)
        self.section_to_chunks = self._group_by_section()
        self.sft_model_path = sft_model_path
        self.base_model = base_model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set seeds for reproducibility
        self._set_seeds(42)
        
        # Initialize canonical citation validator with sections from chunks
        valid_sections = CanonicalCitationValidator.load_valid_sections_from_chunks(chunks_file)
        self.validator = CanonicalCitationValidator(valid_sections)
        print(f"📚 Loaded {len(self.validator.valid_sections)} valid section IDs")
        
        # Create section family mapping for better chunk selection
        self.section_families = self._build_section_family_map()
        
        # Initialize SFT model if provided
        self.sft_model = None
        self.sft_tokenizer = None
        if sft_model_path and Path(sft_model_path).exists():
            # Auto-align base model to adapter base if available
            try:
                cfg_path = Path(sft_model_path) / "adapter_config.json"
                if cfg_path.exists():
                    with open(cfg_path, 'r') as f:
                        _cfg = json.load(f)
                    hinted = _cfg.get("base_model_name_or_path") or _cfg.get("base_model_name")
                    if hinted and hinted != self.base_model:
                        print(f"🔁 Aligning preference drafting base to SFT adapter base: {hinted}")
                        self.base_model = hinted
            except Exception:
                pass
            self._load_sft_model()
        
        # Track rejection types for balance
        self.rejection_counts = {
            "missing_citation": 0,
            "wrong_section": 0, 
            "hallucinated_details": 0,
            "vague_response": 0
        }
        
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
    
    def _set_seeds(self, seed: int):
        """Set seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    
    def _build_section_family_map(self) -> Dict[str, List[Dict]]:
        """Build mapping from section families to chunks for better selection."""
        family_map = {}
        
        for chunk in self.chunks:
            section_id = chunk.get('section_id')
            if section_id:
                # Normalize to canonical before deriving family
                canonical_id = self.validator.normalize_section_id(section_id) or section_id
                family = self.validator.get_section_family(canonical_id)
                if family:
                    if family not in family_map:
                        family_map[family] = []
                    family_map[family].append(chunk)
        
        print(f"📊 Built section family map with {len(family_map)} families")
        return family_map
    
    def _load_sft_model(self):
        """Load SFT model for drafting responses."""
        try:
            print(f"Loading SFT model from {self.sft_model_path}...")
            
            # Load tokenizer
            self.sft_tokenizer = AutoTokenizer.from_pretrained(self.base_model)
            if self.sft_tokenizer.pad_token is None:
                self.sft_tokenizer.pad_token = self.sft_tokenizer.eos_token
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                self.base_model,
                torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # Load SFT adapter
            self.sft_model = PeftModel.from_pretrained(base_model, self.sft_model_path)
            self.sft_model.eval()
            
            print("SFT model loaded successfully for drafting responses")
            
        except Exception as e:
            print(f"Warning: Could not load SFT model: {e}")
            print("Falling back to heuristic response generation")
            self.sft_model = None
            self.sft_tokenizer = None
    
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
        """Get a relevant chunk for the prompt using improved section family matching."""
        
        # First try keyword-based family mapping
        relevant_families = KeywordSectionMapper.get_relevant_families(prompt)
        
        if relevant_families:
            # Try to find chunks from relevant families
            available_families = [f for f in relevant_families if f in self.section_families]
            if available_families:
                family = random.choice(available_families)
                chunks = self.section_families[family]
                return random.choice(chunks)
        
        # Fallback 1: Text-based matching
        prompt_lower = prompt.lower()
        
        # Look for exact section mentions in prompt
        predicted_sections = self.validator.extract_section_ids(prompt)
        if predicted_sections:
            valid_predicted = self.validator.validate_section_ids(predicted_sections)
            if valid_predicted:
                section_id = random.choice(list(valid_predicted))
                if section_id in self.section_to_chunks:
                    chunks = self.section_to_chunks[section_id]
                    return random.choice(chunks)
        
        # Fallback 2: Content-based matching within chunk text
        best_chunk = None
        best_score = 0
        
        for chunk in self.chunks[:50]:  # Sample first 50 for efficiency
            text = chunk.get('original_text', chunk.get('text', '')).lower()
            
            # Simple scoring based on shared keywords
            prompt_words = set(prompt_lower.split())
            text_words = set(text.split())
            
            if prompt_words and text_words:
                score = len(prompt_words.intersection(text_words)) / len(prompt_words.union(text_words))
                if score > best_score:
                    best_score = score
                    best_chunk = chunk
        
        # Fallback 3: Random chunk
        if not best_chunk:
            best_chunk = random.choice(self.chunks)
        
        return best_chunk
    
    def _draft_with_sft(self, prompt: str, chunk: Dict, is_chosen: bool = True) -> str:
        """Draft response using SFT model via chat template."""
        if not self.sft_model or not self.sft_tokenizer:
            return None
        
        try:
            # Prepare system prompt for chosen vs rejected
            if is_chosen:
                system_prompt = (
                    "You are an expert on the Malaysia Employment Act. "
                    "Answer concisely and accurately. Always cite at least one specific Act section in canonical form "
                    "EA-YYYY-NNN[L]*[(N)] at the end of the relevant sentence (e.g., EA-1955-060, EA-2012-044A(2)). "
                    "Only cite sections that apply; do not invent citations."
                )
            else:
                system_prompt = (
                    "You are responding to employment law questions. Provide general but plausible guidance. "
                    "Citations are optional and may be omitted."
                )
            
            # Format using chat template
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ]
            
            # Add structured retrieval context for chosen responses
            if is_chosen and chunk:
                raw_sid = chunk.get('section_id', '')
                section_id = self.validator.normalize_section_id(raw_sid) or raw_sid
                title = chunk.get('title', '')
                excerpt = chunk.get('original_text', chunk.get('text', ''))[:320]
                ctx = f"Section {section_id}{': ' + title if title else ''}. Excerpt: \"{excerpt}\""
                messages.append({"role": "assistant", "content": f"Use this source context: {ctx}"})
                messages.append({"role": "user", "content": "Now answer the original question, grounded in this section."})
            
            formatted_prompt = self.sft_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            
            # Generate response
            enc = self.sft_tokenizer(
                formatted_prompt,
                return_tensors="pt",
                padding=False,
                add_special_tokens=True
            )
            input_ids = enc["input_ids"].to(self.device)
            attention_mask = enc.get("attention_mask")
            if attention_mask is not None:
                attention_mask = attention_mask.to(self.device)
            
            with torch.no_grad():
                generate_kwargs = dict(
                    input_ids=input_ids,
                    max_new_tokens=220,
                    temperature=0.25 if is_chosen else 1.0,
                    top_p=0.90 if is_chosen else 0.95,
                    repetition_penalty=1.05,
                    do_sample=True,
                    pad_token_id=self.sft_tokenizer.eos_token_id,
                    eos_token_id=self.sft_tokenizer.eos_token_id,
                )
                if attention_mask is not None:
                    generate_kwargs["attention_mask"] = attention_mask
                outputs = self.sft_model.generate(**generate_kwargs)
            
            # Slice generated tokens after the prompt length
            prompt_len = input_ids.shape[1]
            response = self.sft_tokenizer.decode(outputs[0][prompt_len:], skip_special_tokens=True).strip()

            # Post-process: ensure canonical citation appears prominently in chosen responses
            if is_chosen:
                try:
                    section_id = chunk.get('section_id', '')
                    predicted = self.validator.extract_section_ids(response)
                    if section_id and (not predicted or section_id not in predicted):
                        # Inject canonical citation into the first sentence/paragraph for stronger supervision
                        first_break = min(
                            [i for i in [response.find('. '), response.find('\n')] if i != -1] or [min(200, len(response))]
                        )
                        insertion_point = first_break + (0 if first_break >= len(response) else 1)
                        # Compose injected snippet
                        injected = f" ({section_id})"
                        if insertion_point <= 0 or insertion_point > len(response):
                            response = response.strip() + injected
                        else:
                            response = response[:insertion_point] + injected + response[insertion_point:]
                except Exception:
                    pass
            return response
            
        except Exception as e:
            print(f"Warning: SFT drafting failed: {e}")
            return None
    
    def _validate_grounding(self, response: str, source_section_id: str) -> Dict[str, Any]:
        """Validate grounding using canonical citation patterns."""
        predicted_sections = self.validator.extract_section_ids(response)
        gold_sections = {self.validator.normalize_section_id(source_section_id) or source_section_id}
        
        citation_em, citation_iou = self.validator.compute_citation_metrics(predicted_sections, gold_sections)
        
        return {
            "gold_section_id": source_section_id,
            "predicted_section_ids": list(predicted_sections),
            "correctly_grounded": citation_em > 0.0,
            "valid_sections_mentioned": len(self.validator.validate_section_ids(predicted_sections)) > 0,
            "grounding_score": citation_iou,
            "citation_em": citation_em,
            "citation_iou": citation_iou
        }
    
    def _generate_chosen_answer(self, prompt: str, chunk: Dict) -> str:
        """Generate a high-quality chosen answer with proper citations."""
        raw_sid = chunk.get('section_id')
        section_id = self.validator.normalize_section_id(raw_sid) or raw_sid
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
    
    def _add_subtle_flaws(self, response: str, correct_section_id: str) -> str:
        """Add subtle flaws to SFT-generated response."""
        # Randomly choose a flaw type
        flaw_type = random.choice(["wrong_section", "remove_citation", "add_hallucination"])
        
        canonical_correct = self.validator.normalize_section_id(correct_section_id) or correct_section_id
        if flaw_type == "wrong_section" and (canonical_correct in response or correct_section_id in response):
            # Use a different valid section ID instead of synthetic ones
            wrong_section = self.validator.get_different_valid_section(canonical_correct)
            if wrong_section:
                # Replace either canonical or legacy occurrence
                if canonical_correct in response:
                    return response.replace(canonical_correct, wrong_section)
                return response.replace(correct_section_id, wrong_section)
        elif flaw_type == "remove_citation":
            # Remove citation patterns using canonical regex
            response = self.validator.CANONICAL_PATTERN.sub('the relevant section', response)
            response = response.replace("(Source: Employment Act Section", "(Source: the relevant section")
            return response
        elif flaw_type == "add_hallucination":
            fake_details = ["exactly 14 days", "triple compensation", "immediate termination rights"]
            fake_detail = random.choice(fake_details)
            return response + f" Additionally, you are entitled to {fake_detail} in such cases."
        
        return response
    
    def _generate_rejected_answer(self, prompt: str, chunk: Dict) -> str:
        """Generate a subtly flawed rejected answer with balanced rejection types."""
        raw_sid = chunk.get('section_id')
        section_id = self.validator.normalize_section_id(raw_sid) or raw_sid
        text = chunk.get('original_text', chunk.get('text', ''))[:300]
        
        # First try SFT drafting for rejected response
        if self.sft_model:
            sft_response = self._draft_with_sft(prompt, chunk, is_chosen=False)
            if sft_response and len(sft_response) > 50:
                # Post-process to add subtle flaws
                return self._add_subtle_flaws(sft_response, section_id)
        
        # Choose rejection type based on balance
        min_count = min(self.rejection_counts.values())
        underrepresented_types = [k for k, v in self.rejection_counts.items() if v == min_count]
        rejection_type = random.choice(underrepresented_types)
        self.rejection_counts[rejection_type] += 1
        
        if rejection_type == "missing_citation":
            answer = f"You are entitled to specific benefits under the Employment Act. "
            answer += f"The law provides: \"{text.strip()}\""
            # No citation provided - this is the flaw
            
        elif rejection_type == "wrong_section":
            # Use a different valid section ID instead of synthetic ones
            wrong_section = self.validator.get_different_valid_section(section_id)
            if not wrong_section:
                # Fallback to missing citation if no other valid sections available
                answer = f"You are entitled to specific benefits under the Employment Act. "
                answer += f"The law provides: \"{text.strip()}\""
            else:
                answer = f"According to Section {wrong_section} of the Employment Act, you have certain entitlements. "
                answer += f"The provision states: \"{text[:200].strip()}...\""
                answer += f" (Source: Employment Act Section {wrong_section})"
            
        elif rejection_type == "hallucinated_details":
            fake_details = ["14 days", "30 calendar days", "6 months notice", "triple compensation"]
            fake_detail = random.choice(fake_details)
            answer = f"Under the Employment Act, you are entitled to {fake_detail} as specified in the legislation. "
            answer += f"The relevant section mentions: \"{text[:150].strip()}...\""
            answer += f" (Source: Employment Act Section {section_id})"
            
        else:  # vague_response
            answer = "The Employment Act contains various provisions related to your question. "
            answer += "You should check the relevant sections or consult with legal counsel for specific details. "
            answer += "Different situations may have different requirements."
        
        return answer
    
    def generate_preference_pairs(self, target_size: int = 60) -> List[Dict]:
        """Generate preference pairs for DPO training with enhanced grounding validation."""
        pairs = []

        # Ensure we cover all prompt types and diversify sections early
        template_types = list(self.prompt_templates.keys())
        section_ids = list(self.section_to_chunks.keys())
        random.shuffle(section_ids)

        # First pass: guarantee diverse section coverage up to target_size
        initial_sections = section_ids[: min(len(section_ids), target_size)]
        for i, section_id in enumerate(initial_sections):
            template_type = template_types[i % len(template_types)]
            prompt = self._generate_prompt(template_type)
            # Choose a chunk from this specific section for grounding diversity
            chunk = random.choice(self.section_to_chunks[section_id])

            # Generate answers
            if self.sft_model:
                chosen = self._draft_with_sft(prompt, chunk, is_chosen=True)
                if not chosen or len(chosen) < 50:
                    chosen = self._generate_chosen_answer(prompt, chunk)
            else:
                chosen = self._generate_chosen_answer(prompt, chunk)

            rejected = self._generate_rejected_answer(prompt, chunk)

            chosen_grounding = self._validate_grounding(chosen, chunk.get('section_id'))
            rejected_grounding = self._validate_grounding(rejected, chunk.get('section_id'))

            pairs.append({
                "pair_id": f"pair_{len(pairs):04d}",
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "source_section": chunk.get('section_id'),
                "template_type": template_type,
                "label_verified": False,
                "chosen_grounding": chosen_grounding,
                "rejected_grounding": rejected_grounding,
                "metadata": {
                    "chunk_id": chunk.get('chunk_id'),
                    "generated_at": datetime.now().isoformat(),
                    "sft_drafted": self.sft_model is not None,
                    "validator_version": "canonical_v1"
                }
            })

        # Fill remaining slots with regular flow
        remaining = target_size - len(pairs)
        for i in range(remaining):
            template_type = template_types[(len(pairs) + i) % len(template_types)]
            prompt = self._generate_prompt(template_type)
            chunk = self._get_relevant_chunk(prompt)

            if self.sft_model:
                chosen = self._draft_with_sft(prompt, chunk, is_chosen=True)
                if not chosen or len(chosen) < 50:
                    chosen = self._generate_chosen_answer(prompt, chunk)
            else:
                chosen = self._generate_chosen_answer(prompt, chunk)

            rejected = self._generate_rejected_answer(prompt, chunk)

            chosen_grounding = self._validate_grounding(chosen, chunk.get('section_id'))
            rejected_grounding = self._validate_grounding(rejected, chunk.get('section_id'))

            pairs.append({
                "pair_id": f"pair_{len(pairs):04d}",
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected,
                "source_section": chunk.get('section_id'),
                "template_type": template_type,
                "label_verified": False,
                "chosen_grounding": chosen_grounding,
                "rejected_grounding": rejected_grounding,
                "metadata": {
                    "chunk_id": chunk.get('chunk_id'),
                    "generated_at": datetime.now().isoformat(),
                    "sft_drafted": self.sft_model is not None,
                    "validator_version": "canonical_v1"
                }
            })

        # Grounding quality gate: ensure >= 60% chosen correctly grounded if possible
        def _chosen_grounded_ratio(items: List[Dict]) -> float:
            if not items:
                return 0.0
            good = sum(1 for p in items if p.get('chosen_grounding', {}).get('correctly_grounded'))
            return good / len(items)

        ratio = _chosen_grounded_ratio(pairs)
        max_rounds = 3
        cap = int(target_size * 1.5)
        rounds = 0
        while ratio < 0.6 and len(pairs) < cap and rounds < max_rounds:
            # Generate a small batch of additional pairs to try to lift grounding ratio
            batch = min(10, target_size // 3 or 1)
            for _ in range(batch):
                template_type = random.choice(template_types)
                prompt = self._generate_prompt(template_type)
                chunk = self._get_relevant_chunk(prompt)

                if self.sft_model:
                    chosen = self._draft_with_sft(prompt, chunk, is_chosen=True)
                    if not chosen or len(chosen) < 50:
                        chosen = self._generate_chosen_answer(prompt, chunk)
                else:
                    chosen = self._generate_chosen_answer(prompt, chunk)

                rejected = self._generate_rejected_answer(prompt, chunk)

                chosen_grounding = self._validate_grounding(chosen, chunk.get('section_id'))
                rejected_grounding = self._validate_grounding(rejected, chunk.get('section_id'))

                pairs.append({
                    "pair_id": f"pair_{len(pairs):04d}",
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected,
                    "source_section": chunk.get('section_id'),
                    "template_type": template_type,
                    "label_verified": False,
                    "chosen_grounding": chosen_grounding,
                    "rejected_grounding": rejected_grounding,
                    "metadata": {
                        "chunk_id": chunk.get('chunk_id'),
                        "generated_at": datetime.now().isoformat(),
                        "sft_drafted": self.sft_model is not None,
                        "validator_version": "canonical_v1"
                    }
                })
            ratio = _chosen_grounded_ratio(pairs)
            rounds += 1

        if ratio < 0.6:
            print(f"⚠️ Chosen grounding ratio below target: {ratio:.1%} (< 60%). Consider increasing pairs or improving prompts.")
        else:
            print(f"✅ Chosen grounding ratio: {ratio:.1%} (>= 60%)")

        random.shuffle(pairs)
        return pairs[:max(len(pairs), target_size)]
    
    def save_preference_pairs(self, pairs: List[Dict], output_file: Path, 
                            train_split: float = 0.8) -> Dict[str, List[str]]:
        """Save preference pairs with split hygiene by source section."""
        
        # Split by source section to avoid leakage
        sections = list(set(pair['source_section'] for pair in pairs))
        random.shuffle(sections)
        
        split_idx = int(len(sections) * train_split)
        train_sections = set(sections[:split_idx])
        eval_sections = set(sections[split_idx:])
        
        train_pairs = [p for p in pairs if p['source_section'] in train_sections]
        eval_pairs = [p for p in pairs if p['source_section'] in eval_sections]
        
        # Store pair IDs for reuse across DPO runs
        train_pair_ids = [p['pair_id'] for p in train_pairs]
        eval_pair_ids = [p['pair_id'] for p in eval_pairs]
        
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
        
        # Create tools directory and save labeling CSV there
        tools_dir = output_file.parent / "tools"
        tools_dir.mkdir(exist_ok=True)
        
        csv_file = tools_dir / f"{output_file.stem}_labeling.csv"
        cli_file = tools_dir / "labeling_cli.py"
        
        self._save_labeling_csv(pairs, csv_file)
        self._create_enhanced_labeling_cli(cli_file)
        
        # Save split metadata with eval pair IDs for reuse
        split_metadata = {
            "train_pair_ids": train_pair_ids,
            "eval_pair_ids": eval_pair_ids,
            "train_sections": list(train_sections),
            "eval_sections": list(eval_sections),
            "split_ratio": train_split,
            "total_pairs": len(pairs),
            "created_at": datetime.now().isoformat(),
            "canonical_validator": True,
            "valid_sections_count": len(self.validator.valid_sections)
        }
        
        with open(output_file.parent / f"{output_file.stem}_split_metadata.json", 'w') as f:
            json.dump(split_metadata, f, indent=2)
        
        # Print statistics
        self._print_dataset_statistics(pairs, train_pairs, eval_pairs, output_file, tools_dir)
        
        return {"train_pair_ids": train_pair_ids, "eval_pair_ids": eval_pair_ids}
    
    def _save_labeling_csv(self, pairs: List[Dict], csv_file: Path):
        """Save compact CSV for fast human labeling."""
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['pair_id', 'prompt', 'chosen_preview', 'rejected_preview', 
                           'chosen_valid_citations', 'rejected_valid_citations',
                           'ok_chosen', 'ok_rejected', 'notes'])
            
            for pair in pairs:
                chosen_preview = pair['chosen'][:100] + "..." if len(pair['chosen']) > 100 else pair['chosen']
                rejected_preview = pair['rejected'][:100] + "..." if len(pair['rejected']) > 100 else pair['rejected']
                
                # Add citation validation info
                chosen_citations = len(pair['chosen_grounding']['predicted_section_ids'])
                rejected_citations = len(pair['rejected_grounding']['predicted_section_ids'])
                
                writer.writerow([
                    pair['pair_id'],
                    pair['prompt'],
                    chosen_preview,
                    rejected_preview,
                    chosen_citations,
                    rejected_citations,
                    '',  # ok_chosen - to be filled by human
                    '',  # ok_rejected - to be filled by human  
                    ''   # notes - to be filled by human
                ])
    
    def _create_enhanced_labeling_cli(self, cli_file: Path):
        """Create enhanced labeling CLI with dry-run and strict modes."""
        cli_script = '''#!/usr/bin/env python3
"""
Enhanced Labeling CLI tool - Update preference pair labels from CSV
Usage: 
  python labeling_cli.py --csv path/to/labeling.csv --jsonl path/to/pairs.jsonl
  python labeling_cli.py --csv path/to/labeling.csv --jsonl path/to/pairs.jsonl --dry-run
  python labeling_cli.py --csv path/to/labeling.csv --jsonl path/to/pairs.jsonl --strict
"""
import json
import csv
import argparse
from pathlib import Path
import re

# Canonical citation pattern
CANONICAL_PATTERN = re.compile(r'\\b(EA-\\d{4}-\\d+[A-Z]*(?:\\(\\d+\\))?)\\b', re.IGNORECASE)

def validate_pair_quality(pair, strict_mode=False):
    """Validate pair quality for strict mode filtering."""
    chosen = pair.get('chosen', '')
    rejected = pair.get('rejected', '')
    source_section = pair.get('source_section', '')
    
    # Extract sections from responses
    chosen_sections = set(CANONICAL_PATTERN.findall(chosen.upper()))
    rejected_sections = set(CANONICAL_PATTERN.findall(rejected.upper()))
    
    issues = []
    
    if strict_mode:
        # Strict mode: chosen must have valid section, rejected must not have gold section
        if not chosen_sections:
            issues.append("chosen_no_citations")
        
        if source_section.upper() in rejected_sections:
            issues.append("rejected_has_gold_section")
    
    return issues

def update_labels_from_csv(csv_file: Path, jsonl_file: Path, dry_run=False, strict_mode=False):
    """Update label_verified field based on CSV feedback."""
    
    # Load CSV feedback
    feedback = {}
    with open(csv_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            pair_id = row['pair_id']
            ok_chosen = row['ok_chosen'].strip()
            ok_rejected = row['ok_rejected'].strip()
            notes = row['notes'].strip()
            
            if ok_chosen in ['1', '0'] and ok_rejected in ['1', '0']:
                feedback[pair_id] = {
                    'ok_chosen': ok_chosen == '1',
                    'ok_rejected': ok_rejected == '1', 
                    'notes': notes,
                    'label_verified': True
                }
    
    print(f"📊 Loaded feedback for {len(feedback)} pairs from CSV")
    
    # Load and update JSONL
    pairs = []
    filtered_count = 0
    strict_filtered_count = 0
    
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            pair = json.loads(line.strip())
            pair_id = pair['pair_id']
            
            # Strict mode validation
            if strict_mode:
                issues = validate_pair_quality(pair, strict_mode=True)
                if issues:
                    strict_filtered_count += 1
                    if not dry_run:
                        continue  # Skip pairs with quality issues
                    else:
                        print(f"⚠️ STRICT: {pair_id} would be filtered: {', '.join(issues)}")
            
            # Apply human feedback
            if pair_id in feedback:
                pair['label_verified'] = feedback[pair_id]['label_verified']
                pair['human_feedback'] = feedback[pair_id]
                
                # Filter out bad pairs based on human feedback
                if not (feedback[pair_id]['ok_chosen'] and feedback[pair_id]['ok_rejected']):
                    filtered_count += 1
                    if not dry_run:
                        continue  # Skip bad pairs
                    else:
                        print(f"❌ HUMAN: {pair_id} would be filtered: human marked as bad")
            
            pairs.append(pair)
    
    if dry_run:
        print(f"\\n📊 DRY RUN SUMMARY:")
        print(f"   Original pairs: {len(pairs)}")
        print(f"   Human feedback available: {len(feedback)}")
        print(f"   Would filter (human): {filtered_count}")
        if strict_mode:
            print(f"   Would filter (strict): {strict_filtered_count}")
        print(f"   Would remain: {len(pairs) - filtered_count - strict_filtered_count}")
        return
    
    # Save updated JSONL
    output_file = jsonl_file.parent / f"{jsonl_file.stem}_verified.jsonl"
    with open(output_file, 'w', encoding='utf-8') as f:
        for pair in pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + '\\n')
    
    verified_count = sum(1 for p in pairs if p.get('label_verified', False))
    print(f"\\n✅ PROCESSING COMPLETE:")
    print(f"   Final pairs: {len(pairs)}")
    print(f"   Verified pairs: {verified_count}")
    print(f"   Filtered (human): {filtered_count}")
    if strict_mode:
        print(f"   Filtered (strict): {strict_filtered_count}")
    print(f"   Saved to: {output_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced preference pair label updater")
    parser.add_argument('--csv', required=True, help='Path to filled labeling CSV')
    parser.add_argument('--jsonl', required=True, help='Path to preference pairs JSONL')
    parser.add_argument('--dry-run', action='store_true', help='Show what would be filtered without making changes')
    parser.add_argument('--strict', action='store_true', help='Enable strict mode: drop pairs where chosen lacks valid sections or rejected contains gold section')
    args = parser.parse_args()
    
    update_labels_from_csv(Path(args.csv), Path(args.jsonl), 
                          dry_run=args.dry_run, strict_mode=args.strict)
'''
        
        # Save CLI script
        with open(cli_file, 'w') as f:
            f.write(cli_script)
        
        # Make executable
        import stat
        cli_file.chmod(cli_file.stat().st_mode | stat.S_IEXEC)
    
    def _print_dataset_statistics(self, pairs: List[Dict], train_pairs: List[Dict], 
                                eval_pairs: List[Dict], output_file: Path, tools_dir: Path):
        """Print comprehensive dataset statistics."""
        print(f"\n✅ Fixed preference pairs saved:")
        print(f"  Total: {len(pairs)} pairs -> {output_file}")
        print(f"  Train: {len(train_pairs)} pairs ({len(set(p['source_section'] for p in train_pairs))} sections)")
        print(f"  Eval:  {len(eval_pairs)} pairs ({len(set(p['source_section'] for p in eval_pairs))} sections)")
        
        # Template distribution
        print(f"\n📊 Template distribution:")
        template_counts = {}
        for pair in pairs:
            template_type = pair['template_type']
            template_counts[template_type] = template_counts.get(template_type, 0) + 1
        
        for template_type, count in template_counts.items():
            print(f"  {template_type}: {count} pairs")
        
        # Rejection type balance
        print(f"\n⚖️ Rejection type balance:")
        for rejection_type, count in self.rejection_counts.items():
            print(f"  {rejection_type}: {count} pairs")
        
        # Canonical grounding statistics
        chosen_grounded = sum(1 for p in pairs if p['chosen_grounding']['correctly_grounded'])
        rejected_grounded = sum(1 for p in pairs if p['rejected_grounding']['correctly_grounded'])
        
        print(f"\n🎯 Canonical grounding validation:")
        print(f"  Chosen correctly grounded: {chosen_grounded}/{len(pairs)} ({chosen_grounded/len(pairs):.1%})")
        print(f"  Rejected correctly grounded: {rejected_grounded}/{len(pairs)} ({rejected_grounded/len(pairs):.1%})")
        print(f"  Valid sections in dataset: {len(self.validator.valid_sections)}")
        
        # Enhanced labeling workflow
        print(f"\n📋 Enhanced labeling workflow:")
        print(f"  CSV file: {tools_dir / f'{output_file.stem}_labeling.csv'}")
        print(f"  CLI tool: {tools_dir / 'labeling_cli.py'}")
        print(f"  Usage:")
        print(f"    # Dry run: python {tools_dir / 'labeling_cli.py'} --csv [csv] --jsonl [jsonl] --dry-run")
        print(f"    # Strict mode: python {tools_dir / 'labeling_cli.py'} --csv [csv] --jsonl [jsonl] --strict")


def main():
    parser = argparse.ArgumentParser(description="Fixed preference pairs generator with canonical patterns")
    parser.add_argument('--chunks', required=True, help='Path to chunks.jsonl file')
    parser.add_argument('--output', required=True, help='Output JSONL file path')
    parser.add_argument('--size', type=int, default=60, 
                       help='Number of preference pairs to generate (default: 60)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--sft-model', help='Path to SFT model for drafting responses')
    parser.add_argument('--base-model', default="meta-llama/Llama-3.1-8B-Instruct",
                       help='Base model for SFT drafting')
    
    args = parser.parse_args()
    
    # Create output directory
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save metadata with canonical validator info
    metadata = {
        "script_args": vars(args),
        "seeds": {"random": args.seed, "numpy": args.seed, "torch": args.seed},
        "timestamp": datetime.now().isoformat(),
        "fixes_applied": {
            "canonical_citation_regex": True,
            "improved_chunk_selection": True,
            "valid_wrong_sections": True,
            "enhanced_labeling_cli": True,
            "unified_grounding_validation": True
        },
        "features": {
            "sft_drafting": args.sft_model is not None,
            "grounding_validation": True,
            "split_hygiene": True,
            "labeling_workflow": True,
            "canonical_patterns": True
        }
    }
    
    with open(output_path.parent / "generation_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Generate preference pairs
    generator = FixedPreferencePairGenerator(
        Path(args.chunks), 
        sft_model_path=Path(args.sft_model) if args.sft_model else None,
        base_model=args.base_model
    )
    
    pairs = generator.generate_preference_pairs(args.size)
    split_info = generator.save_preference_pairs(pairs, output_path)
    
    print(f"\n🎉 FIXED preference pair generation complete!")
    print(f"📊 Generated {len(pairs)} chosen vs rejected pairs")
    print(f"🔧 Applied canonical citation patterns (EA-YYYY-NNN[L]*[(N)])")
    print(f"🎯 Features: Enhanced chunk selection, valid wrong-sections, canonical grounding")
    print(f"📁 Enhanced labeling CLI available in tools/ directory")
    if args.sft_model:
        print(f"🤖 Used SFT model: {args.sft_model}")


if __name__ == "__main__":
    main()
