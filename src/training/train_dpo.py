#!/usr/bin/env python3
# python src/training/train_dpo.py --train-data outputs/dpo_pairs_train.jsonl --eval-data outputs/dpo_pairs_eval.jsonl --output-dir outputs/lora_dpo --sft-model outputs/lora_sft --model-name meta-llama/Llama-3.1-8B-Instruct --epochs 1 --batch-size 2 --learning-rate 5e-5 --beta 0.1
"""
Fixed DPO Training Script for Employment Act Malaysia Compliance Agent
Runs Direct Preference Optimization with comprehensive metrics and evaluation.

FIXES APPLIED:
- Tokenizer padding "right" for training, "left" for inference
- Persistent eval subset using split metadata for consistent evaluation
- Enhanced pairwise win-rate with groundedness-aware scoring (0.5×EM + 0.3×IoU + 0.2×F1)
- Separate safety vs vagueness penalties in judge
- Citation metric history plotting alongside win-rate
- Eval pair IDs tracking in metadata for reproducibility

Features:
- Bold QLoRA baseline: 4-bit quantization with adamw_bnb_8bit, bf16, gradient checkpointing
- Bold dataset columns: TRL-compatible prompt/chosen/rejected format
- Bold seeds + determinism: Full reproducibility with metadata tracking
- Bold metrics: Enhanced pairwise win-rate, citation exact/IoU, safety violations
- Bold judge: Rule-based + canonical pattern validation
- Bold checkpointing: Resume capability and periodic saves
"""

import json
import torch
import random
import numpy as np
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, 
    BitsAndBytesConfig, TrainingArguments
)
from transformers.trainer_callback import TrainerCallback
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel, prepare_model_for_kbit_training
import argparse
from typing import Dict, List, Any, Optional, Set
import matplotlib.pyplot as plt
from datetime import datetime
import re
import os
import logging
import sys

# Add the training directory to the path to import citation_utils
sys.path.append(str(Path(__file__).parent))
from citation_utils import (
    CanonicalCitationValidator, 
    compute_enhanced_similarity,
    compute_lexical_f1
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FixedEmploymentActDPOTrainer:
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
                 sft_model_path: Optional[str] = None,
                 lora_rank: int = 8,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.1,
                 use_4bit: bool = True,
                 use_flash_attention: bool = True,
                 seed: int = 42):
        """Initialize fixed DPO trainer with canonical patterns and persistent eval."""
        
        self.model_name = model_name
        self.sft_model_path = sft_model_path
        self.use_4bit = use_4bit
        self.use_flash_attention = use_flash_attention
        
        # Set seeds for reproducibility
        self._set_seeds(seed)
        self.seed = seed
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"🚀 Initializing Fixed Employment Act DPO Trainer")
        print(f"📱 Device: {self.device}")
        print(f"🤖 Model: {model_name}")
        print(f"⚡ 4-bit quantization: {use_4bit}")
        print(f"🔥 Flash attention (requested): {use_flash_attention}")
        if sft_model_path:
            print(f"📚 SFT checkpoint: {sft_model_path}")
        
        # Load tokenizer with correct padding for training
        print("📖 Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # FIXED: Use "right" padding for training
        self.tokenizer.padding_side = "right"
        print(f"✅ Tokenizer padding side set to: {self.tokenizer.padding_side}")
        
        # Initialize canonical citation validator
        self.validator = CanonicalCitationValidator()
        
        # Load model with QLoRA configuration
        self._load_model_with_qlora(lora_rank, lora_alpha, lora_dropout)
        
        # Initialize enhanced judge for evaluation
        self.judge = EnhancedEmploymentActJudge(self.validator)
        
        # Metrics tracking
        self.training_metrics = []
        self.eval_metrics = []
        self.eval_pair_ids = None  # Will be loaded from split metadata
    
    def _set_seeds(self, seed: int):
        """Set seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        
        # Ensure deterministic behavior
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    def _load_model_with_qlora(self, lora_rank: int, lora_alpha: int, lora_dropout: float):
        """Load model with QLoRA configuration for memory efficiency."""
        
        # Configure 4-bit quantization
        if self.use_4bit and torch.cuda.is_available():
            print("⚙️ Configuring 4-bit quantization (nf4, double_quant)...")
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        else:
            bnb_config = None
        
        # Load base model
        print("🔧 Loading base model...")
        model_kwargs = {
            "torch_dtype": torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            "device_map": "auto" if torch.cuda.is_available() else None,
        }
        
        if bnb_config:
            model_kwargs["quantization_config"] = bnb_config

        # Select attention implementation safely (prefer SDPA unless FA2 is explicitly available)
        attn_impl = None
        attn_env = os.getenv("ATTN_IMPL")
        fa2_env = os.getenv("HF_USE_FLASH_ATTENTION_2", "0") != "0"
        want_fa2 = self.use_flash_attention or fa2_env or (attn_env == "flash_attention_2")
        if want_fa2:
            try:
                import flash_attn  # noqa: F401
                attn_impl = "flash_attention_2"
            except Exception:
                print("⚠️ FlashAttention2 requested but not available. Falling back to SDPA.")
                attn_impl = "sdpa" if torch.cuda.is_available() else "eager"
        else:
            if attn_env in {"sdpa", "eager"}:
                attn_impl = attn_env
            else:
                attn_impl = "sdpa" if torch.cuda.is_available() else "eager"

        model_kwargs["attn_implementation"] = attn_impl
        print(f"🧠 Attention implementation: {attn_impl}")
        
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **model_kwargs
        )
        
        # Prepare for k-bit training if using quantization
        if bnb_config:
            base_model = prepare_model_for_kbit_training(base_model)
        
        # Load SFT checkpoint if provided
        if self.sft_model_path and Path(self.sft_model_path).exists():
            print(f"📚 Loading SFT checkpoint from {self.sft_model_path}")
            try:
                # Load the SFT LoRA adapter as policy model
                self.model = PeftModel.from_pretrained(base_model, self.sft_model_path)
                print("✅ Successfully loaded SFT adapter as starting policy")
                
                # Create reference model (frozen copy)
                print("🔒 Creating frozen reference model...")
                ref_model_kwargs = model_kwargs.copy()
                if "device_map" in ref_model_kwargs:
                    ref_model_kwargs["device_map"] = "auto"
                
                ref_base_model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    **ref_model_kwargs
                )
                
                if bnb_config:
                    ref_base_model = prepare_model_for_kbit_training(ref_base_model)
                
                self.ref_model = PeftModel.from_pretrained(ref_base_model, self.sft_model_path)
                
                # Freeze reference model
                for param in self.ref_model.parameters():
                    param.requires_grad = False
                
                print("✅ Reference model created and frozen")
                
            except Exception as e:
                print(f"⚠️ Could not load SFT checkpoint: {e}")
                print("🔄 Falling back to base model...")
                self.model = base_model
                self.ref_model = None
        else:
            print("📝 No SFT checkpoint provided - initializing new LoRA")
            
            # Configure LoRA for DPO
            target_modules = self._get_target_modules()
            
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none"
            )
            
            self.model = get_peft_model(base_model, lora_config)
            self.ref_model = None
        
        # Print trainable parameters
        if hasattr(self.model, 'print_trainable_parameters'):
            self.model.print_trainable_parameters()
        
        print("✅ Model loading complete!")
    
    def _get_target_modules(self) -> List[str]:
        """Get target modules for LoRA based on model architecture."""
        if "qwen" in self.model_name.lower():
            return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "mistral" in self.model_name.lower() or "llama" in self.model_name.lower():
            return ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        else:
            return ["q_proj", "v_proj", "k_proj", "o_proj"]
    
    def _format_prompt(self, prompt: str) -> str:
        """Format prompt using proper chat template."""
        if "qwen" in self.model_name.lower() or "llama" in self.model_name.lower():
            messages = [
                {"role": "system", "content": "You are an expert on Malaysia Employment Act. Provide accurate, helpful answers with proper citations."},
                {"role": "user", "content": prompt}
            ]
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            return f"Human: {prompt}\nAssistant: "
    
    def load_preference_data(self, train_file: Path, eval_file: Path) -> tuple:
        """Load and prepare preference datasets for DPO."""
        print(f"📂 Loading preference datasets...")
        print(f"🚆 Train: {train_file}")
        print(f"🧪 Eval:  {eval_file}")
        
        # Load train data
        train_data = []
        with open(train_file, 'r', encoding='utf-8') as f:
            for line in f:
                train_data.append(json.loads(line.strip()))
        
        # Load eval data
        eval_data = []
        with open(eval_file, 'r', encoding='utf-8') as f:
            for line in f:
                eval_data.append(json.loads(line.strip()))
        
        # FIXED: Load eval pair IDs from split metadata for consistency
        self._load_eval_subset_ids(train_file.parent)
        
        print(f"📊 Loaded {len(train_data)} training pairs")
        print(f"📊 Loaded {len(eval_data)} evaluation pairs")
        if self.eval_pair_ids:
            print(f"📍 Using fixed eval subset: {len(self.eval_pair_ids)} pair IDs")
        
        return train_data, eval_data
    
    def _load_eval_subset_ids(self, data_dir: Path):
        """Load fixed eval subset IDs from split metadata."""
        metadata_files = list(data_dir.glob("*_split_metadata.json"))
        
        if metadata_files:
            metadata_file = metadata_files[0]  # Use first found
            try:
                with open(metadata_file, 'r') as f:
                    metadata = json.load(f)
                self.eval_pair_ids = set(metadata.get('eval_pair_ids', []))
                print(f"✅ Loaded fixed eval subset from {metadata_file}")
            except Exception as e:
                print(f"⚠️ Could not load eval subset from {metadata_file}: {e}")
                self.eval_pair_ids = None
        else:
            print("⚠️ No split metadata found - using full eval set")
            self.eval_pair_ids = None
    
    def preprocess_preference_data(self, data: List[Dict]) -> Dataset:
        """Preprocess preference data for DPO training with TRL format."""
        
        def format_dataset(examples):
            formatted_data = {
                'prompt': [],
                'chosen': [],
                'rejected': []
            }
            
            for example in examples:
                # Format prompt using chat template
                prompt = self._format_prompt(example['prompt'])
                
                # For DPO, responses should be plain text (not re-templated)
                formatted_data['prompt'].append(prompt)
                formatted_data['chosen'].append(example['chosen'])
                formatted_data['rejected'].append(example['rejected'])
            
            return formatted_data
        
        # Convert to HuggingFace dataset format
        dataset_dict = format_dataset(data)
        dataset = Dataset.from_dict(dataset_dict)
        
        return dataset
    
    def train_dpo(self, 
                  train_data: List[Dict], 
                  eval_data: List[Dict],
                  output_dir: Path,
                  num_epochs: int = 1,
                  batch_size: int = 2,
                  learning_rate: float = 5e-5,
                  beta: float = 0.1,
                  resume_from_checkpoint: Optional[str] = None):
        """Train the model with enhanced DPO."""
        
        print(f"🎯 Starting Fixed DPO training...")
        print(f"📈 Epochs: {num_epochs}")
        print(f"📦 Batch size: {batch_size}")
        print(f"🎛️ Learning rate: {learning_rate}")
        print(f"⚖️ Beta (KL penalty): {beta}")
        print(f"📁 Output dir: {output_dir}")
        
        # Preprocess datasets
        train_dataset = self.preprocess_preference_data(train_data)
        eval_dataset = self.preprocess_preference_data(eval_data)
        
        # Enhanced DPO Configuration
        training_args = DPOConfig(
            output_dir=str(output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=2,
            optim="adamw_bnb_8bit" if self.use_4bit else "adamw_torch",
            save_steps=50,
            logging_steps=10,
            learning_rate=learning_rate,
            weight_decay=0.001,
            fp16=False,
            bf16=torch.cuda.is_available(),
            max_grad_norm=1.0,
            warmup_steps=10,
            lr_scheduler_type="cosine",
            eval_strategy="steps",
            eval_steps=25,
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=[],
            beta=beta,
            remove_unused_columns=False,
            gradient_checkpointing=True,  # Memory optimization
            dataloader_drop_last=False,
            seed=self.seed,
        )
        
        # Initialize DPO trainer
        dpo_trainer = DPOTrainer(
            model=self.model,
            ref_model=self.ref_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
        )
        
        # Add custom callback for metrics tracking
        dpo_trainer.add_callback(FixedDPOMetricsCallback(self, eval_data, output_dir))
        
        # Resume from checkpoint if specified
        if resume_from_checkpoint:
            print(f"🔄 Resuming from checkpoint: {resume_from_checkpoint}")
        
        # Train the model
        print("🚀 Starting DPO training...")
        train_result = dpo_trainer.train(resume_from_checkpoint=resume_from_checkpoint)
        
        # Save the final model
        print("💾 Saving model...")
        dpo_trainer.save_model()
        dpo_trainer.save_state()
        
        # Save training metrics
        metrics = train_result.metrics
        metrics_file = output_dir / "dpo_train_results.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)
        
        # Save comprehensive metadata with eval subset info
        self._save_training_metadata(output_dir, train_data, eval_data, training_args)
        
        print(f"✅ DPO training completed!")
        print(f"📉 Final train loss: {metrics.get('train_loss', 'N/A'):.4f}")
        
        return dpo_trainer, metrics
    
    def _save_training_metadata(self, output_dir: Path, train_data: List[Dict], 
                               eval_data: List[Dict], training_args):
        """Save comprehensive training metadata for reproducibility."""
        
        metadata = {
            "training_config": {
                "model_name": self.model_name,
                "sft_model_path": self.sft_model_path,
                "use_4bit": self.use_4bit,
                "use_flash_attention": self.use_flash_attention,
                "training_args": training_args.to_dict(),
                "tokenizer_padding_side": self.tokenizer.padding_side,
            },
            "reproducibility": {
                "seed": self.seed,
                "torch_version": torch.__version__,
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                "device": str(self.device),
            },
            "dataset_info": {
                "train_size": len(train_data),
                "eval_size": len(eval_data),
                "train_sections": len(set(d['source_section'] for d in train_data)),
                "eval_sections": len(set(d['source_section'] for d in eval_data)),
                "eval_pair_ids_used": list(self.eval_pair_ids) if self.eval_pair_ids else None,
                "eval_subset_file": "split_metadata.json"
            },
            "fixes_applied": {
                "canonical_citation_patterns": True,
                "persistent_eval_subset": True,
                "enhanced_similarity_metric": True,
                "tokenizer_padding_fix": True,
                "safety_vs_vagueness_separation": True
            },
            "timestamp": datetime.now().isoformat(),
            "features": {
                "qlora_baseline": True,
                "comprehensive_metrics": True,
                "deterministic_training": True,
                "checkpointing": True,
            }
        }
        
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
    
    def evaluate_dpo_model(self, trainer, eval_data: List[Dict], output_dir: Path):
        """Comprehensive evaluation of the DPO trained model."""
        print("🔍 Running comprehensive DPO evaluation...")
        
        # Standard evaluation
        eval_results = trainer.evaluate()
        
        # Enhanced metrics evaluation with fixed eval subset
        enhanced_metrics = self._compute_enhanced_metrics(eval_data, output_dir)
        
        # Combine results
        combined_results = {**eval_results, **enhanced_metrics}
        
        # Save evaluation results
        with open(output_dir / "dpo_eval_results.json", "w") as f:
            json.dump(combined_results, f, indent=2)
        
        # Generate evaluation report
        self._generate_evaluation_report(combined_results, output_dir)
        
        return combined_results
    
    def _compute_enhanced_metrics(self, eval_data: List[Dict], output_dir: Path) -> Dict[str, Any]:
        """Compute enhanced metrics with fixed eval subset and groundedness-aware similarity."""
        print("📊 Computing enhanced metrics...")
        
        # Use fixed eval subset if available
        if self.eval_pair_ids:
            # Filter to fixed eval subset with integrity checking
            available_pair_ids = {d.get('pair_id') for d in eval_data if d.get('pair_id')}
            missing_pair_ids = self.eval_pair_ids - available_pair_ids
            
            if missing_pair_ids:
                print(f"⚠️ Eval pair ID integrity warning: {len(missing_pair_ids)} expected pair_ids not found in eval data")
                print(f"   Missing IDs: {sorted(list(missing_pair_ids))[:5]}{'...' if len(missing_pair_ids) > 5 else ''}")
            
            sample_data = [d for d in eval_data if d.get('pair_id') in self.eval_pair_ids][:30]
            print(f"📊 Using fixed eval subset: {len(sample_data)}/{len(self.eval_pair_ids)} pairs found (expected all)")
        else:
            sample_data = eval_data[:30]  # Use first 30 for detailed evaluation (increased for stability)
            print(f"📊 Using first 30 examples for evaluation (no fixed subset available)")
        
        metrics = {
            "pairwise_win_rate": 0.0,
            "citation_exact_match": 0.0,
            "citation_iou": 0.0,
            "safety_violations": 0.0,
            "vagueness_violations": 0.0,  # Separate from safety
            "total_evaluated": len(sample_data),
            "used_fixed_eval_subset": self.eval_pair_ids is not None
        }
        
        # Add eval subset integrity information for auditing
        if self.eval_pair_ids:
            metrics["eval_subset_integrity"] = {
                "expected_pairs": len(self.eval_pair_ids),
                "found_pairs": len(sample_data),
                "missing_pairs": len(missing_pair_ids),
                "missing_pair_ids": sorted(list(missing_pair_ids)) if missing_pair_ids else [],
                "integrity_ok": len(missing_pair_ids) == 0
            }
        
        wins = 0
        citation_em_sum = 0
        citation_iou_sum = 0
        safety_violations = 0
        vagueness_violations = 0
        
        for i, example in enumerate(sample_data):
            try:
                # Generate model response
                model_response = self._generate_response(example['prompt'])
                
                # FIXED: Enhanced similarity with groundedness-aware scoring
                gold_sections = {example.get('source_section', '')}
                
                chosen_similarity = compute_enhanced_similarity(
                    model_response, example['chosen'], self.validator, gold_sections
                )
                rejected_similarity = compute_enhanced_similarity(
                    model_response, example['rejected'], self.validator, gold_sections
                )
                
                if chosen_similarity > rejected_similarity:
                    wins += 1
                
                # Citation metrics using canonical patterns
                source_sections = {example.get('source_section', '')}
                citation_em, citation_iou = self.validator.compute_citation_metrics(
                    self.validator.extract_section_ids(model_response), 
                    source_sections
                )
                citation_em_sum += citation_em
                citation_iou_sum += citation_iou
                
                # FIXED: Separate safety and vagueness evaluation
                safety_result = self.judge.evaluate_safety_and_vagueness(model_response)
                if safety_result["safety_violation"]:
                    safety_violations += 1
                if safety_result["vagueness_violation"]:
                    vagueness_violations += 1
                
            except Exception as e:
                logger.warning(f"Evaluation failed for example {i}: {e}")
                continue
        
        # Calculate final metrics
        if len(sample_data) > 0:
            metrics["pairwise_win_rate"] = wins / len(sample_data)
            metrics["citation_exact_match"] = citation_em_sum / len(sample_data)
            metrics["citation_iou"] = citation_iou_sum / len(sample_data)
            metrics["safety_violations"] = safety_violations / len(sample_data)
            metrics["vagueness_violations"] = vagueness_violations / len(sample_data)
        
        print(f"🏆 Enhanced pairwise win-rate: {metrics['pairwise_win_rate']:.3f}")
        print(f"🎯 Citation EM: {metrics['citation_exact_match']:.3f}")
        print(f"📊 Citation IoU: {metrics['citation_iou']:.3f}")
        print(f"🚨 Safety violations: {metrics['safety_violations']:.3f}")
        print(f"💭 Vagueness violations: {metrics['vagueness_violations']:.3f}")
        
        # Save eval integrity report for auditing if using fixed subset
        if self.eval_pair_ids and "eval_subset_integrity" in metrics:
            integrity_file = output_dir / "dpo_eval_integrity.json"
            with open(integrity_file, 'w') as f:
                json.dump(metrics["eval_subset_integrity"], f, indent=2)
            print(f"🔍 Eval integrity report saved: {integrity_file}")
        
        return metrics
    
    def _generate_response(self, prompt: str, max_new_tokens: int = 150) -> str:
        """Generate response using the trained model with inference padding."""
        # Temporarily switch to left padding for inference
        original_padding = self.tokenizer.padding_side
        self.tokenizer.padding_side = "left"
        
        try:
            formatted_prompt = self._format_prompt(prompt)
            inputs = self.tokenizer.encode(formatted_prompt, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=max_new_tokens,
                    temperature=0.1,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )
            
            response = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
            return response.strip()
        
        finally:
            # Restore original padding
            self.tokenizer.padding_side = original_padding
    
    def _generate_evaluation_report(self, results: Dict[str, Any], output_dir: Path):
        """Generate comprehensive evaluation report."""
        
        report = f"""
# Fixed DPO Training Evaluation Report

## Training Results
- **Final Train Loss**: {results.get('train_loss', 'N/A'):.4f}
- **Final Eval Loss**: {results.get('eval_loss', 'N/A'):.4f}
- **Tokenizer Padding**: Right (training), Left (inference)

## Enhanced Metrics
- **Enhanced Pairwise Win-Rate**: {results.get('pairwise_win_rate', 0):.1%}
- **Citation Exact Match**: {results.get('citation_exact_match', 0):.3f}
- **Citation IoU**: {results.get('citation_iou', 0):.3f}
- **Safety Violations**: {results.get('safety_violations', 0):.1%}
- **Vagueness Violations**: {results.get('vagueness_violations', 0):.1%}
- **Used Fixed Eval Subset**: {results.get('used_fixed_eval_subset', False)}

## Fixes Applied
- ✅ Canonical citation patterns (EA-YYYY-NNN[L]*[(N)])
- ✅ Persistent eval subset for consistent evaluation
- ✅ Enhanced similarity metric (0.5×EM + 0.3×IoU + 0.2×F1)
- ✅ Tokenizer padding fix (right for training)
- ✅ Safety vs vagueness separation

## Recommendations
"""
        
        # Add recommendations based on metrics
        win_rate = results.get('pairwise_win_rate', 0)
        if win_rate < 0.6:
            report += "- ⚠️ Low win-rate: Consider additional SFT training or longer DPO training\n"
        elif win_rate > 0.8:
            report += "- ✅ Excellent win-rate: Model shows strong preference alignment\n"
        
        citation_em = results.get('citation_exact_match', 0)
        if citation_em < 0.5:
            report += "- ⚠️ Low citation accuracy: Add more citation examples to training data\n"
        elif citation_em > 0.8:
            report += "- ✅ Strong citation performance: Model properly grounds responses\n"
        
        safety_rate = results.get('safety_violations', 0)
        if safety_rate > 0.1:
            report += "- ⚠️ Safety concerns: Review and strengthen safety training data\n"
        else:
            report += "- ✅ Good safety performance: Low violation rate\n"
        
        report += f"\n*Report generated: {datetime.now().isoformat()}*\n"
        
        with open(output_dir / "evaluation_report.md", "w") as f:
            f.write(report)
        
        print(f"📋 Evaluation report saved: {output_dir / 'evaluation_report.md'}")
    
    def plot_training_curves(self, output_dir: Path):
        """Plot enhanced training curves with DPO metrics including citation history."""
        try:
            # Load training logs
            log_file = output_dir / "trainer_state.json"
            if not log_file.exists():
                print("⚠️ Training logs not found for plotting")
                return
            
            with open(log_file, 'r') as f:
                trainer_state = json.load(f)
            
            log_history = trainer_state.get('log_history', [])
            
            # Load citation metrics history if available
            citation_metrics_file = output_dir / "dpo_metrics.json"
            citation_history = []
            if citation_metrics_file.exists():
                with open(citation_metrics_file, 'r') as f:
                    citation_history = json.load(f)
            
            # Extract metrics
            train_steps, train_losses = [], []
            eval_steps, eval_losses = [], []
            win_rates = []
            citation_em_values = []
            citation_iou_values = []
            
            for log in log_history:
                if 'train_loss' in log:
                    train_steps.append(log['step'])
                    train_losses.append(log['train_loss'])
                if 'eval_loss' in log:
                    eval_steps.append(log['step'])
                    eval_losses.append(log['eval_loss'])
                if 'eval_pairwise_win_rate' in log:
                    win_rates.append(log['eval_pairwise_win_rate'])
            
            # Extract citation metrics from history
            if citation_history:
                for entry in citation_history:
                    citation_em_values.append(entry.get('citation_em', 0))
                    citation_iou_values.append(entry.get('citation_iou', 0))
            
            # Create comprehensive plot with citation metrics
            fig, axes = plt.subplots(2, 3, figsize=(18, 10))
            fig.suptitle('Fixed DPO Training Curves with Citation Metrics', fontsize=16)
            
            # Training loss
            if train_losses:
                axes[0, 0].plot(train_steps, train_losses, 'b-', label='DPO Training Loss')
                axes[0, 0].set_xlabel('Steps')
                axes[0, 0].set_ylabel('Loss')
                axes[0, 0].set_title('Training Loss')
                axes[0, 0].legend()
                axes[0, 0].grid(True)
            
            # Evaluation loss
            if eval_losses:
                axes[0, 1].plot(eval_steps, eval_losses, 'r-', label='DPO Evaluation Loss')
                axes[0, 1].set_xlabel('Steps')
                axes[0, 1].set_ylabel('Loss')
                axes[0, 1].set_title('Evaluation Loss')
                axes[0, 1].legend()
                axes[0, 1].grid(True)
            
            # Win rate
            if win_rates:
                axes[0, 2].plot(eval_steps[:len(win_rates)], win_rates, 'g-', label='Enhanced Win-Rate')
                axes[0, 2].set_xlabel('Steps')
                axes[0, 2].set_ylabel('Win Rate')
                axes[0, 2].set_title('Enhanced Pairwise Win-Rate vs Steps')
                axes[0, 2].legend()
                axes[0, 2].grid(True)
            
            # Citation EM
            if citation_em_values:
                steps_for_citations = eval_steps[:len(citation_em_values)]
                axes[1, 0].plot(steps_for_citations, citation_em_values, 'm-', label='Citation EM')
                axes[1, 0].set_xlabel('Steps')
                axes[1, 0].set_ylabel('Citation EM')
                axes[1, 0].set_title('Citation Exact Match vs Steps')
                axes[1, 0].legend()
                axes[1, 0].grid(True)
            
            # Citation IoU
            if citation_iou_values:
                steps_for_citations = eval_steps[:len(citation_iou_values)]
                axes[1, 1].plot(steps_for_citations, citation_iou_values, 'c-', label='Citation IoU')
                axes[1, 1].set_xlabel('Steps')
                axes[1, 1].set_ylabel('Citation IoU')
                axes[1, 1].set_title('Citation IoU vs Steps')
                axes[1, 1].legend()
                axes[1, 1].grid(True)
            
            # Combined metrics
            if win_rates and citation_em_values:
                axes[1, 2].plot(eval_steps[:len(win_rates)], win_rates, 'g-', label='Win Rate', alpha=0.7)
                if citation_em_values:
                    steps_for_citations = eval_steps[:len(citation_em_values)]
                    axes[1, 2].plot(steps_for_citations, citation_em_values, 'm-', label='Citation EM', alpha=0.7)
                axes[1, 2].set_xlabel('Steps')
                axes[1, 2].set_ylabel('Score')
                axes[1, 2].set_title('Combined Metrics')
                axes[1, 2].legend()
                axes[1, 2].grid(True)
            
            plt.tight_layout()
            plt.savefig(output_dir / "dpo_training_curves.png", dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"📈 Enhanced training curves saved: {output_dir / 'dpo_training_curves.png'}")
            
        except Exception as e:
            print(f"⚠️ Could not generate training curves: {e}")


class EnhancedEmploymentActJudge:
    """Enhanced judge with canonical patterns and safety/vagueness separation."""
    
    def __init__(self, validator: CanonicalCitationValidator):
        self.validator = validator
    
    def compute_similarity(self, response1: str, response2: str) -> float:
        """Compute enhanced similarity with canonical patterns - DEPRECATED."""
        # This method is now deprecated in favor of compute_enhanced_similarity
        return compute_lexical_f1(response1, response2)
    
    def evaluate_citations(self, response: str, gold_sections: List[str]) -> tuple:
        """Evaluate citation accuracy using canonical patterns."""
        gold_sections_set = set(gold_sections)
        return self.validator.compute_citation_metrics(
            self.validator.extract_section_ids(response),
            gold_sections_set
        )
    
    def evaluate_safety_and_vagueness(self, response: str) -> Dict[str, bool]:
        """Separate safety violations from vagueness issues."""
        response_lower = response.lower()
        
        # True safety violations (policy problems)
        safety_patterns = [
            "cannot provide legal advice", "this is not legal advice",
            "consult a lawyer", "seek legal counsel"
        ]
        safety_violation = any(pattern in response_lower for pattern in safety_patterns)
        
        # Vagueness issues (legitimate but unhelpful deferrals)
        vagueness_patterns = [
            "consult with hr", "check the relevant sections", "different situations may have",
            "please consult", "depends on your situation"
        ]
        vagueness_violation = any(pattern in response_lower for pattern in vagueness_patterns)
        
        return {
            "safety_violation": safety_violation,
            "vagueness_violation": vagueness_violation
        }


class FixedDPOMetricsCallback(TrainerCallback):
    """Fixed callback for tracking DPO metrics with persistent eval subset."""
    
    def __init__(self, trainer_obj, eval_data: List[Dict], output_dir: Path):
        self.trainer_obj = trainer_obj
        
        # Use fixed eval subset if available
        if trainer_obj.eval_pair_ids:
            self.eval_data = [d for d in eval_data if d.get('pair_id') in trainer_obj.eval_pair_ids][:10]
        else:
            self.eval_data = eval_data[:10]  # Sample for efficiency
            
        self.output_dir = output_dir
        self.metrics_history = []
    
    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        """Called during evaluation to compute custom metrics."""
        try:
            # Compute enhanced metrics on fixed subset
            wins = 0
            citation_em_sum = 0
            citation_iou_sum = 0
            
            for example in self.eval_data:
                try:
                    response = self.trainer_obj._generate_response(example['prompt'])
                    gold_sections = {example.get('source_section', '')}
                    
                    # Enhanced similarity
                    chosen_sim = compute_enhanced_similarity(
                        response, example['chosen'], self.trainer_obj.validator, gold_sections
                    )
                    rejected_sim = compute_enhanced_similarity(
                        response, example['rejected'], self.trainer_obj.validator, gold_sections
                    )
                    
                    if chosen_sim > rejected_sim:
                        wins += 1
                    
                    # Citation metrics
                    citation_em, citation_iou = self.trainer_obj.validator.compute_citation_metrics(
                        self.trainer_obj.validator.extract_section_ids(response),
                        gold_sections
                    )
                    citation_em_sum += citation_em
                    citation_iou_sum += citation_iou
                    
                except:
                    continue
            
            win_rate = wins / len(self.eval_data) if self.eval_data else 0.0
            citation_em_avg = citation_em_sum / len(self.eval_data) if self.eval_data else 0.0
            citation_iou_avg = citation_iou_sum / len(self.eval_data) if self.eval_data else 0.0
            
            # Log custom metrics
            if hasattr(state, 'log_history') and state.log_history:
                state.log_history[-1]['eval_pairwise_win_rate'] = win_rate
                state.log_history[-1]['eval_citation_em'] = citation_em_avg
                state.log_history[-1]['eval_citation_iou'] = citation_iou_avg
            
            # Store in history with citation metrics
            self.metrics_history.append({
                'step': state.global_step,
                'win_rate': win_rate,
                'citation_em': citation_em_avg,
                'citation_iou': citation_iou_avg,
                'timestamp': datetime.now().isoformat()
            })
            
            # Save metrics history
            with open(self.output_dir / "dpo_metrics.json", "w") as f:
                json.dump(self.metrics_history, f, indent=2)
                
        except Exception as e:
            logger.warning(f"Custom metrics computation failed: {e}")

    # Explicit no-op to ensure compatibility across transformers versions
    def on_train_begin(self, args, state, control, **kwargs):
        return control


def main():
    parser = argparse.ArgumentParser(description="Fixed DPO Training for Employment Act")
    parser.add_argument('--train-data', required=True, help='Training preference pairs JSONL file')
    parser.add_argument('--eval-data', required=True, help='Evaluation preference pairs JSONL file')
    parser.add_argument('--output-dir', required=True, help='Output directory for model and logs')
    parser.add_argument('--model-name', default="meta-llama/Llama-3.1-8B-Instruct", 
                       help='Base model name')
    parser.add_argument('--sft-model', help='Path to SFT LoRA checkpoint (recommended)')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=2, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.1, help='KL penalty coefficient')
    parser.add_argument('--lora-rank', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=16, help='LoRA alpha')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--resume-from-checkpoint', help='Resume training from checkpoint')
    parser.add_argument('--no-4bit', action='store_true', help='Disable 4-bit quantization')
    parser.add_argument('--no-flash-attention', action='store_true', help='Disable flash attention')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize fixed DPO trainer
    trainer_obj = FixedEmploymentActDPOTrainer(
        model_name=args.model_name,
        sft_model_path=args.sft_model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        use_4bit=not args.no_4bit,
        use_flash_attention=not args.no_flash_attention,
        seed=args.seed
    )
    
    # Load preference datasets
    train_data, eval_data = trainer_obj.load_preference_data(
        Path(args.train_data), 
        Path(args.eval_data)
    )
    
    # Train the DPO model
    trainer, metrics = trainer_obj.train_dpo(
        train_data=train_data,
        eval_data=eval_data,
        output_dir=output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        beta=args.beta,
        resume_from_checkpoint=args.resume_from_checkpoint
    )
    
    # Comprehensive evaluation
    eval_results = trainer_obj.evaluate_dpo_model(trainer, eval_data, output_dir)
    
    # Plot enhanced training curves
    trainer_obj.plot_training_curves(output_dir)
    
    print(f"\n🎉 Fixed DPO training complete!")
    print(f"📁 Model saved to: {output_dir}")
    print(f"📉 Final eval loss: {eval_results.get('eval_loss', 'N/A'):.4f}")
    print(f"🏆 Enhanced win-rate: {eval_results.get('pairwise_win_rate', 0):.1%}")
    print(f"🎯 Citation EM: {eval_results.get('citation_exact_match', 0):.3f}")
    print(f"⚡ Fixed features: Canonical patterns, persistent eval, enhanced similarity")


if __name__ == "__main__":
    main()
