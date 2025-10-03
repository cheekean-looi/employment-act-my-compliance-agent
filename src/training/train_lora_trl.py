#!/usr/bin/env python3
"""
Optional TRL SFTTrainer path for easier auditing and maintenance.
Uses TRL's SFTTrainer with formatting function for cleaner label masking.
"""

import json
import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse

# ML imports
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    BitsAndBytesConfig,
    set_seed,
    EarlyStoppingCallback
)

# Import TensorBoardCallback conditionally
try:
    from transformers import TensorBoardCallback
except ImportError:
    # Fallback for older transformers versions
    try:
        from transformers.integrations import TensorBoardCallback
    except ImportError:
        TensorBoardCallback = None
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    prepare_model_for_kbit_training,
)
from trl import SFTTrainer

try:
    from .eval_utils import load_stable_eval_subset
    from .train_lora import QLoRAConfig, CitationEvaluationCallback, CitationEvaluator
except ImportError:
    # For standalone execution
    import sys
    sys.path.append(str(Path(__file__).parent))
    from eval_utils import load_stable_eval_subset
    from train_lora import QLoRAConfig, CitationEvaluationCallback, CitationEvaluator


class TRLSFTTrainer_Production:
    """TRL SFTTrainer-based trainer with formatting function for cleaner masking."""
    
    def __init__(self, config: QLoRAConfig):
        self.config = config
        # Prefer CUDA → MPS → CPU
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")
        
        # Set seeds for reproducibility
        set_seed(config.seed)
        
        print(f"🚀 Initializing TRL SFTTrainer Production")
        print(f"   Device: {self.device}")
        print(f"   Model: {config.model_name}")
        print(f"   Using TRL SFTTrainer with formatting function")
        
        # Initialize model and tokenizer
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        self.model = self._setup_lora()
        
        print(f"✅ Model initialized with {self.model.num_parameters():,} total parameters")
        print(f"   Trainable parameters: {self.model.num_parameters(only_trainable=True):,}")
    
    def _load_tokenizer(self) -> AutoTokenizer:
        """Load and configure tokenizer."""
        print("📝 Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            trust_remote_code=True,
            use_fast=True
        )
        
        # Configure padding
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        tokenizer.padding_side = "right"
        
        return tokenizer
    
    def _load_model(self) -> AutoModelForCausalLM:
        """Load model with quantization configuration."""
        print("🤖 Loading model...")
        
        # Configure quantization
        if self.config.use_4bit and torch.cuda.is_available():
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type=self.config.bnb_4bit_quant_type,
                bnb_4bit_compute_dtype=getattr(torch, self.config.bnb_4bit_compute_dtype),
                bnb_4bit_use_double_quant=self.config.bnb_4bit_use_double_quant,
            )
        else:
            quantization_config = None
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            quantization_config=quantization_config,
            device_map="auto" if torch.cuda.is_available() else None,
            trust_remote_code=True,
            torch_dtype=torch.bfloat16 if self.config.bf16 else torch.float16,
            attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
        )
        # Move to MPS explicitly if selected
        if self.device.type == "mps":
            model.to(self.device)
        
        # Prepare for k-bit training if using quantization
        if quantization_config:
            model = prepare_model_for_kbit_training(model)
        
        return model
    
    def _setup_lora(self) -> AutoModelForCausalLM:
        """Configure and apply LoRA."""
        print("🔗 Setting up LoRA...")
        
        # Target modules for different model architectures
        if "llama" in self.config.model_name.lower():
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "qwen" in self.config.model_name.lower():
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        elif "mistral" in self.config.model_name.lower():
            target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
        else:
            target_modules = ["q_proj", "v_proj"]
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias=self.config.lora_bias,
            target_modules=target_modules,
        )
        
        model = get_peft_model(self.model, lora_config)
        
        return model
    
    def _formatting_func(self, example):
        """
        Clean formatting function for TRL SFTTrainer.
        
        This is much simpler than custom masking - TRL handles the label masking
        automatically when you provide properly formatted chat templates.
        """
        system_prompt = "You are an expert on Malaysia Employment Act. Provide accurate answers with proper section citations."
        
        # Create conversation
        conversation = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": example["instruction"]},
            {"role": "assistant", "content": example["output"]}
        ]
        
        # Apply chat template
        text = self.tokenizer.apply_chat_template(
            conversation,
            tokenize=False,
            add_generation_prompt=False
        )
        
        return text
    
    def train(self, train_path: Path, eval_path: Path, output_dir: Path):
        """Train the model using TRL SFTTrainer."""
        print(f"🏋️ Starting TRL SFTTrainer training...")
        
        # Load datasets
        train_dataset = load_dataset('json', data_files=str(train_path), split='train')
        eval_dataset = load_dataset('json', data_files=str(eval_path), split='train')
        
        print(f"   Train examples: {len(train_dataset)}")
        print(f"   Eval examples: {len(eval_dataset)}")
        
        # Choose optimizer based on quantization
        optim = "adamw_torch"
        if self.config.use_4bit and torch.cuda.is_available():
            optim = "adamw_bnb_8bit"
            print(f"🔧 Using adamw_bnb_8bit optimizer for 4-bit quantization")
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=self.config.num_epochs,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_eval_batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            learning_rate=self.config.learning_rate,
            lr_scheduler_type=self.config.lr_scheduler_type,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=self.config.weight_decay,
            max_grad_norm=self.config.max_grad_norm,
            logging_steps=self.config.logging_steps,
            eval_steps=self.config.eval_steps,
            save_steps=self.config.save_steps,
            evaluation_strategy=self.config.eval_strategy,
            save_strategy=self.config.save_strategy,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=False,  # For loss
            bf16=self.config.bf16,
            fp16=self.config.fp16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            remove_unused_columns=False,
            seed=self.config.seed,
            optim=optim,
            save_total_limit=3,
            report_to=self.config.report_to,  # Use config report_to for consistency
        )
        
        # Load stable eval subset for consistent evaluation
        eval_subset_path = output_dir / "eval_subset.jsonl"
        eval_subset_examples = load_stable_eval_subset(eval_subset_path, fallback_examples=eval_dataset)
        
        # Setup citation evaluation callback
        valid_section_ids = set()
        for example in eval_subset_examples:
            if 'citations' in example:
                valid_section_ids.update(example['citations'])
        
        evaluator = CitationEvaluator(self.tokenizer, self.model, valid_section_ids)
        citation_callback = CitationEvaluationCallback(evaluator, eval_subset_examples, output_dir)
        
        # Early stopping callback
        early_stopping = EarlyStoppingCallback(
            early_stopping_patience=self.config.early_stopping_patience,
            early_stopping_threshold=0.001
        )
        
        # Prepare callbacks list
        callbacks = [citation_callback, early_stopping]
        
        # Add TensorBoard callback if tensorboard logging is enabled
        if self.config.report_to and "tensorboard" in self.config.report_to:
            if TensorBoardCallback is not None:
                tensorboard_callback = TensorBoardCallback()
                callbacks.append(tensorboard_callback)
                print("📊 Added TensorBoard callback for local logging")
            else:
                print("⚠️ TensorBoard callback not available in this transformers version")
        
        # TRL SFTTrainer - much simpler than custom trainer
        trainer = SFTTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            formatting_func=self._formatting_func,
            tokenizer=self.tokenizer,
            max_seq_length=self.config.max_length,
            packing=False,  # Disable packing for cleaner debugging
            callbacks=callbacks,
        )
        
        # Train
        print("🚀 Starting TRL SFTTrainer training...")
        train_result = trainer.train()
        
        # Save model
        print("💾 Saving model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training results
        train_results = {
            "train_runtime": train_result.metrics["train_runtime"],
            "train_samples_per_second": train_result.metrics["train_samples_per_second"],
            "train_loss": train_result.metrics["train_loss"],
            "epoch": train_result.metrics["epoch"],
            "trainer_type": "TRL_SFTTrainer"
        }
        
        with open(output_dir / "train_results.json", 'w') as f:
            json.dump(train_results, f, indent=2)
        
        # Final evaluation
        print("🧪 Running final citation evaluation...")
        eval_result = evaluator.evaluate_examples(eval_subset_examples)
        eval_result.eval_loss = trainer.evaluate()['eval_loss']
        
        # Add dataset-level presence metrics from validation
        try:
            from .validate_dataset import SFTDatasetValidator
            validator = SFTDatasetValidator()
            examples_with_citations, _, individual_presence_rate = validator.validate_citation_presence_in_output(eval_subset_examples)
            
            # Calculate example-level presence rate
            example_level_rate = (examples_with_citations / len(eval_subset_examples) * 100) if eval_subset_examples else 0.0
            
            # Add presence metrics to eval result
            eval_result.citation_presence_rate = example_level_rate
            eval_result.individual_citation_presence_rate = individual_presence_rate
            
            print(f"📊 Added dataset validation metrics: {example_level_rate:.1f}% examples with citations, {individual_presence_rate:.1f}% individual citations present")
        except Exception as e:
            print(f"⚠️ Could not add dataset validation metrics: {e}")
            # Keep default values (0.0)
        
        # Save evaluation results
        eval_results = asdict(eval_result)
        with open(output_dir / "eval_results.json", 'w') as f:
            json.dump(eval_results, f, indent=2)
        
        # Check for TensorBoard run directory
        tensorboard_run_dir = None
        if self.config.report_to and "tensorboard" in self.config.report_to:
            # TensorBoard typically creates runs/[timestamp] directory
            runs_dir = output_dir / "runs"
            if runs_dir.exists():
                # Get the most recent run directory
                run_dirs = [d for d in runs_dir.iterdir() if d.is_dir()]
                if run_dirs:
                    tensorboard_run_dir = str(max(run_dirs, key=lambda x: x.stat().st_mtime).relative_to(output_dir))
        
        # Save metadata
        metadata = {
            "training_info": {
                "timestamp": datetime.now().isoformat(),
                "trainer_type": "TRL_SFTTrainer",
                "config": asdict(self.config),
                "model_name": self.config.model_name,
                "total_parameters": self.model.num_parameters(),
                "trainable_parameters": self.model.num_parameters(only_trainable=True),
                "logging_backend": self.config.report_to,
                "tensorboard_run_dir": tensorboard_run_dir,
            },
            "results": {
                "train": train_results,
                "eval": eval_results,
            },
            "validation_metrics": {
                "citation_presence_rate": eval_results.get("citation_presence_rate"),
                "individual_citation_presence_rate": eval_results.get("individual_citation_presence_rate"),
                "citation_exact_match": eval_results.get("citation_exact_match"),
                "citation_iou": eval_results.get("citation_iou"),
            },
            "artifacts": {
                "tensorboard_logs": tensorboard_run_dir if tensorboard_run_dir else None,
            },
            "note": "This is the TRL SFTTrainer baseline path for easier auditing"
        }
        
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ TRL SFTTrainer training complete!")
        print(f"   Final train loss: {train_results['train_loss']:.4f}")
        print(f"   Final eval loss: {eval_result.eval_loss:.4f}")
        print(f"   Citation exact match: {eval_result.citation_exact_match:.2%}")
        print(f"📁 Artifacts saved to: {output_dir}")
        
        return trainer, eval_result


def main():
    parser = argparse.ArgumentParser(description="TRL SFTTrainer Production Training")
    
    # Data arguments
    parser.add_argument('--train-data', type=Path, required=True, help='Training JSONL file')
    parser.add_argument('--eval-data', type=Path, required=True, help='Evaluation JSONL file')
    parser.add_argument('--output-dir', type=Path, required=True, help='Output directory')
    
    # Model arguments
    parser.add_argument('--model-name', default="meta-llama/Llama-3.1-8B-Instruct", help='Base model name')
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Create config with Hour 4 defaults
    config = QLoRAConfig(
        model_name=args.model_name,
        num_epochs=args.epochs,
        learning_rate=args.learning_rate,
        seed=args.seed,
    )
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize trainer and train
    trainer = TRLSFTTrainer_Production(config)
    trainer.train(args.train_data, args.eval_data, args.output_dir)
    
    print(f"🎉 TRL SFTTrainer training completed successfully!")
    print(f"📁 Artifacts saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
