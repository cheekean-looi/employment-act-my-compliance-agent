#!/usr/bin/env python3
# python src/training/train_dpo.py --train-data outputs/dpo_pairs_train.jsonl --eval-data outputs/dpo_pairs_eval.jsonl --output-dir outputs/lora_dpo --sft-model outputs/lora_sft --model-name microsoft/DialoGPT-medium --epochs 1 --batch-size 2 --learning-rate 5e-5 --beta 0.1
"""
DPO Training Script for Employment Act Malaysia Compliance Agent
Runs Direct Preference Optimization on chosen vs rejected pairs.
"""

import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import DPOTrainer, DPOConfig
from peft import LoraConfig, get_peft_model, TaskType, PeftModel
import argparse
from typing import Dict, List
import matplotlib.pyplot as plt
from datetime import datetime

class EmploymentActDPOTrainer:
    def __init__(self, 
                 model_name: str = "meta-llama/Llama-3.1-8B-Instruct",
                 sft_model_path: str = None,
                 lora_rank: int = 8,
                 lora_alpha: int = 16,
                 lora_dropout: float = 0.1):
        """Initialize DPO trainer with model and configuration."""
        
        self.model_name = model_name
        self.sft_model_path = sft_model_path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"   Initializing Employment Act DPO Trainer")
        print(f"   Device: {self.device}")
        print(f"   Model: {model_name}")
        if sft_model_path:
            print(f"   SFT checkpoint: {sft_model_path}")
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model
        print("Loading base model...")
            
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,  # Use float32 for MPS compatibility
        )
        
        # Load SFT checkpoint if provided (recommended for better DPO performance)
        if sft_model_path and Path(sft_model_path).exists():
            print(f"  Loading SFT checkpoint from {sft_model_path}")
            print("   Starting DPO from SFT policy for better stability...")
            try:
                # Load the SFT LoRA adapter
                self.model = PeftModel.from_pretrained(base_model, sft_model_path)
                print("   Successfully loaded SFT adapter as starting policy")
            except Exception as e:
                print(f"   Could not load SFT checkpoint: {e}")
                print("   Falling back to base model...")
                self.model = base_model
        else:
            print("   No SFT checkpoint provided - starting DPO from base model")
            print("   Tip: Use --sft-model outputs/lora_sft for better results")
            self.model = base_model
        
        # Configure LoRA for DPO (only if not using SFT checkpoint)
        if not (sft_model_path and Path(sft_model_path).exists()):
            print(f"Setting up new LoRA (rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout})")
            
            # Set target modules based on model architecture
            if "qwen" in model_name.lower():
                target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # For Qwen
            elif "dialo" in model_name.lower():
                target_modules = ["c_attn", "c_proj"]  # For DialoGPT
            else:
                target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]  # For Llama
                
            lora_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM,
                r=lora_rank,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                bias="none"
            )
            
            self.model = get_peft_model(self.model, lora_config)
        
        # Print trainable parameters
        if hasattr(self.model, 'print_trainable_parameters'):
            self.model.print_trainable_parameters()
        else:
            total_params = sum(p.numel() for p in self.model.parameters())
            trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            print(f"trainable params: {trainable_params:,} || all params: {total_params:,} || trainable%: {100 * trainable_params / total_params:.4f}")
    
    def _format_prompt(self, prompt: str) -> str:
        """Format prompt using proper chat template."""
        
        if "qwen" in self.model_name.lower() or "llama" in self.model_name.lower():
            # Use instruction-tuned model chat template
            messages = [
                {"role": "system", "content": "You are an expert on Malaysia Employment Act. Provide accurate, helpful answers with proper citations."},
                {"role": "user", "content": prompt}
            ]
            return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        else:
            # Fallback for other models
            return f"Human: {prompt}\nAssistant: "
    
    def load_preference_data(self, train_file: Path, eval_file: Path) -> tuple:
        """Load and prepare preference datasets for DPO."""
        print(f"   Loading preference datasets...")
        print(f"  Train: {train_file}")
        print(f"  Eval:  {eval_file}")
        
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
        
        print(f"  Loaded {len(train_data)} training pairs")
        print(f"  Loaded {len(eval_data)} evaluation pairs")
        
        return train_data, eval_data
    
    def preprocess_preference_data(self, data: List[Dict]) -> Dataset:
        """Preprocess preference data for DPO training."""
        
        def format_dataset(examples):
            formatted_data = {
                'prompt': [],
                'chosen': [],
                'rejected': []
            }
            
            for example in examples:
                # Format prompt
                prompt = self._format_prompt(example['prompt'])
                
                # Add to dataset
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
                  beta: float = 0.1):
        """Train the model with DPO."""
        
        print(f"   Starting DPO training...")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Beta (KL penalty): {beta}")
        print(f"  Output dir: {output_dir}")
        
        # Preprocess datasets
        train_dataset = self.preprocess_preference_data(train_data)
        eval_dataset = self.preprocess_preference_data(eval_data)
        
        # DPO Configuration
        dpo_config = DPOConfig(
            output_dir=str(output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=2,
            optim="adamw_torch",
            save_steps=20,
            logging_steps=5,
            learning_rate=learning_rate,
            weight_decay=0.001,
            fp16=False,  # Disable for MPS compatibility
            bf16=False,
            max_grad_norm=1.0,
            warmup_steps=5,
            lr_scheduler_type="cosine",
            eval_strategy="steps",
            eval_steps=10,
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=[],  # Disable reporting
            beta=beta,  # KL penalty coefficient
            remove_unused_columns=False,
        )
        
        # Initialize DPO trainer
        dpo_trainer = DPOTrainer(
            model=self.model,
            args=dpo_config,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
        )
        
        # Train the model
        print("   Starting DPO training...")
        train_result = dpo_trainer.train()
        
        # Save the final model
        print("   Saving model...")
        dpo_trainer.save_model()
        dpo_trainer.save_state()
        
        # Save training metrics
        metrics = train_result.metrics
        with open(output_dir / "dpo_train_results.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"   DPO training completed!")
        print(f"   Final train loss: {metrics.get('train_loss', 'N/A'):.4f}")
        
        return dpo_trainer, metrics
    
    def evaluate_dpo_model(self, trainer, eval_data: List[Dict], output_dir: Path):
        """Evaluate the DPO trained model."""
        print("   Evaluating DPO model...")
        
        # Run evaluation
        eval_results = trainer.evaluate()
        
        # Save evaluation results
        with open(output_dir / "dpo_eval_results.json", "w") as f:
            json.dump(eval_results, f, indent=2)
        
        # Test on sample examples
        print("\\n   Testing on sample preference pairs:")
        test_examples = eval_data[:3]  # First 3 examples
        
        for i, example in enumerate(test_examples):
            prompt = self._format_prompt(example['prompt'])
            
            print(f"\\n--- Example {i+1} ---")
            print(f"Prompt: {example['prompt']}")
            print(f"Expected preference: Chosen > Rejected")
            print(f"Chosen response: {example['chosen'][:150]}...")
            print(f"Rejected response: {example['rejected'][:150]}...")
        
        return eval_results
    
    def plot_dpo_training_curves(self, output_dir: Path):
        """Plot DPO training loss curves."""
        try:
            # Load training logs
            log_file = output_dir / "trainer_state.json"
            if log_file.exists():
                with open(log_file, 'r') as f:
                    trainer_state = json.load(f)
                
                log_history = trainer_state.get('log_history', [])
                
                # Extract training and evaluation losses
                train_steps = []
                train_losses = []
                eval_steps = []
                eval_losses = []
                
                for log in log_history:
                    if 'train_loss' in log:
                        train_steps.append(log['step'])
                        train_losses.append(log['train_loss'])
                    if 'eval_loss' in log:
                        eval_steps.append(log['step'])
                        eval_losses.append(log['eval_loss'])
                
                # Create plot
                plt.figure(figsize=(12, 6))
                
                if train_losses:
                    plt.subplot(1, 2, 1)
                    plt.plot(train_steps, train_losses, label='DPO Training Loss', color='blue')
                    plt.xlabel('Steps')
                    plt.ylabel('Loss')
                    plt.title('DPO Training Loss')
                    plt.legend()
                    plt.grid(True)
                
                if eval_losses:
                    plt.subplot(1, 2, 2)
                    plt.plot(eval_steps, eval_losses, label='DPO Evaluation Loss', color='red')
                    plt.xlabel('Steps')
                    plt.ylabel('Loss')
                    plt.title('DPO Evaluation Loss')
                    plt.legend()
                    plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(output_dir / "dpo_training_curves.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"   DPO training curves saved to {output_dir / 'dpo_training_curves.png'}")
                
        except Exception as e:
            print(f"   Could not generate DPO training curves: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train Employment Act DPO model")
    parser.add_argument('--train-data', required=True, help='Training preference pairs JSONL file')
    parser.add_argument('--eval-data', required=True, help='Evaluation preference pairs JSONL file')
    parser.add_argument('--output-dir', required=True, help='Output directory for model and logs')
    parser.add_argument('--model-name', default="Qwen/Qwen2.5-1.5B-Instruct", 
                       help='Base model name')
    parser.add_argument('--sft-model', help='Path to SFT LoRA checkpoint (recommended for better DPO performance)')
    parser.add_argument('--epochs', type=int, default=1, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=2, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=5e-5, help='Learning rate')
    parser.add_argument('--beta', type=float, default=0.1, help='KL penalty coefficient')
    parser.add_argument('--lora-rank', type=int, default=8, help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=16, help='LoRA alpha')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize DPO trainer
    trainer_obj = EmploymentActDPOTrainer(
        model_name=args.model_name,
        sft_model_path=args.sft_model,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha
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
        beta=args.beta
    )
    
    # Evaluate the model
    eval_results = trainer_obj.evaluate_dpo_model(trainer, eval_data, output_dir)
    
    # Plot training curves
    trainer_obj.plot_dpo_training_curves(output_dir)
    
    print(f"\\n   DPO training complete!")
    print(f"   Model saved to: {output_dir}")
    print(f"   Final eval loss: {eval_results.get('eval_loss', 'N/A'):.4f}")


if __name__ == "__main__":
    main()