#!/usr/bin/env python3
# python src/training/train_lora.py --train-data outputs/sft_dataset_train.jsonl --eval-data outputs/sft_dataset_eval.jsonl \utput-dir outputs/lora_sft --model-name "microsoft/DialoGPT-medium" --epochs 3 --batch-size 1 --learning-rate 2e-4 --lora-rank 16 --lora-alpha 32
"""
QLoRA Training Script for Employment Act Malaysia Compliance Agent
Trains Llama-3.1-8B-Instruct with LoRA adapters on our SFT dataset.
"""

import json
import torch
import wandb
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType
import argparse
from typing import Dict, List
import matplotlib.pyplot as plt
import pandas as pd

class EmploymentActTrainer:
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-medium",
                 lora_rank: int = 16,
                 lora_alpha: int = 32,
                 lora_dropout: float = 0.1):
        """Initialize trainer with model and LoRA configuration."""
        
        self.model_name = model_name
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        print(f"Initializing Employment Act QLoRA Trainer")
        print(f"Device: {self.device}")
        print(f"Model: {model_name}")
        
        # Load tokenizer
        print("Loading tokenizer...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model with 4-bit quantization
        print("Loading model with quantization...")
        
        # Use public model for testing
        if "llama" in model_name.lower():
            print("Llama models require authentication. Using DialoGPT instead.")
            model_name = "microsoft/DialoGPT-medium"
            self.model_name = model_name
            
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=torch.float32,  # Use float32 for MPS compatibility
            device_map="auto"
        )
        
        # Configure LoRA
        print(f"Setting up LoRA (rank={lora_rank}, alpha={lora_alpha}, dropout={lora_dropout})")
        
        # Set target modules based on model architecture
        if "dialo" in model_name.lower():
            target_modules = ["c_attn", "c_proj"]  # For DialoGPT
        else:
            target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]  # For Llama
            
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=lora_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none"
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def _format_prompt(self, instruction: str, input_text: str = "", output: str = "") -> str:
        """Format prompt based on model type."""
        
        if "dialo" in self.model_name.lower():
            # Simple format for DialoGPT
            if input_text.strip():
                prompt = f"Human: {instruction}\n{input_text}\nAssistant: "
            else:
                prompt = f"Human: {instruction}\nAssistant: "
            
            if output:
                prompt += output
                
        else:
            # Llama format
            if input_text.strip():
                prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an expert on Malaysia Employment Act. Provide accurate, helpful answers with proper citations.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{instruction}\n\n{input_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            else:
                prompt = f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are an expert on Malaysia Employment Act. Provide accurate, helpful answers with proper citations.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
            
            if output:
                prompt += output + "<|eot_id|>"
        
        return prompt
    
    def load_dataset(self, train_file: Path, eval_file: Path) -> tuple:
        """Load and prepare training datasets."""
        print(f"📚 Loading datasets...")
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
        
        print(f"  Loaded {len(train_data)} training examples")
        print(f"  Loaded {len(eval_data)} evaluation examples")
        
        return train_data, eval_data
    
    def preprocess_data(self, data: List[Dict], max_length: int = 1024) -> Dataset:
        """Preprocess data for training."""
        
        def tokenize_function(examples):
            prompts = []
            for instruction, input_text, output in zip(
                examples['instruction'], 
                examples['input'], 
                examples['output']
            ):
                prompt = self._format_prompt(instruction, input_text, output)
                prompts.append(prompt)
            
            # Tokenize
            tokenized = self.tokenizer(
                prompts,
                truncation=True,
                padding=False,
                max_length=max_length,
                return_tensors=None
            )
            
            # Set labels for language modeling
            tokenized["labels"] = tokenized["input_ids"].copy()
            
            return tokenized
        
        # Convert to HuggingFace dataset
        dataset_dict = {
            'instruction': [ex['instruction'] for ex in data],
            'input': [ex.get('input', '') for ex in data],
            'output': [ex['output'] for ex in data],
            'citations': [ex.get('citations', []) for ex in data]
        }
        
        dataset = Dataset.from_dict(dataset_dict)
        tokenized_dataset = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def train(self, 
              train_data: List[Dict], 
              eval_data: List[Dict],
              output_dir: Path,
              num_epochs: int = 3,
              batch_size: int = 4,
              learning_rate: float = 2e-4,
              eval_steps: int = 50,
              save_steps: int = 100,
              warmup_steps: int = 10,
              gradient_checkpointing: bool = True):
        """Train the model with QLoRA."""
        
        print(f"  Starting training...")
        print(f"  Epochs: {num_epochs}")
        print(f"  Batch size: {batch_size}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Output dir: {output_dir}")
        
        # Preprocess datasets
        train_dataset = self.preprocess_data(train_data)
        eval_dataset = self.preprocess_data(eval_data)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=str(output_dir),
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,
            optim="adamw_torch",
            save_steps=save_steps,
            logging_steps=10,
            learning_rate=learning_rate,
            weight_decay=0.001,
            fp16=False,  # Disable for MPS compatibility
            bf16=False,
            max_grad_norm=1.0,
            warmup_steps=warmup_steps,
            group_by_length=True,
            lr_scheduler_type="cosine",
            eval_strategy="steps",
            eval_steps=eval_steps,
            save_strategy="steps",
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            report_to=[],  # Disable all reporting
            gradient_checkpointing=gradient_checkpointing,
            dataloader_pin_memory=False,
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=self.tokenizer,
            data_collator=data_collator,
        )
        
        # Train the model
        print("   Starting training...")
        train_result = trainer.train()
        
        # Save the final model
        print("   Saving model...")
        trainer.save_model()
        trainer.save_state()
        
        # Save training metrics
        metrics = train_result.metrics
        with open(output_dir / "train_results.json", "w") as f:
            json.dump(metrics, f, indent=2)
        
        print(f"   Training completed!")
        print(f"   Final train loss: {metrics.get('train_loss', 'N/A'):.4f}")
        
        return trainer, metrics
    
    def evaluate_model(self, trainer, eval_data: List[Dict], output_dir: Path):
        """Evaluate the trained model on held-out samples."""
        print("   Evaluating model...")
        
        # Run evaluation
        eval_results = trainer.evaluate()
        
        # Save evaluation results
        with open(output_dir / "eval_results.json", "w") as f:
            json.dump(eval_results, f, indent=2)
        
        # Test on a few examples
        print("\n   Testing on sample examples:")
        test_examples = eval_data[:5]  # First 5 examples
        
        for i, example in enumerate(test_examples):
            prompt = self._format_prompt(example['instruction'], example.get('input', ''))
            
            # Tokenize input
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            
            # Generate response
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_new_tokens=200,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode response
            response = self.tokenizer.decode(outputs[0][len(inputs[0]):], skip_special_tokens=True)
            
            print(f"\n--- Example {i+1} ---")
            print(f"Question: {example['instruction']}")
            print(f"Expected citations: {example.get('citations', [])}")
            print(f"Generated: {response[:200]}...")
        
        return eval_results
    
    def plot_training_curves(self, output_dir: Path):
        """Plot training loss curves."""
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
                    plt.plot(train_steps, train_losses, label='Training Loss', color='blue')
                    plt.xlabel('Steps')
                    plt.ylabel('Loss')
                    plt.title('Training Loss')
                    plt.legend()
                    plt.grid(True)
                
                if eval_losses:
                    plt.subplot(1, 2, 2)
                    plt.plot(eval_steps, eval_losses, label='Evaluation Loss', color='red')
                    plt.xlabel('Steps')
                    plt.ylabel('Loss')
                    plt.title('Evaluation Loss')
                    plt.legend()
                    plt.grid(True)
                
                plt.tight_layout()
                plt.savefig(output_dir / "training_curves.png", dpi=300, bbox_inches='tight')
                plt.close()
                
                print(f"📈 Training curves saved to {output_dir / 'training_curves.png'}")
                
        except Exception as e:
            print(f"⚠️ Could not generate training curves: {e}")


def main():
    parser = argparse.ArgumentParser(description="Train Employment Act QLoRA model")
    parser.add_argument('--train-data', required=True, help='Training JSONL file')
    parser.add_argument('--eval-data', required=True, help='Evaluation JSONL file')
    parser.add_argument('--output-dir', required=True, help='Output directory for model and logs')
    parser.add_argument('--model-name', default="microsoft/DialoGPT-medium", 
                       help='Base model name')
    parser.add_argument('--epochs', type=int, default=3, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=4, help='Training batch size')
    parser.add_argument('--learning-rate', type=float, default=2e-4, help='Learning rate')
    parser.add_argument('--lora-rank', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=32, help='LoRA alpha')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize trainer
    trainer_obj = EmploymentActTrainer(
        model_name=args.model_name,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha
    )
    
    # Load datasets
    train_data, eval_data = trainer_obj.load_dataset(
        Path(args.train_data), 
        Path(args.eval_data)
    )
    
    # Train the model
    trainer, metrics = trainer_obj.train(
        train_data=train_data,
        eval_data=eval_data,
        output_dir=output_dir,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate
    )
    
    # Evaluate the model
    eval_results = trainer_obj.evaluate_model(trainer, eval_data, output_dir)
    
    # Plot training curves
    trainer_obj.plot_training_curves(output_dir)
    
    print(f"\n  Training complete!")
    print(f"   Model saved to: {output_dir}")
    print(f"   Final eval loss: {eval_results.get('eval_loss', 'N/A'):.4f}")


if __name__ == "__main__":
    main()