#!/usr/bin/env python3
# python src/training/train_lora_production.py --train-data outputs/sft_dataset/sft_dataset_train.jsonl --eval-data outputs/sft_dataset/sft_dataset_eval.jsonl --output-dir outputs/lora_sft --model-name meta-llama/Llama-3.1-8B-Instruct
"""
Production-grade QLoRA Training Script for Employment Act Malaysia Compliance Agent.
Features: 4-bit quantization, label masking, citation evaluation, comprehensive artifacts.
"""

import json
import os
import re
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import argparse
import gc
import platform
import sys
from packaging import version
from inspect import signature as _sig

# ML imports
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    set_seed
)
from peft import (
    LoraConfig, 
    get_peft_model, 
    TaskType,
    prepare_model_for_kbit_training,
    PeftModel
)
import bitsandbytes as bnb
from trl import SFTTrainer
from transformers.integrations import TensorBoardCallback
from transformers import TrainerCallback, TrainerState, TrainerControl, EarlyStoppingCallback
import transformers as _transformers

try:
    from .schemas import TrainingConfig, EnvironmentInfo, EvaluationMetrics, CitationMetrics
    from .eval_utils import load_stable_eval_subset
except ImportError:
    # For standalone execution
    sys.path.append(str(Path(__file__).parent))
    from schemas import TrainingConfig, EnvironmentInfo, EvaluationMetrics, CitationMetrics
    from eval_utils import load_stable_eval_subset

    # Load environment variables from a .env file if present (repo root or CWD)
try:
    from dotenv import load_dotenv
    # load from CWD; if running from repo root, this picks up .env there
    load_dotenv()
except Exception:
    pass

# Advisory for HF cache deprecation
try:
    if os.environ.get("TRANSFORMERS_CACHE") and not os.environ.get("HF_HOME"):
        print("ℹ️ Detected TRANSFORMERS_CACHE; prefer HF_HOME for newer transformers versions.")
except Exception:
    pass


class CitationEvaluationCallback(TrainerCallback):
    """Callback to evaluate citation metrics during training."""
    
    def __init__(self, evaluator: 'CitationEvaluator', eval_examples: List[Dict], output_dir: Path):
        self.evaluator = evaluator
        self.eval_examples = eval_examples[:30]  # Fixed 30 samples
        self.output_dir = output_dir
        self.citation_history = []
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def on_evaluate(self, args, state: TrainerState, control: TrainerControl, **kwargs):
        """Run citation evaluation after each evaluation."""
        try:
            # Evaluate citation metrics
            eval_result = self.evaluator.evaluate_examples(self.eval_examples)
            
            # Log to citation history
            history_entry = {
                "step": state.global_step,
                "epoch": state.epoch,
                "eval_loss": eval_result.eval_loss,
                "citation_exact_match": eval_result.citation_exact_match,
                "citation_partial_match": eval_result.citation_partial_match,
                "avg_judge_score": eval_result.avg_judge_score,
                "response_length_avg": eval_result.response_length_avg,
                "valid_json_rate": eval_result.valid_json_rate,
                "total_examples": eval_result.total_examples
            }
            
            self.citation_history.append(history_entry)
            
            # Save to JSONL file
            history_file = self.output_dir / "citation_eval_history.jsonl"
            with open(history_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(history_entry) + '\n')
            
            # Log metrics
            print(f"\n🎯 Citation Eval @ Step {state.global_step}:")
            print(f"   Exact Match: {eval_result.citation_exact_match:.3f}")
            print(f"   Partial Match: {eval_result.citation_partial_match:.3f}")
            print(f"   Judge Score: {eval_result.avg_judge_score:.3f}")
            
        except Exception as e:
            print(f"⚠️ Citation evaluation failed: {e}")


@dataclass
class QLoRAConfig:
    """Configuration for QLoRA training."""
    # Model settings - aligned to spec defaults
    model_name: str = "meta-llama/Llama-3.1-8B-Instruct"  # Spec default
    use_4bit: bool = True
    bnb_4bit_quant_type: str = "nf4"
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_use_double_quant: bool = True
    
    # LoRA settings
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_bias: str = "none"
    
    # Training settings - stability improvements
    num_epochs: int = 2  # Reduced for overfitting prevention
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4  # More conservative LR
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    lr_scheduler_type: str = "cosine"  # Cosine with warmup
    
    # Evaluation and saving
    eval_steps: int = 50
    save_steps: int = 100
    logging_steps: int = 10
    eval_strategy: str = "steps"
    save_strategy: str = "steps"
    load_best_model_at_end: bool = True
    metric_for_best_model: str = "eval_loss"
    early_stopping_patience: int = 3  # Early stopping
    
    # Hardware and precision
    bf16: bool = True
    fp16: bool = False
    gradient_checkpointing: bool = True
    dataloader_num_workers: int = 4
    
    # Tokenization
    max_length: int = 2048
    
    # Modules to save
    modules_to_save: Optional[List[str]] = None  # Can include ["lm_head"] if needed
    
    # Reproducibility
    seed: int = 42
    
    # Reporting
    report_to: Optional[List[str]] = None  # Can be ["wandb"] or ["tensorboard"] or both for multi-backend


@dataclass  
class EvaluationResult:
    """Results from citation evaluation."""
    eval_loss: float
    citation_exact_match: float
    citation_partial_match: float
    citation_precision: float
    citation_recall: float
    citation_f1_score: float
    avg_judge_score: float
    response_length_avg: float
    valid_json_rate: float
    total_examples: int
    # Dataset-level citation presence metrics (from validation)
    citation_presence_rate: float = 0.0
    individual_citation_presence_rate: float = 0.0


class CitationEvaluator:
    """Evaluates model responses for citation accuracy."""
    
    def __init__(self, tokenizer, model, valid_section_ids: set):
        self.tokenizer = tokenizer
        self.model = model
        self.valid_section_ids = valid_section_ids
        
        # Citation extraction pattern
        self.citation_pattern = r'\b(EA-\d{4}-\d+[A-Z]*(?:\(\d+\))?)\b'
    
    def extract_citations(self, text: str) -> List[str]:
        """Extract citation section IDs from text."""
        matches = re.findall(self.citation_pattern, text)
        return list(set(matches))  # Remove duplicates
    
    def compute_citation_match(self, predicted_citations: List[str], 
                             gold_citations: List[str]) -> Tuple[float, float]:
        """Compute exact and partial citation match scores."""
        if not gold_citations:
            return 1.0 if not predicted_citations else 0.0, 1.0
        
        pred_set = set(predicted_citations)
        gold_set = set(gold_citations)
        
        # Exact match: sets are identical
        exact_match = 1.0 if pred_set == gold_set else 0.0
        
        # Partial match: intersection over union
        if not pred_set and not gold_set:
            partial_match = 1.0
        elif not pred_set or not gold_set:
            partial_match = 0.0
        else:
            intersection = len(pred_set & gold_set)
            union = len(pred_set | gold_set)
            partial_match = intersection / union
        
        return exact_match, partial_match
    
    def judge_response_quality(self, response: str, context_text: str) -> float:
        """Lightweight judge scoring for response quality."""
        # Simple heuristic-based scoring (can be replaced with LLM judge)
        score = 0.0
        
        # Length appropriateness (not too short, not too long)
        if 50 <= len(response) <= 500:
            score += 0.3
        
        # Contains specific information (numbers, specific terms)
        if re.search(r'\b\d+\b', response):
            score += 0.2
        
        # Mentions Employment Act or legal terminology
        legal_terms = ['employment act', 'section', 'entitled', 'shall', 'according to']
        if any(term in response.lower() for term in legal_terms):
            score += 0.3
        
        # Not too repetitive
        words = response.lower().split()
        if len(set(words)) / len(words) > 0.6:  # Lexical diversity
            score += 0.2
        
        return min(score, 1.0)
    
    def evaluate_examples(self, eval_examples: List[Dict]) -> EvaluationResult:
        """Evaluate model on held-out examples."""
        print("🧪 Evaluating citation accuracy on held-out examples...")
        
        exact_matches = []
        partial_matches = []
        judge_scores = []
        response_lengths = []
        valid_json_count = 0
        
        for i, example in enumerate(eval_examples[:30]):  # Hour 4 requirement: 30 samples
            instruction = example['instruction']
            gold_citations = example['citations']
            
            # Generate response
            try:
                response = self._generate_response(instruction)
                response_lengths.append(len(response))
                
                # Extract citations from response
                predicted_citations = self.extract_citations(response)
                
                # Compute citation metrics
                exact_match, partial_match = self.compute_citation_match(
                    predicted_citations, gold_citations
                )
                exact_matches.append(exact_match)
                partial_matches.append(partial_match)
                
                # Judge scoring
                judge_score = self.judge_response_quality(response, example.get('context', ''))
                judge_scores.append(judge_score)
                
                # Check if response looks like valid structured output
                if any(term in response.lower() for term in ['section', 'according', 'entitled']):
                    valid_json_count += 1
                
            except Exception as e:
                print(f"⚠️ Error evaluating example {i}: {e}")
                exact_matches.append(0.0)
                partial_matches.append(0.0)
                judge_scores.append(0.0)
                response_lengths.append(0)
        
        # Calculate precision, recall, F1
        precision_scores = []
        recall_scores = []
        
        for exact_match, partial_match in zip(exact_matches, partial_matches):
            # Use partial match as proxy for precision/recall
            precision_scores.append(partial_match)
            recall_scores.append(partial_match)
        
        avg_precision = np.mean(precision_scores)
        avg_recall = np.mean(recall_scores)
        f1_score = 2 * (avg_precision * avg_recall) / (avg_precision + avg_recall) if (avg_precision + avg_recall) > 0 else 0.0
        
        return EvaluationResult(
            eval_loss=0.0,  # Will be filled by trainer
            citation_exact_match=np.mean(exact_matches),
            citation_partial_match=np.mean(partial_matches),
            citation_precision=avg_precision,
            citation_recall=avg_recall,
            citation_f1_score=f1_score,
            avg_judge_score=np.mean(judge_scores),
            response_length_avg=np.mean(response_lengths),
            valid_json_rate=valid_json_count / len(eval_examples[:30]),
            total_examples=len(eval_examples[:30])
        )
    
    def _generate_response(self, instruction: str) -> str:
        """Generate response for a single instruction."""
        # Create chat template
        messages = [
            {"role": "system", "content": "You are an expert on Malaysia Employment Act. Provide accurate answers with proper section citations."},
            {"role": "user", "content": instruction}
        ]
        
        # Apply chat template
        prompt = self.tokenizer.apply_chat_template(
            messages, 
            tokenize=False, 
            add_generation_prompt=True
        )
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=1024
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.1,
                do_sample=True,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        response = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:], 
            skip_special_tokens=True
        )
        
        return response.strip()


class ProductionQLoRATrainer:
    """Production-grade QLoRA trainer with comprehensive evaluation."""
    
    def __init__(self, config: QLoRAConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set seeds for reproducibility
        set_seed(config.seed)
        torch.manual_seed(config.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(config.seed)
        
        print(f"🚀 Initializing Production QLoRA Trainer")
        print(f"   Device: {self.device}")
        print(f"   Model: {config.model_name}")
        print(f"   Precision: {'BF16' if config.bf16 else 'FP16' if config.fp16 else 'FP32'}")
        print(f"   4-bit Quantization: {config.use_4bit}")
        print(f"   Learning Rate: {config.learning_rate}")
        print(f"   Epochs: {config.num_epochs}")
        print(f"   Scheduler: {config.lr_scheduler_type}")

        # Hardware detection and environment info
        self.env_info = self._collect_environment_info()
        self._detect_hardware()
        self._auto_tune_memory_settings()

        # Normalize precision for MPS on older macOS versions
        try:
            if (not torch.cuda.is_available()) and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                mac_ver = platform.mac_ver()[0]
                mac_major = int(mac_ver.split(".")[0]) if mac_ver else 0
                if self.config.bf16 and mac_major < 14:
                    print("⚠️  MPS BF16 unsupported on macOS < 14; switching to FP16.")
                    self.config.bf16 = False
                    self.config.fp16 = True
        except Exception as _e:
            # Non-fatal; default to existing config
            pass
        
        # Initialize model and tokenizer
        self.tokenizer = self._load_tokenizer()
        self.model = self._load_model()
        
        # Configure LoRA
        self.model = self._setup_lora()
        
        # Enable flash attention if available
        self._enable_flash_attention()
        
        print(f"✅ Model initialized with {self.model.num_parameters():,} total parameters")
        print(f"   Trainable parameters: {self.model.num_parameters(only_trainable=True):,}")
        
        trainable_ratio = self.model.num_parameters(only_trainable=True) / self.model.num_parameters() * 100
        print(f"   Trainable ratio: {trainable_ratio:.2f}%")
    
    def _enable_flash_attention(self):
        """Enable flash attention if available."""
        try:
            # Check if flash attention is available and working
            if hasattr(self.model, 'config') and hasattr(self.model.config, 'attn_implementation'):
                if self.model.config.attn_implementation == "flash_attention_2":
                    print("⚡ Flash Attention 2 enabled")
                else:
                    print("🔧 Flash Attention not available, using standard attention")
        except Exception as e:
            print(f"⚠️ Flash Attention check failed: {e}")
    
    def _collect_environment_info(self) -> EnvironmentInfo:
        """Collect comprehensive environment information."""
        env_info = {
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
            "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "torch_version": torch.__version__,
            "python_version": sys.version,
            "platform": platform.platform(),
            "timestamp": datetime.now().isoformat()
        }
        
        # GPU memory info
        if torch.cuda.is_available():
            gpu_memory = []
            for i in range(torch.cuda.device_count()):
                props = torch.cuda.get_device_properties(i)
                gpu_memory.append(props.total_memory / 1e9)
            env_info["gpu_memory_gb"] = gpu_memory
        
        # Package versions
        try:
            import transformers
            env_info["transformers_version"] = transformers.__version__
        except ImportError:
            env_info["transformers_version"] = "unknown"
        
        try:
            import peft
            env_info["peft_version"] = peft.__version__
        except ImportError:
            env_info["peft_version"] = "unknown"
        
        try:
            import bitsandbytes
            env_info["bitsandbytes_version"] = bitsandbytes.__version__
        except ImportError:
            env_info["bitsandbytes_version"] = None
        
        return EnvironmentInfo(**env_info)
    
    def _detect_hardware(self):
        """Detect hardware and provide recommendations."""
        if torch.cuda.is_available():
            total_memory = sum(self.env_info.gpu_memory_gb or [])
            print(f"🔧 GPU Memory: {total_memory:.1f} GB across {self.env_info.gpu_count} GPU(s)")
            print(f"   CUDA Version: {self.env_info.cuda_version}")
            
            if total_memory < 12:
                print("⚠️  Limited GPU memory detected. Consider:")
                print("   - Reducing batch size")
                print("   - Using gradient checkpointing (enabled)")
                print("   - Using 4-bit quantization (enabled)")
        else:
            print("⚠️  No CUDA GPU detected. Training will be very slow on CPU.")
    
    def _auto_tune_memory_settings(self):
        """Auto-tune memory settings based on available GPU memory."""
        if not torch.cuda.is_available() or not self.env_info.gpu_memory_gb:
            print("🔧 No GPU memory info available, using default settings")
            return
        
        max_memory = max(self.env_info.gpu_memory_gb)
        print(f"🔧 Auto-tuning for {max_memory:.1f}GB GPU memory")
        
        # Memory-based heuristics
        if max_memory < 8:  # Less than 8GB - very conservative
            self.config.per_device_train_batch_size = 1
            self.config.gradient_accumulation_steps = max(16, self.config.gradient_accumulation_steps)
            print(f"   Low memory: batch_size=1, grad_accum={self.config.gradient_accumulation_steps}")
            
        elif max_memory < 12:  # 8-12GB - conservative
            self.config.per_device_train_batch_size = 1
            self.config.gradient_accumulation_steps = max(8, self.config.gradient_accumulation_steps)
            print(f"   Medium memory: batch_size=1, grad_accum={self.config.gradient_accumulation_steps}")
            
        elif max_memory > 20:  # >20GB - can be more aggressive
            # Try larger batch size if user didn't specify
            if self.config.per_device_train_batch_size == 1:  # Default value
                self.config.per_device_train_batch_size = 2
            self.config.gradient_accumulation_steps = max(2, self.config.gradient_accumulation_steps // 2)
            print(f"   High memory: batch_size={self.config.per_device_train_batch_size}, grad_accum={self.config.gradient_accumulation_steps}")
        
        # Calculate and display effective batch size
        effective_batch_size = self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps
        if torch.cuda.is_available():
            effective_batch_size *= self.env_info.gpu_count
        print(f"📊 Effective batch size: {effective_batch_size}")
    
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
        
        tokenizer.padding_side = "right"  # Required for causal LM
        
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
            print("⚠️  4-bit quantization disabled (no CUDA or disabled in config)")
        
        # Load model
        # Choose a safe dtype based on backend
        preferred_dtype = torch.float32
        if torch.cuda.is_available():
            preferred_dtype = torch.bfloat16 if self.config.bf16 else (torch.float16 if self.config.fp16 else torch.float32)
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            # On macOS < 14, BF16 on MPS is not supported
            try:
                mac_ver = platform.mac_ver()[0]
                mac_major = int(mac_ver.split(".")[0]) if mac_ver else 0
            except Exception:
                mac_major = 0
            if self.config.bf16 and mac_major >= 14:
                preferred_dtype = torch.bfloat16
            else:
                if self.config.bf16 and mac_major < 14:
                    print("⚠️  Forcing FP16 dtype on MPS (macOS < 14).")
                preferred_dtype = torch.float16 if self.config.fp16 or self.config.bf16 else torch.float32
        else:
            preferred_dtype = torch.float32

        # Select attention implementation with safe defaults on T4 (no FlashAttention 2)
        attn_env = os.getenv("ATTN_IMPL")
        if attn_env in {"flash_attention_2", "sdpa", "eager"}:
            attn_impl = attn_env
        else:
            # Default to disabled unless explicitly enabled via env
            use_fa2 = os.getenv("HF_USE_FLASH_ATTENTION_2", "0") != "0"
            if torch.cuda.is_available() and use_fa2:
                # Transformers will fall back internally if FA2 truly unavailable, but
                # T4 (sm_75) lacks support — prefer SDPA in that case.
                try:
                    cc = torch.cuda.get_device_capability(0)[0]
                    attn_impl = "flash_attention_2" if cc >= 8 else "sdpa"
                except Exception:
                    attn_impl = "sdpa"
            else:
                attn_impl = "sdpa" if torch.cuda.is_available() else "eager"

        # Optional multi-GPU max_memory hints
        max_memory = None
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            try:
                max_memory = {}
                for i in range(torch.cuda.device_count()):
                    total = torch.cuda.get_device_properties(i).total_memory
                    gb = int(total * 0.9 / (1024**3))  # 90% budget
                    max_memory[str(i)] = f"{gb}GiB"
                print(f"🔧 Using max_memory hints: {max_memory}")
            except Exception:
                max_memory = None

        model_kwargs = dict(
            quantization_config=quantization_config,
            trust_remote_code=True,
            torch_dtype=preferred_dtype,
            attn_implementation=attn_impl,
            low_cpu_mem_usage=True,
        )
        # Prefer explicit single-GPU placement to avoid unintended CPU/disk offload
        if torch.cuda.is_available():
            if torch.cuda.device_count() == 1:
                model_kwargs["device_map"] = {"": 0}
                try:
                    total = torch.cuda.get_device_properties(0).total_memory
                    gb = max(1, int(total * 0.9 / (1024**3)))
                    model_kwargs["max_memory"] = {"0": f"{gb}GiB"}
                    print(f"🔧 Using single-GPU max_memory hint: {model_kwargs['max_memory']}")
                except Exception:
                    pass
            else:
                model_kwargs["device_map"] = "auto"
                if max_memory:
                    model_kwargs["max_memory"] = max_memory

        model = AutoModelForCausalLM.from_pretrained(
            self.config.model_name,
            **model_kwargs
        )

        # Move to MPS explicitly if available and not using CUDA
        if not torch.cuda.is_available() and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                model.to("mps")
            except TypeError as e:
                # Fallback if dtype/device move still causes issues
                print(f"⚠️  Could not move model to MPS: {e}. Falling back to CPU.")
                model.to("cpu")
        
        # Prepare for k-bit training if using quantization
        if quantization_config:
            model = prepare_model_for_kbit_training(model)
        
        # Enable gradient checkpointing
        if self.config.gradient_checkpointing:
            model.gradient_checkpointing_enable()
            model.config.use_cache = False  # Required when using gradient checkpointing
        
        return model
    
    def _setup_lora(self) -> PeftModel:
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
            # Default fallback
            target_modules = ["q_proj", "v_proj"]
            print(f"⚠️  Unknown model architecture, using default target modules: {target_modules}")
        
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            lora_dropout=self.config.lora_dropout,
            bias=self.config.lora_bias,
            target_modules=target_modules,
            modules_to_save=self.config.modules_to_save,
        )
        
        model = get_peft_model(self.model, lora_config)
        
        print(f"   LoRA rank: {self.config.lora_rank}")
        print(f"   LoRA alpha: {self.config.lora_alpha}")
        print(f"   Target modules: {target_modules}")
        
        return model
    
    def _load_datasets(self, train_path: Path, eval_path: Path) -> Tuple[Dataset, Dataset]:
        """Load and preprocess datasets."""
        print("📚 Loading datasets...")
        
        # Load JSONL files
        train_dataset = load_dataset('json', data_files=str(train_path), split='train')
        eval_dataset = load_dataset('json', data_files=str(eval_path), split='train')
        
        print(f"   Train examples: {len(train_dataset)}")
        print(f"   Eval examples: {len(eval_dataset)}")
        
        return train_dataset, eval_dataset
    
    def _preprocess_function(self, examples):
        """Preprocess examples with proper label masking."""
        # System prompt for Employment Act assistant
        system_prompt = "You are an expert on Malaysia Employment Act. Provide accurate answers with proper section citations."
        
        inputs = []
        labels = []
        
        for instruction, output in zip(examples['instruction'], examples['output']):
            # Create conversation
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": output}
            ]
            
            # Apply chat template
            full_text = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False
            )
            
            # Tokenize
            tokenized = self.tokenizer(
                full_text,
                truncation=True,
                max_length=self.config.max_length,
                padding="max_length"
            )
            
            # Create labels with masking (only train on assistant response)
            input_ids = tokenized['input_ids']
            labels_ids = input_ids.copy()
            
            # Robust token-boundary label masking - tokenize separately then concatenate
            try:
                # 1. Tokenize system + user messages
                system_user_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": instruction}
                ]
                
                system_user_text = self.tokenizer.apply_chat_template(
                    system_user_messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # 2. Tokenize assistant response separately
                assistant_message = [{"role": "assistant", "content": output}]
                assistant_text = self.tokenizer.apply_chat_template(
                    assistant_message,
                    tokenize=False,
                    add_generation_prompt=False
                ).strip()
                
                # 3. Get token counts for each part
                system_user_tokens = self.tokenizer(
                    system_user_text,
                    truncation=False,
                    padding=False,
                    add_special_tokens=False
                )['input_ids']
                
                assistant_tokens = self.tokenizer(
                    assistant_text,
                    truncation=False,
                    padding=False,
                    add_special_tokens=False
                )['input_ids']
                
                # 4. Mask labels for system+user span only
                system_user_length = len(system_user_tokens)
                
                # Ensure we don't exceed the total length
                mask_length = min(system_user_length, len(labels_ids))
                for i in range(mask_length):
                    labels_ids[i] = -100
                    
            except Exception as e:
                # Fallback to string-based method if tokenize-separately fails
                assistant_start = full_text.find("assistant")
                if assistant_start != -1:
                    assistant_tokens = self.tokenizer(
                        full_text[:assistant_start],
                        truncation=True,
                        max_length=self.config.max_length,
                        padding=False
                    )['input_ids']
                    
                    for i in range(min(len(assistant_tokens), len(labels_ids))):
                        labels_ids[i] = -100
            
            inputs.append(input_ids)
            labels.append(labels_ids)
        
        return {
            'input_ids': inputs,
            'labels': labels,
            'attention_mask': [[1] * len(inp) for inp in inputs]
        }
    
    def train(self, train_path: Path, eval_path: Path, output_dir: Path):
        """Train the model with comprehensive evaluation."""
        print(f"🏋️ Starting training...")
        
        # Load datasets
        train_dataset, eval_dataset = self._load_datasets(train_path, eval_path)
        
        # Preprocess datasets
        print("🔄 Preprocessing datasets...")
        train_dataset = train_dataset.map(
            self._preprocess_function,
            batched=True,
            remove_columns=train_dataset.column_names
        )
        eval_dataset = eval_dataset.map(
            self._preprocess_function,
            batched=True,
            remove_columns=eval_dataset.column_names
        )
        
        # Training arguments
        # Choose optimizer based on quantization
        optim = "adamw_torch"  # Default
        if self.config.use_4bit and torch.cuda.is_available():
            optim = "adamw_bnb_8bit"  # Better for 4-bit quantization
            print(f"🔧 Using adamw_bnb_8bit optimizer for 4-bit quantization")
        else:
            print(f"🔧 Using adamw_torch optimizer")
        
        ta_kwargs = dict(
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
            save_strategy=self.config.save_strategy,
            load_best_model_at_end=self.config.load_best_model_at_end,
            metric_for_best_model=self.config.metric_for_best_model,
            greater_is_better=False,
            bf16=self.config.bf16,
            fp16=self.config.fp16,
            gradient_checkpointing=self.config.gradient_checkpointing,
            dataloader_num_workers=self.config.dataloader_num_workers,
            remove_unused_columns=False,
            report_to=self.config.report_to,
            seed=self.config.seed,
            optim=optim,
            save_total_limit=3,
            resume_from_checkpoint=True,
        )
        # Transformers v5 renamed evaluation_strategy -> eval_strategy. Detect via signature.
        try:
            params = _sig(TrainingArguments.__init__).parameters
            if "eval_strategy" in params:
                ta_kwargs["eval_strategy"] = self.config.eval_strategy
            elif "evaluation_strategy" in params:
                ta_kwargs["evaluation_strategy"] = self.config.eval_strategy
            # else: neither present; skip adding
        except Exception:
            # Fallback to old name
            ta_kwargs["evaluation_strategy"] = self.config.eval_strategy

        training_args = TrainingArguments(**ta_kwargs)
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False,
            pad_to_multiple_of=8
        )
        
        # Load the persistent eval subset with stable IDs
        eval_subset_path = output_dir / "eval_subset.jsonl"
        eval_examples_fallback = load_dataset('json', data_files=str(eval_path), split='train')
        
        eval_subset_examples = load_stable_eval_subset(
            eval_subset_path, 
            fallback_examples=eval_examples_fallback
        )
        
        if eval_subset_path.exists():
            print(f"📌 Loaded persistent eval subset with stable IDs from {eval_subset_path}")
            print(f"   {len(eval_subset_examples)} examples with stable IDs")
        else:
            print("⚠️ No persistent eval subset found, generated stable subset from eval data")
            print(f"   Created {len(eval_subset_examples)} examples with stable IDs")
        
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
            early_stopping_threshold=0.001  # Minimum improvement threshold
        )
        
        # Trainer with callbacks
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=self.tokenizer,
            data_collator=data_collator,
            callbacks=[citation_callback, early_stopping],
        )
        
        # Train
        print("🚀 Starting training...")
        train_result = trainer.train()
        
        # Save model (PEFT adapter only)
        print("💾 Saving model...")
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training results
        train_results = {
            "train_runtime": train_result.metrics["train_runtime"],
            "train_samples_per_second": train_result.metrics["train_samples_per_second"],
            "train_loss": train_result.metrics["train_loss"],
            "epoch": train_result.metrics["epoch"],
        }
        
        with open(output_dir / "train_results.json", 'w') as f:
            json.dump(train_results, f, indent=2)
        
        # Final citation evaluation
        print("🧪 Running final citation evaluation...")
        eval_result = evaluator.evaluate_examples(eval_subset_examples)
        eval_result.eval_loss = trainer.evaluate()['eval_loss']
        
        # Add dataset-level presence metrics from validation (with robust import fallback)
        try:
            try:
                from .validate_dataset import SFTDatasetValidator  # type: ignore
            except Exception:
                # Fallback for script execution without package context
                sys.path.append(str(Path(__file__).parent))
                from validate_dataset import SFTDatasetValidator  # type: ignore

            validator = SFTDatasetValidator()
            examples_with_citations, _, individual_presence_rate = validator.validate_citation_presence_in_output(eval_subset_examples)

            # Calculate example-level presence rate
            example_level_rate = (examples_with_citations / len(eval_subset_examples) * 100) if eval_subset_examples else 0.0

            # Add presence metrics to eval result
            eval_result.citation_presence_rate = example_level_rate
            eval_result.individual_citation_presence_rate = individual_presence_rate

            print(
                f"📊 Added dataset validation metrics: "
                f"{example_level_rate:.1f}% examples with citations, {individual_presence_rate:.1f}% individual citations present"
            )
        except Exception as e:
            print(f"⚠️ Could not add dataset validation metrics: {e}")
            # Keep default values (0.0)
        
        # Save evaluation results
        eval_results = asdict(eval_result)
        with open(output_dir / "eval_results.json", 'w') as f:
            json.dump(eval_results, f, indent=2)

        # Artifact sampling: save a few eval examples with responses and extracted citations
        try:
            samples_file = output_dir / "sft_eval_samples.jsonl"
            max_samples = min(10, len(eval_subset_examples))
            with open(samples_file, 'w', encoding='utf-8') as f:
                for ex in eval_subset_examples[:max_samples]:
                    instr = ex.get('instruction', '')
                    gold = ex.get('citations', [])
                    resp = evaluator._generate_response(instr)
                    preds = evaluator.extract_citations(resp)
                    f.write(json.dumps({
                        "instruction": instr,
                        "gold_citations": gold,
                        "response": resp,
                        "predicted_citations": list(preds)
                    }, ensure_ascii=False) + "\n")
            print(f"   Saved SFT eval samples to {samples_file}")
        except Exception:
            print("⚠️ Could not save SFT eval samples")

        # Plot training curves
        self._plot_training_curves(trainer, output_dir)
        
        # Save comprehensive metadata
        self._save_metadata(output_dir, train_results, eval_results)
        
        # Plot citation curves
        self._plot_citation_curves(output_dir)
        
        print(f"✅ Training complete!")
        print(f"   Final train loss: {train_results['train_loss']:.4f}")
        print(f"   Final eval loss: {eval_result.eval_loss:.4f}")
        print(f"   Citation exact match: {eval_result.citation_exact_match:.2%}")
        print(f"   Citation partial match: {eval_result.citation_partial_match:.2%}")
        print(f"   Citation F1 score: {eval_result.citation_f1_score:.3f}")
        print(f"   Judge score: {eval_result.avg_judge_score:.3f}")
        print(f"\n📁 Artifacts saved to: {output_dir}")
        print(f"   - adapter_model.safetensors")
        print(f"   - adapter_config.json")
        print(f"   - training_curves.png")
        print(f"   - citation_eval_history.jsonl")
        print(f"   - metadata.json")
        
        return trainer, eval_result
    
    def _plot_training_curves(self, trainer, output_dir: Path):
        """Plot and save training curves."""
        print("📊 Plotting training curves...")
        
        # Extract metrics from trainer state
        log_history = trainer.state.log_history
        
        train_losses = []
        eval_losses = []
        steps = []
        eval_steps = []
        
        for log in log_history:
            if 'loss' in log:
                train_losses.append(log['loss'])
                steps.append(log['step'])
            if 'eval_loss' in log:
                eval_losses.append(log['eval_loss'])
                eval_steps.append(log['step'])
        
        # Create plot
        plt.figure(figsize=(12, 6))
        
        # Plot training loss
        plt.subplot(1, 2, 1)
        if train_losses:
            plt.plot(steps, train_losses, label='Training Loss', color='blue')
        if eval_losses and eval_steps:
            plt.plot(eval_steps, eval_losses, label='Validation Loss', color='red', marker='o')
        plt.xlabel('Steps')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Plot learning rate if available
        plt.subplot(1, 2, 2)
        lrs = [log.get('learning_rate', 0) for log in log_history if 'learning_rate' in log]
        lr_steps = [log['step'] for log in log_history if 'learning_rate' in log]
        if lrs:
            plt.plot(lr_steps, lrs, label='Learning Rate', color='green')
            plt.xlabel('Steps')
            plt.ylabel('Learning Rate')
            plt.title('Learning Rate Schedule')
            plt.legend()
            plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "training_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Training curves saved to {output_dir / 'training_curves.png'}")
        # Export CSV for downstream analysis
        try:
            import csv
            csv_path = output_dir / "training_curves.csv"
            lrs = [log.get('learning_rate', 0) for log in log_history if 'learning_rate' in log]
            lr_steps = [log['step'] for log in log_history if 'learning_rate' in log]
            max_len = max(len(steps), len(eval_steps), len(lr_steps))
            with open(csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["step", "train_loss", "eval_step", "eval_loss", "lr_step", "learning_rate"])
                for i in range(max_len):
                    writer.writerow([
                        steps[i] if i < len(steps) else "",
                        train_losses[i] if i < len(train_losses) else "",
                        eval_steps[i] if i < len(eval_steps) else "",
                        eval_losses[i] if i < len(eval_losses) else "",
                        lr_steps[i] if i < len(lr_steps) else "",
                        lrs[i] if i < len(lrs) else "",
                    ])
            print(f"   Training curves CSV saved to {csv_path}")
        except Exception:
            print("⚠️ Could not save training curves CSV")
    
    def _plot_citation_curves(self, output_dir: Path):
        """Plot citation evaluation curves."""
        history_file = output_dir / "citation_eval_history.jsonl"
        
        if not history_file.exists():
            print("⚠️ No citation evaluation history found")
            return
        
        # Load citation history
        citation_data = []
        with open(history_file, 'r') as f:
            for line in f:
                citation_data.append(json.loads(line.strip()))
        
        if not citation_data:
            print("⚠️ No citation evaluation data found")
            return
        
        # Extract data
        steps = [d['step'] for d in citation_data]
        exact_matches = [d['citation_exact_match'] for d in citation_data]
        partial_matches = [d['citation_partial_match'] for d in citation_data]
        judge_scores = [d['avg_judge_score'] for d in citation_data]
        
        # Create plot
        plt.figure(figsize=(15, 5))
        
        # Citation exact match
        plt.subplot(1, 3, 1)
        plt.plot(steps, exact_matches, 'b-', marker='o', label='Exact Match')
        plt.xlabel('Steps')
        plt.ylabel('Citation Exact Match')
        plt.title('Citation Exact Match over Training')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Citation partial match (IoU)
        plt.subplot(1, 3, 2)
        plt.plot(steps, partial_matches, 'g-', marker='s', label='Partial Match (IoU)')
        plt.xlabel('Steps')
        plt.ylabel('Citation Partial Match (IoU)')
        plt.title('Citation IoU over Training')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        # Judge scores
        plt.subplot(1, 3, 3)
        plt.plot(steps, judge_scores, 'r-', marker='^', label='Judge Score')
        plt.xlabel('Steps')
        plt.ylabel('Judge Score')
        plt.title('Response Quality Score over Training')
        plt.grid(True, alpha=0.3)
        plt.ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(output_dir / "citation_curves.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"   Citation curves saved to {output_dir / 'citation_curves.png'}")
    
    def _save_metadata(self, output_dir: Path, train_results: Dict, eval_results: Dict):
        """Save comprehensive training metadata."""
        
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
        
        # Normalize environment info to a dict (supports Pydantic v1/v2, dataclass, or dict)
        try:
            env_info_block = self.env_info.model_dump()  # Pydantic v2
        except Exception:
            try:
                env_info_block = self.env_info.dict()  # Pydantic v1
            except Exception:
                try:
                    env_info_block = asdict(self.env_info)  # dataclass
                except Exception:
                    env_info_block = self.env_info if isinstance(self.env_info, dict) else {}

        metadata = {
            "training_info": {
                "timestamp": datetime.now().isoformat(),
                "config": asdict(self.config),
                "model_name": self.config.model_name,
                "total_parameters": self.model.num_parameters(),
                "trainable_parameters": self.model.num_parameters(only_trainable=True),
                "trainable_ratio": self.model.num_parameters(only_trainable=True) / self.model.num_parameters() * 100,
                "effective_batch_size": self.config.per_device_train_batch_size * self.config.gradient_accumulation_steps,
                "device": str(self.device),
                "logging_backend": self.config.report_to,
                "tensorboard_run_dir": tensorboard_run_dir,
                "cli_args": sys.argv,
            },
            "environment": env_info_block,
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
                "adapter_config": "adapter_config.json",
                "adapter_weights": "adapter_model.safetensors",
                "tokenizer": "tokenizer files",
                "training_curves": "training_curves.png",
                "citation_curves": "citation_curves.png",
                "citation_eval_history": "citation_eval_history.jsonl",
                "train_results": "train_results.json",
                "eval_results": "eval_results.json",
                "tensorboard_logs": tensorboard_run_dir if tensorboard_run_dir else None,
            },
            "versions": {
                "python": sys.version,
                "torch": torch.__version__,
                "transformers": getattr(_transformers, "__version__", None),
            }
        }
        
        with open(output_dir / "metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"   Metadata saved to {output_dir / 'metadata.json'}")

        # Also write accelerator.json with concise device info
        try:
            accel = {
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
                "gpu_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
                "bf16": self.config.bf16,
                "fp16": self.config.fp16,
            }
            with open(output_dir / "accelerator.json", 'w') as f:
                json.dump(accel, f, indent=2)
        except Exception:
            pass


def main():
    parser = argparse.ArgumentParser(description="Production QLoRA Training")
    
    # Data arguments
    parser.add_argument('--train-data', type=Path, required=True, help='Training JSONL file')
    parser.add_argument('--eval-data', type=Path, required=True, help='Evaluation JSONL file')
    parser.add_argument('--output-dir', type=Path, required=True, help='Output directory')
    
    # Model arguments
    parser.add_argument('--model-name', default="meta-llama/Llama-3.2-1B-Instruct", help='Base model name')
    parser.add_argument('--bits', type=int, choices=[4, 8, 16], default=4, help='Quantization bits')
    
    # LoRA arguments
    parser.add_argument('--lora-rank', type=int, default=16, help='LoRA rank')
    parser.add_argument('--lora-alpha', type=int, default=32, help='LoRA alpha')
    parser.add_argument('--lora-dropout', type=float, default=0.1, help='LoRA dropout')
    
    # Training arguments
    parser.add_argument('--epochs', type=int, default=2, help='Number of epochs (Hour 4 default: 2)')
    parser.add_argument('--batch-size', type=int, default=1, help='Per device batch size')
    parser.add_argument('--grad-accumulation', type=int, default=4, help='Gradient accumulation steps (auto-tuned by memory)')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate (Hour 4 default: 1e-4)')
    parser.add_argument('--warmup-ratio', type=float, default=0.1, help='Warmup ratio')
    parser.add_argument('--eval-steps', type=int, default=50, help='Evaluation steps')
    parser.add_argument('--save-steps', type=int, default=100, help='Save steps')
    parser.add_argument('--max-length', type=int, default=2048, help='Maximum sequence length')
    parser.add_argument('--early-stopping-patience', type=int, default=3, help='Early stopping patience')
    
    # Precision arguments
    parser.add_argument('--bf16', action='store_true', default=True, help='Use bfloat16 precision (default)')
    parser.add_argument('--fp16', action='store_true', help='Use float16 precision')
    parser.add_argument('--grad-ckpt', action='store_true', default=True, help='Use gradient checkpointing (default)')
    parser.add_argument('--report-to', action='append', choices=['wandb', 'tensorboard'], default=None,
                        help='Logging backend(s): "wandb" for cloud tracking, "tensorboard" for local logs. Can be specified multiple times for multi-backend logging.')
    
    # Other arguments
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--use-sfttrainer', action='store_true', help='Use TRL SFTTrainer instead of custom trainer (easier auditing)')
    
    args = parser.parse_args()
    
    # Create config
    config = QLoRAConfig(
        model_name=args.model_name,
        use_4bit=(args.bits == 4),
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        num_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accumulation,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        max_length=args.max_length,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=args.grad_ckpt,
        seed=args.seed,
        report_to=args.report_to,  # Already a list from argparse append action
        early_stopping_patience=args.early_stopping_patience,
    )
    
    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize trainer and train
    if args.use_sfttrainer:
        # Use TRL SFTTrainer path for easier auditing
        print("🔄 Using TRL SFTTrainer path for easier auditing")
        try:
            from .train_lora_trl import TRLSFTTrainer_Production
        except ImportError:
            # For standalone execution
            from train_lora_trl import TRLSFTTrainer_Production
        trainer = TRLSFTTrainer_Production(config)
    else:
        # Use custom production trainer (default)
        trainer = ProductionQLoRATrainer(config)
    
    trainer.train(args.train_data, args.eval_data, args.output_dir)
    
    print(f"🎉 Training completed successfully!")
    print(f"📁 Artifacts saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
