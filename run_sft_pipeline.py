#!/usr/bin/env python3
"""
SFT (Supervised Fine-Tuning) Pipeline - Dataset Generation + Model Training

This script orchestrates the complete SFT pipeline:
1. Generate SFT dataset from text chunks with validation
2. Train LoRA adapters with comprehensive monitoring
3. Validate outputs and generate training reports

Key Features:
- ✅ Pydantic schema validation for data quality
- ✅ Stratified sampling to avoid data leakage
- ✅ Citation grounding with canonical patterns
- ✅ QLoRA training with memory optimization
- ✅ Real-time citation evaluation during training
- ✅ Comprehensive validation and error handling
- ✅ Multi-backend logging support

Usage Examples:
    # Complete SFT pipeline
    python run_sft_pipeline.py full --chunks data/processed/chunks.jsonl

    # Only dataset generation
    python run_sft_pipeline.py dataset --chunks data/processed/chunks.jsonl --size 200

    # Only training (existing dataset)
    python run_sft_pipeline.py train --train-data outputs/sft_dataset/sft_dataset_train.jsonl

    # Development mode (small dataset)
    python run_sft_pipeline.py dev --chunks data/processed/chunks.jsonl

    # With configuration file
    python run_sft_pipeline.py full --config config/sft_production.yaml
"""

import argparse
import subprocess
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional
import sys
import os
from dataclasses import dataclass, field, asdict
from enum import Enum


class SFTStage(Enum):
    """SFT pipeline stages."""
    DATASET = "dataset"
    VALIDATION = "validation"
    TRAINING = "training"
    EVALUATION = "evaluation"


@dataclass
class SFTConfig:
    """Configuration for SFT training pipeline."""
    
    # Required paths
    chunks_file: Path
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    
    # Dataset generation config
    dataset_size: int = 200
    dataset_seed: int = 42
    chunk_size: int = 1000
    chunk_stride: int = 150
    
    # Training config
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    epochs: int = 2
    learning_rate: float = 1e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    
    # LoRA config
    lora_rank: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    
    # Advanced training config
    max_length: int = 1024
    warmup_ratio: float = 0.03
    weight_decay: float = 0.001
    lr_scheduler_type: str = "cosine"
    
    # Quantization config
    use_4bit: bool = True
    bf16: bool = True
    fp16: bool = False
    
    # Pipeline control
    skip_dataset: bool = False
    skip_validation: bool = False
    skip_training: bool = False
    train_split: float = 0.8
    
    # Advanced options
    dry_run: bool = False
    verbose: bool = False
    experiment_name: Optional[str] = None
    logging_backends: List[str] = field(default_factory=list)
    use_sfttrainer: bool = False  # Use TRL SFTTrainer instead of custom trainer
    # Auto-select a suitable base model by VRAM and keep consistent suggestions
    auto_select_base: bool = True
    
    @classmethod
    def from_file(cls, config_path: Path) -> 'SFTConfig':
        """Load configuration from YAML or JSON file."""
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        # Convert path strings to Path objects
        if 'chunks_file' in data:
            data['chunks_file'] = Path(data['chunks_file'])
        if 'output_dir' in data:
            data['output_dir'] = Path(data['output_dir'])
            
        return cls(**data)


@dataclass
class SFTState:
    """Track SFT pipeline progress."""
    dataset_completed: bool = False
    validation_completed: bool = False
    training_completed: bool = False
    evaluation_completed: bool = False
    
    dataset_output: Optional[Path] = None
    training_output: Optional[Path] = None
    
    start_time: datetime = field(default_factory=datetime.now)
    last_checkpoint: datetime = field(default_factory=datetime.now)


class SFTPipelineError(Exception):
    """Custom exception for SFT pipeline errors."""
    pass


class SFTPipeline:
    """Complete SFT training pipeline with dataset generation and model training."""
    
    def __init__(self, config: SFTConfig):
        self.config = config
        self.state = SFTState()
        self.setup_logging()
        self.setup_output_dirs()
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if config.experiment_name:
            self.output_dir = config.output_dir / f"sft_{config.experiment_name}_{timestamp}"
        else:
            self.output_dir = config.output_dir / f"sft_{timestamp}"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Define key paths
        self.dataset_dir = self.output_dir / "sft_dataset"
        self.model_dir = self.output_dir / "lora_sft"
        
        # Save config
        with open(self.output_dir / "config.json", 'w') as f:
            json.dump(asdict(config), f, indent=2, default=str)

        # Detect accelerator and apply conservative overrides for MPS/CPU
        try:
            from src.utils.accelerator import get_accelerator
            accel = get_accelerator()
            self.accelerator = accel
            if accel.backend == 'mps':
                # Disable bf16; prefer small batch to reduce memory pressure
                if self.config.bf16:
                    self.logger.info("🍏 MPS detected: disabling bf16 for compatibility")
                    self.config.bf16 = False
                if self.config.batch_size > 2:
                    self.logger.info("🍏 MPS detected: reducing batch size to 2 for stability")
                    self.config.batch_size = 2
            elif accel.backend == 'cpu':
                # Warn on large models
                if '8B' in self.config.model_name or '7B' in self.config.model_name:
                    self.logger.warning("⚠️ Large base model on CPU may be extremely slow. Consider a smaller model (e.g., SmolLM-135M).")
            elif accel.backend == 'cuda' and self.config.auto_select_base:
                # Auto-select an appropriate base model when user didn't explicitly override
                # Heuristic: ≤30 GB → prefer 1B; ≥40 GB → prefer 8B; else 1B
                # Only override if model_name is still at a known default or empty
                try:
                    vram_gb_list = accel.details.get('gpu_memory_gb') if isinstance(accel.details, dict) else None
                    vram_gb = None
                    if vram_gb_list and isinstance(vram_gb_list, list) and vram_gb_list:
                        vram_gb = int(max(vram_gb_list))
                except Exception:
                    vram_gb = None

                desired_1b = "meta-llama/Llama-3.2-1B-Instruct"
                desired_8b = "meta-llama/Llama-3.1-8B-Instruct"

                # Consider it user-overridden if model_name isn't one of our known defaults
                known_defaults = {desired_1b, desired_8b}
                user_overridden = self.config.model_name not in known_defaults

                if not user_overridden:
                    if vram_gb is not None and vram_gb <= 30:
                        if self.config.model_name != desired_1b:
                            self.logger.info("🧠 VRAM ≲30 GB detected; selecting small base: meta-llama/Llama-3.2-1B-Instruct for stability")
                            self.config.model_name = desired_1b
                    elif vram_gb is not None and vram_gb >= 40:
                        if self.config.model_name != desired_8b:
                            self.logger.info("🧠 VRAM ≥40 GB detected; selecting larger base: meta-llama/Llama-3.1-8B-Instruct for quality")
                            self.config.model_name = desired_8b
                    else:
                        # Ambiguous VRAM; prefer the small base
                        if self.config.model_name != desired_1b:
                            self.logger.info("ℹ️ Ambiguous VRAM; defaulting to small base: meta-llama/Llama-3.2-1B-Instruct")
                            self.config.model_name = desired_1b
        except Exception:
            # Best-effort only; continue on failure
            self.accelerator = None
    
    def setup_logging(self):
        """Setup comprehensive logging."""
        log_level = logging.DEBUG if self.config.verbose else logging.INFO
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        
        if self.config.dry_run:
            self.logger.info("🔍 DRY RUN MODE: Commands will be shown but not executed")
    
    def setup_output_dirs(self):
        """Create necessary output directories."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self):
        """Save current pipeline state."""
        self.state.last_checkpoint = datetime.now()
        
        checkpoint_file = self.output_dir / "checkpoint.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(asdict(self.state), f, indent=2, default=str)
        
        self.logger.debug(f"💾 Checkpoint saved: {checkpoint_file}")
    
    def validate_prerequisites(self):
        """Validate all inputs before starting."""
        self.logger.info("🔍 Validating prerequisites...")
        
        if not self.config.chunks_file.exists():
            raise SFTPipelineError(f"Chunks file not found: {self.config.chunks_file}")
        
        # Validate chunks file format
        try:
            with open(self.config.chunks_file, 'r') as f:
                first_line = f.readline()
                json.loads(first_line)  # Validate JSON format
        except Exception as e:
            raise SFTPipelineError(f"Invalid chunks file format: {e}")
        
        self.logger.info("✅ Prerequisites validated")
    
    def run_subprocess_with_monitoring(self, cmd: List[str], stage_name: str) -> int:
        """Run subprocess with real-time monitoring."""
        cmd_str = ' '.join(str(c) for c in cmd)
        self.logger.info(f"🚀 [{stage_name}] Running: {cmd_str}")
        
        if self.config.dry_run:
            self.logger.info(f"🔍 DRY RUN: Would execute command above")
            return 0
        
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, cwd=Path.cwd()
        )
        
        for line in process.stdout:
            line = line.strip()
            if line:
                self.logger.info(f"[{stage_name}] {line}")
        
        process.wait()
        
        if process.returncode != 0:
            raise SFTPipelineError(f"{stage_name} failed with return code {process.returncode}")
        
        return process.returncode
    
    def run_dataset_generation(self) -> Path:
        """Generate SFT dataset with validation."""
        stage_name = "DATASET"
        self.logger.info(f"\n📊 Step 1: Generating SFT dataset ({self.config.dataset_size} examples)...")
        
        self.dataset_dir.mkdir(parents=True, exist_ok=True)
        
        cmd = [
            "python", "src/training/make_sft_dataset.py",
            "--chunks", str(self.config.chunks_file),
            "--output-dir", str(self.dataset_dir),
            "--size", str(self.config.dataset_size),
            "--seed", str(self.config.dataset_seed)
        ]
        
        self.run_subprocess_with_monitoring(cmd, stage_name)
        
        if not self.config.dry_run:
            # Validate outputs
            train_file = self.dataset_dir / "sft_dataset_train.jsonl"
            eval_file = self.dataset_dir / "sft_dataset_eval.jsonl"
            
            if not train_file.exists() or not eval_file.exists():
                raise SFTPipelineError(f"Expected dataset files not found in {self.dataset_dir}")
        
        self.state.dataset_completed = True
        self.state.dataset_output = self.dataset_dir
        self.save_checkpoint()
        
        self.logger.info(f"✅ SFT dataset generated: {self.dataset_dir}")
        return self.dataset_dir
    
    def run_dataset_validation(self) -> bool:
        """Validate generated dataset quality."""
        stage_name = "VALIDATION"
        self.logger.info(f"\n🔍 Step 2: Validating dataset quality...")
        
        train_file = self.dataset_dir / "sft_dataset_train.jsonl"
        eval_file = self.dataset_dir / "sft_dataset_eval.jsonl"
        
        cmd = [
            "python", "src/training/validate_dataset.py",
            "--train-data", str(train_file),
            "--eval-data", str(eval_file),
            "--chunks", str(self.config.chunks_file)
        ]
        
        self.run_subprocess_with_monitoring(cmd, stage_name)
        
        self.state.validation_completed = True
        self.save_checkpoint()
        
        self.logger.info(f"✅ Dataset validation completed")
        return True
    
    def run_sft_training(self) -> Path:
        """Train SFT model with LoRA."""
        stage_name = "TRAINING"
        self.logger.info(f"\n🎯 Step 3: Training SFT model ({self.config.epochs} epochs)...")
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        train_file = self.dataset_dir / "sft_dataset_train.jsonl"
        eval_file = self.dataset_dir / "sft_dataset_eval.jsonl"
        
        # Choose training script based on config
        training_script = "src/training/train_lora_trl.py" if self.config.use_sfttrainer else "src/training/train_lora.py"
        
        cmd = [
            "python", training_script,
            "--train-data", str(train_file),
            "--eval-data", str(eval_file),
            "--output-dir", str(self.model_dir),
            "--model-name", self.config.model_name,
            "--epochs", str(self.config.epochs),
            "--learning-rate", str(self.config.learning_rate),
            "--batch-size", str(self.config.batch_size),
            "--grad-accumulation", str(self.config.gradient_accumulation_steps),
            "--max-length", str(self.config.max_length),
            "--lora-rank", str(self.config.lora_rank),
            "--lora-alpha", str(self.config.lora_alpha),
            "--seed", str(self.config.dataset_seed)
        ]
        
        # Add precision flags
        if self.config.bf16:
            cmd.append("--bf16")
        elif self.config.fp16:
            cmd.append("--fp16")
        
        # Add SFTTrainer flag if needed
        if self.config.use_sfttrainer:
            cmd.append("--use-sfttrainer")
        
        # Add logging backends
        for backend in self.config.logging_backends:
            cmd.extend(["--report-to", backend])
        
        self.run_subprocess_with_monitoring(cmd, stage_name)
        
        if not self.config.dry_run:
            # Validate training output
            adapter_config = self.model_dir / "adapter_config.json"
            if not adapter_config.exists():
                raise SFTPipelineError(f"Training failed - adapter config not found: {adapter_config}")
        
        self.state.training_completed = True
        self.state.training_output = self.model_dir
        self.save_checkpoint()
        
        self.logger.info(f"✅ SFT training completed: {self.model_dir}")
        return self.model_dir
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report."""
        self.logger.info(f"\n📋 Step 4: Generating final report...")
        
        report = {
            "pipeline_info": {
                "timestamp": datetime.now().isoformat(),
                "config": asdict(self.config),
                "state": asdict(self.state),
                "output_directory": str(self.output_dir)
            },
            "artifacts": {
                "dataset_dir": str(self.state.dataset_output) if self.state.dataset_output else None,
                "model_dir": str(self.state.training_output) if self.state.training_output else None,
                "train_file": str(self.dataset_dir / "sft_dataset_train.jsonl") if self.state.dataset_completed else None,
                "eval_file": str(self.dataset_dir / "sft_dataset_eval.jsonl") if self.state.dataset_completed else None,
                "adapter_weights": str(self.model_dir / "adapter_model.safetensors") if self.state.training_completed else None,
            }
        }

        # Attach accelerator info if available
        try:
            if getattr(self, 'accelerator', None) is not None:
                report["pipeline_info"]["accelerator"] = {
                    "backend": self.accelerator.backend,
                    "device": self.accelerator.device_str,
                    "can_use_4bit": self.accelerator.can_use_4bit,
                    "recommended_dtype": self.accelerator.recommended_dtype,
                    "details": self.accelerator.details,
                }
        except Exception:
            pass
        
        # Save report
        report_file = self.output_dir / "sft_final_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"📋 Final report saved: {report_file}")
        return report
    
    def run(self):
        """Main pipeline runner."""
        try:
            self.logger.info("🚀 Starting SFT Training Pipeline")
            self.logger.info(f"📁 Output directory: {self.output_dir}")
            
            # Validate prerequisites
            self.validate_prerequisites()
            
            # Run pipeline stages
            if not self.config.skip_dataset and not self.state.dataset_completed:
                self.run_dataset_generation()
            
            if not self.config.skip_validation and not self.state.validation_completed:
                self.run_dataset_validation()
            
            if not self.config.skip_training and not self.state.training_completed:
                self.run_sft_training()
            
            # Generate final report
            final_report = self.generate_final_report()
            
            # Success message
            duration = datetime.now() - self.state.start_time
            self.logger.info(f"\n🎉 SFT Pipeline Completed Successfully!")
            self.logger.info(f"⏱️ Total duration: {duration}")
            self.logger.info(f"📁 All outputs saved to: {self.output_dir}")
            
            if self.state.training_completed:
                self.logger.info(f"🤖 SFT model ready for RLAIF: {self.model_dir}")
            
            return final_report
            
        except SFTPipelineError as e:
            self.logger.error(f"❌ SFT Pipeline error: {e}")
            sys.exit(1)
        except KeyboardInterrupt:
            self.logger.warning(f"⚠️ Pipeline interrupted by user")
            self.save_checkpoint()
            sys.exit(1)
        except Exception as e:
            self.logger.error(f"💥 Unexpected error: {e}")
            self.save_checkpoint()
            raise e


def create_parser():
    """Create argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="SFT Training Pipeline - Dataset Generation + Model Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Complete SFT pipeline
  %(prog)s full --chunks data/processed/chunks.jsonl

  # Only dataset generation
  %(prog)s dataset --chunks data/processed/chunks.jsonl --size 200

  # Only training (with existing dataset)
  %(prog)s train --train-data outputs/sft_dataset/sft_dataset_train.jsonl

  # Development mode (small dataset)
  %(prog)s dev --chunks data/processed/chunks.jsonl

  # With configuration file
  %(prog)s full --config config/sft_production.yaml
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Pipeline commands')
    
    # Full pipeline
    full_parser = subparsers.add_parser('full', help='Run complete SFT pipeline')
    add_common_args(full_parser)
    
    # Dataset only
    dataset_parser = subparsers.add_parser('dataset', help='Generate SFT dataset only')
    add_dataset_args(dataset_parser)
    
    # Training only
    train_parser = subparsers.add_parser('train', help='Train SFT model only')
    add_training_args(train_parser)
    
    # Development mode
    dev_parser = subparsers.add_parser('dev', help='Development mode (small dataset)')
    add_common_args(dev_parser)
    dev_parser.set_defaults(dataset_size=50, epochs=1, batch_size=2)
    
    return parser


def add_common_args(parser):
    """Add common arguments."""
    parser.add_argument('--config', help='Path to configuration YAML/JSON file')
    parser.add_argument('--chunks', required=True, help='Path to chunks.jsonl file')
    
    # Dataset args
    parser.add_argument('--size', dest='dataset_size', type=int, default=200, help='Dataset size')
    parser.add_argument('--seed', dest='dataset_seed', type=int, default=42, help='Random seed')
    
    # Training args
    parser.add_argument('--model-name', default="meta-llama/Llama-3.2-1B-Instruct", help='Base model')
    parser.add_argument('--epochs', type=int, default=2, help='Training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    
    # Output args
    parser.add_argument('--output-dir', default="outputs", help='Output directory')
    parser.add_argument('--experiment-name', help='Experiment name')
    
    # Advanced args
    parser.add_argument('--use-sfttrainer', action='store_true', help='Use TRL SFTTrainer')
    parser.add_argument('--report-to', action='append', choices=['tensorboard', 'wandb'], 
                       default=[], help='Logging backends')
    parser.add_argument('--no-auto-select-base', dest='auto_select_base', action='store_false',
                       help='Disable VRAM-based auto-selection of base model')
    parser.add_argument('--dry-run', action='store_true', help='Show commands without executing')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')


def add_dataset_args(parser):
    """Add dataset-specific arguments."""
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--chunks', required=True, help='Path to chunks.jsonl file')
    parser.add_argument('--size', dest='dataset_size', type=int, default=200, help='Dataset size')
    parser.add_argument('--seed', dest='dataset_seed', type=int, default=42, help='Random seed')
    parser.add_argument('--output-dir', default="outputs", help='Output directory')
    parser.add_argument('--no-auto-select-base', dest='auto_select_base', action='store_false',
                       help='Disable VRAM-based auto-selection of base model')
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')


def add_training_args(parser):
    """Add training-specific arguments."""
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--train-data', required=True, help='Training data file')
    parser.add_argument('--eval-data', required=True, help='Evaluation data file')
    parser.add_argument('--model-name', default="meta-llama/Llama-3.2-1B-Instruct", help='Base model')
    parser.add_argument('--epochs', type=int, default=2, help='Training epochs')
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--batch-size', type=int, default=4, help='Batch size')
    parser.add_argument('--output-dir', default="outputs", help='Output directory')
    parser.add_argument('--use-sfttrainer', action='store_true', help='Use TRL SFTTrainer')
    parser.add_argument('--report-to', action='append', choices=['tensorboard', 'wandb'], 
                       default=[], help='Logging backends')
    parser.add_argument('--no-auto-select-base', dest='auto_select_base', action='store_false',
                       help='Disable VRAM-based auto-selection of base model')
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    # Capture parser defaults to prevent overriding config with defaults
    parser_defaults = {a.dest: a.default for a in parser._actions if getattr(a, 'dest', None)}
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        # Load configuration
        if hasattr(args, 'config') and args.config:
            config = SFTConfig.from_file(Path(args.config))
            # Override only when CLI values differ from parser defaults
            for key, value in vars(args).items():
                if key == 'config' or key not in SFTConfig.__dataclass_fields__:
                    continue
                default_val = parser_defaults.get(key, None)
                if value is not None and value != default_val:
                    if key in {"chunks_file", "output_dir"}:
                        setattr(config, key, Path(value))
                    else:
                        setattr(config, key, value)
        else:
            # Create config from command line args
            config_dict = {}
            
            # Map command line args to config fields
            arg_mapping = {
                'chunks': 'chunks_file',
                'dataset_size': 'dataset_size',
                'dataset_seed': 'dataset_seed',
                'size': 'dataset_size',
                'seed': 'dataset_seed'
            }
            
            for key, value in vars(args).items():
                if value is not None:
                    config_key = arg_mapping.get(key, key)
                    if config_key in SFTConfig.__dataclass_fields__:
                        config_dict[config_key] = value
            
            # Convert paths
            if 'chunks_file' in config_dict:
                config_dict['chunks_file'] = Path(config_dict['chunks_file'])
            if 'output_dir' in config_dict:
                config_dict['output_dir'] = Path(config_dict['output_dir'])
            
            # Handle subcommand-specific logic
            if args.command == 'dataset':
                config_dict.update({'skip_training': True})
            elif args.command == 'train':
                config_dict.update({'skip_dataset': True, 'skip_validation': True})
            
            config = SFTConfig(**config_dict)
        
        # Initialize and run pipeline
        pipeline = SFTPipeline(config)
        final_report = pipeline.run()
        
        print(f"\n✨ SFT Pipeline Complete!")
        print(f"📁 Check outputs in: {pipeline.output_dir}")
        
    except Exception as e:
        print(f"❌ SFT Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
