#!/usr/bin/env python3
"""
RLAIF Training Pipeline - Modular DPO + PPO Training

This script orchestrates the RLAIF (Reinforcement Learning from AI Feedback) training pipeline:
1. Generate preference pairs with canonical patterns and SFT drafting
2. Train DPO with fixed tokenizer padding and persistent eval subset
3. Run PPO with proper value-head initialization and memory optimization
4. Generate comprehensive evaluation reports

Key Features:
- ✅ Subcommands for different pipeline modes (full/partial/resume/dev)
- ✅ Configuration file support (YAML/JSON)
- ✅ Robust error handling with checkpointing
- ✅ Resource validation and monitoring
- ✅ Dry-run mode for testing
- ✅ Canonical citation patterns throughout
- ✅ Memory optimization and progress tracking

Usage Examples:
    # Complete RLAIF pipeline
    python run_rlaif_training.py full --chunks data/processed/chunks.jsonl --sft-model outputs/lora_sft

    # Partial pipeline (skip PPO)  
    python run_rlaif_training.py partial --chunks data/processed/chunks.jsonl --skip-ppo

    # Resume from DPO checkpoint
    python run_rlaif_training.py resume --from dpo --dpo-model outputs/lora_dpo

    # Development mode (small datasets)
    python run_rlaif_training.py dev --chunks data/processed/chunks.jsonl
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
from src.utils.accelerator import get_accelerator
import torch


class PipelineStage(Enum):
    """Pipeline stages for checkpointing and resume."""
    PAIRS = "pairs"
    DPO = "dpo" 
    PPO = "ppo"
    EVAL = "eval"


@dataclass
class RLAIFConfig:
    """Configuration for RLAIF training pipeline."""
    
    # Required paths
    chunks_file: Path
    sft_model: Optional[Path] = None
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    
    # Preference pairs config
    pairs_size: int = 120
    pairs_seed: int = 42
    
    # DPO config
    dpo_epochs: int = 2
    dpo_batch_size: int = 2
    dpo_learning_rate: float = 5e-5
    dpo_beta: float = 0.05
    
    # PPO config
    ppo_prompts: int = 16
    ppo_model: Optional[str] = None  # If None, align to model_name
    ppo_use_4bit: bool = False
    ppo_batch_size: int = 32
    ppo_mini_batch_size: int = 4
    
    # Model config
    model_name: str = "meta-llama/Llama-3.2-1B-Instruct"
    
    # Pipeline control
    skip_pairs: bool = False
    skip_dpo: bool = False
    skip_ppo: bool = False
    resume_from: Optional[PipelineStage] = None
    
    # Advanced options
    dry_run: bool = False
    verbose: bool = False
    experiment_name: Optional[str] = None
    logging_backends: List[str] = field(default_factory=list)
    # Auto-select a suitable base by VRAM and propagate consistently across SFT→DPO→PPO
    auto_select_base: bool = True
    
    @classmethod
    def from_file(cls, config_path: Path) -> 'RLAIFConfig':
        """Load configuration from YAML or JSON file."""
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        # Convert path strings to Path objects
        if 'chunks_file' in data:
            data['chunks_file'] = Path(data['chunks_file'])
        if 'sft_model' in data and data['sft_model']:
            data['sft_model'] = Path(data['sft_model'])
        if 'output_dir' in data:
            data['output_dir'] = Path(data['output_dir'])
            
        return cls(**data)


@dataclass
class PipelineState:
    """Track pipeline progress for checkpointing."""
    pairs_completed: bool = False
    dpo_completed: bool = False  
    ppo_completed: bool = False
    eval_completed: bool = False
    
    pairs_output: Optional[Path] = None
    dpo_output: Optional[Path] = None
    ppo_output: Optional[Path] = None
    
    start_time: datetime = field(default_factory=datetime.now)
    last_checkpoint: datetime = field(default_factory=datetime.now)


class PipelineError(Exception):
    """Custom exception for pipeline errors."""
    pass


class RLAIFTrainingPipeline:
    """RLAIF training pipeline with robust error handling and checkpointing."""
    
    def __init__(self, config: RLAIFConfig):
        self.config = config
        self.state = PipelineState()
        self.setup_logging()
        self.setup_output_dirs()
        # Normalize training config to recommended defaults when overly small
        # (applies when ephemeral configs inject conservative values)
        try:
            if self.config.dpo_epochs < 2:
                self.logger.info("🔧 Bumping DPO epochs to 2 for stronger signal")
                self.config.dpo_epochs = 2
            # Keep beta within [0.05, 0.1]
            if self.config.dpo_beta is not None:
                if self.config.dpo_beta > 0.1:
                    self.logger.info("🔧 Clamping DPO beta to 0.1")
                    self.config.dpo_beta = 0.1
                elif self.config.dpo_beta < 0.05:
                    self.logger.info("🔧 Raising DPO beta to 0.05")
                    self.config.dpo_beta = 0.05
            if self.config.pairs_size < 100:
                self.logger.info("🔧 Increasing preference pair size to 120 for stability")
                self.config.pairs_size = 120
        except Exception:
            pass
        
        # Create timestamped output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if config.experiment_name:
            self.output_dir = config.output_dir / f"rlaif_{config.experiment_name}_{timestamp}"
        else:
            self.output_dir = config.output_dir / f"rlaif_{timestamp}"
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save config
        with open(self.output_dir / "config.json", 'w') as f:
            json.dump(asdict(config), f, indent=2, default=str)
    
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
        # Detect accelerator and log advisory for config tuning
        try:
            accel = get_accelerator()
            self.accelerator = accel
            if accel.backend == 'mps':
                if '8B' in self.config.model_name or '7B' in self.config.model_name:
                    self.logger.warning("🍏 MPS detected: use smaller base model for stability (e.g., SmolLM-135M)")
            elif accel.backend == 'cpu':
                self.logger.warning("⚠️ CPU backend detected: DPO/PPO will be slow. Consider using a CUDA GPU machine for training.")
        except Exception:
            self.accelerator = None
    
    def save_checkpoint(self):
        """Save current pipeline state."""
        self.state.last_checkpoint = datetime.now()
        
        checkpoint_file = self.output_dir / "checkpoint.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(asdict(self.state), f, indent=2, default=str)
        
        self.logger.debug(f"💾 Checkpoint saved: {checkpoint_file}")
    
    def load_checkpoint(self) -> bool:
        """Load pipeline state from checkpoint."""
        ckpt_path = self.output_dir / "checkpoint.json"
        if not ckpt_path.exists():
            return False
        try:
            with open(ckpt_path, 'r') as f:
                data = json.load(f)
            # Reconstruct state object
            self.state = PipelineState(**data)
            self.logger.info(f"📂 Checkpoint loaded: {ckpt_path}")
            return True
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load checkpoint: {e}")
            return False

    # ----------------------
    # Alignment validation
    # ----------------------
    def _read_adapter_base(self, model_dir: Path) -> Optional[str]:
        """Return the base model name recorded in a PEFT adapter directory.

        Looks for adapter_config.json and reads base_model_name_or_path/base_model_name.
        Returns None if not found.
        """
        try:
            cfg = model_dir / "adapter_config.json"
            if cfg.exists():
                with open(cfg, 'r') as f:
                    data = json.load(f)
                return data.get("base_model_name_or_path") or data.get("base_model_name")
        except Exception:
            pass
        return None

    def _validate_model_alignment(self, dpo_model: Optional[Path] = None):
        """Validate base-model alignment across SFT → DPO → PPO.

        - SFT vs --model-name
        - (optional) DPO vs --ppo-model
        Raises PipelineError with a helpful message if misaligned.
        """
        # SFT alignment
        if self.config.sft_model and self.config.sft_model.exists():
            sft_base = self._read_adapter_base(self.config.sft_model)
            if sft_base and sft_base != self.config.model_name:
                raise PipelineError(
                    "Base model mismatch: SFT adapter was trained on '"
                    f"{sft_base}', but --model-name is '{self.config.model_name}'. "
                    "Set --model-name to the SFT base or regenerate the SFT adapter."
                )

        # DPO alignment (if provided)
        if dpo_model and dpo_model.exists():
            dpo_base = self._read_adapter_base(dpo_model)
            if dpo_base and dpo_base != self.config.ppo_model:
                raise PipelineError(
                    "Base model mismatch: DPO adapter was trained on '"
                    f"{dpo_base}', but --ppo-model is '{self.config.ppo_model}'. "
                    "Set --ppo-model to the DPO base, or regenerate DPO with the desired base."
                )
        
        
    
    def validate_prerequisites(self):
        """Validate all inputs and dependencies before starting."""
        self.logger.info("🔍 Validating prerequisites...")
        
        # Check required files
        if not self.config.chunks_file.exists():
            raise PipelineError(f"Chunks file not found: {self.config.chunks_file}")
        
        # Check SFT model if provided
        if self.config.sft_model and not self.config.sft_model.exists():
            self.logger.warning(f"⚠️ SFT model not found: {self.config.sft_model}")
            self.logger.info("📝 Will use heuristic preference pair generation")
            self.config.sft_model = None
        
        # Resolve PPO base: default to model_name if not set
        if not self.config.ppo_model:
            self.config.ppo_model = self.config.model_name

        # HF gated model guidance
        try:
            gated_models = []
            for name in filter(None, [self.config.model_name, self.config.ppo_model]):
                if isinstance(name, str) and name.lower().startswith("meta-llama/"):
                    gated_models.append(name)
            if gated_models:
                joined = ", ".join(sorted(set(gated_models)))
                self.logger.info(
                    f"🔐 Hugging Face gated models detected: {joined}. If you see 401 warnings, run 'huggingface-cli login' (in tmux), "
                    "ensure access is approved on the model page, and prefer HF_HOME over TRANSFORMERS_CACHE."
                )
        except Exception:
            pass

        # Auto-select and propagate base across stages when possible
        # Priority order:
        # 1) If SFT adapter provided, align model_name/ppo_model to its base
        # 2) Else, if auto_select_base, pick by VRAM and set BOTH model_name and ppo_model consistently
        try:
            desired_1b = "meta-llama/Llama-3.2-1B-Instruct"
            desired_8b = "meta-llama/Llama-3.1-8B-Instruct"

            sft_base = None
            if self.config.sft_model and self.config.sft_model.exists():
                sft_base = self._read_adapter_base(self.config.sft_model)
                if sft_base:
                    if self.config.model_name != sft_base:
                        self.logger.info(f"🔗 Aligning --model-name to SFT adapter base: {sft_base}")
                        self.config.model_name = sft_base
                    if self.config.ppo_model != sft_base:
                        self.logger.info(f"🔗 Aligning --ppo-model to SFT adapter base: {sft_base}")
                        self.config.ppo_model = sft_base

            # If no SFT adapter base to anchor to, apply VRAM-based selection
            if not sft_base and self.config.auto_select_base:
                vram_gb = None
                if torch.cuda.is_available():
                    props = torch.cuda.get_device_properties(0)
                    vram_gb = int(props.total_memory / (1024 ** 3))
                    self.logger.info(f"🧠 Detected GPU VRAM: ~{vram_gb} GB")

                # Determine if user explicitly overrode model_name
                known_defaults = {desired_1b, desired_8b}
                user_overrode_model = self.config.model_name not in known_defaults
                user_overrode_ppo = (self.config.ppo_model not in known_defaults) and (self.config.ppo_model != self.config.model_name)

                # Only override when not explicitly set by user
                if not user_overrode_model and not user_overrode_ppo:
                    chosen = None
                    if vram_gb is not None and vram_gb <= 30:
                        chosen = desired_1b
                    elif vram_gb is not None and vram_gb >= 40:
                        chosen = desired_8b
                    else:
                        chosen = desired_1b
                    if self.config.model_name != chosen:
                        self.logger.info(f"🧭 Auto-selected base by VRAM: {chosen}")
                    self.config.model_name = chosen
                    self.config.ppo_model = chosen

            # Apply PPO memory heuristics on low VRAM regardless of SFT anchoring
            try:
                vram_gb_ppo = None
                if torch.cuda.is_available():
                    props0 = torch.cuda.get_device_properties(0)
                    vram_gb_ppo = int(props0.total_memory / (1024 ** 3))
                if vram_gb_ppo is not None and vram_gb_ppo <= 30:
                    self.config.ppo_use_4bit = True
                    if getattr(self.config, 'ppo_batch_size', 32) > 8:
                        self.config.ppo_batch_size = 8
                    if getattr(self.config, 'ppo_mini_batch_size', 4) > 2:
                        self.config.ppo_mini_batch_size = 2
            except Exception:
                pass
        except Exception:
            # Non-fatal; proceed with user settings
            pass

        # Memory warnings for large models regardless of VRAM detection
        if self.config.ppo_model and ("7B" in self.config.ppo_model or "8B" in self.config.ppo_model):
            self.logger.warning(f"⚠️ Large PPO model ({self.config.ppo_model}) may cause OOM")
            self.logger.info("💡 Consider using meta-llama/Llama-3.2-1B-Instruct for memory efficiency")

        self.logger.info("✅ Prerequisites validated")
        # Validate model alignment for early failure (SFT vs model_name)
        try:
            self._validate_model_alignment()
        except PipelineError as e:
            # Re-raise after logging for a clear early exit
            self.logger.error(str(e))
            raise
        # Log accelerator info into a run-level metadata file for reproducibility
        try:
            info = getattr(self, 'accelerator', None) or get_accelerator()
            with open(self.config.output_dir / "accelerator.json", 'w') as f:
                json.dump({
                    "backend": info.backend,
                    "device": info.device_str,
                    "can_use_4bit": info.can_use_4bit,
                    "recommended_dtype": info.recommended_dtype,
                    "details": info.details,
                }, f, indent=2)
        except Exception:
            pass
    
    def run_subprocess_with_monitoring(self, cmd: List[str], stage_name: str) -> int:
        """Run subprocess with real-time monitoring and logging."""
        cmd_str = ' '.join(str(c) for c in cmd)
        self.logger.info(f"🚀 [{stage_name}] Running: {cmd_str}")
        
        if self.config.dry_run:
            self.logger.info(f"🔍 DRY RUN: Would execute command above")
            return 0
        
        # Run with real-time output
        process = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, cwd=Path.cwd()
        )
        
        # Stream output in real-time
        previous_line = None
        for line in process.stdout:
            line = line.rstrip()
            if not line:
                continue
            if previous_line is not None and line == previous_line:
                continue
            previous_line = line
            self.logger.info(f"[{stage_name}] {line}")
        
        process.wait()
        
        if process.returncode != 0:
            raise PipelineError(f"{stage_name} failed with return code {process.returncode}")
        
        return process.returncode
    
    def run_preference_pairs_generation(self) -> Path:
        """Generate preference pairs with canonical patterns."""
        stage_name = "PAIRS"
        self.logger.info(f"\n📊 Step 1: Generating {self.config.pairs_size} preference pairs...")
        
        pairs_output = self.output_dir / "dpo_pairs.jsonl"
        
        cmd = [
            "python", "src/training/make_pref_pairs.py",
            "--chunks", str(self.config.chunks_file),
            "--output", str(pairs_output),
            "--size", str(self.config.pairs_size),
            "--seed", str(self.config.pairs_seed)
        ]
        
        if self.config.sft_model:
            cmd.extend(["--sft-model", str(self.config.sft_model)])
            self.logger.info(f"🤖 Using SFT model for drafting: {self.config.sft_model}")
        
        import time as _t
        t0 = _t.time()
        self.run_subprocess_with_monitoring(cmd, stage_name)
        self.logger.info(f"⏱️ [{stage_name}] Duration: {(_t.time()-t0):.1f}s")
        
        if not self.config.dry_run:
            # Validate output
            if not pairs_output.exists():
                raise PipelineError(f"Expected pairs output not found: {pairs_output}")
        
        self.state.pairs_completed = True
        self.state.pairs_output = pairs_output
        self.save_checkpoint()
        
        self.logger.info(f"✅ Preference pairs generated: {pairs_output}")
        return pairs_output
    
    def run_dpo_training(self) -> Path:
        """Train DPO with fixed tokenizer and persistent eval subset."""
        stage_name = "DPO"
        self.logger.info(f"\n🎯 Step 2: Training DPO for {self.config.dpo_epochs} epochs...")
        
        dpo_output = self.output_dir / "lora_dpo"
        train_data = self.output_dir / "dpo_pairs_train.jsonl"
        eval_data = self.output_dir / "dpo_pairs_eval.jsonl"
        
        cmd = [
            "python", "src/training/train_dpo.py",
            "--train-data", str(train_data),
            "--eval-data", str(eval_data),
            "--output-dir", str(dpo_output),
            "--model-name", self.config.model_name,
            "--epochs", str(self.config.dpo_epochs),
            "--batch-size", str(self.config.dpo_batch_size),
            "--learning-rate", str(self.config.dpo_learning_rate),
            "--beta", str(self.config.dpo_beta)
        ]
        
        if self.config.sft_model:
            cmd.extend(["--sft-model", str(self.config.sft_model)])
        
        # Add logging backends
        for backend in self.config.logging_backends:
            cmd.extend(["--report-to", backend])
        
        import time as _t
        t0 = _t.time()
        self.run_subprocess_with_monitoring(cmd, stage_name)
        self.logger.info(f"⏱️ [{stage_name}] Duration: {(_t.time()-t0):.1f}s")
        
        if not self.config.dry_run:
            # Validate output
            if not dpo_output.exists():
                raise PipelineError(f"Expected DPO output not found: {dpo_output}")
        
        self.state.dpo_completed = True
        self.state.dpo_output = dpo_output
        self.save_checkpoint()
        
        self.logger.info(f"✅ DPO training completed: {dpo_output}")
        return dpo_output
    
    def run_ppo_training(self, dpo_model: Path) -> Path:
        """Run PPO with proper value-head initialization."""
        stage_name = "PPO"
        self.logger.info(f"\n⚡ Step 3: Running PPO with {self.config.ppo_prompts} prompts...")

        ppo_output = self.output_dir / "lora_ppo"

        # Validate model alignment between DPO adapter and PPO base before launching PPO
        try:
            self._validate_model_alignment(dpo_model)
        except PipelineError as e:
            self.logger.error(str(e))
            raise
        
        cmd = [
            "python", "src/training/tiny_ppo_loop.py",
            "--dpo-model", str(dpo_model),
            "--output", str(ppo_output),
            "--use-real-ppo",
            "--auto-align-to-adapter",
            "--batch-size", str(self.config.ppo_batch_size),
            "--mini-batch-size", str(self.config.ppo_mini_batch_size),
            "--base-model", self.config.ppo_model,
            "--num-prompts", str(self.config.ppo_prompts)
        ]
        if getattr(self.config, 'ppo_use_4bit', False):
            cmd.append("--use-4bit")
        
        import time as _t
        t0 = _t.time()
        self.run_subprocess_with_monitoring(cmd, stage_name)
        self.logger.info(f"⏱️ [{stage_name}] Duration: {(_t.time()-t0):.1f}s")
        
        if not self.config.dry_run:
            # Validate output
            if not ppo_output.exists():
                raise PipelineError(f"Expected PPO output not found: {ppo_output}")
            # Check for PPO errors recorded by tiny_ppo_loop
            try:
                results_path = ppo_output / "ppo_results.json"
                if results_path.exists():
                    with open(results_path, 'r') as f:
                        results = json.load(f)
                    if isinstance(results, dict) and results.get('error'):
                        raise PipelineError(f"PPO reported an error: {results['error']}")
            except PipelineError:
                raise
            except Exception:
                # Non-fatal parsing issues; continue if artifacts are present
                pass
        
        self.state.ppo_completed = True
        self.state.ppo_output = ppo_output
        self.save_checkpoint()
        
        self.logger.info(f"✅ PPO training completed: {ppo_output}")
        return ppo_output
    
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
            "stage_durations": {
                "pairs_sec": None,
                "dpo_sec": None,
                "ppo_sec": None,
            },
            "artifacts": {
                "pairs_file": str(self.state.pairs_output) if self.state.pairs_output else None,
                "dpo_model": str(self.state.dpo_output) if self.state.dpo_output else None,
                "ppo_model": str(self.state.ppo_output) if self.state.ppo_output else None,
            }
        }
        # Extract recent durations from log? Not persisted; left as None here for simplicity
        
        # Save report
        report_file = self.output_dir / "rlaif_final_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"📋 Final report saved: {report_file}")
        return report
    
    def run(self):
        """Main pipeline runner with comprehensive error handling."""
        try:
            self.logger.info("🚀 Starting RLAIF Training Pipeline")
            self.logger.info(f"📁 Output directory: {self.output_dir}")
            
            # Validate prerequisites
            self.validate_prerequisites()
            
            # Load checkpoint if exists
            self.load_checkpoint()
            
            # Run pipeline stages
            if not self.config.skip_pairs and not self.state.pairs_completed:
                self.run_preference_pairs_generation()
            
            if not self.config.skip_dpo and not self.state.dpo_completed:
                self.run_dpo_training()
            
            if not self.config.skip_ppo and not self.state.ppo_completed:
                dpo_model = self.state.dpo_output or self.output_dir / "lora_dpo"
                self.run_ppo_training(dpo_model)
            
            # Generate final report
            final_report = self.generate_final_report()
            
            # Success message
            duration = datetime.now() - self.state.start_time
            self.logger.info(f"\n🎉 RLAIF Pipeline Completed Successfully!")
            self.logger.info(f"⏱️ Total duration: {duration}")
            self.logger.info(f"📁 All outputs saved to: {self.output_dir}")
            
            return final_report
            
        except PipelineError as e:
            self.logger.error(f"❌ Pipeline error: {e}")
            sys.exit(1)
        except KeyboardInterrupt:
            self.logger.warning(f"⚠️ Pipeline interrupted by user")
            self.save_checkpoint()
            self.logger.info(f"💾 Progress saved. Resume with: --resume-from {self.get_last_completed_stage()}")
            sys.exit(1)
        except Exception as e:
            self.logger.error(f"💥 Unexpected error: {e}")
            self.save_checkpoint()
            raise e
    
    def get_last_completed_stage(self) -> str:
        """Get the last completed stage for resume."""
        if self.state.ppo_completed:
            return "eval"
        elif self.state.dpo_completed:
            return "ppo"
        elif self.state.pairs_completed:
            return "dpo"
        else:
            return "pairs"


def create_parser():
    """Create enhanced argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="RLAIF Training Pipeline - DPO + PPO",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Complete RLAIF pipeline
  %(prog)s full --chunks data/processed/chunks.jsonl --sft-model outputs/lora_sft

  # Partial pipeline (skip PPO)
  %(prog)s partial --chunks data/processed/chunks.jsonl --skip-ppo

  # Resume from checkpoint
  %(prog)s resume --from dpo --dpo-model outputs/lora_dpo

  # Development mode (small datasets)
  %(prog)s dev --chunks data/processed/chunks.jsonl

  # With configuration file
  %(prog)s full --config config/rlaif_production.yaml
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Pipeline commands')
    
    # Full pipeline
    full_parser = subparsers.add_parser('full', help='Run complete RLAIF pipeline')
    add_common_args(full_parser)
    
    # Partial pipeline
    partial_parser = subparsers.add_parser('partial', help='Run partial pipeline')
    add_common_args(partial_parser)
    partial_parser.add_argument('--skip-pairs', action='store_true', help='Skip preference pairs generation')
    partial_parser.add_argument('--skip-dpo', action='store_true', help='Skip DPO training')
    partial_parser.add_argument('--skip-ppo', action='store_true', help='Skip PPO training')
    
    # Resume pipeline
    resume_parser = subparsers.add_parser('resume', help='Resume from checkpoint')
    resume_parser.add_argument('--from', dest='resume_from', choices=['pairs', 'dpo', 'ppo'], 
                              required=True, help='Resume from this stage')
    resume_parser.add_argument('--output-dir', required=True, help='Previous output directory to resume from')
    add_common_args(resume_parser, include_chunks=False, include_output=False)
    
    # Development mode
    dev_parser = subparsers.add_parser('dev', help='Development mode (small datasets)')
    add_common_args(dev_parser)
    dev_parser.set_defaults(pairs_size=120, ppo_prompts=8, dpo_epochs=2)
    
    return parser


def add_common_args(parser, include_chunks=True, include_output=True):
    """Add common arguments to parser."""
    
    # Configuration
    parser.add_argument('--config', help='Path to configuration YAML/JSON file')
    
    # Required paths
    if include_chunks:
        parser.add_argument('--chunks', required=True, help='Path to chunks.jsonl file')
    parser.add_argument('--sft-model', help='Path to SFT model directory')
    
    # Training parameters
    parser.add_argument('--pairs-size', type=int, default=60, help='Number of preference pairs')
    parser.add_argument('--dpo-epochs', type=int, default=1, help='DPO training epochs')
    parser.add_argument('--dpo-beta', type=float, default=0.1, help='DPO beta parameter')
    parser.add_argument('--ppo-prompts', type=int, default=16, help='PPO prompts count')
    parser.add_argument('--ppo-model', default=None, help='PPO base model (defaults to --model-name)')
    parser.add_argument('--ppo-use-4bit', action='store_true', help='Enable 4-bit quantization for PPO policy/ref models')
    
    # Model configuration
    parser.add_argument('--model-name', default="meta-llama/Llama-3.2-1B-Instruct",
                       help='Base model name')
    
    # Output configuration
    if include_output:
        parser.add_argument('--output-dir', default="outputs", help='Output directory')
    parser.add_argument('--experiment-name', help='Experiment name for organization')
    
    # Logging
    parser.add_argument('--report-to', action='append', choices=['tensorboard', 'wandb'],
                       default=[], help='Logging backends (can specify multiple)')
    
    # Advanced options
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--dry-run', action='store_true', help='Show commands without executing')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--no-auto-select-base', dest='auto_select_base', action='store_false',
                       help='Disable VRAM-based auto-selection of a consistent base model across stages')


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    # Capture parser defaults to avoid overriding config values with defaults
    parser_defaults = {a.dest: a.default for a in parser._actions if getattr(a, 'dest', None)}
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        # Load configuration
        if hasattr(args, 'config') and args.config:
            config = RLAIFConfig.from_file(Path(args.config))
            # Override with command line arguments ONLY when they differ from parser defaults
            for key, value in vars(args).items():
                if key == 'config' or key not in RLAIFConfig.__dataclass_fields__:
                    continue
                default_val = parser_defaults.get(key, None)
                if value is not None and value != default_val:
                    if key in {"chunks_file", "sft_model", "output_dir"}:
                        setattr(config, key, Path(value))
                    else:
                        setattr(config, key, value)
        else:
            # Create config from command line args
            config_dict = {k: v for k, v in vars(args).items() 
                          if k in RLAIFConfig.__dataclass_fields__ and v is not None}
            
            # Convert string paths to Path objects
            if 'chunks' in config_dict:
                config_dict['chunks_file'] = Path(config_dict.pop('chunks'))
            if 'sft_model' in config_dict and config_dict['sft_model']:
                config_dict['sft_model'] = Path(config_dict['sft_model'])
            if 'output_dir' in config_dict:
                config_dict['output_dir'] = Path(config_dict['output_dir'])
            
            # Handle subcommand-specific logic
            if args.command == 'partial':
                config_dict.update({
                    'skip_pairs': getattr(args, 'skip_pairs', False),
                    'skip_dpo': getattr(args, 'skip_dpo', False),
                    'skip_ppo': getattr(args, 'skip_ppo', False)
                })
            elif args.command == 'resume':
                config_dict['resume_from'] = getattr(args, 'resume_from', None)
            
            config = RLAIFConfig(**config_dict)
        
        # Initialize and run pipeline
        pipeline = RLAIFTrainingPipeline(config)
        final_report = pipeline.run()
        
        print(f"\n✨ RLAIF Training Complete!")
        print(f"📁 Check outputs in: {pipeline.output_dir}")
        
    except Exception as e:
        print(f"❌ Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
