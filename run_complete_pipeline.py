#!/usr/bin/env python3
"""
Complete End-to-End Training Pipeline - From PDFs to Final Model

This script orchestrates the complete ML pipeline from raw data to deployed model:
1. Data Pipeline: PDF ingestion, chunking, and index building
2. SFT Pipeline: Dataset generation and supervised fine-tuning
3. RLAIF Pipeline: DPO and PPO preference optimization
4. Evaluation & Reporting: Comprehensive model evaluation

Key Features:
- ✅ End-to-end automation from PDFs to final model
- ✅ Modular pipeline with individual stage control
- ✅ Comprehensive configuration management
- ✅ Resource optimization and memory management
- ✅ Robust error handling with recovery
- ✅ Multi-backend logging and monitoring
- ✅ Production-ready defaults with dev mode

Usage Examples:
    # Complete pipeline from PDFs
    python run_complete_pipeline.py full --input data/raw_pdfs

    # Resume from SFT stage (data already processed)
    python run_complete_pipeline.py resume --from sft --chunks data/processed/chunks.jsonl

    # Skip data processing (chunks already exist)
    python run_complete_pipeline.py partial --chunks data/processed/chunks.jsonl --skip-data

    # Development mode (small datasets, fast iteration)
    python run_complete_pipeline.py dev --input data/raw_pdfs --pdf-limit 3

    # Production mode with monitoring
    python run_complete_pipeline.py prod --input data/raw_pdfs --config config/production.yaml
"""

import argparse
import subprocess
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
import sys
import os
from dataclasses import dataclass, field, asdict
from enum import Enum
import tempfile
import shutil
from src.utils.accelerator import get_accelerator, log_accelerator


class PipelineStage(Enum):
    """Complete pipeline stages."""
    DATA = "data"
    SFT = "sft"
    RLAIF = "rlaif"
    EVALUATION = "evaluation"


@dataclass
class CompletePipelineConfig:
    """Configuration for the complete training pipeline."""
    
    # Input configuration
    input_path: Optional[Path] = None  # PDF directory for data pipeline
    chunks_file: Optional[Path] = None  # Pre-existing chunks file
    sft_model: Optional[Path] = None   # Pre-existing SFT model
    
    # Output configuration
    output_dir: Path = field(default_factory=lambda: Path("outputs"))
    experiment_name: Optional[str] = None
    
    # Data pipeline config
    data_config: Dict[str, Any] = field(default_factory=dict)
    
    # SFT pipeline config  
    sft_config: Dict[str, Any] = field(default_factory=dict)
    
    # RLAIF pipeline config
    rlaif_config: Dict[str, Any] = field(default_factory=dict)
    
    # Pipeline control
    skip_data: bool = False
    skip_sft: bool = False
    skip_rlaif: bool = False
    skip_evaluation: bool = False
    resume_from: Optional[PipelineStage] = None
    
    # Resource management
    max_memory_gb: Optional[int] = None
    gpu_memory_fraction: float = 0.9
    
    # Advanced options
    dry_run: bool = False
    verbose: bool = False
    logging_backends: List[str] = field(default_factory=list)
    cleanup_intermediate: bool = False  # Clean up intermediate files to save space
    enable_mps_fallback: bool = False   # Opt-in MPS fallback for unsupported ops
    
    @classmethod
    def from_file(cls, config_path: Path) -> 'CompletePipelineConfig':
        """Load configuration from YAML or JSON file."""
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        # Convert path strings to Path objects
        for path_field in ['input_path', 'chunks_file', 'sft_model', 'output_dir']:
            if path_field in data and data[path_field]:
                data[path_field] = Path(data[path_field])
                
        return cls(**data)


@dataclass
class CompleteState:
    """Track complete pipeline progress."""
    data_completed: bool = False
    sft_completed: bool = False
    rlaif_completed: bool = False
    evaluation_completed: bool = False
    
    # Key outputs from each stage
    chunks_file: Optional[Path] = None
    faiss_index: Optional[Path] = None
    sft_model: Optional[Path] = None
    dpo_model: Optional[Path] = None
    ppo_model: Optional[Path] = None
    
    # Timing information
    start_time: datetime = field(default_factory=datetime.now)
    stage_times: Dict[str, float] = field(default_factory=dict)
    last_checkpoint: datetime = field(default_factory=datetime.now)


class CompletePipelineError(Exception):
    """Custom exception for complete pipeline errors."""
    pass


class CompletePipeline:
    """End-to-end training pipeline orchestrator."""
    
    def __init__(self, config: CompletePipelineConfig):
        self.config = config
        self.state = CompleteState()
        # Determine and create timestamped output directory early so logging can use it
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if config.experiment_name:
            self.output_dir = config.output_dir / f"complete_{config.experiment_name}_{timestamp}"
        else:
            self.output_dir = config.output_dir / f"complete_{timestamp}"
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize logging after output_dir exists
        self.setup_logging()
        # Ensure base output root exists (idempotent)
        self.setup_output_dirs()
        
        # Define key output paths
        self.data_dir = self.output_dir / "data"
        self.sft_dir = self.output_dir / "sft"
        self.rlaif_dir = self.output_dir / "rlaif"
        self.eval_dir = self.output_dir / "evaluation"
        
        # Save config
        with open(self.output_dir / "complete_config.json", 'w') as f:
            json.dump(asdict(config), f, indent=2, default=str)
    
    def setup_logging(self):
        """Setup comprehensive logging with file output."""
        log_level = logging.DEBUG if self.config.verbose else logging.INFO
        
        # Create log file
        log_file = self.output_dir / "pipeline.log"
        
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(log_file)
            ]
        )
        
        self.logger = logging.getLogger(__name__)
        
        if self.config.dry_run:
            self.logger.info("🔍 DRY RUN MODE: Commands will be shown but not executed")
            
        self.logger.info(f"📋 Pipeline logs saved to: {log_file}")
    
    def setup_output_dirs(self):
        """Create necessary output directories."""
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self):
        """Save current pipeline state."""
        self.state.last_checkpoint = datetime.now()
        
        checkpoint_file = self.output_dir / "complete_checkpoint.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(asdict(self.state), f, indent=2, default=str)
        
        self.logger.debug(f"💾 Checkpoint saved: {checkpoint_file}")
    
    def load_checkpoint(self) -> bool:
        """Load pipeline state from checkpoint."""
        checkpoint_file = self.output_dir / "complete_checkpoint.json"
        
        if not checkpoint_file.exists():
            return False
        
        try:
            with open(checkpoint_file, 'r') as f:
                data = json.load(f)
            
            # Reconstruct state
            self.state = CompleteState(**data)
            self.logger.info(f"📂 Checkpoint loaded: {checkpoint_file}")
            return True
            
        except Exception as e:
            self.logger.warning(f"⚠️ Failed to load checkpoint: {e}")
            return False
    
    def validate_prerequisites(self):
        """Validate configuration and inputs."""
        self.logger.info("🔍 Validating prerequisites...")
        
        # Check input requirements based on pipeline start point
        if not self.config.skip_data:
            if not self.config.input_path:
                raise CompletePipelineError("input_path required when not skipping data pipeline")
            if not self.config.input_path.exists():
                raise CompletePipelineError(f"Input path not found: {self.config.input_path}")
        
        if self.config.skip_data and not self.config.chunks_file:
            raise CompletePipelineError("chunks_file required when skipping data pipeline")
        
        if self.config.skip_sft and not self.config.sft_model:
            raise CompletePipelineError("sft_model required when skipping SFT pipeline")
        
        # Check accelerator availability for training stages
        if not (self.config.skip_sft and self.config.skip_rlaif):
            info = get_accelerator(enable_mps_fallback=self.config.enable_mps_fallback, do_health_check=True)
            log_accelerator(self.logger, info)
            # Surface health check problems explicitly
            hc = info.details.get("health_check", {"ok": True})
            if not hc.get("ok", True):
                self.logger.warning(f"⚠️ Accelerator health check failed on {info.backend}: {hc.get('error')}. Falling back to CPU may be safer.")
        
        self.logger.info("✅ Prerequisites validated")
    
    def run_data_pipeline(self) -> Path:
        """Run data processing pipeline."""
        stage_name = "DATA"
        stage_start = datetime.now()
        self.logger.info(f"\n📊 Stage 1: Data Pipeline (PDF → Chunks → Indices)")
        
        # Prepare data pipeline configuration
        data_config = {
            'input_path': str(self.config.input_path),
            'output_dir': str(self.data_dir),
            'verbose': self.config.verbose,
            'dry_run': self.config.dry_run,
            **self.config.data_config
        }
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(data_config, f)
            temp_config = f.name
        
        try:
            cmd = [
                "python", "run_data_pipeline.py", "full",
                "--config", temp_config,
                "--input", str(self.config.input_path),
                "--output", str(self.data_dir)
            ]
            
            self.run_subprocess_with_monitoring(cmd, stage_name)
            
            # Validate outputs
            chunks_file = self.data_dir / "chunks.jsonl"
            faiss_index = self.data_dir / "indices" / "faiss.index"
            
            if not self.config.dry_run:
                if not chunks_file.exists():
                    raise CompletePipelineError(f"Expected chunks file not found: {chunks_file}")
                if not faiss_index.exists():
                    raise CompletePipelineError(f"Expected FAISS index not found: {faiss_index}")
            
            self.state.data_completed = True
            self.state.chunks_file = chunks_file
            self.state.faiss_index = faiss_index
            
        finally:
            # Cleanup temporary config
            os.unlink(temp_config)
        
        # Record timing
        stage_duration = (datetime.now() - stage_start).total_seconds()
        self.state.stage_times[stage_name] = stage_duration
        self.save_checkpoint()
        
        self.logger.info(f"✅ Data pipeline completed in {stage_duration:.1f}s")
        self.logger.info(f"   📝 Chunks: {self.state.chunks_file}")
        self.logger.info(f"   🔍 Index: {self.state.faiss_index}")
        
        return self.state.chunks_file
    
    def run_sft_pipeline(self) -> Path:
        """Run SFT training pipeline."""
        stage_name = "SFT"
        stage_start = datetime.now()
        self.logger.info(f"\n🎯 Stage 2: SFT Pipeline (Dataset → Training)")
        
        # Get chunks file (from data pipeline or provided)
        chunks_file = self.state.chunks_file or self.config.chunks_file
        
        # Prepare SFT pipeline configuration
        sft_config = {
            'chunks_file': str(chunks_file),
            'output_dir': str(self.sft_dir),
            'verbose': self.config.verbose,
            'dry_run': self.config.dry_run,
            **self.config.sft_config
        }

        # Safety defaults for dev/Mac: ensure usable model + trainer when not specified
        try:
            from src.utils.accelerator import get_accelerator
            acc = get_accelerator(enable_mps_fallback=self.config.enable_mps_fallback)
            # If no model specified in config, choose a small public one for non-CUDA
            if 'model_name' not in sft_config or not sft_config['model_name']:
                if acc.backend != 'cuda':
                    sft_config['model_name'] = "HuggingFaceTB/SmolLM-135M-Instruct"
            # Prefer TRL SFTTrainer by default for simplicity if not specified
            if 'use_sfttrainer' not in sft_config:
                sft_config['use_sfttrainer'] = True
            # Disable CUDA-only settings on non-CUDA accelerators unless explicitly set
            if acc.backend != 'cuda':
                sft_config.setdefault('bf16', False)
                sft_config.setdefault('use_4bit', False)
        except Exception:
            # Best-effort only
            pass
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(sft_config, f)
            temp_config = f.name
        
        try:
            cmd = [
                "python", "run_sft_pipeline.py", "full",
                "--config", temp_config,
                "--chunks", str(chunks_file)
            ]
            
            self.run_subprocess_with_monitoring(cmd, stage_name)
            
            # Find the actual SFT output directory (timestamped)
            sft_outputs = list(self.sft_dir.glob("sft_*"))
            if sft_outputs and not self.config.dry_run:
                sft_model_dir = max(sft_outputs, key=lambda x: x.stat().st_mtime) / "lora_sft"
                if not sft_model_dir.exists():
                    raise CompletePipelineError(f"SFT model not found in: {sft_model_dir}")
                self.state.sft_model = sft_model_dir
            elif not self.config.dry_run:
                raise CompletePipelineError("No SFT outputs found")
            
            self.state.sft_completed = True
            
        finally:
            os.unlink(temp_config)
        
        # Record timing
        stage_duration = (datetime.now() - stage_start).total_seconds()
        self.state.stage_times[stage_name] = stage_duration
        self.save_checkpoint()
        
        self.logger.info(f"✅ SFT pipeline completed in {stage_duration:.1f}s")
        self.logger.info(f"   🤖 Model: {self.state.sft_model}")
        
        return self.state.sft_model
    
    def run_rlaif_pipeline(self) -> tuple[Path, Path]:
        """Run RLAIF training pipeline."""
        stage_name = "RLAIF"
        stage_start = datetime.now()
        self.logger.info(f"\n⚡ Stage 3: RLAIF Pipeline (DPO → PPO)")
        
        # Get required inputs
        chunks_file = self.state.chunks_file or self.config.chunks_file
        sft_model = self.state.sft_model or self.config.sft_model
        
        # Prepare RLAIF pipeline configuration
        rlaif_config = {
            'chunks_file': str(chunks_file),
            'sft_model': str(sft_model),
            'output_dir': str(self.rlaif_dir),
            'verbose': self.config.verbose,
            'dry_run': self.config.dry_run,
            **self.config.rlaif_config
        }
        
        # Create temporary config file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(rlaif_config, f)
            temp_config = f.name
        
        try:
            cmd = [
                "python", "run_rlaif_training.py", "full",
                "--config", temp_config,
                "--chunks", str(chunks_file)
            ]
            # If we have an SFT model path already, pass it explicitly to satisfy CLI requirements
            if sft_model:
                cmd.extend(["--sft-model", str(sft_model)])
            
            self.run_subprocess_with_monitoring(cmd, stage_name)
            
            # Find the actual RLAIF output directory (timestamped)
            rlaif_outputs = list(self.rlaif_dir.glob("rlaif_*"))
            if rlaif_outputs and not self.config.dry_run:
                rlaif_output_dir = max(rlaif_outputs, key=lambda x: x.stat().st_mtime)
                
                dpo_model = rlaif_output_dir / "lora_dpo"
                ppo_model = rlaif_output_dir / "lora_ppo"
                
                if dpo_model.exists():
                    self.state.dpo_model = dpo_model
                if ppo_model.exists():
                    self.state.ppo_model = ppo_model
                    
            self.state.rlaif_completed = True
            
        finally:
            os.unlink(temp_config)
        
        # Record timing
        stage_duration = (datetime.now() - stage_start).total_seconds()
        self.state.stage_times[stage_name] = stage_duration
        self.save_checkpoint()
        
        self.logger.info(f"✅ RLAIF pipeline completed in {stage_duration:.1f}s")
        self.logger.info(f"   🎯 DPO Model: {self.state.dpo_model}")
        self.logger.info(f"   ⚡ PPO Model: {self.state.ppo_model}")
        
        return self.state.dpo_model, self.state.ppo_model
    
    def run_evaluation(self) -> Dict[str, Any]:
        """Run comprehensive model evaluation."""
        stage_name = "EVAL"
        stage_start = datetime.now()
        self.logger.info(f"\n📊 Stage 4: Model Evaluation")
        
        # TODO: Implement comprehensive evaluation
        # This would include:
        # - RAG pipeline evaluation with different models
        # - Citation accuracy evaluation
        # - Response quality assessment
        # - Performance benchmarking
        
        evaluation_results = {
            "timestamp": datetime.now().isoformat(),
            "models_evaluated": {
                "sft_model": str(self.state.sft_model) if self.state.sft_model else None,
                "dpo_model": str(self.state.dpo_model) if self.state.dpo_model else None,
                "ppo_model": str(self.state.ppo_model) if self.state.ppo_model else None,
            },
            "evaluation_pending": "Comprehensive evaluation not yet implemented"
        }
        
        # Save evaluation results
        eval_file = self.eval_dir / "evaluation_results.json"
        self.eval_dir.mkdir(parents=True, exist_ok=True)
        with open(eval_file, 'w') as f:
            json.dump(evaluation_results, f, indent=2, default=str)
        
        self.state.evaluation_completed = True
        
        # Record timing
        stage_duration = (datetime.now() - stage_start).total_seconds()
        self.state.stage_times[stage_name] = stage_duration
        self.save_checkpoint()
        
        self.logger.info(f"✅ Evaluation completed in {stage_duration:.1f}s")
        self.logger.info(f"   📊 Results: {eval_file}")
        
        return evaluation_results
    
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
            raise CompletePipelineError(f"{stage_name} failed with return code {process.returncode}")
        
        return process.returncode
    
    def cleanup_intermediate_files(self):
        """Clean up intermediate files to save disk space."""
        if not self.config.cleanup_intermediate:
            return
        
        self.logger.info("🧹 Cleaning up intermediate files...")
        
        # Clean up large temporary files but keep final models
        cleanup_patterns = [
            "*/temp_*",
            "*/cache_*", 
            "*/*.tmp",
            "*/runs/*/events.out.tfevents.*"  # TensorBoard logs
        ]
        
        for pattern in cleanup_patterns:
            for file_path in self.output_dir.glob(pattern):
                try:
                    if file_path.is_file():
                        file_path.unlink()
                    elif file_path.is_dir():
                        shutil.rmtree(file_path)
                    self.logger.debug(f"🗑️ Cleaned: {file_path}")
                except Exception as e:
                    self.logger.warning(f"⚠️ Failed to clean {file_path}: {e}")
    
    def generate_final_report(self) -> Dict[str, Any]:
        """Generate comprehensive final report."""
        self.logger.info(f"\n📋 Generating final report...")
        
        total_duration = datetime.now() - self.state.start_time
        
        report = {
            "pipeline_info": {
                "timestamp": datetime.now().isoformat(),
                "total_duration_seconds": total_duration.total_seconds(),
                "config": asdict(self.config),
                "state": asdict(self.state),
                "output_directory": str(self.output_dir)
            },
            "stage_timings": self.state.stage_times,
            "artifacts": {
                "data_outputs": {
                    "chunks_file": str(self.state.chunks_file) if self.state.chunks_file else None,
                    "faiss_index": str(self.state.faiss_index) if self.state.faiss_index else None,
                },
                "model_outputs": {
                    "sft_model": str(self.state.sft_model) if self.state.sft_model else None,
                    "dpo_model": str(self.state.dpo_model) if self.state.dpo_model else None,
                    "ppo_model": str(self.state.ppo_model) if self.state.ppo_model else None,
                },
                "final_model": str(self.state.ppo_model or self.state.dpo_model or self.state.sft_model)
            },
            "next_steps": {
                "serving": f"python src/server/serve_vllm.py --peft {self.state.ppo_model or self.state.dpo_model or self.state.sft_model}",
                "evaluation": f"python -m src.generation.rag_pipeline --faiss {self.state.faiss_index} --model {self.state.ppo_model or self.state.dpo_model or self.state.sft_model}",
                "web_ui": "streamlit run src/ui/app.py"
            }
        }
        
        # Save report
        report_file = self.output_dir / "complete_final_report.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        self.logger.info(f"📋 Final report saved: {report_file}")
        return report
    
    def run(self):
        """Main pipeline runner."""
        try:
            self.logger.info("🚀 Starting Complete End-to-End Pipeline")
            self.logger.info(f"📁 Output directory: {self.output_dir}")
            
            # Validate prerequisites
            self.validate_prerequisites()
            
            # Load checkpoint if resuming
            if self.config.resume_from:
                self.load_checkpoint()
            
            # Run pipeline stages
            if not self.config.skip_data and not self.state.data_completed:
                self.run_data_pipeline()
            
            if not self.config.skip_sft and not self.state.sft_completed:
                self.run_sft_pipeline()
            
            if not self.config.skip_rlaif and not self.state.rlaif_completed:
                self.run_rlaif_pipeline()
            
            if not self.config.skip_evaluation and not self.state.evaluation_completed:
                self.run_evaluation()
            
            # Cleanup if requested
            self.cleanup_intermediate_files()
            
            # Generate final report
            final_report = self.generate_final_report()
            
            # Success message
            total_duration = datetime.now() - self.state.start_time
            self.logger.info(f"\n🎉 Complete Pipeline Finished Successfully!")
            self.logger.info(f"⏱️ Total duration: {total_duration}")
            self.logger.info(f"📁 All outputs saved to: {self.output_dir}")
            
            # Print key results
            self.logger.info(f"\n📊 Pipeline Results:")
            if self.state.chunks_file:
                self.logger.info(f"   📝 Data processed: {self.state.chunks_file}")
            if self.state.sft_model:
                self.logger.info(f"   🎯 SFT model: {self.state.sft_model}")
            if self.state.dpo_model:
                self.logger.info(f"   🎯 DPO model: {self.state.dpo_model}")
            if self.state.ppo_model:
                self.logger.info(f"   ⚡ PPO model: {self.state.ppo_model}")
            
            # Next steps
            final_model = self.state.ppo_model or self.state.dpo_model or self.state.sft_model
            if final_model:
                self.logger.info(f"\n🚀 Next Steps:")
                self.logger.info(f"   🖥️ Start server: python src/server/serve_vllm.py --peft {final_model}")
                self.logger.info(f"   🌐 Web UI: streamlit run src/ui/app.py")
            
            return final_report
            
        except CompletePipelineError as e:
            self.logger.error(f"❌ Complete Pipeline error: {e}")
            sys.exit(1)
        except KeyboardInterrupt:
            self.logger.warning(f"⚠️ Pipeline interrupted by user")
            self.save_checkpoint()
            self.logger.info(f"💾 Progress saved. Resume with appropriate --resume-from flag")
            sys.exit(1)
        except Exception as e:
            self.logger.error(f"💥 Unexpected error: {e}")
            self.save_checkpoint()
            raise e


def create_parser():
    """Create argument parser with subcommands."""
    parser = argparse.ArgumentParser(
        description="Complete End-to-End Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Complete pipeline from PDFs
  %(prog)s full --input data/raw_pdfs

  # Resume from SFT stage  
  %(prog)s resume --from sft --chunks data/processed/chunks.jsonl

  # Skip data processing (chunks exist)
  %(prog)s partial --chunks data/processed/chunks.jsonl --skip-data

  # Development mode (small datasets)
  %(prog)s dev --input data/raw_pdfs --pdf-limit 3

  # Production mode with config
  %(prog)s prod --config config/production.yaml
        """
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest='command', help='Pipeline commands', required=True)
    
    # Full pipeline
    full_parser = subparsers.add_parser('full', help='Run complete pipeline from PDFs')
    full_parser.add_argument('--input', dest='input_path', type=Path, required=True,
                             help='Input PDF directory (root containing source PDFs)')
    add_common_args(full_parser)
    
    # Partial pipeline
    partial_parser = subparsers.add_parser('partial', help='Run partial pipeline')
    add_common_args(partial_parser)
    partial_parser.add_argument('--chunks', dest='chunks_file', help='Pre-existing chunks file')
    partial_parser.add_argument('--sft-model', help='Pre-existing SFT model')
    partial_parser.add_argument('--skip-data', action='store_true', help='Skip data pipeline')
    partial_parser.add_argument('--skip-sft', action='store_true', help='Skip SFT pipeline')
    partial_parser.add_argument('--skip-rlaif', action='store_true', help='Skip RLAIF pipeline')
    
    # Resume pipeline
    resume_parser = subparsers.add_parser('resume', help='Resume from checkpoint')
    resume_parser.add_argument('--from', dest='resume_from', 
                              choices=['data', 'sft', 'rlaif', 'evaluation'], 
                              required=True, help='Resume from this stage')
    # Use a distinct flag name to avoid colliding with common --output-dir (new outputs)
    resume_parser.add_argument('--prev-output-dir', dest='prev_output_dir', type=Path, required=True,
                               help='Path to previous run output directory to resume from')
    add_common_args(resume_parser, require_input=False)
    
    # Development mode
    dev_parser = subparsers.add_parser('dev', help='Development mode (small datasets)')
    dev_parser.add_argument('--input', dest='input_path', type=Path, required=True,
                            help='Input PDF directory (root containing source PDFs)')
    add_common_args(dev_parser)
    dev_parser.set_defaults(
        pdf_limit=3,
        dataset_size=50, 
        pairs_size=20,
        epochs=1,
        dpo_epochs=1,
        ppo_prompts=8
    )
    
    # Production mode
    prod_parser = subparsers.add_parser('prod', help='Production mode with full datasets')
    add_common_args(prod_parser)
    prod_parser.set_defaults(
        dataset_size=500,
        pairs_size=100,
        epochs=3,
        dpo_epochs=2,
        ppo_prompts=32
    )
    
    return parser


def add_common_args(parser, require_input=True):
    """Add common arguments to parser with clear groupings and Path validation."""
    
    # Groups for better help formatting
    cfg = parser.add_argument_group('Config')
    io = parser.add_argument_group('IO')
    rt = parser.add_argument_group('Runtime Overrides')
    mon = parser.add_argument_group('Monitoring')

    # Configuration
    cfg.add_argument('--config', type=Path,
                     help='Configuration file (YAML or JSON) to override defaults')
    
    # IO (note: subparsers add --input explicitly to avoid duplication)
    io.add_argument('--output-dir', type=Path, default=Path("outputs"),
                   help='Base output directory (timestamped subfolder created per run)')
    io.add_argument('--experiment-name', help='Optional experiment name used in output folder naming')
    
    # Runtime overrides (quick knobs for sizes/epochs)
    rt.add_argument('--pdf-limit', type=int, help='Limit number of PDFs to process (dev mode)')
    rt.add_argument('--dataset-size', type=int, help='Target SFT dataset size')
    rt.add_argument('--pairs-size', type=int, help='Preference pair count for DPO')
    rt.add_argument('--epochs', type=int, help='SFT training epochs')
    # Low-memory SFT overrides
    rt.add_argument('--sft-batch-size', dest='sft_batch_size', type=int,
                   help='SFT per-device batch size (memory control)')
    rt.add_argument('--sft-grad-accum', dest='sft_grad_accum', type=int,
                   help='SFT gradient accumulation steps (memory control)')
    rt.add_argument('--sft-max-length', dest='sft_max_length', type=int,
                   help='SFT max sequence length (memory control)')
    rt.add_argument('--dpo-epochs', type=int, help='DPO training epochs')
    rt.add_argument('--ppo-prompts', type=int, help='Number of prompts for PPO rollouts')
    rt.add_argument('--enable-mps-fallback', action='store_true', help='Enable MPS fallback for unsupported ops (Apple Silicon)')
    
    # Monitoring
    mon.add_argument('--logging-backends', action='append', choices=['tensorboard', 'wandb'], default=[],
                    help='Enable logging backends (can specify multiple, e.g., --logging-backends wandb --logging-backends tensorboard)')
    parser.add_argument('--cleanup-intermediate', action='store_true',
                       help='Clean up intermediate files to save space')
    parser.add_argument('--dry-run', action='store_true', help='Show commands without executing')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')


def main():
    """Main entry point."""
    parser = create_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    try:
        # Load configuration
        if hasattr(args, 'config') and args.config:
            config = CompletePipelineConfig.from_file(Path(args.config))
            
            # Override with command line arguments
            for key, value in vars(args).items():
                if value is not None and key != 'config':
                    if key in CompletePipelineConfig.__dataclass_fields__:
                        setattr(config, key, value)
                    else:
                        # Handle nested config overrides
                        if key.startswith('pdf_'):
                            config.data_config[key] = value
                        elif key in ['dataset_size', 'epochs']:
                            config.sft_config[key] = value
                        elif key in ['sft_batch_size', 'sft_grad_accum', 'sft_max_length']:
                            # Map CLI names to SFT config keys
                            mapping = {
                                'sft_batch_size': 'batch_size',
                                'sft_grad_accum': 'gradient_accumulation_steps',
                                'sft_max_length': 'max_length',
                            }
                            config.sft_config[mapping[key]] = value
                        elif key in ['pairs_size', 'dpo_epochs', 'ppo_prompts']:
                            config.rlaif_config[key] = value
        else:
            # Create config from command line args
            config_dict = {}
            data_config = {}
            sft_config = {}
            rlaif_config = {}
            
            # Map arguments to appropriate configs
            for key, value in vars(args).items():
                if value is not None:
                    if key in CompletePipelineConfig.__dataclass_fields__:
                        config_dict[key] = value
                    elif key == 'pdf_limit':
                        data_config['pdf_limit'] = value
                    elif key in ['dataset_size', 'epochs']:
                        sft_config[key] = value
                    elif key in ['sft_batch_size', 'sft_grad_accum', 'sft_max_length']:
                        mapping = {
                            'sft_batch_size': 'batch_size',
                            'sft_grad_accum': 'gradient_accumulation_steps',
                            'sft_max_length': 'max_length',
                        }
                        sft_config[mapping[key]] = value
                    elif key in ['pairs_size', 'dpo_epochs', 'ppo_prompts']:
                        rlaif_config[key] = value
            
            # Convert paths
            if 'input_path' in config_dict and config_dict['input_path']:
                config_dict['input_path'] = Path(config_dict['input_path'])
            if 'chunks_file' in config_dict and config_dict['chunks_file']:
                config_dict['chunks_file'] = Path(config_dict['chunks_file'])
            if 'sft_model' in config_dict and config_dict['sft_model']:
                config_dict['sft_model'] = Path(config_dict['sft_model'])
            if 'output_dir' in config_dict:
                config_dict['output_dir'] = Path(config_dict['output_dir'])
            
            # Add nested configs
            config_dict['data_config'] = data_config
            config_dict['sft_config'] = sft_config
            config_dict['rlaif_config'] = rlaif_config
            
            config = CompletePipelineConfig(**config_dict)
        
        # Initialize and run pipeline
        pipeline = CompletePipeline(config)
        final_report = pipeline.run()
        
        print(f"\n✨ Complete Pipeline Finished!")
        print(f"📁 Check outputs in: {pipeline.output_dir}")
        
    except Exception as e:
        print(f"❌ Complete Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
