#!/usr/bin/env python3
"""
Data Pipeline - PDF Ingestion, Chunking, and Index Building

This script orchestrates the complete data ingestion pipeline:
1. Convert PDF documents to structured text
2. Chunk text with optimal overlap for retrieval
3. Build FAISS and BM25 search indices
4. Validate data quality and coverage

Key Features:
- ✅ Robust PDF processing with error handling
- ✅ Configurable chunking strategies
- ✅ Hybrid index building (FAISS + BM25)
- ✅ Data quality validation and statistics
- ✅ Cross-platform compatibility
- ✅ Progress tracking and resumption

Usage Examples:
    # Complete data pipeline
    python run_data_pipeline.py full --input data/raw_pdfs --output data/processed

    # Only PDF processing
    python run_data_pipeline.py pdf --input data/raw_pdfs --output data/processed/text.jsonl

    # Only chunking (existing text)
    python run_data_pipeline.py chunk --input data/processed/text.jsonl

    # Only index building
    python run_data_pipeline.py index --input data/processed/chunks.jsonl

    # Development mode (small subset)
    python run_data_pipeline.py dev --input data/raw_pdfs --limit 5
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


class DataStage(Enum):
    """Data pipeline stages."""
    PDF_PROCESSING = "pdf"
    CHUNKING = "chunking"
    INDEX_BUILDING = "indexing"
    VALIDATION = "validation"


@dataclass
class DataConfig:
    """Configuration for data processing pipeline."""
    
    # Input/output paths
    input_path: Path
    output_dir: Path = field(default_factory=lambda: Path("data/processed"))
    
    # PDF processing config
    pdf_limit: Optional[int] = None  # Limit number of PDFs (for dev)
    
    # Chunking config
    chunk_size: int = 1000
    chunk_stride: int = 150
    min_chunk_size: int = 100
    
    # Index config
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    
    # Pipeline control
    skip_pdf: bool = False
    skip_chunking: bool = False
    skip_indexing: bool = False
    skip_validation: bool = False
    force_rebuild: bool = False  # Rebuild even if outputs exist
    
    # Advanced options
    dry_run: bool = False
    verbose: bool = False
    experiment_name: Optional[str] = None
    
    @classmethod
    def from_file(cls, config_path: Path) -> 'DataConfig':
        """Load configuration from YAML or JSON file."""
        with open(config_path, 'r') as f:
            if config_path.suffix.lower() in ['.yaml', '.yml']:
                data = yaml.safe_load(f)
            else:
                data = json.load(f)
        
        # Convert path strings to Path objects
        if 'input_path' in data:
            data['input_path'] = Path(data['input_path'])
        if 'output_dir' in data:
            data['output_dir'] = Path(data['output_dir'])
            
        return cls(**data)


@dataclass
class DataState:
    """Track data pipeline progress."""
    pdf_completed: bool = False
    chunking_completed: bool = False
    indexing_completed: bool = False
    validation_completed: bool = False
    
    text_output: Optional[Path] = None
    chunks_output: Optional[Path] = None
    faiss_output: Optional[Path] = None
    store_output: Optional[Path] = None
    
    start_time: datetime = field(default_factory=datetime.now)
    last_checkpoint: datetime = field(default_factory=datetime.now)


class DataPipelineError(Exception):
    """Custom exception for data pipeline errors."""
    pass


class DataPipeline:
    """Complete data processing pipeline from PDFs to search indices."""
    
    def __init__(self, config: DataConfig):
        self.config = config
        self.state = DataState()
        self.setup_logging()
        self.setup_output_dirs()
        
        # Define key output paths
        self.text_file = self.config.output_dir / "text.jsonl"
        self.chunks_file = self.config.output_dir / "chunks.jsonl"
        self.indices_dir = self.config.output_dir / "indices"
        self.faiss_index = self.indices_dir / "faiss.index"
        self.store_file = self.indices_dir / "store.pkl"
        
        # Save config
        config_file = self.config.output_dir / "data_config.json"
        with open(config_file, 'w') as f:
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
        self.indices_dir.mkdir(parents=True, exist_ok=True)
    
    def save_checkpoint(self):
        """Save current pipeline state."""
        self.state.last_checkpoint = datetime.now()
        
        checkpoint_file = self.config.output_dir / "checkpoint.json"
        with open(checkpoint_file, 'w') as f:
            json.dump(asdict(self.state), f, indent=2, default=str)
        
        self.logger.debug(f"💾 Checkpoint saved: {checkpoint_file}")
    
    def validate_prerequisites(self):
        """Validate inputs before starting."""
        self.logger.info("🔍 Validating prerequisites...")
        
        if not self.config.input_path.exists():
            raise DataPipelineError(f"Input path not found: {self.config.input_path}")
        
        # Check if input is directory (for PDFs) or file (for text processing)
        if self.config.input_path.is_dir():
            pdf_files = list(self.config.input_path.glob("*.pdf"))
            if not pdf_files:
                raise DataPipelineError(f"No PDF files found in: {self.config.input_path}")
            
            self.logger.info(f"📄 Found {len(pdf_files)} PDF files")
            if self.config.pdf_limit:
                self.logger.info(f"🔢 Will process first {self.config.pdf_limit} files (dev mode)")
        
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
            raise DataPipelineError(f"{stage_name} failed with return code {process.returncode}")
        
        return process.returncode
    
    def run_pdf_processing(self) -> Path:
        """Convert PDFs to structured text."""
        stage_name = "PDF"
        self.logger.info(f"\n📄 Step 1: Processing PDF documents...")
        
        # Check if output already exists and not forcing rebuild
        if self.text_file.exists() and not self.config.force_rebuild:
            self.logger.info(f"📄 Text file already exists: {self.text_file}")
            self.state.pdf_completed = True
            self.state.text_output = self.text_file
            return self.text_file
        
        cmd = [
            "python", "src/ingest/pdf_to_text.py",
            "--in", str(self.config.input_path),
            "--out", str(self.text_file)
        ]
        
        if self.config.pdf_limit:
            cmd.extend(["--limit", str(self.config.pdf_limit)])
        
        self.run_subprocess_with_monitoring(cmd, stage_name)
        
        if not self.config.dry_run:
            if not self.text_file.exists():
                raise DataPipelineError(f"Expected text output not found: {self.text_file}")
        
        self.state.pdf_completed = True
        self.state.text_output = self.text_file
        self.save_checkpoint()
        
        self.logger.info(f"✅ PDF processing completed: {self.text_file}")
        return self.text_file
    
    def run_chunking(self) -> Path:
        """Chunk text for optimal retrieval."""
        stage_name = "CHUNKING"
        self.logger.info(f"\n📝 Step 2: Chunking text (size={self.config.chunk_size}, stride={self.config.chunk_stride})...")
        
        # Check if output already exists
        if self.chunks_file.exists() and not self.config.force_rebuild:
            self.logger.info(f"📝 Chunks file already exists: {self.chunks_file}")
            self.state.chunking_completed = True
            self.state.chunks_output = self.chunks_file
            return self.chunks_file
        
        # Use existing text file or the one we just created
        input_file = self.state.text_output or self.text_file
        
        cmd = [
            "python", "src/ingest/chunk_text.py",
            "--in", str(input_file),
            "--out", str(self.chunks_file),
            "--chunk", str(self.config.chunk_size),
            "--stride", str(self.config.chunk_stride)
        ]
        
        self.run_subprocess_with_monitoring(cmd, stage_name)
        
        if not self.config.dry_run:
            if not self.chunks_file.exists():
                raise DataPipelineError(f"Expected chunks output not found: {self.chunks_file}")
        
        self.state.chunking_completed = True
        self.state.chunks_output = self.chunks_file
        self.save_checkpoint()
        
        self.logger.info(f"✅ Text chunking completed: {self.chunks_file}")
        return self.chunks_file
    
    def run_index_building(self) -> tuple[Path, Path]:
        """Build FAISS and BM25 search indices."""
        stage_name = "INDEXING"
        self.logger.info(f"\n🔍 Step 3: Building search indices...")
        
        # Check if indices already exist
        if (self.faiss_index.exists() and self.store_file.exists() and 
            not self.config.force_rebuild):
            self.logger.info(f"🔍 Indices already exist: {self.indices_dir}")
            self.state.indexing_completed = True
            self.state.faiss_output = self.faiss_index
            self.state.store_output = self.store_file
            return self.faiss_index, self.store_file
        
        # Use existing chunks file or the one we just created
        input_file = self.state.chunks_output or self.chunks_file
        
        cmd = [
            "python", "src/ingest/build_index.py",
            "--in", str(input_file),
            "--faiss", str(self.faiss_index),
            "--store", str(self.store_file)
        ]
        
        # Add embedding model if specified
        if self.config.embedding_model:
            cmd.extend(["--embedding-model", self.config.embedding_model])
        
        self.run_subprocess_with_monitoring(cmd, stage_name)
        
        if not self.config.dry_run:
            if not self.faiss_index.exists() or not self.store_file.exists():
                raise DataPipelineError(f"Expected index outputs not found in: {self.indices_dir}")
        
        self.state.indexing_completed = True
        self.state.faiss_output = self.faiss_index
        self.state.store_output = self.store_file
        self.save_checkpoint()
        
        self.logger.info(f"✅ Index building completed: {self.indices_dir}")
        return self.faiss_index, self.store_file
    
    def run_validation(self) -> Dict[str, Any]:
        """Validate data quality and generate statistics."""
        stage_name = "VALIDATION"
        self.logger.info(f"\n✅ Step 4: Validating data quality...")
        
        stats = {}
        
        if not self.config.dry_run:
            # Count text documents
            if self.text_file.exists():
                with open(self.text_file, 'r') as f:
                    text_count = sum(1 for _ in f)
                stats['text_documents'] = text_count
                self.logger.info(f"📄 Text documents: {text_count}")
            
            # Count chunks
            if self.chunks_file.exists():
                with open(self.chunks_file, 'r') as f:
                    chunk_count = sum(1 for _ in f)
                stats['chunks_count'] = chunk_count
                self.logger.info(f"📝 Text chunks: {chunk_count}")
                
                # Sample chunk analysis
                with open(self.chunks_file, 'r') as f:
                    first_chunk = json.loads(f.readline())
                    stats['sample_chunk_fields'] = list(first_chunk.keys())
                    stats['sample_chunk_size'] = len(first_chunk.get('text', ''))
            
            # Index statistics
            if self.faiss_index.exists():
                import faiss
                index = faiss.read_index(str(self.faiss_index))
                stats['faiss_index_size'] = index.ntotal
                stats['faiss_dimensions'] = index.d
                self.logger.info(f"🔍 FAISS index: {index.ntotal} vectors, {index.d} dimensions")
        
        # Save validation report
        validation_report = {
            "timestamp": datetime.now().isoformat(),
            "config": asdict(self.config),
            "statistics": stats,
            "files": {
                "text_file": str(self.text_file),
                "chunks_file": str(self.chunks_file),
                "faiss_index": str(self.faiss_index),
                "store_file": str(self.store_file)
            }
        }
        
        validation_file = self.config.output_dir / "validation_report.json"
        with open(validation_file, 'w') as f:
            json.dump(validation_report, f, indent=2, default=str)
        
        self.state.validation_completed = True
        self.save_checkpoint()
        
        self.logger.info(f"✅ Data validation completed: {validation_file}")
        return validation_report
    
    def run(self):
        """Main pipeline runner."""
        try:
            self.logger.info("🚀 Starting Data Processing Pipeline")
            self.logger.info(f"📁 Input: {self.config.input_path}")
            self.logger.info(f"📁 Output: {self.config.output_dir}")
            
            # Validate prerequisites
            self.validate_prerequisites()
            
            # Run pipeline stages
            if not self.config.skip_pdf and not self.state.pdf_completed:
                self.run_pdf_processing()
            
            if not self.config.skip_chunking and not self.state.chunking_completed:
                self.run_chunking()
            
            if not self.config.skip_indexing and not self.state.indexing_completed:
                self.run_index_building()
            
            if not self.config.skip_validation and not self.state.validation_completed:
                validation_report = self.run_validation()
            
            # Success message
            duration = datetime.now() - self.state.start_time
            self.logger.info(f"\n🎉 Data Pipeline Completed Successfully!")
            self.logger.info(f"⏱️ Total duration: {duration}")
            self.logger.info(f"📁 All outputs saved to: {self.config.output_dir}")
            
            # Key outputs for next pipeline stages
            self.logger.info(f"\n📋 Key Outputs:")
            self.logger.info(f"   📝 Chunks: {self.chunks_file}")
            self.logger.info(f"   🔍 FAISS Index: {self.faiss_index}")
            self.logger.info(f"   📦 Store: {self.store_file}")
            
            return validation_report if not self.config.skip_validation else {}
            
        except DataPipelineError as e:
            self.logger.error(f"❌ Data Pipeline error: {e}")
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
        description="Data Processing Pipeline - PDF to Search Indices",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Complete data pipeline
  %(prog)s full --input data/raw_pdfs --output data/processed

  # Only PDF processing
  %(prog)s pdf --input data/raw_pdfs --output data/processed/text.jsonl

  # Only chunking (existing text)
  %(prog)s chunk --input data/processed/text.jsonl

  # Only index building
  %(prog)s index --input data/processed/chunks.jsonl

  # Development mode (small subset)
  %(prog)s dev --input data/raw_pdfs --limit 5
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Pipeline commands')
    
    # Full pipeline
    full_parser = subparsers.add_parser('full', help='Run complete data pipeline')
    add_common_args(full_parser)
    
    # PDF only
    pdf_parser = subparsers.add_parser('pdf', help='Process PDFs to text only')
    add_pdf_args(pdf_parser)
    
    # Chunking only
    chunk_parser = subparsers.add_parser('chunk', help='Chunk text only')
    add_chunk_args(chunk_parser)
    
    # Indexing only
    index_parser = subparsers.add_parser('index', help='Build indices only')
    add_index_args(index_parser)
    
    # Development mode
    dev_parser = subparsers.add_parser('dev', help='Development mode (limited files)')
    add_common_args(dev_parser)
    dev_parser.add_argument('--limit', dest='pdf_limit', type=int, default=5, 
                           help='Limit number of PDFs to process')
    
    return parser


def add_common_args(parser):
    """Add common arguments."""
    parser.add_argument('--config', help='Path to configuration YAML/JSON file')
    parser.add_argument('--input', dest='input_path', required=True, help='Input path (PDFs directory)')
    parser.add_argument('--output', dest='output_dir', default="data/processed", help='Output directory')
    
    # Chunking config
    parser.add_argument('--chunk-size', type=int, default=1000, help='Chunk size in characters')
    parser.add_argument('--chunk-stride', type=int, default=150, help='Chunk overlap in characters')
    
    # Index config
    parser.add_argument('--embedding-model', default="sentence-transformers/all-MiniLM-L6-v2",
                       help='Embedding model for FAISS index')
    
    # Control flags
    parser.add_argument('--force-rebuild', action='store_true', help='Force rebuild even if outputs exist')
    parser.add_argument('--dry-run', action='store_true', help='Show commands without executing')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')


def add_pdf_args(parser):
    """Add PDF-specific arguments."""
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--input', dest='input_path', required=True, help='PDF directory')
    parser.add_argument('--output', dest='output_dir', default="data/processed", help='Output directory')
    parser.add_argument('--limit', dest='pdf_limit', type=int, help='Limit number of PDFs')
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')


def add_chunk_args(parser):
    """Add chunking-specific arguments."""
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--input', dest='input_path', required=True, help='Text JSONL file')
    parser.add_argument('--output', dest='output_dir', default="data/processed", help='Output directory')
    parser.add_argument('--chunk-size', type=int, default=1000, help='Chunk size')
    parser.add_argument('--chunk-stride', type=int, default=150, help='Chunk stride')
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode')
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')


def add_index_args(parser):
    """Add indexing-specific arguments."""
    parser.add_argument('--config', help='Path to configuration file')
    parser.add_argument('--input', dest='input_path', required=True, help='Chunks JSONL file')
    parser.add_argument('--output', dest='output_dir', default="data/processed", help='Output directory')
    parser.add_argument('--embedding-model', default="sentence-transformers/all-MiniLM-L6-v2",
                       help='Embedding model')
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode')
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
            config = DataConfig.from_file(Path(args.config))
            # Override with command line arguments
            for key, value in vars(args).items():
                if value is not None and key != 'config' and key in DataConfig.__dataclass_fields__:
                    setattr(config, key, value)
        else:
            # Create config from command line args
            config_dict = {}
            
            for key, value in vars(args).items():
                if value is not None and key in DataConfig.__dataclass_fields__:
                    config_dict[key] = value
            
            # Convert paths
            if 'input_path' in config_dict:
                config_dict['input_path'] = Path(config_dict['input_path'])
            if 'output_dir' in config_dict:
                config_dict['output_dir'] = Path(config_dict['output_dir'])
            
            # Handle subcommand-specific logic
            if args.command == 'pdf':
                config_dict.update({'skip_chunking': True, 'skip_indexing': True})
            elif args.command == 'chunk':
                config_dict.update({'skip_pdf': True, 'skip_indexing': True})
            elif args.command == 'index':
                config_dict.update({'skip_pdf': True, 'skip_chunking': True})
            
            config = DataConfig(**config_dict)
        
        # Initialize and run pipeline
        pipeline = DataPipeline(config)
        validation_report = pipeline.run()
        
        print(f"\n✨ Data Pipeline Complete!")
        print(f"📁 Check outputs in: {config.output_dir}")
        
    except Exception as e:
        print(f"❌ Data Pipeline failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()