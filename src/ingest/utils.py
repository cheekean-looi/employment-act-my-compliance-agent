#!/usr/bin/env python3
"""
Shared utilities for atomic writes and metadata generation across ingestion pipeline.
"""

import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Dict, Any, Optional, List


def get_git_sha() -> Optional[str]:
    """Get current git SHA if in a git repository."""
    try:
        result = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], 
            capture_output=True, 
            text=True, 
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_pip_freeze() -> Optional[str]:
    """Get pip freeze output for exact package versions."""
    try:
        result = subprocess.run(
            ['pip', 'freeze'], 
            capture_output=True, 
            text=True, 
            check=True
        )
        return result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def get_conda_info() -> Dict[str, str]:
    """Get conda environment information."""
    conda_info = {}
    
    # Get conda environment name
    conda_env = os.environ.get('CONDA_DEFAULT_ENV')
    if conda_env:
        conda_info['conda_env'] = conda_env
    
    # Try to get conda list for this environment
    try:
        result = subprocess.run(
            ['conda', 'list', '--json'], 
            capture_output=True, 
            text=True, 
            check=True
        )
        conda_info['conda_packages'] = result.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    return conda_info


def get_model_revision_info(model_name: str) -> Dict[str, str]:
    """Get model revision/commit info if available."""
    revision_info = {}
    
    try:
        # Try to get model info from transformers
        from transformers import AutoConfig
        config = AutoConfig.from_pretrained(model_name)
        
        # Check for revision info in config
        if hasattr(config, '_commit_hash'):
            revision_info['commit_hash'] = config._commit_hash
        if hasattr(config, '_name_or_path'):
            revision_info['name_or_path'] = config._name_or_path
        if hasattr(config, 'model_type'):
            revision_info['model_type'] = config.model_type
        if hasattr(config, 'architectures'):
            revision_info['architectures'] = config.architectures
            
    except Exception:
        # Fallback to basic model name
        revision_info['model_name'] = model_name
    
    try:
        # Try to get model info from sentence-transformers
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        
        # Get model card info if available
        if hasattr(model, '_model_config') and model._model_config:
            config = model._model_config
            if hasattr(config, '_commit_hash'):
                revision_info['st_commit_hash'] = config._commit_hash
                
        # Try to get tokenizer info
        if hasattr(model, 'tokenizer') and hasattr(model.tokenizer, 'name_or_path'):
            revision_info['tokenizer_path'] = model.tokenizer.name_or_path
            
    except Exception:
        pass
    
    return revision_info


def get_environment_info() -> Dict[str, Any]:
    """Get comprehensive environment information for provenance."""
    env_info = {
        'python_version': platform.python_version(),
        'python_executable': os.sys.executable,
        'platform': platform.platform(),
        'hostname': platform.node(),
        'os_name': os.name,
        'architecture': platform.architecture(),
        'machine': platform.machine(),
        'processor': platform.processor(),
    }
    
    # Add conda information
    conda_info = get_conda_info()
    if conda_info:
        env_info['conda'] = conda_info
    
    # Get pip freeze for exact versions
    pip_freeze = get_pip_freeze()
    if pip_freeze:
        env_info['pip_freeze'] = pip_freeze
    
    # Try to get key package versions
    package_versions = {}
    packages_to_check = [
        ('torch', 'torch'),
        ('transformers', 'transformers'),
        ('faiss', 'faiss'),
        ('fitz', 'pymupdf'),
        ('sentence_transformers', 'sentence-transformers'),
        ('numpy', 'numpy'),
        ('pandas', 'pandas'),
        ('sklearn', 'scikit-learn'),
    ]
    
    for module_name, package_name in packages_to_check:
        try:
            module = __import__(module_name)
            if hasattr(module, '__version__'):
                package_versions[package_name] = module.__version__
            elif hasattr(module, 'version') and hasattr(module.version, '__getitem__'):
                # Handle PyMuPDF version format
                package_versions[package_name] = module.version[0]
        except ImportError:
            pass
    
    if package_versions:
        env_info['key_packages'] = package_versions
    
    return env_info


def compute_file_hash(file_path: Path) -> str:
    """Compute SHA256 hash of a file."""
    sha256_hash = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    except Exception:
        return "hash_error"


def compute_config_hash(config: Dict[str, Any]) -> str:
    """Compute hash of configuration for provenance."""
    config_str = json.dumps(config, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def create_metadata(
    stage: str,
    input_files: List[Path],
    output_files: List[Path],
    config: Dict[str, Any],
    stage_specific: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Create comprehensive metadata for a pipeline stage."""
    timestamp = time.time()
    
    metadata = {
        'stage': stage,
        'timestamp': timestamp,
        'iso_timestamp': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(timestamp)),
        'git_sha': get_git_sha(),
        'environment': get_environment_info(),
        'config': config,
        'config_hash': compute_config_hash(config),
        'input_files': [],
        'output_files': [],
    }
    
    # Add input file information
    for input_file in input_files:
        if input_file.exists():
            metadata['input_files'].append({
                'path': str(input_file),
                'size': input_file.stat().st_size,
                'mtime': input_file.stat().st_mtime,
                'hash': compute_file_hash(input_file)
            })
    
    # Add output file information (will be filled after creation)
    for output_file in output_files:
        metadata['output_files'].append({
            'path': str(output_file),
            'size': None,  # Will be filled after creation
            'hash': None   # Will be filled after creation
        })
    
    # Add stage-specific metadata
    if stage_specific:
        metadata.update(stage_specific)
    
    return metadata


def atomic_write_text(content: str, file_path: Path) -> None:
    """Atomically write text content to a file."""
    temp_path = file_path.with_suffix(file_path.suffix + '.tmp')
    try:
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to temporary file
        with open(temp_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Atomic move
        os.replace(temp_path, file_path)
    except Exception:
        # Clean up temp file if something went wrong
        if temp_path.exists():
            temp_path.unlink()
        raise


def atomic_write_jsonl(lines: List[Dict[str, Any]], file_path: Path) -> None:
    """Atomically write JSONL content to a file."""
    temp_path = file_path.with_suffix(file_path.suffix + '.tmp')
    try:
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to temporary file
        with open(temp_path, 'w', encoding='utf-8') as f:
            for line in lines:
                f.write(json.dumps(line, ensure_ascii=False) + '\n')
        
        # Atomic move
        os.replace(temp_path, file_path)
    except Exception:
        # Clean up temp file if something went wrong
        if temp_path.exists():
            temp_path.unlink()
        raise


def atomic_write_binary(data: bytes, file_path: Path) -> None:
    """Atomically write binary content to a file."""
    temp_path = file_path.with_suffix(file_path.suffix + '.tmp')
    try:
        # Ensure parent directory exists
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Write to temporary file
        with open(temp_path, 'wb') as f:
            f.write(data)
        
        # Atomic move
        os.replace(temp_path, file_path)
    except Exception:
        # Clean up temp file if something went wrong
        if temp_path.exists():
            temp_path.unlink()
        raise


def finalize_metadata(metadata: Dict[str, Any], output_files: List[Path]) -> None:
    """Finalize metadata by adding output file sizes and hashes."""
    for i, output_file in enumerate(output_files):
        if output_file.exists() and i < len(metadata['output_files']):
            metadata['output_files'][i]['size'] = output_file.stat().st_size
            metadata['output_files'][i]['hash'] = compute_file_hash(output_file)


def save_metadata(metadata: Dict[str, Any], metadata_path: Path) -> None:
    """Save metadata to JSON file atomically."""
    atomic_write_text(json.dumps(metadata, indent=2, ensure_ascii=False), metadata_path)


def check_up_to_date(
    output_files: List[Path], 
    input_files: List[Path],
    force: bool = False
) -> bool:
    """Check if outputs are up-to-date relative to inputs."""
    if force:
        return False
    
    # If any output doesn't exist, need to regenerate
    for output_file in output_files:
        if not output_file.exists():
            return False
    
    # If no inputs, outputs are considered up-to-date
    if not input_files:
        return True
    
    # Find the newest input file time
    newest_input_time = 0
    for input_file in input_files:
        if input_file.exists():
            newest_input_time = max(newest_input_time, input_file.stat().st_mtime)
    
    # Find the oldest output file time
    oldest_output_time = float('inf')
    for output_file in output_files:
        if output_file.exists():
            oldest_output_time = min(oldest_output_time, output_file.stat().st_mtime)
    
    # Outputs are up-to-date if oldest output is newer than newest input
    return oldest_output_time > newest_input_time