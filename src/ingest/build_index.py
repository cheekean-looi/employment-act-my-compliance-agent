#!/usr/bin/env python3
# python src/ingest/build_index.py --in data/processed/chunks.jsonl --faiss data/indices/faiss.index --store data/indices/store.pkl --model intfloat/e5-large-v2 --gpu --verify
"""
Build search indices (BM25 + FAISS) for Employment Act chunks.
Creates both sparse (BM25) and dense (FAISS) indices for hybrid retrieval.
"""

import argparse
import json
import pickle
from pathlib import Path
from typing import Dict, List, Any
import os

import numpy as np
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import faiss

try:
    from .utils import (
        create_metadata, finalize_metadata, save_metadata, 
        atomic_write_binary, check_up_to_date, get_model_revision_info
    )
except ImportError:
    # When run as script, use absolute import
    from utils import (
        create_metadata, finalize_metadata, save_metadata, 
        atomic_write_binary, check_up_to_date, get_model_revision_info
    )


def load_chunks(input_file: Path) -> List[Dict[str, Any]]:
    """Load chunks from JSONL file."""
    chunks = []
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            try:
                chunk = json.loads(line.strip())
                chunks.append(chunk)
            except json.JSONDecodeError as e:
                print(f"Warning: Skipping invalid JSON at line {line_num}: {e}")
    
    print(f"Loaded {len(chunks)} chunks from {input_file}")
    return chunks


def create_bm25_index(chunks: List[Dict[str, Any]]) -> BM25Okapi:
    """Create BM25 index from chunk texts."""
    print("Creating BM25 index...")
    
    # Tokenize texts for BM25
    tokenized_corpus = []
    for chunk in chunks:
        # Simple tokenization - split by whitespace and lowercase
        tokens = chunk['text'].lower().split()
        tokenized_corpus.append(tokens)
    
    # Create BM25 index
    bm25 = BM25Okapi(tokenized_corpus)
    
    print(f"BM25 index created with {len(tokenized_corpus)} documents")
    return bm25


def create_dense_embeddings(chunks: List[Dict[str, Any]], model_name: str = "intfloat/e5-large-v2") -> np.ndarray:
    """Create dense embeddings using sentence transformers."""
    print(f"Creating dense embeddings with {model_name}...")
    
    # Load embedding model
    model = SentenceTransformer(model_name)
    
    # Extract texts with E5 passage prefix for proper embedding
    texts = [f"passage: {chunk['text']}" for chunk in chunks]
    
    # Create embeddings in batches
    batch_size = 32
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        batch_embeddings = model.encode(batch_texts, show_progress_bar=True)
        embeddings.append(batch_embeddings)
        
        print(f"Processed {min(i + batch_size, len(texts))}/{len(texts)} chunks")
    
    # Concatenate all embeddings
    embeddings = np.vstack(embeddings)
    
    print(f"Created embeddings: {embeddings.shape}")
    return embeddings


def create_faiss_index(embeddings: np.ndarray, use_gpu: bool = False) -> faiss.Index:
    """Create FAISS index from embeddings."""
    print("Creating FAISS index...")
    
    dimension = embeddings.shape[1]
    n_docs = embeddings.shape[0]
    
    # Normalize embeddings for cosine similarity BEFORE index creation/training
    faiss.normalize_L2(embeddings)
    
    # Choose index type based on dataset size
    if n_docs < 10000:
        # For small datasets, use exact search
        index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        print("Using exact search (IndexFlatIP)")
    else:
        # For larger datasets, use approximate search
        nlist = min(100, max(10, n_docs // 100))  # Number of clusters
        index = faiss.IndexIVFFlat(faiss.IndexFlatIP(dimension), dimension, nlist)
        print(f"Using approximate search (IndexIVFFlat) with {nlist} clusters")
        
        # Train the index on normalized embeddings
        print("Training FAISS index...")
        index.train(embeddings)
    
    # Add embeddings to index
    print("Adding embeddings to index...")
    index.add(embeddings)
    
    # Move to GPU if requested and available
    if use_gpu and faiss.get_num_gpus() > 0:
        print("Moving index to GPU...")
        index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index)
    
    print(f"FAISS index created with {index.ntotal} vectors")
    return index


def save_indices_and_metadata(
    chunks: List[Dict[str, Any]],
    bm25: BM25Okapi,
    faiss_index: faiss.Index,
    faiss_path: Path,
    store_path: Path
) -> None:
    """Save indices and metadata atomically."""
    print("Saving indices and metadata...")
    
    # Ensure output directories exist
    faiss_path.parent.mkdir(parents=True, exist_ok=True)
    store_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Save FAISS index atomically
    temp_faiss_path = faiss_path.with_suffix(faiss_path.suffix + '.tmp')
    try:
        # Move back to CPU before saving if it was on GPU
        if hasattr(faiss_index, 'index'):  # GPU index
            cpu_index = faiss.index_gpu_to_cpu(faiss_index)
            faiss.write_index(cpu_index, str(temp_faiss_path))
        else:
            faiss.write_index(faiss_index, str(temp_faiss_path))
        
        # Atomic move
        os.replace(temp_faiss_path, faiss_path)
    except Exception:
        if temp_faiss_path.exists():
            temp_faiss_path.unlink()
        raise
    
    # Prepare metadata store
    store_data = {
        'chunks': chunks,
        'bm25_index': bm25,
        'chunk_count': len(chunks),
        'embedding_dimension': faiss_index.d if hasattr(faiss_index, 'd') else None,
        'index_type': type(faiss_index).__name__,
        'bm25_serialization': 'pickled (python-version-bound)',
        'bm25_portability_note': 'BM25 index is pickled and bound to Python version. '
                                'For cross-environment deployment, use --bm25-rebuild to '
                                'regenerate BM25 from tokenized corpus instead of unpickling.'
    }
    
    # Save metadata and BM25 index atomically
    temp_store_path = store_path.with_suffix(store_path.suffix + '.tmp')
    try:
        with open(temp_store_path, 'wb') as f:
            pickle.dump(store_data, f)
        
        # Atomic move
        os.replace(temp_store_path, store_path)
    except Exception:
        if temp_store_path.exists():
            temp_store_path.unlink()
        raise
    
    print(f"FAISS index saved to: {faiss_path}")
    print(f"Metadata and BM25 index saved to: {store_path}")


def build_indices(
    input_file: Path,
    faiss_path: Path,
    store_path: Path,
    embedding_model: str = "intfloat/e5-large-v2",
    use_gpu: bool = False,
    force: bool = False,
    bm25_rebuild: bool = False
) -> None:
    """Build both BM25 and FAISS indices with atomic writes."""
    # Check if outputs are up-to-date
    metadata_file = faiss_path.parent / 'build_index.metadata.json'
    if check_up_to_date([faiss_path, store_path, metadata_file], [input_file], force):
        print(f"Output files are up-to-date. Use --force to regenerate.")
        return
    
    # Load chunks
    chunks = load_chunks(input_file)
    
    if not chunks:
        print("No chunks found. Exiting.")
        return
    
    # Get model revision info for provenance
    model_revision_info = get_model_revision_info(embedding_model)
    
    # Create metadata
    config = {
        'stage': 'build_index',
        'embedding_model': embedding_model,
        'embedding_model_revision': model_revision_info,
        'use_gpu': use_gpu,
        'e5_prefixes_applied': True,
        'normalization_before_training': True,
        'batch_size': 32,
        'bm25_rebuild_requested': bm25_rebuild
    }
    
    metadata = create_metadata(
        stage='build_index',
        input_files=[input_file],
        output_files=[faiss_path, store_path],
        config=config
    )
    
    # Create BM25 index
    bm25 = create_bm25_index(chunks)
    
    # Create dense embeddings
    embeddings = create_dense_embeddings(chunks, embedding_model)
    
    # Create FAISS index
    faiss_index = create_faiss_index(embeddings, use_gpu)
    
    # Add index stats to metadata
    metadata['index_stats'] = {
        'chunk_count': len(chunks),
        'embedding_dimension': faiss_index.d,
        'index_type': type(faiss_index).__name__,
        'index_total_vectors': faiss_index.ntotal,
        'bm25_vocab_size': len(bm25.idf) if hasattr(bm25, 'idf') else None
    }
    
    # If using IVF index, add training parameters
    if hasattr(faiss_index, 'nlist'):
        metadata['index_stats']['ivf_nlist'] = faiss_index.nlist
        metadata['index_stats']['ivf_nprobe'] = getattr(faiss_index, 'nprobe', None)
    
    # Save everything
    save_indices_and_metadata(chunks, bm25, faiss_index, faiss_path, store_path)
    
    # Finalize and save metadata
    finalize_metadata(metadata, [faiss_path, store_path])
    save_metadata(metadata, metadata_file)
    
    # Print summary
    print("\n" + "="*50)
    print("INDEX BUILDING COMPLETE")
    print("="*50)
    print(f"Processed {len(chunks)} chunks")
    print(f"BM25 index: Ready")
    print(f"FAISS index: {faiss_index.ntotal} vectors, dimension {faiss_index.d}")
    print(f"Embedding model: {embedding_model}")
    print(f"E5 prefixes applied: passage/query")
    print(f"Files created:")
    print(f"  - {faiss_path}")
    print(f"  - {store_path}")
    print(f"  - {metadata_file}")


def verify_indices(faiss_path: Path, store_path: Path) -> None:
    """Verify that indices can be loaded correctly."""
    print("\nVerifying indices...")
    
    try:
        # Load FAISS index
        faiss_index = faiss.read_index(str(faiss_path))
        print(f"✓ FAISS index loaded: {faiss_index.ntotal} vectors")
        
        # Load metadata
        with open(store_path, 'rb') as f:
            store_data = pickle.load(f)
        
        print(f"✓ Metadata loaded: {len(store_data['chunks'])} chunks")
        print(f"✓ BM25 index loaded: {type(store_data['bm25_index']).__name__}")
        
        print("All indices verified successfully!")
        
    except Exception as e:
        print(f"✗ Error verifying indices: {e}")


def main():
    parser = argparse.ArgumentParser(description="Build BM25 and FAISS indices for Employment Act chunks")
    parser.add_argument('--in', dest='input_file', required=True,
                        help='Input JSONL file with chunks')
    parser.add_argument('--faiss', dest='faiss_path', required=True,
                        help='Output path for FAISS index file')
    parser.add_argument('--store', dest='store_path', required=True,
                        help='Output path for metadata and BM25 index (pickle file)')
    parser.add_argument('--model', default="intfloat/e5-large-v2",
                        help='Sentence transformer model for embeddings')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for FAISS index (if available)')
    parser.add_argument('--verify', action='store_true',
                        help='Verify indices after creation')
    parser.add_argument('--force', action='store_true',
                        help='Force regeneration even if outputs are up-to-date')
    parser.add_argument('--bm25-rebuild', action='store_true',
                        help='Rebuild BM25 index instead of loading from pickle (for cross-env compatibility)')
    
    args = parser.parse_args()
    
    input_file = Path(args.input_file)
    faiss_path = Path(args.faiss_path)
    store_path = Path(args.store_path)
    
    if not input_file.exists():
        print(f"Error: Input file {input_file} does not exist")
        return
    
    # Check GPU availability
    if args.gpu:
        gpu_count = faiss.get_num_gpus()
        if gpu_count > 0:
            print(f"GPU mode enabled. Found {gpu_count} GPU(s)")
        else:
            print("Warning: GPU requested but no GPUs found. Using CPU.")
            args.gpu = False
    
    # Build indices
    build_indices(input_file, faiss_path, store_path, args.model, args.gpu, args.force, getattr(args, 'bm25_rebuild', False))
    
    # Verify if requested
    if args.verify:
        verify_indices(faiss_path, store_path)


if __name__ == "__main__":
    main()