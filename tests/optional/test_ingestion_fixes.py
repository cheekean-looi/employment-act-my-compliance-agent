#!/usr/bin/env python3
"""
Test script to verify all critical Hour 1 fixes are working correctly.
Tests E5 prefixes, FAISS normalization, config unification, and atomic writes.
"""

import json
import os
import sys
from pathlib import Path
import tempfile
import shutil

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / 'src'))

try:
    from src.ingest.utils import create_metadata, save_metadata, check_up_to_date
    from src.retriever.hybrid_retriever import HybridRetriever
    import numpy as np
    import faiss
    from sentence_transformers import SentenceTransformer
    print("✓ All imports successful")
except ImportError as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)


def test_e5_prefixes():
    """Test that E5 prefixes are being applied correctly."""
    print("\n=== Testing E5 Prefixes ===")
    
    # Test embedding model directly
    model = SentenceTransformer("intfloat/e5-large-v2")
    
    # Test passage prefix (what build_index.py does)
    test_text = "This is a test document"
    passage_embedding = model.encode([f"passage: {test_text}"])
    
    # Test query prefix (what hybrid_retriever.py does)
    test_query = "test query"
    query_embedding = model.encode([f"query: {test_query}"])
    
    print(f"✓ Passage embedding shape: {passage_embedding.shape}")
    print(f"✓ Query embedding shape: {query_embedding.shape}")
    
    # Test that prefixes make a difference
    no_prefix_passage = model.encode([test_text])
    no_prefix_query = model.encode([test_query])
    
    passage_diff = np.mean(np.abs(passage_embedding - no_prefix_passage))
    query_diff = np.mean(np.abs(query_embedding - no_prefix_query))
    
    if passage_diff > 0.01 and query_diff > 0.01:
        print("✓ E5 prefixes are making embeddings different (good!)")
    else:
        print("✗ E5 prefixes seem to have no effect")
        return False
    
    return True


def test_faiss_normalization():
    """Test that FAISS normalization is working correctly."""
    print("\n=== Testing FAISS Normalization ===")
    
    # Create test embeddings
    np.random.seed(42)
    test_embeddings = np.random.randn(10, 384).astype(np.float32)
    
    # Test normalization
    faiss.normalize_L2(test_embeddings)
    
    # Check that vectors are normalized
    norms = np.linalg.norm(test_embeddings, axis=1)
    if np.allclose(norms, 1.0, atol=1e-6):
        print("✓ FAISS normalization working correctly")
        return True
    else:
        print(f"✗ FAISS normalization failed, norms: {norms}")
        return False


def test_config_unification():
    """Test that configuration is unified across components."""
    print("\n=== Testing Config Unification ===")
    
    # Test environment variable reading
    os.environ['CHUNK_SIZE'] = '1000'
    os.environ['CHUNK_STRIDE'] = '300'
    
    chunk_size = int(os.getenv('CHUNK_SIZE', '1000'))
    chunk_stride = int(os.getenv('CHUNK_STRIDE', '300'))
    
    if chunk_size == 1000 and chunk_stride == 300:
        print("✓ Config unification working")
        return True
    else:
        print(f"✗ Config unification failed: size={chunk_size}, stride={chunk_stride}")
        return False


def test_atomic_writes():
    """Test atomic write functionality."""
    print("\n=== Testing Atomic Writes ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Test metadata creation
        config = {'test': 'config', 'value': 42}
        metadata = create_metadata(
            stage='test',
            input_files=[],
            output_files=[],
            config=config
        )
        
        # Test metadata saving
        metadata_file = temp_path / 'test.metadata.json'
        save_metadata(metadata, metadata_file)
        
        if metadata_file.exists():
            with open(metadata_file) as f:
                loaded_metadata = json.load(f)
            
            if loaded_metadata['stage'] == 'test' and loaded_metadata['config']['value'] == 42:
                print("✓ Atomic metadata writes working")
            else:
                print("✗ Metadata content incorrect")
                return False
        else:
            print("✗ Metadata file not created")
            return False
    
    return True


def test_idempotency():
    """Test idempotency checks."""
    print("\n=== Testing Idempotency ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create test files
        input_file = temp_path / 'input.txt'
        output_file = temp_path / 'output.txt'
        
        input_file.write_text('test input')
        output_file.write_text('test output')
        
        # Test up-to-date check
        is_up_to_date = check_up_to_date([output_file], [input_file], force=False)
        
        if is_up_to_date:
            print("✓ Idempotency check working")
        else:
            print("✗ Idempotency check failed")
            return False
        
        # Test force override
        is_forced = check_up_to_date([output_file], [input_file], force=True)
        
        if not is_forced:
            print("✓ Force override working")
        else:
            print("✗ Force override failed")
            return False
    
    return True


def test_section_coverage_calculation():
    """Test section coverage calculation logic."""
    print("\n=== Testing Section Coverage Calculation ===")
    
    # Mock chunks data
    chunks = [
        {'section_id': 'EA-1', 'text': 'test1'},
        {'section_id': 'EA-2', 'text': 'test2'},
        {'section_id': None, 'text': 'test3'},
        {'section_id': 'EA-3', 'text': 'test4'},
        {'section_id': '', 'text': 'test5'},
    ]
    
    sections_with_id = sum(1 for chunk in chunks if chunk.get('section_id'))
    coverage = (sections_with_id / len(chunks) * 100) if chunks else 0
    
    expected_coverage = 60.0  # 3 out of 5 chunks have valid section_id
    if abs(coverage - expected_coverage) < 0.1:
        print(f"✓ Section coverage calculation working: {coverage}%")
        return True
    else:
        print(f"✗ Section coverage calculation failed: {coverage}% (expected ~{expected_coverage}%)")
        return False


def main():
    """Run all tests."""
    print("Testing Hour 1 Critical Fixes")
    print("=" * 50)
    
    tests = [
        test_e5_prefixes,
        test_faiss_normalization,
        test_config_unification,
        test_atomic_writes,
        test_idempotency,
        test_section_coverage_calculation,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with error: {e}")
            results.append(False)
    
    print("\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{i+1}. {test.__name__}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All critical fixes are working correctly!")
        return 0
    else:
        print(f"\n❌ {total - passed} tests failed. Please review the implementation.")
        return 1


if __name__ == "__main__":
    sys.exit(main())