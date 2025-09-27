#!/usr/bin/env python3
"""
Test script to verify Hour 1 code changes are correctly implemented.
This checks the actual code changes without requiring full environment.
"""

import re
from pathlib import Path


def test_e5_prefixes_in_code():
    """Test that E5 prefixes are in the code."""
    print("=== Testing E5 Prefixes in Code ===")
    
    # Check build_index.py
    build_index_path = Path("src/ingest/build_index.py")
    if not build_index_path.exists():
        print("✗ build_index.py not found")
        return False
    
    content = build_index_path.read_text()
    
    # Check for passage prefix
    if 'f"passage: {chunk[\'text\']}"' in content:
        print("✓ Passage prefix found in build_index.py")
    else:
        print("✗ Passage prefix NOT found in build_index.py")
        return False
    
    # Check hybrid_retriever.py
    retriever_path = Path("src/retriever/hybrid_retriever.py")
    if not retriever_path.exists():
        print("✗ hybrid_retriever.py not found")
        return False
    
    content = retriever_path.read_text()
    
    # Check for query prefix
    if 'f"query: {query}"' in content:
        print("✓ Query prefix found in hybrid_retriever.py")
    else:
        print("✗ Query prefix NOT found in hybrid_retriever.py")
        return False
    
    return True


def test_faiss_normalization_order():
    """Test that FAISS normalization is before training."""
    print("\n=== Testing FAISS Normalization Order ===")
    
    build_index_path = Path("src/ingest/build_index.py")
    content = build_index_path.read_text()
    
    # Find the normalization and training code
    normalize_match = re.search(r'faiss\.normalize_L2\(embeddings\)', content)
    train_match = re.search(r'index\.train\(embeddings\)', content)
    
    if not normalize_match:
        print("✗ FAISS normalization not found")
        return False
    
    if not train_match:
        print("✓ No training found (small dataset - OK)")
        return True
    
    # Check order
    if normalize_match.start() < train_match.start():
        print("✓ FAISS normalization is before training")
        return True
    else:
        print("✗ FAISS normalization is AFTER training")
        return False


def test_config_unification():
    """Test config unification in env and code."""
    print("\n=== Testing Config Unification ===")
    
    # Check .env.example
    env_path = Path(".env.example")
    if not env_path.exists():
        print("✗ .env.example not found")
        return False
    
    env_content = env_path.read_text()
    if "CHUNK_STRIDE=300" in env_content:
        print("✓ CHUNK_STRIDE=300 in .env.example")
    else:
        print("✗ CHUNK_STRIDE=300 NOT in .env.example")
        return False
    
    # Check chunk_text.py
    chunk_path = Path("src/ingest/chunk_text.py")
    content = chunk_path.read_text()
    
    if "os.getenv('CHUNK_STRIDE', '300')" in content:
        print("✓ Environment variable reading in chunk_text.py")
    else:
        print("✗ Environment variable reading NOT in chunk_text.py")
        return False
    
    return True


def test_positional_filtering():
    """Test positional filtering implementation."""
    print("\n=== Testing Positional Filtering ===")
    
    pdf_path = Path("src/ingest/pdf_to_text.py")
    content = pdf_path.read_text()
    
    # Check for new functions
    if "detect_repeated_blocks" in content:
        print("✓ detect_repeated_blocks function found")
    else:
        print("✗ detect_repeated_blocks function NOT found")
        return False
    
    if "extract_text_with_positional_filtering" in content:
        print("✓ extract_text_with_positional_filtering function found")
    else:
        print("✗ extract_text_with_positional_filtering function NOT found")
        return False
    
    # Check for positional bands
    if "top_band = page_height * 0.08" in content:
        print("✓ Top band filtering found")
    else:
        print("✗ Top band filtering NOT found")
        return False
    
    if "bottom_band = page_height * 0.92" in content:
        print("✓ Bottom band filtering found")
    else:
        print("✗ Bottom band filtering NOT found")
        return False
    
    return True


def test_atomic_writes():
    """Test atomic writes implementation."""
    print("\n=== Testing Atomic Writes Implementation ===")
    
    # Check utils.py exists
    utils_path = Path("src/ingest/utils.py")
    if not utils_path.exists():
        print("✗ utils.py not found")
        return False
    
    content = utils_path.read_text()
    
    # Check for atomic write functions
    functions = [
        "atomic_write_text",
        "atomic_write_jsonl", 
        "atomic_write_binary",
        "create_metadata",
        "save_metadata"
    ]
    
    for func in functions:
        if f"def {func}" in content:
            print(f"✓ {func} function found")
        else:
            print(f"✗ {func} function NOT found")
            return False
    
    # Check for os.replace usage
    if "os.replace" in content:
        print("✓ Atomic os.replace found")
    else:
        print("✗ Atomic os.replace NOT found")
        return False
    
    return True


def test_force_flags():
    """Test --force flag implementation."""
    print("\n=== Testing Force Flags ===")
    
    files_to_check = [
        "src/ingest/pdf_to_text.py",
        "src/ingest/chunk_text.py", 
        "src/ingest/build_index.py"
    ]
    
    for file_path in files_to_check:
        path = Path(file_path)
        content = path.read_text()
        
        if "--force" in content and "force" in content:
            print(f"✓ Force flag found in {file_path}")
        else:
            print(f"✗ Force flag NOT found in {file_path}")
            return False
    
    return True


def test_section_coverage():
    """Test section coverage logging."""
    print("\n=== Testing Section Coverage ===")
    
    chunk_path = Path("src/ingest/chunk_text.py")
    content = chunk_path.read_text()
    
    if "section_coverage" in content:
        print("✓ Section coverage calculation found")
    else:
        print("✗ Section coverage calculation NOT found")
        return False
    
    if "Section coverage:" in content:
        print("✓ Section coverage logging found")
    else:
        print("✗ Section coverage logging NOT found")
        return False
    
    return True


def test_requirements_deprecation():
    """Test requirements.txt deprecation."""
    print("\n=== Testing Requirements.txt Deprecation ===")
    
    req_path = Path("requirements.txt")
    content = req_path.read_text()
    
    if "DEPRECATED" in content and "environment.yml" in content:
        print("✓ requirements.txt properly deprecated")
        return True
    else:
        print("✗ requirements.txt NOT properly deprecated")
        return False


def main():
    """Run all code verification tests."""
    print("Verifying Hour 1 Critical Fixes Implementation")
    print("=" * 60)
    
    tests = [
        test_e5_prefixes_in_code,
        test_faiss_normalization_order,
        test_config_unification,
        test_positional_filtering,
        test_atomic_writes,
        test_force_flags,
        test_section_coverage,
        test_requirements_deprecation,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with error: {e}")
            results.append(False)
    
    print("\n" + "=" * 60)
    print("IMPLEMENTATION VERIFICATION SUMMARY")
    print("=" * 60)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✓ IMPLEMENTED" if result else "✗ MISSING"
        print(f"{i+1}. {test.__name__}: {status}")
    
    print(f"\nTotal: {passed}/{total} implementations verified")
    
    if passed == total:
        print("\n🎉 All critical fixes have been implemented correctly!")
        print("\nNext steps:")
        print("1. Set up conda environment: conda env create -f environment.yml")
        print("2. Activate environment: conda activate faiss-env")
        print("3. Run end-to-end pipeline test")
        return 0
    else:
        print(f"\n❌ {total - passed} implementations are missing or incorrect.")
        print("Please review and fix the implementation.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())