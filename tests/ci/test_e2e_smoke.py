#!/usr/bin/env python3
"""
Smoke test for end-to-end ingestion pipeline (PDF → text → chunks → index).
This test verifies the pipeline works without requiring full environment setup.
Suitable for CI/CD to catch breaking changes.
"""

import json
import os
import sys
import tempfile
from pathlib import Path
import shutil
import subprocess


def create_test_pdf_content() -> str:
    """Create mock PDF content for testing."""
    return """Employment Act 1955

Section 60A - Overtime
(1) Any employee may be required by his employer to work for more than the normal hours of work.
(2) No employee shall work for more than twelve hours in any one day.

Section 61 - Payment for overtime
(1) An employee shall be entitled to receive payment for overtime at the rate of not less than one and a half times his hourly rate of pay.

Page 1 of 1
© Government of Malaysia 2023"""


def create_mock_text_jsonl(temp_dir: Path) -> Path:
    """Create mock text.jsonl file for testing."""
    text_file = temp_dir / 'text.jsonl'
    
    test_doc = {
        'id': 'test_employment_act',
        'text': create_test_pdf_content(),
        'metadata': {
            'title': 'Employment Act 1955',
            'filename': 'test.pdf',
            'file_size': 1000
        },
        'source_file': 'test.pdf',
        'page_count': 1,
        'text_length': len(create_test_pdf_content()),
        'extraction_method': 'mock'
    }
    
    with open(text_file, 'w') as f:
        f.write(json.dumps(test_doc) + '\n')
    
    return text_file


def test_chunk_text_module():
    """Test chunk_text.py can process mock data."""
    print("=== Testing chunk_text.py ===")
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        
        # Create input file
        input_file = create_mock_text_jsonl(temp_path)
        output_file = temp_path / 'chunks.jsonl'
        
        # Run chunk_text.py
        cmd = [
            sys.executable, 'src/ingest/chunk_text.py',
            '--in', str(input_file),
            '--out', str(output_file),
            '--chunk', '500',
            '--stride', '150'
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            print("✓ chunk_text.py executed successfully")
            
            # Check output
            if output_file.exists():
                with open(output_file) as f:
                    chunks = [json.loads(line) for line in f]
                
                if len(chunks) > 0:
                    print(f"✓ Generated {len(chunks)} chunks")
                    
                    # Check chunk structure
                    chunk = chunks[0]
                    required_fields = ['chunk_id', 'text', 'section_id', 'chunk_index']
                    
                    if all(field in chunk for field in required_fields):
                        print("✓ Chunk structure is correct")
                        return True
                    else:
                        print("✗ Chunk missing required fields")
                        return False
                else:
                    print("✗ No chunks generated")
                    return False
            else:
                print("✗ Output file not created")
                return False
                
        except subprocess.CalledProcessError as e:
            print(f"✗ chunk_text.py failed: {e}")
            print(f"Stderr: {e.stderr}")
            return False


def test_chunk_text_imports():
    """Test that chunk_text.py imports work."""
    print("=== Testing chunk_text.py imports ===")
    
    try:
        # Test the main imports
        sys.path.insert(0, 'src')
        from src.ingest.utils import create_metadata, atomic_write_jsonl
        from src.ingest.chunk_text import detect_section_id, create_chunk_id
        print("✓ All critical imports successful")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def test_utils_module():
    """Test utils.py functionality."""
    print("=== Testing utils.py ===")
    
    try:
        sys.path.insert(0, 'src')
        from src.ingest.utils import compute_file_hash, get_environment_info, compute_config_hash
        
        # Test config hash
        config = {'test': 'config', 'value': 123}
        hash_val = compute_config_hash(config)
        if len(hash_val) == 16:  # Should be 16 char hash
            print("✓ Config hash generation working")
        else:
            print("✗ Config hash has wrong length")
            return False
        
        # Test environment info
        env_info = get_environment_info()
        if 'python_version' in env_info and 'platform' in env_info:
            print("✓ Environment info generation working")
            return True
        else:
            print("✗ Environment info missing required fields")
            return False
            
    except Exception as e:
        print(f"✗ Utils test failed: {e}")
        return False


def test_section_detection():
    """Test section detection logic."""
    print("=== Testing Section Detection ===")
    
    try:
        sys.path.insert(0, 'src')
        from src.ingest.chunk_text import detect_section_id, extract_section_title
        
        # Test cases
        test_cases = [
            ("Section 60A - Overtime", "EA-60A"),
            ("60A. Overtime payment", "EA-60A"),
            ("60A Overtime provisions", "EA-60A"),
            ("Part III of the Act", "EA-III"),
            ("No section here", None)
        ]
        
        passed = 0
        for text, expected in test_cases:
            result = detect_section_id(text)
            if result == expected:
                passed += 1
            else:
                print(f"✗ Section detection failed: '{text}' -> {result} (expected {expected})")
        
        if passed == len(test_cases):
            print(f"✓ All {len(test_cases)} section detection tests passed")
            return True
        else:
            print(f"✗ {len(test_cases) - passed} section detection tests failed")
            return False
            
    except Exception as e:
        print(f"✗ Section detection test failed: {e}")
        return False


def test_config_unification():
    """Test that config is properly unified."""
    print("=== Testing Config Unification ===")
    
    # Check .env.example
    env_path = Path('.env.example')
    if not env_path.exists():
        print("✗ .env.example not found")
        return False
    
    env_content = env_path.read_text()
    if "CHUNK_STRIDE=300" in env_content:
        print("✓ CHUNK_STRIDE=300 found in .env.example")
    else:
        print("✗ CHUNK_STRIDE=300 NOT found in .env.example")
        return False
    
    # Check chunk_text.py reads from env
    chunk_path = Path('src/ingest/chunk_text.py')
    chunk_content = chunk_path.read_text()
    
    if "os.getenv('CHUNK_STRIDE'" in chunk_content:
        print("✓ Environment variable reading found in chunk_text.py")
        return True
    else:
        print("✗ Environment variable reading NOT found in chunk_text.py")
        return False


def test_metadata_structure():
    """Test metadata structure is correct."""
    print("=== Testing Metadata Structure ===")
    
    try:
        sys.path.insert(0, 'src')
        from src.ingest.utils import create_metadata
        
        # Create test metadata
        metadata = create_metadata(
            stage='test',
            input_files=[],
            output_files=[],
            config={'test': 'value'}
        )
        
        # Check required fields
        required_fields = [
            'stage', 'timestamp', 'git_sha', 'environment', 
            'config', 'config_hash', 'input_files', 'output_files'
        ]
        
        missing_fields = [field for field in required_fields if field not in metadata]
        
        if not missing_fields:
            print("✓ Metadata structure is correct")
            return True
        else:
            print(f"✗ Metadata missing fields: {missing_fields}")
            return False
            
    except Exception as e:
        print(f"✗ Metadata structure test failed: {e}")
        return False


def main():
    """Run all smoke tests."""
    print("Running End-to-End Smoke Test")
    print("=" * 50)
    print("This test verifies core functionality without requiring full environment setup")
    print()
    
    tests = [
        test_utils_module,
        test_config_unification,
        test_metadata_structure,
        test_section_detection,
        test_chunk_text_imports,
        test_chunk_text_module,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
            print()
        except Exception as e:
            print(f"✗ Test {test.__name__} failed with error: {e}")
            results.append(False)
            print()
    
    print("=" * 50)
    print("SMOKE TEST SUMMARY")
    print("=" * 50)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{i+1}. {test.__name__}: {status}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All smoke tests passed! Pipeline is functional.")
        print("\nTo run full end-to-end test:")
        print("1. conda activate faiss-env")
        print("2. Run with real PDFs using the ingestion pipeline")
        return 0
    else:
        print(f"\n❌ {total - passed} smoke tests failed.")
        print("Pipeline has issues that need to be addressed.")
        return 1


if __name__ == "__main__":
    sys.exit(main())