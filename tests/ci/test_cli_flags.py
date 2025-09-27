#!/usr/bin/env python3
"""Test CLI flags are properly wired."""

import subprocess
import sys

def test_build_index_flags():
    """Test build_index.py CLI flags."""
    print("=== Testing build_index.py CLI flags ===")
    
    try:
        result = subprocess.run([
            sys.executable, 'src/ingest/build_index.py', '--help'
        ], capture_output=True, text=True, check=True)
        
        help_text = result.stdout
        
        if '--bm25-rebuild' in help_text:
            print("✓ --bm25-rebuild flag found in build_index.py")
        else:
            print("✗ --bm25-rebuild flag NOT found in build_index.py")
            return False
            
        if 'cross-env compatibility' in help_text:
            print("✓ BM25 rebuild help text is descriptive")
        else:
            print("✗ BM25 rebuild help text missing description")
            
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ build_index.py help failed: {e.stderr}")
        return False


def test_pdf_to_text_flags():
    """Test pdf_to_text.py CLI flags."""
    print("\n=== Testing pdf_to_text.py CLI flags ===")
    
    try:
        result = subprocess.run([
            sys.executable, 'src/ingest/pdf_to_text.py', '--help'
        ], capture_output=True, text=True, check=True)
        
        help_text = result.stdout
        
        if '--top-band' in help_text:
            print("✓ --top-band flag found in pdf_to_text.py")
        else:
            print("✗ --top-band flag NOT found in pdf_to_text.py")
            return False
            
        if '--bottom-band' in help_text:
            print("✓ --bottom-band flag found in pdf_to_text.py")
        else:
            print("✗ --bottom-band flag NOT found in pdf_to_text.py")
            return False
            
        if 'header band percentage' in help_text:
            print("✓ Band help text is descriptive")
        else:
            print("✗ Band help text missing description")
            
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"✗ pdf_to_text.py help failed: {e.stderr}")
        return False


def main():
    """Test all CLI flags."""
    print("Testing CLI Flag Implementation")
    print("=" * 40)
    
    tests = [
        test_pdf_to_text_flags,
        test_build_index_flags,
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} failed: {e}")
            results.append(False)
    
    passed = sum(results)
    total = len(results)
    
    print(f"\n{'='*40}")
    print(f"CLI FLAGS TEST SUMMARY")
    print(f"{'='*40}")
    print(f"Total: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All CLI flags are properly implemented!")
        return 0
    else:
        print("❌ Some CLI flags are missing or incorrect.")
        return 1


if __name__ == "__main__":
    sys.exit(main())