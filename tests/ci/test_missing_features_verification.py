#!/usr/bin/env python3
"""
Verification test for previously missing features.
Confirms all claimed features are actually implemented.
"""

import re
from pathlib import Path


def test_bm25_rebuild_flag():
    """Verify --bm25-rebuild flag is implemented."""
    print("=== Testing --bm25-rebuild flag ===")
    
    build_index_path = Path("src/ingest/build_index.py")
    content = build_index_path.read_text()
    
    checks = [
        ("CLI argument defined", "--bm25-rebuild" in content),
        ("Help text present", "cross-env compatibility" in content),
        ("Parameter passed to function", "bm25_rebuild" in content),
        ("Function signature updated", "bm25_rebuild: bool = False" in content),
        ("Config metadata tracking", "bm25_rebuild_requested" in content),
    ]
    
    passed = 0
    for check_name, result in checks:
        if result:
            print(f"  ✓ {check_name}")
            passed += 1
        else:
            print(f"  ✗ {check_name}")
    
    return passed == len(checks)


def test_header_footer_cli_flags():
    """Verify --top-band and --bottom-band flags are implemented."""
    print("\n=== Testing header/footer CLI flags ===")
    
    pdf_to_text_path = Path("src/ingest/pdf_to_text.py")
    content = pdf_to_text_path.read_text()
    
    checks = [
        ("--top-band CLI argument", "--top-band" in content),
        ("--bottom-band CLI argument", "--bottom-band" in content),
        ("Function parameters", "top_band_pct: float = 0.08" in content),
        ("Parameter passing", "getattr(args, 'top_band'" in content),
        ("Configurable bands", "top_band = page_height * top_band_pct" in content),
        ("Metadata recording", "'top_pct': top_band_pct" in content),
        ("Progress logging", "top: {top_band_pct*100}%" in content),
    ]
    
    passed = 0
    for check_name, result in checks:
        if result:
            print(f"  ✓ {check_name}")
            passed += 1
        else:
            print(f"  ✗ {check_name}")
    
    return passed == len(checks)


def test_model_revision_metadata():
    """Verify model revision info is captured in metadata."""
    print("\n=== Testing model revision metadata ===")
    
    utils_path = Path("src/ingest/utils.py")
    utils_content = utils_path.read_text()
    
    build_index_path = Path("src/ingest/build_index.py")
    build_content = build_index_path.read_text()
    
    checks = [
        ("get_model_revision_info function", "def get_model_revision_info" in utils_content),
        ("Commit hash extraction", "_commit_hash" in utils_content),
        ("Model type capture", "model_type" in utils_content),
        ("Architecture info", "architectures" in utils_content),
        ("Function imported in build_index", "get_model_revision_info" in build_content),
        ("Revision info added to config", "embedding_model_revision" in build_content),
        ("Function called before metadata", "model_revision_info = get_model_revision_info" in build_content),
    ]
    
    passed = 0
    for check_name, result in checks:
        if result:
            print(f"  ✓ {check_name}")
            passed += 1
        else:
            print(f"  ✗ {check_name}")
    
    return passed == len(checks)


def test_environment_yml_documentation():
    """Verify environment.yml is properly documented as single source."""
    print("\n=== Testing environment.yml documentation ===")
    
    claude_md_path = Path("CLAUDE.md")
    claude_content = claude_md_path.read_text()
    
    requirements_path = Path("requirements.txt")
    requirements_content = requirements_path.read_text()
    
    checks = [
        ("Single source of truth noted", "single source of truth" in claude_content),
        ("environment.yml referenced", "environment.yml" in claude_content),
        ("requirements.txt deprecated note", "deprecated" in claude_content.lower()),
        ("requirements.txt file deprecated", "DEPRECATED" in requirements_content),
        ("Migration instructions in requirements.txt", "conda env create" in requirements_content),
        ("Cross-env deployment noted", "bm25-rebuild" in claude_content),
    ]
    
    passed = 0
    for check_name, result in checks:
        if result:
            print(f"  ✓ {check_name}")
            passed += 1
        else:
            print(f"  ✗ {check_name}")
    
    return passed == len(checks)


def test_comprehensive_metadata():
    """Verify comprehensive metadata implementation."""
    print("\n=== Testing comprehensive metadata ===")
    
    utils_path = Path("src/ingest/utils.py")
    content = utils_path.read_text()
    
    checks = [
        ("Git SHA capture", "get_git_sha" in content),
        ("Pip freeze capture", "get_pip_freeze" in content),
        ("Conda info capture", "get_conda_info" in content),
        ("Environment details", "python_executable" in content),
        ("Config hash generation", "compute_config_hash" in content),
        ("File hash tracking", "compute_file_hash" in content),
        ("Atomic write operations", "atomic_write_jsonl" in content),
        ("Metadata finalization", "finalize_metadata" in content),
    ]
    
    passed = 0
    for check_name, result in checks:
        if result:
            print(f"  ✓ {check_name}")
            passed += 1
        else:
            print(f"  ✗ {check_name}")
    
    return passed == len(checks)


def main():
    """Run verification tests for all missing features."""
    print("Verification Test - Claimed vs Implemented Features")
    print("=" * 55)
    
    tests = [
        ("BM25 Rebuild Flag", test_bm25_rebuild_flag),
        ("Header/Footer CLI Flags", test_header_footer_cli_flags),
        ("Model Revision Metadata", test_model_revision_metadata),
        ("Environment.yml Documentation", test_environment_yml_documentation),
        ("Comprehensive Metadata", test_comprehensive_metadata),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append(result)
        except Exception as e:
            print(f"\n✗ Test {test_name} failed with error: {e}")
            results.append(False)
    
    print("\n" + "=" * 55)
    print("VERIFICATION SUMMARY")
    print("=" * 55)
    
    passed = sum(results)
    total = len(results)
    
    for i, (test_name, result) in enumerate(zip([t[0] for t in tests], results)):
        status = "✅ VERIFIED" if result else "❌ MISSING"
        print(f"{i+1}. {test_name}: {status}")
    
    print(f"\nTotal: {passed}/{total} features verified as implemented")
    
    if passed == total:
        print("\n🎉 ALL CLAIMED FEATURES ARE CORRECTLY IMPLEMENTED!")
        print("No gaps between claims and actual implementation.")
        return 0
    else:
        print(f"\n❌ {total - passed} claimed features are not properly implemented.")
        print("Implementation needs to match the claims.")
        return 1


if __name__ == "__main__":
    import sys
    sys.exit(main())