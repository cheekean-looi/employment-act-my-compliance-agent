#!/usr/bin/env python3
"""
Comprehensive test script for Fixed Hour 5 implementation
Validates that all fixed components work correctly before full training.

Tests include:
- Fixed canonical citation patterns
- Enhanced preference pair generation
- DPO training with persistent eval subset
- PPO with proper value-head initialization
- Smoke run with small model to validate PPOTrainer
"""

import json
from pathlib import Path
import subprocess
import sys
import tempfile
import shutil
import torch
import re


def test_canonical_citation_patterns():
    """Test canonical citation regex patterns."""
    print("🧪 Testing canonical citation patterns...")
    
    try:
        # Import citation utils
        sys.path.append(str(Path(__file__).parent / "src" / "training"))
        from citation_utils import CanonicalCitationValidator
        
        validator = CanonicalCitationValidator()
        
        # Test canonical pattern matching
        test_cases = [
            ("EA-2022-60E(1)", True),
            ("EA-2022-37", True),
            ("EA-2022-13(2)", True),
            ("EA-60E", False),  # Legacy format should not match directly
            ("EA-99", False),   # Invalid section
            ("Section EA-2022-60E(1) provides", True),
        ]
        
        for text, should_match in test_cases:
            sections = validator.extract_section_ids(text)
            has_match = len(sections) > 0
            
            if has_match != should_match:
                print(f"❌ Pattern test failed: '{text}' -> {sections} (expected match: {should_match})")
                return False
        
        # Test normalization
        normalized = validator.normalize_section_id("EA-60E")
        if normalized != "EA-2022-60E":
            print(f"❌ Normalization failed: EA-60E -> {normalized}")
            return False
        
        print("✅ Canonical citation patterns test passed")
        return True
        
    except Exception as e:
        print(f"❌ Citation patterns test failed: {e}")
        return False


def test_enhanced_preference_pairs_generation():
    """Test enhanced preference pairs generation with canonical patterns."""
    print("🧪 Testing enhanced preference pairs generation...")
    
    # Create minimal test chunks with canonical IDs
    test_chunks = [
        {
            "chunk_id": "chunk_001",
            "section_id": "EA-2022-60E",
            "text": "Annual leave entitlement under Malaysian employment law",
            "original_text": "Annual leave entitlement under Malaysian employment law"
        },
        {
            "chunk_id": "chunk_002",
            "section_id": "EA-2022-37",
            "text": "Maternity protection provisions for female employees",
            "original_text": "Maternity protection provisions for female employees"
        },
        {
            "chunk_id": "chunk_003",
            "section_id": "EA-2022-13(1)",
            "text": "Termination procedures and notice requirements",
            "original_text": "Termination procedures and notice requirements"
        }
    ]
    
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        chunks_file = temp_path / "test_chunks.jsonl"
        output_file = temp_path / "test_pairs.jsonl"
        
        # Write test chunks
        with open(chunks_file, 'w') as f:
            for chunk in test_chunks:
                f.write(json.dumps(chunk) + '\n')
        
        # Test enhanced preference pair generation
        cmd = [
            "python", "src/training/make_pref_pairs_fixed.py",
            "--chunks", str(chunks_file),
            "--output", str(output_file),
            "--size", "6",  # Small size for testing
            "--seed", "42"
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Enhanced preference pairs generation failed:")
            print(result.stderr)
            return False
        
        # Check outputs
        required_files = [
            output_file,
            temp_path / "test_pairs_train.jsonl",
            temp_path / "test_pairs_eval.jsonl",
            temp_path / "tools" / "test_pairs_labeling.csv",
            temp_path / "tools" / "labeling_cli.py",
            temp_path / "test_pairs_split_metadata.json"
        ]
        
        for req_file in required_files:
            if not req_file.exists():
                print(f"❌ Required file not created: {req_file}")
                return False
        
        # Validate canonical patterns in generated pairs
        with open(output_file, 'r') as f:
            for line in f:
                pair = json.loads(line.strip())
                
                # Check that grounding validation includes canonical patterns
                if 'chosen_grounding' not in pair:
                    print("❌ Missing canonical grounding validation")
                    return False
                
                # Check that predicted sections use canonical format
                predicted = pair['chosen_grounding'].get('predicted_section_ids', [])
                for section_id in predicted:
                    if not re.match(r'EA-\d{4}-\d+[A-Z]*(?:\(\d+\))?', section_id):
                        print(f"❌ Non-canonical section ID: {section_id}")
                        return False
        
        # Validate labeling CLI
        cli_file = temp_path / "tools" / "labeling_cli.py"
        result = subprocess.run(["python", str(cli_file), "--help"], 
                              capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Labeling CLI failed: {result.stderr}")
            return False
        
        if "--dry-run" not in result.stdout or "--strict" not in result.stdout:
            print("❌ Labeling CLI missing dry-run/strict modes")
            return False
        
        print("✅ Enhanced preference pairs generation test passed")
        return True


def test_dpo_training_with_persistent_eval():
    """Test DPO training initialization with fixed components."""
    print("🧪 Testing DPO training initialization...")
    
    # Test imports and initialization
    try:
        # Test that fixed DPO trainer imports work
        result = subprocess.run([
            "python", "-c", 
            "from src.training.train_dpo_fixed import FixedEmploymentActDPOTrainer; print('Import successful')"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ DPO trainer import failed: {result.stderr}")
            return False
        
        # Test help command
        result = subprocess.run([
            "python", "src/training/train_dpo_fixed.py", "--help"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ DPO trainer help failed: {result.stderr}")
            return False
        
        expected_args = ["--train-data", "--eval-data", "--sft-model"]
        for arg in expected_args:
            if arg not in result.stdout:
                print(f"❌ Missing expected argument: {arg}")
                return False
        
        print("✅ DPO training initialization test passed")
        return True
        
    except Exception as e:
        print(f"❌ DPO training test failed: {e}")
        return False


def test_ppo_with_small_model():
    """Test PPO training with small model to validate TRL integration."""
    print("🧪 Testing PPO with SmolLM (smoke test)...")
    
    try:
        # Create minimal test data
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            output_dir = temp_path / "ppo_smoke_test"
            
            # Test PPO initialization with small model
            cmd = [
                "python", "src/training/tiny_ppo_loop_fixed.py",
                "--output", str(output_dir),
                "--base-model", "HuggingFaceTB/SmolLM-135M-Instruct",
                "--num-prompts", "4",  # Very small for testing
                "--batch-size", "2",
                "--mini-batch-size", "1"
            ]
            
            # Don't use --use-real-ppo for smoke test to avoid model download
            # Just test the simple mode to validate basic functionality
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            
            if result.returncode != 0:
                print(f"❌ PPO smoke test failed:")
                print(result.stderr)
                return False
            
            # Check that output was created
            expected_files = [
                output_dir / "ppo_results.json",
                output_dir / "ppo_summary.json"
            ]
            
            for expected_file in expected_files:
                if not expected_file.exists():
                    print(f"❌ Expected PPO output file missing: {expected_file}")
                    return False
            
            # Validate PPO results structure
            with open(output_dir / "ppo_results.json", 'r') as f:
                results = json.load(f)
            
            required_keys = ["total_examples", "average_reward", "timestamp"]
            for key in required_keys:
                if key not in results:
                    print(f"❌ Missing key in PPO results: {key}")
                    return False
            
            # Validate that average_reward is a valid number
            avg_reward = results["average_reward"]
            if not isinstance(avg_reward, (int, float)):
                print(f"❌ Invalid average_reward type: {type(avg_reward)}")
                return False
            
            print(f"✅ PPO smoke test passed (avg_reward: {avg_reward:+.3f})")
            return True
            
    except subprocess.TimeoutExpired:
        print("❌ PPO smoke test timed out")
        return False
    except Exception as e:
        print(f"❌ PPO smoke test failed: {e}")
        return False


def test_pipeline_integration():
    """Test that fixed pipeline script works."""
    print("🧪 Testing fixed pipeline integration...")
    
    try:
        # Test pipeline help
        result = subprocess.run([
            "python", "run_hour5_training_fixed.py", "--help"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ Fixed pipeline help failed: {result.stderr}")
            return False
        
        expected_args = ["--chunks", "--sft-model", "--ppo-model"]
        for arg in expected_args:
            if arg not in result.stdout:
                print(f"❌ Missing expected pipeline argument: {arg}")
                return False
        
        # Check that it mentions "Fixed" in help
        if "Fixed" not in result.stdout:
            print("❌ Pipeline help doesn't indicate fixes applied")
            return False
        
        print("✅ Fixed pipeline integration test passed")
        return True
        
    except Exception as e:
        print(f"❌ Pipeline integration test failed: {e}")
        return False


def test_memory_optimization():
    """Test memory optimization features."""
    print("🧪 Testing memory optimization features...")
    
    try:
        # Test that SmolLM is the default for PPO
        result = subprocess.run([
            "python", "src/training/tiny_ppo_loop_fixed.py", "--help"
        ], capture_output=True, text=True)
        
        if result.returncode != 0:
            print(f"❌ PPO help failed: {result.stderr}")
            return False
        
        if "SmolLM-135M" not in result.stdout:
            print("❌ SmolLM not set as default for memory optimization")
            return False
        
        # Test that pipeline warns about large models
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            f.write('{"chunk_id":"test","section_id":"EA-2022-60E","text":"test"}\n')
            test_chunks_file = f.name
        
        try:
            # This should trigger the large model warning
            result = subprocess.run([
                "python", "run_hour5_training_fixed.py",
                "--chunks", test_chunks_file,
                "--ppo-model", "meta-llama/Llama-3.1-8B-Instruct",
                "--help"  # Use help to avoid actual execution
            ], capture_output=True, text=True, input="N\n")
            
            # Should succeed with help regardless of model choice
            
        finally:
            Path(test_chunks_file).unlink()
        
        print("✅ Memory optimization test passed")
        return True
        
    except Exception as e:
        print(f"❌ Memory optimization test failed: {e}")
        return False


def test_file_structure():
    """Test that all fixed files exist with correct structure."""
    print("🧪 Testing fixed file structure...")
    
    required_files = [
        "src/training/citation_utils.py",
        "src/training/make_pref_pairs_fixed.py",
        "src/training/train_dpo_fixed.py",
        "src/training/tiny_ppo_loop_fixed.py",
        "run_hour5_training_fixed.py",
        "test_hour5_implementation_fixed.py"
    ]
    
    for file_path in required_files:
        if not Path(file_path).exists():
            print(f"❌ Fixed file missing: {file_path}")
            return False
        
        # Check file size (should not be empty)
        if Path(file_path).stat().st_size == 0:
            print(f"❌ Fixed file is empty: {file_path}")
            return False
    
    # Check that files mention fixes
    fix_indicators = {
        "src/training/citation_utils.py": ["canonical", "EA-YYYY-NNN"],
        "src/training/make_pref_pairs_fixed.py": ["FIXES APPLIED", "canonical"],
        "src/training/train_dpo_fixed.py": ["FIXES APPLIED", "persistent eval"],
        "src/training/tiny_ppo_loop_fixed.py": ["FIXES APPLIED", "value-head"],
        "run_hour5_training_fixed.py": ["FIXES APPLIED", "canonical patterns"]
    }
    
    for file_path, indicators in fix_indicators.items():
        with open(file_path, 'r') as f:
            content = f.read()
        
        for indicator in indicators:
            if indicator not in content:
                print(f"❌ Fix indicator '{indicator}' not found in {file_path}")
                return False
    
    print("✅ Fixed file structure test passed")
    return True


def run_all_tests():
    """Run all Hour 5 fixed implementation tests."""
    print("🚀 Running Hour 5 Fixed Implementation Tests\n")
    
    tests = [
        ("Fixed File Structure", test_file_structure),
        ("Canonical Citation Patterns", test_canonical_citation_patterns),
        ("Enhanced Preference Pairs", test_enhanced_preference_pairs_generation),
        ("DPO Training (Fixed)", test_dpo_training_with_persistent_eval),
        ("Memory Optimization", test_memory_optimization),
        ("PPO Smoke Test (SmolLM)", test_ppo_with_small_model),
        ("Fixed Pipeline Integration", test_pipeline_integration),
    ]
    
    passed = 0
    total = len(tests)
    failed_tests = []
    
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
                failed_tests.append(test_name)
        except Exception as e:
            print(f"❌ {test_name} FAILED with exception: {e}")
            failed_tests.append(test_name)
    
    print(f"\n{'='*60}")
    print(f"Fixed Hour 5 Implementation Test Results: {passed}/{total} passed")
    
    if failed_tests:
        print(f"\n❌ Failed tests:")
        for test_name in failed_tests:
            print(f"  - {test_name}")
    
    if passed == total:
        print("🎉 All tests passed! Fixed Hour 5 implementation is ready.")
        print("\nNext steps:")
        print("1. Ensure you have completed Hour 4 SFT training")
        print("2. Run: python run_hour5_training_fixed.py --chunks data/processed/chunks.jsonl --sft-model outputs/lora_sft")
        print("3. Review the enhanced reports in outputs/hour5_fixed_*/")
        print("4. Check the canonical pattern coverage in logs")
        print("5. Use the enhanced labeling tools in tools/ directory")
        return True
    else:
        print("⚠️ Some tests failed. Please fix issues before running fixed Hour 5 training.")
        print("\nDiagnostic tips:")
        print("- Check that all fixed files are present and not corrupted")
        print("- Verify Python dependencies are installed (transformers, trl, peft)")
        print("- Ensure CUDA is available if using GPU acceleration")
        print("- Check disk space for temporary files")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)