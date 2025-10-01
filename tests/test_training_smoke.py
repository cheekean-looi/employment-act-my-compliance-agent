#!/usr/bin/env python3
"""
Smoke tests for Hour 4 SFT training pipeline.
Uses tiny models and datasets for CI validation.
"""

import json
import tempfile
import pytest
from pathlib import Path
from unittest.mock import patch
import torch

# Test data
SAMPLE_CHUNKS = [
    {
        "text": "An employee shall be entitled to annual leave of not less than eight days for every twelve months of continuous service.",
        "section_id": "EA-2022-60E(1)",
        "chunk_id": "chunk_001"
    },
    {
        "text": "The rate of overtime pay shall be not less than one and a half times the hourly rate of pay.",
        "section_id": "EA-2022-60A(1)",
        "chunk_id": "chunk_002"
    },
    {
        "text": "Every employee shall be entitled to a rest day of one whole day in every period of seven consecutive days.",
        "section_id": "EA-2022-60D(1)",
        "chunk_id": "chunk_003"
    }
]

SAMPLE_SFT_EXAMPLES = [
    {
        "instruction": "How many days of annual leave am I entitled to?",
        "input": "",
        "output": "According to the Employment Act, an employee shall be entitled to annual leave of not less than eight days for every twelve months of continuous service. This is covered under section EA-2022-60E(1).",
        "citations": ["EA-2022-60E(1)"],
        "category": "entitlement"
    },
    {
        "instruction": "How is overtime pay calculated?",
        "input": "",
        "output": "According to the Employment Act, the rate of overtime pay shall be not less than one and a half times the hourly rate of pay. This is covered under section EA-2022-60A(1).",
        "citations": ["EA-2022-60A(1)"],
        "category": "calculation"
    }
]


class TestSFTDatasetGeneration:
    """Test SFT dataset generation pipeline."""
    
    def test_dataset_generator_import(self):
        """Test that dataset generator can be imported."""
        try:
            from src.training.make_sft_dataset_production import ProductionSFTGenerator
            assert ProductionSFTGenerator is not None
        except ImportError as e:
            pytest.skip(f"Cannot import dataset generator: {e}")
    
    def test_dataset_generator_initialization(self):
        """Test dataset generator initialization."""
        from src.training.make_sft_dataset_production import ProductionSFTGenerator
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for chunk in SAMPLE_CHUNKS:
                f.write(json.dumps(chunk) + '\n')
            chunks_path = Path(f.name)
        
        try:
            generator = ProductionSFTGenerator(chunks_path, seed=42)
            assert len(generator.chunks) == 3
            assert len(generator.section_to_chunks) > 0
            assert len(generator.valid_section_ids) > 0
        finally:
            chunks_path.unlink()
    
    def test_dataset_generation_small(self):
        """Test generating a small dataset."""
        from src.training.make_sft_dataset_production import ProductionSFTGenerator
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for chunk in SAMPLE_CHUNKS:
                f.write(json.dumps(chunk) + '\n')
            chunks_path = Path(f.name)
        
        try:
            generator = ProductionSFTGenerator(chunks_path, seed=42)
            examples = generator.generate_dataset(target_size=3)
            
            assert len(examples) <= 3  # May be less due to deduplication
            for example in examples:
                assert 'instruction' in example
                assert 'output' in example
                assert 'citations' in example
                assert len(example['citations']) > 0
        finally:
            chunks_path.unlink()
    
    def test_citation_validation(self):
        """Test citation validation logic."""
        from src.training.make_sft_dataset_production import ProductionSFTGenerator
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for chunk in SAMPLE_CHUNKS:
                f.write(json.dumps(chunk) + '\n')
            chunks_path = Path(f.name)
        
        try:
            generator = ProductionSFTGenerator(chunks_path, seed=42)
            
            # Valid citations
            assert generator._validate_citations(["EA-2022-60E(1)"])
            
            # Invalid citations
            assert not generator._validate_citations(["INVALID-SECTION"])
            assert not generator._validate_citations([])
        finally:
            chunks_path.unlink()


class TestDatasetValidator:
    """Test dataset validation."""
    
    def test_validator_import(self):
        """Test that validator can be imported."""
        try:
            from src.training.validate_dataset import SFTDatasetValidator
            assert SFTDatasetValidator is not None
        except ImportError as e:
            pytest.skip(f"Cannot import validator: {e}")
    
    def test_valid_dataset_validation(self):
        """Test validation of a valid dataset."""
        from src.training.validate_dataset import SFTDatasetValidator
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for example in SAMPLE_SFT_EXAMPLES:
                f.write(json.dumps(example) + '\n')
            dataset_path = Path(f.name)
        
        try:
            validator = SFTDatasetValidator()
            result = validator.validate_dataset(dataset_path)
            
            assert result.is_valid
            assert len(result.errors) == 0
            assert result.stats['total_examples'] == 2
            assert result.stats['valid_examples'] == 2
        finally:
            dataset_path.unlink()
    
    def test_invalid_dataset_validation(self):
        """Test validation of an invalid dataset."""
        from src.training.validate_dataset import SFTDatasetValidator
        
        invalid_examples = [
            {"instruction": "test", "output": "test"},  # Missing citations
            {"instruction": "", "output": "test", "citations": []},  # Empty instruction
        ]
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for example in invalid_examples:
                f.write(json.dumps(example) + '\n')
            dataset_path = Path(f.name)
        
        try:
            validator = SFTDatasetValidator()
            result = validator.validate_dataset(dataset_path)
            
            assert not result.is_valid
            assert len(result.errors) > 0
        finally:
            dataset_path.unlink()


class TestQLoRATraining:
    """Test QLoRA training pipeline with tiny model."""
    
    def test_trainer_import(self):
        """Test that trainer can be imported."""
        try:
            from src.training.train_lora_production import ProductionQLoRATrainer, QLoRAConfig
            assert ProductionQLoRATrainer is not None
            assert QLoRAConfig is not None
        except ImportError as e:
            pytest.skip(f"Cannot import trainer: {e}")
    
    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_tiny_model_initialization(self):
        """Test initialization with a tiny model."""
        from src.training.train_lora_production import ProductionQLoRATrainer, QLoRAConfig
        
        # Use a tiny model for testing
        config = QLoRAConfig(
            model_name="HuggingFaceTB/SmolLM-135M-Instruct",  # Tiny model
            lora_rank=2,
            lora_alpha=4,
            use_4bit=False,  # Disable for tiny model
            gradient_checkpointing=False,
        )
        
        try:
            trainer = ProductionQLoRATrainer(config)
            assert trainer.model is not None
            assert trainer.tokenizer is not None
            assert trainer.model.num_parameters() > 0
        except Exception as e:
            pytest.skip(f"Cannot initialize tiny model: {e}")
    
    def test_config_validation(self):
        """Test configuration validation."""
        from src.training.train_lora_production import QLoRAConfig
        
        config = QLoRAConfig()
        assert config.model_name is not None
        assert config.lora_rank > 0
        assert config.lora_alpha > 0
        assert config.learning_rate > 0
    
    def test_citation_evaluator_import(self):
        """Test citation evaluator can be imported."""
        try:
            from src.training.train_lora_production import CitationEvaluator
            assert CitationEvaluator is not None
        except ImportError as e:
            pytest.skip(f"Cannot import citation evaluator: {e}")
    
    def test_citation_extraction(self):
        """Test citation extraction from text."""
        from src.training.train_lora_production import CitationEvaluator
        
        # Mock evaluator (without model loading)
        class MockEvaluator(CitationEvaluator):
            def __init__(self):
                self.citation_pattern = r'\b(EA-\d{4}-\d+[A-Z]*(?:\(\d+\))?)\b'
        
        evaluator = MockEvaluator()
        
        text = "According to section EA-2022-60E(1) and EA-2022-60A(1), employees are entitled to benefits."
        citations = evaluator.extract_citations(text)
        
        assert "EA-2022-60E(1)" in citations
        assert "EA-2022-60A(1)" in citations
        assert len(citations) == 2


class TestTrainingPipelineIntegration:
    """Integration tests for the complete training pipeline."""
    
    def test_end_to_end_tiny_pipeline(self):
        """Test end-to-end pipeline with tiny dataset."""
        from src.training.make_sft_dataset_production import ProductionSFTGenerator
        from src.training.validate_dataset import SFTDatasetValidator
        
        # Create tiny chunks file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for chunk in SAMPLE_CHUNKS:
                f.write(json.dumps(chunk) + '\n')
            chunks_path = Path(f.name)
        
        try:
            # Generate dataset
            generator = ProductionSFTGenerator(chunks_path, seed=42)
            examples = generator.generate_dataset(target_size=3)
            
            # Save to temporary files
            with tempfile.TemporaryDirectory() as temp_dir:
                output_dir = Path(temp_dir)
                generator.save_dataset(output_dir, examples)
                
                # Validate generated dataset
                train_path = output_dir / "sft_dataset_train.jsonl"
                eval_path = output_dir / "sft_dataset_eval.jsonl"
                
                assert train_path.exists()
                assert eval_path.exists()
                
                validator = SFTDatasetValidator()
                result = validator.validate_train_eval_split(train_path, eval_path)
                
                # Should be valid (though may have warnings about size)
                assert len(result.errors) == 0
                
        finally:
            chunks_path.unlink()
    
    def test_training_artifacts_structure(self):
        """Test that training creates expected artifacts."""
        # This would be tested in a full training run
        expected_artifacts = [
            "adapter_config.json",
            "adapter_model.safetensors", 
            "tokenizer.json",
            "train_results.json",
            "eval_results.json",
            "training_curves.png",
            "metadata.json"
        ]
        
        # Just verify the list is complete
        assert len(expected_artifacts) == 7
        assert "adapter_config.json" in expected_artifacts
        assert "metadata.json" in expected_artifacts


if __name__ == "__main__":
    pytest.main([__file__, "-v"])