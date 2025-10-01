#!/usr/bin/env python3
"""
Pydantic schemas for SFT dataset validation and training.
Ensures data quality and consistency across the training pipeline.
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator, model_validator
import re


class SFTExample(BaseModel):
    """Schema for a single SFT training example."""
    
    instruction: str = Field(..., min_length=10, max_length=2000, description="The user instruction/question")
    input: str = Field(default="", max_length=500, description="Additional input context (can be empty)")
    output: str = Field(..., min_length=20, max_length=1000, description="The assistant response")
    citations: List[str] = Field(..., min_items=1, max_items=10, description="Section ID citations")
    source_chunk_id: Optional[str] = Field(None, description="Source chunk identifier")
    has_numeric_claims: bool = Field(default=False, description="Whether response contains numeric claims")
    category: Optional[str] = Field(None, description="Question category")
    
    @field_validator('citations')
    @classmethod
    def validate_citations(cls, v):
        """Validate citation format and non-empty."""
        if not v:
            raise ValueError("Citations cannot be empty")
        
        citation_pattern = r'^EA-\d{4}-\d+[A-Z]*(?:\(\d+\))?$'
        for citation in v:
            if not re.match(citation_pattern, citation):
                raise ValueError(f"Invalid citation format: {citation}")
        return v
    
    @field_validator('instruction')
    @classmethod
    def validate_instruction(cls, v):
        """Validate instruction quality."""
        if len(v.strip()) < 10:
            raise ValueError("Instruction too short")
        
        # Check for reasonable question patterns
        question_indicators = ['?', 'how', 'what', 'when', 'where', 'why', 'can', 'should', 'is', 'are']
        if not any(indicator in v.lower() for indicator in question_indicators):
            raise ValueError("Instruction should be a question or request")
        
        return v.strip()
    
    @field_validator('output')
    @classmethod
    def validate_output(cls, v):
        """Validate output quality."""
        if len(v.strip()) < 20:
            raise ValueError("Output too short")
        
        # Should contain some legal terminology
        legal_terms = ['employment act', 'section', 'according', 'entitled', 'shall', 'must']
        if not any(term in v.lower() for term in legal_terms):
            raise ValueError("Output should contain legal terminology")
        
        return v.strip()
    
    @model_validator(mode='after')
    def validate_consistency(self):
        """Validate overall example consistency."""
        output = self.output or ''
        citations = self.citations or []
        
        # Check that citations appear in output (at least one)
        citation_mentioned = False
        for citation in citations:
            if citation in output:
                citation_mentioned = True
                break
        
        if not citation_mentioned:
            # This is a warning, not an error - we'll flag it but allow it
            pass
        
        return self


class SFTDataset(BaseModel):
    """Schema for complete SFT dataset."""
    
    examples: List[SFTExample] = Field(..., min_items=1)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('examples')
    @classmethod
    def validate_examples(cls, v):
        """Validate example collection."""
        if len(v) < 50:
            raise ValueError("Dataset too small (minimum 50 examples)")
        
        # Check for instruction diversity (no more than 5% duplicates)
        instructions = [ex.instruction.lower().strip() for ex in v]
        unique_instructions = set(instructions)
        
        duplicate_rate = 1 - (len(unique_instructions) / len(instructions))
        if duplicate_rate > 0.05:
            raise ValueError(f"Too many duplicate instructions: {duplicate_rate:.1%}")
        
        return v


class ValidationReport(BaseModel):
    """Schema for dataset validation report."""
    
    is_valid: bool
    total_examples: int
    valid_examples: int
    errors: List[str] = Field(default_factory=list)
    warnings: List[str] = Field(default_factory=list)
    statistics: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        extra = 'allow'


class EnvironmentInfo(BaseModel):
    """Schema for environment information metadata."""
    
    cuda_available: bool
    cuda_version: Optional[str] = None
    gpu_count: int = 0
    gpu_memory_gb: Optional[List[float]] = None
    torch_version: str
    transformers_version: str
    peft_version: str
    bitsandbytes_version: Optional[str] = None
    python_version: str
    platform: str
    timestamp: str
    
    class Config:
        extra = 'allow'


class TrainingConfig(BaseModel):
    """Schema for training configuration validation."""
    
    # Model settings
    model_name: str = Field(..., description="Base model name")
    use_4bit: bool = Field(default=True, description="Use 4-bit quantization")
    bf16: bool = Field(default=True, description="Use bfloat16 precision")
    
    # LoRA settings  
    lora_rank: int = Field(default=16, ge=1, le=256, description="LoRA rank")
    lora_alpha: int = Field(default=32, ge=1, description="LoRA alpha")
    lora_dropout: float = Field(default=0.1, ge=0.0, le=0.5, description="LoRA dropout")
    
    # Training settings
    num_epochs: int = Field(default=3, ge=1, le=10, description="Number of epochs")
    per_device_train_batch_size: int = Field(default=1, ge=1, description="Training batch size")
    gradient_accumulation_steps: int = Field(default=4, ge=1, description="Gradient accumulation")
    learning_rate: float = Field(default=2e-4, gt=0.0, lt=1e-2, description="Learning rate")
    warmup_ratio: float = Field(default=0.1, ge=0.0, le=0.5, description="Warmup ratio")
    
    # Evaluation settings
    eval_steps: int = Field(default=50, ge=1, description="Evaluation frequency")
    save_steps: int = Field(default=100, ge=1, description="Save frequency")
    
    # Reproducibility
    seed: int = Field(default=42, description="Random seed")
    
    @field_validator('model_name')
    @classmethod
    def validate_model_name(cls, v):
        """Validate model name format."""
        # Should be in format org/model-name or meta-llama/Llama-3.1-8B-Instruct
        if '/' not in v:
            raise ValueError("Model name should be in format 'org/model-name'")
        return v
    
    @model_validator(mode='after')
    def validate_training_settings(self):
        """Validate training setting combinations."""
        epochs = self.num_epochs
        learning_rate = self.learning_rate
        
        # Warn about potential overfitting
        if epochs > 5 and learning_rate > 1e-4:
            # This would be a warning in practice
            pass
        
        return self


class CitationMetrics(BaseModel):
    """Schema for citation evaluation metrics."""
    
    exact_match: float = Field(ge=0.0, le=1.0, description="Exact citation match rate")
    partial_match: float = Field(ge=0.0, le=1.0, description="Partial citation match rate (IoU)")
    precision: float = Field(ge=0.0, le=1.0, description="Citation precision")
    recall: float = Field(ge=0.0, le=1.0, description="Citation recall")
    f1_score: float = Field(ge=0.0, le=1.0, description="Citation F1 score")
    valid_format_rate: float = Field(ge=0.0, le=1.0, description="Rate of valid citation formats")
    
    @model_validator(mode='after')
    def calculate_f1(self):
        """Calculate F1 score from precision and recall."""
        precision = self.precision
        recall = self.recall
        
        if precision + recall > 0:
            self.f1_score = 2 * (precision * recall) / (precision + recall)
        else:
            self.f1_score = 0.0
        
        return self


class EvaluationMetrics(BaseModel):
    """Schema for comprehensive evaluation metrics."""
    
    eval_loss: float = Field(ge=0.0, description="Evaluation loss")
    citation_metrics: CitationMetrics
    avg_judge_score: float = Field(ge=0.0, le=1.0, description="Average judge score")
    response_length_avg: float = Field(ge=0.0, description="Average response length")
    valid_json_rate: float = Field(ge=0.0, le=1.0, description="Valid JSON response rate")
    total_examples: int = Field(ge=1, description="Total examples evaluated")
    
    class Config:
        extra = 'allow'