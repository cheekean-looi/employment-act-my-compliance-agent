#!/usr/bin/env python3
"""
Pydantic schemas for FastAPI endpoints.
Production-grade request/response models with validation.
"""

from typing import List, Dict, Any, Optional, Union
from pydantic import BaseModel, Field, validator
from datetime import datetime


class QueryRequest(BaseModel):
    """Request for /answer endpoint."""
    query: str = Field(..., min_length=1, max_length=2000, description="User query")
    max_tokens: Optional[int] = Field(512, ge=1, le=2048, description="Maximum response tokens")
    temperature: Optional[float] = Field(0.1, ge=0.0, le=2.0, description="Generation temperature")
    top_k: Optional[int] = Field(8, ge=1, le=20, description="Retrieval top-k")
    
    @validator('query')
    def query_not_empty(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()


class Citation(BaseModel):
    """Citation with section and snippet."""
    section_id: str = Field(..., description="Employment Act section identifier")
    snippet: str = Field(..., description="Relevant text snippet")


class TokenUsage(BaseModel):
    """Token usage statistics."""
    prompt_tokens: int = Field(..., ge=0)
    completion_tokens: int = Field(..., ge=0)
    total_tokens: int = Field(..., ge=0)


class GuardrailsReport(BaseModel):
    """Simplified guardrails report for API response."""
    timestamp: str
    decision: str  # "allowed" | "refused" | "escalated"
    confidence: float
    processing_time_ms: float
    input_flags: List[str]
    output_flags: List[str]
    triggered_rules: List[str]
    config_version: str


class AnswerResponse(BaseModel):
    """Response from /answer endpoint."""
    answer: str = Field(..., description="Generated answer")
    citations: List[Citation] = Field(default=[], description="Supporting citations")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Response confidence")
    should_escalate: bool = Field(default=False, description="Whether to escalate to human")
    safety_flags: List[str] = Field(default=[], description="Safety flags raised")
    guardrails_report: Optional[GuardrailsReport] = Field(None, description="Detailed guardrails report")
    latency_ms: float = Field(..., ge=0, description="Total request latency")
    token_usage: TokenUsage = Field(..., description="Token usage statistics")
    cache_hit: bool = Field(default=False, description="Whether response was cached")
    routing_metadata: Optional[Dict[str, Any]] = Field(None, description="Model routing decision metadata")


class SeveranceRequest(BaseModel):
    """Request for severance calculation tool."""
    monthly_wage: float = Field(..., gt=0, le=50000, description="Monthly wage in MYR")
    years_of_service: float = Field(..., ge=0, le=60, description="Years of service")
    termination_reason: str = Field(..., description="Reason for termination")
    annual_leave_days: Optional[int] = Field(None, ge=0, le=365, description="Unused annual leave days")
    
    @validator('termination_reason')
    def valid_termination_reason(cls, v):
        valid_reasons = [
            'resignation', 'dismissal_with_cause', 'dismissal_without_cause',
            'redundancy', 'retirement', 'contract_expiry', 'mutual_agreement'
        ]
        if v.lower() not in valid_reasons:
            raise ValueError(f'Termination reason must be one of: {", ".join(valid_reasons)}')
        return v.lower()


class SeveranceResponse(BaseModel):
    """Response from severance calculation."""
    severance_pay: float = Field(..., description="Calculated severance pay in MYR")
    notice_pay: Optional[float] = Field(None, description="Notice pay in lieu")
    annual_leave_compensation: Optional[float] = Field(None, description="Unused leave compensation")
    total_compensation: float = Field(..., description="Total compensation amount")
    calculation_breakdown: Dict[str, Any] = Field(..., description="Detailed calculation breakdown")
    employment_act_references: List[str] = Field(default=[], description="Relevant EA sections")
    latency_ms: float = Field(..., ge=0, description="Calculation latency")


class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Overall health status")
    version: str = Field(..., description="Git SHA or version")
    rag_ready: bool = Field(..., description="RAG pipeline ready")
    vllm_ready: bool = Field(..., description="vLLM service ready")
    indices_ok: bool = Field(..., description="Search indices available")
    uptime_s: float = Field(..., ge=0, description="Service uptime in seconds")
    cache_status: str = Field(..., description="Cache layer status")
    guardrails_version: str = Field(..., description="Guardrails config version")


class SectionRequest(BaseModel):
    """Request for section snippet retrieval."""
    section_id: str = Field(..., description="Employment Act section ID")


class SectionResponse(BaseModel):
    """Response with section snippet."""
    section_id: str = Field(..., description="Section identifier")
    title: Optional[str] = Field(None, description="Section title")
    full_text: str = Field(..., description="Complete section text")
    related_sections: List[str] = Field(default=[], description="Related section IDs")


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Human-readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    request_id: Optional[str] = Field(None, description="Request ID for tracking")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())


class StreamingChunk(BaseModel):
    """Streaming response chunk."""
    chunk_type: str = Field(..., description="Type: 'token', 'citation', 'metadata'")
    content: str = Field(..., description="Chunk content")
    sequence: int = Field(..., description="Sequence number")
    is_final: bool = Field(default=False, description="Whether this is the final chunk")


# Cache-related models
class CacheConfig(BaseModel):
    """Cache configuration."""
    enabled: bool = Field(default=True)
    ttl_seconds: int = Field(default=1800)  # 30 minutes
    max_size: int = Field(default=1000)  # In-memory cache size
    redis_url: Optional[str] = Field(None)


# Metrics and telemetry models
class RequestMetrics(BaseModel):
    """Request metrics for observability."""
    request_id: str
    endpoint: str
    method: str
    start_time: datetime
    duration_ms: float
    status_code: int
    cache_hit: bool = False
    tokens_used: Optional[int] = None
    user_id: Optional[str] = None


class PerformanceMetrics(BaseModel):
    """Performance metrics summary."""
    requests_per_minute: float
    avg_latency_ms: float
    p95_latency_ms: float
    cache_hit_rate: float
    error_rate: float
    token_usage_per_minute: float