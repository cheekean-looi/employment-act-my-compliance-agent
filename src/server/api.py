#!/usr/bin/env python3
"""
FastAPI application for Employment Act Malaysia compliance agent.
Production-grade API with caching, observability, and error handling.
"""

import os
import time
import json
import asyncio
from typing import Dict, Any, List, Optional
from datetime import datetime
from pathlib import Path

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import Response
from starlette.responses import StreamingResponse
from fastapi.exception_handlers import http_exception_handler
from contextlib import asynccontextmanager
import uvicorn

# Optional rate limiting
try:
    from slowapi import Limiter
    from slowapi.util import get_remote_address
    from slowapi.errors import RateLimitExceeded
except ImportError:
    Limiter = None
    RateLimitExceeded = Exception

# Import schemas and dependencies
from src.server.schemas import (
    QueryRequest, AnswerResponse, SeveranceRequest, SeveranceResponse,
    HealthResponse, SectionRequest, SectionResponse, ErrorResponse,
    Citation, TokenUsage, GuardrailsReport
)
from src.server.deps import (
    get_rag_pipeline, get_guardrails_engine, get_severance_calculator,
    get_vllm_client, get_cache, get_cache_key_builder, get_config_hash,
    get_model_router, lifespan_manager, get_dependencies, validate_config
)
from src.server.telemetry import (
    get_logger, get_metrics, get_otel_manager, TelemetryMiddleware, create_telemetry_context,
    track_function
)
from src.server.security import (
    SecurityMiddleware, SecurityConfig, require_api_key, require_permission,
    api_key_manager, APIKey
)
from src.server.vllm_client import GenerationConfig, VLLMClient
from src.server.cache import CacheKeyBuilder

# Global startup time for uptime calculation
startup_time = time.time()


@asynccontextmanager
async def app_lifespan(app: FastAPI):
    """Application lifespan manager."""
    async with lifespan_manager(app):
        yield


# Create FastAPI app
app = FastAPI(
    title="Employment Act Malaysia Compliance Agent",
    description="Production-grade API for employment law guidance with RAG and guardrails",
    version="1.0.0",
    lifespan=app_lifespan
)

# Add telemetry middleware
logger = get_logger()
metrics = get_metrics()
otel_manager = get_otel_manager()
app.add_middleware(TelemetryMiddleware, logger=logger, metrics=metrics)

# Add security middleware
security_config = SecurityConfig()
app.add_middleware(SecurityMiddleware, config=security_config)

# Add OpenTelemetry instrumentation
otel_manager.instrument_fastapi(app)

# CORS middleware
allowed_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:8501").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Set up rate limiter if available
if Limiter is not None:
    limiter = Limiter(key_func=get_remote_address)
    app.state.limiter = limiter
    
    @app.exception_handler(RateLimitExceeded)
    async def rate_limit_handler(request: Request, exc: RateLimitExceeded):
        from fastapi.responses import JSONResponse
        from datetime import datetime
        
        # Extract retry-after from exception if available
        retry_after = getattr(exc, 'retry_after', 60)
        
        error_response = ErrorResponse(
            error="RateLimitExceeded",
            message=f"Rate limit exceeded. Try again later.",
            request_id=getattr(request.state, 'request_id', None),
            timestamp=datetime.utcnow().isoformat()
        )
        
        headers = {"Retry-After": str(retry_after)}
        
        return JSONResponse(
            status_code=429,
            content=error_response.dict(),
            headers=headers
        )
else:
    class _NoLimiter:
        def limit(self, *args, **kwargs):
            def decorator(f):
                return f
            return decorator
    limiter = _NoLimiter()


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler with telemetry."""
    try:
        telemetry = create_telemetry_context(request)
        logger.log_error(exc, request_id=getattr(request.state, 'request_id', None))
    except Exception:
        # Fallback if telemetry fails
        print(f"Exception in global handler: {exc}")
    
    # Return structured error response as JSONResponse (not HTTPException)
    error_response = ErrorResponse(
        error=type(exc).__name__,
        message=str(exc),
        request_id=getattr(request.state, 'request_id', None),
        timestamp=datetime.utcnow().isoformat()
    )
    
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content=error_response.dict()
    )


@app.post("/answer", response_model=AnswerResponse)
@limiter.limit("10/minute")
async def answer_query(
    request: Request,
    query_request: QueryRequest,
    rag_pipeline=Depends(get_rag_pipeline),
    guardrails_engine=Depends(get_guardrails_engine),
    vllm_client=Depends(get_vllm_client),
    model_router=Depends(get_model_router),
    cache=Depends(get_cache),
    cache_key_builder=Depends(get_cache_key_builder),
    config_hash=Depends(get_config_hash),
    api_key: Optional[APIKey] = Depends(require_api_key)
) -> AnswerResponse:
    """
    Process user query with RAG pipeline and return structured answer.
    Includes caching, guardrails, and full observability.
    """
    start_time = time.time()
    telemetry = create_telemetry_context(request)
    
    try:
        # Step 1: Quick cache check with query-only key to avoid retrieval on repeats
        guardrails_version = guardrails_engine.config_version
        query_only_key = cache_key_builder.build_query_key(
            query_request.query,
            [],
            config_hash,
            guardrails_version
        )
        cached_response = await cache.get(query_only_key)
        if cached_response:
            # telemetry.log_cache_hit(query_only_key, hit=True)
            
            # Add cache metadata
            cached_response["cache_hit"] = True
            cached_response["latency_ms"] = (time.time() - start_time) * 1000
            
            return AnswerResponse(**cached_response)
        
        # telemetry.log_cache_hit(query_only_key, hit=False)
        
        # Step 2: Retrieve to build full cache key with context
        # Offload heavy retrieval to worker thread to avoid blocking event loop
        retrieval_result = await asyncio.to_thread(
            rag_pipeline.retrieve_and_evaluate, 
            query_request.query
        )
        context_ids = [
            chunk.get('doc_id', f"chunk_{i}") 
            for i, chunk in enumerate(retrieval_result['filtered_chunks'])
        ]
        
        cache_key = cache_key_builder.build_query_key(
            query_request.query,
            context_ids,
            config_hash,
            guardrails_version
        )
        
        # Step 3: Check for early refusal
        if retrieval_result['guardrails_result'] and retrieval_result['guardrails_result'].should_refuse:
            # Handle refusal case
            refusal_response = rag_pipeline.handle_refusal(
                query_request.query, 
                retrieval_result['guardrails_result']
            )
            
            # Convert to API response format
            response_data = _build_answer_response(
                refusal_response,
                start_time,
                cache_hit=False,
                was_cached=False
            )
            
            # Cache the refusal (shorter TTL)
            await cache.set(cache_key, response_data, ttl=300)  # 5 minutes
            await cache.set(query_only_key, response_data, ttl=300)
            
            return AnswerResponse(**response_data)
        
        # Step 4: Check for insufficient context
        if not retrieval_result['filtered_chunks']:
            insufficient_response = rag_pipeline.handle_insufficient_context(
                query_request.query,
                retrieval_result['retrieved_chunks']
            )
            
            response_data = _build_answer_response(
                insufficient_response,
                start_time,
                cache_hit=False,
                was_cached=False
            )
            
            # Cache with medium TTL
            await cache.set(cache_key, response_data, ttl=900)  # 15 minutes
            await cache.set(query_only_key, response_data, ttl=900)
            
            return AnswerResponse(**response_data)
        
        # Step 4.5: Model routing decision - choose between 8B and 70B based on escalation signals
        # Extract guardrails flags correctly from object (not dict)
        gr = retrieval_result.get('guardrails_result')
        guardrails_flags = getattr(gr, 'safety_flags', []) if gr else []
        user_escalation = query_request.query.lower().find('complex') != -1  # Simple user signal detection
        
        routing_decision = model_router.should_escalate(
            query_request.query,
            retrieval_result['filtered_chunks'],
            guardrails_flags,
            user_escalation
        )
        
        # Get model configuration based on routing decision
        model_config = model_router.get_model_config(routing_decision.model_tier)
        
        # Step 5: Generate LLM response using JSON pipeline with routed model
        # Track token usage across generation calls
        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_tokens = 0
        used_model = None
        
        async def llm_generate_func(prompt: str, json_constrained: bool = False):
            """Enhanced async wrapper that applies model routing configuration."""
            nonlocal total_prompt_tokens, total_completion_tokens, total_tokens, used_model
            
            # Create generation config with routing and JSON constraints
            routed_config = GenerationConfig(
                max_tokens=model_config.get("max_tokens", 512),
                temperature=model_config.get("temperature", 0.0),
                top_p=model_config.get("top_p", 0.9),
                top_k=model_config.get("top_k", 6),
                json_only=json_constrained,
                response_format={"type": "json_object"} if json_constrained else None
            )
            
            # Generate response with routed model and return text
            routed_model_name = model_config.get("model_name")
            generation_result = await vllm_client.generate(prompt, routed_config, model_name=routed_model_name)
            
            # Accumulate token usage
            total_prompt_tokens += generation_result.prompt_tokens
            total_completion_tokens += generation_result.completion_tokens
            total_tokens += generation_result.total_tokens
            used_model = generation_result.model
            
            return generation_result.text
        
        llm_start = time.time()
        
        # Use the optimized JSON pipeline that reuses pre-computed retrieval_result (avoids double retrieval)
        response_data = await rag_pipeline.process_query_with_json_from_retrieval(
            query_request.query,
            retrieval_result,
            llm_generate_func=llm_generate_func
        )
        llm_latency = (time.time() - llm_start) * 1000
        
        # Add token usage information to response
        token_usage = {
            "prompt_tokens": total_prompt_tokens,
            "completion_tokens": total_completion_tokens,
            "total_tokens": total_tokens,
            "model": used_model or model_config.get("model_name", "unknown")
        }
        response_data["token_usage"] = token_usage
        
        # Extract token usage and model info
        model_name = token_usage.get("model", "unknown")
        cfg = get_dependencies().config
        
        # Prepare routing metadata for telemetry
        routing_metadata = {
            "model_tier": routing_decision.model_tier.value,
            "escalation_signals": routing_decision.escalation_signals,
            "confidence_threshold": routing_decision.confidence_threshold,
            "reasoning": routing_decision.reasoning
        }
        
        # Log routing decision for audit
        telemetry.log_routing_decision(
            model_tier=routing_decision.model_tier.value,
            escalation_signals=routing_decision.escalation_signals,
            confidence_threshold=routing_decision.confidence_threshold,
            reasoning=routing_decision.reasoning,
            original_model=cfg.get("default_model", "llama-3.1-8b-instruct"),
            routed_model=model_name
        )
        
        # Log LLM call with structured data including routing metadata
        telemetry.log_llm_generation(
            model=model_name,
            prompt_tokens=token_usage.get("prompt_tokens", 0),
            completion_tokens=token_usage.get("completion_tokens", 0),
            latency_ms=llm_latency,
            routing_metadata=routing_metadata
        )
        
        # Step 6: Response already validated and auto-repaired by JSON pipeline
        # The process_query_with_json() returns a structured dict with guaranteed citations
        
        # Step 7: Build final API response with enhanced metadata including routing
        final_response = _build_json_answer_response(
            response_data,
            retrieval_result,
            token_usage,
            start_time,
            llm_latency,
            routing_decision,
            cache_hit=False
        )
        
        # Step 8: Cache successful response
        await cache.set(cache_key, final_response, ttl=1800)  # 30 minutes
        await cache.set(query_only_key, final_response, ttl=1800)
        
        return AnswerResponse(**final_response)
        
    except Exception as e:
        logger.log_error(e, request_id=getattr(request.state, 'request_id', None))
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process query: {str(e)}"
        )


@app.post("/tool/severance", response_model=SeveranceResponse)
async def calculate_severance(
    request: Request,
    severance_request: SeveranceRequest,
    calculator=Depends(get_severance_calculator),
    cache=Depends(get_cache),
    cache_key_builder=Depends(get_cache_key_builder)
) -> SeveranceResponse:
    """Calculate severance pay using Employment Act formulas."""
    start_time = time.time()
    telemetry = create_telemetry_context(request)
    
    try:
        # Build cache key
        cache_key = cache_key_builder.build_severance_key(
            severance_request.monthly_wage,
            severance_request.years_of_service,
            severance_request.termination_reason,
            severance_request.annual_leave_days
        )
        
        # Check cache
        cached_result = await cache.get(cache_key)
        if cached_result:
            telemetry.log_cache_hit(cache_key, hit=True)
            cached_result["latency_ms"] = (time.time() - start_time) * 1000
            return SeveranceResponse(**cached_result)
        
        telemetry.log_cache_hit(cache_key, hit=False)
        
        # Calculate severance
        result = calculator.calculate_severance(
            monthly_wage=severance_request.monthly_wage,
            years_of_service=severance_request.years_of_service,
            termination_reason=severance_request.termination_reason
        )
        
        # Calculate annual leave compensation if provided
        annual_leave_compensation = None
        if severance_request.annual_leave_days:
            daily_wage = severance_request.monthly_wage / 30
            annual_leave_compensation = daily_wage * severance_request.annual_leave_days
        
        # Build response
        total_compensation = result["severance_pay"]
        if result.get("notice_pay"):
            total_compensation += result["notice_pay"]
        if annual_leave_compensation:
            total_compensation += annual_leave_compensation
        
        response_data = {
            "severance_pay": result["severance_pay"],
            "notice_pay": result.get("notice_pay"),
            "annual_leave_compensation": annual_leave_compensation,
            "total_compensation": total_compensation,
            "calculation_breakdown": result,
            "employment_act_references": result.get("legal_references", []),
            "latency_ms": (time.time() - start_time) * 1000
        }
        
        # Cache result
        await cache.set(cache_key, response_data, ttl=3600)  # 1 hour (calculations don't change)
        
        return SeveranceResponse(**response_data)
        
    except Exception as e:
        logger.log_error(e, request_id=getattr(request.state, 'request_id', None))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Severance calculation failed: {str(e)}"
        )


@app.get("/section/{section_id}", response_model=SectionResponse)
async def get_section(
    section_id: str,
    rag_pipeline=Depends(get_rag_pipeline),
    cache=Depends(get_cache),
    cache_key_builder=Depends(get_cache_key_builder)
) -> SectionResponse:
    """Retrieve Employment Act section by ID."""
    try:
        # Build cache key
        cache_key = cache_key_builder.build_section_key(section_id)
        
        # Check cache
        cached_section = await cache.get(cache_key)
        if cached_section:
            return SectionResponse(**cached_section)
        
        # Get section from retriever store
        retriever = rag_pipeline.retriever
        
        # Search for section in stored chunks
        if hasattr(retriever, 'store') and retriever.store:
            # Look through stored documents for section
            for doc_id, doc_data in retriever.store.items():
                if section_id.lower() in doc_data.get('text', '').lower():
                    section_data = {
                        "section_id": section_id,
                        "title": f"Section {section_id}",
                        "full_text": doc_data.get('text', ''),
                        "related_sections": []  # Could be enhanced
                    }
                    
                    # Cache section
                    await cache.set(cache_key, section_data, ttl=7200)  # 2 hours
                    
                    return SectionResponse(**section_data)
        
        # Section not found
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Section {section_id} not found"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.log_error(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve section"
        )


@app.get("/health", response_model=HealthResponse)
async def health_check() -> HealthResponse:
    """Comprehensive health check endpoint."""
    try:
        deps = get_dependencies()
        health = await deps.health_check()
        
        # Calculate uptime
        uptime_s = time.time() - startup_time
        
        # Determine overall status
        critical_checks = ["rag_ready", "indices_ok"]
        status_ok = all(health.get(check, False) for check in critical_checks)
        overall_status = "ok" if status_ok else "degraded"
        
        # Get git version (or fallback)
        version = os.getenv("GIT_SHA", "development")
        
        return HealthResponse(
            status=overall_status,
            version=version,
            rag_ready=health.get("rag_ready", False),
            vllm_ready=health.get("vllm_ready", False),
            indices_ok=health.get("indices_ok", False),
            uptime_s=uptime_s,
            cache_status=health.get("cache_status", "unknown"),
            guardrails_version=health.get("guardrails_version", "unknown")
        )
        
    except Exception as e:
        logger.log_error(e)
        return HealthResponse(
            status="error",
            version="unknown",
            rag_ready=False,
            vllm_ready=False,
            indices_ok=False,
            uptime_s=time.time() - startup_time,
            cache_status="error",
            guardrails_version="error"
        )


@app.get("/metrics")
async def get_metrics() -> Response:
    """Prometheus metrics endpoint."""
    try:
        # Import Prometheus client for proper content type
        from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
        
        # Get metrics in Prometheus format
        metrics_data = generate_latest()
        
        # Return with proper Prometheus content type
        return Response(
            content=metrics_data,
            media_type=CONTENT_TYPE_LATEST
        )
    except ImportError:
        # Fallback if prometheus_client not available
        try:
            metrics_data = metrics.generate_metrics()
            return Response(
                content=metrics_data,
                media_type="text/plain; version=0.0.4; charset=utf-8"
            )
        except Exception as e:
            logger.log_error(e)
            return Response(
                content="# Metrics unavailable\n",
                media_type="text/plain; version=0.0.4; charset=utf-8"
            )
    except Exception as e:
        logger.log_error(e)
        return Response(
            content="# Metrics unavailable\n",
            media_type="text/plain; version=0.0.4; charset=utf-8"
        )


@app.get("/api-keys/stats")
async def get_api_key_stats(
    api_key: Optional[APIKey] = Depends(require_api_key)
) -> Dict[str, Any]:
    """Get API key usage statistics (admin only)."""
    if not api_key:
        # If API keys are disabled, return basic info
        return {
            "api_keys_enabled": False,
            "total_keys": 0,
            "message": "API key authentication is disabled"
        }
    
    try:
        stats = api_key_manager.get_stats()
        stats["api_keys_enabled"] = True
        return stats
    except Exception as e:
        logger.log_error(e)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve API key statistics"
        )


@app.get("/config/validate")
async def validate_configuration() -> Dict[str, Any]:
    """Validate application configuration."""
    try:
        validation = validate_config()
        return validation
    except Exception as e:
        logger.log_error(e)
        return {"valid": False, "errors": [str(e)], "warnings": []}


# Helper functions
def _build_answer_response(response_dict: Dict[str, Any], 
                          start_time: float,
                          cache_hit: bool = False,
                          was_cached: bool = False) -> Dict[str, Any]:
    """Build standardized answer response."""
    latency_ms = (time.time() - start_time) * 1000
    
    # Default token usage if not provided
    token_usage = {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0
    }
    
    return {
        "answer": response_dict.get("answer", ""),
        "citations": response_dict.get("citations", []),
        "confidence": response_dict.get("confidence", 0.5),
        "should_escalate": response_dict.get("should_escalate", False),
        "safety_flags": response_dict.get("safety_flags", []),
        "guardrails_report": None,  # Simplified for now
        "latency_ms": latency_ms,
        "token_usage": token_usage,
        "cache_hit": cache_hit
    }


def _build_answer_response_with_metadata(response_dict: Dict[str, Any],
                                        retrieval_result: Dict[str, Any],
                                        generation_result,
                                        start_time: float,
                                        cache_hit: bool = False) -> Dict[str, Any]:
    """Build answer response with full metadata."""
    latency_ms = (time.time() - start_time) * 1000
    
    # Build guardrails report
    guardrails_report = None
    if retrieval_result.get('guardrails_result') and hasattr(retrieval_result['guardrails_result'], 'report'):
        gr = retrieval_result['guardrails_result'].report
        guardrails_report = {
            "timestamp": gr.timestamp,
            "decision": gr.decision,
            "confidence": gr.confidence,
            "processing_time_ms": gr.processing_time_ms,
            "input_flags": gr.input_flags,
            "output_flags": getattr(gr, 'output_flags', []),
            "triggered_rules": gr.triggered_rules,
            "config_version": gr.config_version
        }
    
    # Build token usage
    token_usage = {
        "prompt_tokens": generation_result.prompt_tokens,
        "completion_tokens": generation_result.completion_tokens,
        "total_tokens": generation_result.total_tokens
    }
    
    return {
        "answer": response_dict.get("answer", ""),
        "citations": response_dict.get("citations", []),
        "confidence": response_dict.get("confidence", 0.5),
        "should_escalate": response_dict.get("should_escalate", False),
        "safety_flags": response_dict.get("safety_flags", []),
        "guardrails_report": guardrails_report,
        "latency_ms": latency_ms,
        "token_usage": token_usage,
        "cache_hit": cache_hit
    }


def _build_json_answer_response(response_data: Dict[str, Any],
                                retrieval_result: Dict[str, Any],
                                token_usage: Dict[str, Any],
                                start_time: float,
                                llm_latency: float,
                                routing_decision=None,
                                cache_hit: bool = False) -> Dict[str, Any]:
    """Build answer response from JSON pipeline output with metadata."""
    total_latency_ms = (time.time() - start_time) * 1000
    
    # Build guardrails report from both input and output guardrails
    guardrails_report = None
    if retrieval_result.get('guardrails_result') and hasattr(retrieval_result['guardrails_result'], 'report'):
        gr = retrieval_result['guardrails_result'].report
        # Merge with any output guardrails from JSON response
        output_guardrails = response_data.get("guardrails_report", {})
        
        guardrails_report = {
            "timestamp": output_guardrails.get("timestamp", gr.timestamp),
            "decision": output_guardrails.get("decision", gr.decision),
            "confidence": output_guardrails.get("confidence", gr.confidence),
            "processing_time_ms": gr.processing_time_ms,
            "input_flags": gr.input_flags,
            "output_flags": output_guardrails.get("output_flags", []),
            "triggered_rules": gr.triggered_rules,
            "config_version": gr.config_version
        }
    
    # Use token usage from JSON response or fallback to provided
    final_token_usage = response_data.get("token_usage", token_usage)
    
    # Add routing metadata if available
    routing_metadata = {}
    if routing_decision:
        routing_metadata = {
            "model_tier": routing_decision.model_tier.value,  # Use .value for enum serialization
            "escalation_signals": routing_decision.escalation_signals,
            "confidence_threshold": routing_decision.confidence_threshold,  # Fix: confidence_score -> confidence_threshold
            "reasoning": routing_decision.reasoning
        }
    
    response_dict = {
        "answer": response_data.get("answer", ""),
        "citations": response_data.get("citations", []),
        "confidence": response_data.get("confidence", 0.5),
        "should_escalate": response_data.get("should_escalate", False),
        "safety_flags": response_data.get("safety_flags", []),
        "guardrails_report": guardrails_report,
        "latency_ms": total_latency_ms,
        "token_usage": final_token_usage,
        "cache_hit": cache_hit
    }
    
    # Add routing metadata if present
    if routing_metadata:
        response_dict["routing_metadata"] = routing_metadata
    
    return response_dict


if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "src.server.api:app",
        host="0.0.0.0",
        port=int(os.getenv("API_PORT", "8001")),
        reload=True,
        log_level="info"
    )

# Streaming endpoint for incremental tokens (SSE)
@app.post("/answer/stream")
async def answer_query_stream(
    request: Request,
    query_request: QueryRequest,
    rag_pipeline=Depends(get_rag_pipeline),
    vllm_client=Depends(get_vllm_client),
    model_router=Depends(get_model_router),
):
    """Stream tokens for an answer via Server-Sent Events."""
    # Offload heavy retrieval to worker thread to avoid blocking event loop
    retrieval_result = await asyncio.to_thread(
        rag_pipeline.retrieve_and_evaluate, 
        query_request.query
    )
    if retrieval_result['guardrails_result'] and retrieval_result['guardrails_result'].should_refuse:
        async def gen_refusal():
            yield "data: {\"chunk_type\":\"metadata\",\"content\":\"refused\"}\n\n"
            yield "data: [DONE]\n\n"
        return StreamingResponse(gen_refusal(), media_type="text/event-stream")

    # Apply model routing for parity with /answer endpoint
    gr = retrieval_result.get('guardrails_result')
    guardrails_flags = getattr(gr, 'safety_flags', []) if gr else []
    user_escalation = query_request.query.lower().find('complex') != -1
    
    routing_decision = model_router.should_escalate(
        query_request.query,
        retrieval_result['filtered_chunks'],
        guardrails_flags,
        user_escalation
    )
    
    # Get model configuration based on routing decision
    model_config = model_router.get_model_config(routing_decision.model_tier)

    prompt = rag_pipeline.generate_prompt(query_request.query, retrieval_result['filtered_chunks'])
    cfg = get_dependencies().config
    generation_config = GenerationConfig(
        max_tokens=(query_request.max_tokens or cfg.get("max_tokens", 512)),
        temperature=(query_request.temperature if query_request.temperature is not None else cfg.get("temperature", 0.0)),
        json_only=False
    )

    async def token_stream():
        start_time = time.time()
        token_count = 0
        
        try:
            # Emit initial metadata chunk with routing information (clean JSON format)
            routing_metadata = {
                "chunk_type": "metadata",
                "model_tier": routing_decision.model_tier.value,
                "effective_model": model_config.get("model_name"),
                "escalation_signals": routing_decision.escalation_signals,
                "confidence_threshold": routing_decision.confidence_threshold,
                "reasoning": routing_decision.reasoning
            }
            yield f"data: {json.dumps(routing_metadata)}\n\n"
            
            # Use routed model configuration
            effective_model_name = model_config.get("model_name")
            async for token in vllm_client.generate_stream(
                prompt, 
                generation_config,
                model_name=effective_model_name
            ):
                token_count += 1
                token_chunk = {
                    "chunk_type": "token",
                    "content": token
                }
                yield f"data: {json.dumps(token_chunk)}\n\n"
                
        except Exception as e:
            error_chunk = {
                "chunk_type": "error",
                "content": str(e).replace("\n", " "),
                "status": "error"
            }
            yield f"data: {json.dumps(error_chunk)}\n\n"
            
            # Final error status metadata
            error_status = {
                "chunk_type": "status",
                "status": "error",
                "latency_ms": round((time.time() - start_time) * 1000, 2)
            }
            yield f"data: {json.dumps(error_status)}\n\n"
        finally:
            # Emit final metadata chunk with usage statistics (if no error occurred)
            if 'error_chunk' not in locals():
                end_time = time.time()
                latency_ms = (end_time - start_time) * 1000
                
                # Estimate token usage (rough approximation)
                estimated_prompt_tokens = len(prompt.split()) * 1.3
                estimated_completion_tokens = token_count
                
                usage_metadata = {
                    "chunk_type": "usage",
                    "latency_ms": round(latency_ms, 2),
                    "estimated_prompt_tokens": int(estimated_prompt_tokens),
                    "estimated_completion_tokens": estimated_completion_tokens,
                    "estimated_total_tokens": int(estimated_prompt_tokens + estimated_completion_tokens),
                    "model": effective_model_name
                }
                yield f"data: {json.dumps(usage_metadata)}\n\n"
                
                # Final success status metadata
                success_status = {
                    "chunk_type": "status",
                    "status": "completed"
                }
                yield f"data: {json.dumps(success_status)}\n\n"
            
            yield "data: [DONE]\n\n"

    return StreamingResponse(token_stream(), media_type="text/event-stream")
