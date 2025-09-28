#!/usr/bin/env python3
"""
Telemetry and observability setup for Employment Act API.
Includes structured logging, metrics collection, and request tracing.
"""

import os
import time
import uuid
import logging
import json
from typing import Dict, Any, Optional
from datetime import datetime
from contextlib import asynccontextmanager
from functools import wraps

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
import structlog

# Optional Prometheus metrics
try:
    from prometheus_client import Counter, Histogram, Gauge, generate_latest, CONTENT_TYPE_LATEST
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False

# Optional OpenTelemetry support
try:
    from opentelemetry import trace, metrics as otel_metrics
    from opentelemetry.exporter.jaeger.thrift import JaegerExporter
    from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
    from opentelemetry.exporter.prometheus import PrometheusMetricReader
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
    from opentelemetry.instrumentation.httpx import HTTPXClientInstrumentor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.semconv.resource import ResourceAttributes
    from opentelemetry.trace.status import Status, StatusCode
    OTEL_AVAILABLE = True
except ImportError:
    OTEL_AVAILABLE = False


class StructuredLogger:
    """Structured logger with JSON output and request context."""
    
    def __init__(self, service_name: str = "employment-act-api"):
        """Initialize structured logging."""
        self.service_name = service_name
        
        # Configure structlog only for this specific logger, not globally
        self.logger = structlog.get_logger(service=service_name)
        
        # Only configure structlog if not already configured
        if not hasattr(structlog, '_CONFIGURED'):
            structlog.configure(
                processors=[
                    structlog.processors.TimeStamper(fmt="iso"),
                    structlog.processors.add_log_level,
                    structlog.processors.StackInfoRenderer(),
                    structlog.processors.JSONRenderer()
                ],
                wrapper_class=structlog.make_filtering_bound_logger(
                    logging.INFO if os.getenv("LOG_LEVEL", "INFO") == "INFO" else logging.DEBUG
                ),
                logger_factory=structlog.WriteLoggerFactory(),
                cache_logger_on_first_use=True,
            )
            structlog._CONFIGURED = True
    
    def log_request(self, 
                   request_id: str,
                   method: str,
                   path: str,
                   status_code: int,
                   duration_ms: float,
                   user_id: Optional[str] = None,
                   **kwargs):
        """Log request with structured data."""
        # Ensure kwargs don't conflict with event parameter
        safe_kwargs = kwargs.copy()
        safe_kwargs.pop('event', None)  # Remove event if present to avoid conflicts
        
        self.logger.info(
            "request_completed",
            request_id=request_id,
            method=method,
            path=path,
            status_code=status_code,
            duration_ms=duration_ms,
            user_id=user_id,
            **safe_kwargs
        )
    
    def log_cache_event(self, 
                       event: str,
                       cache_key: str,
                       hit: bool,
                       latency_ms: float):
        """Log cache events."""
        self.logger.info(
            "cache_event",
            event=event,
            cache_key=cache_key[:16] + "...",  # Truncate for privacy
            hit=hit,
            latency_ms=latency_ms
        )
    
    def log_llm_call(self,
                    model: str,
                    prompt_tokens: int,
                    completion_tokens: int,
                    latency_ms: float,
                    cache_hit: bool = False,
                    routing_metadata: Optional[Dict[str, Any]] = None):
        """Log LLM generation calls with optional routing metadata."""
        log_data = {
            "model": model,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "latency_ms": latency_ms,
            "cache_hit": cache_hit
        }
        
        # Add routing metadata if provided
        if routing_metadata:
            log_data.update({
                "routing_tier": routing_metadata.get("model_tier"),
                "escalation_signals": routing_metadata.get("escalation_signals", []),
                "confidence_threshold": routing_metadata.get("confidence_threshold"),
                "routing_reasoning": routing_metadata.get("reasoning")
            })
        
        self.logger.info("llm_generation", **log_data)
    
    def log_routing_decision(self,
                           model_tier: str,
                           escalation_signals: list,
                           confidence_threshold: float,
                           reasoning: str,
                           original_model: str,
                           routed_model: str):
        """Log model routing decisions for audit and analysis."""
        self.logger.info(
            "model_routing",
            model_tier=model_tier,
            escalation_signals=escalation_signals,
            confidence_threshold=confidence_threshold,
            reasoning=reasoning,
            original_model=original_model,
            routed_model=routed_model
        )
    
    def log_guardrails_event(self,
                           decision: str,
                           triggered_rules: list,
                           processing_time_ms: float,
                           confidence: float):
        """Log guardrails decisions."""
        self.logger.info(
            "guardrails_decision",
            decision=decision,
            rules_triggered=len(triggered_rules),
            processing_time_ms=processing_time_ms,
            confidence=confidence
        )
    
    def log_error(self, 
                 error: Exception,
                 request_id: Optional[str] = None,
                 context: Optional[Dict[str, Any]] = None):
        """Log errors with context."""
        # Ensure context doesn't conflict with event parameter
        safe_context = (context or {}).copy()
        safe_context.pop('event', None)  # Remove event if present to avoid conflicts
        
        self.logger.error(
            "error_occurred",
            error_type=type(error).__name__,
            error_message=str(error),
            request_id=request_id,
            context=safe_context
        )


class MetricsCollector:
    """Prometheus metrics collector."""
    
    def __init__(self, enabled: bool = True):
        """Initialize metrics collector."""
        self.enabled = enabled and PROMETHEUS_AVAILABLE
        
        if self.enabled:
            # Request metrics
            self.request_count = Counter(
                'employment_act_requests_total',
                'Total number of requests',
                ['method', 'endpoint', 'status']
            )
            
            self.request_duration = Histogram(
                'employment_act_request_duration_seconds',
                'Request duration in seconds',
                ['method', 'endpoint']
            )
            
            # Cache metrics
            self.cache_operations = Counter(
                'employment_act_cache_operations_total',
                'Cache operations',
                ['operation', 'result']
            )
            
            # LLM metrics
            self.llm_requests = Counter(
                'employment_act_llm_requests_total',
                'LLM requests',
                ['model', 'cached']
            )
            
            self.llm_tokens = Counter(
                'employment_act_llm_tokens_total',
                'LLM tokens used',
                ['model', 'type']  # type: prompt, completion
            )
            
            self.llm_duration = Histogram(
                'employment_act_llm_duration_seconds',
                'LLM request duration',
                ['model']
            )
            
            # Guardrails metrics
            self.guardrails_decisions = Counter(
                'employment_act_guardrails_decisions_total',
                'Guardrails decisions',
                ['decision']  # allowed, refused, escalated
            )
            
            # System metrics
            self.active_requests = Gauge(
                'employment_act_active_requests',
                'Currently active requests'
            )
            
            self.cache_size = Gauge(
                'employment_act_cache_size',
                'Cache size',
                ['tier']  # l1, l2
            )
    
    def record_request(self, method: str, endpoint: str, status: int, duration: float):
        """Record request metrics."""
        if not self.enabled:
            return
        
        self.request_count.labels(
            method=method,
            endpoint=endpoint,
            status=str(status)
        ).inc()
        
        self.request_duration.labels(
            method=method,
            endpoint=endpoint
        ).observe(duration)
    
    def record_cache_operation(self, operation: str, hit: bool):
        """Record cache metrics."""
        if not self.enabled:
            return
        
        result = "hit" if hit else "miss"
        self.cache_operations.labels(
            operation=operation,
            result=result
        ).inc()
    
    def record_llm_call(self, model: str, prompt_tokens: int, completion_tokens: int, 
                       duration: float, cached: bool = False):
        """Record LLM metrics."""
        if not self.enabled:
            return
        
        self.llm_requests.labels(
            model=model,
            cached=str(cached)
        ).inc()
        
        self.llm_tokens.labels(
            model=model,
            type="prompt"
        ).inc(prompt_tokens)
        
        self.llm_tokens.labels(
            model=model,
            type="completion"
        ).inc(completion_tokens)
        
        self.llm_duration.labels(model=model).observe(duration)
    
    def record_guardrails_decision(self, decision: str):
        """Record guardrails decision."""
        if not self.enabled:
            return
        
        self.guardrails_decisions.labels(decision=decision).inc()
    
    def set_active_requests(self, count: int):
        """Set active request count."""
        if self.enabled:
            self.active_requests.set(count)
    
    def set_cache_size(self, tier: str, size: int):
        """Set cache size."""
        if self.enabled:
            self.cache_size.labels(tier=tier).set(size)
    
    def generate_metrics(self) -> str:
        """Generate Prometheus metrics."""
        if self.enabled:
            return generate_latest()
        return ""


class TelemetryMiddleware(BaseHTTPMiddleware):
    """FastAPI middleware for request telemetry."""
    
    def __init__(self, app, logger: StructuredLogger, metrics: MetricsCollector):
        super().__init__(app)
        self.logger = logger
        self.metrics = metrics
        self.active_requests = 0
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Process request with telemetry."""
        # Generate request ID
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        
        # Extract user ID if available (from auth header, etc.)
        user_id = request.headers.get("X-User-ID")
        
        # Track active requests
        self.active_requests += 1
        self.metrics.set_active_requests(self.active_requests)
        
        start_time = time.time()
        
        try:
            # Process request
            response = await call_next(request)
            
            # Calculate duration
            duration = time.time() - start_time
            duration_ms = duration * 1000
            
            # Extract endpoint pattern
            endpoint = self._get_endpoint_pattern(request)
            
            # Log request
            self.logger.log_request(
                request_id=request_id,
                method=request.method,
                path=endpoint,
                status_code=response.status_code,
                duration_ms=duration_ms,
                user_id=user_id
            )
            
            # Record metrics
            self.metrics.record_request(
                method=request.method,
                endpoint=endpoint,
                status=response.status_code,
                duration=duration
            )
            
            # Add request ID to response headers
            response.headers["X-Request-ID"] = request_id
            
            return response
            
        except Exception as e:
            # Log error
            duration = time.time() - start_time
            duration_ms = duration * 1000
            
            self.logger.log_error(e, request_id=request_id, context={
                "method": request.method,
                "path": str(request.url),
                "duration_ms": duration_ms
            })
            
            # Record error metrics
            endpoint = self._get_endpoint_pattern(request)
            self.metrics.record_request(
                method=request.method,
                endpoint=endpoint,
                status=500,
                duration=duration
            )
            
            raise
            
        finally:
            # Update active requests
            self.active_requests -= 1
            self.metrics.set_active_requests(self.active_requests)
    
    def _get_endpoint_pattern(self, request: Request) -> str:
        """Extract endpoint pattern from request."""
        path = request.url.path
        
        # Normalize dynamic paths
        if path.startswith("/section/"):
            return "/section/{section_id}"
        elif path.startswith("/api/v1/"):
            return path
        else:
            return path


class OpenTelemetryManager:
    """OpenTelemetry distributed tracing and metrics manager."""
    
    def __init__(self):
        self.enabled = OTEL_AVAILABLE and self._should_enable()
        self.tracer = None
        self.meter = None
        
        if self.enabled:
            self._setup_tracing()
            self._setup_metrics()
            self._setup_instrumentation()
    
    def _should_enable(self) -> bool:
        """Check if OpenTelemetry should be enabled."""
        return bool(
            os.getenv("JAEGER_ENDPOINT") or 
            os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or
            os.getenv("OTEL_TRACES_EXPORTER")
        )
    
    def _setup_tracing(self):
        """Setup distributed tracing."""
        try:
            # Create resource
            resource = Resource.create({
                ResourceAttributes.SERVICE_NAME: "employment-act-api",
                ResourceAttributes.SERVICE_VERSION: "1.0.0",
                ResourceAttributes.DEPLOYMENT_ENVIRONMENT: os.getenv("ENVIRONMENT", "development"),
                "deployment.id": os.getenv("DEPLOYMENT_ID", "local"),
            })
            
            # Create tracer provider
            provider = TracerProvider(resource=resource)
            
            # Add exporters based on environment
            jaeger_endpoint = os.getenv("JAEGER_ENDPOINT")
            if jaeger_endpoint:
                parts = jaeger_endpoint.replace("http://", "").split(":")
                host = parts[0]
                port = int(parts[1]) if len(parts) > 1 else 14268
                
                jaeger_exporter = JaegerExporter(
                    agent_host_name=host,
                    agent_port=port
                )
                provider.add_span_processor(BatchSpanProcessor(jaeger_exporter))
            
            otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
            if otlp_endpoint:
                otlp_exporter = OTLPSpanExporter(endpoint=otlp_endpoint)
                provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
            
            # Set global tracer provider
            trace.set_tracer_provider(provider)
            self.tracer = trace.get_tracer("employment-act-api")
            
        except Exception as e:
            print(f"Failed to setup OpenTelemetry tracing: {e}")
            self.enabled = False
    
    def _setup_metrics(self):
        """Setup OpenTelemetry metrics."""
        try:
            if not self.enabled:
                return
            
            # Create resource
            resource = Resource.create({
                ResourceAttributes.SERVICE_NAME: "employment-act-api",
                ResourceAttributes.SERVICE_VERSION: "1.0.0",
            })
            
            # Create meter provider
            provider = MeterProvider(resource=resource)
            otel_metrics.set_meter_provider(provider)
            self.meter = otel_metrics.get_meter("employment-act-api")
            
        except Exception as e:
            print(f"Failed to setup OpenTelemetry metrics: {e}")
    
    def _setup_instrumentation(self):
        """Setup automatic instrumentation."""
        try:
            if not self.enabled:
                return
            
            # HTTP client instrumentation
            HTTPXClientInstrumentor().instrument()
            
        except Exception as e:
            print(f"Failed to setup OpenTelemetry instrumentation: {e}")
    
    def instrument_fastapi(self, app):
        """Instrument FastAPI application."""
        if self.enabled:
            try:
                FastAPIInstrumentor.instrument_app(
                    app,
                    excluded_urls="/health,/metrics"
                )
            except Exception as e:
                print(f"Failed to instrument FastAPI: {e}")
    
    def start_span(self, name: str, **attributes):
        """Start a new span."""
        if self.enabled and self.tracer:
            span = self.tracer.start_span(name)
            for key, value in attributes.items():
                span.set_attribute(key, value)
            return span
        return None
    
    def record_exception(self, span, exception: Exception):
        """Record exception in span."""
        if span and self.enabled:
            span.record_exception(exception)
            span.set_status(Status(StatusCode.ERROR, str(exception)))


class TelemetryContext:
    """Context manager for telemetry in request handlers."""
    
    def __init__(self, request: Request, logger: StructuredLogger, metrics: MetricsCollector):
        self.request = request
        self.logger = logger
        self.metrics = metrics
        self.request_id = getattr(request.state, 'request_id', 'unknown')
        self.start_time = time.time()
    
    def log_cache_hit(self, cache_key: str, hit: bool):
        """Log cache operation."""
        latency_ms = (time.time() - self.start_time) * 1000
        
        self.logger.log_cache_event(
            event="get",
            cache_key=cache_key,
            hit=hit,
            latency_ms=latency_ms
        )
        
        self.metrics.record_cache_operation("get", hit)
    
    def log_llm_generation(self, model: str, prompt_tokens: int, 
                          completion_tokens: int, latency_ms: float, cached: bool = False,
                          routing_metadata: Optional[Dict[str, Any]] = None):
        """Log LLM generation with optional routing metadata."""
        self.logger.log_llm_call(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            cache_hit=cached,
            routing_metadata=routing_metadata
        )
        
        self.metrics.record_llm_call(
            model=model,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            duration=latency_ms / 1000,
            cached=cached
        )
    
    def log_guardrails_decision(self, decision: str, triggered_rules: list, 
                               processing_time_ms: float, confidence: float):
        """Log guardrails decision."""
        self.logger.log_guardrails_event(
            decision=decision,
            triggered_rules=triggered_rules,
            processing_time_ms=processing_time_ms,
            confidence=confidence
        )
        
        self.metrics.record_guardrails_decision(decision)
    
    def log_routing_decision(self, model_tier: str, escalation_signals: list,
                           confidence_threshold: float, reasoning: str,
                           original_model: str, routed_model: str):
        """Log model routing decision for audit and analysis."""
        self.logger.log_routing_decision(
            model_tier=model_tier,
            escalation_signals=escalation_signals,
            confidence_threshold=confidence_threshold,
            reasoning=reasoning,
            original_model=original_model,
            routed_model=routed_model
        )


# Global telemetry instances
_logger: Optional[StructuredLogger] = None
_metrics: Optional[MetricsCollector] = None
_otel_manager: Optional[OpenTelemetryManager] = None


def get_logger() -> StructuredLogger:
    """Get global logger instance."""
    global _logger
    if _logger is None:
        _logger = StructuredLogger()
    return _logger


def get_metrics() -> MetricsCollector:
    """Get global metrics collector."""
    global _metrics
    if _metrics is None:
        metrics_enabled = os.getenv("METRICS_ENABLED", "true").lower() == "true"
        _metrics = MetricsCollector(enabled=metrics_enabled)
    return _metrics


def get_otel_manager() -> OpenTelemetryManager:
    """Get global OpenTelemetry manager."""
    global _otel_manager
    if _otel_manager is None:
        _otel_manager = OpenTelemetryManager()
    return _otel_manager


def create_telemetry_context(request: Request) -> TelemetryContext:
    """Create telemetry context for request."""
    return TelemetryContext(request, get_logger(), get_metrics())


# Decorator for function telemetry
def track_function(operation_name: str):
    """Decorator to track function execution."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            start_time = time.time()
            logger = get_logger()
            
            try:
                result = await func(*args, **kwargs)
                
                duration_ms = (time.time() - start_time) * 1000
                logger.logger.info(
                    "function_completed",
                    operation=operation_name,
                    duration_ms=duration_ms,
                    success=True
                )
                
                return result
                
            except Exception as e:
                duration_ms = (time.time() - start_time) * 1000
                logger.logger.error(
                    "function_failed",
                    operation=operation_name,
                    duration_ms=duration_ms,
                    error=str(e)
                )
                raise
        
        return wrapper
    return decorator


if __name__ == "__main__":
    # Test telemetry
    logger = get_logger()
    metrics = get_metrics()
    
    # Test logging
    logger.log_request("test-123", "GET", "/answer", 200, 150.5)
    logger.log_llm_call("test-model", 100, 50, 2000.0)
    
    # Test metrics
    metrics.record_request("GET", "/answer", 200, 0.15)
    metrics.record_llm_call("test-model", 100, 50, 2.0)
    
    print("Telemetry test completed")
    
    if PROMETHEUS_AVAILABLE:
        print("Prometheus metrics:")
        print(metrics.generate_metrics())