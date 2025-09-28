#!/usr/bin/env python3
"""
vLLM client with retries, circuit breaker, and usage tracking.
Production-grade HTTP client for vLLM inference service.
"""

import asyncio
import time
import json
import logging
from typing import Dict, Any, Optional, AsyncGenerator, Tuple
from dataclasses import dataclass
from enum import Enum
import httpx
import backoff
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class CircuitBreakerState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


@dataclass
class GenerationConfig:
    """Configuration for text generation."""
    max_tokens: int = 512
    temperature: float = 0.0
    top_p: float = 0.9
    top_k: int = 50
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    stop_sequences: list = None
    json_only: bool = True
    response_format: Optional[Dict[str, Any]] = None  # {"type": "json_object"} for constrained JSON
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for API call."""
        config = {
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "frequency_penalty": self.frequency_penalty,
            "presence_penalty": self.presence_penalty
        }
        if self.stop_sequences:
            config["stop"] = self.stop_sequences
        
        # Add response format constraint if specified
        if self.response_format:
            config["response_format"] = self.response_format
        elif self.json_only:
            # Default JSON constraint when json_only is True
            config["response_format"] = {"type": "json_object"}
        
        return config


@dataclass
class GenerationResult:
    """Result from text generation."""
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency_ms: float
    model: str
    finish_reason: str = "stop"


class CircuitBreaker:
    """Circuit breaker for vLLM service calls."""
    
    def __init__(self, 
                 failure_threshold: int = 5,
                 recovery_timeout: float = 60.0,
                 expected_exception: type = Exception):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.expected_exception = expected_exception
        
        self.failure_count = 0
        self.last_failure_time = None
        self.state = CircuitBreakerState.CLOSED
    
    def call(self, func, *args, **kwargs):
        """Execute function with circuit breaker logic."""
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
            else:
                raise Exception("Circuit breaker is OPEN")
        
        try:
            result = func(*args, **kwargs)
            self._on_success()
            return result
        except self.expected_exception as e:
            self._on_failure()
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if enough time has passed to attempt reset."""
        return (time.time() - self.last_failure_time) >= self.recovery_timeout
    
    def _on_success(self):
        """Reset circuit breaker on successful call."""
        self.failure_count = 0
        self.state = CircuitBreakerState.CLOSED
    
    def _on_failure(self):
        """Handle failure and potentially open circuit."""
        self.failure_count += 1
        self.last_failure_time = time.time()
        
        if self.failure_count >= self.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            logger.warning(f"Circuit breaker opened after {self.failure_count} failures")


class VLLMClient:
    """
    Production-grade client for vLLM inference service.
    Features: retries, circuit breaker, timeouts, usage tracking.
    """
    
    def __init__(self, 
                 base_url: str,
                 model_name: str,
                 timeout: float = 60.0,
                 max_retries: int = 3,
                 circuit_breaker_threshold: int = 5):
        """Initialize vLLM client."""
        self.base_url = base_url.rstrip('/')
        self.model_name = model_name
        self.timeout = timeout
        self.max_retries = max_retries
        
        # HTTP client with custom timeout
        self.client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout),
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100)
        )
        
        # Circuit breaker for fault tolerance
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=circuit_breaker_threshold,
            recovery_timeout=60.0,
            expected_exception=(httpx.HTTPError, asyncio.TimeoutError)
        )
        
        # Metrics
        self.total_requests = 0
        self.failed_requests = 0
        self.total_tokens = 0
        
        logger.info(f"Initialized vLLM client for {base_url} with model {model_name}")
    
    async def generate(self, 
                      prompt: str, 
                      config: Optional[GenerationConfig] = None,
                      model_name: Optional[str] = None) -> GenerationResult:
        """
        Generate text completion for prompt.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            model_name: Override model name for this request (for routing)
            
        Returns:
            GenerationResult with text and usage stats
        """
        if config is None:
            config = GenerationConfig()
        
        start_time = time.time()
        
        # Use provided model name or default
        effective_model = model_name or self.model_name
        
        try:
            result = await self._generate_with_retries(prompt, config, effective_model)
            
            # Calculate latency
            latency_ms = (time.time() - start_time) * 1000
            
            # Update metrics
            self.total_requests += 1
            self.total_tokens += result["usage"]["total_tokens"]
            
            # Create result object
            return GenerationResult(
                text=result["text"],
                prompt_tokens=result["usage"]["prompt_tokens"],
                completion_tokens=result["usage"]["completion_tokens"],
                total_tokens=result["usage"]["total_tokens"],
                latency_ms=latency_ms,
                model=effective_model,
                finish_reason=result.get("finish_reason", "stop")
            )
            
        except Exception as e:
            self.failed_requests += 1
            logger.error(f"Generation failed: {e}")
            raise
    
    async def generate_stream(self, 
                            prompt: str, 
                            config: Optional[GenerationConfig] = None,
                            model_name: Optional[str] = None) -> AsyncGenerator[str, None]:
        """
        Stream text generation token by token.
        
        Args:
            prompt: Input prompt
            config: Generation configuration
            model_name: Override model name for this request (for routing)
            
        Yields:
            Generated tokens as strings
        """
        if config is None:
            config = GenerationConfig()
        
        # Use provided model name or default
        effective_model = model_name or self.model_name
        
        try:
            async for chunk in self._stream_with_retries(prompt, config, effective_model):
                yield chunk
        except Exception as e:
            self.failed_requests += 1
            logger.error(f"Streaming generation failed: {e}")
            raise
    
    @backoff.on_exception(
        backoff.expo,
        (httpx.HTTPError, asyncio.TimeoutError),
        max_tries=3,
        max_time=300
    )
    async def _generate_with_retries(self, prompt: str, config: GenerationConfig, model_name: str) -> Dict[str, Any]:
        """Generate with exponential backoff retries."""
        return await self.circuit_breaker.call(self._make_generation_request, prompt, config, model_name)
    
    async def _make_generation_request(self, prompt: str, config: GenerationConfig, model_name: str) -> Dict[str, Any]:
        """Make the actual HTTP request to vLLM."""
        payload = {
            "prompt": prompt,
            "model": model_name,
            **config.to_dict()
        }
        
        # Add usage tracking
        payload["logprobs"] = None  # Disable to save bandwidth
        payload["echo"] = False
        
        response = await self.client.post(
            f"{self.base_url}/v1/completions",
            json=payload,
            headers={"Content-Type": "application/json"}
        )
        
        response.raise_for_status()
        result = response.json()
        
        # Extract text and usage from vLLM response format
        if "choices" not in result or not result["choices"]:
            raise ValueError("Invalid response format from vLLM")
        
        choice = result["choices"][0]
        text = choice["text"]
        
        # Extract usage statistics
        usage = result.get("usage", {})
        if not usage:
            # Estimate tokens if not provided
            estimated_prompt = len(prompt.split()) * 1.3  # Rough estimation
            estimated_completion = len(text.split()) * 1.3
            usage = {
                "prompt_tokens": int(estimated_prompt),
                "completion_tokens": int(estimated_completion),
                "total_tokens": int(estimated_prompt + estimated_completion)
            }
        
        return {
            "text": text,
            "usage": usage,
            "finish_reason": choice.get("finish_reason", "stop")
        }

    @staticmethod
    def repair_json_from_text(text: str) -> Optional[str]:
        """Attempt to extract a JSON object from free-form text.

        Finds the first '{' and the last '}' and returns the enclosed substring.
        Returns None if extraction fails.
        """
        try:
            start = text.find('{')
            end = text.rfind('}')
            if start != -1 and end != -1 and start < end:
                candidate = text[start:end+1]
                # Basic sanity check by attempting to load
                json.loads(candidate)
                return candidate
        except Exception:
            return None
        return None
    
    async def _stream_with_retries(self, prompt: str, config: GenerationConfig, model_name: str) -> AsyncGenerator[str, None]:
        """Stream generation with retries."""
        payload = {
            "prompt": prompt,
            "model": model_name,
            "stream": True,
            **config.to_dict()
        }
        
        try:
            async with self.client.stream(
                "POST",
                f"{self.base_url}/v1/completions",
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                response.raise_for_status()
                
                async for line in response.aiter_lines():
                    if line.startswith("data: "):
                        data_str = line[6:]  # Remove "data: " prefix
                        if data_str.strip() == "[DONE]":
                            break
                        
                        try:
                            data = json.loads(data_str)
                            if "choices" in data and data["choices"]:
                                choice = data["choices"][0]
                                if "text" in choice:
                                    yield choice["text"]
                        except json.JSONDecodeError:
                            continue
                            
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            raise
    
    async def health_check(self) -> bool:
        """Check if vLLM service is healthy."""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception as e:
            logger.warning(f"Health check failed: {e}")
            return False
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information from vLLM."""
        try:
            response = await self.client.get(f"{self.base_url}/v1/models")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {}
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get client metrics."""
        error_rate = self.failed_requests / self.total_requests if self.total_requests > 0 else 0
        
        return {
            "total_requests": self.total_requests,
            "failed_requests": self.failed_requests,
            "error_rate": error_rate,
            "total_tokens": self.total_tokens,
            "circuit_breaker_state": self.circuit_breaker.state.value,
            "circuit_breaker_failures": self.circuit_breaker.failure_count
        }
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()
    
    @asynccontextmanager
    async def managed_client(self):
        """Async context manager for automatic cleanup."""
        try:
            yield self
        finally:
            await self.close()


# Factory function for dependency injection
def create_vllm_client(base_url: str, 
                      model_name: str, 
                      **kwargs) -> VLLMClient:
    """Factory function to create vLLM client."""
    return VLLMClient(base_url, model_name, **kwargs)


# Test function
async def test_vllm_client():
    """Test vLLM client functionality."""
    client = VLLMClient(
        base_url="http://localhost:8000",
        model_name="test-model"
    )
    
    try:
        # Test health check
        healthy = await client.health_check()
        print(f"vLLM healthy: {healthy}")
        
        if healthy:
            # Test generation
            result = await client.generate(
                "What is the Employment Act of Malaysia?",
                GenerationConfig(max_tokens=100, temperature=0.1)
            )
            print(f"Generated: {result.text[:100]}...")
            print(f"Tokens: {result.total_tokens}, Latency: {result.latency_ms:.1f}ms")
            
            # Test streaming
            print("\nStreaming test:")
            async for chunk in client.generate_stream("Tell me about", GenerationConfig(max_tokens=50)):
                print(chunk, end="", flush=True)
            print()
            
        # Print metrics
        metrics = client.get_metrics()
        print(f"\nMetrics: {metrics}")
        
    finally:
        await client.close()


if __name__ == "__main__":
    asyncio.run(test_vllm_client())
