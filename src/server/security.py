#!/usr/bin/env python3
"""
Security middleware and authentication for Employment Act Malaysia compliance agent.
Implements API key authentication, security headers, and request validation.
"""

import os
import time
import hashlib
import secrets
from typing import Dict, Any, Optional, List, Set
from datetime import datetime, timedelta
from dataclasses import dataclass

from fastapi import HTTPException, status, Request, Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.responses import Response

# Import for structured logging
from .telemetry import get_logger

logger = get_logger()


@dataclass
class APIKey:
    """API key data structure."""
    key_id: str
    key_hash: str
    name: str
    permissions: Set[str]
    created_at: datetime
    last_used: Optional[datetime] = None
    usage_count: int = 0
    rate_limit: int = 100  # requests per minute
    is_active: bool = True


class SecurityConfig:
    """Security configuration."""
    
    def __init__(self):
        # API Key settings
        self.api_keys_enabled = os.getenv("API_KEYS_ENABLED", "true").lower() == "true"
        self.require_auth_endpoints = set(os.getenv("REQUIRE_AUTH_ENDPOINTS", "/answer,/section,/severance").split(","))
        self.public_endpoints = set(os.getenv("PUBLIC_ENDPOINTS", "/health,/metrics,/docs,/openapi.json").split(","))
        
        # Security headers
        self.security_headers_enabled = os.getenv("SECURITY_HEADERS_ENABLED", "true").lower() == "true"
        self.cors_origins = os.getenv("CORS_ORIGINS", "http://localhost:8501").split(",")
        
        # Rate limiting
        self.rate_limit_enabled = os.getenv("RATE_LIMIT_ENABLED", "true").lower() == "true"
        self.default_rate_limit = int(os.getenv("DEFAULT_RATE_LIMIT", "100"))  # per minute
        
        # Content Security Policy
        self.csp_policy = os.getenv("CSP_POLICY", 
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data:; "
            "font-src 'self';"
        )


class APIKeyManager:
    """Manages API keys and authentication."""
    
    def __init__(self):
        self.config = SecurityConfig()
        self.api_keys: Dict[str, APIKey] = {}
        self.usage_tracker: Dict[str, Dict[str, int]] = {}  # key_id -> {minute_timestamp -> count}
        self._load_api_keys()
    
    def _load_api_keys(self):
        """Load API keys from environment or file."""
        # Load from environment variable (comma-separated key:name:permissions)
        api_keys_env = os.getenv("API_KEYS", "")
        if api_keys_env:
            for key_data in api_keys_env.split(","):
                if ":" in key_data:
                    parts = key_data.split(":")
                    key = parts[0]
                    name = parts[1] if len(parts) > 1 else "default"
                    permissions = set(parts[2].split("|")) if len(parts) > 2 else {"read", "write"}
                    self.add_api_key(key, name, permissions)
        
        # Create default key if none exist and in development
        if not self.api_keys and os.getenv("ENVIRONMENT", "development") == "development":
            default_key = "ea-dev-" + secrets.token_urlsafe(32)
            self.add_api_key(default_key, "development", {"read", "write"})
            logger.logger.info("Created development API key", api_key_prefix=default_key[:10])
    
    def add_api_key(self, key: str, name: str, permissions: Set[str]) -> str:
        """Add a new API key."""
        key_id = hashlib.sha256(key.encode()).hexdigest()[:16]
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            name=name,
            permissions=permissions,
            created_at=datetime.utcnow()
        )
        
        self.api_keys[key_hash] = api_key
        return key_id
    
    def validate_api_key(self, key: str) -> Optional[APIKey]:
        """Validate API key and return key info."""
        if not key or not self.config.api_keys_enabled:
            return None
        
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        api_key = self.api_keys.get(key_hash)
        
        if not api_key or not api_key.is_active:
            return None
        
        # Update usage
        api_key.last_used = datetime.utcnow()
        api_key.usage_count += 1
        
        # Check rate limit
        if self._is_rate_limited(api_key):
            raise HTTPException(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                detail="Rate limit exceeded"
            )
        
        return api_key
    
    def _is_rate_limited(self, api_key: APIKey) -> bool:
        """Check if API key is rate limited."""
        if not self.config.rate_limit_enabled:
            return False
        
        current_minute = int(time.time() // 60)
        key_usage = self.usage_tracker.setdefault(api_key.key_id, {})
        
        # Clean old entries (keep last 5 minutes)
        cutoff = current_minute - 5
        key_usage = {k: v for k, v in key_usage.items() if k >= cutoff}
        self.usage_tracker[api_key.key_id] = key_usage
        
        # Count requests in current minute
        current_count = key_usage.get(current_minute, 0)
        if current_count >= api_key.rate_limit:
            return True
        
        # Increment counter
        key_usage[current_minute] = current_count + 1
        return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get API key usage statistics."""
        stats = {
            "total_keys": len(self.api_keys),
            "active_keys": sum(1 for k in self.api_keys.values() if k.is_active),
            "usage_by_key": {}
        }
        
        for api_key in self.api_keys.values():
            stats["usage_by_key"][api_key.key_id] = {
                "name": api_key.name,
                "usage_count": api_key.usage_count,
                "last_used": api_key.last_used.isoformat() if api_key.last_used else None,
                "permissions": list(api_key.permissions),
                "is_active": api_key.is_active
            }
        
        return stats


class SecurityMiddleware(BaseHTTPMiddleware):
    """Security middleware for headers and basic protection."""
    
    def __init__(self, app, config: SecurityConfig):
        super().__init__(app)
        self.config = config
    
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        """Apply security headers and checks."""
        # Process request
        response = await call_next(request)
        
        # Add security headers
        if self.config.security_headers_enabled:
            self._add_security_headers(response)
        
        return response
    
    def _add_security_headers(self, response: Response):
        """Add comprehensive security headers."""
        # Prevent XSS
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        
        # HTTPS enforcement (in production)
        if os.getenv("ENVIRONMENT") == "production":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"
        
        # Content Security Policy
        response.headers["Content-Security-Policy"] = self.config.csp_policy
        
        # Referrer Policy
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"
        
        # Permissions Policy
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=(), "
            "payment=(), usb=(), magnetometer=(), gyroscope=()"
        )
        
        # Cache control for sensitive endpoints
        response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, private"
        response.headers["Pragma"] = "no-cache"
        response.headers["Expires"] = "0"


# Authentication dependency
security = HTTPBearer(auto_error=False)
api_key_manager = APIKeyManager()


async def get_api_key(credentials: Optional[HTTPAuthorizationCredentials] = Depends(security)) -> Optional[APIKey]:
    """Dependency to extract and validate API key."""
    if not api_key_manager.config.api_keys_enabled:
        return None
    
    if not credentials:
        return None
    
    return api_key_manager.validate_api_key(credentials.credentials)


async def require_api_key(request: Request) -> APIKey:
    """Dependency that requires valid API key."""
    # Check if endpoint requires authentication
    path = request.url.path
    
    if path in api_key_manager.config.public_endpoints:
        return None
    
    if not api_key_manager.config.api_keys_enabled:
        return None
    
    if path not in api_key_manager.config.require_auth_endpoints:
        return None
    
    # Extract API key from header or query parameter
    api_key = None
    
    # Try Authorization header
    auth_header = request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        api_key = auth_header[7:]
    
    # Try X-API-Key header
    if not api_key:
        api_key = request.headers.get("X-API-Key")
    
    # Try query parameter
    if not api_key:
        api_key = request.query_params.get("api_key")
    
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="API key required",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    validated_key = api_key_manager.validate_api_key(api_key)
    if not validated_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
            headers={"WWW-Authenticate": "Bearer"}
        )
    
    return validated_key


async def require_permission(permission: str):
    """Dependency factory for permission checking."""
    async def check_permission(api_key: APIKey = Depends(require_api_key)) -> APIKey:
        if not api_key:
            return api_key  # Public endpoint
        
        if permission not in api_key.permissions:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission '{permission}' required"
            )
        
        return api_key
    
    return check_permission


def create_security_middleware(app):
    """Create and configure security middleware."""
    config = SecurityConfig()
    return SecurityMiddleware(app, config)


# Utility functions for key generation
def generate_api_key(prefix: str = "ea") -> str:
    """Generate a secure API key."""
    return f"{prefix}-{secrets.token_urlsafe(32)}"


def hash_api_key(key: str) -> str:
    """Hash API key for storage."""
    return hashlib.sha256(key.encode()).hexdigest()


# Test function
def test_security():
    """Test security components."""
    manager = APIKeyManager()
    
    # Generate test key
    test_key = generate_api_key("test")
    key_id = manager.add_api_key(test_key, "test-key", {"read"})
    
    print(f"Generated test key: {test_key}")
    print(f"Key ID: {key_id}")
    
    # Validate key
    validated = manager.validate_api_key(test_key)
    print(f"Validation result: {validated.name if validated else 'Failed'}")
    
    # Get stats
    stats = manager.get_stats()
    print(f"Stats: {stats}")


if __name__ == "__main__":
    test_security()