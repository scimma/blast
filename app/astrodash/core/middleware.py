import time
import re
import secrets
from typing import Callable, Optional, List
from fastapi import Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import JSONResponse
from astrodash.config.settings import get_settings
from astrodash.config.logging import get_logger

logger = get_logger(__name__)

class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """Add comprehensive security headers to all responses."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        response = await call_next(request)

        # Essential security headers
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["X-Frame-Options"] = "DENY"
        response.headers["X-XSS-Protection"] = "1; mode=block"
        response.headers["Referrer-Policy"] = "strict-origin-when-cross-origin"

        # HSTS header (only for HTTPS)
        if request.url.scheme == "https":
            response.headers["Strict-Transport-Security"] = "max-age=31536000; includeSubDomains; preload"

        # Additional security headers
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; "
            "script-src 'self' 'unsafe-inline' 'unsafe-eval'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self' https:; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'"
        )
        response.headers["Permissions-Policy"] = (
            "geolocation=(), microphone=(), camera=(), "
            "payment=(), usb=(), magnetometer=(), gyroscope=(), "
            "accelerometer=(), ambient-light-sensor=(), autoplay=()"
        )
        response.headers["X-Permitted-Cross-Domain-Policies"] = "none"
        response.headers["X-Download-Options"] = "noopen"
        response.headers["X-DNS-Prefetch-Control"] = "off"

        # Remove server information
        if "server" in response.headers:
            del response.headers["server"]

        return response

class RateLimitingMiddleware(BaseHTTPMiddleware):
    """Implement rate limiting to prevent abuse."""

    def __init__(self, app, requests_per_minute: int = 60, burst_limit: int = 10):
        super().__init__(app)
        self.requests_per_minute = requests_per_minute
        self.burst_limit = burst_limit
        self.request_counts = {}
        self.last_cleanup = time.time()

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Clean up old entries every 5 minutes
        current_time = time.time()
        if current_time - self.last_cleanup > 300:
            self._cleanup_old_entries(current_time)
            self.last_cleanup = current_time

        # Get client identifier (IP address)
        client_ip = self._get_client_ip(request)

        # Check rate limit
        if not self._check_rate_limit(client_ip, current_time):
            logger.warning(f"Rate limit exceeded for IP: {client_ip}")
            return JSONResponse(
                status_code=status.HTTP_429_TOO_MANY_REQUESTS,
                content={
                    "detail": "Rate limit exceeded. Please try again later.",
                    "retry_after": 60
                },
                headers={"Retry-After": "60"}
            )

        response = await call_next(request)
        return response

    def _get_client_ip(self, request: Request) -> str:
        """Extract real client IP considering proxies."""
        # Check for forwarded headers (in order of preference)
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            # Take the first IP in the chain
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        # Fallback to direct client IP
        if request.client:
            return request.client.host

        return "unknown"

    def _check_rate_limit(self, client_ip: str, current_time: float) -> bool:
        """Check if request is within rate limits."""
        if client_ip not in self.request_counts:
            self.request_counts[client_ip] = []

        # Remove requests older than 1 minute
        self.request_counts[client_ip] = [
            req_time for req_time in self.request_counts[client_ip]
            if current_time - req_time < 60
        ]

        # Check burst limit
        if len(self.request_counts[client_ip]) >= self.burst_limit:
            return False

        # Check rate limit
        if len(self.request_counts[client_ip]) >= self.requests_per_minute:
            return False

        # Add current request
        self.request_counts[client_ip].append(current_time)
        return True

    def _cleanup_old_entries(self, current_time: float):
        """Clean up old rate limiting entries."""
        cutoff_time = current_time - 300  # 5 minutes
        for client_ip in list(self.request_counts.keys()):
            self.request_counts[client_ip] = [
                req_time for req_time in self.request_counts[client_ip]
                if req_time > cutoff_time
            ]
            if not self.request_counts[client_ip]:
                del self.request_counts[client_ip]

class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """Enhanced request logging with security considerations."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        start_time = time.time()

        # Generate request ID for tracking
        request_id = secrets.token_hex(8)

        # Log request with security considerations
        client_ip = self._get_client_ip(request)
        user_agent = request.headers.get("user-agent", "unknown")

        # Sanitize sensitive information
        sanitized_path = self._sanitize_path(request.url.path)

        logger.info(
            f"Request {request_id}: {request.method} {sanitized_path} "
            f"from {client_ip} (UA: {user_agent[:100]})"
        )

        try:
            response = await call_next(request)
        except Exception as e:
            logger.error(f"Request {request_id} failed: {str(e)}")
            raise
        finally:
            # Log response
            process_time = time.time() - start_time
            status_code = getattr(response, 'status_code', 500)

            # Log based on status code
            if status_code >= 400:
                logger.warning(
                    f"Request {request_id}: {status_code} in {process_time:.3f}s "
                    f"({request.method} {sanitized_path})"
                )
            else:
                logger.info(
                    f"Request {request_id}: {status_code} in {process_time:.3f}s "
                    f"({request.method} {sanitized_path})"
                )

        return response

    def _get_client_ip(self, request: Request) -> str:
        """Extract real client IP considering proxies."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        if request.client:
            return request.client.host

        return "unknown"

    def _sanitize_path(self, path: str) -> str:
        """Sanitize path to remove sensitive information."""
        # Remove potential sensitive parameters
        sensitive_patterns = [
            r'/api/v1/.*?password.*?',
            r'/api/v1/.*?token.*?',
            r'/api/v1/.*?secret.*?',
            r'/api/v1/.*?key.*?'
        ]

        sanitized_path = path
        for pattern in sensitive_patterns:
            sanitized_path = re.sub(pattern, '[REDACTED]', sanitized_path, flags=re.IGNORECASE)

        return sanitized_path

class InputValidationMiddleware(BaseHTTPMiddleware):
    """Validate and sanitize input to prevent injection attacks."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        # Check for suspicious patterns in headers
        if self._has_suspicious_headers(request):
            logger.warning(f"Suspicious headers detected from {self._get_client_ip(request)}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Invalid request headers"}
            )

        # Check for suspicious patterns in query parameters
        if self._has_suspicious_query_params(request):
            logger.warning(f"Suspicious query parameters detected from {self._get_client_ip(request)}")
            return JSONResponse(
                status_code=status.HTTP_400_BAD_REQUEST,
                content={"detail": "Invalid query parameters"}
            )

        response = await call_next(request)
        return response

    def _has_suspicious_headers(self, request: Request) -> bool:
        """Check for suspicious patterns in headers."""
        suspicious_patterns = [
            r'<script',
            r'javascript:',
            r'data:text/html',
            r'vbscript:',
            r'onload=',
            r'onerror=',
            r'<iframe',
            r'<object',
            r'<embed'
        ]

        for header_name, header_value in request.headers.items():
            header_value_lower = header_value.lower()
            for pattern in suspicious_patterns:
                if re.search(pattern, header_value_lower, re.IGNORECASE):
                    return True

        return False

    def _has_suspicious_query_params(self, request: Request) -> bool:
        """Check for suspicious patterns in query parameters."""
        suspicious_patterns = [
            r'<script',
            r'javascript:',
            r'data:text/html',
            r'vbscript:',
            r'onload=',
            r'onerror=',
            r'<iframe',
            r'<object',
            r'<embed',
            r'../',
            r'..\\',
            r'%00',
            r'%0d',
            r'%0a'
        ]

        query_string = str(request.url.query)
        query_string_lower = query_string.lower()

        for pattern in suspicious_patterns:
            if re.search(pattern, query_string_lower, re.IGNORECASE):
                return True

        return False

    def _get_client_ip(self, request: Request) -> str:
        """Extract real client IP considering proxies."""
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            return forwarded_for.split(",")[0].strip()

        real_ip = request.headers.get("X-Real-IP")
        if real_ip:
            return real_ip

        if request.client:
            return request.client.host

        return "unknown"

class ErrorHandlingMiddleware(BaseHTTPMiddleware):
    """Handle errors gracefully and securely."""

    async def dispatch(self, request: Request, call_next: Callable) -> Response:
        try:
            response = await call_next(request)
            return response
        except Exception as e:
            logger.error(f"Unhandled exception in {request.method} {request.url.path}: {str(e)}")

            # Don't expose internal errors in production
            if get_settings().environment == "production":
                return JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content={"detail": "Internal server error"}
                )
            else:
                # In development, provide more details
                return JSONResponse(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    content={"detail": f"Internal server error: {str(e)}"}
                )

def setup_middleware(app):
    """Setup all middleware for the application with security-first approach."""
    settings = get_settings()

    # Security middleware (order matters!)

    # 1. Error handling middleware (first to catch all errors)
    app.add_middleware(ErrorHandlingMiddleware)

    # 2. Input validation middleware (early validation)
    app.add_middleware(InputValidationMiddleware)

    # 3. Rate limiting middleware (prevent abuse)
    app.add_middleware(
        RateLimitingMiddleware,
        requests_per_minute=settings.rate_limit_requests_per_minute,
        burst_limit=settings.rate_limit_burst_limit
    )

    # 4. Trusted host middleware for web app only
    if (settings.allowed_hosts and
        settings.allowed_hosts != ["*"] and
        len(settings.allowed_hosts) > 0 and
        not all(host == "*" for host in settings.allowed_hosts)):
        logger.info(f"Applying TrustedHostMiddleware with allowed hosts: {settings.allowed_hosts}")
        app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=settings.allowed_hosts
        )
    else:
        logger.info("Skipping TrustedHostMiddleware - API mode (allowing all hosts)")

    # 5. CORS middleware (configured securely for API usage)
    if settings.cors_origins and settings.cors_origins != ["*"]:
        # Specific CORS origins configured
        app.add_middleware(
            CORSMiddleware,
            allow_origins=settings.cors_origins,
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["*"],
            expose_headers=["Content-Length", "Content-Type"],
            max_age=3600,
        )
        logger.info(f"Applied CORS middleware with specific origins: {settings.cors_origins}")
    else:
        # Allow all origins for API usage (common for APIs)
        app.add_middleware(
            CORSMiddleware,
            allow_origins=["*"],
            allow_credentials=True,
            allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
            allow_headers=["*"],
            expose_headers=["Content-Length", "Content-Type"],
            max_age=3600,
        )
        logger.info("Applied CORS middleware with allow_origins=['*'] for API usage")

    # 6. Security headers middleware (add security headers)
    app.add_middleware(SecurityHeadersMiddleware)

    # 7. Request logging middleware (last to log everything)
    app.add_middleware(RequestLoggingMiddleware)

    logger.info("Middleware setup completed with security-first approach for API usage")
