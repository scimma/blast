import time
from typing import Dict, Any
from datetime import datetime
from collections import defaultdict
import psutil
import os
from astrodash.config.logging import get_logger

logger = get_logger(__name__)

class MetricsCollector:
    """Collect and track application metrics."""

    def __init__(self):
        self.request_counts = defaultdict(int)
        self.response_times = defaultdict(list)
        self.error_counts = defaultdict(int)
        self.start_time = time.time()

    def record_request(self, endpoint: str, method: str, status_code: int, response_time: float):
        """Record a request metric."""
        key = f"{method} {endpoint}"
        self.request_counts[key] += 1

        if status_code >= 400:
            self.error_counts[key] += 1

        self.response_times[key].append(response_time)

        # Keep only last 1000 response times per endpoint
        if len(self.response_times[key]) > 1000:
            self.response_times[key] = self.response_times[key][-1000:]

    def get_metrics(self) -> Dict[str, Any]:
        """Get current metrics."""
        uptime = time.time() - self.start_time

        # Calculate average response times
        avg_response_times = {}
        for endpoint, times in self.response_times.items():
            if times:
                avg_response_times[endpoint] = sum(times) / len(times)

        # System metrics
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')

        return {
            "uptime_seconds": uptime,
            "total_requests": sum(self.request_counts.values()),
            "total_errors": sum(self.error_counts.values()),
            "request_counts": dict(self.request_counts),
            "error_counts": dict(self.error_counts),
            "avg_response_times": avg_response_times,
            "system": {
                "cpu_percent": cpu_percent,
                "memory_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "disk_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3)
            }
        }

# Global metrics collector
metrics = MetricsCollector()

def get_health_status() -> Dict[str, Any]:
    """Get comprehensive health status."""
    try:
        # Basic health checks
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "version": "1.0.0"
        }

        # Check disk space
        disk = psutil.disk_usage('/')
        if disk.percent > 90:
            health_status["status"] = "warning"
            health_status["disk_warning"] = f"Disk usage: {disk.percent}%"

        # Check memory
        memory = psutil.virtual_memory()
        if memory.percent > 90:
            health_status["status"] = "warning"
            health_status["memory_warning"] = f"Memory usage: {memory.percent}%"

        # Add metrics
        health_status["metrics"] = metrics.get_metrics()

        return health_status

    except Exception as e:
        logger.error(f"Error getting health status: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

def log_performance_metrics(endpoint: str, method: str, status_code: int, response_time: float):
    """Log performance metrics."""
    metrics.record_request(endpoint, method, status_code, response_time)

    # Log slow requests
    if response_time > 5.0:  # 5 seconds
        logger.warning(f"Slow request: {method} {endpoint} took {response_time:.2f}s")

    # Log errors
    if status_code >= 500:
        logger.error(f"Server error: {method} {endpoint} returned {status_code}")
    elif status_code >= 400:
        logger.warning(f"Client error: {method} {endpoint} returned {status_code}")
