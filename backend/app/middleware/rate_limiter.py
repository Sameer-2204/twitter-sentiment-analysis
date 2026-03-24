"""
middleware/rate_limiter.py — Simple in-memory rate limiter for the
prediction endpoints.

Uses a sliding-window approach: for each IP address, keep a list of
timestamps.  On every request, prune timestamps older than the window
and check if the count exceeds the limit.

This is intentionally simple (no Redis / external store) and works
correctly with a single-worker deployment.
"""

from __future__ import annotations

import logging
import time
from collections import defaultdict
from typing import Dict, List

from fastapi import HTTPException, Request

from app.config import get_settings

logger = logging.getLogger(__name__)


class RateLimiter:
    """In-memory per-IP rate limiter.

    Parameters
    ----------
    max_requests : int
        Maximum number of requests allowed within the window.
    window_seconds : int
        Sliding window size in seconds.
    """

    def __init__(
        self,
        max_requests: int | None = None,
        window_seconds: int | None = None,
    ) -> None:
        settings = get_settings()
        self.max_requests: int = max_requests or settings.RATE_LIMIT_MAX_REQUESTS
        self.window_seconds: int = window_seconds or settings.RATE_LIMIT_WINDOW_SECONDS
        self._requests: Dict[str, List[float]] = defaultdict(list)

    def _get_client_ip(self, request: Request) -> str:
        """Extract client IP from the request."""
        # Check X-Forwarded-For header (Railway / reverse proxy)
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        return request.client.host if request.client else "unknown"

    def check_rate_limit(self, request: Request) -> None:
        """Check if the client IP has exceeded the rate limit.

        Raises
        ------
        HTTPException
            429 Too Many Requests if the limit is exceeded.
        """
        ip = self._get_client_ip(request)
        now = time.time()
        window_start = now - self.window_seconds

        # Prune timestamps outside the window
        self._requests[ip] = [
            ts for ts in self._requests[ip] if ts > window_start
        ]

        if len(self._requests[ip]) >= self.max_requests:
            retry_after = int(self.window_seconds - (now - self._requests[ip][0]))
            logger.warning(
                "Rate limit exceeded for IP %s (%d/%d in %ds)",
                ip,
                len(self._requests[ip]),
                self.max_requests,
                self.window_seconds,
            )
            raise HTTPException(
                status_code=429,
                detail=(
                    f"Too many requests. Limit: {self.max_requests} "
                    f"per {self.window_seconds}s. "
                    f"Retry after {max(1, retry_after)}s."
                ),
                headers={"Retry-After": str(max(1, retry_after))},
            )

        # Record this request
        self._requests[ip].append(now)

    def cleanup(self) -> None:
        """Remove stale entries from the internal dict."""
        now = time.time()
        window_start = now - self.window_seconds
        stale_ips = [
            ip for ip, timestamps in self._requests.items()
            if not timestamps or timestamps[-1] < window_start
        ]
        for ip in stale_ips:
            del self._requests[ip]


# Module-level singleton
rate_limiter = RateLimiter()
