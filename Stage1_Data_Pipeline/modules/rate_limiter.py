#!/usr/bin/env python3
"""
Thread-Safe Rate Limiter Module
"""

import time
import threading
import random
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class RateLimiterManager:
    """
    Thread-safe rate limiter using token bucket algorithm.
    Implements backoff handling for 429 responses.
    
    Each worker process gets its own instance. With max_workers=5 and 9 req/s per worker,
    the system stays conservative relative to the actual global limit.
    """
    
    def __init__(self, requests_per_second: float, burst_size: Optional[int] = None,
                 name: str = "rate_limiter"):
        """
        Initialize rate limiter.
        
        Args:
            requests_per_second: Maximum sustained request rate
            burst_size: Maximum burst size (tokens). Defaults to requests_per_second / 2
            name: Name for logging purposes
        """
        self.name = name
        self.requests_per_second = requests_per_second
        self.burst_size = burst_size if burst_size is not None else max(1, int(requests_per_second / 2))
        
        # Token bucket parameters
        self.tokens_per_second = requests_per_second
        self.max_tokens = float(self.burst_size)
        self.tokens = float(self.burst_size)
        
        # Timing
        self.last_update = time.time()
        self.min_interval = 1.0 / requests_per_second if requests_per_second > 0 else 0
        
        # Backoff handling
        self.backoff_until = 0.0
        self.backoff_duration = 5.0
        self.max_backoff = 300.0
        
        # Thread safety
        self.lock = threading.Lock()
        
        logger.debug(f"Initialized {name}: {requests_per_second:.2f} req/s, "
                     f"burst={self.burst_size}, interval={self.min_interval:.3f}s")
    
    def _refill_tokens(self, current_time: float):
        """Refill tokens based on elapsed time (token bucket algorithm)."""
        elapsed = current_time - self.last_update
        
        # Add tokens based on elapsed time
        tokens_to_add = elapsed * self.tokens_per_second
        self.tokens = min(self.tokens + tokens_to_add, self.max_tokens)
        self.last_update = current_time
    
    def acquire(self, tokens: float = 1.0, timeout: Optional[float] = 60.0) -> bool:
        """
        Acquire tokens from the rate limiter (blocking).
        
        Args:
            tokens: Number of tokens to acquire (default 1.0)
            timeout: Maximum time to wait in seconds (None for infinite)
        
        Returns:
            True if tokens acquired, False if timeout
        """
        start_time = time.time()
        
        while True:
            with self.lock:
                current_time = time.time()
                
                # Check if we're in a backoff period
                if current_time < self.backoff_until:
                    backoff_remaining = self.backoff_until - current_time
                    if timeout and (current_time - start_time + backoff_remaining) > timeout:
                        logger.error(f"{self.name}: Timeout during backoff")
                        return False
                    
                    sleep_duration = min(0.5, backoff_remaining)
                    
                elif current_time >= self.backoff_until:
                    # Backoff period is over - reset duration
                    if self.backoff_until > 0:
                        self.backoff_duration = 5.0  # Reset to base backoff
                    
                    # Refill tokens
                    self._refill_tokens(current_time)
                    
                    # Check if we have enough tokens
                    if self.tokens >= tokens:
                        # Consume tokens and return immediately
                        self.tokens -= tokens
                        logger.debug(f"{self.name}: Acquired {tokens} tokens, {self.tokens:.2f} remaining")
                        return True
                    else:
                        # Not enough tokens - calculate sleep time
                        if timeout and (current_time - start_time) >= timeout:
                            logger.error(f"{self.name}: Timeout waiting for tokens")
                            return False
                        
                        # Calculate how long until we have enough tokens
                        time_to_refill = (tokens - self.tokens) / self.tokens_per_second
                        sleep_duration = max(self.min_interval * 0.1, min(0.5, time_to_refill))
                else:
                    sleep_duration = 0.01
            
            # Sleep outside lock to allow other threads to proceed
            if sleep_duration > 0:
                time.sleep(sleep_duration)
            else:
                continue
    
    async def acquire_async(self, tokens: float = 1.0, timeout: Optional[float] = 60.0) -> bool:
        """
        Async version of acquire for use with asyncio.
        
        Args:
            tokens: Number of tokens to acquire
            timeout: Maximum time to wait in seconds
            
        Returns:
            True if tokens acquired, False if timeout
        """
        import asyncio
        
        start_time = time.time()
        
        while True:
            with self.lock:
                current_time = time.time()
                
                # Check backoff
                if current_time < self.backoff_until:
                    backoff_remaining = self.backoff_until - current_time
                    if timeout and (current_time - start_time + backoff_remaining) > timeout:
                        logger.error(f"{self.name}: Async timeout during backoff")
                        return False
                    
                    sleep_time = min(0.5, backoff_remaining)
                    
                elif current_time >= self.backoff_until:
                    # Backoff period is over
                    if self.backoff_until > 0:
                        self.backoff_duration = 5.0  # Reset to base backoff
                    
                    # Refill tokens
                    self._refill_tokens(current_time)
                    
                    # Check if we have enough tokens
                    if self.tokens >= tokens:
                        self.tokens -= tokens
                        logger.debug(f"{self.name}: Async acquired {tokens} tokens, {self.tokens:.2f} remaining")
                        return True
                    else:
                        # Check timeout
                        if timeout and (current_time - start_time) >= timeout:
                            logger.error(f"{self.name}: Async timeout waiting for tokens")
                            return False
                        
                        # Calculate sleep time
                        time_to_refill = (tokens - self.tokens) / self.tokens_per_second
                        sleep_time = max(self.min_interval * 0.1, min(0.5, time_to_refill))
                else:
                    sleep_time = 0.01
            
            # Async sleep outside the lock
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)
            else:
                continue
    
    def report_rate_limit_error(self, retry_after: Optional[float] = None):
        """
        Report a rate limit error (e.g., HTTP 429) to trigger backoff.
        
        Args:
            retry_after: Suggested retry delay from API (seconds)
        """
        with self.lock:
            current_time = time.time()
            
            # Use provided retry_after or exponential backoff
            if retry_after:
                backoff_duration = retry_after
            else:
                # Exponential backoff if already in backoff
                if current_time < self.backoff_until:
                    self.backoff_duration = min(
                        self.backoff_duration * 2,
                        self.max_backoff
                    )
                backoff_duration = self.backoff_duration
            
            self.backoff_until = current_time + backoff_duration
            logger.warning(f"{self.name}: Rate limit error - backing off for {backoff_duration:.1f}s")
    
    def get_stats(self) -> dict:
        """Get current rate limiter statistics."""
        with self.lock:
            current_time = time.time()
            self._refill_tokens(current_time)
            
            in_backoff = current_time < self.backoff_until
            backoff_remaining = max(0, self.backoff_until - current_time)
            
            return {
                'name': self.name,
                'tokens': self.tokens,
                'max_tokens': self.max_tokens,
                'requests_per_second': self.requests_per_second,
                'in_backoff': in_backoff,
                'backoff_remaining': backoff_remaining,
            }
    
    def wait_if_needed(self, timeout: float = 360.0):
        """Alias for acquire() for backwards compatibility."""
        return self.acquire(timeout=timeout)
    
    def handle_429_error(self):
        """Alias for report_rate_limit_error() for backwards compatibility."""
        return self.report_rate_limit_error()
    
    def reset_backoff(self):
        """Reset backoff state (called after successful request for backwards compatibility)."""
        with self.lock:
            self.backoff_duration = 5.0
    
    async def async_wait_if_needed(self, timeout: float = 360.0):
        """Async alias for acquire_async() for backwards compatibility."""
        return await self.acquire_async(timeout=timeout)


# Global rate limiter
_rate_limiters = {}
_rate_limiter_lock = threading.Lock()


def get_rate_limiter(name: str, requests_per_second: float, 
                    burst_size: Optional[int] = None) -> RateLimiterManager:
    """
    Get or create a global rate limiter instance.
    
    Args:
        name: Unique name for the rate limiter
        requests_per_second: Maximum request rate
        burst_size: Maximum burst size
        
    Returns:
        RateLimiterManager instance
    """
    with _rate_limiter_lock:
        if name not in _rate_limiters:
            _rate_limiters[name] = RateLimiterManager(
                requests_per_second=requests_per_second,
                burst_size=burst_size,
                name=name
            )
        return _rate_limiters[name]


# Pre-configured rate limiters for different services
def get_sec_rate_limiter() -> RateLimiterManager:
    """Get SEC API rate limiter"""
    return get_rate_limiter('SEC_API', requests_per_second=10, burst_size=1)


def get_company_rate_limiter() -> RateLimiterManager:
    """Get company ticker lookup rate limiter"""
    return get_rate_limiter('COMPANY_LOOKUP', requests_per_second=50.0, burst_size=100)


def get_yfinance_rate_limiter() -> RateLimiterManager:
    """Get Yahoo Finance rate limiter"""
    return get_rate_limiter('YFINANCE', requests_per_second=50.0, burst_size=100)


def get_hackernews_rate_limiter() -> RateLimiterManager:
    """Get Hacker News API rate limiter"""
    return get_rate_limiter('HACKER_NEWS', requests_per_second=50.0, burst_size=100)


def get_news_rate_limiter() -> RateLimiterManager:
    """Get generic news rate limiter"""
    return get_rate_limiter('NEWS_API', requests_per_second=50.0, burst_size=100)


class PerDomainRateLimiter:
    """
    Per-domain rate limiter for web scraping with anti-bot features.
    Each domain is rate-limited independently with jitter and random delays.
    """
    
    def __init__(self, base_delay: float = 7.0, jitter: float = 3.0, name: str = "per_domain_limiter"):
        """
        Initialize per-domain rate limiter.
        
        Args:
            base_delay: Base delay in seconds between requests to the same domain
            jitter: Maximum random jitter to add/subtract (±jitter seconds)
            name: Name for logging purposes
        """
        self.name = name
        self.base_delay = base_delay
        self.jitter = jitter
        
        # Domain state: domain -> last_access_time
        self.domain_state = {}
        self.lock = threading.Lock()
        
        # Rotate user agents to avoid getting blocked
        self.user_agents = [
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:121.0) Gecko/20100101 Firefox/121.0',
            'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.1 Safari/605.1.15',
            'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36 Edg/119.0.0.0',
        ]
        
        logger.info(f"Initialized {name}: {base_delay}s ±{jitter}s per domain")
    
    def _extract_domain(self, url: str) -> str:
        """Extract domain from URL."""
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            return parsed.netloc or parsed.path.split('/')[0]
        except Exception:
            return url
    
    def _calculate_delay(self) -> float:
        """Calculate delay with random jitter."""
        jitter_amount = random.uniform(-self.jitter, self.jitter)
        return max(1.0, self.base_delay + jitter_amount)  # Minimum 1 second
    
    def acquire(self, url: str, timeout: Optional[float] = 60.0) -> bool:
        """
        Acquire permission to make a request to the given URL's domain.
        
        Args:
            url: URL to extract domain from
            timeout: Maximum time to wait in seconds
        
        Returns:
            True if permission acquired, False if timeout
        """
        domain = self._extract_domain(url)
        start_time = time.time()
        
        while True:
            with self.lock:
                current_time = time.time()
                last_access = self.domain_state.get(domain, 0.0)
                
                # Calculate required delay for this domain
                required_delay = self._calculate_delay()
                time_since_last = current_time - last_access
                
                if time_since_last >= required_delay or last_access == 0.0:
                    # Grant access
                    self.domain_state[domain] = current_time
                    logger.debug(f"{self.name}: Acquired access to {domain}")
                    return True
                else:
                    # Check timeout
                    if timeout and (current_time - start_time) >= timeout:
                        logger.error(f"{self.name}: Timeout waiting for {domain}")
                        return False
            
            # Wait before checking again
            wait_time = min(0.5, 1.0 - time_since_last)
            time.sleep(wait_time)
    
    def get_random_user_agent(self) -> str:
        """Get a random user agent string for anti-bot bypass."""
        return random.choice(self.user_agents)
    
    def add_anti_bot_delay(self):
        """Add a small random delay to mimic human behavior."""
        # Random delay between 0.5 and 2.5 seconds
        delay = random.uniform(0.5, 2.5)
        time.sleep(delay)
    
    def reset_backoff(self):
        """Reset backoff (no-op for per-domain limiter, kept for compatibility)."""
        pass
    
    def report_rate_limit_error(self, url: Optional[str] = None):
        """Report a rate limit error for a specific domain."""
        if url:
            domain = self._extract_domain(url)
            with self.lock:
                # Add extra penalty: set last access to future time
                penalty = self.base_delay * 3  # 3x the normal delay
                self.domain_state[domain] = time.time() + penalty
                logger.warning(f"{self.name}: Rate limit error for {domain}, adding {penalty:.1f}s penalty")
        else:
            logger.warning(f"{self.name}: Rate limit error reported (no URL provided)")
    
    def get_stats(self) -> dict:
        """Get current rate limiter statistics."""
        with self.lock:
            return {
                'name': self.name,
                'base_delay': self.base_delay,
                'jitter': self.jitter,
                'tracked_domains': len(self.domain_state),
                'domains': list(self.domain_state.keys())[:10]
            }


# Global per-domain rate limiter for article extraction
_per_domain_limiter = None
_per_domain_limiter_lock = threading.Lock()


def get_article_extraction_rate_limiter() -> PerDomainRateLimiter:
    """Get per-domain article extraction rate limiter."""
    global _per_domain_limiter
    with _per_domain_limiter_lock:
        if _per_domain_limiter is None:
            _per_domain_limiter = PerDomainRateLimiter(
                base_delay=7.0,
                jitter=3.0,
                name='ARTICLE_EXTRACTION_PER_DOMAIN'
            )
        return _per_domain_limiter