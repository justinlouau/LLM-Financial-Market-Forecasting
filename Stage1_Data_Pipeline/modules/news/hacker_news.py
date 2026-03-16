#!/usr/bin/env python3
"""
Hacker News Search Module
==========================

This module searches Hacker News for articles related to a company
and returns a list of article URLs using a prioritized search strategy
to minimize false positives:

1. Search by company name (most precise)
2. Fallback to combined ticker + company name (avoids ticker-only false positives)
3. Further fallback to industry or competitor searches (broadens context)
"""

import logging
import re
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry as URLRetry
from datetime import datetime
from typing import List, Dict, Any
import sys
import time
from pathlib import Path

# Add parent directory to path to import rate limiter
sys.path.insert(0, str(Path(__file__).parent.parent))
from rate_limiter import get_hackernews_rate_limiter

logger = logging.getLogger(__name__)

# Create session for Hacker News API
_HN_SESSION = None

def _get_hn_session() -> requests.Session:
    """
    Get requests Session for Hacker News API calls.

    Returns:
        requests.Session configured with connection pooling
    """
    global _HN_SESSION
    
    if _HN_SESSION is not None:
        return _HN_SESSION
    
    _HN_SESSION = requests.Session()
    
    # Configure adapter with connection pooling
    adapter = HTTPAdapter(
        pool_connections=5,
        pool_maxsize=10,
        max_retries=URLRetry(total=0, connect=0, backoff_factor=0)
    )
    
    _HN_SESSION.mount('https://', adapter)
    _HN_SESSION.mount('http://', adapter)
    
    return _HN_SESSION


def simplify_company_name(name: str) -> str:
    """
    Simplify company name by removing common suffixes.
    """
    # Common company suffixes to remove (case-insensitive)
    suffixes = [
        r'\s+Corporation$',
        r'\s+Inc\.?$',
        r'\s+Incorporated$',
        r'\s+Limited$',
        r'\s+Ltd\.?$',
        r'\s+L\.?L\.?C\.?$',
        r'\s+plc$',
        r'\s+Co\.?$',
        r'\s+Company$',
        r'\s+Group$',
        r',?\s+National Association$',
        r'\s+Technologies$',
        r'\s+Holdings?$',
        r'\s+Services?$',
        r'\s+Systems?$',
    ]
    
    simplified = name
    for suffix in suffixes:
        simplified = re.sub(suffix, '', simplified, flags=re.IGNORECASE)
    
    return simplified.strip()


def validate_date_range(start_date: str, end_date: str) -> tuple:
    """
    Validate date range and return datetime objects and timestamps.
    
    Args:
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        
    Returns:
        Tuple of (start_dt, end_dt, start_timestamp, end_timestamp)
    """
    try:
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')
        
        if start_dt > end_dt:
            raise ValueError("Start date must be before end date")
        
        if end_dt > datetime.now():
            logger.error("End date is in the future, using current date")
            end_dt = datetime.now()
        
        start_timestamp = int(start_dt.timestamp())
        end_timestamp = int(end_dt.timestamp())
        
        return start_dt, end_dt, start_timestamp, end_timestamp
        
    except ValueError as e:
        logger.error(f"Date validation error: {e}")
        raise e


def search_hackernews_stories(company_name: str, start_timestamp: int, 
                            end_timestamp: int, max_results: int = 50) -> List[Dict[str, Any]]:
    """
    Search Hacker News for stories mentioning the company within date range.
    Includes retry logic with exponential backoff for transient network errors.
    Uses persistent session to reuse TCP connections.
    
    Args:
        company_name: Company name to search for
        start_timestamp: Start date as Unix timestamp
        end_timestamp: End date as Unix timestamp  
        max_results: Maximum number of results to return
        
    Returns:
        List of HN story data dictionaries
    """
    base_url = "https://hn.algolia.com/api/v1/search"
    
    params = {
        'query': company_name,
        'tags': 'story',
        'numericFilters': f'created_at_i>={start_timestamp},created_at_i<={end_timestamp}',
        'hitsPerPage': min(max_results, 1000),
        'page': 0
    }
    
    # Retry configuration
    max_retries = 3
    base_delay = 1.0
    backoff_factor = 2.0
    
    session = _get_hn_session()
    
    for attempt in range(max_retries):
        try:
            # Enforce rate limit before making request with timeout
            rate_limiter = get_hackernews_rate_limiter()
            if not rate_limiter.acquire(timeout=15.0):
                logger.error(f"Rate limiter timeout for Hacker News, skipping")
                return []
            
            response = session.get(base_url, params=params, timeout=30)
            response.raise_for_status()
            
            data = response.json()
            stories = data.get('hits', [])
            
            # Reset backoff on success
            rate_limiter.reset_backoff()
            
            return stories
            
        except requests.exceptions.ConnectionError as e:
            # DNS or connection errors - retry with backoff
            if attempt < max_retries - 1:
                delay = base_delay * (backoff_factor ** attempt)
                logger.warning(f"Connection error fetching HN stories (attempt {attempt+1}/{max_retries}), retrying in {delay:.1f}s: {type(e).__name__}")
                time.sleep(delay)
            else:
                logger.error(f"Connection error fetching HN stories after {max_retries} attempts: {e}")
                return []
        except requests.exceptions.Timeout as e:
            # Timeout errors - retry with backoff
            if attempt < max_retries - 1:
                delay = base_delay * (backoff_factor ** attempt)
                logger.warning(f"Timeout fetching HN stories (attempt {attempt+1}/{max_retries}), retrying in {delay:.1f}s")
                time.sleep(delay)
            else:
                logger.error(f"Timeout fetching HN stories after {max_retries} attempts: {e}")
                return []
        except requests.exceptions.HTTPError as e:
            # HTTP errors - only retry on 5xx errors
            if 500 <= response.status_code < 600 and attempt < max_retries - 1:
                delay = base_delay * (backoff_factor ** attempt)
                logger.warning(f"HTTP {response.status_code} from HN API (attempt {attempt+1}/{max_retries}), retrying in {delay:.1f}s")
                time.sleep(delay)
            else:
                logger.error(f"Error fetching HN stories: {e}")
                # Report rate limit error if applicable
                if response.status_code == 429:
                    rate_limiter = get_hackernews_rate_limiter()
                    rate_limiter.report_rate_limit_error()
                return []
        except Exception as e:
            # Other errors - don't retry, just log and return
            logger.error(f"Error fetching HN stories: {e}")
            
            # Report rate limit error if applicable
            if 'rate' in str(e).lower() or '429' in str(e):
                rate_limiter = get_hackernews_rate_limiter()
                rate_limiter.report_rate_limit_error()
            
            return []
    
    return []


def get_hacker_news_urls(company_name: str, start_date: str, end_date: str,
                        max_results: int = 50, ticker: str = None, 
                        industry: str = None, competitors: List[str] = None) -> List[str]:
    """
    Get article URLs from Hacker News for a company within a date range.
    
    Uses a prioritized search strategy:
    1. Search by simplified company name (removes "Inc.", "Corporation", etc.)
    2. If insufficient results, search by ticker + simplified name
    3. If still insufficient, search by industry or simplified competitor names
    
    Args:
        company_name: Company name to search for (primary search term)
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        max_results: Maximum number of article URLs to return
        ticker: Optional stock ticker symbol (for fallback search)
        industry: Optional industry name (for fallback search)
        competitors: Optional list of competitor names (for fallback search)
        
    Returns:
        List of article URLs from Hacker News stories
    """
    
    try:
        # Validate dates and get timestamps
        _, _, start_timestamp, end_timestamp = validate_date_range(start_date, end_date)
        
        urls = []
        seen_urls = set()
        
        # Simplify the company name for better search results
        simple_name = simplify_company_name(company_name)
        
        # Helper function to extract URLs from stories
        def extract_urls_from_stories(stories: List[Dict[str, Any]]) -> List[str]:
            extracted = []
            for story in stories:
                points = story.get('points', 0)
                num_comments = story.get('num_comments', 0)
                url = story.get('url')
                
                # Only include stories with engagement and a URL
                if (points > 0 or num_comments > 0) and url and url not in seen_urls:
                    seen_urls.add(url)
                    extracted.append(url)
            return extracted
        
        # Priority 1: Search by simplified company name (most precise)
        stories = search_hackernews_stories(simple_name, start_timestamp, end_timestamp, max_results)
        urls.extend(extract_urls_from_stories(stories))
        
        # Priority 2: If insufficient results and ticker available, search by combined "ticker simple_name"
        # This helps avoid false positives from short tickers while still capturing ticker-based mentions
        if len(urls) < max_results * 0.5 and ticker:
            # Only use ticker if it's not too short (3+ characters) to avoid false positives
            if len(ticker) >= 3:
                remaining = max_results - len(urls)
                combined_query = f"{ticker} {simple_name}"
                ticker_stories = search_hackernews_stories(combined_query, start_timestamp, end_timestamp, remaining)
                new_urls = extract_urls_from_stories(ticker_stories)
                urls.extend(new_urls)
        
        # Priority 3: If still insufficient and industry/competitors available, broaden search
        if len(urls) < max_results * 0.3:
            remaining = max_results - len(urls)
            
            # Try industry search with multiple variations
            if industry:
                industry_stories = search_hackernews_stories(industry, start_timestamp, end_timestamp, remaining // 2)
                new_urls = extract_urls_from_stories(industry_stories)
                urls.extend(new_urls)
                
                # If industry is multi-word, try searching for just the main keyword
                # E.g., "specialty chemicals" -> try "chemicals", "software services" -> "software"
                if len(urls) < max_results * 0.3 and ' ' in industry:
                    # Extract the last significant word (usually the category)
                    words = industry.split()
                    # Skip common modifiers and use the main category word
                    skip_words = {'specialty', 'integrated', 'diversified', 'advanced', 'general', 'industrial', 'commercial', 'consumer'}
                    main_keywords = [w for w in words if w.lower() not in skip_words]
                    if main_keywords:
                        main_keyword = main_keywords[-1]  # Usually the last word is the main category
                        if main_keyword.lower() != industry.lower():
                            alt_industry_stories = search_hackernews_stories(main_keyword, start_timestamp, end_timestamp, remaining // 2)
                            new_urls = extract_urls_from_stories(alt_industry_stories)
                            urls.extend(new_urls)
            
            # Try competitor search if available (use simplified names)
            if competitors and len(urls) < max_results * 0.5:
                remaining = max_results - len(urls)
                # Simplify competitor names
                simple_competitors = [simplify_company_name(comp) for comp in competitors[:3]]
                for i, competitor in enumerate(simple_competitors):  # Limit to top 3 competitors
                    if len(urls) >= max_results:
                        break
                    competitor_stories = search_hackernews_stories(
                        competitor, start_timestamp, end_timestamp, remaining // len(simple_competitors)
                    )
                    new_urls = extract_urls_from_stories(competitor_stories)
                    urls.extend(new_urls)
        
        if not urls:
            logger.error(f"No Hacker News stories found for '{company_name}' with any search strategy")
            return []
        
        return urls[:max_results]
        
    except Exception as e:
        logger.error(f"Error getting Hacker News URLs for '{company_name}': {e}")
        return []


# # Backward compatibility - CLI interface
# if __name__ == "__main__":
#     def main():
#         """Simple CLI for testing the Hacker News URL extraction."""
#         print("=== Hacker News URL Extraction ===")
#         company = input("Company name: ").strip()
#         if not company:
#             print("Company name required")
#             return
        
#         start_date = input("Start date (YYYY-MM-DD): ").strip()
#         end_date = input("End date (YYYY-MM-DD): ").strip()
        
#         max_results_in = input("Max results (default 50): ").strip()
#         try:
#             max_results = int(max_results_in) if max_results_in else 50
#         except ValueError:
#             max_results = 50
        
#         try:
#             urls = get_hacker_news_urls(company, start_date, end_date, max_results)
#             print(f"\nFound {len(urls)} URLs:")
#             for i, url in enumerate(urls, 1):
#                 print(f"{i}. {url}")
#         except Exception as e:
#             print(f"Error: {e}")
    
#     main()
