import logging
import json
import os
from datetime import datetime, timedelta
from typing import Tuple, Optional
from threading import Lock

# Configure logging
logging.basicConfig(level=logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# In-memory cache for company tickers (thread-safe)
_COMPANY_TICKERS_CACHE = None
_CACHE_LOCK = Lock()


def _load_company_tickers_cache() -> Optional[dict]:
    """
    Load company tickers from file into memory cache (thread-safe).
    
    Returns:
        Dictionary of company tickers or None if failed
    """
    global _COMPANY_TICKERS_CACHE
    
    # Double-check locking pattern
    if _COMPANY_TICKERS_CACHE is not None:
        return _COMPANY_TICKERS_CACHE
    
    with _CACHE_LOCK:
        # Check again after acquiring lock
        if _COMPANY_TICKERS_CACHE is not None:
            return _COMPANY_TICKERS_CACHE
        
        try:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            company_tickers_path = os.path.join(current_dir, 'company_tickers.json')
            
            with open(company_tickers_path, 'r') as f:
                _COMPANY_TICKERS_CACHE = json.load(f)
            
            logger.debug(f"Loaded {len(_COMPANY_TICKERS_CACHE)} company tickers into memory cache")
            return _COMPANY_TICKERS_CACHE
        except Exception as e:
            logger.error(f"Error loading company tickers cache: {e}")
            return None


def get_company_name(ticker: str) -> Optional[str]:
    """
    Get the company name for a given ticker from in-memory cache.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Company name or None if not found
    """
    try:
        tickers_data = _load_company_tickers_cache()
        if not tickers_data:
            return None
        
        # Search for the ticker in the cached data
        for entry in tickers_data.values():
            if entry.get('ticker') == ticker.upper():
                return entry.get('title')
        
        logger.debug(f"Ticker '{ticker}' not found in company_tickers")
        return None
    except Exception as e:
        logger.error(f"Error getting company name for {ticker}: {e}")
        return None


def resolve_company_ticker(company: str) -> str:
    """
    Convert company name to stock ticker if needed.
    
    Args:
        company: Company name or ticker symbol
        
    Returns:
        Ticker symbol (or original input if already a ticker)
    """
    company_to_ticker = {
        'atlassian': 'TEAM',
        'apple': 'AAPL',
        'microsoft': 'MSFT',
        'tesla': 'TSLA',
        'google': 'GOOGL',
        'alphabet': 'GOOGL',
        'amazon': 'AMZN',
        'meta': 'META',
        'facebook': 'META',
        'netflix': 'NFLX',
        'nvidia': 'NVDA',
    }
    
    if company.isupper() and len(company) <= 5:
        return company
    
    company_lower = company.lower()
    if company_lower in company_to_ticker:
        return company_to_ticker[company_lower]
    
    logger.error(f"Could not resolve ticker for '{company}'")
    return company.upper() if len(company) <= 5 else company


def clear_delisted_stocks_cache(cache_path: Optional[str] = None) -> bool:
    """
    Delete the delisted stocks cache file (best-effort).
    If `cache_path` is None, defaults to `delisted_stocks.json` in the same `modules` directory.

    Returns:
        True if the file was removed, False otherwise.
    """
    try:
        if cache_path is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            cache_path = os.path.join(current_dir, 'delisted_stocks.json')

        if os.path.exists(cache_path):
            os.remove(cache_path)
            logger.info(f"Removed delisted stocks cache at {cache_path}")
            return True
        else:
            logger.debug(f"Delisted stocks cache not found at {cache_path}")
            return False
    except Exception as e:
        logger.warning(f"Failed to remove delisted stocks cache at {cache_path}: {e}")
        return False

