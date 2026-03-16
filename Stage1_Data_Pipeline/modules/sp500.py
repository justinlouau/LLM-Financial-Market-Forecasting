import requests
import logging
import json
import pandas as pd

from io import StringIO
from pathlib import Path
from threading import Lock
from datetime import datetime
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry as URLRetry

logger = logging.getLogger(__name__)

# Session for Wikipedia requests
_WIKIPEDIA_SESSION = None

def _get_wikipedia_session() -> requests.Session:
    """
    Get a requests Session for Wikipedia connections.
    
    Returns:
        requests.Session configured with connection pooling
    """
    global _WIKIPEDIA_SESSION
    
    if _WIKIPEDIA_SESSION is not None:
        return _WIKIPEDIA_SESSION
    
    # Create session with connection pooling
    _WIKIPEDIA_SESSION = requests.Session()
    
    # Configure adapter with connection pooling
    adapter = HTTPAdapter(
        pool_connections=10,
        pool_maxsize=20,
        max_retries=URLRetry(
            total=0,
            connect=0,
            backoff_factor=0
        )
    )
    
    _WIKIPEDIA_SESSION.mount('https://', adapter)
    _WIKIPEDIA_SESSION.mount('http://', adapter)
    
    return _WIKIPEDIA_SESSION

# Path to delisted stocks cache file
DELISTED_CACHE_FILE = Path(__file__).parent / "delisted_stocks.json"

# Thread lock for delisted stocks cache writes
_DELISTED_STOCKS_LOCK = Lock()

# Default known delisted stocks (core set that's always included)
DEFAULT_DELISTED_STOCKS = {
    'ANSS',  # Ansys
    'CTLT',  # Catalent
    'DFS',   # Discover Financial
    'HES',   # Hess Corporation
    'JNPR',  # Juniper Networks
    'MRO',   # Marathon Oil
    'PXD',   # Pioneer Natural Resources
}

# Runtime cache of delisted stocks (includes defaults + discovered at runtime)
_DELISTED_STOCKS_CACHE = None

# Track newly discovered delisted stocks during this session
_NEWLY_DELISTED_STOCKS = set()

# Flag to track if we have unsaved changes
_DELISTED_STOCKS_DIRTY = False

def load_delisted_stocks() -> set:
    """
    Load delisted stocks from cache file or return defaults.
    
    Returns:
        Set of delisted ticker symbols
    """
    global _DELISTED_STOCKS_CACHE
    
    if _DELISTED_STOCKS_CACHE is not None:
        return _DELISTED_STOCKS_CACHE
    
    # Start with defaults
    delisted = DEFAULT_DELISTED_STOCKS.copy()
    
    # Try to load from cache file
    if DELISTED_CACHE_FILE.exists():
        try:
            with open(DELISTED_CACHE_FILE, 'r') as f:
                cached_data = json.load(f)
                if isinstance(cached_data, dict) and 'delisted' in cached_data:
                    delisted.update(cached_data['delisted'])
                    logger.debug(f"Loaded {len(delisted)} delisted stocks from cache")
        except Exception as e:
            logger.debug(f"Could not load delisted stocks cache: {e}")
    
    _DELISTED_STOCKS_CACHE = delisted
    return delisted


def add_delisted_stock(ticker: str) -> None:
    """
    Add a ticker to the delisted stocks list (in-memory only).
    Changes are batched in memory and saved to disk periodically via save_delisted_stocks_cache().
    This prevents file descriptor exhaustion from repeated writes.
    
    Args:
        ticker: Stock ticker symbol
    """
    global _DELISTED_STOCKS_CACHE, _NEWLY_DELISTED_STOCKS, _DELISTED_STOCKS_DIRTY
    
    # Use lock to prevent concurrent access
    with _DELISTED_STOCKS_LOCK:
        # Load current set if not already loaded
        if _DELISTED_STOCKS_CACHE is None:
            load_delisted_stocks()
        
        # Add to set if not already there
        if ticker not in _DELISTED_STOCKS_CACHE:
            _DELISTED_STOCKS_CACHE.add(ticker)
            _NEWLY_DELISTED_STOCKS.add(ticker)
            _DELISTED_STOCKS_DIRTY = True
            logger.debug(f"Marked {ticker} as delisted (will save to cache on shutdown)")


def save_delisted_stocks_cache() -> None:
    """
    Write all delisted stocks to cache file. Call this once at the end of pipeline execution.
    This prevents file descriptor exhaustion from repeated writes during processing.
    """
    global _DELISTED_STOCKS_CACHE, _NEWLY_DELISTED_STOCKS, _DELISTED_STOCKS_DIRTY
    
    if not _DELISTED_STOCKS_DIRTY or _DELISTED_STOCKS_CACHE is None:
        return
    
    # Use lock to prevent concurrent writes
    with _DELISTED_STOCKS_LOCK:
        if not _NEWLY_DELISTED_STOCKS:
            _DELISTED_STOCKS_DIRTY = False
            return
        
        try:
            cache_data = {
                'delisted': sorted(list(_DELISTED_STOCKS_CACHE)),
                'last_updated': datetime.now().isoformat()
            }
            with open(DELISTED_CACHE_FILE, 'w') as f:
                json.dump(cache_data, f, indent=2)
            logger.debug(f"Saved {len(_NEWLY_DELISTED_STOCKS)} newly delisted stocks to cache")
            _NEWLY_DELISTED_STOCKS.clear()
            _DELISTED_STOCKS_DIRTY = False
        except Exception as e:
            logger.warning(f"Could not save delisted stocks cache: {e}")


def close_wikipedia_session() -> None:
    """
    Close the persistent Wikipedia session to release file descriptors.
    Call this at the end of pipeline execution.
    """
    global _WIKIPEDIA_SESSION
    
    if _WIKIPEDIA_SESSION is not None:
        try:
            _WIKIPEDIA_SESSION.close()
            logger.debug("Closed Wikipedia session")
        except Exception as e:
            logger.warning(f"Error closing Wikipedia session: {e}")
        finally:
            _WIKIPEDIA_SESSION = None


def get_delisted_stocks() -> set:
    """Get the current set of delisted stocks (thread-safe getter)."""
    if _DELISTED_STOCKS_CACHE is None:
        return load_delisted_stocks()
    return _DELISTED_STOCKS_CACHE

def get_sp500_constituents(date_str, exclude_delisted=True):
    """
    Gets the S&P 500 constituents for a specific date.

    Args:
        date_str: The date in 'YYYY-MM-DD' format.
        exclude_delisted: If True, exclude stocks that have been delisted and 
                         cannot be retrieved via yfinance (default: True)

    Returns:
        A list of S&P 500 tickers for the given date, or None if an error occurs.
    """
    try:
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'}
        
        session = _get_wikipedia_session()
        response = session.get(url, headers=headers, timeout=30)
        response.raise_for_status() 

        # Use pandas to read the HTML tables from the response text
        tables = pd.read_html(StringIO(response.text))
        
        # The Wikipedia page has exactly 2 tables:
        # Table 0: Current constituents
        # Table 1: Historical changes
        if len(tables) < 2:
            raise ValueError("Expected at least 2 tables on Wikipedia page")
        
        current_constituents_df = tables[0]
        changes_df = tables[1]
        
        current_tickers = set(current_constituents_df['Symbol'])

        # Handle multi-level columns in changes table
        if isinstance(changes_df.columns, pd.MultiIndex):
            changes_df.columns = changes_df.columns.droplevel(0)
        
        changes_df.columns = ['Date', 'Added Ticker', 'Added Security', 'Removed Ticker', 'Removed Security', 'Reason']
        changes_df['Date'] = pd.to_datetime(changes_df['Date'], errors='coerce')
        changes_df.dropna(subset=['Date'], inplace=True)

        target_date = datetime.strptime(date_str, '%Y-%m-%d')
        historical_tickers = current_tickers.copy()

        for _, row in changes_df.sort_values('Date', ascending=False).iterrows():
            change_date = row['Date']
            if change_date > target_date:
                if isinstance(row['Added Ticker'], str):
                    historical_tickers.discard(row['Added Ticker'])
                if isinstance(row['Removed Ticker'], str):
                    historical_tickers.add(row['Removed Ticker'])

        # Filter out any empty strings or invalid tickers
        valid_tickers = [t for t in historical_tickers if t and isinstance(t, str) and len(t) > 0]
        
        # Filter out delisted stocks if requested
        if exclude_delisted:
            before_count = len(valid_tickers)
            delisted = get_delisted_stocks()
            valid_tickers = [t for t in valid_tickers if t not in delisted]
            excluded_count = before_count - len(valid_tickers)
            if excluded_count > 0:
                logger.info(f"Excluded {excluded_count} delisted stocks for {date_str} (delisted: {len(delisted)} total)")
        
        return sorted(list(valid_tickers))

    except Exception as e:
        logger.error(f"An error occurred fetching S&P 500 constituents: {e}")
        return None