import re
import html
import json
import time
import logging
import asyncio
import requests
import concurrent.futures

from datetime import datetime
from typing import Optional, Dict, Any, List
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry as URLRetry
from pathlib import Path

from .rate_limiter import get_sec_rate_limiter
from .sp500 import add_delisted_stock


logger = logging.getLogger(__name__)

# Create session for SEC requests to reuse connections
_SEC_SESSION = None

def _get_sec_session() -> requests.Session:
    """
    Get a persistent requests Session for SEC API calls.
    This reuses TCP connections and reduces DNS lookup overhead significantly.
    
    Returns:
        requests.Session configured with connection pooling
    """
    global _SEC_SESSION
    
    if _SEC_SESSION is not None:
        return _SEC_SESSION
    
    # Create session with connection pooling and HTTP connection pooling
    _SEC_SESSION = requests.Session()
    
    # Configure adapter with connection pooling
    adapter = HTTPAdapter(
        pool_connections=10,
        pool_maxsize=20,
        max_retries=URLRetry(total=0, connect=0, backoff_factor=0)
    )
    
    _SEC_SESSION.mount('https://', adapter)
    _SEC_SESSION.mount('http://', adapter)
    
    return _SEC_SESSION


def _requests_get_with_retry(url: str, **kwargs) -> Optional[requests.Response]:
    """
    Make HTTP GET request with retry logic for transient failures (DNS, timeouts).
    Uses persistent session to reuse TCP connections and reduce DNS overhead.
    
    Args:
        url: URL to request
        **kwargs: Additional arguments to pass to session.get
        
    Returns:
        Response object or None if all retries failed
    """
    session = _get_sec_session()
    max_retries = 3
    base_delay = 0.5
    backoff_factor = 2.0
    
    for attempt in range(max_retries):
        try:
            response = session.get(url, **kwargs)
            return response
        except requests.exceptions.ConnectionError as e:
            # DNS or connection errors - retry with backoff
            if attempt < max_retries - 1:
                delay = base_delay * (backoff_factor ** attempt)
                logger.debug(f"Connection error on {url} (attempt {attempt+1}/{max_retries}), retrying in {delay:.1f}s: {type(e).__name__}")
                time.sleep(delay)
            else:
                logger.error(f"Connection error on {url} after {max_retries} attempts: {e}")
                return None
        except requests.exceptions.Timeout as e:
            # Timeout errors - retry with backoff
            if attempt < max_retries - 1:
                delay = base_delay * (backoff_factor ** attempt)
                logger.debug(f"Timeout on {url} (attempt {attempt+1}/{max_retries}), retrying in {delay:.1f}s")
                time.sleep(delay)
            else:
                logger.error(f"Timeout on {url} after {max_retries} attempts: {e}")
                return None
        except Exception as e:
            # Other errors - don't retry, just return None
            logger.error(f"Error requesting {url}: {e}")
            return None
    
    return None


# In-memory cache for company tickers (loaded once per session)
_COMPANY_TICKERS_CACHE = None
_CACHE_REFRESH_ATTEMPTED = False  # Track if we've attempted to refresh this session

# In-memory cache for ticker lookups (CIK mappings)
_TICKER_LOOKUP_CACHE = None
TICKER_LOOKUP_CACHE_FILE = Path(__file__).parent / "ticker_cik_lookup.json"

# Track pending ticker lookups to batch-write (avoid file descriptor exhaustion)
_TICKER_LOOKUP_PENDING = {}
_TICKER_LOOKUP_DIRTY = False
_TICKER_LOOKUP_LOCK = None


def load_ticker_lookup_cache() -> Dict[str, Optional[str]]:
    """
    Load cached ticker -> CIK mappings from disk.
    
    Returns:
        Dictionary mapping ticker -> CIK (or None if lookup failed)
    """
    global _TICKER_LOOKUP_CACHE
    
    if _TICKER_LOOKUP_CACHE is not None:
        return _TICKER_LOOKUP_CACHE
    
    _TICKER_LOOKUP_CACHE = {}
    
    if TICKER_LOOKUP_CACHE_FILE.exists():
        try:
            with open(TICKER_LOOKUP_CACHE_FILE, 'r') as f:
                cache_data = json.load(f)
                if isinstance(cache_data, dict):
                    _TICKER_LOOKUP_CACHE = cache_data
                    logger.debug(f"Loaded {len(_TICKER_LOOKUP_CACHE)} ticker lookups from cache")
        except Exception as e:
            logger.debug(f"Could not load ticker lookup cache: {e}")
    
    return _TICKER_LOOKUP_CACHE


def save_ticker_lookup(ticker: str, cik: Optional[str]) -> None:
    """
    Queue a ticker -> CIK mapping to the cache (in-memory only).
    Changes are batched and saved to disk via flush_ticker_lookup_cache().
    This prevents file descriptor exhaustion from repeated writes.
    
    Args:
        ticker: Stock ticker symbol
        cik: CIK value (or None if lookup failed)
    """
    global _TICKER_LOOKUP_CACHE, _TICKER_LOOKUP_PENDING, _TICKER_LOOKUP_DIRTY
    
    if _TICKER_LOOKUP_CACHE is None:
        load_ticker_lookup_cache()
    
    _TICKER_LOOKUP_CACHE[ticker] = cik
    _TICKER_LOOKUP_PENDING[ticker] = cik
    _TICKER_LOOKUP_DIRTY = True
    logger.debug(f"Queued ticker '{ticker}' -> CIK '{cik}' (will save on flush)")


def flush_ticker_lookup_cache() -> None:
    """
    Write all pending ticker lookups to disk. Call this once at the end of pipeline execution.
    This prevents file descriptor exhaustion from repeated writes during processing.
    """
    global _TICKER_LOOKUP_CACHE, _TICKER_LOOKUP_PENDING, _TICKER_LOOKUP_DIRTY
    
    if not _TICKER_LOOKUP_DIRTY or _TICKER_LOOKUP_CACHE is None:
        return
    
    if not _TICKER_LOOKUP_PENDING:
        _TICKER_LOOKUP_DIRTY = False
        return
    
    try:
        with open(TICKER_LOOKUP_CACHE_FILE, 'w') as f:
            json.dump(_TICKER_LOOKUP_CACHE, f, indent=2)
        logger.debug(f"Flushed {len(_TICKER_LOOKUP_PENDING)} ticker lookups to cache")
        _TICKER_LOOKUP_PENDING.clear()
        _TICKER_LOOKUP_DIRTY = False
    except Exception as e:
        logger.warning(f"Could not save ticker lookup cache: {e}")


def close_sec_session() -> None:
    """
    Close the persistent SEC session to release file descriptors.
    Call this at the end of pipeline execution.
    """
    global _SEC_SESSION
    
    if _SEC_SESSION is not None:
        try:
            _SEC_SESSION.close()
            logger.debug("Closed SEC session")
        except Exception as e:
            logger.warning(f"Error closing SEC session: {e}")
        finally:
            _SEC_SESSION = None


def get_cached_cik(ticker: str) -> Optional[Optional[str]]:
    """
    Get CIK from cache if available.
    
    Returns:
        (found, cik) tuple where found is bool indicating if in cache
    """
    cache = load_ticker_lookup_cache()
    if ticker in cache:
        return True, cache[ticker]
    return False, None

def refresh_company_tickers_cache(headers: dict, force: bool = False) -> Optional[dict]:
    """
    Fetch the latest company tickers from SEC and update the local cache file.
    
    Args:
        headers: HTTP headers for SEC requests
        force: If True, refresh even if already attempted this session
        
    Returns:
        Dictionary of company tickers or None if failed
    """
    global _COMPANY_TICKERS_CACHE, _CACHE_REFRESH_ATTEMPTED
    
    # Skip if already attempted this session (unless forced)
    if _CACHE_REFRESH_ATTEMPTED and not force:
        return _COMPANY_TICKERS_CACHE
    
    _CACHE_REFRESH_ATTEMPTED = True
    local_file = Path(__file__).parent / "company_tickers.json"
    backup_file = Path(__file__).parent / "company_tickers.json.backup"
    
    try:
        # Fetch fresh data from SEC
        rate_limiter = get_sec_rate_limiter()
        rate_limiter.wait_if_needed()
        
        tickers_url = "https://www.sec.gov/files/company_tickers.json"
        response = _requests_get_with_retry(tickers_url, headers=headers, timeout=30)
        
        if response is None:
            print("Failed to refresh company_tickers.json: network error")
            return None
        
        if response.status_code == 429:
            rate_limiter.handle_429_error()
            print("Rate limited while fetching company tickers, using cached version")
            return None
        
        response.raise_for_status()
        new_data = response.json()
        
        # Backup existing file before replacing
        if local_file.exists():
            import shutil
            shutil.copy2(local_file, backup_file)
        
        # Save the new data
        with open(local_file, 'w') as f:
            json.dump(new_data, f, indent=2)
        
        _COMPANY_TICKERS_CACHE = new_data
        # Suppress verbose cache update messages in multiprocessing contexts
        logger.debug(f"Updated company_tickers.json cache with {len(new_data)} companies")
        rate_limiter.reset_backoff()
        return new_data
        
    except Exception as e:
        print(f"Failed to refresh company_tickers.json: {e}")
        # If backup exists and no cache, try to use backup
        if _COMPANY_TICKERS_CACHE is None and backup_file.exists():
            try:
                with open(backup_file, 'r') as f:
                    _COMPANY_TICKERS_CACHE = json.load(f)
                print("Using backup company_tickers.json file")
            except Exception as backup_error:
                print(f"Failed to load backup: {backup_error}")
        return None

def get_company_tickers(headers: dict, refresh: bool = False) -> Optional[dict]:
    """
    Get company tickers from local cache file (or SEC API if explicitly requested).
    
    For performance: company tickers should be refreshed ONCE at startup via
    explicit refresh_company_tickers_cache() call in __main__, then all worker
    processes read from the local file to avoid repeated SEC API calls.
    
    Args:
        headers: HTTP headers for SEC requests
        refresh: If True, refresh from SEC API. Default False uses local file only.
        
    Returns:
        Dictionary of company tickers or None if failed
    """
    global _COMPANY_TICKERS_CACHE
    
    # If we have cache and not refreshing, return it
    if _COMPANY_TICKERS_CACHE is not None and not refresh:
        return _COMPANY_TICKERS_CACHE
    
    # Try to refresh cache if this is first call or refresh requested
    if refresh and _COMPANY_TICKERS_CACHE is None:
        fresh_data = refresh_company_tickers_cache(headers)
        if fresh_data is not None:
            return fresh_data
    
    # Fall back to local file if refresh failed or not attempted
    if _COMPANY_TICKERS_CACHE is None:
        local_file = Path(__file__).parent / "company_tickers.json"
        if local_file.exists():
            try:
                with open(local_file, 'r') as f:
                    _COMPANY_TICKERS_CACHE = json.load(f)
                return _COMPANY_TICKERS_CACHE
            except (json.JSONDecodeError, IOError) as e:
                print(f"Failed to load local file: {e}")
        else:
            print(f"Local file not found: {local_file}")
    
    return _COMPANY_TICKERS_CACHE

def search_cik_by_ticker(ticker: str, headers: dict, target_date: Optional[str] = None) -> Optional[str]:
    """
    Search for a company's CIK by ticker using SEC's search API.
    This is a fallback when the ticker is not in the local company_tickers.json file.
    If multiple CIKs are found, prefer the one with recent filings before target_date.
    
    Args:
        ticker: The stock ticker symbol
        headers: HTTP headers for SEC requests
        target_date: Optional target date to filter CIKs with filings before this date
        
    Returns:
        CIK string (zero-padded to 10 digits) or None if not found
    """
    # Check cache first to avoid SEC API lookup
    found_in_cache, cached_cik = get_cached_cik(ticker)
    if found_in_cache:
        if cached_cik:
            logger.debug(f"Found cached CIK {cached_cik} for ticker '{ticker}'")
        return cached_cik
    
    try:
        # SEC's company search endpoint - use company name search for better results
        search_url = f"https://www.sec.gov/cgi-bin/browse-edgar"
        params = {
            'action': 'getcompany',
            'CIK': ticker,
            'type': '',
            'dateb': '',
            'owner': 'exclude',
            'output': 'atom',
            'count': '100'
        }
        
        rate_limiter = get_sec_rate_limiter()
        rate_limiter.wait_if_needed()
        
        response = _requests_get_with_retry(search_url, params=params, headers=headers, timeout=30)
        
        if response is None:
            logger.debug(f"Failed to search CIK for ticker '{ticker}': network error")
            save_ticker_lookup(ticker, None)
            return None
        
        if response.status_code == 200:
            # Parse the atom feed to extract all CIKs and their filing info
            import re
            
            # Extract all CIKs from filing URLs
            cik_matches = re.findall(r'CIK=(\d+)', response.text)
            unique_ciks = list(dict.fromkeys(cik_matches))
            
            if not unique_ciks:
                # Try direct CIK extraction from XML
                cik_match = re.search(r'<cik>(\d+)</cik>', response.text)
                if cik_match:
                    cik = cik_match.group(1).zfill(10)
                    print(f"Found CIK {cik} for ticker '{ticker}' via SEC search API")
                    save_ticker_lookup(ticker, cik)
                    return cik
                # Not found - cache None to avoid future lookups
                save_ticker_lookup(ticker, None)
                return None
            
            # If we have a target date, find CIK with filings before that date
            if target_date and len(unique_ciks) > 1:
                target_datetime = datetime.strptime(target_date, "%Y-%m-%d")
                
                # Extract filing entries with dates
                entries = re.findall(r'<entry>(.*?)</entry>', response.text, re.DOTALL)
                
                for entry in entries:
                    filing_date_match = re.search(r'<filing-date>(.*?)</filing-date>', entry)
                    cik_match = re.search(r'CIK=(\d+)', entry)
                    
                    if filing_date_match and cik_match:
                        filing_date_str = filing_date_match.group(1)
                        filing_datetime = datetime.strptime(filing_date_str, "%Y-%m-%d")
                        
                        # If this filing is before target date, use this CIK
                        if filing_datetime < target_datetime:
                            cik = cik_match.group(1).zfill(10)
                            print(f"Found CIK {cik} for ticker '{ticker}' with filings before {target_date}")
                            save_ticker_lookup(ticker, cik)
                            return cik
            
            # If no target date filtering or no match found, use the first/most common CIK
            # The most common CIK in the results is likely the main company
            from collections import Counter
            cik_counter = Counter(cik_matches)
            most_common_cik = cik_counter.most_common(1)[0][0] if cik_counter else unique_ciks[0]
            cik = most_common_cik.zfill(10)
            print(f"Found CIK {cik} for ticker '{ticker}' via SEC search API")
            save_ticker_lookup(ticker, cik)
            return cik
        
        # Cache None for not found to avoid further lookups
        save_ticker_lookup(ticker, None)
        return None
        
    except Exception as e:
        print(f"Error searching for CIK via SEC API: {e}")
        # On error, cache None to avoid retry
        save_ticker_lookup(ticker, None)
        return None


def download_all_sec_filings(company_ticker: str, form_type: str, target_date: Optional[str] = None, 
                            min_date: Optional[str] = None, max_filings: int = 10) -> List[tuple]:
    """
    Downloads ALL SEC filings of a specific form type for a given company within a date range.

    Args:
        company_ticker: The stock ticker of the company (e.g., "AAPL").
        form_type: The form type to download (e.g., "10-K", "10-Q", "8-K").
        target_date: The cutoff date in YYYY-MM-DD format. Only filings before this date will be considered.
        min_date: The minimum date in YYYY-MM-DD format. Only filings after this date will be considered.
        max_filings: Maximum number of filings to retrieve (default: 10)
    
    Returns:
        List of tuples: [(paragraphs_list, metadata_dict), ...] sorted by date (newest first)
        where paragraphs_list is a list of paragraph strings
    """
    headers = {'User-Agent': "z5218709 Student z5218709@student.unsw.edu.au"}

    target_datetime = None
    if target_date:
        try:
            target_datetime = datetime.strptime(target_date, "%Y-%m-%d")
        except ValueError:
            print(f"Error: Invalid date format '{target_date}'. Please use YYYY-MM-DD format.")
            return []
    
    min_datetime = None
    if min_date:
        try:
            min_datetime = datetime.strptime(min_date, "%Y-%m-%d")
        except ValueError:
            print(f"Error: Invalid min_date format '{min_date}'. Please use YYYY-MM-DD format.")
            return []

    companies = get_company_tickers(headers)
    if companies is None:
        return []

    # Normalize ticker for SEC lookup
    normalized_ticker = company_ticker.replace('.', '-').replace('/', '-').upper()
    
    cik = None
    for company in companies.values():
        if company['ticker'] == company_ticker.upper():
            cik = str(company['cik_str']).zfill(10)
            break
        if company['ticker'] == normalized_ticker:
            cik = str(company['cik_str']).zfill(10)
            break

    if not cik:
        print(f"Ticker '{company_ticker}' not found in company_tickers.json, searching SEC API...")
        cik = search_cik_by_ticker(company_ticker, headers, target_date)
        if not cik:
            cik = search_cik_by_ticker(normalized_ticker, headers, target_date)
    
    if not cik:
        print(f"Error: Could not find CIK for ticker '{company_ticker}'")
        return []

    submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    rate_limiter = get_sec_rate_limiter()
    
    try:
        rate_limiter.wait_if_needed()
        submissions_response = _requests_get_with_retry(submissions_url, headers=headers, timeout=30)
        
        if submissions_response is None:
            logger.error(f"Failed to fetch submissions for CIK {cik}: network error")
            return []
        
        if submissions_response.status_code == 429:
            rate_limiter.handle_429_error()
            return []
        
        submissions_response.raise_for_status()
        submissions_data = submissions_response.json()
        rate_limiter.reset_backoff()
    except Exception as e:
        logger.error(f"Failed to fetch submissions for CIK {cik}: {e}")
        return []

    # Collect all matching filings
    matching_filings = []
    
    def collect_filings(accession_numbers, form_types, primary_documents, filing_dates):
        for i in range(len(form_types)):
            if form_types[i].upper() == form_type.upper():
                filing_date_str = filing_dates[i]
                filing_datetime = datetime.strptime(filing_date_str, "%Y-%m-%d")
                
                # Check date range
                if target_datetime and filing_datetime >= target_datetime:
                    continue
                if min_datetime and filing_datetime <= min_datetime:
                    continue
                
                matching_filings.append({
                    'datetime': filing_datetime,
                    'date_str': filing_date_str,
                    'accession_number': accession_numbers[i],
                    'primary_document': primary_documents[i]
                })
    
    # Search recent filings
    recent_filings = submissions_data['filings']['recent']
    collect_filings(
        recent_filings['accessionNumber'],
        recent_filings['form'],
        recent_filings['primaryDocument'],
        recent_filings['filingDate']
    )
    
    # Sort by date (newest first) and limit
    matching_filings.sort(key=lambda x: x['datetime'], reverse=True)
    matching_filings = matching_filings[:max_filings]
    
    # Download each filing
    results = []
    for filing in matching_filings:
        accession_number_raw = filing['accession_number']
        primary_document_name = filing['primary_document']
        filing_date_str = filing['date_str']
        accession_number_clean = accession_number_raw.replace('-', '')
        cik_unpadded = str(int(cik))
        
        filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik_unpadded}/{accession_number_clean}/{primary_document_name}"
        
        try:
            rate_limiter.wait_if_needed()
            filing_response = _requests_get_with_retry(filing_url, headers=headers, timeout=30)
            
            if filing_response is None:
                logger.error(f"Failed to download {form_type} filed on {filing_date_str}: network error")
                continue
            
            if filing_response.status_code == 429:
                rate_limiter.handle_429_error()
                continue
            
            filing_response.raise_for_status()
            rate_limiter.reset_backoff()
            
            cleaned_content = clean_html_content(filing_response.text)
            paragraphs = split_into_paragraphs(cleaned_content)
            
            filing_metadata = {
                'filing_date': filing_date_str,
                'filing_url': filing_url,
                'accession_number': accession_number_raw,
                'form_type': form_type,
                'company_ticker': company_ticker,
                'cik': cik,
                'primary_document': primary_document_name
            }
            
            results.append((paragraphs, filing_metadata))
        except Exception as e:
            logger.error(f"Failed to download {form_type} filed on {filing_date_str}: {e}")
            continue
    
    return results


def download_latest_sec_filing(company_ticker: str, form_type: str, target_date: Optional[str] = None, save_to_file: bool = True):
    """
    Downloads the most recent SEC filing of a specific form type for a given company prior to a specified date.

    Args:
        company_ticker: The stock ticker of the company (e.g., "AAPL").
        form_type: The form type to download (e.g., "10-K", "10-Q", "8-K").
        target_date: The cutoff date in YYYY-MM-DD format. If provided, only filings 
                    submitted before this date will be considered. If None, gets the latest filing.
        save_to_file: Whether to save the content to a file. If False, returns the content as array of paragraphs.
    
    Returns:
        If save_to_file is True: filename of saved file or None if failed
        If save_to_file is False: tuple of (paragraphs_list, metadata_dict) or (None, None) if failed
    """
    headers = {'User-Agent': "z5218709 Student z5218709@student.unsw.edu.au"}

    target_datetime = None
    if target_date:
        try:
            target_datetime = datetime.strptime(target_date, "%Y-%m-%d")
        except ValueError:
            print(f"Error: Invalid date format '{target_date}'. Please use YYYY-MM-DD format.")
            return None if save_to_file else (None, None)

    companies = get_company_tickers(headers)
    if companies is None:
        return None if save_to_file else (None, None)

    # Normalize ticker for SEC lookup (replace . with -, / with -)
    normalized_ticker = company_ticker.replace('.', '-').replace('/', '-').upper()
    
    cik = None
    for company in companies.values():
        # Try exact match first
        if company['ticker'] == company_ticker.upper():
            cik = str(company['cik_str']).zfill(10)
            break
        # Try normalized match
        if company['ticker'] == normalized_ticker:
            cik = str(company['cik_str']).zfill(10)
            break

    # Fallback: Search SEC API directly if not found in local file
    if not cik:
        print(f"Ticker '{company_ticker}' not found in company_tickers.json, searching SEC API...")
        cik = search_cik_by_ticker(company_ticker, headers, target_date)
        
        if not cik:
            # Try normalized ticker as fallback
            cik = search_cik_by_ticker(normalized_ticker, headers, target_date)
    
    if not cik:
        print(f"Error: Could not find CIK for ticker '{company_ticker}' (tried '{company_ticker.upper()}', '{normalized_ticker}', and SEC API search)")
        # Mark as potentially delisted if CIK not found
        add_delisted_stock(company_ticker)
        return None if save_to_file else (None, None)

    submissions_url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    max_retries = 3
    retry_count = 0
    rate_limiter = get_sec_rate_limiter()
    
    while retry_count < max_retries:
        try:
            rate_limiter.wait_if_needed()
            submissions_response = _requests_get_with_retry(submissions_url, headers=headers, timeout=30) 
            
            if submissions_response is None:
                logger.error(f"Failed to fetch submissions for CIK {cik}: network error")
                retry_count += 1
                if retry_count < max_retries:
                    logger.info(f"Retrying submissions fetch ({retry_count}/{max_retries})...")
                    continue
                else:
                    logger.error(f"Max retries reached for submissions URL")
                    return None if save_to_file else (None, None)
            
            if submissions_response.status_code == 429:
                rate_limiter.handle_429_error()
                retry_count += 1
                if retry_count < max_retries:
                    print(f"Retrying ({retry_count}/{max_retries})...")
                    continue
                else:
                    print(f"Max retries reached for submissions URL")
                    return None if save_to_file else (None, None)
            
            submissions_response.raise_for_status()
            submissions_data = submissions_response.json()
            rate_limiter.reset_backoff()
            break
            
        except requests.exceptions.RequestException as e:
            print(f"Failed to fetch submissions for CIK {cik}: {e}")
            retry_count += 1
            if retry_count >= max_retries:
                return None if save_to_file else (None, None)
            time.sleep(2)

    filing_found = False
    best_filing = None
    
    # Helper function to search through a set of filings
    def search_filings(accession_numbers, form_types, primary_documents, filing_dates):
        nonlocal best_filing
        for i in range(len(form_types)):
            if form_types[i].upper() == form_type.upper():
                filing_date_str = filing_dates[i]
                filing_datetime = datetime.strptime(filing_date_str, "%Y-%m-%d")
                
                # Skip if after target date
                if target_datetime and filing_datetime >= target_datetime:
                    continue
                
                # Update best_filing if this is more recent
                if best_filing is None or filing_datetime > best_filing['datetime']:
                    best_filing = {
                        'datetime': filing_datetime,
                        'date_str': filing_date_str,
                        'accession_number': accession_numbers[i],
                        'primary_document': primary_documents[i]
                    }
    
    # Search recent filings first
    recent_filings = submissions_data['filings']['recent']
    search_filings(
        recent_filings['accessionNumber'],
        recent_filings['form'],
        recent_filings['primaryDocument'],
        recent_filings['filingDate']
    )
    
    # If no filing found in recent, search archived files
    if best_filing is None and 'files' in submissions_data['filings']:
        archived_files = submissions_data['filings']['files']
        
        for file_info in archived_files:
            file_name = file_info['name']
            
            # Check if this archive might contain relevant filings
            # (based on date range, if target_date is specified)
            if target_datetime:
                file_to = datetime.strptime(file_info['filingTo'], "%Y-%m-%d")
                # Skip archives that only contain filings after our target date
                if file_to >= target_datetime:
                    continue
            
            # Fetch the archived submissions file
            archive_url = f"https://data.sec.gov/submissions/{file_name}"
            
            max_retries = 3
            retry_count = 0
            rate_limiter = get_sec_rate_limiter()
            
            while retry_count < max_retries:
                try:
                    rate_limiter.wait_if_needed()
                    archive_response = _requests_get_with_retry(archive_url, headers=headers, timeout=30)
                    
                    if archive_response is None:
                        logger.error(f"Failed to fetch archive {file_name}: network error")
                        retry_count += 1
                        if retry_count < max_retries:
                            logger.info(f"Retrying archive fetch ({retry_count}/{max_retries})...")
                            continue
                        else:
                            logger.error(f"Max retries reached for archive {file_name}")
                            break
                    
                    if archive_response.status_code == 429:
                        rate_limiter.handle_429_error()
                        retry_count += 1
                        if retry_count < max_retries:
                            logger.info(f"Retrying archive fetch ({retry_count}/{max_retries})...")
                            continue
                        else:
                            logger.error(f"Max retries reached for archive {file_name}")
                            break
                    
                    archive_response.raise_for_status()
                    archive_data = archive_response.json()
                    rate_limiter.reset_backoff()
                    
                    # Search this archive
                    search_filings(
                        archive_data['accessionNumber'],
                        archive_data['form'],
                        archive_data['primaryDocument'],
                        archive_data['filingDate']
                    )
                    break
                    
                except requests.exceptions.RequestException as e:
                    print(f"Failed to fetch archive {file_name}: {e}")
                    retry_count += 1
                    if retry_count >= max_retries:
                        break
                    time.sleep(2)
            
            # If we found a filing, we can stop searching archives
            # (they're ordered from newest to oldest)
            if best_filing is not None:
                break
    
    # Now download the best filing we found
    if best_filing is not None:
        accession_number_raw = best_filing['accession_number']
        primary_document_name = best_filing['primary_document']
        filing_date_str = best_filing['date_str']
        accession_number_clean = accession_number_raw.replace('-', '')
        cik_unpadded = str(int(cik))
        
        filing_url = f"https://www.sec.gov/Archives/edgar/data/{cik_unpadded}/{accession_number_clean}/{primary_document_name}"

        max_retries = 3
        retry_count = 0
        filing_response = None
        rate_limiter = get_sec_rate_limiter()
        
        while retry_count < max_retries:
            try:
                rate_limiter.wait_if_needed()
                filing_response = _requests_get_with_retry(filing_url, headers=headers, timeout=30)
                
                if filing_response is None:
                    logger.error(f"Failed to download filing from {filing_url}: network error")
                    retry_count += 1
                    if retry_count < max_retries:
                        logger.info(f"Retrying filing download ({retry_count}/{max_retries})...")
                        continue
                    else:
                        logger.error(f"Max retries reached for filing download")
                        return None if save_to_file else (None, None)
                
                if filing_response.status_code == 429:
                    rate_limiter.handle_429_error()
                    retry_count += 1
                    if retry_count < max_retries:
                        logger.info(f"Retrying filing download ({retry_count}/{max_retries})...")
                        continue
                    else:
                        logger.error(f"Max retries reached for filing download")
                        return None if save_to_file else (None, None)
                
                filing_response.raise_for_status()
                rate_limiter.reset_backoff()
                break
                
            except requests.exceptions.RequestException as e:
                print(f"Failed to download filing from {filing_url}: {e}")
                retry_count += 1
                if retry_count >= max_retries:
                    return None if save_to_file else (None, None)
                time.sleep(2)
        
        if filing_response is None:
            print(f"Failed to download filing after retries")
            return None if save_to_file else (None, None)

        cleaned_content = clean_html_content(filing_response.text)
        paragraphs = split_into_paragraphs(cleaned_content)

        filing_metadata = {
            'filing_date': filing_date_str,
            'filing_url': filing_url,
            'accession_number': accession_number_raw,
            'form_type': form_type,
            'company_ticker': company_ticker,
            'cik': cik,
            'primary_document': primary_document_name
        }

        if save_to_file:
            output_filename = f"{company_ticker}_{form_type.replace('/', '-')}.txt"
            with open(output_filename, "w", encoding="utf-8") as f:
                # Save paragraphs separated by double newlines
                f.write('\n\n'.join(paragraphs))
            filing_found = True
            return output_filename
        else:
            filing_found = True
            return paragraphs, filing_metadata

    if not filing_found:
        if target_date:
            print(f"No {form_type} filings were found before {target_date} for {company_ticker}")
        else:
            print(f"No {form_type} filings were found in the recent submissions for {company_ticker}")
        
        return None if save_to_file else (None, None)

################################################################################
# Helper Functions
################################################################################

def split_into_paragraphs(text: str, min_words: int = 20) -> List[str]:
    """
    Split cleaned text into an array of paragraphs with a minimum word count.
    Paragraphs with fewer than min_words are merged with the next paragraph.
    
    Args:
        text: Cleaned text content
        min_words: Minimum number of words per paragraph (default: 20)
        
    Returns:
        List of paragraph strings, with empty paragraphs filtered out and 
        short paragraphs merged with subsequent ones
    """
    # Split by double newlines (paragraph breaks)
    raw_paragraphs = re.split(r'\n\n+', text)
    
    # Filter out empty paragraphs and strip whitespace
    raw_paragraphs = [p.strip() for p in raw_paragraphs if p.strip()]
    
    # Merge paragraphs that are too short
    merged_paragraphs = []
    accumulated = ""
    
    for i, para in enumerate(raw_paragraphs):
        # Add to accumulated text
        if accumulated:
            accumulated += "\n\n" + para
        else:
            accumulated = para
        
        # Count words in accumulated text
        word_count = len(accumulated.split())
        
        # If we meet the minimum OR it's the last paragraph, add it
        if word_count >= min_words or i == len(raw_paragraphs) - 1:
            merged_paragraphs.append(accumulated)
            accumulated = ""
    
    # Handle any remaining accumulated text (shouldn't happen, but just in case)
    if accumulated:
        if merged_paragraphs:
            # Merge with the last paragraph
            merged_paragraphs[-1] += "\n\n" + accumulated
        else:
            # No paragraphs yet, add it anyway
            merged_paragraphs.append(accumulated)
    
    return merged_paragraphs


def clean_html_content(html_content: str) -> str:
    """
    Clean HTML content by removing ng HTML entities to text,
    while preserving meaningful formatting based on tag types.
    
    Args:
        html_content: Raw HTML content from SEC filing
        
    Returns:
        Cleaned text content with appropriate formatting preserved
    """
    # First, decode HTML entities (like &#160; &#8217; &#8220; &#8221; etc.)
    cleaned_text = html.unescape(html_content)
    
    # Remove script and style elements completely
    cleaned_text = re.sub(r'<(script|style).*?</\1>', '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove XML declarations and XBRL namespace declarations
    cleaned_text = re.sub(r'<\?xml.*?\?>', '', cleaned_text, flags=re.DOTALL)
    cleaned_text = re.sub(r'<!DOCTYPE.*?>', '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
    
    # Remove XBRL context definitions and other XBRL metadata
    cleaned_text = re.sub(r'<(ix:header|ix:hidden|ix:resources).*?</\1>', '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
    cleaned_text = re.sub(r'<xbrli:context.*?</xbrli:context>', '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
    
    # Replace block-level elements with line breaks to preserve structure
    # Headings get extra spacing for emphasis
    cleaned_text = re.sub(r'<(h[1-6])\b[^>]*>', '\n\n', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'</(h[1-6])>', '\n\n', cleaned_text, flags=re.IGNORECASE)
    
    # Paragraphs, divs, and sections get single line breaks
    cleaned_text = re.sub(r'<(p|div|section|article)\b[^>]*>', '\n', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'</(p|div|section|article)>', '\n', cleaned_text, flags=re.IGNORECASE)
    
    # List items get line breaks with bullet points
    cleaned_text = re.sub(r'<(li)\b[^>]*>', '\n• ', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'</(li)>', '', cleaned_text, flags=re.IGNORECASE)
    
    # Definition list items
    cleaned_text = re.sub(r'<(dt)\b[^>]*>', '\n', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'</(dt)>', ':', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'<(dd)\b[^>]*>', ' ', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'</(dd)>', '\n', cleaned_text, flags=re.IGNORECASE)
    
    # Tables: handle table structure better - create CSV-like format
    # First mark tables for special processing
    cleaned_text = re.sub(r'<table\b[^>]*>', '\n\n__TABLE_START__\n', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'</table>', '\n__TABLE_END__\n\n', cleaned_text, flags=re.IGNORECASE)
    
    # Within tables, handle rows and cells more carefully
    cleaned_text = re.sub(r'<tr\b[^>]*>', '\n__ROW__', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'</tr>', '__ROW_END__', cleaned_text, flags=re.IGNORECASE)
    
    # For table cells, preserve content and add separators
    cleaned_text = re.sub(r'<(td|th)\b[^>]*>', '__CELL__', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'</(td|th)>', '__CELL_END__', cleaned_text, flags=re.IGNORECASE)
    
    # Line breaks for other block elements
    cleaned_text = re.sub(r'<(br|hr)\b[^>]*/?>', '\n', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'<(blockquote|pre|address)\b[^>]*>', '\n', cleaned_text, flags=re.IGNORECASE)
    cleaned_text = re.sub(r'</(blockquote|pre|address)>', '\n', cleaned_text, flags=re.IGNORECASE)
    
    # Remove remaining HTML/XML tags (mostly inline elements that don't affect formatting)
    # This handles both regular tags and self-closing tags
    cleaned_text = re.sub(r'<[^>]+>', '', cleaned_text)
    
    # Post-process tables to create proper formatting
    def process_table_content(match):
        table_content = match.group(1)
        
        # Process each row
        rows = re.split(r'__ROW__.*?__ROW_END__', table_content)
        row_matches = re.findall(r'__ROW__(.*?)__ROW_END__', table_content, re.DOTALL)
        
        processed_rows = []
        for row in row_matches:
            # Extract cell contents
            cells = re.findall(r'__CELL__(.*?)__CELL_END__', row, re.DOTALL)
            if cells:
                # Clean up each cell content
                clean_cells = []
                for cell in cells:
                    # Normalize whitespace within cells and strip
                    cell_clean = re.sub(r'\s+', ' ', cell.strip())
                    # Handle empty cells
                    if not cell_clean:
                        cell_clean = ""
                    clean_cells.append(cell_clean)
                
                # Combine adjacent dollar signs with numbers
                i = 0
                while i < len(clean_cells) - 1:
                    # Check if current cell is just a dollar sign and next cell is a number/parentheses
                    if (clean_cells[i] == '$' and 
                        clean_cells[i + 1] and 
                        re.match(r'^[\d,().-]+$', clean_cells[i + 1])):
                        # Combine dollar sign with number
                        clean_cells[i] = f"$ {clean_cells[i + 1]}"
                        clean_cells.pop(i + 1)
                    # Check if current cell ends with opening parenthesis and next is a number
                    elif (clean_cells[i] and clean_cells[i + 1] and
                          clean_cells[i].endswith('(') and 
                          re.match(r'^[\d,.-]+$', clean_cells[i + 1])):
                        # Combine opening parenthesis with number
                        clean_cells[i] = f"{clean_cells[i]}{clean_cells[i + 1]}"
                        clean_cells.pop(i + 1)
                    # Check if current cell is number and next cell is closing parenthesis
                    elif (clean_cells[i] and clean_cells[i + 1] == ')' and 
                          re.match(r'^[\d,.-]+$', clean_cells[i])):
                        # Combine number with closing parenthesis
                        clean_cells[i] = f"{clean_cells[i]}{clean_cells[i + 1]}"
                        clean_cells.pop(i + 1)
                    # Check if current cell is dollar sign followed by opening parenthesis and next is number
                    elif (clean_cells[i] == '$ (' and 
                          clean_cells[i + 1] and 
                          re.match(r'^[\d,.-]+$', clean_cells[i + 1])):
                        # Combine dollar with parenthesis and number
                        clean_cells[i] = f"$ ({clean_cells[i + 1]}"
                        clean_cells.pop(i + 1)
                    # Check if current cell is just opening parenthesis and next is a number
                    elif (clean_cells[i] == '(' and 
                          clean_cells[i + 1] and 
                          re.match(r'^[\d,.-]+$', clean_cells[i + 1])):
                        # Combine opening parenthesis with number
                        clean_cells[i] = f"({clean_cells[i + 1]}"
                        clean_cells.pop(i + 1)
                    else:
                        i += 1
                
                # Second pass: handle remaining parentheses combinations
                i = 0
                while i < len(clean_cells) - 1:
                    # Check for patterns like "$ (46,446" followed by " | )"
                    if (clean_cells[i] and clean_cells[i + 1] == ')' and
                        re.search(r'\$ \([\d,.-]+$', clean_cells[i])):
                        clean_cells[i] = f"{clean_cells[i]})"
                        clean_cells.pop(i + 1)
                    # Check for any number followed by closing parenthesis
                    elif (clean_cells[i] and clean_cells[i + 1] == ')' and
                          re.search(r'[\d,.-]+$', clean_cells[i])):
                        clean_cells[i] = f"{clean_cells[i]})"
                        clean_cells.pop(i + 1)
                    else:
                        i += 1
                
                # Third pass: combine percentages and parentheses with percentages
                i = 0
                while i < len(clean_cells) - 1:
                    # Check for number followed by %
                    if (clean_cells[i] and clean_cells[i + 1] == '%' and
                        re.search(r'[\d,.-]+\)?$', clean_cells[i])):
                        clean_cells[i] = f"{clean_cells[i]}%"
                        clean_cells.pop(i + 1)
                    # Check for ')' followed by '%'
                    elif (clean_cells[i] == ')' and clean_cells[i + 1] == '%'):
                        clean_cells[i] = ")%"
                        clean_cells.pop(i + 1)
                    else:
                        i += 1
                
                # Fourth pass: fix remaining parenthetical percentages
                i = 0
                while i < len(clean_cells):
                    if clean_cells[i] == ')%' and i > 0:
                        # Look back to find the number and add opening parenthesis
                        prev_cell = clean_cells[i-1]
                        if re.search(r'[\d,.-]+$', prev_cell):
                            clean_cells[i-1] = f"({prev_cell})%"
                            clean_cells.pop(i)
                            i -= 1
                    # Fix double parentheses
                    elif '((' in clean_cells[i]:
                        clean_cells[i] = clean_cells[i].replace('((', '(')
                    i += 1
                
                # Join cells with pipe separators, filtering out empty cells at the end
                while clean_cells and clean_cells[-1] == "":
                    clean_cells.pop()
                
                if clean_cells:  # Only add row if it has content
                    processed_rows.append(' | '.join(clean_cells))
        
        # Join rows with newlines
        return '\n'.join(processed_rows)
    
    # Apply table processing
    cleaned_text = re.sub(r'__TABLE_START__\n(.*?)\n__TABLE_END__', process_table_content, cleaned_text, flags=re.DOTALL)
    
    # Clean up any remaining table markers
    cleaned_text = re.sub(r'__TABLE_START__|__TABLE_END__|__ROW__|__ROW_END__|__CELL__|__CELL_END__', '', cleaned_text)
    
    # Final cleanup for table formatting
    # Remove multiple consecutive pipes with optional spaces
    cleaned_text = re.sub(r'(\s*\|\s*){3,}', ' | ', cleaned_text)
    # Clean up double pipes
    cleaned_text = re.sub(r'\s*\|\s*\|\s*', ' | ', cleaned_text)
    # Remove leading and trailing pipes on lines
    cleaned_text = re.sub(r'^\s*\|\s*', '', cleaned_text, flags=re.MULTILINE)
    cleaned_text = re.sub(r'\s*\|\s*$', '', cleaned_text, flags=re.MULTILINE)
    
    # Clean up excessive whitespace while preserving intentional line breaks
    # First, normalize multiple spaces to single spaces (but preserve line breaks and tabs)
    cleaned_text = re.sub(r'[ ]+', ' ', cleaned_text)
    
    # Clean up excessive line breaks (more than 2 consecutive)
    cleaned_text = re.sub(r'\n\s*\n\s*\n+', '\n\n', cleaned_text)
    
    # Clean up trailing spaces on lines
    cleaned_text = re.sub(r'[ ]+\n', '\n', cleaned_text)
    
    # Clean up empty lines with only spaces
    cleaned_text = re.sub(r'^\s*$', '', cleaned_text, flags=re.MULTILINE)
    
    # Strip leading and trailing whitespace
    cleaned_text = cleaned_text.strip()
    
    return cleaned_text

################################################################################
# Aggregation Function
################################################################################

async def get_financial_reports_async(ticker: str, target_date: str, token_limits: Optional[Dict[str, int]] = None) -> Dict[str, Any]:
    """
    Retrieve the latest financial reports (10-K, 10-Q, 8-K) before the target date asynchronously.
    Returns full content as arrays of paragraphs without filtering or truncation.
    
    Filtering logic:
    - 10-Q: Only include if dated after the latest 10-K
    - 8-K: Retrieve ALL 8-Ks dated after the latest 10-Q (up to 10 filings)
    - If no reports meet the criteria, they are excluded (set to None)

    Args:
        ticker: Stock ticker symbol
        target_date: Target date in YYYY-MM-DD format
        token_limits: Ignored (kept for backward compatibility)

    Returns:
        Dictionary mapping report types to dict with 'paragraphs' (list) and 'metadata' or None if unavailable.
        For 8-K, returns a list of reports instead of a single report.
    """
    reports: Dict[str, Any] = {}

    async def fetch_report(report_type: str, min_date: Optional[str] = None) -> tuple[str, Any]:
        try:
            loop = asyncio.get_event_loop()
            paragraphs, metadata = await loop.run_in_executor(None, download_latest_sec_filing, ticker, report_type, target_date, False)
            if paragraphs is not None and metadata is not None:
                # Check if report meets the minimum date requirement
                if min_date:
                    filing_date = metadata.get('filing_date')
                    if filing_date:
                        try:
                            filing_dt = datetime.strptime(filing_date, "%Y-%m-%d")
                            min_dt = datetime.strptime(min_date, "%Y-%m-%d")
                            
                            # Exclude if filing is not after the minimum date
                            if filing_dt <= min_dt:
                                return (report_type, None)
                        except ValueError as e:
                            print(f"Error parsing dates for {report_type}: {e}")
                            return (report_type, None)
                
                # Return paragraphs array
                return (report_type, {
                    'paragraphs': paragraphs,
                    'metadata': metadata
                })
            else:
                return (report_type, None)
        except Exception as e:
            print(f"Failed to download {report_type} report for {ticker}: {e}")
            return (report_type, None)
    
    async def fetch_all_8ks(min_date: Optional[str] = None) -> tuple[str, Any]:
        """Fetch ALL 8-Ks after the minimum date."""
        try:
            loop = asyncio.get_event_loop()
            # Download all 8-Ks in the date range
            all_8ks = await loop.run_in_executor(None, download_all_sec_filings, ticker, '8-K', target_date, min_date, 10)
            
            if not all_8ks:
                return ('8-K', None)
            
            # Process each 8-K - return paragraphs array
            processed_8ks = []
            
            for paragraphs, metadata in all_8ks:
                processed_8ks.append({
                    'paragraphs': paragraphs,
                    'metadata': metadata
                })
            
            print(f"Retrieved {len(processed_8ks)} 8-K report(s) for {ticker} after {min_date}")
            return ('8-K', processed_8ks if processed_8ks else None)
            
        except Exception as e:
            print(f"Failed to download 8-K reports for {ticker}: {e}")
            return ('8-K', None)
    
    # Step 1 & 2: Fetch 10-K and 10-Q IN PARALLEL (independent of each other)
    ten_k_task = fetch_report('10-K')
    ten_q_task = fetch_report('10-Q', min_date=None)
    
    ten_k_result, ten_q_result = await asyncio.gather(ten_k_task, ten_q_task, return_exceptions=True)
    
    # Handle results
    reports['10-K'] = ten_k_result[1] if not isinstance(ten_k_result, Exception) else None
    reports['10-Q'] = ten_q_result[1] if not isinstance(ten_q_result, Exception) else None
    
    # Step 3: Filter 10-Q if needed (only include if after 10-K)
    ten_k_date = None
    if reports['10-K'] and reports['10-K'].get('metadata'):
        ten_k_date = reports['10-K']['metadata'].get('filing_date')
    
    if reports['10-Q'] and ten_k_date:
        ten_q_date = reports['10-Q']['metadata'].get('filing_date')
        try:
            ten_q_dt = datetime.strptime(ten_q_date, "%Y-%m-%d")
            ten_k_dt = datetime.strptime(ten_k_date, "%Y-%m-%d")
            if ten_q_dt <= ten_k_dt:
                reports['10-Q'] = None  # Exclude if not after 10-K
        except ValueError:
            pass
    
    # Step 4: Fetch 8-Ks after 10-Q or 10-K (EARLY EXIT: skip if no 10-K found)
    if reports['10-K'] is None:
        # No 10-K found - skip 8-K fetching to save SEC API calls
        reports['8-K'] = None
    else:
        ten_q_date = None
        if reports['10-Q'] and reports['10-Q'].get('metadata'):
            ten_q_date = reports['10-Q']['metadata'].get('filing_date')
        
        min_date_for_8k = ten_q_date if ten_q_date else ten_k_date
        report_type, report_data = await fetch_all_8ks(min_date=min_date_for_8k)
        reports['8-K'] = report_data

    return reports


def get_financial_reports(ticker: str, target_date: str) -> Dict[str, Any]:
    """
    Synchronous wrapper for get_financial_reports_async.
    Retrieve the latest financial reports (10-K, 10-Q, 8-K) before the target date.
    Content is returned as arrays of paragraphs.

    Args:
        ticker: Stock ticker symbol
        target_date: Target date in YYYY-MM-DD format

    Returns:
        Dictionary mapping report types to dict with 'paragraphs' (list) and 'metadata' or None if unavailable.
    """
    # Run the async function in an event loop
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, get_financial_reports_async(ticker, target_date))
                # Add timeout to prevent hanging (8 minutes max for all 3 reports)
                return future.result(timeout=480)
        else:
            return loop.run_until_complete(get_financial_reports_async(ticker, target_date))
    except RuntimeError:
        # No event loop exists, create one
        return asyncio.run(get_financial_reports_async(ticker, target_date))
    except (TimeoutError, concurrent.futures.TimeoutError):
        print(f"Financial reports timeout (480s) for {ticker}")
        return {'10-K': None, '10-Q': None, '8-K': None}

if __name__ == "__main__":
    ticker_input = input("Enter the company ticker (e.g., AAPL): ")
    form_type_input = input("Enter the form type (e.g., 10-K, 10-Q, 8-K): ")
    target_date_input = input("Enter the target date (YYYY-MM-DD) or leave blank for latest: ")

    if not target_date_input.strip():
        target_date_input = None

    download_latest_sec_filing(
        company_ticker=ticker_input, 
        form_type=form_type_input, 
        target_date=target_date_input
    )