"""
Stock Price Data Fetcher using yfinance

Provides a simple function to fetch historical stock data using the yfinance library.

Usage:
    from src.data_pipeline.stock_prices import fetch_stock_data
    df = fetch_stock_data('AAPL', '2023-01-01', '2024-01-01')
"""

import pandas as pd
import logging
from typing import Optional
import yfinance as yf
from .rate_limiter import get_yfinance_rate_limiter
from .sp500 import add_delisted_stock

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def fetch_stock_data(ticker: str, from_date: str, to_date: str) -> Optional[pd.DataFrame]:
    """
    Fetch historical stock data for a given ticker and date range using yfinance.
    
    Args:
        ticker: Stock symbol (e.g., 'AAPL', 'MSFT')
        from_date: Start date in YYYY-MM-DD format
        to_date: End date in YYYY-MM-DD format
        
    Returns:
        DataFrame with cleaned stock data (Date, Open, High, Low, Close, Volume)
        or None if request fails
    """
    try:
        # Acquire rate limit token before making request with timeout
        rate_limiter = get_yfinance_rate_limiter()
        if not rate_limiter.acquire(timeout=15.0):
            logger.error(f"Rate limiter timeout for {ticker}, skipping stock data")
            return None
        
        stock = yf.Ticker(ticker)
        df = stock.history(start=from_date, end=to_date, auto_adjust=False)
        
        # Reset backoff on success
        rate_limiter.reset_backoff()
        
        if df is None or df.empty:
            logger.error(f"No data found for {ticker} in the specified date range")
            # Mark as potentially delisted so we skip it next time
            add_delisted_stock(ticker)
            return None
        
        df = df.reset_index()
        
        # Select only needed columns
        columns_to_keep = ['Date', 'Open', 'High', 'Low', 'Close', 'Volume']
        available_columns = [col for col in columns_to_keep if col in df.columns]
        
        return df[available_columns]
        
    except Exception as e:
        logger.error(f"Error fetching data for {ticker}: {e}")
        
        # Mark as potentially delisted if error looks like missing data
        if 'delisted' in str(e).lower() or 'timezone' in str(e).lower() or "doesn't exist" in str(e).lower():
            add_delisted_stock(ticker)
        
        # Report rate limit error if it looks like throttling
        if 'rate' in str(e).lower() or '429' in str(e):
            rate_limiter = get_yfinance_rate_limiter()
            rate_limiter.report_rate_limit_error()
        
        return None


# if __name__ == "__main__":
#     ticker = "AAPL"
#     start_date = "2023-01-01"
#     end_date = "2024-01-01"
    
#     df = fetch_stock_data(ticker, start_date, end_date)
