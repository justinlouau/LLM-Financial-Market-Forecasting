"""
Data Pipeline Package

This package contains modules for retrieving various types of financial and news data:
- stock_prices: Stock price data from Nasdaq API
- financial_reports: SEC filing downloads (10-K, 10-Q, 8-K)
- news: News articles from multiple sources (Hacker News, Google News)
"""

# Suppress third-party library INFO/WARNING messages before importing modules
import logging
logging.getLogger('newspaper').setLevel(logging.CRITICAL)
logging.getLogger('urllib3').setLevel(logging.CRITICAL)
logging.getLogger('requests').setLevel(logging.CRITICAL)
logging.getLogger('feedparser').setLevel(logging.CRITICAL)
logging.getLogger('bs4').setLevel(logging.CRITICAL)

# Suppress newspaper's internal loggers
logging.getLogger('newspaper.article').setLevel(logging.CRITICAL)
logging.getLogger('newspaper.source').setLevel(logging.CRITICAL)
logging.getLogger('newspaper.network').setLevel(logging.CRITICAL)
logging.getLogger('newspaper.utils').setLevel(logging.CRITICAL)

from .stock_prices import fetch_stock_data
from .financial_reports import download_latest_sec_filing

__all__ = [
    'fetch_stock_data',
    'download_latest_sec_filing'
]