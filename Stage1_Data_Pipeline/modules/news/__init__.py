"""
News submodule for the data pipeline.

This submodule contains components for fetching and extracting news articles
from multiple sources.
"""

from .hacker_news import get_hacker_news_urls

__all__ = [
    'get_hacker_news_urls',
]
