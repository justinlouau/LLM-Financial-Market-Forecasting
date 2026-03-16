#!/usr/bin/env python3
"""
News Aggregation Module
"""

import logging
import yfinance as yf

from typing import List, Dict, Any, Optional
from .news.hacker_news import get_hacker_news_urls

logger = logging.getLogger(__name__) 

def get_company_name_from_ticker(ticker: str) -> Optional[str]:
    """
    Get the company name from a stock ticker using yfinance.
    
    Args:
        ticker: Stock ticker symbol
        
    Returns:
        Company name or None if not found
    """
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        # Try longName first, then shortName
        company_name = info.get('longName', info.get('shortName', ''))
        
        if company_name:
            # Clean up the company name suffixes for better search results
            for suffix in [', Inc.', ' Inc.', ', Corp.', ' Corp.', ', Ltd.', ' Ltd.', 
                          ', LLC', ' LLC', ', LP', ' LP', ', PLC', ' PLC']:
                if company_name.endswith(suffix):
                    company_name = company_name[:-len(suffix)]
            
            return company_name.strip()
        return None
        
    except Exception:
        return None


def get_company_news_urls(ticker: str, start_date: str, end_date: str, 
                         max_articles: int = 50) -> List[Dict[str, str]]:
    """
    Get company-specific news article URLs from Hacker News.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        max_articles: Maximum number of URLs to return
        
    Returns:
        List of dicts with 'url' and 'source' keys
    """
    all_urls = []
    seen = set()
    
    # Get company name for better search results
    company_name = get_company_name_from_ticker(ticker)
    if company_name is None:
        company_name = ticker
    
    # Get Hacker News URLs
    try:
        hn_urls = get_hacker_news_urls(
            company_name=company_name,
            start_date=start_date,
            end_date=end_date,
            max_results=max_articles,
            ticker=ticker
        )
        
        for url in hn_urls:
            if url not in seen:
                seen.add(url)
                all_urls.append({'url': url, 'source': 'hacker_news', 'type': 'company'})
        
    except Exception as e:
        logger.error(f"Failed to get Hacker News URLs for {ticker}: {e}")
    
    return all_urls[:max_articles]

def get_news_article_metadata(ticker: str, start_date: str, end_date: str,
                             max_articles: int = 20) -> Dict[str, Any]:
    """
    Get news article metadata (title and publish date) without extracting full content.
    
    This is a lightweight version that only retrieves metadata from Hacker News API,
    avoiding the expensive article content extraction process.
    
    Args:
        ticker: Stock ticker symbol
        start_date: Start date in YYYY-MM-DD format
        end_date: End date in YYYY-MM-DD format
        max_articles: Maximum number of articles to return
        
    Returns:
        Dictionary containing:
            - articles: List of article metadata (title, publish_date, url)
            - total_articles: Total number of articles found
    """
    try:
        from datetime import datetime
        from .news.hacker_news import search_hackernews_stories, validate_date_range
        
        # Get company name and context
        company_name = get_company_name_from_ticker(ticker)
        if company_name is None:
            company_name = ticker
        
        # Validate dates and get timestamps
        _, _, start_timestamp, end_timestamp = validate_date_range(start_date, end_date)
        
        # Search Hacker News stories
        stories = search_hackernews_stories(company_name, start_timestamp, end_timestamp, max_articles)
        
        # Extract metadata from stories
        articles = []
        for story in stories:
            # Only include stories with engagement and a URL
            points = story.get('points', 0)
            num_comments = story.get('num_comments', 0)
            url = story.get('url')
            title = story.get('title', '')
            created_at = story.get('created_at')
            
            if (points > 0 or num_comments > 0) and url and title:
                # Convert created_at to ISO format date string
                publish_date = ''
                if created_at:
                    try:
                        dt = datetime.fromisoformat(created_at.replace('Z', '+00:00'))
                        publish_date = dt.strftime('%Y-%m-%d')
                    except:
                        pass
                
                articles.append({
                    'title': title,
                    'publish_date': publish_date,
                    'url': url
                })
        
        # Sort by date (most recent first) and limit to max_articles
        articles.sort(key=lambda x: x.get('publish_date', ''), reverse=True)
        articles = articles[:max_articles]
        
        return {
            'articles': articles,
            'total_articles': len(articles)
        }
        
    except Exception as e:
        logger.error(f"Error getting news article metadata for {ticker}: {e}", exc_info=True)
        return {
            'articles': [],
            'total_articles': 0,
            'error': str(e)
        }
