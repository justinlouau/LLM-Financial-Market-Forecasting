"""
data.py

This script implements the data pipeline to fetch stock prices, financial reports, and news articles for companies for a given date range.
"""

import logging
import os
import json
from datetime import datetime, timedelta
from typing import Dict, Any
import asyncio
import gc

# Import data pipeline module functions
from modules.stock_prices import fetch_stock_data
from modules.financial_reports import get_financial_reports_async, refresh_company_tickers_cache, flush_ticker_lookup_cache, close_sec_session
from modules.news_aggregator import get_news_article_metadata
from modules.utils import resolve_company_ticker, get_company_name, clear_delisted_stocks_cache
from modules.sp500 import get_sp500_constituents, save_delisted_stocks_cache, close_wikipedia_session

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def run_sp500_pipeline(date: str):
    """Wrapper to run the async pipeline."""
    try:
        asyncio.run(run_sp500_pipeline_async(date))
    finally:
        close_sec_session()
        close_wikipedia_session()
        flush_ticker_lookup_cache()
        save_delisted_stocks_cache()

async def run_sp500_pipeline_async(date: str):
    """
    Run S&P 500 pipeline using asyncio with a semaphore for concurrency control.
    """
    logger.debug(f"[SP500] Starting S&P 500 pipeline for date {date}")
    base_output_dir = "output"
    max_concurrent = 3
    
    # Update company tickers from SEC website and retrieve S&P 500 constituents for the given date
    headers = {'User-Agent': "z5218709 Student z5218709@student.unsw.edu.au"}
    refresh_company_tickers_cache(headers, force=True)
    tickers = get_sp500_constituents(date)
    logger.debug(f"[SP500] Found {len(tickers) if tickers else 0} tickers")
    
    # Exit with error if no tickers found for the date
    if not tickers:
        logger.error(f"[SP500] Failed to get S&P 500 constituents for {date}")
        raise RuntimeError(f"Failed to get S&P 500 constituents for {date}")

    # Initialize results dictionary
    print(f"Processing {len(tickers)} tickers with {max_concurrent} concurrent tasks...\n")   
    results = {'successful': 0, 'failed': 0, 'processed': 0}

    # Create a semaphore to limit concurrent tasks
    semaphore = asyncio.Semaphore(max_concurrent)
    os.makedirs(base_output_dir, exist_ok=True)    
    
    async def process_ticker_with_semaphore(ticker):
        """Process a ticker while respecting the concurrency semaphore."""
        async with semaphore:
            try:
                output = await run_ticker_async(ticker, date, base_output_dir)
                return ticker, output
            except Exception as e:
                logger.error(f"[SP500] Error processing {ticker}: {e}")
                return ticker, {'ticker': ticker, 'error': str(e)}
    
    # Create tasks for all tickers
    tasks = [process_ticker_with_semaphore(ticker) for ticker in tickers]
    
    # Process results as they complete
    for coro in asyncio.as_completed(tasks):
        try:
            ticker, output = await asyncio.wait_for(coro, timeout=60)
            results['processed'] += 1
            
            # Check if stock_data has any values in the arrays
            has_stock_data = False
            if output.get('stock_data') and isinstance(output['stock_data'], dict):
                has_stock_data = any(output['stock_data'].get(key, []) for key in ['open', 'high', 'low', 'close', 'volume'])
            
            if has_stock_data or output.get('latest_10k'):
                results['successful'] += 1
                logger.info(f"✓ [{results['processed']}/{len(tickers)}] {ticker} - Success")
            else:
                results['failed'] += 1
                logger.error(f"✗ [{results['processed']}/{len(tickers)}] {ticker} - Missing data")
        except asyncio.TimeoutError:
            results['failed'] += 1
            results['processed'] += 1
            logger.error(f"✗ [{results['processed']}/{len(tickers)}] Ticker: Timeout after 60s")
        except Exception as e:
            results['failed'] += 1
            results['processed'] += 1
            logger.error(f"✗ [{results['processed']}/{len(tickers)}] Error: {e}")
        finally:
            gc.collect()
            await asyncio.sleep(0.01)
    
    # Final results summary
    print(f"\nS&P 500: Processed {results['processed']}, Successful: {results['successful']}, Failed: {results['failed']}")

async def run_ticker_async(ticker: str, date: str, base_output_dir: str = "output") -> Dict[str, Any]:
    """
    Run the data pipeline for a single ticker.
    
    Args:
        ticker: Stock ticker symbol (e.g., 'AAPL', 'MSFT')
        date: The reference date in 'YYYY-MM-DD' format
        base_output_dir: Base directory for output files
        save_output_file: Whether to save the output JSON file
    
    Returns:
        Dict containing the result for the ticker
    """
    try:
        logger.debug(f"[Ticker: {ticker}] Starting async pipeline")
        output = await run_pipeline_async(company=ticker, date=date)

        # Save output to JSON file
        ticker_safe = ticker.replace('.', '_').replace('/', '_')
        date_safe = date.replace('-', '')
        output_path = os.path.join(base_output_dir, f"{ticker_safe}_{date_safe}_data.json")
        os.makedirs(base_output_dir, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, ensure_ascii=False, indent=2, default=str)
        logger.debug(f"[Ticker: {ticker}] Output saved to {output_path}")
        
        # Cleanup and return
        gc.collect()
        return output
        
    except Exception as e:
        logger.error(f"[Ticker: {ticker}] Failed to process: {e}")
        gc.collect()  # Clean up on error too
        return {
            'ticker': ticker,
            'error': str(e)
        }

async def run_pipeline_async(company: str, date: str) -> Dict[str, Any]:
    """
    Retrieve data for a single company and date, including stock prices, financial reports, and news articles.
    
    Returns:
        - Company Name
        - Company Stock Ticker
        - Date Range
        - Stock Data (Array)
        - Latest 10K Report
        - Latest 10Q report since last 10K
        - Any 8K reports since last 10Q
        - Top 20 most recent hacker news articles about company (deduplicated)
    """
    logger.debug(f"[Pipeline] Starting async pipeline for {company} with target date {date}")
    result: Dict[str, Any] = {
        'company_name': company,
        'ticker': None,
        'date_range': None,
        'stock_data': {
            'open': [],
            'high': [],
            'low': [],
            'close': [],
            'volume': []
        },
        'latest_10k': None,
        'latest_10q': None,
        'eight_k_reports': [],
        'hacker_news_articles': []
    }

    try:
        logger.debug(f"[Pipeline] Calculating date range for {company}")
        # Always fetch only the past 365 days for stock data
        end_date = date
        start_date = (datetime.strptime(date, "%Y-%m-%d") - timedelta(days=365)).strftime("%Y-%m-%d")
        result['date_range'] = {'start': start_date, 'end': end_date}
        logger.debug(f"[Pipeline] Date range (fixed 365 days): {start_date} to {end_date}")

        logger.debug(f"[Pipeline] Resolving ticker for {company}")
        ticker = resolve_company_ticker(company)
        result['ticker'] = ticker
        logger.debug(f"[Pipeline] Ticker resolved: {ticker}")
        
        # Resolve the actual company name from the ticker
        actual_company_name = get_company_name(ticker)
        if actual_company_name:
            result['company_name'] = actual_company_name
            logger.debug(f"[Pipeline] Company name resolved: {actual_company_name}")
        
        # Handle special ticker formats (e.g., BRK.B -> BRK-B for yfinance)
        yf_ticker = ticker.replace('.', '-')
        logger.debug(f"[Pipeline] Yahoo Finance ticker: {yf_ticker}")

        loop = asyncio.get_event_loop()
        
        async def fetch_stock_async():
            logger.debug(f"[Pipeline] Starting stock data fetch for {yf_ticker}")
            return await loop.run_in_executor(None, fetch_stock_data, yf_ticker, start_date, end_date)
        
        async def fetch_news_async():
            logger.debug(f"[Pipeline] Starting news fetch for {ticker}")
            # Get top 20 HackerNews article metadata (title and publish_date only)
            return await loop.run_in_executor(
                None, 
                get_news_article_metadata,
                ticker,
                start_date, 
                end_date,
                20  # max_articles - top 20
            )
        
        logger.debug(f"[Pipeline] Starting parallel fetch: stock data, financial reports, news")
        # Fetch stock data, financial reports, and news articles in parallel
        stock_df, financial_reports, news_articles = await asyncio.gather(
            fetch_stock_async(),
            get_financial_reports_async(ticker, date, None),  # No token limits
            fetch_news_async(),
            return_exceptions=True
        )
        logger.debug(f"[Pipeline] Parallel fetch completed for {company}")
        
        # Process stock data
        if isinstance(stock_df, Exception):
            logger.error(f"Stock data retrieval failed for {ticker}: {stock_df}")
            result['stock_data'] = {'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}
        elif stock_df is None:
            logger.error(f"Stock data retrieval failed for {ticker}")
            result['stock_data'] = {'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}
        else:
            try:
                required_columns = {'Date', 'Open', 'High', 'Low', 'Close', 'Volume'}
                if required_columns.issubset(stock_df.columns):
                    recent_df = stock_df.sort_values('Date')
                    
                    # Extract each OHLCV component as separate arrays and truncate to 256 points
                    def truncate(arr):
                        return arr[-256:] if len(arr) > 256 else arr

                    opens = truncate(recent_df['Open'].dropna().tolist())
                    highs = truncate(recent_df['High'].dropna().tolist())
                    lows = truncate(recent_df['Low'].dropna().tolist())
                    closes = truncate(recent_df['Close'].dropna().tolist())
                    volumes = truncate(recent_df['Volume'].dropna().tolist())

                    result['stock_data'] = {
                        'open': opens if opens else [],
                        'high': highs if highs else [],
                        'low': lows if lows else [],
                        'close': closes if closes else [],
                        'volume': volumes if volumes else []
                    }
                else:
                    logger.error(f"Stock DataFrame missing required columns. Has: {stock_df.columns.tolist()}")
                    result['stock_data'] = {'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}
            except Exception as e:
                logger.error(f"Failed processing stock data: {e}")
                result['stock_data'] = {'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}

        # Process financial reports
        if isinstance(financial_reports, Exception):
            logger.error(f"Financial reports retrieval failed for {ticker}: {financial_reports}")
        elif financial_reports:
            # Latest 10-K
            if financial_reports.get('10-K'):
                result['latest_10k'] = financial_reports['10-K']
            
            # Latest 10-Q since last 10-K
            if financial_reports.get('10-Q'):
                result['latest_10q'] = financial_reports['10-Q']
            
            # Any 8-K reports since last 10-Q
            if financial_reports.get('8-K'):
                eight_k_data = financial_reports['8-K']
                if isinstance(eight_k_data, list):
                    result['eight_k_reports'] = eight_k_data
                elif eight_k_data:
                    result['eight_k_reports'] = [eight_k_data]

        # Process news articles (HackerNews top 20 - metadata only)
        if isinstance(news_articles, Exception):
            logger.error(f"News articles retrieval failed for {ticker}: {news_articles}")
        elif news_articles and isinstance(news_articles, dict):
            articles = news_articles.get('articles', [])
            # Already contains only title and publish_date from metadata function
            result['hacker_news_articles'] = articles[:20]

        logger.debug(f"[Pipeline] Successfully completed pipeline for {company}")

    except Exception as e:
        logger.exception(f"Critical pipeline error: {e}")

    return result

def generate_date_list(start_date_str, end_date_str, step=1):
  start_date = datetime.strptime(start_date_str, "%Y-%m-%d").date()
  end_date = datetime.strptime(end_date_str, "%Y-%m-%d").date()
  
  date_list = []
  current_date = start_date
  
  while current_date <= end_date:
    date_list.append(current_date.strftime("%Y-%m-%d"))
    current_date += timedelta(days=step)
    
  return date_list

if __name__ == "__main__":    
    logger.info("Starting data pipeline execution")

    # Set Date Range
    start_date = "2020-01-01"
    end_date = "2025-12-31"
    interval_days = 90
    logger.info(f"Date range: {start_date} to {end_date}")
    
    date_list = generate_date_list(start_date, end_date, step=interval_days)
    logger.info(f"Generated {len(date_list)} date checkpoints (step: {interval_days} days)")
    logger.debug(f"Date list: {date_list}")

    clear_delisted_stocks_cache()
    for i, date_str in enumerate(date_list, 1):
        logger.info(f"Processing checkpoint {i}/{len(date_list)}: {date_str}")
        print(f"Processing date: {date_str} ({i}/{len(date_list)})")
        run_sp500_pipeline(date_str)

    logger.info("Pipeline execution completed")
    clear_delisted_stocks_cache()
