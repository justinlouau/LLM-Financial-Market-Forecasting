"""
training.py

Generates training dataset by augmenting existing JSON files with future stock data.
"""

import json
import time
import pandas as pd
import yfinance as yf

from pathlib import Path
from datetime import datetime, timedelta

def _series_to_list(series_or_df):
    if isinstance(series_or_df, pd.DataFrame):
        if series_or_df.shape[1] == 1:
            series_or_df = series_or_df.iloc[:, 0]
        else:
            series_or_df = series_or_df.mean(axis=1)
    return series_or_df.dropna().tolist()


def get_future_stock_data(ticker, last_date):
    # Get next 3 months of data from last_date in OHLCV format
    try:
        start_date_obj = datetime.strptime(last_date, "%Y-%m-%d")
        start = (start_date_obj + timedelta(days=1)).strftime("%Y-%m-%d")
        end = (start_date_obj + timedelta(days=90)).strftime("%Y-%m-%d")
        df = yf.download(ticker, start=start, end=end, auto_adjust=True, progress=False)
        
        if df.empty:
            print(f"Warning: No data found for {ticker} (possibly delisted).")
            return {'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}

        df = df.sort_index()

        if isinstance(df.columns, pd.MultiIndex):
            if ticker in df.columns.get_level_values(-1):
                df = df.xs(ticker, level=-1, axis=1)
            else:
                df = df.droplevel(-1, axis=1)
        
        # Extract OHLCV as separate arrays
        opens = _series_to_list(df['Open'])
        highs = _series_to_list(df['High'])
        lows = _series_to_list(df['Low'])
        closes = _series_to_list(df['Close'])
        volumes = _series_to_list(df['Volume'])
        
        return {
            'open': opens if opens else [],
            'high': highs if highs else [],
            'low': lows if lows else [],
            'close': closes if closes else [],
            'volume': volumes if volumes else []
        }
        
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        return {'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}

def process_files():
    input_dir = Path("output_filtered")
    output_dir = Path("output_training")
    output_dir.mkdir(exist_ok=True)
    
    files = list(input_dir.glob("*.json"))
    total_files = len(files)

    for i, file in enumerate(files):
        print(f"Processing {i+1}/{total_files}: {file.name}")
        
        with open(file, "r") as f:
            data = json.load(f)

        if "stock_data" in data:
            data["past_stock_data"] = data.pop("stock_data")

        ticker = data.get("ticker") or file.stem.split("_")[0]
        
        last_date = None
        if "date_range" in data and "end" in data["date_range"]:
            last_date = data["date_range"]["end"]

        if ticker and last_date:
            future_data = get_future_stock_data(ticker, last_date)
            data["future_stock_data"] = future_data
            time.sleep(1) 
        else:
            data["future_stock_data"] = {'open': [], 'high': [], 'low': [], 'close': [], 'volume': []}

        out_path = output_dir / file.name
        with open(out_path, "w") as f:
            json.dump(data, f, indent=2)

if __name__ == "__main__":
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    except:
        pass

    process_files()