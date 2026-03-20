import os
import json
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import numpy as np
from scipy.stats import norm
from sklearn.metrics import f1_score, matthews_corrcoef
import argparse
import time
import random
import re
import warnings

# Suppress pandas performance warnings that are not critical for this script
warnings.simplefilter(action='ignore', category=pd.errors.PerformanceWarning)

# ================= CONFIGURATION =================
# Thesis Parameters
CONFIDENCE_THRESHOLD = 0.55   # Confidence required to enter a trade
INITIAL_CAPITAL = 100000      # $100k simulated portfolio
RISK_FREE_RATE = 0.04         # 4% annual risk-free rate
TRANSACTION_COST = 0.0005     # 5 bps round-trip trading cost/slippage penalty
UNCHANGED_THRESHOLD = 0.005   # 0.5% threshold to define "Unchanged" class

# Simple in-memory cache for yfinance results.
_CACHE_FILE = os.path.join(os.path.dirname(__file__), ".yfinance_market_data_cache.json")
_MARKET_DATA_CACHE = {}
_BULK_MARKET_DATA_CACHE = {}


def _load_market_data_cache():
    """Load the persisted cache from disk."""
    global _MARKET_DATA_CACHE
    try:
        if os.path.exists(_CACHE_FILE):
            with open(_CACHE_FILE, "r", encoding="utf-8") as f:
                _MARKET_DATA_CACHE = json.load(f)
    except Exception:
        _MARKET_DATA_CACHE = {}


def _save_market_data_cache():
    """Persist the cache to disk so it survives between runs."""
    try:
        tmp_path = _CACHE_FILE + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(_MARKET_DATA_CACHE, f)
        os.replace(tmp_path, _CACHE_FILE)
    except Exception:
        pass

_load_market_data_cache()


def _is_rate_limit_error(exc):
    """Detect common rate-limit / Too Many Requests signals."""
    msg = str(exc).lower()
    if "429" in msg or "rate limit" in msg or "too many requests" in msg:
        return True
    if hasattr(exc, "response") and getattr(exc.response, "status_code", None) == 429:
        return True
    return False

# =================================================

def parse_filename(filename):
    """Extracts ticker and date from filename like: AEP_20240828_data_run_1"""
    try:
        match = re.search(r'(\d{8})', filename)
        if not match: return None, None
        date_str = match.group(1)
        ticker_part = filename[:match.start()]
        ticker = ticker_part.rstrip('_')
        return ticker, date_str
    except Exception:
        return None, None

def get_market_data(ticker, date_str, max_retries=5, initial_delay=1.0):
    """Fetch historical price data for a given ticker + date."""
    yf_ticker = ticker.replace("_", "-").replace(" ", "-").replace(".", "-").upper()
    cache_key = f"{yf_ticker}|{date_str}"
    if cache_key in _MARKET_DATA_CACHE:
        return _MARKET_DATA_CACHE[cache_key]

    target_date = datetime.strptime(date_str, "%Y%m%d")
    start_date = target_date - timedelta(days=40)
    end_date = target_date + timedelta(days=10)

    attempt = 0
    delay = initial_delay
    while attempt < max_retries:
        attempt += 1
        try:
            dat = yf.Ticker(yf_ticker)
            df = dat.history(start=start_date, end=end_date, auto_adjust=False)

            if df is None or df.empty:
                raise RuntimeError("No data returned from yfinance")

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            df['Returns'] = df['Close'].pct_change()
            df_tz_naive = df.copy()
            df_tz_naive.index = df_tz_naive.index.tz_localize(None)

            past_dates = df_tz_naive[df_tz_naive.index <= target_date]
            if past_dates.empty: return None

            base_loc = df_tz_naive.index.get_loc(past_dates.index[-1])
            if base_loc + 1 >= len(df): return None

            base_day = df.iloc[base_loc]
            next_day = df.iloc[base_loc + 1]

            if 'Stock Splits' in df.columns and next_day['Stock Splits'] != 0:
                return None

            result = {
                'base_date': str(df.index[base_loc].date()),
                'base_close': float(base_day['Close']),
                'actual_next_date': str(df.index[base_loc + 1].date()),
                'actual_next_open': float(next_day['Open']),
                'actual_next_close': float(next_day['Close']),
            }
            _MARKET_DATA_CACHE[cache_key] = result
            _save_market_data_cache()
            return result
        except Exception as e:
            if attempt >= max_retries: return None
            time.sleep(delay * (1 + random.uniform(0, 0.5)))
            delay = min(delay * 2, 60)
    return None

def get_bulk_market_data(tickers, start_date, end_date):
    """Efficiently fetches and caches bulk market data for multiple tickers."""
    global _BULK_MARKET_DATA_CACHE
    tickers = sorted(list(set(tickers)))
    cache_key = f"{','.join(tickers)}|{start_date.strftime('%Y%m%d')}|{end_date.strftime('%Y%m%d')}"
    if cache_key in _BULK_MARKET_DATA_CACHE:
        return _BULK_MARKET_DATA_CACHE[cache_key]
    
    try:
        df = yf.download(tickers, start=start_date, end=end_date, auto_adjust=False, progress=False)
        _BULK_MARKET_DATA_CACHE[cache_key] = df
        return df
    except Exception as e:
        print(f"Warning: Bulk data download failed: {e}")
        return pd.DataFrame()

def calculate_multiclass_brier(probs, outcome_one_hot):
    return np.sum((probs - outcome_one_hot) ** 2)

def calculate_rps(probs, actual_dir):
    p_up, p_down, p_unchanged = probs[0], probs[1], probs[2]
    cp1 = p_down
    cp2 = p_down + p_unchanged
    
    if actual_dir == -1: ca1, ca2 = 1.0, 1.0
    elif actual_dir == 0: ca1, ca2 = 0.0, 1.0
    else: ca1, ca2 = 0.0, 0.0
    return (cp1 - ca1)**2 + (cp2 - ca2)**2

def benchmark_predictions(folder_path):
    """Processes all JSON prediction files in a given folder."""
    results = []
    files = [f for f in os.listdir(folder_path) if f.endswith('.json')]
    print(f"Found {len(files)} files in '{os.path.basename(folder_path)}'. Processing...")

    for i, filename in enumerate(files):
        if i > 0 and i % 500 == 0:
            print(f"  ...processed {i}/{len(files)} files...")

        ticker, date_str = parse_filename(filename)
        if not ticker: continue
        
        pred_price, pred_direction, probs = None, None, np.array([np.nan, np.nan, np.nan])
        
        try:
            with open(os.path.join(folder_path, filename), 'r') as f:
                data = json.load(f)
            
            output = data.get('output', {})
            if isinstance(output, dict):
                try: pred_price = float(output.get('next_day_closing_price'))
                except (ValueError, TypeError): pass
                try: pred_direction = int(output.get('next_day_direction'))
                except (ValueError, TypeError): pass
                
                forecast = output.get('forecast', {})
                if isinstance(forecast, dict):
                    prob_up = forecast.get('up', 0.0)
                    prob_down = forecast.get('down', 0.0)
                    prob_unchanged = forecast.get('unchanged', 0.0)
                    total_prob = prob_up + prob_down + prob_unchanged
                    if total_prob > 0:
                        probs = np.array([prob_up, prob_down, prob_unchanged]) / total_prob
        except Exception:
            pass 

        market_data = get_market_data(ticker, date_str)
        time.sleep(0.02) # Small delay to respect API limits
        
        if not market_data: continue

        actual_price = market_data['actual_next_close']
        base_price = market_data['base_close']
        pct_change = (actual_price - base_price) / base_price
        
        if abs(pct_change) <= UNCHANGED_THRESHOLD: actual_dir, outcome_vector = 0, np.array([0, 0, 1])
        elif pct_change > 0: actual_dir, outcome_vector = 1, np.array([1, 0, 0])
        else: actual_dir, outcome_vector = -1, np.array([0, 1, 0])

        results.append({
            'Ticker': ticker,
            'Date': market_data.get('actual_next_date'),
            'Base_Price': base_price,
            'Actual_Price': actual_price,
            'Pred_Price': pred_price,
            'Actual_Return': (actual_price - market_data['actual_next_open']) / market_data['actual_next_open'],
            'Pred_Direction': pred_direction,
            'Actual_Direction': actual_dir,
            'Prob_Up': probs[0], 'Prob_Down': probs[1],
            'Brier_Score': calculate_multiclass_brier(probs, outcome_vector) if not np.isnan(probs).any() else np.nan,
            'RPS': calculate_rps(probs, actual_dir) if not np.isnan(probs).any() else np.nan,
            'Abs_Error_Pct': abs((pred_price - actual_price) / actual_price) if pred_price is not None else np.nan
        })

    return pd.DataFrame(results)

def run_trading_simulation(df, full_date_range):
    """Runs a trading simulation over a fixed date range for fair comparison."""
    if df.empty:
        return 0.0, 0.0, 0.0, pd.DataFrame({'Date': full_date_range.date, 'Cumulative_Strategy': INITIAL_CAPITAL})

    df['Date'] = pd.to_datetime(df['Date']).dt.date
    
    if 'Prob_Up' in df.columns and 'Prob_Down' in df.columns:
        df['Position'] = 0
        df.loc[df['Prob_Up'] > CONFIDENCE_THRESHOLD, 'Position'] = 1
        df.loc[df['Prob_Down'] > CONFIDENCE_THRESHOLD, 'Position'] = -1
    else:
        if 'Position' in df.columns:
            df['Position'] = df['Position'].fillna(0).astype(int)
        else:
            df['Position'] = 0

    df['Trade_Return'] = (df['Position'] * df['Actual_Return']) - (abs(df['Position']) * TRANSACTION_COST)
    
    daily_trade_counts = df.groupby('Date')['Ticker'].transform('count')
    df['Portfolio_Contribution'] = df['Trade_Return'] / daily_trade_counts
    
    daily_returns = df.groupby('Date')['Portfolio_Contribution'].sum().reset_index()
    daily_returns.rename(columns={'Portfolio_Contribution': 'Daily_Return'}, inplace=True)
    
    all_dates_df = pd.DataFrame({'Date': full_date_range.date})
    daily_df = pd.merge(all_dates_df, daily_returns, on='Date', how='left')
    
    daily_risk_free = (1 + RISK_FREE_RATE) ** (1/252) - 1
    daily_df['Daily_Return'] = daily_df['Daily_Return'].fillna(daily_risk_free)
    daily_df = daily_df.sort_values(by='Date')
    
    daily_df['Cumulative_Strategy'] = (1 + daily_df['Daily_Return']).cumprod() * INITIAL_CAPITAL
    
    total_return = (daily_df['Cumulative_Strategy'].iloc[-1] / INITIAL_CAPITAL) - 1
    years = len(full_date_range) / 252.0
    annual_return = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0.0
    
    excess_returns = daily_df['Daily_Return'] - daily_risk_free
    std_excess = excess_returns.std()
    
    sharpe = (excess_returns.mean() / std_excess) * np.sqrt(252) if std_excess > 0 else 0.0

    daily_df['High_Water_Mark'] = daily_df['Cumulative_Strategy'].cummax()
    daily_df['Drawdown'] = (daily_df['Cumulative_Strategy'] / daily_df['High_Water_Mark']) - 1
    max_drawdown = daily_df['Drawdown'].min()

    return annual_return, sharpe, max_drawdown, df

def generate_baseline_df(all_tickers, full_date_range, strategy):
    """Generates a DataFrame for Random or BRSF baseline strategies."""
    print(f"Generating '{strategy}' baseline for {len(all_tickers)} tickers over {len(full_date_range)} days...")
    fetch_start = full_date_range.min() - timedelta(days=5)
    fetch_end = full_date_range.max() + timedelta(days=1)
    
    market_data = get_bulk_market_data(list(all_tickers), fetch_start, fetch_end)
    if market_data.empty or 'Close' not in market_data: return pd.DataFrame()

    results = []
    for ticker in all_tickers:
        if len(all_tickers) > 1:
            if ticker not in market_data.columns.get_level_values(1): continue
            df = market_data.xs(ticker, level=1, axis=1).copy()
        else:
            df = market_data.copy()

        df.dropna(subset=['Open', 'Close'], inplace=True)
        if df.empty: continue

        df['Actual_Return'] = (df['Close'] - df['Open']) / df['Open']
        
        if strategy == 'random':
            df['Position'] = np.random.choice([-1, 0, 1], size=len(df))
        elif strategy == 'brsf':
            signal = df['Close'].shift(1) - df['Close'].shift(2)
            df['Position'] = np.sign(signal).fillna(0).astype(int)
        
        df = df.reindex(full_date_range).copy()
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Date'}, inplace=True)
        df['Ticker'] = ticker
        results.append(df[['Date', 'Ticker', 'Position', 'Actual_Return']])

    if not results: return pd.DataFrame()
    final_df = pd.concat(results, ignore_index=True)
    return final_df.dropna(subset=['Position', 'Actual_Return'])

def calculate_metrics(df, full_date_range):
    """Calculates all required metrics for a given dataframe, checking for column existence."""
    metrics = {}
    
    # Financial Metrics (These are always applicable)
    ar, sharpe, max_dd, _ = run_trading_simulation(df.copy(), full_date_range)
    metrics['Ann. Return'] = f"{ar*100:.2f}%"
    metrics['Sharpe'] = f"{sharpe:.3f}"
    metrics['Max Drawdown'] = f"{max_dd*100:.2f}%"

    # Probabilistic Metrics - Only calculate if columns exist
    if 'Brier_Score' in df.columns and 'RPS' in df.columns:
        df_prob = df.dropna(subset=['Brier_Score', 'RPS'])
        metrics['RPS'] = f"{df_prob['RPS'].mean():.4f}" if not df_prob.empty else "N/A"
        metrics['Brier Score'] = f"{df_prob['Brier_Score'].mean():.4f}" if not df_prob.empty else "N/A"
    else:
        metrics['RPS'] = "N/A"
        metrics['Brier Score'] = "N/A"
    
    # Classification Metrics - Only calculate if columns exist
    if 'Pred_Direction' in df.columns and 'Actual_Direction' in df.columns:
        df_class = df.dropna(subset=['Pred_Direction', 'Actual_Direction'])
        if not df_class.empty and len(df_class['Actual_Direction'].unique()) > 1:
            metrics['Accuracy'] = f"{(df_class['Pred_Direction'] == df_class['Actual_Direction']).mean()*100:.2f}%"
            metrics['F1 (Wgt)'] = f"{f1_score(df_class['Actual_Direction'], df_class['Pred_Direction'], average='weighted', zero_division=0):.4f}"
            metrics['MCC'] = f"{matthews_corrcoef(df_class['Actual_Direction'], df_class['Pred_Direction']):.4f}"
        else:
            metrics['Accuracy'] = metrics['F1 (Wgt)'] = metrics['MCC'] = "N/A"
    else:
        metrics['Accuracy'] = "N/A"
        metrics['F1 (Wgt)'] = "N/A"
        metrics['MCC'] = "N/A"

    # Regression Metrics - Only calculate if columns exist
    if 'Pred_Price' in df.columns and 'Actual_Price' in df.columns:
        df_reg = df.dropna(subset=['Pred_Price', 'Actual_Price'])
        if not df_reg.empty:
            metrics['RMSE'] = f"{np.sqrt(((df_reg['Actual_Price'] - df_reg['Pred_Price']) ** 2).mean()):.4f}"
            metrics['sMAPE'] = f"{(2 * (df_reg['Actual_Price'] - df_reg['Pred_Price']).abs() / (df_reg['Actual_Price'].abs() + df_reg['Pred_Price'].abs())).mean() * 100:.2f}%"
            metrics['MdAPE'] = f"{df_reg['Abs_Error_Pct'].median() * 100:.2f}%"
        else:
            metrics['RMSE'] = metrics['sMAPE'] = metrics['MdAPE'] = "N/A"
    else:
        metrics['RMSE'] = "N/A"
        metrics['sMAPE'] = "N/A"
        metrics['MdAPE'] = "N/A"
        
    return metrics

def print_model_report(model_name, df, full_date_range):
    """Prints a full report for a model, including aggregated and per-ticker metrics."""
    print("\n" + "="*80)
    print(f"EVALUATION REPORT: {model_name.upper()}")
    print("="*80)

    # 1. Aggregated Metrics
    print("\n--- AGGREGATED METRICS (ALL TICKERS) ---")
    agg_metrics = calculate_metrics(df, full_date_range)
    agg_df = pd.DataFrame([agg_metrics], index=['Overall']).T
    print(agg_df)
    
    # 2. Per-Ticker Breakdown
    if 'Ticker' in df.columns and df['Ticker'].nunique() > 1:
        print("\n--- PER-TICKER METRICS BREAKDOWN ---")
        
        ticker_metrics_list = []
        for ticker, group in df.groupby('Ticker'):
            ticker_metrics = calculate_metrics(group, full_date_range)
            ticker_metrics['Ticker'] = ticker
            ticker_metrics_list.append(ticker_metrics)

        if ticker_metrics_list:
            ticker_df = pd.DataFrame(ticker_metrics_list).set_index('Ticker')
            ticker_df = ticker_df[agg_df.index]
            with pd.option_context('display.max_rows', None, 'display.max_columns', None, 'display.width', 1000):
                print(ticker_df)

    print("\n" + "="*80)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark LLM prediction folders with detailed, comparative analysis.")
    parser.add_argument("base_folder", type=str, help="Path to the base folder containing model sub-folders")
    args = parser.parse_args()

    if not os.path.isdir(args.base_folder):
        print(f"ERROR: Base folder not found at {args.base_folder}")
        exit(1)

    model_dirs = [os.path.join(args.base_folder, d) for d in os.listdir(args.base_folder) if os.path.isdir(os.path.join(args.base_folder, d))]
    
    if not model_dirs:
        print(f"ERROR: No model sub-folders found in {args.base_folder}")
        exit(1)

    all_results = {}
    all_tickers = set()
    all_dates = []

    for model_dir in model_dirs:
        model_name = os.path.basename(model_dir)
        df = benchmark_predictions(model_dir)
        if not df.empty:
            df['Date'] = pd.to_datetime(df['Date'])
            all_results[model_name] = df
            all_tickers.update(df['Ticker'].unique())
            all_dates.extend(df['Date'].unique())
    
    if not all_dates:
        print("Processing complete, but no valid data was found to generate a report.")
        exit()

    min_date, max_date = min(all_dates), max(all_dates)
    full_date_range = pd.date_range(start=min_date, end=max_date, freq='B')
    print(f"\nGlobal evaluation period set from {min_date.date()} to {max_date.date()}.")

    random_baseline_df = generate_baseline_df(all_tickers, full_date_range, 'random')
    brsf_baseline_df = generate_baseline_df(all_tickers, full_date_range, 'brsf')
    if not random_baseline_df.empty: all_results['BASELINE_Random'] = random_baseline_df
    if not brsf_baseline_df.empty: all_results['BASELINE_BRSF'] = brsf_baseline_df

    for model_name, df in sorted(all_results.items()):
        print_model_report(model_name, df, full_date_range)
        
    print("\nAll benchmarks completed!")