import os
import json
import argparse
import time
import re
import pandas as pd
import yfinance as yf
from pathlib import Path
from datetime import datetime, timedelta

def extract_json_from_text(text):
    """
    Attempts to find the largest outer JSON object enclosed in {}
    within a string containing other text (from Script 1).

    The model sometimes returns the entire JSON object wrapped as a
    quoted string (i.e. the file content is a JSON string whose value
    is itself another JSON).  ``json.loads`` will happily decode that
    to a Python ``str``, which later code interprets as invalid.
    To handle that we parse recursively until we no longer get a plain
    string, or fall back to the brace‑search methods below.
    """
    if not isinstance(text, str):
        return None

    # Prase JSON recursively to handle nested quoted JSON strings
    try:
        data = json.loads(text)
        while isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError:
                break
        return data
    except json.JSONDecodeError:
        pass

    # Fallback 1: Find largest outer {...}
    start_index = text.find('{')
    end_index = text.rfind('}')
    if start_index != -1 and end_index != -1 and end_index > start_index:
        json_candidate = text[start_index : end_index + 1]
        try:
            return json.loads(json_candidate)
        except json.JSONDecodeError:
            pass

    # Fallback 2: Stack-based extraction of first {...} block
    stack = []
    start = None
    for i, c in enumerate(text):
        if c == '{':
            if not stack:
                start = i
            stack.append(c)
        elif c == '}':
            if stack:
                stack.pop()
                if not stack and start is not None:
                    json_candidate = text[start:i+1]
                    try:
                        return json.loads(json_candidate)
                    except json.JSONDecodeError:
                        break
    return None

def parse_filename_info(filename):
    """
    Extracts ticker and date from filename like: AEP_20240828_data.json
    Returns: ticker, date_str
    """
    try:
        # Regex to find 8 digit date
        match = re.search(r'(\d{8})', filename)
        if not match:
            return None, None
        date_str = match.group(1)

        # Extract ticker
        ticker_part = filename[:match.start()]
        ticker = ticker_part.rstrip('_').replace("_", "-").upper()
        return ticker, date_str
    except Exception:
        return None, None

def get_ground_truth(ticker, date_str):
    """
    Fetches actual market data for T and T+1 using yfinance (from Script 2).
    """
    try:
        start_date = datetime.strptime(date_str, "%Y%m%d")
        end_date = start_date + timedelta(days=10)

        # Fetch history
        dat = yf.Ticker(ticker)
        df = dat.history(start=start_date, end=end_date)

        if df is None or df.empty or len(df) < 2:
            return None

        # Handle MultiIndex
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        # Get T (Base) and T+1 (Target)
        base_close = df.iloc[0]['Close']
        next_close = df.iloc[1]['Close']
        
        # Ensure native python types
        base_close = base_close.item() if hasattr(base_close, 'item') else float(base_close)
        next_close = next_close.item() if hasattr(next_close, 'item') else float(next_close)

        # Determine direction (1: Up, -1: Down, 0: Flat)
        if next_close > base_close:
            direction = 1
        elif next_close < base_close:
            direction = -1
        else:
            direction = 0

        return {
            'base_close': base_close,
            'actual_close': next_close,
            'actual_direction': direction
        }
    except Exception as e:
        print(f"  [Error] Market data fetch failed for {ticker}: {e}")
        return None

def evaluate_output(output_str, ground_truth):
    """
    Parses an output string and scores it against ground truth.
    Returns a dict with 'is_valid', 'direction_correct', 'abs_error_pct'.
    """
    data = extract_json_from_text(output_str)
    
    # 1. Validation Check
    if not data or 'output' not in data:
        return {'is_valid': False, 'score': -999}
    
    out_node = data['output']
    if 'next_day_closing_price' not in out_node or 'next_day_direction' not in out_node:
         return {'is_valid': False, 'score': -999}

    try:
        pred_price = float(out_node['next_day_closing_price'])
        pred_dir = int(out_node['next_day_direction'])
    except ValueError:
        return {'is_valid': False, 'score': -999}

    # 2. Logic Check
    actual_price = ground_truth['actual_close']
    actual_dir = ground_truth['actual_direction']
    
    # Direction Accuracy (High priority)
    direction_match = (pred_dir == actual_dir)
    
    # Price Precision (Tie-breaker)
    # Lower error is better
    abs_error_pct = abs((pred_price - actual_price) / actual_price)
    
    return {
        'is_valid': True,
        'direction_correct': direction_match,
        'abs_error_pct': abs_error_pct,
        'pred_price': pred_price,
        'pred_dir': pred_dir
    }

def select_winner(stats1, stats2):
    """
    Decides which output is 'chosen'. 
    Returns: 1 if output1 is better, 2 if output2 is better, 0 if both invalid.
    """
    # Criterion 1: Validity
    if stats1['is_valid'] and not stats2['is_valid']: return 1
    if not stats1['is_valid'] and stats2['is_valid']: return 2
    if not stats1['is_valid'] and not stats2['is_valid']: return 0

    # Criterion 2: Direction Accuracy
    if stats1['direction_correct'] and not stats2['direction_correct']: return 1
    if not stats1['direction_correct'] and stats2['direction_correct']: return 2

    # Criterion 3: Price Precision (if direction is tied)
    # Smaller error wins
    if stats1['abs_error_pct'] < stats2['abs_error_pct']:
        return 1
    else:
        return 2

# ================= MAIN PROCESS =================

def process_dpo_dataset(input_folder, output_file):
    input_path = Path(input_folder)
    results = []
    
    files = list(input_path.glob('*.json'))
    print(f"Found {len(files)} files to process...")

    # Cache for market data to avoid re-fetching same Ticker/Date
    market_cache = {} 

    processed_count = 0
    skipped_count = 0

    for file_path in files:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                entry = json.load(f)

            # Ensure required keys exist
            if not all(k in entry for k in ('input', 'output1', 'output2')):
                print(f"[Skip] Missing keys in {file_path.name}")
                skipped_count += 1
                continue

            # 1. Identify Context (Ticker/Date)
            # Try parsing filename first
            ticker, date_str = parse_filename_info(file_path.name)
            
            # If not in filename, check if it's stored in the input dict (optional backup)
            if not ticker and 'meta' in entry:
                ticker = entry['meta'].get('ticker')
                date_str = entry['meta'].get('date')

            if not ticker or not date_str:
                print(f"[Skip] Could not identify Ticker/Date for {file_path.name}")
                skipped_count += 1
                continue

            # 2. Get Ground Truth (Market Data)
            cache_key = f"{ticker}_{date_str}"
            if cache_key in market_cache:
                ground_truth = market_cache[cache_key]
            else:
                ground_truth = get_ground_truth(ticker, date_str)
                time.sleep(0.1) # Rate limit protection
                market_cache[cache_key] = ground_truth

            if not ground_truth:
                print(f"[Skip] No market data found for {ticker} on {date_str}")
                skipped_count += 1
                continue

            # 3. Evaluate Models
            stats1 = evaluate_output(entry['output1'], ground_truth)
            stats2 = evaluate_output(entry['output2'], ground_truth)

            # 4. Pick Winner
            winner = select_winner(stats1, stats2)

            if winner == 0:
                print(f"[Skip] Both outputs invalid for {file_path.name}")
                skipped_count += 1
                continue
            
            # Construct DPO Entry
            dpo_entry = {
                "input": entry['input'],
                "timeseries": entry.get('timeseries', []),
                "ticker": ticker,
                "date": date_str,
                "metrics": {
                    "actual_close": ground_truth['actual_close'],
                    "winner_error": stats1['abs_error_pct'] if winner == 1 else stats2['abs_error_pct']
                }
            }

            if winner == 1:
                dpo_entry['chosen'] = entry['output1']
                dpo_entry['rejected'] = entry['output2']
            else:
                dpo_entry['chosen'] = entry['output2']
                dpo_entry['rejected'] = entry['output1']

            results.append(dpo_entry)
            processed_count += 1
            
            if processed_count % 10 == 0:
                print(f"Processed {processed_count} files...")

        except Exception as e:
            print(f"[Error] Failed processing {file_path.name}: {e}")
            skipped_count += 1

    # Write Result to JSONL
    print(f"\nWriting {len(results)} pairs to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f_out:
        for item in results:
            f_out.write(json.dumps(item) + '\n')

    print("-" * 30)
    print(f"Completed.")
    print(f"Successfully generated: {processed_count}")
    print(f"Skipped/Failed:         {skipped_count}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate LLM outputs against Market Data to generate DPO dataset.")
    parser.add_argument("input_folder", type=str, help="Folder containing raw .json files with output1/output2")
    parser.add_argument("--output", type=str, default="dpo_training_data.jsonl", help="Output JSONL filename")
    
    args = parser.parse_args()
    
    if os.path.exists(args.input_folder):
        process_dpo_dataset(args.input_folder, args.output)
    else:
        print("Error: Input folder does not exist.")