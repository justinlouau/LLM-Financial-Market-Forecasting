import os
import re
import time
import json
import argparse
import numpy as np
import pandas as pd
import yfinance as yf

from datetime import datetime, timedelta
from scipy.stats import norm
from sklearn.metrics import f1_score, matthews_corrcoef

# ================= CONFIGURATION =================
# Thesis Parameters
CONFIDENCE_THRESHOLD = 0.55   # Confidence required to enter a trade
INITIAL_CAPITAL = 100000      # $100k simulated portfolio
RISK_FREE_RATE = 0.04         # 4% annual risk-free rate
TRANSACTION_COST = 0.0005     # 5 bps round-trip trading cost/slippage penalty
UNCHANGED_THRESHOLD = 0.005   # 0.5% threshold to define "Unchanged" class
# =================================================

def parse_filename(filename):
    """Extracts ticker and date from filename like: AEP_20240828_data_run_1"""
    try:
        match = re.search(r'(\d{8})', filename)
        if not match:
            return None, None
        date_str = match.group(1)
        ticker_part = filename[:match.start()]
        ticker = ticker_part.rstrip('_')
        return ticker, date_str
    except Exception as e:
        return None, None

def get_market_data(ticker, date_str):
    try:
        yf_ticker = ticker.replace("_", "-").replace(" ", "-").replace(".", "-").upper()
        target_date = datetime.strptime(date_str, "%Y%m%d")
        
        # Look back 40 days to guarantee enough data for a 20-day historical volatility window
        start_date = target_date - timedelta(days=40)
        end_date = target_date + timedelta(days=10)

        dat = yf.Ticker(yf_ticker)
        df = dat.history(start=start_date, end=end_date, auto_adjust=False)

        if df is None or df.empty: return None
        
        if isinstance(df.columns, pd.MultiIndex): 
            df.columns = df.columns.get_level_values(0)

        # Calculate daily returns for historical volatility
        df['Returns'] = df['Close'].pct_change()

        # Drop timezone awareness to compare dates safely
        df_tz_naive = df.copy()
        df_tz_naive.index = df_tz_naive.index.tz_localize(None)

        # Find the last valid trading day exactly on or before our target date
        past_dates = df_tz_naive[df_tz_naive.index <= target_date]
        if past_dates.empty: return None
        
        base_loc = df_tz_naive.index.get_loc(past_dates.index[-1])
        if base_loc + 1 >= len(df): return None

        base_day = df.iloc[base_loc]
        next_day = df.iloc[base_loc + 1]
        
        # Calculate 20-day historical volatility dynamically (Academic CRPS fix)
        vol_window = df['Returns'].iloc[max(0, base_loc-19):base_loc+1]
        hist_vol = vol_window.std()
        
        # Fallback if insufficient data or zero volatility
        if pd.isna(hist_vol) or hist_vol == 0:
            hist_vol = 0.02

        # Safety filter for bad data
        if float(next_day['Open']) <= 0 or float(base_day['Close']) <= 0:
            return None

        # Prevent false massive drops from stock splits
        if 'Stock Splits' in df.columns and next_day['Stock Splits'] != 0:
            print(f"Skipping {ticker} on {date_str} due to corporate action (Split).")
            return None

        return {
            'base_date': df.index[base_loc],
            'base_close': float(base_day['Close']),
            'actual_next_date': df.index[base_loc + 1],
            'actual_next_open': float(next_day['Open']),
            'actual_next_close': float(next_day['Close']),
            'historical_vol': hist_vol
        }
    except Exception as e:
        print(f"CRITICAL ERROR fetching {ticker}: {str(e)}")
        return None

def calculate_multiclass_brier(probs, outcome_one_hot):
    """Calculates the Brier Score for multiclass classification."""
    return np.sum((probs - outcome_one_hot) ** 2)

def calculate_rps(probs, actual_dir):
    """Calculates Ranked Probability Score (RPS) for ordinal multiclass."""
    # probs is[prob_up, prob_down, prob_unchanged]
    p_up, p_down, p_unchanged = probs[0], probs[1], probs[2]
    
    # Cumulative predicted probabilities (Ordinal sequence: Down -> Unchanged -> Up)
    cp1 = p_down
    cp2 = p_down + p_unchanged
    
    # Cumulative actual outcomes
    if actual_dir == -1:    # Down
        ca1, ca2 = 1.0, 1.0
    elif actual_dir == 0:   # Unchanged
        ca1, ca2 = 0.0, 1.0
    else:                   # Up
        ca1, ca2 = 0.0, 0.0
        
    return (cp1 - ca1)**2 + (cp2 - ca2)**2

def calculate_crps(pred_price, actual_price, current_price, hist_vol):
    """Calculates Continuous Ranked Probability Score (CRPS) dynamically."""
    try:
        sigma = current_price * hist_vol
        z = (actual_price - pred_price) / sigma
        pdf = norm.pdf(z)
        cdf = norm.cdf(z)
        crps = sigma * (z * (2 * cdf - 1) + 2 * pdf - 1 / np.sqrt(np.pi))
        return crps
    except Exception:
        return np.nan

def benchmark_predictions(folder_path):
    results = []
    files =[f for f in os.listdir(folder_path) if f.endswith('.json')]
    print(f"Found {len(files)} files. Processing...")

    for i, filename in enumerate(files):
        if i % 50 == 0:
            print(f"Processing file {i}/{len(files)}...")

        ticker, date_str = parse_filename(filename)
        if not ticker:
            continue

        try:
            with open(os.path.join(folder_path, filename), 'r') as f:
                data = json.load(f)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

        output = data.get('output', {})
        if not isinstance(output, dict):
            continue

        pred_price = output.get('next_day_closing_price')
        if pred_price is not None:
            try:
                pred_price = float(pred_price)
            except ValueError:
                pred_price = None

        try:
            pred_direction = int(output.get('next_day_direction'))
        except (ValueError, TypeError):
            pred_direction = None

        forecast = output.get('forecast', {})
        if not isinstance(forecast, dict): forecast = {}
        
        prob_up = forecast.get('up', 0.0)
        prob_down = forecast.get('down', 0.0)
        prob_unchanged = forecast.get('unchanged', 0.0)

        total_prob = prob_up + prob_down + prob_unchanged
        if total_prob > 0:
            probs = np.array([prob_up, prob_down, prob_unchanged]) / total_prob
        else:
            probs = np.array([0.33, 0.33, 0.33])

        market_data = get_market_data(ticker, date_str)
        time.sleep(0.05)
        
        if not market_data:
            continue

        actual_price = market_data['actual_next_close']
        base_price = market_data['base_close']
        hist_vol = market_data['historical_vol']

        pct_change = (actual_price - base_price) / base_price
        
        # Widened the unchanged threshold to prevent sparse class imbalance
        if abs(pct_change) <= UNCHANGED_THRESHOLD:
            actual_dir = 0
            outcome_vector = np.array([0, 0, 1])
        elif pct_change > 0:
            actual_dir = 1
            outcome_vector = np.array([1, 0, 0])
        else:
            actual_dir = -1
            outcome_vector = np.array([0, 1, 0])

        brier_score = calculate_multiclass_brier(probs, outcome_vector)
        rps_score = calculate_rps(probs, actual_dir)
        
        crps_score = None
        if pred_price is not None:
            crps_score = calculate_crps(pred_price, actual_price, base_price, hist_vol)

        abs_error_pct = abs((pred_price - actual_price) / actual_price) if pred_price else None

        results.append({
            'Ticker': ticker,
            'Date': market_data.get('actual_next_date'),
            'Base_Price': base_price,
            'Actual_Price': actual_price,
            'Pred_Price': pred_price,
            'Actual_Return': (actual_price - market_data['actual_next_open']) / market_data['actual_next_open'],

            'Pred_Direction': pred_direction,
            'Actual_Direction': actual_dir,

            'Prob_Up': probs[0],
            'Prob_Down': probs[1],
            'Brier_Score': brier_score,
            'RPS': rps_score,
            'CRPS': crps_score,
            'Abs_Error_Pct': abs_error_pct
        })

    return pd.DataFrame(results)

def run_trading_simulation(df):
    if df.empty:
        return 0.0, 0.0, 0.0, 0.0, df
        
    df['Date'] = pd.to_datetime(df['Date']).dt.date
    df = df.sort_values(by='Date')
    
    # 1. Flag positions based on model's confidence threshold
    df['Position'] = np.where(df['Prob_Up'] > CONFIDENCE_THRESHOLD, 1, 0)
    
    # Enable SHORT selling for high confidence downward predictions
    df['Position'] = np.where(df['Prob_Down'] > CONFIDENCE_THRESHOLD, -1, df['Position'])

    # Apply Returns + Transaction/Slippage Costs
    df['Strategy_Return'] = (df['Position'] * df['Actual_Return']) - (abs(df['Position']) * TRANSACTION_COST)
    
    # 2. Calculate daily returns exclusively across active positions
    active_trades = df[df['Position'] != 0]
    
    if not active_trades.empty:
        daily_returns = active_trades.groupby('Date')['Strategy_Return'].mean().reset_index()
        daily_returns.rename(columns={'Strategy_Return': 'Daily_Return'}, inplace=True)
    else:
        daily_returns = pd.DataFrame(columns=['Date', 'Daily_Return'])
    
    # 3. Handle cash drag correctly (uninvested cash earns risk-free rate)
    all_dates = pd.DataFrame({'Date': df['Date'].dropna().unique()})
    daily_df = pd.merge(all_dates, daily_returns, on='Date', how='left')
    
    daily_risk_free = (1 + RISK_FREE_RATE) ** (1/252) - 1
    
    # Replace NaNs (cash-only days) with the daily risk-free rate
    daily_df['Daily_Return'] = daily_df['Daily_Return'].fillna(daily_risk_free)
    daily_df = daily_df.sort_values(by='Date')
    
    if daily_df.empty:
        return 0.0, 0.0, 0.0, 0.0, df

    daily_df['Cumulative_Strategy'] = (1 + daily_df['Daily_Return']).cumprod() * INITIAL_CAPITAL
    
    start_date = daily_df['Date'].min()
    end_date = daily_df['Date'].max()
    days_passed = (end_date - start_date).days if start_date != end_date else 1
    years = max(days_passed / 365.25, 0.01)
    
    total_return = (daily_df['Cumulative_Strategy'].iloc[-1] / INITIAL_CAPITAL) - 1
    
    # Calculate regular annualized and compound annualized
    regular_annual_return = total_return / years
    annual_return = (1 + total_return) ** (1 / years) - 1
    
    # Sharpe Ratio (cash days now yield exactly 0.0 excess return)
    excess_returns = daily_df['Daily_Return'] - daily_risk_free
    std_excess = excess_returns.std()
    
    if len(excess_returns) < 2 or pd.isna(std_excess) or std_excess == 0:
        sharpe = 0
    else:
        sharpe = (excess_returns.mean() / std_excess) * np.sqrt(252)

    daily_df['High_Water_Mark'] = daily_df['Cumulative_Strategy'].cummax()
    daily_df['Drawdown'] = (daily_df['Cumulative_Strategy'] / daily_df['High_Water_Mark']) - 1
    max_drawdown = daily_df['Drawdown'].min()

    return annual_return, regular_annual_return, sharpe, max_drawdown, df

def print_thesis_report(df, data_folder=None):
    if df.empty:
        print("No data processed.")
        return

    model_name = os.path.basename(os.path.normpath(data_folder)) if data_folder else 'model'
    model_name_clean = re.sub(r'[^A-Za-z0-9_\-]', '_', str(model_name))
    csv_filename = f"{model_name_clean}_results.csv"
    txt_filename = f"{model_name_clean}_results.txt"

    # Separate dataframes to bypass NaNs securely for distinct metrics
    df_clean = df.dropna(subset=['Pred_Direction', 'Actual_Direction'])
    df_reg = df.dropna(subset=['Pred_Price', 'Actual_Price'])

    # 1. Financial Metrics
    ar, reg_ar, sharpe, max_dd, sim_df = run_trading_simulation(df)

    # 2. Probabilistic Metrics
    avg_brier = df['Brier_Score'].mean()
    avg_crps = df['CRPS'].mean()
    avg_rps = df['RPS'].mean()

    # 3. Reasoning / Direction Metrics 
    if not df_clean.empty:
        accuracy = (df_clean['Pred_Direction'] == df_clean['Actual_Direction']).mean()
        f1_weight = f1_score(df_clean['Actual_Direction'], df_clean['Pred_Direction'], average='weighted')
        f1_macro = f1_score(df_clean['Actual_Direction'], df_clean['Pred_Direction'], average='macro')
        mcc = matthews_corrcoef(df_clean['Actual_Direction'], df_clean['Pred_Direction'])
    else:
        accuracy = f1_weight = f1_macro = mcc = np.nan

    # 4. Regression Metrics 
    if not df_reg.empty:
        mdape = df_reg['Abs_Error_Pct'].median() * 100
        mape = df_reg['Abs_Error_Pct'].mean() * 100
        rmse = np.sqrt(((df_reg['Actual_Price'] - df_reg['Pred_Price']) ** 2).mean())
        mae = (df_reg['Actual_Price'] - df_reg['Pred_Price']).abs().mean()
        
        # sMAPE
        smape = (2 * (df_reg['Actual_Price'] - df_reg['Pred_Price']).abs() / 
                 (df_reg['Actual_Price'].abs() + df_reg['Pred_Price'].abs())).mean() * 100
    else:
        mdape = mape = rmse = mae = smape = np.nan

    report_lines =[
        "\n" + "="*60,
        "THESIS RESULTS: MODERATE-AGGRESSIVE STRATEGY",
        "="*60,
        f"Total Samples:         {len(df)}",
        "-" * 60,
        "1. FINANCIAL PERFORMANCE (Risk-Adjusted)",
        f"   Regular Annual Ret:   {reg_ar*100:.2f}%   (Simple Annualized)",
        f"   Annual Return (CAGR): {ar*100:.2f}%   (Higher is better)",
        f"   Sharpe Ratio:         {sharpe:.4f}   (Higher is better; >1.0 is ideal)",
        f"   Max Drawdown:         {max_dd*100:.2f}%   (Lower is better; less negative is better)",
        "-" * 60,
        "2. PROBABILISTIC FORECASTING (Uncertainty)",
        f"   CRPS:                 {avg_crps:.4f}   (Lower is better; 0 is perfect)",
        f"   RPS:                  {avg_rps:.4f}   (Lower is better; 0 is perfect)",
        f"   Brier Score:          {avg_brier:.4f}   (Lower is better; 0 is perfect)",
        "-" * 60,
        "3. CLASSIFICATION & ACCURACY (Historical)",
        f"   MCC Score:            {mcc:.4f}   (Higher is better; 1 is perfect, 0 is random)",
        f"   Direction Accuracy:   {accuracy*100:.2f}%   (Higher is better)",
        f"   F1 Score (Weighted):  {f1_weight:.4f}   (Higher is better; 1 is perfect)",
        f"   F1 Score (Macro):     {f1_macro:.4f}   (Higher is better; 1 is perfect)",
        "-" * 60,
        "4. PRICE PRECISION",
        f"   RMSE:                 {rmse:.4f}   (Lower is better)",
        f"   MAE:                  {mae:.4f}   (Lower is better)",
        f"   MAPE:                 {mape:.2f}%   (Lower is better)",
        f"   sMAPE:                {smape:.2f}%   (Lower is better)",
        f"   MdAPE (Median):       {mdape:.2f}%   (Lower is better; mitigates outliers)",
        "="*60
    ]

    report_str = "\n".join(report_lines)
    print(report_str)

    # Export
    df.to_csv(csv_filename, index=False)
    with open(txt_filename, 'w') as f:
        f.write(report_str + "\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark LLM predictions.")
    parser.add_argument("data_folder", type=str, help="Path to the data folder")
    args = parser.parse_args()

    if os.path.exists(args.data_folder):
        results_df = benchmark_predictions(args.data_folder)
        print_thesis_report(results_df, args.data_folder)
    else:
        print(f"ERROR: Data folder not found at {args.data_folder}")