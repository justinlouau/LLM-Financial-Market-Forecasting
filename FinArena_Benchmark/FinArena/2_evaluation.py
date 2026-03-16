# 03_evaluate_and_report.py
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

# Configuration
PREDICTIONS_FILE = 'predictions.csv'
INITIAL_CAPITAL = 100000.0

# Define risk profiles as defined in FinArena
RISK_PROFILES = {
    'Conservative': {'buy_pct': 0.50, 'sell_pct': 1.00},
    'M.Conservative': {'buy_pct': 0.70, 'sell_pct': 1.00},
    'M.Aggressive': {'buy_pct': 1.00, 'sell_pct': 0.50},
    'Aggressive': {'buy_pct': 1.00, 'sell_pct': 0.30},
}

def calculate_simulation_metrics(portfolio_history, days):
    """Calculates AR, MD, and SR from a list of daily portfolio values."""
    portfolio_series = pd.Series(portfolio_history)
    
    # 1. Annualized Return (AR)
    total_return = (portfolio_series.iloc[-1] / portfolio_series.iloc[0]) - 1
    annualized_return = ((1 + total_return) ** (365 / days)) - 1
    
    # 2. Maximum Drawdown (MD)
    running_max = portfolio_series.cummax()
    drawdown = (running_max - portfolio_series) / running_max
    max_drawdown = drawdown.max()
    
    # 3. Sharpe Ratio (SR)
    daily_returns = portfolio_series.pct_change().dropna()
    if daily_returns.std() == 0:
        sharpe_ratio = 0.0
    else:
        sharpe_ratio = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252)
        
    return annualized_return, max_drawdown, sharpe_ratio

# --- Main Execution ---
print("Loading predictions...")
df = pd.read_csv(PREDICTIONS_FILE, parse_dates=['date'])
tickers = df['ticker'].unique()

# Task 1: Stock Movement Prediction Evaluation
print("\n--- Task 1: Stock Movement Prediction Metrics ---")
metrics = []
for ticker in tickers:
    ticker_df = df[df['ticker'] == ticker]
    acc = accuracy_score(ticker_df['target'], ticker_df['prediction'])
    f1 = f1_score(ticker_df['target'], ticker_df['prediction'])
    metrics.append({'ticker': ticker, 'Accuracy': acc, 'F1-Score': f1})

metrics_df = pd.DataFrame(metrics).set_index('ticker')
print(metrics_df)
print("\nSummary:")
print(metrics_df.agg(['mean', 'std']))


# Task 2: Stock Trading Simulation
print("\n--- Task 2: Stock Trading Simulation ---")
simulation_results = []

# Store detailed metrics for each ticker and profile
detailed_metrics = {
    'AR': {},  # {profile_name: {ticker: value}}
    'MD': {},
    'SR': {}
}

for profile_name, params in RISK_PROFILES.items():
    profile_metrics = {'AR': [], 'MD': [], 'SR': []}
    
    # Initialize nested dictionaries for this profile
    detailed_metrics['AR'][profile_name] = {}
    detailed_metrics['MD'][profile_name] = {}
    detailed_metrics['SR'][profile_name] = {}
    
    for ticker in tickers:
        ticker_df = df[df['ticker'] == ticker].copy()
        
        # Simulate trading
        capital = INITIAL_CAPITAL
        shares = 0
        portfolio_history = [INITIAL_CAPITAL]
        
        for _, row in ticker_df.iterrows():
            # Buy signal
            if row['prediction'] == 1 and capital > 1:
                investment = capital * params['buy_pct']
                shares_to_buy = investment / row['close']
                capital -= investment
                shares += shares_to_buy
            # Sell signal
            elif row['prediction'] == 0 and shares > 0:
                shares_to_sell = shares * params['sell_pct']
                cash_from_sale = shares_to_sell * row['close']
                shares -= shares_to_sell
                capital += cash_from_sale
            
            # Record portfolio value for the day
            portfolio_value = capital + (shares * row['close'])
            portfolio_history.append(portfolio_value)

        # Calculate metrics for this ticker and profile
        num_days = (ticker_df['date'].iloc[-1] - ticker_df['date'].iloc[0]).days
        ar, md, sr = calculate_simulation_metrics(portfolio_history, num_days)
        profile_metrics['AR'].append(ar)
        profile_metrics['MD'].append(md)
        profile_metrics['SR'].append(sr)
        
        # Store detailed metrics for this ticker and profile
        detailed_metrics['AR'][profile_name][ticker] = ar
        detailed_metrics['MD'][profile_name][ticker] = md
        detailed_metrics['SR'][profile_name][ticker] = sr

    # Average metrics for the profile across all tickers
    simulation_results.append({
        'Profile': profile_name,
        'Mean AR': np.mean(profile_metrics['AR']),
        'Mean MD': np.mean(profile_metrics['MD']),
        'Mean SR': np.mean(profile_metrics['SR'])
    })

# Print summary table
results_df = pd.DataFrame(simulation_results).set_index('Profile')
print("Summary Results:")
print(results_df)

# Print detailed tables for each metric
print("\n--- Detailed Trading Metrics by Stock ---")

# Annualized Return (AR) Table
print("\n--- Annualized Return (AR) by Stock ---")
ar_df = pd.DataFrame(detailed_metrics['AR']).T
ar_df.index.name = 'Profile'
print(ar_df)
print("\nAR Summary:")
print(ar_df.agg(['mean', 'std']))

# Maximum Drawdown (MD) Table
print("\n--- Maximum Drawdown (MD) by Stock ---")
md_df = pd.DataFrame(detailed_metrics['MD']).T
md_df.index.name = 'Profile'
print(md_df)
print("\nMD Summary:")
print(md_df.agg(['mean', 'std']))

# Sharpe Ratio (SR) Table
print("\n--- Sharpe Ratio (SR) by Stock ---")
sr_df = pd.DataFrame(detailed_metrics['SR']).T
sr_df.index.name = 'Profile'
print(sr_df)
print("\nSR Summary:")
print(sr_df.agg(['mean', 'std']))