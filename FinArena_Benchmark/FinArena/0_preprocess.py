# 01_preprocess_data.py
import pandas as pd
import os

# --- Configuration ---
DATASET_PATH = 'FinArena-low-cost-dataset'
COMPANIES = {
    'AMZN': 'Amazon', 'GOOG': 'Google', 'MSFT': 'Microsoft', 
    'NVDA': 'Nvidia', 'TSLA': 'Tesla',
    # We exclude FinArena's non-SP&500 companies for this thesis
    # '002594': 'BYD', '300750': 'CATL', '600941': 'CMCC',
    # '688047': 'Loongson', '600519': 'MOUTAI'
}
START_DATE = '2023-01-01'
END_DATE = '2024-03-30'
OUTPUT_FILE = 'processed_data.csv'

def load_and_prepare_data(ticker, company_name, base_path):
    """Loads, merges, and prepares all data for a single company."""
    print(f"Processing {ticker} ({company_name})...")

    # 1. Load Stock Prices
    price_path = os.path.join(base_path, 'stock', f'{ticker}.csv')
    stock_df = pd.read_csv(price_path)
    stock_df['date'] = pd.to_datetime(stock_df['date'])
    stock_df.rename(columns={'ticker': 'ticker_price'}, inplace=True)
    
    # 2. Load News Data
    news_path = os.path.join(base_path, 'news', f'{company_name}news.json')
    news_df = pd.read_json(news_path)
    news_df.rename(columns={'Date': 'date', 'Title': 'news_title', 'Text': 'news_text'}, inplace=True)
    news_df['date'] = pd.to_datetime(news_df['date'])
    news_df = news_df.sort_values('date').drop_duplicates(subset=['date'], keep='last')

    # 3. Load Financial Statements
    statement_path = os.path.join(base_path, 'statement')
    balance_df = pd.read_json(os.path.join(statement_path, f'{company_name}_balance_sheet_quarter.json'))
    cash_df = pd.read_json(os.path.join(statement_path, f'{company_name}_cash_flow_quarter.json'))
    income_df = pd.read_json(os.path.join(statement_path, f'{company_name}_income_statement_quarter.json'))
    
    statement_dfs = {'balance': balance_df, 'cash': cash_df, 'income': income_df}
    merged_statements = None

    for name, df in statement_dfs.items():
        if df.empty: continue
        df.rename(columns={'report_date': 'date'}, inplace=True)
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date').drop_duplicates(subset=['date'], keep='last')

        # Add prefix to avoid column name collisions
        df = df.add_prefix(f'{name}_')
        df.rename(columns={f'{name}_date': 'date'}, inplace=True)
        
        # Merge statements together
        if merged_statements is None:
            merged_statements = df
        else:
            merged_statements = pd.merge_asof(merged_statements, df, on='date', direction='backward')

    # 4. Merge all data sources
    merged_df = stock_df.sort_values('date')
    merged_df = pd.merge_asof(merged_df, news_df, on='date', direction='backward')

    if merged_statements is not None:
        # Drop ticker columns that might not exist to avoid errors
        columns_to_drop = [col for col in ['balance_ticker', 'cash_ticker', 'income_ticker'] if col in merged_statements.columns]
        if columns_to_drop:
            merged_statements = merged_statements.drop(columns=columns_to_drop)
        merged_df = pd.merge_asof(merged_df, merged_statements, on='date', direction='backward')

    # 5. Create Target Variable
    # The target for day 't' is if the price goes up on day 't+1'
    merged_df['target'] = (merged_df['close'].shift(-1) > merged_df['close']).astype(int)
    
    # Add ticker column for identification
    merged_df['ticker'] = ticker
    
    return merged_df

# --- Main Execution ---
all_company_data = []
for ticker, name in COMPANIES.items():
    try:
        company_df = load_and_prepare_data(ticker, name, os.path.join(DATASET_PATH, 'metadata'))
        all_company_data.append(company_df)
    except FileNotFoundError as e:
        print(f"Skipping {ticker}: {e}")
    except Exception as e:
        print(f"Error processing {ticker}: {e}")
        import traceback
        traceback.print_exc()

# Combine all companies into a single DataFrame
if not all_company_data:
    raise ValueError("No company data was successfully loaded. Please check file paths and data availability.")

final_df = pd.concat(all_company_data, ignore_index=True)

# Filter by date range and drop rows with no target
final_df = final_df[
    (final_df['date'] >= START_DATE) &
    (final_df['date'] <= END_DATE)
]
final_df.dropna(subset=['target'], inplace=True)

# Save to file
final_df.to_csv(OUTPUT_FILE, index=False)
print(f"\nPreprocessing complete. Data saved to {OUTPUT_FILE}")
print(f"Total rows: {len(final_df)}")
print("Sample of the data:")
print(final_df[['ticker', 'date', 'close', 'news_title', 'balance_total_assets', 'target']].head())