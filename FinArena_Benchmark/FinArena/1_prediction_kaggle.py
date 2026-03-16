# 02_run_prediction.py
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm

# Configuration
INPUT_FILE = '/kaggle/input/processed-data/processed_data.csv'
OUTPUT_FILE = '/kaggle/working/predictions.csv'
TEST_START_DATE = '2024-01-01'
TEST_END_DATE = '2024-03-27'
PRICE_HISTORY_WINDOW = 30 

def predict_movement_with_llm(full_df: pd.DataFrame, current_index: int) -> int:
    """
    Analyzes data for a specific day using the Time Series LLM and returns a prediction.
    
    Args:
        full_df: The complete DataFrame of test data.
        current_index: The integer index of the row to be predicted.
        
    Returns:
        1 for a predicted RISE, 0 for a predicted FALL.
    """
    row = full_df.loc[current_index]
    
    # Get historical time series data
    start_index = max(0, current_index - PRICE_HISTORY_WINDOW)
    
    # Ensure we only get data for the same ticker up to the current point
    ticker_df = full_df[full_df['ticker'] == row['ticker']]
    historical_slice = ticker_df.loc[:current_index].iloc[-PRICE_HISTORY_WINDOW-1:]
    
    timeseries_data = historical_slice['close'].values.astype(np.float32)

    # Get textual news and financial data
    news_text = f"Title: {row['news_title']}. Body: {row['news_text']}"

    # Format financial data for the prompt
    net_profit = f"{row['income_net_profit']/1e9:.2f}B" if pd.notna(row['income_net_profit']) else "N/A"
    total_assets = f"{row['balance_total_assets']/1e9:.2f}B" if pd.notna(row['balance_total_assets']) else "N/A"

    # Construct Prompt
    prompt_template = f"""Analyze the following financial data for the stock {row['ticker']}.
Recent News:
{news_text}

Key Financials (most recent quarter):
- Net Profit: {net_profit}
- Total Assets: {total_assets}

Recent Price History (up to today):
<ts><ts/>

Based on all available information, will the stock price RISE or FALL on the next trading day?
Response with a single word RISE or FALL"""

    prompt = f"<|im_start|>system\nYou are an expert financial analyst providing stock predictions.<|im_end|><|im_start|>user\n{prompt_template}<|im_end|><|im_start|>assistant\n"
    
    try:
        # Run the LLM Inference
        inputs = processor(
            text=[prompt],
            timeseries=[timeseries_data],
            padding=True,
            return_tensors="pt"
        )
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        torch.cuda.empty_cache()

        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=150, do_sample=False) # do_sample=False for more deterministic output

        generated_text = tokenizer.decode(outputs[0][len(inputs['input_ids'][0]):], skip_special_tokens=True)
        
        # Parse the Output
        last_words = generated_text.upper().split()[-6:]
        if "RISE" in last_words:
            return 1
        elif "FALL" in last_words:
            return 0
        else:
            # Fallback if the model doesn't follow instructions
            print(f"  [Warning] LLM output for {row['ticker']} on {row['date'].date()} was ambiguous. Defaulting to FALL.")
            print(f"  LLM Response: '{generated_text}'")
            return 0 # Conservative default

    except Exception as e:
        print(f"  [Error] Failed to get prediction for {row['ticker']} on {row['date'].date()}: {e}")
        return 0 # Return a default value on error

# ==============================================================================
# === 3. MAIN EXECUTION LOOP ===
# ==============================================================================
print("Loading preprocessed data...")
df = pd.read_csv(INPUT_FILE, parse_dates=['date'])

# Filter for the test period
test_df = df[
    (df['date'] >= TEST_START_DATE) &
    (df['date'] <= TEST_END_DATE)
].copy().reset_index(drop=True)

if test_df.empty:
    raise ValueError("Test data is empty. Check dates and input file.")

print(f"Generating predictions for {len(test_df)} trading days using the Time Series LLM...")

predictions = []
# Iterate with a progress bar
for idx in tqdm(range(len(test_df)), desc="Making Predictions"):
    prediction = predict_movement_with_llm(test_df, idx)
    predictions.append(prediction)

# Add the generated predictions as a new column
test_df['prediction'] = predictions

# Save the predictions for the evaluation script
output_cols = ['ticker', 'date', 'close', 'target', 'prediction']
test_df[output_cols].to_csv(OUTPUT_FILE, index=False)

print(f"\nPredictions saved to {OUTPUT_FILE}")
print("Sample of predictions:")
print(test_df[output_cols].head())