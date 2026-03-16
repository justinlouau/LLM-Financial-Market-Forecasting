import os
import json
import copy
import tiktoken

def check_token_length(text: str, max_tokens: int, label: str) -> None:
    """
    Checks token length using tiktoken and warns if the limit is exceeded.
    """
    try:
        encoding = tiktoken.get_encoding("cl100k_base")
        token_count = len(encoding.encode(text))
        if token_count > max_tokens:
            print(
                f"Warning: {label} is {token_count} tokens, which exceeds the "
                f"limit of {max_tokens}. Consider reducing input size."
            )
    except Exception as e:
        print(f"Warning: Failed to count tokens for {label}: {e}")

def create_prompt_and_ts(financial_data: dict, stock_data: dict):
    """
    Creates a ChatTS-formatted prompt and synchronized timeseries_data.
    Includes <ts><ts/> tags and appends to the array ONLY for series that exist.
    """
    financial_document_str = json.dumps(financial_data, indent=4)

    # Check length
    check_token_length(financial_document_str, max_tokens=10000, label="financial document")
    
    example_json = {
        "chain_of_thought": "Step-by-step analysis...",
        "output": {
            "next_day_direction": 1,
            "next_day_closing_price": 172.2,
            "forecast": {"up": 0.60, "down": 0.20, "unchanged": 0.20},
        }
    }
    example_str = json.dumps(example_json, indent=4)

    # Define the canonical order and keys
    features_map = [
        ("Open", "open"),
        ("High", "high"),
        ("Low", "low"),
        ("Close", "close"),
        ("Volume", "volume")
    ]

    ts_text_parts = []
    timeseries_data = []

    # Dynamically build the prompt & array synchronously based on available data
    for label, key in features_map:
        data = stock_data.get(key, [])
        if data and len(data) > 0:
            try:
                # Calculate simple stats for the text description
                val_min = float(min(data))
                val_max = float(max(data))
                # ChatTS standard placeholder format
                part = f"{label} (min:{val_min:.2f}, max:{val_max:.2f}):  <ts><ts/>"
                
                ts_text_parts.append(part)
                timeseries_data.append(data)
            except Exception:
                continue

    ts_text = " ".join(ts_text_parts)

    prompt_template = f"""You are a financial analyst LLM. Given the following financial document and time series data, perform chain-of-thought reasoning to predict the stock price movement.

Your response must be a valid JSON object strictly adhering to the following schema example:
{example_str}

---
FINANCIAL DOCUMENT:
{financial_document_str}
---
TIME SERIES DATA:
{ts_text}
"""
    return prompt_template, timeseries_data


def create_training_set_with_context():
    filtered_folder = 'output_filtered'
    training_data_folder = 'output_training_data_json' 
    output_file = 'training_set.jsonl' 

    if not os.path.exists(filtered_folder) or not os.path.exists(training_data_folder):
        print("Error: Input directories missing.")
        return

    filtered_files = sorted([f for f in os.listdir(filtered_folder) if f.endswith('.json')])

    with open(output_file, 'w', encoding='utf-8') as outfile:
        for filename in filtered_files:
            filtered_path = os.path.join(filtered_folder, filename)
            training_data_path = os.path.join(training_data_folder, filename)

            if not os.path.exists(training_data_path):
                continue

            try:
                with open(filtered_path, 'r', encoding='utf-8') as f:
                    filtered_data = json.load(f)

                with open(training_data_path, 'r', encoding='utf-8') as f:
                    training_text = f.read()

                try:
                    training_data = json.loads(training_text)
                except json.JSONDecodeError:
                    training_data = training_text

                # Prepare context and stock data for prompt creation
                context_data = copy.deepcopy(filtered_data)
                raw_stock = context_data.pop('stock_data', {})
                
                # Normalize keys
                stock_data = {
                    'open': raw_stock.get('open', []),
                    'high': raw_stock.get('high', []),
                    'low': raw_stock.get('low', []),
                    'close': raw_stock.get('close', []),
                    'volume': raw_stock.get('volume', [])
                }

                # Construct prompt (Input) & aligned time series
                prompt_text, timeseries_data = create_prompt_and_ts(context_data, stock_data)

                # Skip if there are 0 valid time series
                if not timeseries_data:
                    print(f"Skipping {filename}: No valid time series data.")
                    continue

                # Construct response string (Output)
                if isinstance(training_data, dict):
                    reasoning = training_data.get('chain_of_thought', '')
                    prediction = training_data.get('output', {})
                    if reasoning:
                        response = f"{reasoning}\n\nFinal Prediction: {json.dumps(prediction)}"
                    else:
                        response = json.dumps(training_data)
                else:
                    response = str(training_data).strip() if str(training_data).strip() else 'No rationale provided.'

                # Create the JSONL entry
                jsonl_entry = {
                    "input": prompt_text,
                    "output": response,
                    "timeseries": timeseries_data
                }

                # Write as a single line in the JSONL file
                outfile.write(json.dumps(jsonl_entry) + '\n')

            except Exception as e:
                print(f"Error processing {filename}: {e}")

    print(f"Done! Created {output_file}")

if __name__ == '__main__':
    create_training_set_with_context()