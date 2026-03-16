import os
import re
import json
import time
import pandas as pd
import yfinance as yf

from pydantic import BaseModel, Field
from vllm import LLM, SamplingParams
from vllm.sampling_params import GuidedDecodingParams
from datetime import datetime, timedelta
from pathlib import Path
from typing import List

# ================= CONFIGURATION =================
SCRIPT_DIR = Path(__file__).parent.absolute()

_auto = {
    'MAX_MODEL_LEN': 10000,
    'BATCH_SIZE': 128,
    'GPU_MEMORY_UTILIZATION': 0.95,
    'TENSOR_PARALLEL_SIZE': 1
}

MODEL_NAME = os.getenv("MODEL_NAME", "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B")
MAX_MODEL_LEN = int(os.getenv("MAX_MODEL_LEN", str(_auto['MAX_MODEL_LEN'])))
BATCH_SIZE = int(os.getenv("BATCH_SIZE", str(_auto['BATCH_SIZE'])))
GPU_MEMORY_UTILIZATION = float(os.getenv("GPU_MEMORY_UTILIZATION", str(_auto['GPU_MEMORY_UTILIZATION'])))
TENSOR_PARALLEL_SIZE = int(os.getenv("TENSOR_PARALLEL_SIZE", str(_auto['TENSOR_PARALLEL_SIZE'])))

INPUT_DIR = Path(os.getenv("INPUT_DIR", "output_filtered"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "output_training_data"))
MODEL_CACHE_DIR = Path(os.getenv("MODEL_CACHE_DIR", str(SCRIPT_DIR / "models")))
MODEL_CACHE_DIR.mkdir(parents=True, exist_ok=True)

MAX_RETRIES = 3
INITIAL_BACKOFF = 1

# Prompt Template
SUPERVISOR_PROMPT = """You are an expert financial analyst AI creating high-quality training data for a stock prediction model.
You are provided with a financial document containing historical data and news up to a specific base date.

GROUND TRUTH OUTCOME (The Actual Future):
- Base Date Close: ${base_price:.2f}
- Actual Next Day Close: ${actual_price:.2f}
- Target Direction: {actual_dir} (1 = Up, -1 = Down, 0 = Unchanged)

TASK:
Write a highly detailed, logical 'chain_of_thought' that analyzes the provided financial document and flawlessly reverse-engineers the ground truth outcome. 
Your reasoning MUST sound be predictive (written before the market opened), highlighting the specific bullish or bearish signals in the text which justify the ground truth outcome. You must ONLY reference facts, metrics, and news explicitly stated in the provided document. Do NOT hallucinate or invent positive/negative news to justify the outcome. If the text is neutral, explain how subtle market context in the text led to the result.

REQUIREMENTS:
1. Synthesize the text into a chain of though reasoning leading logically to the ground truth Outcome.
2. Fill the 'output' EXACTLY with the Target Direction and Actual Next Day Close price.
3. In 'forecast', assign a probability of > 0.65 to the true direction, and distribute the rest.
4. Output ONLY a valid JSON object matching the requested schema. Start with '{{'.

Financial Document:
---
{document}
---
Output:
"""

# --- SCHEMAS ---
class Forecast(BaseModel):
    up: float
    down: float
    unchanged: float

class FinancialPrediction(BaseModel):
    next_day_direction: int
    next_day_closing_price: float
    forecast: Forecast

class ReasoningResponse(BaseModel):
    chain_of_thought: str = Field(description="Detailed financial reasoning based strictly on historical context that leads to the true outcome.")
    output: FinancialPrediction


# --- UTILITIES ---
def parse_filename(filename):
    """Extracts ticker and date from filename (eg. AEP_20240828_data_run_1)"""
    try:
        match = re.search(r'(\d{8})', filename)
        if not match: return None, None
        date_str = match.group(1)
        ticker_part = filename[:match.start()]
        ticker = ticker_part.rstrip('_')
        return ticker, date_str
    except Exception:
        return None, None

def get_market_data(ticker, date_str):
    """Fetches Base (T) and Target (T+1) market data using evaluation logic."""
    try:
        yf_ticker = ticker.replace("_", "-").replace(" ", "-").replace(".", "-").upper()
        start_date = datetime.strptime(date_str, "%Y%m%d")
        end_date = start_date + timedelta(days=10)

        dat = yf.Ticker(yf_ticker)
        df = dat.history(start=start_date, end=end_date)

        if df is None or df.empty or len(df) < 2:
            return None

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        base_day = df.iloc[0]
        next_day = df.iloc[1]

        base_close = base_day['Close'].item() if hasattr(base_day['Close'], 'item') else float(base_day['Close'])
        next_close = next_day['Close'].item() if hasattr(next_day['Close'], 'item') else float(next_day['Close'])
        
        # Determine actual direction
        if next_close > base_close: actual_dir = 1
        elif next_close < base_close: actual_dir = -1
        else: actual_dir = 0

        return {
            'base_date': str(df.index[0].date()),
            'base_close': base_close,
            'actual_next_date': str(df.index[1].date()),
            'actual_next_close': next_close,
            'actual_dir': actual_dir
        }
    except Exception as e:
        print(f"Error fetching {ticker}: {str(e)}")
        return None


class TrainingDataGenerator:
    def __init__(self, model_name, max_model_len, gpu_util):
        print(f"Initializing vLLM (Context: {max_model_len})...")
        
        self.llm = LLM(
            model=model_name,
            max_model_len=max_model_len,
            gpu_memory_utilization=gpu_util,
            trust_remote_code=True,
            dtype="bfloat16",
            download_dir=str(MODEL_CACHE_DIR),
            tensor_parallel_size=TENSOR_PARALLEL_SIZE,
            guided_decoding_backend="outlines" 
        )
        
        self.schema = ReasoningResponse.model_json_schema()
        
        self.sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.95,
            max_tokens=2048,
            guided_decoding=GuidedDecodingParams(json=self.schema)
        )
        
        self.tokenizer = self.llm.get_tokenizer()
        self.tokenizer.model_max_length = max_model_len
        self.max_context = max_model_len

    def truncate_document(self, document: str) -> str:
        tokens = self.tokenizer.encode(document)
        limit = self.max_context
        
        if len(tokens) > limit:
            tokens = tokens[:limit]
            document = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return document

    def generate_continuous(self, prompts: List[str]) -> List[str]:
        outputs = self.llm.generate(prompts, sampling_params=self.sampling_params)
        return [out.outputs[0].text.strip() for out in outputs]


def main():
    # Try to increase open file limits for large batch processing
    try:
        import resource
        soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
        resource.setrlimit(resource.RLIMIT_NOFILE, (hard, hard))
    except Exception:
        pass

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_files = sorted([f for f in INPUT_DIR.glob("*.json") if f.is_file()])
    if not input_files:
        print(f"No input files found in {INPUT_DIR}")
        return

    processed_files = set(f.name for f in OUTPUT_DIR.glob("*.json"))
    files_to_process = [f for f in input_files if f.name not in processed_files]
    
    if not files_to_process:
        print("All files already processed.")
        return
        
    total_files = len(files_to_process)
    print(f"Found {total_files} files to process.")

    generator = TrainingDataGenerator(MODEL_NAME, MAX_MODEL_LEN, GPU_MEMORY_UTILIZATION)

    idx = 0
    total_processed = 0
    while idx < total_files:
        batch_files = files_to_process[idx:idx+BATCH_SIZE]
        
        valid_files =[]
        prompts = []
        metadata_list =[]

        # 1. Fetch market data dynamically for the batch
        print(f"Fetching ground-truth market data for batch {idx//BATCH_SIZE+1}...")
        for file_obj in batch_files:
            ticker, date_str = parse_filename(file_obj.name)
            if not ticker: continue
            
            market_data = get_market_data(ticker, date_str)
            time.sleep(0.05)
            
            if not market_data:
                continue

            try:
                # Load existing document data
                with open(file_obj, 'r') as f:
                    doc_data = json.load(f)
                
                # Convert the raw dictionary to a formatted string for the prompt
                doc_text = json.dumps(doc_data, indent=2)
                doc_truncated = generator.truncate_document(doc_text)
                
                # Embed Next Day Target into Prompt
                prompt = SUPERVISOR_PROMPT.format(
                    base_price=market_data['base_close'],
                    actual_price=market_data['actual_next_close'],
                    actual_dir=market_data['actual_dir'],
                    document=doc_truncated
                )
                
                prompts.append(prompt)
                valid_files.append(file_obj)
                metadata_list.append((doc_data, market_data))
                
            except Exception as e:
                print(f"Failed to read/format {file_obj.name}: {e}")

        if not prompts:
            print("No valid data for this batch (likely market data missing). Skipping...")
            idx += BATCH_SIZE
            continue

        # 2. Run LLM Generation
        print(f"Generating teacher responses for {len(prompts)} files...")
        batch_outputs =[]
        for attempt in range(MAX_RETRIES):
            try:
                batch_outputs = generator.generate_continuous(prompts)
                break
            except Exception as e:
                print(f"Error during generation: {e}. Retrying ({attempt+1}/{MAX_RETRIES})...")
                time.sleep(INITIAL_BACKOFF * (2 ** attempt))
        else:
            print(f"Batch failed after {MAX_RETRIES} attempts. Skipping.")
            idx += BATCH_SIZE
            continue

        # 3. Save Final Ground-Truth Synthesized Data
        for file_obj, raw_output, (orig_doc, market_data) in zip(valid_files, batch_outputs, metadata_list):
            out_path = OUTPUT_DIR / file_obj.name

            with open(out_path, "w") as outfile:
                json.dump(raw_output, outfile, indent=2)

            # Increment overall counter and log every 100 files
            total_processed += 1
            if total_processed % 100 == 0:
                print(f"Processed {total_processed}/{total_files} files")

        print(f"Successfully processed batch {idx//BATCH_SIZE+1}.")
        idx += BATCH_SIZE

if __name__ == "__main__":
    main()