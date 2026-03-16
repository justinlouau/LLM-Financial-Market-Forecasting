import os
import json
import torch
import gc
import traceback
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
)
import numpy as np 
from concurrent.futures import ThreadPoolExecutor

# TF32 Optimization for Hopper (H200)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Configuration 
MODELS_TO_BENCHMARK =[
    ("Base-8B", "bytedance-research/ChatTS-8B"),
    ("ChatTS-8B-Official-Finance-LoRA-Merged", "/srv/scratch/z5218709/thesis/ChatTS-Training/exports/ChatTS-8B-Official-Finance-LoRA-Merged"),
    ("ChatTS-8B-Stage-4-Base", "/srv/scratch/z5218709/thesis/ChatTS-Training/exports/ChatTS-8B-Stage-4-Base"),
    ("ChatTS-8B-Stage-4-Custom", "/srv/scratch/z5218709/thesis/ChatTS-Training/exports/ChatTS-8B-Stage-4-Custom"),
]

INPUT_DIR = "output_filtered"
BASE_OUTPUT_DIR = "output"

MAX_TS_POINTS = 256
MAX_NEW_TOKENS = 2048
MAX_INPUT_TOKENS = 10000

BATCH_SIZE = 32

def save_outputs_async(filenames, generated_texts, output_dir):
    """Saves generated outputs to disk in a background thread."""
    for filename, text in zip(filenames, generated_texts):
        base_name = os.path.splitext(filename)[0]
        output_path = os.path.join(output_dir, f"{base_name}.txt")
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)

def create_prompt(financial_data: dict, ts_stats: dict) -> str:
    financial_document_str = json.dumps(financial_data, indent=4)

    # Create a string representation of the stats for the LLM to read
    # Format: "Series 1 (Open): Min 150.0, Max 180.0..."
    stats_str = "\n".join([
        f"{k.capitalize()}: Min {v['min']:.4f}, Max {v['max']:.4f}" 
        for k, v in ts_stats.items()
    ])

    example_json = {
        "chain_of_thought": "Step-by-step analysis...",
        "output": {
            "next_day_direction": 1,
            "next_day_closing_price": 172.2,
            "forecast": {"up": 0.60, "down": 0.20, "unchanged": 0.20},
        }
    }
    
    return f"""<|im_start|>system
You are a financial analyst LLM. Given the following financial document and time series data, perform chain-of-thought reasoning to predict the stock price movement.

The time series data provided is normalized. Use the following statistics to understand the real scale of values:
{stats_str}

Your response must be a valid JSON object strictly adhering to the following schema example:
{json.dumps(example_json, indent=4)}

<|im_end|>
<|im_start|>user
---
FINANCIAL DOCUMENT:
{financial_document_str}
---
TIME SERIES DATA:
Here are the stock OHLCV time series (Open, High, Low, Close, Volume): <ts><ts/> <ts><ts/> <ts><ts/> <ts><ts/> <ts><ts/>
---
<|im_end|>
<|im_start|>assistant
"""

def preload_data(pending_files):
    print("Pre-loading and processing dataset into RAM...")
    dataset = []
    
    ts_keys = ['open', 'high', 'low', 'close', 'volume']
    
    for filename in pending_files:
        try:
            with open(os.path.join(INPUT_DIR, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)

            time_series_data = data.pop("stock_data", None)
            if not time_series_data or not isinstance(time_series_data, dict) or not time_series_data.get('open'):
                continue

            ts_tensors = []
            ts_stats = {}
            
            for k in ts_keys:
                vals = time_series_data.get(k, [])
                if len(vals) > MAX_TS_POINTS:
                    vals = vals[-MAX_TS_POINTS:]
                
                # Convert to Numpy and calculate stats for normalization
                vals_np = np.array(vals, dtype=np.float32)
                if len(vals_np) > 0:
                    v_min = float(vals_np.min())
                    v_max = float(vals_np.max())
                else:
                    v_min, v_max = 0.0, 1.0

                ts_stats[k] = {"min": v_min, "max": v_max}
                ts_tensors.append(vals_np)

            prompt = create_prompt(data, ts_stats)
            
            dataset.append({
                "filename": filename,
                "prompt": prompt,
                "ts": ts_tensors
            })
            
        except Exception as e:
            print(f"[ERROR] Reading {filename}: {e}")
            
    print(f"Successfully pre-loaded {len(dataset)} items.")
    return dataset

def evaluate_model(model_name, model_path):
    current_output_dir = os.path.join(BASE_OUTPUT_DIR, model_name)
    os.makedirs(current_output_dir, exist_ok=True)

    print(f"\n{'='*60}\nStarting Evaluation for: {model_name}\n{'='*60}")

    chosen_dtype = torch.bfloat16
    chosen_attn = "flash_attention_2"

    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        processor.tokenizer.padding_side = "left" 
        if getattr(processor.tokenizer, "pad_token", None) is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            trust_remote_code=True,
            torch_dtype=chosen_dtype,
            device_map="cuda:0",
            attn_implementation=chosen_attn,
            low_cpu_mem_usage=True
        )
        model.eval()
        
    except Exception as e:
        print(f"[ERROR] Failed to load model {model_name}: {e}")
        return

    all_files =[f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]
    pending_files =[f for f in all_files if not os.path.exists(os.path.join(current_output_dir, f"{os.path.splitext(f)[0]}.txt"))]
    
    # Pre-load all data to remove CPU bottleneck
    dataset = preload_data(pending_files)

    # Sort exactly by prompt length to tightly group sequence lengths and minimize padding VRAM/compute
    dataset.sort(key=lambda x: len(x["prompt"]))
    total_files = len(dataset)

    # Thread executor for non-blocking file writes
    executor = ThreadPoolExecutor(max_workers=4)

    # Batched Processing Loop
    i = 0
    current_batch_size = BATCH_SIZE
    
    processed_files = 0
    while i < total_files:
        # Dynamic batch sizing logic for OOM recovery
        batch_items = dataset[i : i + current_batch_size]
        
        valid_filenames = [item["filename"] for item in batch_items]
        batch_prompts = [item["prompt"] for item in batch_items]
        
        # Flatten TS tensors for the processor
        batch_ts =[]
        for item in batch_items:
            batch_ts.extend(item["ts"])

        print(f"Processing batch[{i+1} to {min(i+current_batch_size, total_files)}] / {total_files} (Batch Size: {current_batch_size})...")

        try:
            with torch.inference_mode():
                inputs = processor(
                    text=batch_prompts,
                    timeseries=batch_ts,
                    padding=True,
                    return_tensors="pt"
                )

                for k, v in inputs.items():
                    if torch.is_floating_point(v):
                        inputs[k] = v.to(device=model.device, dtype=model.dtype)
                    else:
                        inputs[k] = v.to(device=model.device)

                input_length = inputs.input_ids.shape[1]
                
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=0.7,
                    top_p=0.9,
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=processor.tokenizer.eos_token_id,
                    use_cache=True
                )

                generated_texts = processor.tokenizer.batch_decode(
                    output_ids[:, input_length:], 
                    skip_special_tokens=True
                )

                # Submit to background thread (Non-blocking) so GPU can proceed immediately
                executor.submit(save_outputs_async, valid_filenames, generated_texts, current_output_dir)

                processed_files += len(valid_filenames)
                if processed_files % 100 < current_batch_size or processed_files == total_files:
                    print(f"Progress: {processed_files} / {total_files} files processed.")

                del inputs, output_ids
                
            # Move to next batch, reset batch size if it was lowered due to OOM
            i += current_batch_size
            current_batch_size = BATCH_SIZE 

        except torch.cuda.OutOfMemoryError:
            # Catch OOM, clear cache, and halve the batch size
            print(f"  [WARN] OOM Error encountered. Halving batch size from {current_batch_size} to {max(1, current_batch_size // 2)} and retrying.")
            torch.cuda.empty_cache()
            if current_batch_size == 1:
                print(f"  [ERROR] Cannot process sequence even with batch size 1. Skipping {valid_filenames[0]}.")
                i += 1
            else:
                current_batch_size = max(1, current_batch_size // 2)
                
        except Exception as e:
            print(f"  [ERROR] Inference failed for batch starting with {valid_filenames[0]}")
            traceback.print_exc()
            i += current_batch_size # Skip batch on non-OOM errors

    # Cleanup 
    executor.shutdown(wait=True)
    del model
    del processor
    gc.collect()
    torch.cuda.empty_cache()
    
    print(f"Finished evaluation for {model_name}.\n")

def main():
    print("--- Starting ChatTS-8B Benchmark Suite ---")
    for name, path in MODELS_TO_BENCHMARK:
        evaluate_model(name, path)
    print("--- Benchmark Suite Finished ---")

if __name__ == "__main__":
    main()