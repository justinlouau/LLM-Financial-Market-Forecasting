import os
import json
import torch
import gc
import traceback
import argparse
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoProcessor,
)
import numpy as np 

# TF32 Optimization for Hopper (H200)
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Configuration
MODELS_TO_BENCHMARK =[
    ("ChatTS-8B-SFT", "/srv/scratch/thesis/ChatTS-Training/exports/ChatTS-8B-SFT"),
]

INPUT_DIR = "output_filtered"
BASE_OUTPUT_DIR = "output_dpo_pairs" 

MAX_TS_POINTS = 256
MAX_NEW_TOKENS = 2048
MAX_INPUT_TOKENS = 32768

BATCH_SIZE = 8

def parse_args():
    parser = argparse.ArgumentParser(description="Run DPO Generation with Sharding")
    parser.add_argument("--shard-id", type=int, default=0, help="Index of the current shard (0 to num_shards-1)")
    parser.add_argument("--num-shards", type=int, default=1, help="Total number of shards/GPUs")
    return parser.parse_args()

def create_prompt(financial_data: dict, ts_stats: dict) -> str:
    # OPTIMIZATION 1: Minify JSON to drastically reduce token count & KV Cache memory
    financial_document_str = json.dumps(financial_data, separators=(',', ':'))

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
{json.dumps(example_json, separators=(',', ':'))}

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
    
    ts_keys =['open', 'high', 'low', 'close', 'volume']
    
    for filename in pending_files:
        try:
            with open(os.path.join(INPUT_DIR, filename), 'r', encoding='utf-8') as f:
                data = json.load(f)

            time_series_data = data.pop("stock_data", None)
            if not time_series_data or not isinstance(time_series_data, dict) or not time_series_data.get('open'):
                continue

            ts_tensors = []
            raw_ts_lists =[] 
            ts_stats = {} 
            
            for k in ts_keys:
                vals = time_series_data.get(k, [])
                if len(vals) > MAX_TS_POINTS:
                    vals = vals[-MAX_TS_POINTS:]
                
                vals_np = np.array(vals, dtype=np.float32)
                raw_ts_lists.append(vals)

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
                "prompt_length": len(prompt),
                "ts": ts_tensors,
                "raw_ts": raw_ts_lists
            })
            
        except Exception as e:
            print(f"[ERROR] Reading {filename}: {e}")
            
    # Sort by prompt string length instead of raw file size to reduce padding
    dataset.sort(key=lambda x: x["prompt_length"])
    
    print(f"Successfully pre-loaded {len(dataset)} items.")
    return dataset

def evaluate_model(model_name, model_path, shard_id, num_shards):
    current_output_dir = os.path.join(BASE_OUTPUT_DIR, model_name)
    os.makedirs(current_output_dir, exist_ok=True)

    print(f"\n{'='*60}\nStarting Evaluation for: {model_name} (Shard {shard_id+1}/{num_shards})\n{'='*60}")

    chosen_dtype = torch.bfloat16
    chosen_attn = "flash_attention_2"

    try:
        processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        processor.tokenizer.padding_side = "left" 
        if getattr(processor.tokenizer, "pad_token", None) is None:
            processor.tokenizer.pad_token = processor.tokenizer.eos_token
        
        eos_token_ids = [processor.tokenizer.eos_token_id]
        try:
            im_end_id = processor.tokenizer.encode("<|im_end|>", add_special_tokens=False)
            if isinstance(im_end_id, list):
                im_end_id = im_end_id[0]
            if im_end_id not in eos_token_ids:
                eos_token_ids.append(im_end_id)
        except Exception:
            pass
        
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

    # SHARDING LOGIC
    # 1. Get ALL files and sort them deterministically so every worker sees the same list
    all_files =[f for f in os.listdir(INPUT_DIR) if f.endswith('.json')]
    all_files.sort() 

    # 2. Divide files evenly based on modulo arithmetic
    sharded_files =[f for i, f in enumerate(all_files) if i % num_shards == shard_id]
    
    # 3. Filter out files that have already been generated (for resumption)
    pending_files =[f for f in sharded_files if not os.path.exists(os.path.join(current_output_dir, f))]
            
    # Pre-load data for this specific GPU's shard
    dataset = preload_data(pending_files)
    total_files = len(dataset)

    i = 0
    current_batch_size = BATCH_SIZE
    processed_files = 0
    progress_counter = 0
    
    while i < total_files:
        batch_items = dataset[i : i + current_batch_size]
        
        valid_filenames = [item["filename"] for item in batch_items]
        batch_prompts = [item["prompt"] for item in batch_items]
        
        batch_ts =[]
        for item in batch_items:
            batch_ts.extend(item["ts"])

        print(f"[Shard {shard_id}] Processing batch[{i+1} to {min(i+current_batch_size, total_files)}] / {total_files} (Batch Size: {current_batch_size})...")

        try:
            with torch.inference_mode():
                inputs = processor(
                    text=batch_prompts,
                    timeseries=batch_ts,
                    padding=True,
                    return_tensors="pt"
                ).to(model.device)

                for k, v in inputs.items():
                    if torch.is_floating_point(v):
                        inputs[k] = v.to(dtype=model.dtype)

                input_length = inputs.input_ids.shape[1]
                
                output_ids = model.generate(
                    **inputs,
                    max_new_tokens=MAX_NEW_TOKENS,
                    do_sample=True,
                    temperature=0.8, 
                    top_p=0.95,
                    num_return_sequences=2, 
                    pad_token_id=processor.tokenizer.pad_token_id,
                    eos_token_id=eos_token_ids,
                    use_cache=True 
                )

                generated_texts = processor.tokenizer.batch_decode(
                    output_ids[:, input_length:], 
                    skip_special_tokens=True
                )

                for idx, item in enumerate(batch_items):
                    filename = item["filename"]
                    output1 = generated_texts[idx * 2]
                    output2 = generated_texts[(idx * 2) + 1]
                    
                    dpo_data = {
                        "input": item["prompt"],
                        "output1": output1,
                        "output2": output2,
                        "timeseries": item["raw_ts"]
                    }

                    output_path = os.path.join(current_output_dir, filename) 
                    with open(output_path, 'w', encoding='utf-8') as f:
                        json.dump(dpo_data, f, indent=4)

                    processed_files += 1
                    progress_counter += 1
                    if progress_counter >= 100 or processed_files == total_files:
                        print(f"[Shard {shard_id}] Progress: {processed_files} / {total_files} files processed.")
                        progress_counter = 0

                del inputs, output_ids
                
            i += current_batch_size

        except torch.cuda.OutOfMemoryError:
            print(f"[WARN] Shard {shard_id} OOM Error. Halving batch size from {current_batch_size} to {max(1, current_batch_size // 2)} and retrying.", flush=True)
            
            if 'inputs' in locals():
                del inputs
            if 'output_ids' in locals():
                del output_ids
            gc.collect()
            torch.cuda.empty_cache()
            
            if current_batch_size == 1:
                print(f"  [ERROR] Cannot process sequence even with batch size 1. Skipping {valid_filenames[0]}.", flush=True)
                i += 1
            else:
                current_batch_size = max(1, current_batch_size // 2)
                
        except Exception as e:
            print(f"  [ERROR] Inference failed for batch starting with {valid_filenames[0]}")
            traceback.print_exc()
            i += current_batch_size

    del model
    del processor
    gc.collect()
    torch.cuda.empty_cache()
    print(f"Finished evaluation for {model_name} on Shard {shard_id}.\n")

def main():
    args = parse_args()
    print(f"--- Starting ChatTS-8B DPO Generation Suite (Shard {args.shard_id+1}/{args.num_shards}) ---")
    for name, path in MODELS_TO_BENCHMARK:
        evaluate_model(name, path, args.shard_id, args.num_shards)
    print(f"--- DPO Generation Suite Finished for Shard {args.shard_id+1} ---")

if __name__ == "__main__":
    main()