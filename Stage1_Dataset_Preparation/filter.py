# filter.py
from __future__ import annotations

import argparse
import json
import copy
import concurrent.futures
from pathlib import Path
from typing import Any, Dict, List, Callable, Tuple
import tiktoken

def truncate_stock_data(data: Any) -> Any:
    """
    Recursively traverse and truncate floats to 2 decimal places in stock_data.
    """
    if isinstance(data, float):
        return round(data, 2)
    elif isinstance(data, dict):
        return {k: truncate_stock_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [truncate_stock_data(v) for v in data]
    return data

def make_token_counter(encoding_name: str = "qwen") -> Callable[[str], int]:
    """
    Return an estimate_tokens(text) function using tiktoken when available.
    """
    # 1. Fallback if tiktoken is missing
    if tiktoken is None:
        return lambda s: len(s.split()) if s else 0

    # 2. Try to load the specific encoding, fallback to cl100k_base, then simple count
    try:
        enc = tiktoken.get_encoding(encoding_name)
    except Exception:
        try:
            enc = tiktoken.get_encoding("cl100k_base")
        except Exception:
            return lambda s: len(s.split()) if s else 0

    def _encode_count(s: str) -> int:
        if not s:
            return 0
        try:
            return len(enc.encode(s))
        except Exception:
            return len(s.split())

    return _encode_count


def strip_metadata_and_join_paragraphs(data: Dict[str, Any], in_place: bool = False) -> Dict[str, Any]:
    """
    Strips scoring metadata and joins paragraphs for SEC reports.
    
    Args:
        data: The dictionary to process.
        in_place: If True, modifies the data directly. If False, creates a deep copy first.
    """
    processed_data = data if in_place else copy.deepcopy(data)

    # Truncate stock data floats to 2 decimal places
    if "stock_data" in processed_data:
        processed_data["stock_data"] = truncate_stock_data(processed_data["stock_data"])

    # Strip metadata from news articles
    for key in processed_data.keys():
        if ("news" in key or "article" in key) and isinstance(processed_data[key], list):
            for item in processed_data[key]:
                if isinstance(item, dict):
                    item.pop("score", None)

    # Strip metadata from and join paragraphs in SEC sections
    sec_keys = ['sec_report', 'latest_10k', 'latest_10q', 'eight_k_reports']
    
    def process_report_list(report_list: List[Dict]):
        for report in report_list:
            if isinstance(report, dict) and "paragraphs" in report:
                # Extract text, join, and remove the list of paragraph objects
                texts = [
                    p['text'] for p in report.get("paragraphs", []) 
                    if isinstance(p, dict) and 'text' in p
                ]
                report['report'] = '\n\n'.join(texts)
                del report["paragraphs"]

    for sec_key in sec_keys:
        sec_section = processed_data.get(sec_key)
        if isinstance(sec_section, list):
            process_report_list(sec_section)
        elif isinstance(sec_section, dict):
            process_report_list([sec_section])

    return processed_data


def calculate_json_tokens(obj: Any, estimate_tokens_fn: Callable[[str], int]) -> int:
    """
    Helper to dump JSON and count tokens. 
    Assumes obj is already in the final structure (metadata stripped).
    """
    json_str = json.dumps(obj, ensure_ascii=False, separators=(',', ':'))
    return estimate_tokens_fn(json_str)


def get_final_token_count(obj: Any, estimate_tokens_fn: Callable[[str], int], destructive: bool = False) -> int:
    """
    Estimates token count of the final processed form.
    
    Args:
        destructive: If True, allows modifying 'obj' in place to save copying time. 
                     Only use this on temporary objects!
    """
    if not isinstance(obj, dict):
        return 0
    
    processed_obj = strip_metadata_and_join_paragraphs(obj, in_place=destructive)
    return calculate_json_tokens(processed_obj, estimate_tokens_fn)


def apply_filters(data: Dict[str, Any], token_limit: int, estimate_tokens_fn: Callable[[str], int]) -> Dict[str, Any]:
    """
    Apply successive filters until token count is below token_limit.
    """
    working_data = copy.deepcopy(data)

    filter_levels = [
        {'type': 'remove_news', 'score': 0},
        {'type': 'strip_news_body', 'score': 1},
        {'type': 'remove_sec_para', 'score': 0},
        {'type': 'strip_news_body', 'scores': [2, 3]},
        {'type': 'remove_sec_para', 'score': 1},
        {'type': 'remove_sec_para', 'score': 2},
        {'type': 'remove_sec_para', 'score': 3},
        {'type': 'remove_sec_para', 'score': 4},
        {'type': 'remove_sec_para', 'score': 5},
        {'type': 'remove_sec_para', 'score': 6},
        {'type': 'remove_sec_para', 'score': 7},
        {'type': 'remove_sec_para', 'score': 8},
        {'type': 'remove_sec_para', 'score': 9},
    ]

    news_keys = [k for k in working_data if "news" in k or "article" in k]
    sec_keys = ['sec_report', 'latest_10k', 'latest_10q', 'eight_k_reports']

    for level in filter_levels:
        snapshot_before_level = copy.deepcopy(working_data)
        actions_to_perform: List[Tuple[str, Any, Any, str]] = [] 

        # Step 1: Identify Candidates
        if level['type'] == 'remove_news':
            for key in news_keys:
                if isinstance(working_data.get(key), list):
                    indices = [i for i, art in enumerate(working_data[key]) 
                               if isinstance(art, dict) and art.get("score") == level['score']]
                    for index in sorted(indices, reverse=True):
                        actions_to_perform.append(('pop_list_item', key, index, None))

        elif level['type'] == 'strip_news_body':
            scores = level.get('scores', [level.get('score')])
            for key in news_keys:
                if isinstance(working_data.get(key), list):
                    for i, article in enumerate(working_data[key]):
                        if isinstance(article, dict) and article.get("score") in scores and 'text' in article:
                            actions_to_perform.append(('remove_dict_key', key, i, 'text'))

        elif level['type'] == 'remove_sec_para':
            # Gather all paragraphs to remove across all reports
            for key in sec_keys:
                sec_section = working_data.get(key)
                if isinstance(sec_section, list):
                    for r_idx, report in enumerate(sec_section):
                        if isinstance(report, dict) and 'paragraphs' in report:
                            indices = [p_idx for p_idx, p in enumerate(report['paragraphs']) 
                                       if isinstance(p, dict) and p.get("score") == level['score']]
                            for p_idx in sorted(indices, reverse=True):
                                actions_to_perform.append(('pop_para_from_list', key, r_idx, p_idx))
                elif isinstance(sec_section, dict) and 'paragraphs' in sec_section:
                     indices = [p_idx for p_idx, p in enumerate(sec_section['paragraphs']) 
                                if isinstance(p, dict) and p.get("score") == level['score']]
                     for p_idx in sorted(indices, reverse=True):
                         actions_to_perform.append(('pop_para_from_dict', key, None, p_idx))

        if not actions_to_perform:
            continue

        # Step 2: Helper function to execute an action on a given data dict
        def execute_action(d, action):
            act_type, k, idx1, idx2 = action
            if act_type == 'pop_list_item':
                if k in d and idx1 < len(d[k]):
                    d[k].pop(idx1)
            elif act_type == 'remove_dict_key':
                if k in d and idx1 < len(d[k]):
                    d[k][idx1].pop(idx2, None)
            elif act_type == 'pop_para_from_list':
                d[k][idx1]['paragraphs'].pop(idx2)
            elif act_type == 'pop_para_from_dict':
                d[k]['paragraphs'].pop(idx2)

        # Step 3: Apply all actions to the working copy
        for action in actions_to_perform:
            execute_action(working_data, action)

        current_tokens = get_final_token_count(working_data, estimate_tokens_fn)
        
        if current_tokens <= token_limit:
            fine_grained_data = snapshot_before_level
            
            # Apply actions one by one and check
            for action in actions_to_perform:
                execute_action(fine_grained_data, action)
                
                # Make a copy to test the token count
                test_copy = copy.deepcopy(fine_grained_data)
                if get_final_token_count(test_copy, estimate_tokens_fn, destructive=True) <= token_limit:
                    return fine_grained_data
            
            # Fallback if loop finishes without returning
            return fine_grained_data

    # If all filters applied and still over limit:
    print(f"Warning: Could not reduce below {token_limit} tokens. Returning minimal data.")
    return {
        "company_name": data.get("company_name"),
        "ticker": data.get("ticker"),
        "date_range": data.get("date_range"),
        "stock_data": data.get("stock_data"),
    }


def process_file(args_ns: argparse.Namespace, file_path: Path, tokenizer_name: str):
    """
    Orchestrates the filtering and processing for a single file.
    Note: We pass tokenizer_name and init inside to be process-safe.
    """
    estimate_tokens_fn = make_token_counter(tokenizer_name)

    try:
        with file_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except (IOError, json.JSONDecodeError) as e:
        print(f"Skipping {file_path.name}: Failed to read or parse JSON. Error: {e}")
        return

    # Truncate stock_data immediately after loading
    if "stock_data" in data:
        data["stock_data"] = truncate_stock_data(data["stock_data"])

    # Check initial size
    original_tokens = get_final_token_count(data, estimate_tokens_fn)
    processed_data = data

    if original_tokens > args_ns.token_count:
        processed_data = apply_filters(data, args_ns.token_count, estimate_tokens_fn)
    
    if args_ns.debug:
        intermediate_path = args_ns.intermediate_dir / file_path.name
        with intermediate_path.open("w", encoding="utf-8") as f:
            json.dump(processed_data, f, ensure_ascii=False, indent=2)

    # Final transformation
    final_data = strip_metadata_and_join_paragraphs(processed_data, in_place=True)

    output_path = args_ns.output_dir / file_path.name
    final_minified_str = json.dumps(final_data, ensure_ascii=False, separators=(',', ':'))
    
    with output_path.open("w", encoding="utf-8") as f:
        f.write(final_minified_str)
    
    final_tokens = estimate_tokens_fn(final_minified_str)
    print(f"Processed {file_path.name} -> {output_path.name} (Original: {original_tokens}, Final: {final_tokens} tokens)")


def main():
    parser = argparse.ArgumentParser(description="Filter and process JSON files to a token limit.")
    parser.add_argument("--token-count", type=int, required=True, help="Mandatory: The target token count.")
    parser.add_argument("--input-dir", type=Path, default="./output_processed", help="Directory containing input JSON files.")
    parser.add_argument("--output-dir", type=Path, default="./output_filtered", help="Directory to save final processed JSON files.")
    parser.add_argument("--tokenizer", type=str, default="qwen", help="Tokenizer to use for counting.")
    parser.add_argument("--debug", action="store_true", help="If set, saves intermediate filtered file.")
    parser.add_argument("--intermediate-dir", type=Path, default="./output_filtered_intermediate", help="Directory for intermediate files if --debug is set.")

    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.debug:
        args.intermediate_dir.mkdir(parents=True, exist_ok=True)

    input_files = list(args.input_dir.glob("*.json"))
    files_to_process = []
    for p in input_files:
        output_path = args.output_dir / p.name
        # Resume logic - skip if output file already exists and is non-empty
        if output_path.exists() and output_path.stat().st_size > 0:
            print(f"Skipping {p.name}: Output already exists and is non-empty.")
            continue
        files_to_process.append(p)

    if not files_to_process:
        print(f"No new JSON files to process in {args.input_dir}")
        return

    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [
            executor.submit(process_file, args, p, args.tokenizer)
            for p in files_to_process
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                print(f"An error occurred during file processing: {e}")


if __name__ == "__main__":
    main()