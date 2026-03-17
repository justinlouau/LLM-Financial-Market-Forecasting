import json
import os
import argparse
from pathlib import Path

def extract_jsons_from_text(text):
    """
    Attempts to find all valid JSON objects enclosed in {} 
    within a string containing other text.
    Returns a list of parsed JSON objects.
    """
    # 1. Try parsing the whole text first 
    try:
        return [json.loads(text)]
    except json.JSONDecodeError:
        pass

    # 2. Extract all JSON candidates using JSONDecoder
    jsons =[]
    decoder = json.JSONDecoder()
    idx = 0
    while idx < len(text):
        idx = text.find('{', idx)
        if idx == -1:
            break
        try:
            result, index = decoder.raw_decode(text[idx:])
            jsons.append(result)
            # Move the index forward by the length of the parsed JSON
            idx += index
        except json.JSONDecodeError:
            # If the bracket wasn't a valid JSON start, move forward by 1
            idx += 1
            
    return jsons

def process_json_folder(input_folder_path):
    input_path = Path(input_folder_path)
    
    # Check if input directory exists
    if not input_path.exists() or not input_path.is_dir():
        print(f"Error: The input folder '{input_folder_path}' does not exist or is not a directory.")
        return

    # Define the output folder name
    input_path = input_path.resolve()
    output_path = input_path.parent / (input_path.name + "_clean")
    
    # Create the output directory
    os.makedirs(output_path, exist_ok=True)
    
    print(f"Source Folder: {input_path}")
    print(f"Target Folder: {output_path}\n")

    valid_files_count = 0
    invalid_files_count = 0
    
    # Dictionary to track the status of each Ticker_Date group
    group_tracker = {}

    # Look for *.txt files
    for file_path in input_path.glob('*.txt'):
        
        # 1. Identify Metadata (Ticker + Date)
        filename_stem = file_path.stem # Filename without extension
        
        if "_data_run_" in filename_stem:
            group_id = filename_stem.split("_data_run_")[0]
        else:
            group_id = filename_stem

        if group_id not in group_tracker:
            group_tracker[group_id] = False

        # 2. Process Data
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()

            # Extract all potential JSON blocks from mixed text
            candidate_jsons = extract_jsons_from_text(raw_content)

            valid_data = None
            error_msg = "No JSON brackets '{}' found or no valid JSON block parsed."

            # Iterate through all extracted JSONs backwards to get last valid JSON
            for data in reversed(candidate_jsons):
                if not isinstance(data, dict):
                    error_msg = "Extracted JSON is not a dict"
                    continue
                
                # Wrap JSON if needed
                if 'chain_of_thought' not in data and 'output' not in data:
                    if 'next_day_direction' in data and 'next_day_closing_price' in data:
                        data = {
                            "chain_of_thought": raw_content.strip(),
                            "output": data
                        }

                # Perform schema validation on the candidate
                if 'chain_of_thought' not in data or not isinstance(data['chain_of_thought'], str):
                    error_msg = "missing or invalid 'chain_of_thought'"
                    continue
                if 'output' not in data or not isinstance(data['output'], dict):
                    error_msg = "missing or invalid 'output' key"
                    continue

                # Check required output fields
                output = data['output']
                if 'next_day_direction' not in output or not isinstance(output['next_day_direction'], int):
                    error_msg = "missing or invalid 'next_day_direction' in 'output'"
                    continue
                if 'next_day_closing_price' not in output or not isinstance(output['next_day_closing_price'], (float, int)):
                    error_msg = "missing or invalid 'next_day_closing_price' in 'output'"
                    continue
                if 'forecast' not in output or not isinstance(output['forecast'], dict):
                    error_msg = "missing or invalid 'forecast' in 'output'"
                    continue
                    
                forecast = output['forecast']
                valid_forecast = True
                
                for key in ['up', 'down', 'unchanged']:
                    if key not in forecast or not isinstance(forecast[key], (float, int)):
                        error_msg = f"missing or invalid '{key}' in 'forecast'"
                        valid_forecast = False
                        break
                
                # If all checks passed, we found our valid JSON! Stop looking.
                if valid_forecast:
                    valid_data = data
                    break 

            if valid_data is not None:
                # Save as .json
                output_filename = file_path.with_suffix('.json').name
                output_file_path = output_path / output_filename

                with open(output_file_path, 'w', encoding='utf-8') as f_out:
                    json.dump(valid_data, f_out, indent=4, ensure_ascii=False)
                print(f"[OK] {file_path.name} -> {output_filename}")
                valid_files_count += 1
                group_tracker[group_id] = True
            else:
                # No candidates matched the schema
                print(f"[SKIP] Invalid format in {file_path.name}: {error_msg}")
                invalid_files_count += 1

        except Exception as e:
            print(f"[ERROR] Could not process {file_path.name}: {e}")
            invalid_files_count += 1

    # Calculate stats
    failed_groups =[gid for gid, success in group_tracker.items() if not success]
    
    print("-" * 30)
    print(f"Processing Complete.")
    print(f"Individual Files Valid:   {valid_files_count}")
    print(f"Individual Files Invalid: {invalid_files_count}")
    print("-" * 30)
    print(f"Total Ticker/Date Groups found: {len(group_tracker)}")
    print(f"Groups with NO valid runs:      {len(failed_groups)}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean JSON/TXT files and identify completely failed runs.")
    parser.add_argument("folder", type=str, help="The path to the folder containing the files.")
    args = parser.parse_args()

    process_json_folder(args.folder)