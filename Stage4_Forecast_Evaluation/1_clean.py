import json
import os
import argparse
from pathlib import Path

def extract_json_from_text(text):
    """
    Attempts to find the largest outer JSON object enclosed in {} 
    within a string containing other text.
    """
    try:
        # 1. Try parsing the whole text first (fast path for pure JSON)
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Fallback to extracting content between the first { and the last }
    start_index = text.find('{')
    end_index = text.rfind('}')

    if start_index != -1 and end_index != -1 and end_index > start_index:
        # Extract the content from the first { to the last }
        json_candidate = text[start_index : end_index + 1]
        try:
            return json.loads(json_candidate)
        except json.JSONDecodeError as e:
            # Re-raise with specific context if extraction failed to produce valid JSON
            raise ValueError(f"Extracted content is not valid JSON: {e}")
    else:
        raise ValueError("No JSON brackets '{}' found in text.")

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
        
        # 1. IDENTIFY THE GROUP (Ticker + Date)
        filename_stem = file_path.stem # Filename without extension
        
        if "_data_run_" in filename_stem:
            group_id = filename_stem.split("_data_run_")[0]
        else:
            group_id = filename_stem

        if group_id not in group_tracker:
            group_tracker[group_id] = False

        # 2. PROCESS JSON
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                raw_content = f.read()

            # Extract JSON from mixed text
            data = extract_json_from_text(raw_content)


            # Perform schema validation on the extracted JSON
            if not isinstance(data, dict):
                print(f"[SKIP] Invalid format in {file_path.name}: not a dict")
                invalid_files_count += 1
                continue
            if 'chain_of_thought' not in data or not isinstance(data['chain_of_thought'], str):
                print(f"[SKIP] Invalid format in {file_path.name}: missing or invalid 'chain_of_thought'")
                invalid_files_count += 1
                continue
            if 'output' not in data or not isinstance(data['output'], dict):
                print(f"[SKIP] Invalid format in {file_path.name}: missing or invalid 'output' key")
                invalid_files_count += 1
                continue

            # Check required output fields
            output = data['output']
            if 'next_day_direction' not in output or not isinstance(output['next_day_direction'], int):
                print(f"[SKIP] Invalid format in {file_path.name}: missing or invalid 'next_day_direction' in 'output'")
                invalid_files_count += 1
                continue
            if 'next_day_closing_price' not in output or not (isinstance(output['next_day_closing_price'], float) or isinstance(output['next_day_closing_price'], int)):
                print(f"[SKIP] Invalid format in {file_path.name}: missing or invalid 'next_day_closing_price' in 'output'")
                invalid_files_count += 1
                continue
            if 'forecast' not in output or not isinstance(output['forecast'], dict):
                print(f"[SKIP] Invalid format in {file_path.name}: missing or invalid 'forecast' in 'output'")
                invalid_files_count += 1
                continue
            forecast = output['forecast']
            for key in ['up', 'down', 'unchanged']:
                if key not in forecast or not (isinstance(forecast[key], float) or isinstance(forecast[key], int)):
                    print(f"[SKIP] Invalid format in {file_path.name}: missing or invalid '{key}' in 'forecast'")
                    invalid_files_count += 1
                    break
            else:
                # Save as .json
                output_filename = file_path.with_suffix('.json').name
                output_file_path = output_path / output_filename

                with open(output_file_path, 'w', encoding='utf-8') as f_out:
                    json.dump(data, f_out, indent=4, ensure_ascii=False)
                print(f"[OK] {file_path.name} -> {output_filename}")
                valid_files_count += 1
                group_tracker[group_id] = True
                continue

            # Save as .json
            output_filename = file_path.with_suffix('.json').name
            output_file_path = output_path / output_filename

            with open(output_file_path, 'w', encoding='utf-8') as f_out:
                json.dump(data, f_out, indent=4, ensure_ascii=False)
            print(f"[OK] {file_path.name} -> {output_filename}")
            valid_files_count += 1
            group_tracker[group_id] = True

        except (json.JSONDecodeError, ValueError) as e:
            print(f"[SKIP] Invalid JSON in {file_path.name}: {e}")
            invalid_files_count += 1
            
        except Exception as e:
            print(f"[ERROR] Could not process {file_path.name}: {e}")
            invalid_files_count += 1

    # Calculate stats
    failed_groups = [gid for gid, success in group_tracker.items() if not success]
    
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