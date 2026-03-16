import os
import json

# Configuration
INPUT_DIR = 'output_training_data'
OUTPUT_DIR = 'output_training_data_json'
FAILED_DIR = 'output_training_data_failed'

def extract_json_objects(text):
    """
    Scans the string and extracts all valid top-level JSON objects.
    Returns a list of valid dictionaries.

    The input file sometimes contains a single JSON *string* whose value
    is the actual object (see example in issue).  If that happens we
    recursively decode the string and re-run the extraction so callers
    still receive a proper dict.
    """
    # Try to parse json
    if isinstance(text, str):
        try:
            maybe = json.loads(text)
            if isinstance(maybe, str):
                return extract_json_objects(maybe)
            elif isinstance(maybe, dict):
                return [maybe]
        except json.JSONDecodeError:
            pass

    decoder = json.JSONDecoder()
    pos = 0
    valid_objects = []

    # Loop through the text and extract valid JSON objects
    while pos < len(text):
        start_index = text.find('{', pos)
        
        # If no more braces, stop
        if start_index == -1:
            break
        
        try:
            obj, end_index = decoder.raw_decode(text, idx=start_index)
            valid_objects.append(obj)
            pos = end_index
        except json.JSONDecodeError:
            pos = start_index + 1
            
    return valid_objects

def process_files():
    # 1. Create output directories
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(FAILED_DIR, exist_ok=True)
    
    if not os.path.exists(INPUT_DIR):
        print(f"Error: Input directory '{INPUT_DIR}' does not exist.")
        return

    processed_count = 0
    failed_count = 0

    # 2. Iterate over files
    for filename in os.listdir(INPUT_DIR):
        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)
        failed_path = os.path.join(FAILED_DIR, filename)

        if not os.path.isfile(input_path):
            continue

        content = ""

        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                content = f.read()

            # 3. Extract all valid JSONs found in the file
            valid_jsons = extract_json_objects(content)

            if not valid_jsons:
                raise ValueError("No valid JSON object found in file.")

            # 4. Select the last valid json object found
            final_json = valid_jsons[-1]

            # 5. Success: Write the LAST valid JSON found
            with open(output_path, 'w', encoding='utf-8') as f_out:
                json.dump(final_json, f_out, indent=4)
            
            processed_count += 1

        except Exception as e:
            print(f"Failed {filename}: {e}")
            
            with open(failed_path, 'w', encoding='utf-8') as f_fail:
                f_fail.write(content)
            
            failed_count += 1

    print(f"Processing Complete.")
    print(f"Success: {processed_count} (saved to /{OUTPUT_DIR})")
    print(f"Failed:  {failed_count} (saved to /{FAILED_DIR})")

if __name__ == "__main__":
    process_files()