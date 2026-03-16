import pandas as pd
import os
from collections import defaultdict
from typing import List, Dict
import time

# Configuration
INPUT_FILENAME = 'FNSPID/nasdaq_exteral_data.csv'
OUTPUT_DIR = 'news'

TARGET_STOCKS = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'NFLX']
CHUNK_SIZE = 10000 

def process_large_csv_optimized(input_file: str, output_dir: str, target_symbols: List[str], chunk_size: int = 10000):
    """
    Optimized function to process large CSV files by reading in chunks.
    Saves each target stock to its own CSV file.
    
    Args:
        input_file: Path to the input CSV file
        output_dir: Directory to save individual stock CSV files
        target_symbols: List of stock symbols to extract
        chunk_size: Number of rows to process at a time
    """
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found.")
        return

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Dictionary to store file handles for each stock symbol
    file_handles: Dict[str, any] = {}
    headers_written: Dict[str, bool] = {}
    stock_counts: Dict[str, int] = defaultdict(int)
    
    try:
        print(f"Starting to process large CSV file: {input_file}")
        print(f"Target symbols: {target_symbols}")
        print(f"Chunk size: {chunk_size}")
        
        start_time = time.time()
        total_rows_processed = 0
        chunks_processed = 0
        
        # Read the CSV file in chunks
        for chunk in pd.read_csv(input_file, chunksize=chunk_size):
            chunks_processed += 1
            chunk_start_time = time.time()
            
            # Filter the chunk for target symbols
            filtered_chunk = chunk[chunk['Stock_symbol'].isin(target_symbols)]
            
            if not filtered_chunk.empty:
                # Group by stock symbol and write to respective files
                for symbol in target_symbols:
                    symbol_data = filtered_chunk[filtered_chunk['Stock_symbol'] == symbol]
                    
                    if not symbol_data.empty:
                        output_file = os.path.join(output_dir, f'{symbol}.csv')
                        
                        # Write header only for the first chunk of each symbol
                        if symbol not in headers_written:
                            symbol_data.to_csv(output_file, mode='w', index=False, encoding='utf-8')
                            headers_written[symbol] = True
                        else:
                            # Append data without header
                            symbol_data.to_csv(output_file, mode='a', header=False, index=False, encoding='utf-8')
                        
                        stock_counts[symbol] += len(symbol_data)
            
            total_rows_processed += len(chunk)
            chunk_time = time.time() - chunk_start_time
            
            # Progress update every 100 chunks
            if chunks_processed % 100 == 0:
                elapsed_time = time.time() - start_time
                print(f"Processed {chunks_processed} chunks ({total_rows_processed:,} rows) in {elapsed_time:.2f}s. "
                      f"Chunk processing time: {chunk_time:.3f}s")
        
        total_time = time.time() - start_time
        
        print("\n" + "="*50)
        print("Processing complete!")
        print(f"Total time: {total_time:.2f} seconds")
        print(f"Total rows processed: {total_rows_processed:,}")
        print(f"Total chunks processed: {chunks_processed}")
        print(f"Average rows per second: {total_rows_processed/total_time:.0f}")
        
        # Print results for each stock
        print("\nResults by stock symbol:")
        for symbol in target_symbols:
            count = stock_counts[symbol]
            if count > 0:
                output_file = os.path.join(output_dir, f'{symbol}.csv')
                print(f"  {symbol}: {count:,} articles saved to '{output_file}'")
            else:
                print(f"  {symbol}: No articles found")
        
        total_articles = sum(stock_counts.values())
        print(f"\nTotal articles saved: {total_articles:,}")
        
    except Exception as e:
        print(f"An error occurred: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Clean up any open file handles
        for handle in file_handles.values():
            if handle and not handle.closed:
                handle.close()


# --- Main execution ---
if __name__ == "__main__":
    # Display file information
    if os.path.exists(INPUT_FILENAME):
        print(f"Input file: {INPUT_FILENAME}")
        print(f"Output directory: {OUTPUT_DIR}")
        print("-" * 50)
    
    process_large_csv_optimized(INPUT_FILENAME, OUTPUT_DIR, TARGET_STOCKS, CHUNK_SIZE)