import shutil
from pathlib import Path
from typing import List

def copy_stock_files(tickers: List[str], 
                    source_dir: str = "../FNSPID/full_history", 
                    dest_dir: str = "../prices",
                    overwrite: bool = False) -> None:
    """
    Copy stock ticker CSV files from source directory to destination directory.
    
    Args:
        tickers: List of stock ticker symbols (e.g., ['AAPL', 'GOOGL', 'MSFT'])
        source_dir: Path to the source directory containing the stock files
        dest_dir: Path to the destination directory where files will be copied
        overwrite: Whether to overwrite existing files in destination
    """
    
    # Get the script directory to resolve relative paths
    script_dir = Path(__file__).parent
    source_path = script_dir / source_dir
    dest_path = script_dir / dest_dir
    
    # Create destination directory if it doesn't exist
    dest_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Source directory: {source_path}")
    print(f"Destination directory: {dest_path}")
    print(f"Copying {len(tickers)} stock files...")
    
    copied_files = []
    missing_files = []
    skipped_files = []
    
    for ticker in tickers:
        # Handle both uppercase and lowercase ticker names
        ticker_upper = ticker.upper()
        ticker_lower = ticker.lower()
        
        # Check for file existence (try both cases)
        source_file_upper = source_path / f"{ticker_upper}.csv"
        source_file_lower = source_path / f"{ticker_lower}.csv"
        
        source_file = None
        if source_file_upper.exists():
            source_file = source_file_upper
        elif source_file_lower.exists():
            source_file = source_file_lower
        
        if source_file is None:
            missing_files.append(ticker)
            print(f"❌ File not found: {ticker}")
            continue
        
        # Set destination file path (use uppercase convention)
        dest_file = dest_path / f"{ticker_upper}.csv"
        
        # Check if destination file already exists
        if dest_file.exists() and not overwrite:
            skipped_files.append(ticker)
            print(f"⏭️  Skipped (already exists): {ticker}")
            continue
        
        try:
            shutil.copy2(source_file, dest_file)
            copied_files.append(ticker)
            print(f"✅ Copied: {ticker}")
        except Exception as e:
            print(f"❌ Error copying {ticker}: {e}")
    
    # Print summary
    print(f"\n📊 Summary:")
    print(f"  Successfully copied: {len(copied_files)} files")
    print(f"  Missing files: {len(missing_files)} files")
    print(f"  Skipped files: {len(skipped_files)} files")
    
    if missing_files:
        print(f"\n❌ Missing files: {', '.join(missing_files)}")
    
    if skipped_files:
        print(f"\n⏭️  Skipped files: {', '.join(skipped_files)}")

if __name__ == "__main__":
    tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN', 'TSLA', 'NVDA', 'NFLX']
    copy_stock_files(tickers)

