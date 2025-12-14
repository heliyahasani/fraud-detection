"""
Download and setup the fraud detection dataset.

Usage:
    uv run python data/download_data.py
"""

import shutil
import subprocess
from pathlib import Path

import kagglehub

# Get the directory where THIS script is located
SCRIPT_DIR = Path(__file__).parent.resolve()

# Define paths relative to script location
RAW_DIR = SCRIPT_DIR / "raw"
INTERIM_DIR = SCRIPT_DIR / "interim"
PROCESSED_DIR = SCRIPT_DIR / "processed"
MERGE_SCRIPT = INTERIM_DIR / "merge_data.py"


def main():
    print("Downloading fraud detection dataset from Kaggle...")
    print(f"Script location: {SCRIPT_DIR}")

    # Create folders
    RAW_DIR.mkdir(exist_ok=True)
    INTERIM_DIR.mkdir(exist_ok=True)
    PROCESSED_DIR.mkdir(exist_ok=True)

    # Download dataset
    download_path = Path(kagglehub.dataset_download("kartik2112/fraud-detection"))

    # Copy CSV files to raw/
    print(f"\nCopying files to: {RAW_DIR}")
    for csv_file in download_path.glob("*.csv"):
        dest = RAW_DIR / csv_file.name
        shutil.copy(csv_file, dest)
        size_mb = dest.stat().st_size / (1024 * 1024)
        print(f"  {csv_file.name}: {size_mb:.1f} MB")

    # Run merge script from interim directory
    print(f"\nRunning merge script: {MERGE_SCRIPT}")
    subprocess.run(["python", "merge_data.py"], cwd=INTERIM_DIR)

    print(f"\nDone! Data: {INTERIM_DIR}")


if __name__ == "__main__":
    main()
