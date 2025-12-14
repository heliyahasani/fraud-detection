"""
WHY MERGE?
----------
The original Kaggle dataset comes pre-split, but we want full control over:
- Train/test split using random split with custom percentages (e.g., 70/30, 80/20)
- Ability to create validation sets
- Flexibility to experiment with different split strategies
"""

import pandas as pd
from pathlib import Path

# Paths
RAW_DIR = Path("../raw")
OUTPUT_FILE = Path("transactions_merged.csv")


def merge_datasets():
    """Merge train and test CSVs into one file."""

    # Load original files
    train = pd.read_csv(RAW_DIR / "fraudTrain.csv")
    test = pd.read_csv(RAW_DIR / "fraudTest.csv")

    print(f"Original Train: {len(train):,} rows")
    print(f"Original Test: {len(test):,} rows")

    # Merge
    df = pd.concat([train, test], ignore_index=True)
    print(f"Merged Total: {len(df):,} rows")

    # Save
    df.to_csv(OUTPUT_FILE, index=False)
    print(f"\nSaved to: {OUTPUT_FILE}")

    # Summary
    print(f"\nDataset Summary:")
    print(f"  Rows: {len(df):,}")
    print(f"  Columns: {len(df.columns)}")

    return df


if __name__ == "__main__":
    merge_datasets()
