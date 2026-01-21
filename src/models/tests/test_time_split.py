"""
Time split validation test for Phase 5.

This test validates that time_utils.py correctly:
1. Splits data temporally
2. Enforces label awareness
3. Prevents temporal leakage

Run from project root:
    python -m src.models.tests.test_time_split

Or from src/models/tests/:
    python test_time_split.py
"""

import sys
import os
from pathlib import Path

# Add project root to path (handles both run methods)
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import duckdb
import pandas as pd

from src.models.time_utils import (
    calculate_split_dates,
    split_data_with_label_awareness,
    validate_no_leakage
)


def main():
    print("\n" + "="*70)
    print("PHASE 5 - TIME SPLIT VALIDATION")
    print("="*70 + "\n")
    
    # Construct path to data (relative to project root)
    data_path = project_root / "data" / "processed" / "full_features.duckdb"
    
    if not data_path.exists():
        print(f"❌ ERROR: Data file not found at {data_path}")
        print(f"   Current directory: {os.getcwd()}")
        print(f"   Project root: {project_root}")
        sys.exit(1)
    
    # Load Phase 4 data
    print("Loading Phase 4 feature data...")
    con = duckdb.connect(str(data_path), read_only=True)
    df = con.execute("SELECT * FROM training_data").df()
    con.close()
    
    print(f"✅ Loaded {len(df):,} transactions")
    print(f"   Columns: {len(df.columns)}")
    print(f"   Date range: {df['event_timestamp'].min()} → {df['event_timestamp'].max()}")
    
    # Calculate split dates
    print("\nCalculating split dates...")
    train_end_date, test_start_date = calculate_split_dates(
        df,
        test_window_days=30,
        buffer_hours=1
    )
    
    # Split data with label awareness
    print("\nSplitting data with label awareness...")
    train_df, test_df = split_data_with_label_awareness(
        df,
        train_end_date,
        test_start_date,
        verbose=True
    )
    
    # Validate no leakage
    print("\nValidating temporal correctness...")
    validate_no_leakage(train_df, test_df, test_start_date)
    
    # Show sample from each set
    print("\nSAMPLE FROM TRAIN SET (first 3 rows):")
    print(train_df[['transaction_id', 'event_timestamp', 'is_fraud', 'payer_txn_count_5min']].head(3))
    
    print("\nSAMPLE FROM TEST SET (first 3 rows):")
    print(test_df[['transaction_id', 'event_timestamp', 'is_fraud', 'payer_txn_count_5min']].head(3))
    
    # Additional validation stats
    print("\n" + "="*70)
    print("ADDITIONAL STATS")
    print("="*70)
    print(f"Train fraud rate:     {(train_df['is_fraud']==1.0).sum()/len(train_df)*100:.2f}%")
    print(f"Test fraud rate:      {(test_df['is_fraud']==1.0).sum()/len(test_df)*100:.2f}%")
    print(f"Train date range:     {train_df['event_timestamp'].min()} → {train_df['event_timestamp'].max()}")
    print(f"Test date range:      {test_df['event_timestamp'].min()} → {test_df['event_timestamp'].max()}")
    print(f"Train duration:       {(train_df['event_timestamp'].max() - train_df['event_timestamp'].min()).days} days")
    print(f"Test duration:        {(test_df['event_timestamp'].max() - test_df['event_timestamp'].min()).days} days")
    print("="*70)
    
    print("\n" + "="*70)
    print("✅ TIME SPLIT VALIDATION COMPLETE")
    print("="*70 + "\n")
    
    return train_df, test_df


if __name__ == "__main__":
    main()
