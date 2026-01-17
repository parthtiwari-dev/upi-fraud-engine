import pandas as pd
import sys
from pathlib import Path

def validate_data():
    project_root = Path(__file__).parent.parent
    path = project_root / "data" / "processed" / "full_upi_dataset_injected.csv"
    
    print("🚀 Starting Validation...")
    # Use low_memory=False because we haven't optimized dtypes yet
    df = pd.read_csv(path, parse_dates=['event_timestamp', 'label_available_timestamp'], low_memory=False)
    
    print(f"Loaded {len(df)} rows.")

    # --- Check 1: Unique IDs ---
    print("\n[1] Checking IDs...")
    if df['transaction_id'].is_unique:
        print("✅ transaction_id is unique.")
    else:
        print(f"❌ DUPLICATE IDs FOUND: {df['transaction_id'].duplicated().sum()}")

    # --- Check 2: Time Order ---
    print("\n[2] Checking Time Sort...")
    if df['event_timestamp'].is_monotonic_increasing:
        print("✅ Data is strictly sorted by time.")
    else:
        print("❌ Data is NOT sorted! (Did you forget to sort in injector?)")

    # --- Check 3: Label Delay ---
    print("\n[3] Checking Label Delay...")
    # Assert label is strictly AFTER event
    time_travel_mask = df['label_available_timestamp'] <= df['event_timestamp']
    bad_rows = df[time_travel_mask]
    
    if len(bad_rows) == 0:
        print("✅ No time travel detected (Labels arrive after events).")
    else:
        print(f"❌ {len(bad_rows)} rows have invalid label availability!")

    # --- Check 4: Fraud Summary ---
    print("\n[4] Fraud Summary")
    fraud_count = df['is_fraud'].sum()
    fraud_rate = (fraud_count / len(df)) * 100
    print(f"Total Fraud: {fraud_count}")
    print(f"Fraud Rate:  {fraud_rate:.2f}%")
    
    print("\nPattern Breakdown:")
    # Value counts including NaNs
    print(df['fraud_pattern'].value_counts(dropna=False))

if __name__ == "__main__":
    validate_data()
