import pandas as pd
import numpy as np 
from datetime import datetime, timedelta
import hashlib
import os
from pathlib import Path
script_dir = Path(__file__).parent
project_root = script_dir.parent

def load_and_merge_identity(txn_path , id_path):
    """
    Function 1: load_and_merge_identity(txn_path, id_path)
    Load the transaction CSV.
    Load the identity CSV.
    Do a LEFT JOIN on TransactionID.
    Return the merged dataframe.
    """
    df_train_trans = pd.read_csv(txn_path)

    df_train_id = pd.read_csv(id_path, low_memory=False)

    df_merge = pd.merge(df_train_trans, df_train_id, on="TransactionID", how="left")

    return df_merge

def standardize_columns(df, is_train_set):
    """ 
    Function 2: standardize_columns(df, is_train_set)
    Rename TransactionID → transaction_id.
    Rename TransactionAmt → amount.
    The Time Conversion:
    Pick a fake start date (e.g., January 1, 2023).
    Convert TransactionDT (which is in seconds) into a real datetime:
    event_timestamp = start_date + TransactionDT seconds.

    The Label Logic:
    If this is the train set: Rename isFraud → is_fraud.
    If this is the test set: Create a column is_fraud and set it to None (because test data has no labels).

    The Label Delay:
    Create label_available_timestamp = event_timestamp + 48 hours.
    Why: This column tells your system "don't look at the label until this time."
    Add a column currency = 'INR' (hardcode). 
    """
    df = df.copy()

    df.rename(columns = {
        'TransactionID': 'transaction_id',
        'TransactionAmt': 'amount'
    }, inplace = True)

    start_date = datetime(2025, 1 ,1)

    df['event_timestamp'] = start_date + pd.to_timedelta(df["TransactionDT"], unit='s')
    
    if is_train_set:
        df.rename(columns = {'isFraud' : 'is_fraud'}, inplace = True)
    else:
        df["is_fraud"] = np.nan

    df['label_available_timestamp'] = df['event_timestamp'] + pd.Timedelta(hours=48)

    df['currency'] = 'INR'

    return df
SALT = "upi_phase1_v1"  # fixed salt (constant) for stability


def stable_hash(text: str, n=12) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:n]


def map_to_upi(df):
    """ 
    Function 3: map_to_upi(df)
    Payer (Who Sent Money):
    Hash the card columns: payer_id = hash(card1 + card2).
    Create a UPI address: payer_vpa = f"user_{payer_id}@upi".
    Payee (Who Received Money):
    Use ProductCD as a proxy for merchant.
    Create payee_vpa = f"merchant_{ProductCD}@upi".

    Device:
    If DeviceInfo exists: Hash it → device_id.
    If DeviceInfo is missing: Set device_id = "unknown_device".
    Copy DeviceType as-is. 
    """
    df = df.copy()

    # Combine all 6 card columns safely
    card_cols = [col for col in ["card1","card2","card3","card4","card5","card6"] if col in df.columns]
    card_fingerprint = (
        df[card_cols].fillna("").astype(str).agg("|".join, axis=1)
    )
    df["payer_id"] = (SALT + "|" + card_fingerprint).map(stable_hash)
    df["payer_vpa"] = "user_" + df["payer_id"] + "@upi"

    # Merchant proxy
    df["payee_vpa"] = "merchant_" + df["ProductCD"].astype(str) + "@upi"

    # Device hashing with missing handling
    device_text = df["DeviceInfo"].fillna("")
    df["device_id"] = device_text.apply(
    lambda x: stable_hash(SALT + "|dev|" + x) if x else "unknown_device"
    )

    # Copy as-is
    df["DeviceType"] = df.get("DeviceType")

    return df


if __name__ == "__main__":
    # Put all your path definition and checking logic here
    print("🚀 Starting Manual Checks...")
    
    # Define paths
    train_txn_path = project_root / "data" / "raw" / "train_transaction.csv"
    train_id_path = project_root / "data" / "raw" / "train_identity.csv"

    # --- 1. Identity Merge Check ---
    print("\n--- 1. Checking Merge Logic ---")
    
    # Load raw transaction file just to get the true row count
    # (using usecols=['TransactionID'] makes this super fast)
    df_raw_txn = pd.read_csv(train_txn_path, usecols=['TransactionID'])
    initial_len = len(df_raw_txn)
    
    # Perform the actual load and merge
    df = load_and_merge_identity(train_txn_path, train_id_path)
    merged_len = len(df)
    
    print(f"Original Rows: {initial_len}")
    print(f"Merged Rows:   {merged_len}")
    
    # CRITICAL ASSERTION: Ensure no rows were dropped or duplicated
    assert merged_len == initial_len, f"❌ Row count mismatch! Expected {initial_len}, got {merged_len}"
    print("✅ Merge Row Count Safe")


    # --- 2. Standardization Check ---
    print("\n--- 2. Checking Column Standardization ---")
    df_std = standardize_columns(df, is_train_set=True)
    
    print("Preview of Standardized Columns:")
    print(df_std[[
        "transaction_id",
        "amount",
        "event_timestamp",
        "is_fraud",
        "label_available_timestamp",
        "currency"
    ]].head())
    
    # Basic data type checks
    assert pd.api.types.is_datetime64_any_dtype(df_std['event_timestamp']), "❌ event_timestamp is not datetime"
    assert pd.api.types.is_datetime64_any_dtype(df_std['label_available_timestamp']), "❌ label_available_timestamp is not datetime"
    print("✅ Timestamp types are correct")


    # --- 3. UPI Mapping Check ---
    print("\n--- 3. Checking UPI Mapping ---")
    df_upi = map_to_upi(df_std)
    
    print("Preview of UPI Columns:")
    print(df_upi[[
        "transaction_id",
        "payer_vpa",
        "payee_vpa",
        "device_id",
        "DeviceType"
    ]].head())
    
    # Check "unknown_device" logic works (assuming dataset has missing devices)
    missing_devices = (df_upi["device_id"] == "unknown_device").sum()
    print(f"Found {missing_devices} transactions with 'unknown_device' (This is expected)")


    # --- 4. Determinism Check ---
    print("\n--- 4. Checking Hashing Determinism ---")
    # Run the function twice on fresh copies
    df_upi_1 = map_to_upi(df_std.copy())
    df_upi_2 = map_to_upi(df_std.copy())
    
    # Check if payer_id is identical in both runs
    is_deterministic = (df_upi_1["payer_id"] == df_upi_2["payer_id"]).all()
    
    if is_deterministic:
        print("✅ Hashing is deterministic (stable across runs)")
    else:
        print("❌ Hashing is NOT deterministic! Check your salt or hash function.")
        
    print("\n✨ All checks completed.")
