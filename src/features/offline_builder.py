"""
Batch Feature Builder (The Time Machine).
Generates training data by replaying history row-by-row.
Enforces strict point-in-time correctness by passing the full state
and letting feature_definitions logic handle the time filtering.
"""

import pandas as pd
import duckdb
from typing import List
from datetime import datetime
from tqdm import tqdm # Progress bar is essential for O(N^2) loops

from src.features.feature_definitions import compute_all_features
from src.features.schema import FeatureVector

def load_transactions(duckdb_path: str) -> pd.DataFrame:
    """Loads raw transactions sorted by time (for deterministic replay)."""
    print(f"Loading data from {duckdb_path}...")
    con = duckdb.connect(duckdb_path)
    # We sort by time to simulate the natural flow of events,
    # but the correctness logic relies on timestamps, not sort order.
    query = "SELECT * FROM transactions"

    df = con.execute(query).df()
    con.close()

    df["transaction_id"] = df["transaction_id"].astype(str)
    df['event_timestamp'] = pd.to_datetime(df['event_timestamp'])
    
    assert df["event_timestamp"].notna().all(), "Null event_timestamp found"

    if 'label_available_timestamp' in df.columns:
        df['label_available_timestamp'] = pd.to_datetime(df['label_available_timestamp'])

    df = df.sort_values("event_timestamp").reset_index(drop=True)
    assert df["event_timestamp"].is_monotonic_increasing, "event_timestamp must be sorted"


    return df

def build_features_batch(df: pd.DataFrame) -> pd.DataFrame:
    """
    The Main Loop: Iterates through history and builds features.

    WARNING: This is O(N^2). It is designed for correctness validation
    on smaller datasets (<100k rows), not for massive scale production.
    """
    assert df["event_timestamp"].is_monotonic_increasing, "event_timestamp must be sorted"

    feature_vectors: List[dict] = []

    print(f"Building features for {len(df)} transactions...")

    # Iterate row by row
    # tqdm gives us a progress bar so we know it's not frozen
    for i in tqdm(range(len(df)), desc="Time Machine Replay"):

        # 1. Extract current transaction as a dictionary
        current_txn = df.iloc[i].to_dict()

        # 2. Compute Features
        # We pass the FULL DataFrame. The strict (< T) filtering happens inside.
        try:
            fv: FeatureVector = compute_all_features(
                current_txn=current_txn,
                full_df=df
            )

            # 3. Collect Result (dump pydantic to dict for dataframe creation)
            feature_vectors.append(fv.model_dump())


        except Exception as e:
            print(f"Error processing transaction {current_txn.get('transaction_id')}: {e}")
            # In production, we might skip or fail. Here we raise to debug.
            raise e

    # 4. Materialize Output
    return pd.DataFrame(feature_vectors)

def save_features(feature_df: pd.DataFrame, output_path: str):
    print(f"Saving {len(feature_df)} feature vectors to {output_path}...")
    con = duckdb.connect(output_path)
    con.register("feature_df", feature_df)
    con.execute("CREATE OR REPLACE TABLE features AS SELECT * FROM feature_df")
    con.close()
    print("✅ Success.")


if __name__ == "__main__":
    # Config
    INPUT_DB = "data/processed/transactions.duckdb"
    OUTPUT_DB = "data/processed/15k_features.duckdb"

    # limit for testing speed (optional, remove for full run)
    # df = load_transactions(INPUT_DB).head(1000) 

    try:
        # 1. Load
        raw_df = load_transactions(INPUT_DB).head(15000)

        # 2. Build (The slow, correct part)
        features_df = build_features_batch(raw_df)

        # 3. Save
        save_features(features_df, OUTPUT_DB)

        print("\nSample Feature Vector:")
        print(features_df.iloc[-1])

    except Exception as e:
        print(f"\n❌ Job Failed: {e}")
