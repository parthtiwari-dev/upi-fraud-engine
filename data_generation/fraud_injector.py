import pandas as pd
import numpy as np 
import os
import sys
from pathlib import Path
import random

sys.path.append(str(Path(__file__).parent.parent)) 


def inject_device_rings(df):
    """
    Simulate 'Device Farming' rings where multiple fraudsters share a single device.
    
    Logic:
    1. Identify existing fraud transactions.
    2. Select a subset (2%) to form a ring.
    3. Overwrite their device_id to a shared 'hacker' device ID.
    4. Tag them for future validation.
    """
    df = df.copy()

    fraud_idxs = df[df["is_fraud"] == 1].index.tolist()

    if not fraud_idxs:
        print("no fraud rows found to inject patterns")
        return df
    
    num_to_inject = max(5, int(len(fraud_idxs) * 0.02))

    ring_indices = random.sample(fraud_idxs, num_to_inject)

    cluster_size = 5
    for i in range(0, len(ring_indices), cluster_size):
        chunk = ring_indices[i : i + cluster_size]
        
        fake_device_id = f"device_hacker_ring_{i // cluster_size}"
        
        df.loc[chunk, 'device_id'] = fake_device_id
        
        if 'fraud_pattern' not in df.columns:
            df['fraud_pattern'] = None
            
        df.loc[chunk, 'fraud_pattern'] = 'device_ring'
        
    print(f"   -> Injected 'device_ring' into {len(ring_indices)} rows ({len(ring_indices)//cluster_size} rings created).")
    
    return df


def inject_velocity_spikes(df):
    """
    Simulate 'Velocity Attacks' where a user makes many transactions in seconds.
    
    Logic:
    1. Find legitimate users (is_fraud=0) who have at least 10 transactions.
    2. Pick a random victim user.
    3. Select 10 of their transactions.
    4. Overwrite timestamps to occur within a 5-minute window.
    5. Flip label to fraud (is_fraud=1) and tag pattern.
    """
    df = df.copy()
    
    user_counts = df[df['is_fraud'] == 0].groupby('payer_id')['transaction_id'].count()
    
    candidates = user_counts[user_counts >= 10].index.tolist()
    
    if not candidates:
        print("Warning: No users with >= 10 transactions found for velocity injection.")
        return df
        
    victim_id = random.choice(candidates)
    
    victim_txns = df[df['payer_id'] == victim_id].sample(n=10)
    victim_indices = victim_txns.index
    
    base_time = victim_txns['event_timestamp'].min()
    
    new_times = [base_time + pd.Timedelta(seconds=i*30) for i in range(10)]
    
    df.loc[victim_indices, 'event_timestamp'] = new_times
    
    df.loc[victim_indices, 'is_fraud'] = 1
    
    if 'fraud_pattern' not in df.columns:
        df['fraud_pattern'] = None
    df.loc[victim_indices, 'fraud_pattern'] = 'velocity_spike'
    
    print(f"   -> Injected 'velocity_spike' for user {victim_id} (10 txns in 5 mins).")
    
    return df


def inject_time_anomalies(df):
    """
    Simulate 'Nighttime Attacks' where transactions happen at suspicious hours (2 AM - 4 AM).
    
    Logic:
    1. Filter transactions happening between 02:00 and 04:00.
    2. Exclude rows that are already fraud.
    3. Pick a small fraction (e.g., 1%) of these legitimate night transactions.
    4. Flip label to fraud (is_fraud=1) and tag pattern.
    """
    df = df.copy()
    
    night_mask = (df['event_timestamp'].dt.hour >= 2) & (df['event_timestamp'].dt.hour <= 4)
    
    candidates = df[night_mask & (df['is_fraud'] == 0)].index.tolist()
    
    if not candidates:
        print("Warning: No legitimate transactions found between 2 AM and 4 AM.")
        return df
        
    num_to_inject = max(10, int(len(candidates) * 0.01))
    
    victim_indices = random.sample(candidates, num_to_inject)
    
    df.loc[victim_indices, 'is_fraud'] = 1
    
    if 'fraud_pattern' not in df.columns:
        df['fraud_pattern'] = None
    df.loc[victim_indices, 'fraud_pattern'] = 'time_anomaly'
    
    print(f"   -> Injected 'time_anomaly' into {len(victim_indices)} rows (Hours 2-4 AM).")
    
    return df


if __name__ == "__main__":
    project_root = Path(__file__).parent.parent
    input_path = project_root / "data" / "processed" / "full_upi_dataset.csv"
    output_path = project_root / "data" / "processed" / "full_upi_dataset_injected.csv"
    
    print("🚀 Starting Fraud Injection Pipeline...")
    print("1. Loading Data...")
    df = pd.read_csv(input_path, parse_dates=['event_timestamp'], low_memory=False)
    
    print("\n2. Injecting Fraud Patterns...")
    df = inject_device_rings(df)
    df = inject_velocity_spikes(df)
    df = inject_time_anomalies(df)

    print("\n3. Re-sorting by time (Critical)...")
    df.sort_values(by="event_timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"\n4. Saving to {output_path}...")
    df.to_csv(output_path, index=False)
    print("✅ DONE! Injected dataset ready for Feature Engineering.")
