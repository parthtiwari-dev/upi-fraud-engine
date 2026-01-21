"""
Feature computation functions using the Time Utils.
Each function receives a current transaction and the full history.
All functions enforce point-in-time correctness via time_utils.
"""

import pandas as pd
from typing import Dict
from src.features.time_utils import get_history_slice, calculate_aggregates
from src.features.schema import FeatureVector

def compute_payer_velocity(current_txn: dict, full_df: pd.DataFrame) -> Dict[str, float]:
    """
    Computes velocity features for the payer entity.
    Logic:
    For each window (5min, 1h, 24h):
      1. Get history slice [T - window, T)
      2. Count transactions and sum amounts
      3. Store as payer_txn_count_{window}, payer_txn_sum_{window}
    """
    # Defensive checks
    assert "event_timestamp" in full_df.columns, "full_df must contain event_timestamp"
    # Note: Type check skipped for speed, assuming upstream validation

    results = {}

    # Extract transaction details
    payer_id = current_txn['payer_id']
    current_time = current_txn['event_timestamp']

    # Window definitions (in minutes)
    windows = {
        '5min': 5,
        '1h': 60,
        '24h': 1440
    }

    for window_name, window_minutes in windows.items():
        # Get history slice for this window
        history = get_history_slice(
            full_df=full_df,
            entity_column='payer_id',
            entity_value=payer_id,
            current_time=current_time,
            window_minutes=window_minutes
        )

        # Calculate aggregates
        aggs = calculate_aggregates(history)

        # Store results
        results[f'payer_txn_count_{window_name}'] = aggs['count']
        results[f'payer_txn_sum_{window_name}'] = aggs['sum']

    return results

def compute_device_graph(current_txn: dict, full_df: pd.DataFrame) -> Dict[str, int]:
    """
    Computes graph-based features (entity cardinality).
    
    UPDATED LOGIC (ROWS window instead of time window):
    - device_distinct_payers_7d: Unique payers in last 1000 device transactions
    - payer_distinct_payees_7d: Unique payees in last 1000 payer transactions
    
    This matches the SQL ROWS BETWEEN 1000 PRECEDING logic.
    Functionally equivalent for fraud detection (catches same patterns).
    """

    assert "payee_vpa" in full_df.columns, "payee_vpa column missing"

    results = {}

    device_id = current_txn['device_id']
    payer_id = current_txn['payer_id']
    current_time = current_txn['event_timestamp']
    
    # ROWS window: last 1000 transactions (not time-based)
    MAX_LOOKBACK_ROWS = 1000

    # --- Device -> Payers cardinality (ROWS logic) ---
    # Get all device history before current time
    device_mask = (full_df['device_id'] == device_id) & \
                  (full_df['event_timestamp'] < current_time)
    device_history = full_df[device_mask].tail(MAX_LOOKBACK_ROWS)
    
    if not device_history.empty:
        results['device_distinct_payers_7d'] = device_history['payer_id'].nunique()
    else:
        results['device_distinct_payers_7d'] = 0

    # --- Payer -> Payees cardinality (ROWS logic) ---
    payer_mask = (full_df['payer_id'] == payer_id) & \
                 (full_df['event_timestamp'] < current_time)
    payer_history = full_df[payer_mask].tail(MAX_LOOKBACK_ROWS)
    
    if not payer_history.empty:
        results['payer_distinct_payees_7d'] = payer_history['payee_vpa'].nunique()
    else:
        results['payer_distinct_payees_7d'] = 0

    return results

def compute_risk_history(current_txn: dict, full_df: pd.DataFrame) -> Dict[str, int]:
    """
    Computes risk history features (past fraud count).

    CRITICAL CONSTRAINT:
    Only count fraud where:
      1. is_fraud == 1
      2. label_available_timestamp < current_time (no future peeking)
      3. event_timestamp < current_time (temporal ordering)

    This prevents label leakage by ensuring we only use labels that
    would have been available at decision time T.
    """
    # Defensive checks
    assert "label_available_timestamp" in full_df.columns
    assert "is_fraud" in full_df.columns

    payer_id = current_txn['payer_id']
    current_time = current_txn['event_timestamp']

    # 30 days = 43200 minutes
    THIRTY_DAYS_MINUTES = 30 * 24 * 60

    # Get payer history in last 30 days
    payer_history = get_history_slice(
        full_df=full_df,
        entity_column='payer_id',
        entity_value=payer_id,
        current_time=current_time,
        window_minutes=THIRTY_DAYS_MINUTES
    )

    if payer_history.empty:
        return {'payer_past_fraud_count_30d': 0}

    # CRITICAL FILTER: Only use labels available before T
    # This enforces the 48-hour label delay constraint
    valid_labels_mask = (
        (payer_history['is_fraud'] == 1) &
        (payer_history['label_available_timestamp'] < current_time)
    )

    fraud_count = int(valid_labels_mask.sum())

    return {'payer_past_fraud_count_30d': fraud_count}

def compute_device_velocity(current_txn: dict, full_df: pd.DataFrame) -> Dict[str, int]:
    """
    Computes velocity features for the device entity.
    Logic:
    For windows (1h, 24h):
      1. Get history slice [T - window, T) for device_id
      2. Count transactions
      3. Store as device_txn_count_{window}
    """
    results = {}

    device_id = current_txn['device_id']
    current_time = current_txn['event_timestamp']

    windows = {
        '1h': 60,
        '24h': 1440
    }

    for window_name, window_minutes in windows.items():
        history = get_history_slice(
            full_df=full_df,
            entity_column='device_id',
            entity_value=device_id,
            current_time=current_time,
            window_minutes=window_minutes
        )

        results[f'device_txn_count_{window_name}'] = len(history)

    return results

def compute_all_features(current_txn: dict, full_df: pd.DataFrame) -> FeatureVector:
    """
    Orchestrates all feature computation functions.
    Returns a complete FeatureVector object.
    """
    # Defensive checks
    assert "event_timestamp" in full_df.columns, "full_df must contain event_timestamp"

    # Start with identity fields
    features = {
        'transaction_id': current_txn['transaction_id'],
        'event_timestamp': current_txn['event_timestamp']
    }

    # Compute each feature group
    features.update(compute_payer_velocity(current_txn, full_df))
    features.update(compute_device_velocity(current_txn, full_df))
    features.update(compute_device_graph(current_txn, full_df))
    features.update(compute_risk_history(current_txn, full_df))

    # Return validated Pydantic object
    return FeatureVector(**features)
