import pandas as pd
from datetime import datetime, timedelta

def get_history_slice(
        full_df: pd.DataFrame,
        entity_column: str,
        entity_value: str,
        current_time: datetime,
        window_minutes: int
        ) -> pd.DataFrame:
    
    """
    Retrieves the correct slice of history for an entity without leakage.

    Logic:
    1. Filter by entity (payer_id, device_id, etc.)
    2. Filter by time: [T - window, T)
       CRITICAL: strictly LESS THAN (<) current_time to avoid leakage.
    """
    
    assert "event_timestamp" in full_df.columns, "full_df must contain event_timestamp column"

    # 1. Filter by Entity
    # (In a real DB, this is a WHERE clause. In Pandas, it's boolean indexing)
    entity_mask = full_df[entity_column] == entity_value

    # Optimize: If entity not found, return empty immediately
    if not entity_mask.any():
        return pd.DataFrame(columns=full_df.columns)

    # 2. Time Boundaries
    start_time = current_time - timedelta(minutes=window_minutes)

    # 3. Apply Time Filter
    # STRICTLY LESS THAN (<) is the "Time Machine" enforcement.
    time_mask = (full_df['event_timestamp'] >= start_time) & \
                (full_df['event_timestamp'] < current_time)

    return full_df[entity_mask & time_mask]

def calculate_aggregates(slice_df: pd.DataFrame) -> dict:
    """
    Computes standard aggregations (count, sum) on a history slice.
    """
    if slice_df.empty:
        return {"count": 0, "sum": 0.0}

    return {
        "count": len(slice_df),
        "sum": float(slice_df['amount'].sum())
    }
