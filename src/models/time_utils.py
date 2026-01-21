"""
Time-based splitting with label awareness.

This module is the PRIMARY defense against temporal leakage in Phase 5.

Key Guarantees:
1. Training data ends BEFORE test data starts (temporal correctness)
2. Training labels are only used if they were available before test window
3. No test distribution leaks into training preprocessing

All downstream pipelines (Stage 1, Stage 2) MUST use these splits.
"""

import pandas as pd
from datetime import timedelta
from typing import Tuple
import warnings


def calculate_split_dates(
    df: pd.DataFrame,
    test_window_days: int = 30,
    buffer_hours: int = 1
) -> Tuple[pd.Timestamp, pd.Timestamp]:
    """
    Calculate train/test split dates with temporal buffer.
    
    Design:
    - Test window is the LAST N days of data
    - Training window is everything BEFORE test window (with buffer)
    - Buffer prevents boundary cases where features computed near split
    
    Args:
        df: Feature DataFrame with 'event_timestamp' column
        test_window_days: Size of test window (default 30 days)
        buffer_hours: Time buffer between train and test (default 1 hour)
    
    Returns:
        (train_end_date, test_start_date)
    
    Raises:
        ValueError: If resulting training window is too small (<7 days)
    
    Example:
        >>> train_end, test_start = calculate_split_dates(df, test_window_days=30)
        >>> # Test window: [test_start, max_date]
        >>> # Train window: [min_date, train_end]
    """
    # Validate input
    if 'event_timestamp' not in df.columns:
        raise ValueError("DataFrame must contain 'event_timestamp' column")
    
    if df['event_timestamp'].isna().any():
        raise ValueError("event_timestamp contains null values")
    
    # Calculate dates
    max_event_time = df['event_timestamp'].max()
    min_event_time = df['event_timestamp'].min()
    
    test_start_date = max_event_time - pd.Timedelta(days=test_window_days)
    train_end_date = test_start_date - pd.Timedelta(hours=buffer_hours)
    
    # Validate minimum training window
    training_days = (train_end_date - min_event_time).days
    if training_days < 7:
        raise ValueError(
            f"Training window too small: {training_days} days. "
            f"Need at least 7 days. Reduce test_window_days or use more data."
        )
    
    # Log split information
    total_days = (max_event_time - min_event_time).days
    print(f"\n{'='*70}")
    print(f"TIME SPLIT CALCULATION")
    print(f"{'='*70}")
    print(f"Data range:       {min_event_time} → {max_event_time}")
    print(f"Total days:       {total_days} days")
    print(f"Train window:     {min_event_time} → {train_end_date} ({training_days} days)")
    print(f"Buffer:           {buffer_hours} hour(s)")
    print(f"Test window:      {test_start_date} → {max_event_time} ({test_window_days} days)")
    print(f"{'='*70}\n")
    
    return train_end_date, test_start_date


def split_data_with_label_awareness(
    df: pd.DataFrame,
    train_end_date: pd.Timestamp,
    test_start_date: pd.Timestamp,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into train/test with label availability enforcement.
    
    CRITICAL CONSTRAINT:
    Training labels are ONLY included if they were available BEFORE the test window.
    This prevents using "future knowledge" during training.
    
    Training Set Criteria:
    1. event_timestamp < train_end_date (temporal correctness)
    2. label_available_timestamp < test_start_date (label awareness)
    
    Test Set Criteria:
    1. event_timestamp >= test_start_date (future data)
    
    Args:
        df: Feature DataFrame with required columns:
            - event_timestamp
            - label_available_timestamp (can be null for unlabeled data)
            - is_fraud (0.0, 1.0, or NaN)
        train_end_date: End of training window (exclusive)
        test_start_date: Start of test window (inclusive)
        verbose: Print diagnostics
    
    Returns:
        (train_df, test_df)
    
    Raises:
        ValueError: If temporal overlap detected or required columns missing
    
    Example:
        >>> train_df, test_df = split_data_with_label_awareness(
        ...     df, train_end_date, test_start_date
        ... )
        >>> assert (train_df['event_timestamp'] < train_end_date).all()
        >>> assert (test_df['event_timestamp'] >= test_start_date).all()
    """
    # Validate required columns
    required_cols = ['event_timestamp', 'label_available_timestamp', 'is_fraud']
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # Temporal split
    train_temporal_mask = df['event_timestamp'] < train_end_date
    test_temporal_mask = df['event_timestamp'] >= test_start_date
    
    # Label awareness for training
    # Only use labels that were available BEFORE test window
    label_available_mask = df['label_available_timestamp'] < test_start_date
    train_label_aware_mask = train_temporal_mask & label_available_mask
    
    # Create splits
    train_df = df[train_label_aware_mask].copy()
    test_df = df[test_temporal_mask].copy()
    
    # Validation: No temporal overlap
    train_max = train_df['event_timestamp'].max()
    test_min = test_df['event_timestamp'].min()
    
    if train_max >= test_min:
        raise ValueError(
            f"TEMPORAL OVERLAP DETECTED!\n"
            f"Train max: {train_max}\n"
            f"Test min:  {test_min}\n"
            f"This violates temporal correctness."
        )
    
    # Diagnostics
    if verbose:
        total_rows = len(df)
        train_rows = len(train_df)
        test_rows = len(test_df)
        excluded_rows = total_rows - train_rows - test_rows
        
        train_labeled = train_df['is_fraud'].notna().sum()
        train_fraud = (train_df['is_fraud'] == 1.0).sum()
        train_legit = (train_df['is_fraud'] == 0.0).sum()
        
        test_labeled = test_df['is_fraud'].notna().sum()
        test_fraud = (test_df['is_fraud'] == 1.0).sum()
        test_legit = (test_df['is_fraud'] == 0.0).sum()
        
        print(f"\n{'='*70}")
        print(f"TRAIN/TEST SPLIT WITH LABEL AWARENESS")
        print(f"{'='*70}")
        print(f"Total rows:       {total_rows:,}")
        print(f"Train rows:       {train_rows:,} ({train_rows/total_rows*100:.1f}%)")
        print(f"Test rows:        {test_rows:,} ({test_rows/total_rows*100:.1f}%)")
        print(f"Excluded rows:    {excluded_rows:,} (label not yet available)")
        print(f"")
        print(f"TRAIN SET:")
        print(f"  Labeled:        {train_labeled:,} ({train_labeled/train_rows*100:.1f}%)")
        print(f"  Fraud:          {train_fraud:,} ({train_fraud/train_labeled*100:.2f}% of labeled)" if train_labeled > 0 else "  Fraud:          0")
        print(f"  Legitimate:     {train_legit:,}")
        print(f"")
        print(f"TEST SET:")
        print(f"  Labeled:        {test_labeled:,} ({test_labeled/test_rows*100:.1f}%)")
        print(f"  Fraud:          {test_fraud:,} ({test_fraud/test_labeled*100:.2f}% of labeled)" if test_labeled > 0 else "  Fraud:          0")
        print(f"  Legitimate:     {test_legit:,}")
        print(f"{'='*70}\n")
        
        # Warning if too many unlabeled in train
        if train_labeled / train_rows < 0.80:
            warnings.warn(
                f"Only {train_labeled/train_rows*100:.1f}% of training data is labeled! "
                f"This may impact supervised model quality. "
                f"Consider adjusting test_window_days to allow more labels to 'arrive'.",
                UserWarning
            )
    
    return train_df, test_df


def validate_no_leakage(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    test_start_date: pd.Timestamp
) -> bool:
    """
    Audit function to validate temporal correctness.
    
    Checks:
    1. No temporal overlap (train ends before test starts)
    2. All training labels were available before test window
    3. No test data in training set
    
    Args:
        train_df: Training DataFrame
        test_df: Test DataFrame
        test_start_date: Start of test window
    
    Returns:
        True if all checks pass
    
    Raises:
        AssertionError: If any leakage detected
    """
    print(f"\n{'='*70}")
    print(f"LEAKAGE VALIDATION")
    print(f"{'='*70}")
    
    # Check 1: No temporal overlap
    train_max_time = train_df['event_timestamp'].max()
    test_min_time = test_df['event_timestamp'].min()
    
    assert train_max_time < test_min_time, (
        f"TEMPORAL OVERLAP: Train max ({train_max_time}) >= Test min ({test_min_time})"
    )
    print(f"✅ No temporal overlap")
    
    # Check 2: All training labels were available before test
    if 'label_available_timestamp' in train_df.columns:
        late_labels = (train_df['label_available_timestamp'] >= test_start_date).sum()
        assert late_labels == 0, (
            f"LABEL LEAKAGE: {late_labels} training labels arrived after test window"
        )
        print(f"✅ All training labels available before test window")
    
    # Check 3: No duplicate transaction IDs across train/test
    if 'transaction_id' in train_df.columns and 'transaction_id' in test_df.columns:
        train_ids = set(train_df['transaction_id'])
        test_ids = set(test_df['transaction_id'])
        overlap = train_ids & test_ids
        assert len(overlap) == 0, (
            f"DATA LEAKAGE: {len(overlap)} transactions appear in both train and test"
        )
        print(f"✅ No transaction ID overlap")
    
    print(f"{'='*70}")
    print(f"🎉 ALL LEAKAGE CHECKS PASSED")
    print(f"{'='*70}\n")
    
    return True
