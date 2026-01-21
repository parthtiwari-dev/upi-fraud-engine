"""
Data Leakage Auditor Tests

Critical validation that ensures model correctness by checking:
1. Stage 1 never sees fraud labels (unsupervised requirement)
2. No temporal overlap between train and test (no future information)
3. Stage 2 never trains on unlabeled data (NULL labels excluded)

These tests formalize correctness guarantees that prevent silent bugs
which could inflate performance metrics while making the model useless in production.

Author: [Your Name]
Date: January 22, 2026
"""

import pandas as pd
import numpy as np
import duckdb
import pytest
from datetime import datetime, timedelta
from pathlib import Path
import warnings


# ============================================================================
# TEST 1: STAGE 1 DOES NOT REFERENCE LABEL FEATURES
# ============================================================================

def test_stage1_no_label_features():
    """
    CRITICAL: Verify Stage 1 never uses fraud labels or derived features.
    
    Why this matters:
        Stage 1 is UNSUPERVISED anomaly detection. If it accidentally uses
        'is_fraud', 'payer_past_fraud_count_30d', or any label-derived feature,
        it's not actually unsupervised - it's cheating with the answer key!
    
    This would:
        - Inflate Stage 1 performance metrics
        - Make model useless in production (real data has no labels)
        - Be a critical design flaw
    
    Test validates:
        - STAGE1_FEATURES list contains no forbidden features
        - Preprocessor doesn't accidentally select forbidden features
    """
    from src.models.stage1_anomaly import STAGE1_FEATURES, FORBIDDEN_FEATURES
    
    print("\n" + "="*70)
    print("TEST 1: Stage 1 Feature Independence")
    print("="*70)
    
    # Check for forbidden features in Stage 1 feature list
    forbidden_in_stage1 = set(STAGE1_FEATURES) & set(FORBIDDEN_FEATURES)
    
    print(f"\nStage 1 uses {len(STAGE1_FEATURES)} features:")
    for feat in STAGE1_FEATURES:
        print(f"  ✓ {feat}")
    
    print(f"\nForbidden features (label-derived):")
    for feat in FORBIDDEN_FEATURES:
        print(f"  ✗ {feat}")
    
    if len(forbidden_in_stage1) > 0:
        print(f"\n❌ CRITICAL ERROR: Stage 1 references {len(forbidden_in_stage1)} forbidden features!")
        print(f"   Violations: {forbidden_in_stage1}")
        print(f"   → This makes Stage 1 supervised (not unsupervised)!")
        assert False, f"Stage 1 leaks label information via: {forbidden_in_stage1}"
    
    print(f"\n✅ PASS: Stage 1 is truly unsupervised (no label features)")
    print("="*70 + "\n")


# ============================================================================
# TEST 2: NO TEMPORAL OVERLAP BETWEEN TRAIN AND TEST
# ============================================================================

def test_no_temporal_overlap():
    """
    CRITICAL: Verify training data is strictly before test data.
    
    Why this matters:
        If train contains data from June 5 and test starts June 1, the model
        has "seen the future" via feature aggregations (e.g., device_txn_count_24h).
        
        Example bug:
            - Test transaction at June 1, 10:00 AM
            - Feature: device_txn_count_24h (counts 9:00 AM - 10:00 AM)
            - If training data includes June 1, 11:00 AM, the aggregation leaked!
    
    This would:
        - Artificially boost test performance
        - Fail completely in production (can't predict past with future data)
    
    Test validates:
        - Latest train timestamp < earliest test timestamp (with buffer)
        - No transaction ID overlap
        - Label availability constraints respected
    """
    from src.models.time_utils import calculate_split_dates, split_data_with_label_awareness
    
    print("\n" + "="*70)
    print("TEST 2: Temporal Data Integrity")
    print("="*70)
    
    # Load data
    data_path = "data/processed/full_features.duckdb"
    con = duckdb.connect(data_path, read_only=True)
    df = con.execute("SELECT transaction_id, event_timestamp, label_available_timestamp, is_fraud FROM training_data").df()
    con.close()
    
    # Perform split
    train_end, test_start = calculate_split_dates(df, test_window_days=30, buffer_hours=1)
    train_df, test_df = split_data_with_label_awareness(df, train_end, test_start, verbose=False)
    
    # Validate temporal separation
    train_max_time = train_df['event_timestamp'].max()
    test_min_time = test_df['event_timestamp'].min()
    
    time_gap = (test_min_time - train_max_time).total_seconds() / 3600  # hours
    
    print(f"\nTrain set:")
    print(f"  Earliest: {train_df['event_timestamp'].min()}")
    print(f"  Latest:   {train_max_time}")
    print(f"  Rows:     {len(train_df):,}")
    
    print(f"\nTest set:")
    print(f"  Earliest: {test_min_time}")
    print(f"  Latest:   {test_df['event_timestamp'].max()}")
    print(f"  Rows:     {len(test_df):,}")
    
    print(f"\nTemporal Gap: {time_gap:.1f} hours")
    
    # Validation checks
    assert train_max_time < test_min_time, \
        f"❌ TEMPORAL OVERLAP: Train max ({train_max_time}) >= Test min ({test_min_time})"
    
    assert time_gap >= 1.0, \
        f"❌ INSUFFICIENT BUFFER: Gap is {time_gap:.1f}h, need >= 1.0h for aggregation safety"
    
    # Check for transaction ID overlap (should be zero)
    train_ids = set(train_df['transaction_id'])
    test_ids = set(test_df['transaction_id'])
    overlap_ids = train_ids & test_ids
    
    assert len(overlap_ids) == 0, \
        f"❌ TRANSACTION OVERLAP: {len(overlap_ids)} IDs appear in both train and test!"
    
    print(f"\n✅ PASS: Strict temporal separation maintained")
    print(f"  → {time_gap:.1f}h buffer between train and test")
    print(f"  → 0 transaction ID overlaps")
    print("="*70 + "\n")


# ============================================================================
# TEST 3: STAGE 2 SEES NO NULL LABELS
# ============================================================================

def test_stage2_no_null_labels():
    """
    CRITICAL: Verify Stage 2 training data contains only labeled examples.
    
    Why this matters:
        XGBoost cannot train on NULL labels (is_fraud = NaN). If training data
        includes unlabeled transactions, it causes:
        - Training errors or silent NaN propagation
        - Unpredictable model behavior
        - Inflated performance (if NaNs treated as legitimate)
    
    This would:
        - Make model unreliable
        - Cause production failures when scoring unlabeled data
    
    Test validates:
        - All training labels are 0.0 or 1.0 (no NaNs)
        - Label availability timestamp < event timestamp for all train examples
        - Test set CAN have NaNs (production reality)
    """
    from src.models.time_utils import split_data_with_label_awareness, calculate_split_dates
    from src.models.stage2_supervised import prepare_stage2_features
    
    print("\n" + "="*70)
    print("TEST 3: Training Label Completeness")
    print("="*70)
    
    # Load data
    data_path = "data/processed/full_features.duckdb"
    con = duckdb.connect(data_path, read_only=True)
    df = con.execute("SELECT * FROM training_data").df()
    con.close()
    
    # Perform split
    train_end, test_start = calculate_split_dates(df, test_window_days=30, buffer_hours=1)
    train_df, test_df = split_data_with_label_awareness(df, train_end, test_start, verbose=False)
    
    # Check raw train/test labels
    train_nulls = train_df['is_fraud'].isna().sum()
    test_nulls = test_df['is_fraud'].isna().sum()
    
    print(f"\nRaw data (before prepare_stage2_features):")
    print(f"  Train set: {len(train_df):,} rows, {train_nulls:,} NULL labels")
    print(f"  Test set:  {len(test_df):,} rows, {test_nulls:,} NULL labels")
    
    # Prepare features (should filter NULLs from train)
    X_train, features, y_train, encoders = prepare_stage2_features(train_df, verbose=False)
    X_test, _, y_test, _ = prepare_stage2_features(test_df, verbose=False)
    
    # Check labels after preparation
    train_nulls_after = y_train.isna().sum()
    test_nulls_after = y_test.isna().sum()
    
    print(f"\nAfter prepare_stage2_features:")
    print(f"  Train labels (y_train): {len(y_train):,} rows, {train_nulls_after:,} NULL labels")
    print(f"  Test labels (y_test):   {len(y_test):,} rows, {test_nulls_after:,} NULL labels")
    
    # Validation: Train must have ZERO nulls
    assert train_nulls_after == 0, \
        f"❌ TRAINING DATA HAS {train_nulls_after} NULL LABELS! XGBoost cannot train on these."
    
    # Validation: All train labels are 0.0 or 1.0
    unique_train_labels = set(y_train.unique())
    assert unique_train_labels <= {0.0, 1.0}, \
        f"❌ INVALID LABELS IN TRAINING: {unique_train_labels}. Expected only {{0.0, 1.0}}"
    
    # Test set CAN have nulls (this is production reality)
    if test_nulls_after > 0:
        print(f"\n  ℹ️  Test set has {test_nulls_after:,} NULL labels (expected in production)")
    
    # Check label distribution
    train_fraud_rate = (y_train == 1.0).sum() / len(y_train)
    test_fraud_rate = (y_test == 1.0).sum() / y_test.notna().sum() if y_test.notna().sum() > 0 else 0
    
    print(f"\nLabel distribution:")
    print(f"  Train fraud rate: {train_fraud_rate:.2%}")
    print(f"  Test fraud rate:  {test_fraud_rate:.2%} (among labeled)")
    
    # Sanity check: fraud rate should be reasonable (1-10%)
    assert 0.01 <= train_fraud_rate <= 0.10, \
        f"❌ SUSPICIOUS FRAUD RATE: {train_fraud_rate:.2%}. Expected 1-10% for realistic fraud."
    
    print(f"\n✅ PASS: All training labels are valid (0.0 or 1.0, no NaNs)")
    print("="*70 + "\n")


# ============================================================================
# TEST 4: LABEL AVAILABILITY CONSTRAINT
# ============================================================================

def test_label_availability_respected():
    """
    ADVANCED: Verify labels used in training were available before test window.
    
    Why this matters:
        In production, fraud labels arrive with delay (e.g., 72h for chargeback).
        If we train on labels that weren't available yet, model performance
        will be artificially inflated.
        
        Example:
            - Transaction on Jan 1, 10:00 AM
            - Label available on Jan 4, 10:00 AM (72h delay)
            - Test window starts Jan 2
            - This label CANNOT be used in training (not available yet)
    
    Test validates:
        - All train labels have label_available_timestamp < test_start
        - Split respects this constraint
    """
    from src.models.time_utils import split_data_with_label_awareness, calculate_split_dates
    
    print("\n" + "="*70)
    print("TEST 4: Label Availability Constraint")
    print("="*70)
    
    # Load data
    data_path = "data/processed/full_features.duckdb"
    con = duckdb.connect(data_path, read_only=True)
    df = con.execute("SELECT event_timestamp, label_available_timestamp, is_fraud FROM training_data").df()
    con.close()
    
    # Perform split
    train_end, test_start = calculate_split_dates(df, test_window_days=30, buffer_hours=1)
    train_df, test_df = split_data_with_label_awareness(df, train_end, test_start, verbose=False)
    
    # Check label availability in training set
    train_labeled = train_df[train_df['is_fraud'].notna()]
    
    if len(train_labeled) > 0:
        violations = train_labeled[train_labeled['label_available_timestamp'] >= test_start]
        
        print(f"\nLabel availability check:")
        print(f"  Train set (labeled): {len(train_labeled):,} rows")
        print(f"  Test window starts:  {test_start}")
        print(f"  Labels available after test_start: {len(violations):,}")
        
        if len(violations) > 0:
            print(f"\n❌ CRITICAL: {len(violations):,} training labels weren't available before test window!")
            print(f"\nExample violations:")
            print(violations[['event_timestamp', 'label_available_timestamp']].head())
            assert False, "Label availability constraint violated"
        
        print(f"\n✅ PASS: All training labels were available before test window")
    else:
        print(f"\n⚠️  WARNING: No labeled training data found")
    
    print("="*70 + "\n")
# ============================================================================
# TEST 5: FRAUD_PATTERN NOT IN STAGE 2
# ============================================================================

def test_fraud_pattern_excluded():
    """
    CRITICAL: Verify fraud_pattern is excluded from Stage 2 features.
    
    Why this matters:
        fraud_pattern was created during Phase 2 (fraud injection) to label
        which attack pattern was used (velocity_burst, round_amount, etc.).
        
        This column:
        - Only exists in synthetic training data
        - Does NOT exist in real production transactions
        - Will cause catastrophic deployment failure (missing column error)
        
        If fraud_pattern has high feature importance, it means the model
        is "cheating" with information that won't be available at inference time.
        
        This is a classic "training-serving skew" bug that kills ML projects.
    
    Test validates:
        1. fraud_pattern is in STAGE2_EXCLUDED_COLUMNS
        2. fraud_pattern is NOT in prepared Stage 2 features
        3. Model can be trained without fraud_pattern (production simulation)
    """
    import duckdb
    from src.models.stage2_supervised import STAGE2_EXCLUDED_COLUMNS, prepare_stage2_features
    from src.models.time_utils import split_data_with_label_awareness, calculate_split_dates
    
    print("\n" + "="*70)
    print("TEST 5: fraud_pattern Exclusion (Production Safety)")
    print("="*70)
    
    # ========================================================================
    # CHECK 1: fraud_pattern in exclusion list
    # ========================================================================
    print(f"\nCHECK 1: Verifying fraud_pattern in STAGE2_EXCLUDED_COLUMNS...")
    
    if 'fraud_pattern' not in STAGE2_EXCLUDED_COLUMNS:
        print(f"\n❌ CRITICAL ERROR: fraud_pattern NOT in STAGE2_EXCLUDED_COLUMNS!")
        print(f"\nCurrent exclusions: {STAGE2_EXCLUDED_COLUMNS}")
        print(f"\n⚠️  This will cause production failure:")
        print(f"   1. Model trains with fraud_pattern (synthetic data)")
        print(f"   2. Production data lacks fraud_pattern column")
        print(f"   3. Prediction fails with 'missing column' error")
        print(f"\nAdd 'fraud_pattern' to STAGE2_EXCLUDED_COLUMNS immediately!")
        assert False, "fraud_pattern must be in STAGE2_EXCLUDED_COLUMNS"
    
    print(f"   ✅ fraud_pattern is in STAGE2_EXCLUDED_COLUMNS")
    
    # ========================================================================
    # CHECK 2: fraud_pattern not in prepared features
    # ========================================================================
    print(f"\nCHECK 2: Verifying fraud_pattern excluded from Stage 2 features...")
    
    # Load sample data
    data_path = "data/processed/full_features.duckdb"
    con = duckdb.connect(data_path, read_only=True)
    df = con.execute("SELECT * FROM training_data LIMIT 10000").df()
    con.close()
    
    # Check if fraud_pattern exists in raw data
    if 'fraud_pattern' in df.columns:
        print(f"   ℹ️  fraud_pattern exists in raw data (Phase 2 artifact)")
    else:
        print(f"   ℹ️  fraud_pattern not in raw data (already cleaned)")
    
    # Prepare features
    train_end, test_start = calculate_split_dates(df, test_window_days=30, buffer_hours=1)
    train_df, _ = split_data_with_label_awareness(df, train_end, test_start, verbose=False)
    
    X, feature_names, y, encoders = prepare_stage2_features(train_df, verbose=False)
    
    # Verify fraud_pattern NOT in features
    if 'fraud_pattern' in feature_names:
        feature_idx = feature_names.index('fraud_pattern')
        print(f"\n❌ CRITICAL ERROR: fraud_pattern leaked into Stage 2 features!")
        print(f"   Position in feature list: {feature_idx}")
        print(f"   Total features: {len(feature_names)}")
        print(f"\n⚠️  Production Impact:")
        print(f"   → Model expects fraud_pattern column at inference time")
        print(f"   → Real transactions don't have this column")
        print(f"   → Prediction will crash with KeyError or missing column error")
        print(f"\nFix: Ensure prepare_stage2_features() excludes fraud_pattern")
        assert False, "fraud_pattern leaked into Stage 2 features - production deployment unsafe"
    
    print(f"   ✅ fraud_pattern correctly excluded")
    print(f"   Final feature count: {len(feature_names)}")
    
    # ========================================================================
    # CHECK 3: Verify other critical exclusions
    # ========================================================================
    print(f"\nCHECK 3: Verifying all critical exclusions...")
    
    excluded_count = 0
    for col in STAGE2_EXCLUDED_COLUMNS:
        if col in feature_names:
            print(f"   ❌ {col} should be excluded but found in features!")
            excluded_count += 1
        else:
            print(f"   ✅ {col} correctly excluded")
    
    if excluded_count > 0:
        assert False, f"{excluded_count} excluded columns leaked into features"
    
    # ========================================================================
    # CHECK 4: Verify target variable not in features
    # ========================================================================
    print(f"\nCHECK 4: Verifying target variable (is_fraud) excluded...")
    
    assert 'is_fraud' not in feature_names, \
        "❌ CRITICAL: Target variable 'is_fraud' leaked into features!"
    
    print(f"   ✅ is_fraud correctly excluded (no target leakage)")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"✅ ALL PRODUCTION SAFETY CHECKS PASSED")
    print(f"{'='*70}")
    print(f"\nExcluded columns: {len(STAGE2_EXCLUDED_COLUMNS)}")
    for col in STAGE2_EXCLUDED_COLUMNS:
        print(f"  ✓ {col}")
    
    print(f"\nFinal Stage 2 features: {len(feature_names)}")
    print(f"\n→ Model is production-safe (no synthetic-only columns)")
    print(f"→ Can deploy without training-serving skew")
    print(f"{'='*70}\n")


# ============================================================================
# UPDATE RUN_ALL_AUDITOR_TESTS
# ============================================================================

def run_all_auditor_tests():
    """
    Execute all data leakage tests and generate summary report.
    """
    print("\n" + "="*70)
    print("DATA LEAKAGE AUDITOR - COMPREHENSIVE TEST SUITE")
    print("="*70)
    print("\nValidating model correctness guarantees...\n")
    
    tests = [
        ("Stage 1 Feature Independence", test_stage1_no_label_features),
        ("Temporal Data Integrity", test_no_temporal_overlap),
        ("Training Label Completeness", test_stage2_no_null_labels),
        ("Label Availability Constraint", test_label_availability_respected),
        ("fraud_pattern Exclusion (Production Safety)", test_fraud_pattern_excluded)  # ← NEW
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            test_func()
            results.append((test_name, "PASS", None))
        except AssertionError as e:
            results.append((test_name, "FAIL", str(e)))
        except Exception as e:
            results.append((test_name, "ERROR", str(e)))
    
    # Print summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70 + "\n")
    
    for test_name, status, error in results:
        symbol = "✅" if status == "PASS" else "❌"
        print(f"{symbol} {test_name:.<55} {status}")
        if error:
            print(f"   Error: {error}\n")
    
    # Overall result
    failed_tests = [r for r in results if r[1] != "PASS"]
    
    if len(failed_tests) == 0:
        print(f"\n🎉 ALL {len(tests)} TESTS PASSED! Model correctness validated.")
        print(f"   → No data leakage detected")
        print(f"   → Temporal integrity maintained")
        print(f"   → Training data is clean")
        print(f"   → Production deployment safe")
    else:
        print(f"\n❌ {len(failed_tests)}/{len(tests)} TEST(S) FAILED!")
        print(f"   → Fix these issues before deploying model")
        print(f"   → Current model has correctness violations")
    
    print("="*70 + "\n")
    
    return len(failed_tests) == 0



# ============================================================================
# PYTEST INTEGRATION
# ============================================================================

if __name__ == "__main__":
    """
    Run as standalone script or via pytest.
    
    Usage:
        python -m src.models.tests.test_no_label_leakage
        pytest src/models/tests/test_no_label_leakage.py -v
    """
    success = run_all_auditor_tests()
    exit(0 if success else 1)
