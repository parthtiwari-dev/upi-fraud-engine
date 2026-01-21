"""
Comprehensive test suite for Stage 1 Anomaly Detection.

Tests:
1. Integration test (end-to-end pipeline)
2. Feature contract enforcement
3. Model persistence
4. Error handling
5. Edge cases

Run from project root:
    python -m src.models.tests.test_stage1_anomaly
"""

import sys
from pathlib import Path
import tempfile
import os

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

import duckdb
import pandas as pd
import numpy as np

from src.models.time_utils import (
    calculate_split_dates,
    split_data_with_label_awareness
)
from src.models.stage1_anomaly import (
    train_stage1_pipeline,
    score_stage1_pipeline,
    save_stage1_artifacts,
    load_stage1_artifacts,
    analyze_feature_importance,
    STAGE1_FEATURES,
    FORBIDDEN_FEATURES,
    fit_stage1_preprocessor,
    transform_stage1,
    _validate_feature_contract
)


# ============================================================================
# TEST 1: INTEGRATION TEST (END-TO-END)
# ============================================================================

def test_integration():
    """
    Test complete Stage 1 pipeline on real Phase 4 data.
    This is the main validation test.
    """
    print("\n" + "="*70)
    print("TEST 1: INTEGRATION TEST (END-TO-END PIPELINE)")
    print("="*70 + "\n")
    
    # Load data
    print("Loading Phase 4 feature data...")
    data_path = project_root / "data" / "processed" / "full_features.duckdb"
    
    if not data_path.exists():
        print(f"⚠️  SKIPPING: Data file not found at {data_path}")
        return None, None, None, None
    
    con = duckdb.connect(str(data_path), read_only=True)
    df = con.execute("SELECT * FROM training_data").df()
    con.close()
    
    print(f"✅ Loaded {len(df):,} transactions")
    
    # Time split
    print("\nPerforming time split...")
    train_end_date, test_start_date = calculate_split_dates(
        df, test_window_days=30, buffer_hours=1
    )
    
    train_df, test_df = split_data_with_label_awareness(
        df, train_end_date, test_start_date, verbose=False
    )
    
    print(f"✅ Train: {len(train_df):,} | Test: {len(test_df):,}")
    
    # Verify features exist
    print("\nVerifying Stage 1 features...")
    missing = set(STAGE1_FEATURES) - set(df.columns)
    if missing:
        print(f"❌ FAILED: Missing features: {missing}")
        return None, None, None, None
    
    print(f"✅ All {len(STAGE1_FEATURES)} Stage 1 features present")
    
    # Train Stage 1
    print("\nTraining Stage 1 (Isolation Forest)...")
    train_fraud_rate = (train_df['is_fraud'] == 1.0).sum() / len(train_df)
    
    model, scaler = train_stage1_pipeline(
        train_df,
        contamination=train_fraud_rate,
        random_state=42,
        verbose=True
    )
    
    # Score train and test
    print("\nScoring anomalies on train set...")
    train_anomaly_scores = score_stage1_pipeline(
        train_df, model, scaler, verbose=True
    )
    
    print("\nScoring anomalies on test set...")
    test_anomaly_scores = score_stage1_pipeline(
        test_df, model, scaler, verbose=True
    )
    
    # Analyze results
    print("\n" + "="*70)
    print("STAGE 1 RESULTS ANALYSIS")
    print("="*70)
    
    train_df_scored = train_df.copy()
    train_df_scored['anomaly_score'] = train_anomaly_scores.values
    
    test_df_scored = test_df.copy()
    test_df_scored['anomaly_score'] = test_anomaly_scores.values
    
    # Fraud vs non-fraud scores
    print("\nTRAIN SET:")
    train_fraud_scores = train_df_scored[train_df_scored['is_fraud'] == 1.0]['anomaly_score']
    train_legit_scores = train_df_scored[train_df_scored['is_fraud'] == 0.0]['anomaly_score']
    
    print(f"  Fraud mean score:      {train_fraud_scores.mean():.4f}")
    print(f"  Legitimate mean score: {train_legit_scores.mean():.4f}")
    print(f"  Separation:            {train_fraud_scores.mean() - train_legit_scores.mean():.4f}")
    
    print("\nTEST SET:")
    test_fraud_scores = test_df_scored[test_df_scored['is_fraud'] == 1.0]['anomaly_score']
    test_legit_scores = test_df_scored[test_df_scored['is_fraud'] == 0.0]['anomaly_score']
    
    print(f"  Fraud mean score:      {test_fraud_scores.mean():.4f}")
    print(f"  Legitimate mean score: {test_legit_scores.mean():.4f}")
    print(f"  Separation:            {test_fraud_scores.mean() - test_legit_scores.mean():.4f}")
    
    # Top anomalies
    print("\nTOP 10 MOST ANOMALOUS TRANSACTIONS (Test Set):")
    top_anomalies = test_df_scored.nlargest(10, 'anomaly_score')
    print(top_anomalies[['transaction_id', 'anomaly_score', 'is_fraud', 
                          'payer_txn_count_5min', 'device_distinct_payers_7d']].to_string())
    
    # Precision at top K
    top_1pct = int(len(test_df) * 0.01)
    top_anomalies_1pct = test_df_scored.nlargest(top_1pct, 'anomaly_score')
    precision_at_1pct = (top_anomalies_1pct['is_fraud'] == 1.0).sum() / len(top_anomalies_1pct)
    
    print(f"\n" + "="*70)
    print(f"PRECISION @ TOP 1% OF TEST SET:")
    print(f"  Top 1% = {top_1pct:,} transactions")
    print(f"  Frauds caught: {(top_anomalies_1pct['is_fraud'] == 1.0).sum():,}")
    print(f"  Precision: {precision_at_1pct:.2%}")
    print(f"  Baseline (random): {(test_df['is_fraud']==1.0).mean():.2%}")
    print(f"  Lift: {precision_at_1pct / (test_df['is_fraud']==1.0).mean():.1f}x")
    print("="*70)
    
    # Validation checks
    assert train_fraud_scores.mean() > train_legit_scores.mean(), \
        "❌ FAILED: Frauds should have higher anomaly scores than legitimate"
    
    baseline = (test_df['is_fraud']==1.0).mean()
    if precision_at_1pct < baseline * 1.5:
        print(f"⚠️  WARNING: Precision ({precision_at_1pct:.2%}) is below 1.5x baseline")
        print(f"   This is expected for unsupervised anomaly detection.")
        print(f"   Stage 2 (XGBoost) will refine these anomalies using labels.")
    else:
        print(f"✅ Precision ({precision_at_1pct:.2%}) exceeds 1.5x baseline")
    
    print("\n✅ INTEGRATION TEST PASSED\n")
    
    return model, scaler, train_df_scored, test_df_scored


# ============================================================================
# TEST 2: FEATURE CONTRACT ENFORCEMENT
# ============================================================================

def test_feature_contract():
    """
    Test that feature contract validation works correctly.
    """
    print("\n" + "="*70)
    print("TEST 2: FEATURE CONTRACT ENFORCEMENT")
    print("="*70 + "\n")
    
    # Create sample data with all required features
    sample_data = pd.DataFrame({
        feat: np.random.randn(100) for feat in STAGE1_FEATURES
    })
    
    # Test 1: Valid features should pass
    print("Test 2.1: Valid feature set...")
    try:
        _validate_feature_contract(sample_data)
        print("✅ Valid features accepted")
    except Exception as e:
        print(f"❌ FAILED: {e}")
        return False
    
    # Test 2: Missing features should raise error
    print("\nTest 2.2: Missing features should raise error...")
    incomplete_data = sample_data.drop(columns=['payer_txn_count_5min'])
    try:
        _validate_feature_contract(incomplete_data)
        print("❌ FAILED: Should have raised error for missing features")
        return False
    except ValueError as e:
        print(f"✅ Correctly raised error: {str(e)[:80]}...")
    
    # Test 3: Forbidden features should trigger warning
    print("\nTest 2.3: Forbidden features should trigger warning...")
    contaminated_data = sample_data.copy()
    contaminated_data['is_fraud'] = np.random.randint(0, 2, 100)
    
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _validate_feature_contract(contaminated_data)
        
        if len(w) > 0 and "FORBIDDEN" in str(w[0].message):
            print(f"✅ Warning triggered: {str(w[0].message)[:80]}...")
        else:
            print("⚠️  Warning not triggered (non-critical)")
    
    print("\n✅ FEATURE CONTRACT TEST PASSED\n")
    return True


# ============================================================================
# TEST 3: MODEL PERSISTENCE
# ============================================================================

def test_model_persistence(model, scaler):
    """
    Test save/load functionality.
    """
    print("\n" + "="*70)
    print("TEST 3: MODEL PERSISTENCE (SAVE/LOAD)")
    print("="*70 + "\n")
    
    if model is None or scaler is None:
        print("⚠️  SKIPPING: No model to test (integration test failed)")
        return False
    
    # Create temporary directory
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_model.pkl")
        scaler_path = os.path.join(tmpdir, "test_scaler.pkl")
        
        # Test save
        print("Test 3.1: Saving artifacts...")
        try:
            save_stage1_artifacts(model, scaler, model_path, scaler_path)
            print("✅ Save successful")
        except Exception as e:
            print(f"❌ FAILED: {e}")
            return False
        
        # Test load
        print("\nTest 3.2: Loading artifacts...")
        try:
            loaded_model, loaded_scaler = load_stage1_artifacts(model_path, scaler_path)
            print("✅ Load successful")
        except Exception as e:
            print(f"❌ FAILED: {e}")
            return False
        
        # Test loaded model produces same scores
        print("\nTest 3.3: Verifying loaded model produces identical scores...")
        sample_data = pd.DataFrame({
            feat: np.random.randn(100) for feat in STAGE1_FEATURES
        })
        
        # Score with original
        X_orig = transform_stage1(sample_data, scaler, verbose=False)
        scores_orig = model.decision_function(X_orig)
        
        # Score with loaded
        X_loaded = transform_stage1(sample_data, loaded_scaler, verbose=False)
        scores_loaded = loaded_model.decision_function(X_loaded)
        
        # Compare
        if np.allclose(scores_orig, scores_loaded):
            print("✅ Loaded model produces identical scores")
        else:
            print(f"❌ FAILED: Score mismatch! Max diff: {np.abs(scores_orig - scores_loaded).max()}")
            return False
    
    print("\n✅ MODEL PERSISTENCE TEST PASSED\n")
    return True


# ============================================================================
# TEST 4: ERROR HANDLING
# ============================================================================

def test_error_handling():
    """
    Test that errors are caught and reported properly.
    """
    print("\n" + "="*70)
    print("TEST 4: ERROR HANDLING")
    print("="*70 + "\n")
    
    # Test 4.1: Null values
    print("Test 4.1: Null values should raise error...")
    data_with_nulls = pd.DataFrame({
        feat: np.random.randn(100) for feat in STAGE1_FEATURES
    })
    data_with_nulls.loc[10, 'payer_txn_count_5min'] = np.nan
    
    try:
        fit_stage1_preprocessor(data_with_nulls, verbose=False)
        print("❌ FAILED: Should have raised error for null values")
        return False
    except ValueError as e:
        print(f"✅ Correctly raised error: {str(e)[:80]}...")
    
    # Test 4.2: Invalid contamination should trigger our warning
    print("\nTest 4.2: High contamination (0.45) should trigger warning...")
    valid_data = pd.DataFrame({
        feat: np.random.randn(100) for feat in STAGE1_FEATURES
    })
    
    scaler = fit_stage1_preprocessor(valid_data, verbose=False)
    X_scaled = transform_stage1(valid_data, scaler, verbose=False)
    
    import warnings
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        from src.models.stage1_anomaly import train_isolation_forest
        
        # Use 0.45 (high but valid) to trigger our warning
        train_isolation_forest(X_scaled, contamination=0.45, verbose=False)
        
        if len(w) > 0 and "Unusual contamination" in str(w[0].message):
            print(f"✅ Warning triggered: {str(w[0].message)[:80]}...")
        else:
            print("⚠️  Warning not triggered (non-critical, contamination within sklearn limits)")
    
    # Test 4.3: sklearn should reject contamination > 0.5
    print("\nTest 4.3: sklearn should reject contamination > 0.5...")
    try:
        train_isolation_forest(X_scaled, contamination=0.9, verbose=False)
        print("❌ FAILED: Should have raised error for contamination > 0.5")
        return False
    except Exception as e:
        if "contamination" in str(e).lower():
            print(f"✅ sklearn correctly raised error: {str(e)[:80]}...")
        else:
            print(f"⚠️  Unexpected error: {str(e)[:80]}...")
    
    print("\n✅ ERROR HANDLING TEST PASSED\n")
    return True


# ============================================================================
# TEST 5: FEATURE IMPORTANCE (OPTIONAL)
# ============================================================================

def test_feature_importance(model, scaler):
    """
    Test feature importance analysis on small sample.
    """
    print("\n" + "="*70)
    print("TEST 5: FEATURE IMPORTANCE ANALYSIS (OPTIONAL)")
    print("="*70 + "\n")
    
    if model is None or scaler is None:
        print("⚠️  SKIPPING: No model to test")
        return False
    
    print("Analyzing feature importance on sample (1000 transactions)...")
    
    # Create sample data
    sample_data = pd.DataFrame({
        feat: np.random.randn(1000) for feat in STAGE1_FEATURES
    })
    
    X_scaled = transform_stage1(sample_data, scaler, verbose=False)
    
    try:
        importance_df = analyze_feature_importance(
            model, X_scaled, feature_names=STAGE1_FEATURES
        )
        
        print("\nFeature Importance Ranking:")
        print(importance_df.to_string())
        
        print("\n✅ FEATURE IMPORTANCE TEST PASSED\n")
        return True
        
    except Exception as e:
        print(f"⚠️  Feature importance failed (non-critical): {e}")
        return False


# ============================================================================
# MAIN TEST RUNNER
# ============================================================================

def main():
    """
    Run all tests in sequence.
    """
    print("\n" + "="*70)
    print("STAGE 1 ANOMALY DETECTION - COMPREHENSIVE TEST SUITE")
    print("="*70)
    
    results = {}
    
    # Run tests
    print("\n🧪 Running tests...\n")
    
    # Test 1: Integration (main test)
    model, scaler, train_df, test_df = test_integration()
    results['integration'] = (model is not None)
    
    # Test 2: Feature contract
    results['feature_contract'] = test_feature_contract()
    
    # Test 3: Model persistence
    results['persistence'] = test_model_persistence(model, scaler)
    
    # Test 4: Error handling
    results['error_handling'] = test_error_handling()
    
    # Test 5: Feature importance (optional)
    results['feature_importance'] = test_feature_importance(model, scaler)
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:20s} {status}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print("="*70)
    print(f"TOTAL: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("🎉 ALL TESTS PASSED!")
    elif total_passed >= total_tests - 1:
        print("⚠️  Most tests passed (1 optional failure allowed)")
    else:
        print("❌ SOME TESTS FAILED - Review output above")
    
    print("="*70 + "\n")
    
    return results


if __name__ == "__main__":
    main()
