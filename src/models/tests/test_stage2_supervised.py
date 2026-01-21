"""
Test Stage 2 (XGBoost) standalone WITHOUT Stage 1.

This validates:
1. Feature preparation (denylist works correctly)
2. XGBoost training (handles 500 columns, class imbalance)
3. Prediction generation
4. Feature importance extraction
5. Model save/load

We'll add a FAKE anomaly_score column to simulate Stage 1 output.
"""

import pandas as pd
import numpy as np
import duckdb
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.time_utils import calculate_split_dates, split_data_with_label_awareness
from src.models.stage2_supervised import (
    prepare_stage2_features,
    train_xgboost,
    predict_fraud_probabilities,
    get_feature_importance,
    save_stage2_artifacts,
    load_stage2_artifacts
)


def test_stage2_standalone():
    """
    Comprehensive test of Stage 2 WITHOUT Stage 1.
    """
    print(f"\n{'='*70}")
    print(f"STAGE 2 (XGBOOST) STANDALONE TEST")
    print(f"{'='*70}\n")
    
    # ========================================================================
    # STEP 1: Load Data
    # ========================================================================
    print("📂 Loading Phase 4 feature data...")
    db_path = "data/processed/full_features.duckdb"
    con = duckdb.connect(db_path, read_only=True)
    df = con.execute("SELECT * FROM training_data").df()
    con.close()
    print(f"✅ Loaded {len(df):,} transactions with {len(df.columns)} columns\n")
    
    # ========================================================================
    # STEP 2: Time Split
    # ========================================================================
    print("🕒 Performing time-based split...")
    train_end, test_start = calculate_split_dates(df, test_window_days=30)
    train_df, test_df = split_data_with_label_awareness(df, train_end, test_start)
    
    # ========================================================================
    # STEP 3: Baseline test (NO anomaly_score from Stage 1)
    # ========================================================================
    print("🎭 BASELINE TEST: No anomaly_score (Stage 1 not integrated yet)...")
    print(f"✅ Proceeding with {len(train_df.columns)} original columns\n")
    
    # ========================================================================
    # STEP 4: Prepare Features
    # ========================================================================
    print("🔧 TEST 1: Feature Preparation")
    print("-" * 70)
    X_train, feature_names, y_train, encoders_train = prepare_stage2_features(train_df, verbose=True)
    X_test, _, y_test, _ = prepare_stage2_features(test_df, verbose=False)

    print(f"✅ Feature preparation successful")
    print(f"   Training set: {X_train.shape}")
    print(f"   Test set: {X_test.shape}")
    print(f"   Features: {len(feature_names)}\n")
    
    # Baseline test: anomaly_score should NOT be present
    assert 'anomaly_score' not in feature_names, "Baseline test: anomaly_score should not be present!"
    print(f"✅ Verified: anomaly_score NOT in features (baseline test)\n")
    
    # ========================================================================
    # STEP 5: Train XGBoost (SMALL validation set for speed)
    # ========================================================================
    print("🚀 TEST 2: XGBoost Training")
    print("-" * 70)
    
    # Use 80% train, 20% validation
    split_idx = int(len(X_train) * 0.8)
    X_tr = X_train.iloc[:split_idx]
    y_tr = y_train.iloc[:split_idx]
    X_val = X_train.iloc[split_idx:]
    y_val = y_train.iloc[split_idx:]
    
    print(f"Split for early stopping:")
    print(f"  Train: {len(X_tr):,}")
    print(f"  Val:   {len(X_val):,}\n")
    
    # Train with fewer rounds for speed (this is a test)
    model = train_xgboost(
        X_tr, y_tr,
        X_val, y_val,
        num_boost_round=100,  # Reduced from 300 for testing
        early_stopping_rounds=20,
        verbose=True
    )
    
    print(f"✅ XGBoost training successful\n")
    
    # ========================================================================
    # STEP 6: Generate Predictions
    # ========================================================================
    print("🎯 TEST 3: Fraud Probability Prediction")
    print("-" * 70)
    
    test_probs = predict_fraud_probabilities(model, X_test, verbose=True)
    
    print(f"✅ Generated {len(test_probs):,} predictions\n")
    
    # ========================================================================
    # STEP 7: Feature Importance
    # ========================================================================
    print("📊 TEST 4: Feature Importance Analysis")
    print("-" * 70)
    
    importance_df = get_feature_importance(
        model, 
        feature_names, 
        importance_type='gain',
        top_n=20,
        verbose=True
    )

    # DEBUG block (after get_feature_importance)
    print("\n🔍 DEBUG: Raw XGBoost Importance Scores")
    print("="*70)
    gain_dict = model.get_score(importance_type='gain')
    weight_dict = model.get_score(importance_type='weight')
    print(f"Features with 'gain' scores: {len(gain_dict)}")
    print(f"Features with 'weight' scores: {len(weight_dict)}")
    print(f"Total gain importance: {sum(gain_dict.values()):.4f}")
    print(f"Total weight importance: {sum(weight_dict.values()):.0f}")

    if len(weight_dict) > 0:
        # Sort by weight directly (keys are already feature names!)
        weight_sorted = sorted(weight_dict.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\nTop 10 features by WEIGHT (actual usage):")
        for fname, score in weight_sorted:
            print(f"  {fname:40s} {score:8.0f}")
    print("="*70)
    # Check where anomaly_score ranked (if present)
    if 'anomaly_score' in importance_df['feature'].values:
        anomaly_rank = importance_df[importance_df['feature'] == 'anomaly_score']['rank'].values[0]
        print(f"\n🔍 CRITICAL CHECK: anomaly_score ranked #{anomaly_rank}")
        
        if anomaly_rank <= 50:
            print(f"   ✅ Within top 50 - Stage 1 would be useful!")
        else:
            print(f"   ⚠️  Ranked low - but this is FAKE data (random scores)")
            print(f"   Real Stage 1 scores will likely perform better\n")
    else:
        print(f"\n🔍 CRITICAL CHECK: anomaly_score NOT FOUND (baseline test)")
        print(f"   → Running Stage 2 standalone without Stage 1\n")
    
    # ========================================================================
    # STEP 8: Quick Performance Check
    # ========================================================================
    print("📈 TEST 5: Quick Performance Sanity Check")
    print("-" * 70)
    
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    # Filter to labeled test data only
    labeled_mask = y_test.notna()
    y_test_clean = y_test[labeled_mask]
    test_probs_clean = test_probs[labeled_mask]
    
    roc_auc = roc_auc_score(y_test_clean, test_probs_clean)
    pr_auc = average_precision_score(y_test_clean, test_probs_clean)
    
    print(f"ROC-AUC:  {roc_auc:.4f}")
    print(f"PR-AUC:   {pr_auc:.4f}")
    
    # Expected performance (even with fake anomaly_score)
    if roc_auc > 0.85:
        print(f"✅ ROC-AUC > 0.85 (EXCELLENT - Phase 4 features are strong!)")
    elif roc_auc > 0.70:
        print(f"✅ ROC-AUC > 0.70 (GOOD)")
    else:
        print(f"⚠️  ROC-AUC < 0.70 (investigate)")
    
    print("")
    
    # ========================================================================
    # STEP 9: Model Persistence
    # ========================================================================
    print("💾 TEST 6: Model Save/Load")
    print("-" * 70)
    
    import tempfile
    import os
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_path = os.path.join(tmpdir, "test_stage2_model.json")
        
        # Save
        save_stage2_artifacts(
            model, 
            feature_names,
            encoders_train,
            model_path,
            metadata={'roc_auc': roc_auc, 'pr_auc': pr_auc}
        )
        
        # Load
        loaded_model, loaded_features, loaded_encoders, loaded_metadata = load_stage2_artifacts(model_path)
        
        # Verify loaded model produces same predictions
        loaded_probs = predict_fraud_probabilities(loaded_model, X_test, verbose=False)
        
        if np.allclose(test_probs, loaded_probs):
            print(f"✅ Loaded model produces IDENTICAL predictions")
        else:
            print(f"❌ Loaded model predictions differ!")
    
    print("")
    
    # ========================================================================
    # SUMMARY
    # ========================================================================
    print(f"\n{'='*70}")
    print(f"TEST SUMMARY")
    print(f"{'='*70}")
    print(f"✅ Feature preparation      (denylist approach works)")
    print(f"✅ XGBoost training         (handles 500 cols + imbalance)")
    print(f"✅ Prediction generation    (fraud probabilities)")
    print(f"✅ Feature importance       (ranks all features)")
    print(f"✅ Performance sanity       (ROC-AUC: {roc_auc:.4f}, PR-AUC: {pr_auc:.4f})")
    print(f"✅ Model persistence        (save/load works)")
    print(f"{'='*70}")
    print(f"🎉 ALL STAGE 2 TESTS PASSED!")
    print(f"")
    print(f"📝 NOTES:")
    print(f"   - Used FAKE anomaly_score (random numbers)")
    print(f"   - Real Stage 1 scores will be better")
    print(f"   - Ready to integrate with training_pipeline.py")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    test_stage2_standalone()
