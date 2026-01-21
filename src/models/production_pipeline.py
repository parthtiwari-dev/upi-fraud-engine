"""
Production Fraud Detection Pipeline (Stage 2 Only)

BASELINE MODEL - OPTIMIZED CONFIGURATION
After A/B testing two-stage vs baseline, baseline performed better:
  - Baseline (Stage 2 only): 0.9106 ROC-AUC
  - Two-stage (IF + XGBoost): 0.9008 ROC-AUC

Dropped Stage 1 because:
  1. Stage 1 features redundant with Phase 4 features
  2. Anomaly score ranked #228/484 (minimal impact)
  3. Simpler is better (Occam's razor)

This is the PRODUCTION deployment pipeline.

Author: [Your Name]
Date: January 22, 2026
"""

import pandas as pd
import numpy as np
import duckdb
from pathlib import Path
from datetime import datetime
from typing import Dict
import json

from src.models.time_utils import (
    calculate_split_dates,
    split_data_with_label_awareness,
    validate_no_leakage
)
from src.models.stage2_supervised import (
    prepare_stage2_features,
    train_xgboost,
    predict_fraud_probabilities,
    get_feature_importance,
    save_stage2_artifacts
)
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    confusion_matrix
)


class ProductionConfig:
    """Optimized production configuration."""
    DATA_PATH = "data/processed/full_features.duckdb"
    OUTPUT_DIR = "models/production"
    TEST_WINDOW_DAYS = 30
    BUFFER_HOURS = 1
    NUM_BOOST_ROUND = 300
    EARLY_STOPPING = 20
    VALIDATION_SPLIT = 0.2


def train_production_model(config: ProductionConfig = None, verbose: bool = True) -> Dict:
    """
    Train production-optimized fraud detection model (Stage 2 only).
    
    This is the FINAL model based on A/B testing results.
    Uses only XGBoost with 483 Phase 4 features (no Stage 1 anomaly score).
    
    Args:
        config: Pipeline configuration (uses defaults if None)
        verbose: Print detailed progress
    
    Returns:
        Dict with performance metrics and artifact paths
    """
    if config is None:
        config = ProductionConfig()
    
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'model_type': 'XGBoost (Stage 2 Only - Production Baseline)',
        'config': vars(config)
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"PRODUCTION FRAUD DETECTION PIPELINE")
        print(f"Model: XGBoost (Stage 2 Only - No Stage 1)")
        print(f"Rationale: Baseline outperformed two-stage (0.9106 vs 0.9008)")
        print(f"{'='*70}\n")
    
    # ========================================================================
    # STEP 1: Load Data
    # ========================================================================
    if verbose:
        print(f"📂 STEP 1: Loading Phase 4 feature data...")
        print(f"   Source: {config.DATA_PATH}")
    
    con = duckdb.connect(config.DATA_PATH, read_only=True)
    df = con.execute("SELECT * FROM training_data").df()
    con.close()
    
    results['data'] = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'fraud_rate': (df['is_fraud'] == 1.0).sum() / len(df)
    }
    
    if verbose:
        print(f"   ✅ Loaded {len(df):,} transactions")
        print(f"   Columns: {len(df.columns)}")
        print(f"   Fraud rate: {results['data']['fraud_rate']:.2%}\n")
    
    # ========================================================================
    # STEP 2: Time-Based Train/Test Split
    # ========================================================================
    if verbose:
        print(f"🕒 STEP 2: Performing time-based split...")
    
    train_end, test_start = calculate_split_dates(
        df,
        test_window_days=config.TEST_WINDOW_DAYS,
        buffer_hours=config.BUFFER_HOURS
    )
    
    train_df, test_df = split_data_with_label_awareness(
        df, train_end, test_start, verbose=verbose
    )
    
    # Validate no temporal leakage
    validate_no_leakage(train_df, test_df, test_start)
    
    results['split'] = {
        'train_rows': len(train_df),
        'test_rows': len(test_df),
        'train_fraud_rate': (train_df['is_fraud'] == 1.0).sum() / len(train_df),
        'test_fraud_rate': (test_df['is_fraud'] == 1.0).sum() / len(test_df)
    }
    
    # ========================================================================
    # STEP 3: Prepare Features (NO Stage 1 anomaly_score)
    # ========================================================================
    if verbose:
        print(f"\n🔧 STEP 3: Preparing features (Stage 2 only)...")
    
    X_train_full, feature_names, y_train_full, encoders = prepare_stage2_features(
        train_df, verbose=verbose
    )
    
    # Train/val split for early stopping
    val_size = int(len(X_train_full) * config.VALIDATION_SPLIT)
    X_train = X_train_full.iloc[:-val_size]
    y_train = y_train_full.iloc[:-val_size]
    X_val = X_train_full.iloc[-val_size:]
    y_val = y_train_full.iloc[-val_size:]
    
    if verbose:
        print(f"\n   Split for early stopping:")
        print(f"   Train: {len(X_train):,}")
        print(f"   Val:   {len(X_val):,}")
    
    # Prepare test features
    X_test, _, y_test, _ = prepare_stage2_features(test_df, verbose=False)
    
    results['features'] = {
        'num_features': len(feature_names),
        'has_anomaly_score': 'anomaly_score' in feature_names
    }
    
    if verbose:
        print(f"\n   ✅ Final features: {len(feature_names)}")
        if 'anomaly_score' in feature_names:
            print(f"   ⚠️  WARNING: anomaly_score detected (should not be present)")
        else:
            print(f"   ✅ No anomaly_score (baseline configuration)")
    
    # ========================================================================
    # STEP 4: Train XGBoost
    # ========================================================================
    if verbose:
        print(f"\n🚀 STEP 4: Training XGBoost (Stage 2)...")
    
    model = train_xgboost(
        X_train, y_train, X_val, y_val,
        num_boost_round=config.NUM_BOOST_ROUND,
        early_stopping_rounds=config.EARLY_STOPPING,
        verbose=verbose
    )
    
    results['training'] = {
        'best_iteration': model.best_iteration
    }
    
    # ========================================================================
    # STEP 5: Feature Importance Analysis
    # ========================================================================
    if verbose:
        print(f"\n📊 STEP 5: Analyzing feature importance...")
    
    importance_df = get_feature_importance(
        model, feature_names,
        importance_type='gain',
        top_n=20,
        verbose=verbose
    )
    
    # Save feature importance
    importance_path = output_dir / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    
    # ========================================================================
    # STEP 6: Generate Predictions & Evaluate
    # ========================================================================
    if verbose:
        print(f"\n🎯 STEP 6: Generating predictions and evaluating...")
    
    test_probs = predict_fraud_probabilities(model, X_test, verbose=verbose)
    
    # Filter to labeled test data
    labeled_mask = y_test.notna()
    y_test_clean = y_test[labeled_mask]
    test_probs_clean = test_probs[labeled_mask]
    
    # Core metrics
    roc_auc = roc_auc_score(y_test_clean, test_probs_clean)
    pr_auc = average_precision_score(y_test_clean, test_probs_clean)
    
    results['performance'] = {
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc)
    }
    
    # Precision at 0.5% alert budget
    threshold_0_5pct = np.percentile(test_probs_clean, 99.5)
    y_pred_0_5pct = (test_probs_clean >= threshold_0_5pct).astype(int)
    cm = confusion_matrix(y_test_clean, y_pred_0_5pct)
    tn, fp, fn, tp = cm.ravel()
    
    precision_0_5pct = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall_0_5pct = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    
    results['business_metrics'] = {
        'precision_at_0.5pct_budget': float(precision_0_5pct),
        'recall_at_0.5pct_budget': float(recall_0_5pct),
        'threshold_0.5pct': float(threshold_0_5pct)
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"PRODUCTION MODEL PERFORMANCE")
        print(f"{'='*70}")
        print(f"\nStandard Metrics:")
        print(f"  ROC-AUC:  {roc_auc:.4f}")
        print(f"  PR-AUC:   {pr_auc:.4f}")
        
        print(f"\nBusiness Metrics (0.5% alert budget):")
        print(f"  Precision: {precision_0_5pct:.2%}")
        print(f"  Recall:    {recall_0_5pct:.2%}")
        print(f"  Threshold: {threshold_0_5pct:.4f}")
        
        print(f"\nModel Configuration:")
        print(f"  Features:       {len(feature_names)}")
        print(f"  Best iteration: {model.best_iteration}")
        print(f"  Model type:     XGBoost (Stage 2 only)")
        print(f"{'='*70}\n")
    
    # ========================================================================
    # STEP 7: Save Production Artifacts
    # ========================================================================
    if verbose:
        print(f"💾 STEP 7: Saving production artifacts...")
    
    model_path = output_dir / "fraud_detector.json"
    
    save_stage2_artifacts(
        model, feature_names, encoders, str(model_path),
        metadata={
            'model_type': 'XGBoost (Stage 2 Only - Production)',
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'precision_at_0.5pct': precision_0_5pct,
            'recall_at_0.5pct': recall_0_5pct,
            'num_features': len(feature_names),
            'best_iteration': model.best_iteration,
            'training_date': datetime.now().isoformat(),
            'ab_test_winner': 'Baseline (Stage 2 only) outperformed two-stage'
        }
    )
    
    results['artifacts'] = {
        'model_path': str(model_path),
        'feature_importance': str(importance_path)
    }
    
    # Save pipeline results
    results_path = output_dir / "pipeline_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    if verbose:
        print(f"   ✅ Model saved to {model_path}")
        print(f"   ✅ Feature importance saved to {importance_path}")
        print(f"   ✅ Results saved to {results_path}\n")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    if verbose:
        print(f"{'='*70}")
        print(f"🎉 PRODUCTION PIPELINE COMPLETE!")
        print(f"{'='*70}")
        print(f"\nFinal Model:")
        print(f"  Type:        XGBoost (Stage 2 Only)")
        print(f"  ROC-AUC:     {roc_auc:.4f}")
        print(f"  PR-AUC:      {pr_auc:.4f}")
        print(f"  Features:    {len(feature_names)}")
        print(f"  Location:    {output_dir}/")
        
        print(f"\nWhy This Configuration?")
        print(f"  ✅ Simpler than two-stage (no Stage 1)")
        print(f"  ✅ Better performance (0.9106 vs 0.9008)")
        print(f"  ✅ Production-ready (483 robust features)")
        
        print(f"\nNext Steps:")
        print(f"  1. Run evaluation: python -c 'from src.evaluation.evaluation ...'")
        print(f"  2. Deploy for inference (batch or API)")
        print(f"  3. Monitor performance in production")
        print(f"\n{'='*70}\n")
    
    return results


# ============================================================================
# COMPARISON WITH A/B TEST
# ============================================================================

def compare_with_ab_test(baseline_results: Dict = None):
    """
    Compare production model with A/B test results.
    """
    print(f"\n{'='*70}")
    print(f"PRODUCTION vs A/B TEST COMPARISON")
    print(f"{'='*70}\n")
    
    # Run production pipeline
    prod_results = train_production_model(verbose=True)
    
    # Default A/B test results
    if baseline_results is None:
        baseline_results = {
            'two_stage': {'roc_auc': 0.9008, 'pr_auc': 0.5572},
            'baseline': {'roc_auc': 0.9106, 'pr_auc': 0.5529}
        }
    
    print(f"\n{'='*70}")
    print(f"PERFORMANCE SUMMARY")
    print(f"{'='*70}\n")
    
    print(f"{'Configuration':<30} {'ROC-AUC':<12} {'PR-AUC':<12} {'Status':<20}")
    print(f"{'-'*70}")
    print(f"{'Two-Stage (IF + XGBoost)':<30} {baseline_results['two_stage']['roc_auc']:<12.4f} {baseline_results['two_stage']['pr_auc']:<12.4f} {'❌ Rejected':<20}")
    print(f"{'Baseline (XGBoost only)':<30} {baseline_results['baseline']['roc_auc']:<12.4f} {baseline_results['baseline']['pr_auc']:<12.4f} {'✅ Selected':<20}")
    print(f"{'Production (This Run)':<30} {prod_results['performance']['roc_auc']:<12.4f} {prod_results['performance']['pr_auc']:<12.4f} {'✅ Deployed':<20}")
    print(f"{'='*70}\n")
    
    return prod_results


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Train production fraud detection model.
    
    Usage:
        python -m src.models.production_pipeline
    """
    results = compare_with_ab_test()
    
    print("✅ Production model training complete!")
    print(f"   Model: {results['artifacts']['model_path']}")
    print(f"   ROC-AUC: {results['performance']['roc_auc']:.4f}")
    print(f"\nReady for deployment! 🚀")
