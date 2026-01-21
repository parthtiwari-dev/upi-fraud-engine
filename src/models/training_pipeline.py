"""
Phase 5: Two-Stage Fraud Detection Training Pipeline

Purpose:
End-to-end orchestration of Stage 1 (Isolation Forest) + Stage 2 (XGBoost)
for UPI fraud detection with comprehensive evaluation and artifact management.

Pipeline Flow:
1. Load Phase 4 feature data
2. Time-based train/test split (with label awareness)
3. Train Stage 1 (unsupervised anomaly detection)
4. Score anomalies on train and test sets
5. Train Stage 2 (supervised fraud classification)
6. Evaluate two-stage system vs baseline
7. Save all model artifacts and evaluation reports

Author: [Your Name]
Date: January 22, 2026
Phase: 5 (Model Training & Evaluation)
"""

import pandas as pd
import numpy as np
import duckdb
import pickle
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, Tuple
import warnings

# Import Phase 5 modules
from src.models.time_utils import (
    calculate_split_dates,
    split_data_with_label_awareness,
    validate_no_leakage
)
from src.models.stage1_anomaly import (
    train_stage1_pipeline,
    score_stage1_pipeline
)
from src.models.stage2_supervised import (
    prepare_stage2_features,
    train_xgboost,
    predict_fraud_probabilities,
    get_feature_importance,
    save_stage2_artifacts
)

# Evaluation metrics
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    classification_report
)


# ============================================================================
# CONFIGURATION
# ============================================================================

class PipelineConfig:
    """Configuration for training pipeline."""
    
    # Data paths
    DATA_PATH = "data/processed/full_features.duckdb"
    OUTPUT_DIR = "models/phase5_two_stage"
    
    # Time split parameters
    TEST_WINDOW_DAYS = 30
    BUFFER_HOURS = 1
    
    # Stage 1 (Isolation Forest) parameters
    STAGE1_CONTAMINATION = None  # Auto-calculated from fraud rate
    STAGE1_N_ESTIMATORS = 100
    STAGE1_RANDOM_STATE = 42
    
    # Stage 2 (XGBoost) parameters
    STAGE2_NUM_BOOST_ROUND = 300
    STAGE2_EARLY_STOPPING = 20
    STAGE2_VALIDATION_SPLIT = 0.2  # 20% of train for validation
    
    # Evaluation
    DECISION_THRESHOLDS = [0.5, 0.7, 0.9, 0.95]  # For precision analysis


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_two_stage_pipeline(
    config: PipelineConfig = None,
    verbose: bool = True
) -> Dict:
    """
    Execute complete two-stage fraud detection training pipeline.
    
    Args:
        config: Pipeline configuration (uses defaults if None)
        verbose: Print detailed progress
    
    Returns:
        Dict with all results, metrics, and artifact paths
    """
    if config is None:
        config = PipelineConfig()
    
    # Create output directory
    output_dir = Path(config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': vars(config),
        'stage1': {},
        'stage2': {},
        'evaluation': {},
        'artifacts': {}
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"TWO-STAGE FRAUD DETECTION TRAINING PIPELINE")
        print(f"Phase 5 - UPI Fraud Detection System")
        print(f"{'='*70}\n")
    
    # ========================================================================
    # STEP 1: Load Phase 4 Feature Data
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
        df,
        train_end,
        test_start,
        verbose=verbose
    )
    
    # Validate no temporal leakage
    validate_no_leakage(train_df, test_df, test_start)
    
    results['split'] = {
        'train_rows': len(train_df),
        'test_rows': len(test_df),
        'train_fraud_rate': (train_df['is_fraud'] == 1.0).sum() / len(train_df),
        'test_fraud_rate': (test_df['is_fraud'] == 1.0).sum() / len(test_df),
    }
    
    # ========================================================================
    # STEP 3: Train Stage 1 (Isolation Forest)
    # ========================================================================
    if verbose:
        print(f"\n🌲 STEP 3: Training Stage 1 (Isolation Forest)...")
    
    # Calculate contamination from actual fraud rate
    contamination = results['data']['fraud_rate']
    
    stage1_model, stage1_scaler = train_stage1_pipeline(
        train_df,
        contamination=contamination,
        random_state=config.STAGE1_RANDOM_STATE,
        verbose=verbose
    )
    
    results['stage1']['contamination'] = contamination
    results['stage1']['n_estimators'] = config.STAGE1_N_ESTIMATORS
    
    # ========================================================================
    # STEP 4: Score Anomalies (Add anomaly_score column)
    # ========================================================================
    if verbose:
        print(f"\n🎯 STEP 4: Scoring anomalies on train and test sets...")
    
    # Score train set
    train_anomaly_scores = score_stage1_pipeline(
        train_df,
        stage1_model,
        stage1_scaler,
        verbose=verbose
    )
    train_df['anomaly_score'] = train_anomaly_scores
    
    # Score test set
    test_anomaly_scores = score_stage1_pipeline(
        test_df,
        stage1_model,
        stage1_scaler,
        verbose=verbose
    )
    test_df['anomaly_score'] = test_anomaly_scores
    
    if verbose:
        print(f"   ✅ Added anomaly_score column to both sets")
        print(f"   Train: {len(train_df.columns)} columns (490 + anomaly_score)")
        print(f"   Test:  {len(test_df.columns)} columns (490 + anomaly_score)\n")
    
    # ========================================================================
    # STEP 5: Prepare Features for Stage 2
    # ========================================================================
    if verbose:
        print(f"🔧 STEP 5: Preparing features for Stage 2...")
    
    # Prepare train features (with validation split for early stopping)
    X_train_full, feature_names, y_train_full, encoders = prepare_stage2_features(
        train_df,
        verbose=verbose
    )
    
    # Split train into train/val for early stopping
    val_size = int(len(X_train_full) * config.STAGE2_VALIDATION_SPLIT)
    
    X_train = X_train_full.iloc[:-val_size]
    y_train = y_train_full.iloc[:-val_size]
    X_val = X_train_full.iloc[-val_size:]
    y_val = y_train_full.iloc[-val_size:]
    
    if verbose:
        print(f"\n   Split for early stopping:")
        print(f"   Train: {len(X_train):,}")
        print(f"   Val:   {len(X_val):,}")
    
    # Prepare test features
    X_test, _, y_test, _ = prepare_stage2_features(
        test_df,
        verbose=False
    )
    
    results['stage2']['num_features'] = len(feature_names)
    results['stage2']['has_anomaly_score'] = 'anomaly_score' in feature_names
    
    # ========================================================================
    # STEP 6: Train Stage 2 (XGBoost)
    # ========================================================================
    if verbose:
        print(f"\n🚀 STEP 6: Training Stage 2 (XGBoost)...")
    
    stage2_model = train_xgboost(
        X_train,
        y_train,
        X_val,
        y_val,
        num_boost_round=config.STAGE2_NUM_BOOST_ROUND,
        early_stopping_rounds=config.STAGE2_EARLY_STOPPING,
        verbose=verbose
    )
    
    results['stage2']['best_iteration'] = stage2_model.best_iteration
    
    # ========================================================================
    # STEP 7: Feature Importance Analysis
    # ========================================================================
    if verbose:
        print(f"\n📊 STEP 7: Analyzing feature importance...")
    
    importance_df = get_feature_importance(
        stage2_model,
        feature_names,
        importance_type='gain',
        top_n=20,
        verbose=verbose
    )
    
    # Save feature importance
    importance_path = output_dir / "feature_importance.csv"
    importance_df.to_csv(importance_path, index=False)
    results['artifacts']['feature_importance'] = str(importance_path)
    
    # Check anomaly_score rank
    if 'anomaly_score' in importance_df['feature'].values:
        anomaly_rank = importance_df[importance_df['feature'] == 'anomaly_score']['rank'].values[0]
        anomaly_importance = importance_df[importance_df['feature'] == 'anomaly_score']['importance'].values[0]
        results['stage1']['anomaly_score_rank'] = int(anomaly_rank)
        results['stage1']['anomaly_score_importance'] = float(anomaly_importance)
    
    # ========================================================================
    # STEP 8: Generate Predictions
    # ========================================================================
    if verbose:
        print(f"\n🎯 STEP 8: Generating fraud probability predictions...")
    
    test_probs = predict_fraud_probabilities(stage2_model, X_test, verbose=verbose)
    
    # ========================================================================
    # STEP 9: Comprehensive Evaluation
    # ========================================================================
    if verbose:
        print(f"\n📈 STEP 9: Evaluating two-stage system...")
    
    # Filter to labeled test data
    labeled_mask = y_test.notna()
    y_test_clean = y_test[labeled_mask]
    test_probs_clean = test_probs[labeled_mask]
    
    # Core metrics
    roc_auc = roc_auc_score(y_test_clean, test_probs_clean)
    pr_auc = average_precision_score(y_test_clean, test_probs_clean)
    
    results['evaluation']['roc_auc'] = float(roc_auc)
    results['evaluation']['pr_auc'] = float(pr_auc)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"EVALUATION RESULTS (TWO-STAGE SYSTEM)")
        print(f"{'='*70}")
        print(f"ROC-AUC:  {roc_auc:.4f}")
        print(f"PR-AUC:   {pr_auc:.4f}")
    
    # Precision/Recall at different thresholds
    precision, recall, thresholds = precision_recall_curve(y_test_clean, test_probs_clean)
    
    threshold_metrics = []
    for target_precision in config.DECISION_THRESHOLDS:
        # Find threshold closest to target precision
        valid_idx = precision >= target_precision
        if valid_idx.sum() > 0:
            idx = np.where(valid_idx)[0][-1]  # Last index with precision >= target
            threshold_metrics.append({
                'target_precision': target_precision,
                'actual_precision': float(precision[idx]),
                'recall': float(recall[idx]),
                'threshold': float(thresholds[idx]) if idx < len(thresholds) else 1.0
            })
    
    results['evaluation']['threshold_metrics'] = threshold_metrics
    
    if verbose:
        print(f"\nPrecision/Recall Trade-offs:")
        for tm in threshold_metrics:
            print(f"  Precision ≥ {tm['target_precision']:.0%}: Recall = {tm['recall']:.2%} @ threshold {tm['threshold']:.3f}")
    
    # Confusion matrix at default 0.5 threshold
    y_pred_binary = (test_probs_clean >= 0.5).astype(int)
    cm = confusion_matrix(y_test_clean, y_pred_binary)
    
    results['evaluation']['confusion_matrix'] = {
        'tn': int(cm[0, 0]),
        'fp': int(cm[0, 1]),
        'fn': int(cm[1, 0]),
        'tp': int(cm[1, 1])
    }
    
    if verbose:
        print(f"\nConfusion Matrix (threshold=0.5):")
        print(f"  TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
        print(f"  FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")
        print(f"{'='*70}\n")
    
    # ========================================================================
    # STEP 10: Save All Artifacts
    # ========================================================================
    if verbose:
        print(f"💾 STEP 10: Saving model artifacts...")
    
    # Save Stage 1
    stage1_model_path = output_dir / "stage1_isolation_forest.pkl"
    stage1_scaler_path = output_dir / "stage1_scaler.pkl"
    
    with open(stage1_model_path, 'wb') as f:
        pickle.dump(stage1_model, f)
    with open(stage1_scaler_path, 'wb') as f:
        pickle.dump(stage1_scaler, f)
    
    results['artifacts']['stage1_model'] = str(stage1_model_path)
    results['artifacts']['stage1_scaler'] = str(stage1_scaler_path)
    
    if verbose:
        print(f"   ✅ Stage 1 artifacts saved")
    
    # Save Stage 2 (with encoders)
    stage2_model_path = output_dir / "stage2_xgboost.json"
    
    save_stage2_artifacts(
        stage2_model,
        feature_names,
        encoders,
        str(stage2_model_path),
        metadata={
            'roc_auc': roc_auc,
            'pr_auc': pr_auc,
            'best_iteration': stage2_model.best_iteration,
            'anomaly_score_rank': results['stage1'].get('anomaly_score_rank', None)
        }
    )
    
    results['artifacts']['stage2_model'] = str(stage2_model_path)
    
    if verbose:
        print(f"   ✅ Stage 2 artifacts saved")
    
    # Save pipeline results
    results_path = output_dir / "pipeline_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    results['artifacts']['pipeline_results'] = str(results_path)
    
    if verbose:
        print(f"   ✅ Pipeline results saved to {results_path}")
    
    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    if verbose:
        print(f"\n{'='*70}")
        print(f"🎉 TWO-STAGE PIPELINE COMPLETE!")
        print(f"{'='*70}")
        print(f"\nPerformance Summary:")
        print(f"  ROC-AUC: {roc_auc:.4f}")
        print(f"  PR-AUC:  {pr_auc:.4f}")
        if 'anomaly_score_rank' in results['stage1']:
            print(f"  Anomaly Score Rank: #{results['stage1']['anomaly_score_rank']}")
        print(f"\nArtifacts saved to: {output_dir}")
        print(f"  - Stage 1 model + scaler")
        print(f"  - Stage 2 XGBoost + encoders + metadata")
        print(f"  - Feature importance analysis")
        print(f"  - Complete pipeline results (JSON)")
        print(f"\n{'='*70}\n")
    
    return results


# ============================================================================
# COMPARISON: Two-Stage vs Baseline
# ============================================================================

def compare_with_baseline(
    baseline_roc_auc: float = 0.9106,
    baseline_pr_auc: float = 0.5529
):
    """
    Run two-stage pipeline and compare to Stage 2 only baseline.
    
    Args:
        baseline_roc_auc: ROC-AUC from Stage 2 standalone test
        baseline_pr_auc: PR-AUC from Stage 2 standalone test
    """
    print(f"\n{'='*70}")
    print(f"BASELINE vs TWO-STAGE COMPARISON")
    print(f"{'='*70}\n")
    
    print(f"Running two-stage pipeline...")
    results = run_two_stage_pipeline(verbose=True)
    
    # Extract metrics
    two_stage_roc = results['evaluation']['roc_auc']
    two_stage_pr = results['evaluation']['pr_auc']
    
    # Calculate improvements
    roc_improvement = two_stage_roc - baseline_roc_auc
    pr_improvement = two_stage_pr - baseline_pr_auc
    
    roc_pct = (roc_improvement / baseline_roc_auc) * 100
    pr_pct = (pr_improvement / baseline_pr_auc) * 100
    
    # Print comparison
    print(f"\n{'='*70}")
    print(f"PERFORMANCE COMPARISON")
    print(f"{'='*70}")
    print(f"\n{'Metric':<20} {'Baseline':<15} {'Two-Stage':<15} {'Δ':<10} {'%':<10}")
    print(f"{'-'*70}")
    print(f"{'ROC-AUC':<20} {baseline_roc_auc:<15.4f} {two_stage_roc:<15.4f} {roc_improvement:<10.4f} {roc_pct:<10.2f}%")
    print(f"{'PR-AUC':<20} {baseline_pr_auc:<15.4f} {two_stage_pr:<15.4f} {pr_improvement:<10.4f} {pr_pct:<10.2f}%")
    print(f"{'='*70}\n")
    
    # Decision recommendation
    if roc_improvement >= 0.005:  # 0.5% improvement
        print(f"✅ RECOMMENDATION: KEEP Stage 1")
        print(f"   Two-stage system shows meaningful improvement (+{roc_pct:.2f}%)")
    elif roc_improvement >= 0.002:  # 0.2% improvement
        print(f"⚠️  RECOMMENDATION: Optional - marginal gain")
        print(f"   Improvement is small but Stage 1 adds architectural value")
    else:
        print(f"❌ RECOMMENDATION: DROP Stage 1 for production simplicity")
        print(f"   Improvement < 0.2% - Stage 2 alone is sufficient")
    
    print(f"\n{'='*70}\n")
    
    return results


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Run two-stage training pipeline and compare to baseline.
    """
    # Run comparison
    results = compare_with_baseline(
        baseline_roc_auc=0.9106,  # From your baseline test
        baseline_pr_auc=0.5529
    )
    
    print("Pipeline execution complete!")
    print(f"Check models/phase5_two_stage/ for all artifacts.")
