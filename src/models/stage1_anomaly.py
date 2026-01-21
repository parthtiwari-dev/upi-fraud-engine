"""
Stage 1: Unsupervised Anomaly Detection (Isolation Forest)

Purpose:
Detect behavioral anomalies WITHOUT using fraud labels.
This stage identifies structurally unusual transactions based on velocity,
graph density, and device patterns.

Design Constraint:
- MUST NOT see label-derived features (payer_past_fraud_count_30d)
- MUST NOT see fraud labels during training
- Output is a continuous anomaly score in [0, 1]

This is the discovery layer. Stage 2 is the refinement layer.
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional
import warnings
import pickle


# ============================================================================
# FEATURE CONTRACT (STRICT)
# ============================================================================
# Stage 1 ONLY uses behavioral features, never label-derived information

STAGE1_FEATURES = [
    # Velocity features (6)
    'payer_txn_count_5min',
    'payer_txn_sum_5min',
    'payer_txn_count_1h',
    'payer_txn_sum_1h',
    'payer_txn_count_24h',
    'payer_txn_sum_24h',
    
    # Device velocity (2)
    'device_txn_count_1h',
    'device_txn_count_24h',
    
    # Graph density - ROWS-based, not time-based (2)
    'device_distinct_payers_7d',
    'payer_distinct_payees_7d',
]

# Features that MUST NEVER appear in Stage 1
FORBIDDEN_FEATURES = [
    'payer_past_fraud_count_30d',  # Label-derived
    'is_fraud',                     # Target variable
    'label_available_timestamp',    # Label metadata
]


# ============================================================================
# PREPROCESSING
# ============================================================================

def fit_stage1_preprocessor(train_df: pd.DataFrame, verbose: bool = True) -> StandardScaler:
    """
    Fit StandardScaler on TRAINING data only.
    
    This is the PRIMARY defense against test distribution leakage.
    The scaler learns mean/std ONLY from training data, then applies
    the SAME transformation to test data.
    
    Args:
        train_df: Training DataFrame with Stage 1 features
        verbose: Print diagnostics
    
    Returns:
        Fitted StandardScaler
    
    Raises:
        ValueError: If required features missing or forbidden features present
    
    Example:
        >>> scaler = fit_stage1_preprocessor(train_df)
        >>> # Later: X_train_scaled = scaler.transform(X_train)
    """
    # Validate feature contract
    _validate_feature_contract(train_df)
    
    # Extract features
    X_train = train_df[STAGE1_FEATURES].copy()
    
    # Check for nulls (should be impossible after Phase 4, but defensive)
    null_counts = X_train.isnull().sum()
    if null_counts.sum() > 0:
        raise ValueError(
            f"NULL values detected in Stage 1 features:\n{null_counts[null_counts > 0]}"
        )
    
    # Fit scaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"STAGE 1 PREPROCESSOR FITTED")
        print(f"{'='*70}")
        print(f"Features used:       {len(STAGE1_FEATURES)}")
        print(f"Training samples:    {len(X_train):,}")
        print(f"")
        print(f"Feature Statistics (PRE-SCALING):")
        print(f"{X_train.describe().T[['mean', 'std', 'min', 'max']]}")
        print(f"{'='*70}\n")
    
    return scaler


def transform_stage1(
    df: pd.DataFrame,
    scaler: StandardScaler,
    verbose: bool = False
) -> np.ndarray:
    """
    Transform data using FITTED scaler.
    
    Critical: This applies the TRAINING distribution's mean/std to
    any dataset (train or test), preventing test distribution leakage.
    
    Args:
        df: DataFrame with Stage 1 features
        scaler: Fitted StandardScaler from fit_stage1_preprocessor()
        verbose: Print diagnostics
    
    Returns:
        Scaled feature matrix (np.ndarray)
    
    Example:
        >>> X_train_scaled = transform_stage1(train_df, scaler)
        >>> X_test_scaled = transform_stage1(test_df, scaler)  # Same scaler!
    """
    # Validate features
    _validate_feature_contract(df)
    
    # Extract and scale
    X = df[STAGE1_FEATURES].copy()
    X_scaled = scaler.transform(X)
    
    if verbose:
        print(f"Transformed {len(X):,} samples using fitted scaler")
    
    return X_scaled


def _validate_feature_contract(df: pd.DataFrame) -> None:
    """
    Enforce Stage 1 feature contract.
    
    Raises:
        ValueError: If contract violated
    """
    # Check required features exist
    missing_features = set(STAGE1_FEATURES) - set(df.columns)
    if missing_features:
        raise ValueError(
            f"Missing required Stage 1 features: {missing_features}\n"
            f"Expected: {STAGE1_FEATURES}"
        )
    
    # Check forbidden features NOT present
    forbidden_present = set(FORBIDDEN_FEATURES) & set(df.columns)
    if forbidden_present:
        warnings.warn(
            f"⚠️  FORBIDDEN features detected in DataFrame: {forbidden_present}\n"
            f"Stage 1 will ignore these, but their presence indicates a design error.",
            UserWarning
        )


# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_isolation_forest(
    X_train_scaled: np.ndarray,
    contamination: float,
    random_state: int = 42,
    n_estimators: int = 100,
    verbose: bool = True
) -> IsolationForest:
    """
    Train Isolation Forest on scaled training data.
    
    Design:
    - contamination = actual fraud rate (~3.6% for your data)
    - n_estimators = 100 (sklearn default, good balance)
    - max_samples = "auto" (256 samples per tree, efficient)
    
    Args:
        X_train_scaled: Scaled feature matrix from transform_stage1()
        contamination: Expected fraud rate (e.g., 0.036 for 3.6%)
        random_state: Reproducibility seed
        n_estimators: Number of isolation trees
        verbose: Print training info
    
    Returns:
        Trained IsolationForest model
    
    Example:
        >>> model = train_isolation_forest(X_train_scaled, contamination=0.036)
    """
    if not 0 < contamination < 0.5:
        warnings.warn(
            f"Unusual contamination rate: {contamination:.1%}. "
            f"Typical fraud rates are 1-10%.",
            UserWarning
        )
    
    # Initialize model
    model = IsolationForest(
        contamination=contamination,
        n_estimators=n_estimators,
        max_samples='auto',  # 256 samples per tree
        random_state=random_state,
        n_jobs=-1,           # Use all CPU cores
        verbose=0
    )
    
    # Train
    if verbose:
        print(f"\n{'='*70}")
        print(f"TRAINING ISOLATION FOREST (STAGE 1)")
        print(f"{'='*70}")
        print(f"Samples:             {len(X_train_scaled):,}")
        print(f"Features:            {X_train_scaled.shape[1]}")
        print(f"Contamination:       {contamination:.2%} (expected fraud rate)")
        print(f"n_estimators:        {n_estimators}")
        print(f"max_samples:         auto (256 per tree)")
        print(f"{'='*70}")
        print(f"Training...")
    
    model.fit(X_train_scaled)
    
    if verbose:
        print(f"✅ Training complete")
        print(f"{'='*70}\n")
    
    return model


# ============================================================================
# PREDICTION (ANOMALY SCORING)
# ============================================================================

def predict_anomaly_scores(
    model: IsolationForest,
    X_scaled: np.ndarray,
    verbose: bool = True
) -> pd.Series:
    """
    Generate anomaly scores in [0, 1] range.
    
    Isolation Forest outputs:
    - decision_function(): raw scores (negative = anomaly)
    - score_samples(): normalized scores
    
    We convert to [0, 1] where:
    - 0 = most normal
    - 1 = most anomalous
    
    Args:
        model: Trained IsolationForest
        X_scaled: Scaled feature matrix
        verbose: Print score distribution
    
    Returns:
        pd.Series of anomaly scores in [0, 1]
    
    Example:
        >>> anomaly_scores = predict_anomaly_scores(model, X_test_scaled)
        >>> # Higher scores = more anomalous
    """
    # Get raw scores (negative = anomaly)
    raw_scores = model.decision_function(X_scaled)
    
    # Normalize to [0, 1]
    # Min-max scaling: (x - min) / (max - min)
    min_score = raw_scores.min()
    max_score = raw_scores.max()
    
    if max_score == min_score:
        warnings.warn("All anomaly scores identical - model may not have learned")
        normalized_scores = np.zeros(len(raw_scores))
    else:
        normalized_scores = (raw_scores - min_score) / (max_score - min_score)
    
    # Invert so high = anomalous (raw scores: low = anomalous)
    anomaly_scores = 1 - normalized_scores
    
    # Convert to Series
    anomaly_scores = pd.Series(anomaly_scores, name='anomaly_score')
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"ANOMALY SCORE DISTRIBUTION")
        print(f"{'='*70}")
        print(f"Samples scored:      {len(anomaly_scores):,}")
        print(f"")
        print(f"Score Statistics:")
        print(f"  Mean:              {anomaly_scores.mean():.4f}")
        print(f"  Std:               {anomaly_scores.std():.4f}")
        print(f"  Min:               {anomaly_scores.min():.4f}")
        print(f"  25th percentile:   {anomaly_scores.quantile(0.25):.4f}")
        print(f"  50th percentile:   {anomaly_scores.quantile(0.50):.4f}")
        print(f"  75th percentile:   {anomaly_scores.quantile(0.75):.4f}")
        print(f"  95th percentile:   {anomaly_scores.quantile(0.95):.4f}")
        print(f"  99th percentile:   {anomaly_scores.quantile(0.99):.4f}")
        print(f"  Max:               {anomaly_scores.max():.4f}")
        print(f"{'='*70}\n")
    
    return anomaly_scores


# ============================================================================
# PIPELINE WRAPPER
# ============================================================================

def train_stage1_pipeline(
    train_df: pd.DataFrame,
    contamination: float,
    random_state: int = 42,
    verbose: bool = True
) -> Tuple[IsolationForest, StandardScaler]:
    """
    Complete Stage 1 training pipeline.
    
    Steps:
    1. Fit preprocessor (StandardScaler) on training data
    2. Transform training data
    3. Train Isolation Forest
    
    Args:
        train_df: Training DataFrame with Stage 1 features
        contamination: Expected fraud rate
        random_state: Reproducibility seed
        verbose: Print progress
    
    Returns:
        (trained_model, fitted_scaler)
    
    Example:
        >>> model, scaler = train_stage1_pipeline(train_df, contamination=0.036)
        >>> # Save for deployment
        >>> pickle.dump(model, open('stage1_model.pkl', 'wb'))
        >>> pickle.dump(scaler, open('stage1_scaler.pkl', 'wb'))
    """
    # Step 1: Fit scaler
    scaler = fit_stage1_preprocessor(train_df, verbose=verbose)
    
    # Step 2: Transform training data
    X_train_scaled = transform_stage1(train_df, scaler, verbose=verbose)
    
    # Step 3: Train model
    model = train_isolation_forest(
        X_train_scaled,
        contamination=contamination,
        random_state=random_state,
        verbose=verbose
    )
    
    return model, scaler


def score_stage1_pipeline(
    df: pd.DataFrame,
    model: IsolationForest,
    scaler: StandardScaler,
    verbose: bool = True
) -> pd.Series:
    """
    Score data using trained Stage 1 pipeline.
    
    Steps:
    1. Transform data using fitted scaler
    2. Predict anomaly scores
    
    Args:
        df: DataFrame with Stage 1 features (train or test)
        model: Trained IsolationForest
        scaler: Fitted StandardScaler
        verbose: Print diagnostics
    
    Returns:
        pd.Series of anomaly scores [0, 1]
    
    Example:
        >>> test_anomaly_scores = score_stage1_pipeline(test_df, model, scaler)
    """
    # Transform
    X_scaled = transform_stage1(df, scaler, verbose=verbose)
    
    # Score
    anomaly_scores = predict_anomaly_scores(model, X_scaled, verbose=verbose)
    
    return anomaly_scores

# ============================================================================
# MODEL PERSISTENCE
# ============================================================================

def save_stage1_artifacts(
    model: IsolationForest,
    scaler: StandardScaler,
    model_path: str = "models/stage1_model.pkl",
    scaler_path: str = "models/stage1_scaler.pkl"
) -> None:
    """
    Save trained Stage 1 artifacts for deployment.
    
    Args:
        model: Trained IsolationForest
        scaler: Fitted StandardScaler
        model_path: Path to save model
        scaler_path: Path to save scaler
    
    Example:
        >>> save_stage1_artifacts(model, scaler)
        >>> # Later: model, scaler = load_stage1_artifacts()
    """
    import os
    os.makedirs(os.path.dirname(model_path), exist_ok=True)
    
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"✅ Saved Stage 1 model to {model_path}")
    print(f"✅ Saved Stage 1 scaler to {scaler_path}")


def load_stage1_artifacts(
    model_path: str = "models/stage1_model.pkl",
    scaler_path: str = "models/stage1_scaler.pkl"
) -> Tuple[IsolationForest, StandardScaler]:
    """
    Load trained Stage 1 artifacts.
    
    Returns:
        (model, scaler)
    
    Example:
        >>> model, scaler = load_stage1_artifacts()
        >>> scores = score_stage1_pipeline(new_data, model, scaler)
    """
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    print(f"✅ Loaded Stage 1 model from {model_path}")
    print(f"✅ Loaded Stage 1 scaler from {scaler_path}")
    
    return model, scaler

# ============================================================================
# FEATURE ANALYSIS (OPTIONAL)
# ============================================================================

def analyze_feature_importance(
    model: IsolationForest,
    X_scaled: np.ndarray,
    feature_names: list = None
) -> pd.DataFrame:
    """
    Approximate feature importance by permutation.
    
    Note: This is computationally expensive. Use on sample data.
    
    Args:
        model: Trained IsolationForest
        X_scaled: Scaled feature matrix (sample, not full dataset)
        feature_names: List of feature names (default: STAGE1_FEATURES)
    
    Returns:
        DataFrame with feature importance scores
    """
    if feature_names is None:
        feature_names = STAGE1_FEATURES
    
    # Baseline scores
    baseline_scores = model.decision_function(X_scaled)
    
    importances = []
    
    for i, feature in enumerate(feature_names):
        # Permute feature i
        X_permuted = X_scaled.copy()
        np.random.shuffle(X_permuted[:, i])
        
        # Re-score
        permuted_scores = model.decision_function(X_permuted)
        
        # Importance = change in anomaly detection
        importance = np.abs(baseline_scores - permuted_scores).mean()
        importances.append(importance)
    
    # Create DataFrame
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return importance_df

