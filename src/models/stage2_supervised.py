"""
Stage 2: Supervised Fraud Classification (XGBoost)

Purpose:
Train XGBoost classifier using ALL Phase 4 features + Stage 1 anomaly scores
to predict fraud with high precision under alert budget constraints.

Design Contract:
- Input: Full feature set (490 Phase 4 + 1 anomaly_score = 491 total)
- Uses: Denylist-based feature selection (exclude identifiers/metadata)
- Imbalance: scale_pos_weight (NOT sample_weight, NOT SMOTE)
- Metric: AUC-PR (precision-recall, imbalance-aware)
- Output: Fraud probability [0, 1] for each transaction

This is the refinement layer. Stage 1 is the discovery layer.
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Tuple, List, Dict, Optional
import warnings
import pickle
import json


# ============================================================================
# FEATURE SELECTION CONTRACT (DENYLIST APPROACH)
# ============================================================================
# We exclude metadata/identifiers, use EVERYTHING else automatically.
# This prevents silent bugs if new features are added in Phase 4.

STAGE2_EXCLUDED_COLUMNS = [
    'transaction_id',           # Identifier (not predictive)
    'event_timestamp',          # Time metadata (not predictive)
    'label_available_timestamp', # Label metadata (not predictive)
    'is_fraud',                 # TARGET VARIABLE (must exclude!)
    'fraud_pattern' 
]

# Optional: Additional columns to exclude if present
OPTIONAL_EXCLUSIONS = [
    'payer_id',      # High cardinality identifier
    'payee_vpa',     # High cardinality identifier  
    'device_id',     # High cardinality identifier
    'currency',      # Constant (always INR in your data)
]


# ============================================================================
# FEATURE PREPARATION
# ============================================================================

def prepare_stage2_features(
    df: pd.DataFrame,
    exclude_optional: bool = True,
    allow_nulls: bool = True,  # NEW PARAMETER
    verbose: bool = True
) -> Tuple[pd.DataFrame, List[str], pd.Series, Dict]:
    """
    Prepare features for XGBoost training using denylist approach.
    
    Design Philosophy:
    - Exclude metadata/identifiers (denylist)
    - Include EVERYTHING else automatically (prevents schema drift bugs)
    - Allow nulls in raw Vesta features (XGBoost handles them natively)
    - Validate that anomaly_score exists (Stage 1 must have run)
    
    Args:
        df: Full feature DataFrame (should have 491 columns after Stage 1)
        exclude_optional: Also exclude high-cardinality IDs (recommended)
        allow_nulls: Allow null values (XGBoost handles natively)
        verbose: Print feature selection details
    
    Returns:
        (X, feature_names, y)
        - X: Feature matrix (DataFrame)
        - feature_names: List of column names used
        - y: Target variable (Series)
    
    Raises:
        ValueError: If is_fraud missing, or anomaly_score missing, or no features remain
    
    Example:
        >>> X, features, y = prepare_stage2_features(train_df)
        >>> print(f"Training on {len(features)} features")
    """
    # Validate required columns
    if 'is_fraud' not in df.columns:
        raise ValueError("Target variable 'is_fraud' not found in DataFrame")
    
    if 'anomaly_score' not in df.columns:
        warnings.warn(
            "⚠️  'anomaly_score' not found! Stage 1 must run before Stage 2.\n"
            "If running Stage 2 standalone, ensure anomaly_score is added.",
            UserWarning
        )
    
    # Build exclusion list
    exclude_cols = STAGE2_EXCLUDED_COLUMNS.copy()
    if exclude_optional:
        exclude_cols.extend(OPTIONAL_EXCLUSIONS)
    
    # Extract target
    y = df['is_fraud'].copy()
    
    # Filter labeled data only (is_fraud = 0.0 or 1.0)
    labeled_mask = y.notna()
    if not labeled_mask.all():
        n_unlabeled = (~labeled_mask).sum()
        warnings.warn(
            f"Filtering out {n_unlabeled:,} unlabeled transactions. "
            f"Stage 2 only trains on labeled data.",
            UserWarning
        )
        df = df[labeled_mask].copy()
        y = y[labeled_mask].copy()
    
    # Select features (denylist approach)
    available_cols = set(df.columns)
    excluded_present = set(exclude_cols) & available_cols
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    if len(feature_cols) == 0:
        raise ValueError("No features remaining after exclusion! Check denylist.")
    
    X = df[feature_cols].copy()
    # ========================================================================
    # CATEGORICAL FEATURE HANDLING (PRODUCTION-GRADE)
    # ========================================================================
    # XGBoost 1.7+ supports native categorical features with enable_categorical=True
    # BUT requires 'category' dtype, not 'object'. We'll use Label Encoding instead.
    
    from sklearn.preprocessing import LabelEncoder
    
    # Identify column types
    numeric_cols = X.select_dtypes(include=['int64', 'float64', 'bool']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    # Store encoders for later use (needed for deployment)
    label_encoders = {}
    
    if len(categorical_cols) > 0:
        if verbose:
            print(f"\n📊 ENCODING {len(categorical_cols)} CATEGORICAL FEATURES:")
            print(f"   Method: Label Encoding (string → integer)")
            print(f"   Categorical columns: {categorical_cols[:10]}...")  # Show first 10
        
        for col in categorical_cols:
            le = LabelEncoder()
            
            # Handle nulls: LabelEncoder doesn't support nulls
            # Strategy: Fill nulls with special value '__MISSING__' before encoding
            X[col] = X[col].fillna('__MISSING__')
            
            # Fit and transform
            X[col] = le.fit_transform(X[col])
            
            # Store encoder for deployment (to encode test data the same way)
            label_encoders[col] = le
        
        if verbose:
            print(f"   ✅ Encoded {len(categorical_cols)} features")
            print(f"   All features now numeric (int/float)\n")
    
    # Update feature list (keep same names, but now they're all numeric)
    feature_cols = numeric_cols + categorical_cols
    
    # Check for nulls (informational only if allow_nulls=True)
    null_counts = X.isnull().sum()
    if null_counts.sum() > 0:
        null_features = null_counts[null_counts > 0]
        
        if allow_nulls:
            # Just report, don't error
            if verbose:
                print(f"\n⚠️  NULL VALUES DETECTED (XGBoost will handle natively):")
                print(f"   Features with nulls: {len(null_features)}")
                print(f"   Total null count: {null_counts.sum():,}")
                print(f"   Top 5 null-heavy features:")
                print(f"{null_features.nlargest(5)}")
                print(f"   → XGBoost treats missing values as informative signal")
        else:
            # Strict mode - raise error
            raise ValueError(
                f"NULL values detected in {len(null_features)} features:\n"
                f"{null_features.head(10)}\n"
                f"Set allow_nulls=True to let XGBoost handle them."
            )
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"STAGE 2 FEATURE PREPARATION")
        print(f"{'='*70}")
        print(f"Input rows:          {len(df):,}")
        print(f"Labeled rows:        {len(X):,}")
        print(f"Input columns:       {len(df.columns)}")
        print(f"Excluded columns:    {len(excluded_present)}")
        print(f"  - {sorted(excluded_present)}")
        print(f"Final features:      {len(feature_cols)}")
        print(f"")
        print(f"Target distribution:")
        print(f"  Legitimate (0.0):  {(y == 0.0).sum():,} ({(y == 0.0).sum()/len(y)*100:.2f}%)")
        print(f"  Fraud (1.0):       {(y == 1.0).sum():,} ({(y == 1.0).sum()/len(y)*100:.2f}%)")
        print(f"")
        
        # Check if Stage 1 anomaly_score is present
        if 'anomaly_score' in feature_cols:
            print(f"✅ Stage 1 anomaly_score detected (col #{feature_cols.index('anomaly_score')+1})")
        else:
            print(f"⚠️  Stage 1 anomaly_score NOT FOUND (running Stage 2 standalone)")
        
        print(f"{'='*70}\n")
    
    return X, feature_cols, y, label_encoders 



# ============================================================================
# MODEL TRAINING
# ============================================================================

def train_xgboost(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: Optional[pd.DataFrame] = None,
    y_val: Optional[pd.Series] = None,
    params: Optional[Dict] = None,
    num_boost_round: int = 300,
    early_stopping_rounds: int = 50,
    verbose: bool = True
) -> xgb.Booster:
    """
    Train XGBoost classifier with fraud-optimized hyperparameters.
    
    Key Design Decisions:
    - scale_pos_weight: Handles 3.6% class imbalance (auto-calculated)
    - eval_metric: 'aucpr' (PR-AUC, better for imbalanced data than ROC-AUC)
    - max_depth: 6 (prevents overfitting on 500K samples)
    - learning_rate: 0.1 (standard)
    - early_stopping: Prevents overtraining
    
    Args:
        X_train: Training features
        y_train: Training labels (0.0 or 1.0)
        X_val: Validation features (optional, enables early stopping)
        y_val: Validation labels (optional)
        params: Custom hyperparameters (overrides defaults)
        num_boost_round: Max boosting iterations
        early_stopping_rounds: Stop if no improvement (requires validation set)
        verbose: Print training progress
    
    Returns:
        Trained xgb.Booster model
    
    Example:
        >>> model = train_xgboost(X_train, y_train, X_val, y_val)
        >>> # Later: predictions = model.predict(xgb.DMatrix(X_test))
    """
    # Calculate class imbalance weight
    n_legitimate = (y_train == 0.0).sum()
    n_fraud = (y_train == 1.0).sum()
    scale_pos_weight = n_legitimate / n_fraud
    
    # Default hyperparameters (fraud-optimized)
    default_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'aucpr',           # PR-AUC (better for imbalance)
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,                 # Prevent overfitting
        'colsample_bytree': 0.8,          # Use 80% features per tree
        'scale_pos_weight': scale_pos_weight,  # Handle imbalance
        'min_child_weight': 1,
        'gamma': 0,
        'reg_alpha': 0,                   # L1 regularization
        'reg_lambda': 1,                  # L2 regularization
        'seed': 42,
        'nthread': -1,                    # Use all CPU cores
    }
    
    # Override with custom params if provided
    if params is not None:
        default_params.update(params)
        # Warn if user tried to set scale_pos_weight manually
        if 'scale_pos_weight' in params:
            warnings.warn(
                f"scale_pos_weight manually set to {params['scale_pos_weight']}. "
                f"Auto-calculated value was {scale_pos_weight:.2f}",
                UserWarning
            )
    
    # Create DMatrix (XGBoost's internal data structure)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    
    # Setup evaluation
    evals = [(dtrain, 'train')]
    if X_val is not None and y_val is not None:
        dval = xgb.DMatrix(X_val, label=y_val)
        evals.append((dval, 'val'))
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"TRAINING XGBOOST (STAGE 2)")
        print(f"{'='*70}")
        print(f"Training samples:    {len(X_train):,}")
        print(f"Features:            {X_train.shape[1]}")
        print(f"Fraud rate:          {n_fraud/len(y_train)*100:.2f}% ({n_fraud:,} frauds)")
        print(f"scale_pos_weight:    {scale_pos_weight:.2f}")
        print(f"")
        print(f"Hyperparameters:")
        for k, v in sorted(default_params.items()):
            print(f"  {k:20s} = {v}")
        print(f"")
        print(f"Training with early stopping (patience={early_stopping_rounds})...")
        print(f"{'='*70}\n")
    
    # Train
    model = xgb.train(
        params=default_params,
        dtrain=dtrain,
        num_boost_round=num_boost_round,
        evals=evals,
        early_stopping_rounds=early_stopping_rounds if X_val is not None else None,
        verbose_eval=20 if verbose else False  # Print every 20 iterations
    )
    
    if verbose:
        best_iteration = model.best_iteration if X_val is not None else num_boost_round
        best_score = model.best_score if X_val is not None else "N/A"
        print(f"\n{'='*70}")
        print(f"✅ Training complete")
        print(f"Best iteration:      {best_iteration}")
        if X_val is not None:
            print(f"Best val PR-AUC:     {best_score:.4f}")
        print(f"{'='*70}\n")
    
    return model


# ============================================================================
# PREDICTION
# ============================================================================

def predict_fraud_probabilities(
    model: xgb.Booster,
    X: pd.DataFrame,
    verbose: bool = True
) -> pd.Series:
    """
    Generate fraud probabilities for transactions.
    
    Args:
        model: Trained XGBoost model
        X: Feature matrix (must have same columns as training)
        verbose: Print prediction statistics
    
    Returns:
        pd.Series of fraud probabilities [0, 1]
    
    Example:
        >>> fraud_probs = predict_fraud_probabilities(model, X_test)
        >>> # fraud_probs[i] = 0.85 means 85% probability transaction i is fraud
    """
    dtest = xgb.DMatrix(X)
    predictions = model.predict(dtest)
    
    # Convert to Series
    fraud_probs = pd.Series(predictions, index=X.index, name='fraud_probability')
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"FRAUD PROBABILITY PREDICTIONS")
        print(f"{'='*70}")
        print(f"Samples predicted:   {len(fraud_probs):,}")
        print(f"")
        print(f"Probability Distribution:")
        print(f"  Mean:              {fraud_probs.mean():.4f}")
        print(f"  Std:               {fraud_probs.std():.4f}")
        print(f"  Min:               {fraud_probs.min():.4f}")
        print(f"  25th percentile:   {fraud_probs.quantile(0.25):.4f}")
        print(f"  50th percentile:   {fraud_probs.quantile(0.50):.4f}")
        print(f"  75th percentile:   {fraud_probs.quantile(0.75):.4f}")
        print(f"  95th percentile:   {fraud_probs.quantile(0.95):.4f}")
        print(f"  99th percentile:   {fraud_probs.quantile(0.99):.4f}")
        print(f"  Max:               {fraud_probs.max():.4f}")
        print(f"{'='*70}\n")
    
    return fraud_probs


# ============================================================================
# FEATURE IMPORTANCE
# ============================================================================

def get_feature_importance(
    model: xgb.Booster,
    feature_names: List[str],
    importance_type: str = 'gain',
    top_n: int = 20,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Extract and rank feature importance from trained XGBoost model.
    
    Importance Types:
    - 'gain': Average gain when feature is used (RECOMMENDED)
    - 'weight': Number of times feature is used for splitting
    - 'cover': Average coverage of feature when used
    
    IMPORTANT: Handles both f0/f1/f2 format AND direct feature name format.
    Automatically falls back to 'weight' if 'gain' returns all zeros.
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature names (from prepare_stage2_features)
        importance_type: Type of importance metric
        top_n: Return top N features (default 20)
        verbose: Print top features
    
    Returns:
        DataFrame with columns ['feature', 'importance', 'rank']
        sorted by importance (descending)
    """
    # Get importance scores
    importance_dict = model.get_score(importance_type=importance_type)
    
    # Check if all scores are zero (can happen with early stopping)
    total_importance = sum(importance_dict.values())
    
    if total_importance == 0 and importance_type == 'gain':
        # Fallback to 'weight' (more reliable)
        if verbose:
            print(f"⚠️  'gain' importance returned all zeros. Falling back to 'weight'.\n")
        importance_dict = model.get_score(importance_type='weight')
        importance_type = 'weight'  # Update for display
        total_importance = sum(importance_dict.values())
    
    if total_importance == 0:
        warnings.warn(
            "All feature importance scores are zero! Model may not have learned anything.",
            UserWarning
        )
    
    # Detect key format: XGBoost can return either 'f0' or feature names directly
    sample_key = list(importance_dict.keys())[0] if importance_dict else 'f0'
    uses_f_notation = sample_key.startswith('f') and sample_key[1:].isdigit()
    
    # Map to feature names
    importance_scores = []
    
    if uses_f_notation:
        # Keys are 'f0', 'f1', 'f2' → map to feature names
        for i, fname in enumerate(feature_names):
            xgb_fname = f'f{i}'
            score = importance_dict.get(xgb_fname, 0.0)
            importance_scores.append({
                'feature': fname,
                'importance': score,
            })
    else:
        # Keys are already feature names → use directly
        for fname in feature_names:
            score = importance_dict.get(fname, 0.0)
            importance_scores.append({
                'feature': fname,
                'importance': score,
            })
    
    # Sort by importance
    importance_df = pd.DataFrame(importance_scores)
    importance_df = importance_df.sort_values('importance', ascending=False).reset_index(drop=True)
    importance_df['rank'] = importance_df.index + 1
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"FEATURE IMPORTANCE (Top {top_n}, type={importance_type})")
        print(f"{'='*70}")
        
        top_features = importance_df.head(top_n)
        
        # Check if anomaly_score is in top features
        if 'anomaly_score' in top_features['feature'].values:
            anomaly_rank = importance_df[importance_df['feature'] == 'anomaly_score']['rank'].values[0]
            anomaly_importance = importance_df[importance_df['feature'] == 'anomaly_score']['importance'].values[0]
            print(f"✅ STAGE 1 ANOMALY_SCORE DETECTED: Rank #{anomaly_rank} (importance: {anomaly_importance:.2f})")
            print(f"   → Stage 1 contributed useful signal to Stage 2!")
        else:
            if 'anomaly_score' in importance_df['feature'].values:
                anomaly_rank = importance_df[importance_df['feature'] == 'anomaly_score']['rank'].values[0]
                anomaly_importance = importance_df[importance_df['feature'] == 'anomaly_score']['importance'].values[0]
                print(f"⚠️  STAGE 1 ANOMALY_SCORE: Rank #{anomaly_rank} (importance: {anomaly_importance:.2f})")
                print(f"   → Stage 1 had minimal impact. Consider removing.")
            else:
                print(f"ℹ️  ANOMALY_SCORE not found in features (running without Stage 1)")
        
        print(f"")
        print(top_features.to_string(index=False))
        print(f"{'='*70}\n")
    
    return importance_df



# ============================================================================
# MODEL PERSISTENCE
# ============================================================================

def save_stage2_artifacts(
    model: xgb.Booster,
    feature_names: List[str],
    label_encoders: Dict,  # ← NEW PARAMETER
    model_path: str,
    metadata_path: Optional[str] = None,
    metadata: Optional[Dict] = None
) -> None:
    """
    Save trained Stage 2 model, encoders, and metadata for deployment.
    
    Saves:
    1. XGBoost model (.json format)
    2. Feature names (.txt, one per line)
    3. Label encoders (.pkl for categorical features)
    4. Metadata (.json with training info, metrics, etc.)
    
    Args:
        model: Trained XGBoost model
        feature_names: List of feature names (order matters!)
        label_encoders: Dict of LabelEncoder objects (from prepare_stage2_features)
        model_path: Path to save model (e.g., 'models/stage2_xgboost.json')
        metadata_path: Path to save metadata (default: same dir as model)
        metadata: Dict of metadata to save (metrics, params, etc.)
    """
    import os
    
    # Save model
    model.save_model(model_path)
    print(f"✅ Saved Stage 2 model to {model_path}")
    
    # Save feature names
    feature_path = model_path.replace('.json', '_features.txt')
    with open(feature_path, 'w') as f:
        f.write('\n'.join(feature_names))
    print(f"✅ Saved feature names to {feature_path}")
    
    # Save label encoders
    encoder_path = model_path.replace('.json', '_encoders.pkl')
    with open(encoder_path, 'wb') as f:
        pickle.dump(label_encoders, f)
    print(f"✅ Saved {len(label_encoders)} label encoders to {encoder_path}")
    
    # Save metadata
    if metadata_path is None:
        metadata_path = model_path.replace('.json', '_metadata.json')
    
    if metadata is None:
        metadata = {}
    
    metadata['num_features'] = len(feature_names)
    metadata['num_categorical'] = len(label_encoders)
    metadata['model_path'] = model_path
    metadata['feature_path'] = feature_path
    metadata['encoder_path'] = encoder_path
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Saved metadata to {metadata_path}")


def load_stage2_artifacts(model_path: str) -> Tuple[xgb.Booster, List[str], Dict, Dict]:
    """
    Load trained Stage 2 model, encoders, and metadata.
    
    Args:
        model_path: Path to saved model
    
    Returns:
        (model, feature_names, label_encoders, metadata)
    """
    # Load model
    model = xgb.Booster()
    model.load_model(model_path)
    print(f"✅ Loaded Stage 2 model from {model_path}")
    
    # Load feature names
    feature_path = model_path.replace('.json', '_features.txt')
    with open(feature_path, 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    print(f"✅ Loaded {len(feature_names)} feature names from {feature_path}")
    
    # Load label encoders
    encoder_path = model_path.replace('.json', '_encoders.pkl')
    with open(encoder_path, 'rb') as f:
        label_encoders = pickle.load(f)
    print(f"✅ Loaded {len(label_encoders)} label encoders from {encoder_path}")
    
    # Load metadata
    metadata_path = model_path.replace('.json', '_metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"✅ Loaded metadata from {metadata_path}")
    
    return model, feature_names, label_encoders, metadata


def load_stage2_artifacts(model_path: str) -> Tuple[xgb.Booster, List[str], Dict, Dict]:
    """
    Load trained Stage 2 model, encoders, and metadata.
    
    Args:
        model_path: Path to saved model
    
    Returns:
        (model, feature_names, label_encoders, metadata)
    """
    # Load model
    model = xgb.Booster()
    model.load_model(model_path)
    print(f"✅ Loaded Stage 2 model from {model_path}")
    
    # Load feature names
    feature_path = model_path.replace('.json', '_features.txt')
    with open(feature_path, 'r') as f:
        feature_names = [line.strip() for line in f.readlines()]
    print(f"✅ Loaded {len(feature_names)} feature names from {feature_path}")
    
    # Load label encoders
    encoder_path = model_path.replace('.json', '_encoders.pkl')
    with open(encoder_path, 'rb') as f:
        label_encoders = pickle.load(f)
    print(f"✅ Loaded {len(label_encoders)} label encoders from {encoder_path}")
    
    # Load metadata
    metadata_path = model_path.replace('.json', '_metadata.json')
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    print(f"✅ Loaded metadata from {metadata_path}")
    
    return model, feature_names, label_encoders, metadata  # ← 4 values

