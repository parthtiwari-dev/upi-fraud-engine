"""
Single Transaction Fraud Prediction for Real-Time API

This bridges the gap between:
- Raw transaction input (Transaction schema)
- Feature engineering (online_builder.py)
- Model prediction (production model)

Usage:
    predictor = FraudPredictor("models/production/fraud_detector.json")
    result = predictor.predict_single(raw_transaction_dict)
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from typing import Dict, Optional
from datetime import datetime
import re
import time

from src.ingestion.schema import Transaction  # For validation
from src.features.online_builder import OnlineFeatureStore
from src.models.stage2_supervised import load_stage2_artifacts


def extract_payer_id_from_vpa(payer_vpa: str) -> str:
    """
    Extract payer_id from payer_vpa.
    
    Phase 1 mapping: payer_vpa = f"user_{payer_id}@upi"
    Reverse: payer_id = payer_vpa.replace("user_", "").replace("@upi", "")
    
    Args:
        payer_vpa: Format "user_{payer_id}@upi"
    
    Returns:
        payer_id: Extracted ID
    """
    # Handle both formats: "user_abc123@upi" or just "abc123@upi"
    if payer_vpa.startswith("user_"):
        payer_id = payer_vpa.replace("user_", "").replace("@upi", "")
    else:
        # Fallback: extract before @
        payer_id = payer_vpa.split("@")[0]
    
    return payer_id


class FraudPredictor:
    """
    Single-transaction fraud prediction service.
    
    This is the CORE inference logic that will be used by:
    - Phase 6: Backtesting (day-by-day replay)
    - Phase 8: FastAPI (HTTP endpoints)
    - UI: Direct Python calls
    """
    
    def __init__(self, model_path: str):
        """
        Load model artifacts and initialize feature store.
        
        Args:
            model_path: Path to saved XGBoost model JSON
        """
        # Load model, feature names, encoders, metadata
        self.model, self.feature_names, self.encoders, self.metadata = load_stage2_artifacts(model_path)
        
        # Initialize online feature store (maintains state)
        self.feature_store = OnlineFeatureStore()
        
        print(f"✅ FraudPredictor initialized")
        print(f"   Model: {self.metadata.get('model_type', 'XGBoost')}")
        print(f"   Features: {len(self.feature_names)}")
        if 'roc_auc' in self.metadata:
            print(f"   ROC-AUC: {self.metadata['roc_auc']:.4f}")
    
    def predict_single(self, transaction: Dict) -> Dict:
        """
        Predict fraud probability for a single raw transaction.
        
        Complete pipeline:
        1. Validate transaction schema
        2. Extract payer_id from payer_vpa
        3. Compute engineered features (online_builder)
        4. Merge with raw Vesta features
        5. Apply model preprocessing (encoders, feature selection)
        6. Predict fraud probability
        7. Update feature store state
        
        Args:
            transaction: Raw transaction dict with:
                - Required: transaction_id, event_timestamp, amount, payer_vpa, payee_vpa, device_id
                - Optional: All Vesta features (V1, V2, ..., V339, C1, C2, ..., C14, etc.)
                - Optional: payer_id (if not provided, extracted from payer_vpa)
        
        Returns:
            {
                'transaction_id': str,
                'fraud_probability': float,  # [0, 1]
                'should_alert': bool,        # True if prob > threshold
                'risk_tier': str,             # 'low', 'medium', 'high', 'critical'
                'threshold_used': float,
                'latency_ms': float
            }
        """
        start_time = time.time()
        
        try:
            # Step 1: Validate transaction schema
            txn_validated = Transaction(**transaction)
            txn_dict = txn_validated.model_dump()
            
            # Step 2: Extract payer_id if not provided
            if 'payer_id' not in txn_dict:
                txn_dict['payer_id'] = extract_payer_id_from_vpa(txn_dict['payer_vpa'])
            
            # Step 3: Compute engineered features via online_builder
            # This returns FeatureVector (11 features)
            feature_vector = self.feature_store.get_features(txn_dict)
            engineered_features = feature_vector.model_dump()
            
            # Step 4: Combine engineered + raw Vesta features
            # txn_dict already has all Vesta features (V1, V2, ..., C1, C2, etc.)
            # Merge them together
            all_features = {**txn_dict, **engineered_features}
            
            # Step 5: Convert to DataFrame (single row)
            feature_df = pd.DataFrame([all_features])
            
            # Step 6: Prepare features exactly as training (apply encoders, select features)
            X = self._prepare_features_for_model(feature_df)
            
            # Validate shape matches model expectations
            if X.shape[1] != len(self.feature_names):
                raise ValueError(
                    f"Feature count mismatch: got {X.shape[1]} columns, "
                    f"expected {len(self.feature_names)}"
                )
            
            # Step 7: Predict
            # CRITICAL: Build DMatrix without feature name validation
            # The model has 482 features - we provide them in exact order
            dtest = xgb.DMatrix(X.values)
            
            # Disable feature validation during predict
            with np.errstate(all='ignore'):
                fraud_prob = float(self.model.predict(dtest, validate_features=False)[0])
            
            # Step 8: Ingest transaction into feature store (for next prediction)
            # This updates the history for future feature computation
            self.feature_store.ingest(txn_dict)
            
            # Step 9: Determine alert and risk tier
            threshold = self.metadata.get('threshold_0.5pct', 0.994)  # From training
            should_alert = fraud_prob >= threshold
            risk_tier = self._get_risk_tier(fraud_prob)
            
            latency_ms = (time.time() - start_time) * 1000
            
            return {
                'transaction_id': txn_dict.get('transaction_id', 'unknown'),
                'fraud_probability': float(fraud_prob),
                'should_alert': should_alert,
                'risk_tier': risk_tier,
                'threshold_used': float(threshold),
                'latency_ms': latency_ms
            }
            
        except Exception as e:
            # Log error and re-raise with context
            raise ValueError(
                f"Prediction failed for transaction {transaction.get('transaction_id', 'unknown')}: {str(e)}"
            ) from e
    
    
    def _prepare_features_for_model(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply same preprocessing as training:
        1. Select features in correct order (as saved in feature_names)
        2. Apply label encoders to categoricals
        3. Handle missing values
        
        This MUST match exactly what prepare_stage2_features() does in training.
        """
        # CRITICAL: Build feature matrix in EXACT order of self.feature_names
        # This determines the column order, which MUST match DMatrix feature_names order
        X_list = []
        
        for feat_name in self.feature_names:
            if feat_name in df.columns:
                col_data = df[feat_name].values
            else:
                # Feature missing - fill with NaN (XGBoost will handle)
                col_data = np.full(len(df), np.nan, dtype=np.float64)
            X_list.append(col_data)
        
        # Create numpy array (rows x features) - shape must be (n_rows, n_features)
        X_array = np.column_stack(X_list)
        
        # Convert to DataFrame ONLY for encoding operations
        X = pd.DataFrame(X_array, columns=self.feature_names, index=df.index)
        
        # Apply label encoders (same as training)
        # Training uses '__MISSING__' (double underscores)!
        for col, encoder in self.encoders.items():
            if col in X.columns:
                # Handle missing/unseen categories
                # CRITICAL: Use '__MISSING__' to match training!
                X[col] = X[col].fillna('__MISSING__')
                
                # Transform: if value is in encoder classes, use it
                # Otherwise, use '__MISSING__' (which should be in encoder classes)
                def safe_encode(x):
                    if isinstance(x, float) and np.isnan(x):
                        x = '__MISSING__'
                    if isinstance(x, str) and x == '__MISSING__':
                        x = '__MISSING__'
                    else:
                        x = str(x)
                    
                    if x in encoder.classes_:
                        return encoder.transform([x])[0]
                    elif '__MISSING__' in encoder.classes_:
                        return encoder.transform(['__MISSING__'])[0]
                    else:
                        # Fallback: use first class
                        return encoder.transform([encoder.classes_[0]])[0]
                
                X[col] = X[col].apply(safe_encode)
        
        # Ensure numeric types (XGBoost requirement)
        X = X.astype(np.float64)
        
        return X
    
    def _get_risk_tier(self, fraud_prob: float) -> str:
        """Map fraud probability to risk tier."""
        if fraud_prob >= 0.9:
            return 'critical'
        elif fraud_prob >= 0.7:
            return 'high'
        elif fraud_prob >= 0.3:
            return 'medium'
        else:
            return 'low'


# Convenience function for quick testing
def predict_fraud(
    transaction: Dict, 
    model_path: str = "models/production/fraud_detector.json"
) -> Dict:
    """
    One-liner for single transaction prediction.
    
    Args:
        transaction: Raw transaction dict
        model_path: Path to model
    
    Returns:
        Prediction result dict
    """
    predictor = FraudPredictor(model_path)
    return predictor.predict_single(transaction)


if __name__ == "__main__":
    # Example usage
    sample_transaction = {
        'transaction_id': 'TXN_TEST_001',
        'event_timestamp': datetime.now(),
        'amount': 5000.0,
        'payer_vpa': 'user_abc123@upi',  # Will extract payer_id = 'abc123'
        'payee_vpa': 'merchant_xyz@upi',
        'device_id': 'device_xyz',
        # Add all Vesta features (V1-V339, C1-C14, etc.)
        'V1': 1.23,
        'V2': 0.45,
        # ... (all other Vesta features)
        # If missing, they'll be filled with NaN (XGBoost handles nulls)
    }
    
    result = predict_fraud(sample_transaction)
    print(f"\nPrediction Result:")
    print(f"  Transaction ID: {result['transaction_id']}")
    print(f"  Fraud Probability: {result['fraud_probability']:.4f}")
    print(f"  Should Alert: {result['should_alert']}")
    print(f"  Risk Tier: {result['risk_tier']}")
    print(f"  Latency: {result['latency_ms']:.2f}ms")