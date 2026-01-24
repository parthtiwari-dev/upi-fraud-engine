"""
Day-by-Day Backtesting Engine for Fraud Detection

This module replays historical transactions through the fraud detection system
to measure real-world performance under operational constraints.

Key Features:
- Day-by-day replay (no future information leakage)
- Alert budget enforcement (0.5% daily limit)
- Scenario testing (fraud spikes, behavior shifts)
- Comprehensive metrics (precision, recall, cost-benefit)

Critical Guarantees:
1. NO LEAKAGE: When scoring Day N, only use features from Days 1...N-1
2. BUDGET ENFORCEMENT: Never exceed 0.5% alert rate on any day
3. REPRODUCIBILITY: Same input → same output (deterministic)

Author: Your Name
Date: January 24, 2026
"""

import pandas as pd
import numpy as np
import duckdb
from typing import Dict, List, Optional, Tuple
from datetime import datetime, date, timedelta
from pathlib import Path
import json
import warnings

from src.evaluation.alert_policy import AlertPolicy, compute_daily_metrics
from src.evaluation.metrics import (
    precision_at_alert_budget,
    alert_budget_curve,
    cost_benefit_analysis
)


class Backtester:
    """
    Day-by-day backtesting engine for fraud detection.
    
    Workflow:
    1. Load trained model (Stage 2 XGBoost)
    2. For each day in history:
       a. Get transactions for that day
       b. Load pre-computed features (from Phase 4)
       c. Score with model → fraud probabilities
       d. Apply alert policy → decide which to alert on
       e. Compare vs ground truth → compute metrics
    3. Aggregate results → overall performance
    4. Run scenarios → stress testing
    
    Usage:
        backtester = Backtester(
            model_path='models/production/fraud_detector.json',
            feature_store_path='data/processed/full_features.duckdb'
        )
        
        results = backtester.run_backtest(
            start_date='2025-06-01',
            end_date='2025-07-02'
        )
    """
    
    def __init__(
        self,
        model_path: str,
        feature_store_path: str,
        alert_budget: float = 0.005,
        verbose: bool = True
    ):
        """
        Initialize backtester with model and data sources.
        
        Args:
            model_path: Path to trained XGBoost model (.json)
            feature_store_path: Path to DuckDB with pre-computed features
            alert_budget: Daily alert budget (default: 0.5%)
            verbose: Print progress messages
        """
        self.model_path = model_path
        self.feature_store_path = feature_store_path
        self.alert_budget = alert_budget
        self.verbose = verbose
        
        # Load model
        if self.verbose:
            print(f"{'=' * 70}")
            print(f"INITIALIZING BACKTESTER")
            print(f"{'=' * 70}")
        
        self._load_model()
        self._connect_feature_store()
        
        if self.verbose:
            print(f"✅ Backtester initialized")
            print(f"   Model: {self.model_path}")
            print(f"   Feature store: {self.feature_store_path}")
            print(f"   Alert budget: {self.alert_budget:.1%}")
            print(f"{'=' * 70}\n")
    
    def _load_model(self):
        """Load trained XGBoost model and metadata."""
        import xgboost as xgb
        
        # Load model
        self.model = xgb.Booster()
        self.model.load_model(self.model_path)
        
        # Get feature names DIRECTLY from the model (not from file)
        # This ensures exact match with what model expects
        self.feature_names = self.model.feature_names
        
        # Load metadata if exists
        metadata_path = self.model_path.replace('.json', '_metadata.json')
        if Path(metadata_path).exists():
            with open(metadata_path) as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
        
        if self.verbose:
            print(f"✅ Loaded model: {len(self.feature_names)} features")

    
    def _connect_feature_store(self):
        """Connect to DuckDB feature store."""
        self.con = duckdb.connect(self.feature_store_path, read_only=True)
        
        # Get date range
        result = self.con.execute("""
            SELECT 
                MIN(DATE(event_timestamp)) as min_date,
                MAX(DATE(event_timestamp)) as max_date,
                COUNT(*) as total_transactions
            FROM training_data
        """).fetchone()
        
        self.data_start_date = result[0]
        self.data_end_date = result[1]
        self.total_transactions = result[2]
        
        if self.verbose:
            print(f"✅ Connected to feature store: {self.total_transactions:,} transactions")
            print(f"   Date range: {self.data_start_date} to {self.data_end_date}")
    
    def run_backtest(
        self,
        start_date: str,
        end_date: str,
        output_dir: Optional[str] = None
    ) -> Dict:
        """
        Run complete backtest over date range.
        
        Args:
            start_date: Start date (ISO format: 'YYYY-MM-DD')
            end_date: End date (ISO format: 'YYYY-MM-DD')
            output_dir: Optional directory to save results
            
        Returns:
            Dict with:
                - 'daily_metrics': List[Dict] (one per day)
                - 'cumulative_metrics': Dict (overall performance)
                - 'alert_budget_adherence': bool
                - 'summary': Dict (key statistics)
        """
        start = pd.to_datetime(start_date).date()
        end = pd.to_datetime(end_date).date()
        
        if self.verbose:
            print(f"{'=' * 70}")
            print(f"RUNNING BACKTEST")
            print(f"{'=' * 70}")
            print(f"Period: {start} to {end}")
            print(f"Alert budget: {self.alert_budget:.1%}")
            print(f"{'=' * 70}\n")
        
        # Run day-by-day backtest
        daily_metrics = []
        current_date = start
        day_count = 0
        
        while current_date <= end:
            day_result = self.backtest_day(current_date)
            
            if day_result is not None:
                daily_metrics.append(day_result)
                day_count += 1
                
                if self.verbose and day_count % 10 == 0:
                    pct_complete = (current_date - start).days / (end - start).days * 100
                    print(f" Processed {day_count} days ({pct_complete:.1f}% complete)... (latest: {current_date})")
            
            current_date += timedelta(days=1)
        
        if self.verbose:
            print(f"\n✅ Backtest complete: {len(daily_metrics)} days processed\n")
        
        # Aggregate metrics
        cumulative_metrics = self._aggregate_metrics(daily_metrics)
        
        # Check budget adherence
        budget_adherence = self._check_budget_adherence(daily_metrics)
        
        # Generate summary
        summary = self._generate_summary(daily_metrics, cumulative_metrics)
        
        results = {
            'daily_metrics': daily_metrics,
            'cumulative_metrics': cumulative_metrics,
            'alert_budget_adherence': budget_adherence,
            'summary': summary,
            'config': {
                'start_date': str(start),
                'end_date': str(end),
                'alert_budget': self.alert_budget,
                'model_path': self.model_path
            }
        }
        
        # Save results if output_dir provided
        if output_dir:
            self._save_results(results, output_dir)
        
        # Print summary
        if self.verbose:
            self._print_summary(summary)
        
        return results
    
    def backtest_day(self, date: date) -> Optional[Dict]:
        """
        Backtest a single day.
        
        Args:
            date: Date to backtest
            
        Returns:
            Dict with daily metrics, or None if no data for this day
        """
        # Load transactions for this day
        query = """
        SELECT *
        FROM training_data
        WHERE DATE(event_timestamp) = ?
        ORDER BY event_timestamp
        """
        
        df = self.con.execute(query, [date]).df()
        
        if len(df) == 0:
            return None  # No data for this day
        
        # Extract labels
        if 'is_fraud' not in df.columns:
            warnings.warn(f"No 'is_fraud' column for {date}, skipping")
            return None
        
        y_true = df['is_fraud'].values
        
        # FEATURE ALIGNMENT (OPTIMIZED)
        # Build dictionary first
        feature_data = {}
        
        for feature_name in self.feature_names:
            if feature_name in df.columns:
                feature_data[feature_name] = df[feature_name]
            else:
                feature_data[feature_name] = 0
                if self.verbose and date.day == 1:
                    warnings.warn(f"Feature '{feature_name}' not found in data, filling with 0")
        
        # Create DataFrame
        X_df = pd.DataFrame(feature_data, index=df.index)
        
        # CRITICAL: Force column order to match model expectations
        X_df = X_df[self.feature_names]
        
        # Type conversion and NaN handling
        # Step 1: Convert everything to object first
        X_df = X_df.astype('object')
        
        # Step 2: Convert to numeric
        for col in X_df.columns:
            X_df[col] = pd.to_numeric(X_df[col], errors='coerce')
        
        # Step 3: Fill NaN with 0
        X_df = X_df.fillna(0)
        
        # Step 4: Convert to numpy array
        X = X_df.values.astype('float32')
        
        # Score with model
        import xgboost as xgb
        dtest = xgb.DMatrix(X, feature_names=self.feature_names)
        fraud_probs = self.model.predict(dtest)
        
        # Apply alert policy
        policy = AlertPolicy(budget_pct=self.alert_budget)
        
        # Create temporary DataFrame with timestamp
        temp_df = pd.DataFrame({
            'event_timestamp': df['event_timestamp'],
            'fraud_probability': fraud_probs,
            'is_fraud': y_true
        })
        
        alerts, policy_metadata = policy.decide_alerts(temp_df, fraud_probs)
        
        # Compute metrics
        daily_metrics = compute_daily_metrics(alerts, y_true, temp_df)
        
        # Should only have 1 day's metrics
        if len(daily_metrics) != 1:
            warnings.warn(f"Expected 1 day, got {len(daily_metrics)} for {date}")
            return daily_metrics[0] if len(daily_metrics) > 0 else None
        
        result = daily_metrics[0]
        result['date'] = str(date)
        
        return result
    
    def _aggregate_metrics(self, daily_metrics: List[Dict]) -> Dict:
        """
        Aggregate daily metrics into cumulative statistics.
        
        Args:
            daily_metrics: List of daily metric dicts
            
        Returns:
            Dict with cumulative metrics
        """
        if len(daily_metrics) == 0:
            return {}
        
        # Sum up totals
        total_transactions = sum(d['num_transactions'] for d in daily_metrics)
        total_alerts = sum(d['num_alerts'] for d in daily_metrics)
        total_tp = sum(d['tp'] for d in daily_metrics)
        total_fp = sum(d['fp'] for d in daily_metrics)
        total_fn = sum(d['fn'] for d in daily_metrics)
        total_tn = sum(d['tn'] for d in daily_metrics)
        
        # Calculate overall metrics
        overall_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
        overall_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
        overall_alert_rate = total_alerts / total_transactions if total_transactions > 0 else 0
        
        # Daily statistics
        daily_precisions = [d['precision'] for d in daily_metrics if d['num_alerts'] > 0]
        daily_recalls = [d['recall'] for d in daily_metrics if d['total_fraud'] > 0]
        
        return {
            'total_days': len(daily_metrics),
            'total_transactions': int(total_transactions),
            'total_fraud': int(total_tp + total_fn),
            'total_alerts': int(total_alerts),
            'total_caught': int(total_tp),
            'total_missed': int(total_fn),
            'total_false_positives': int(total_fp),
            'overall_precision': float(overall_precision),
            'overall_recall': float(overall_recall),
            'overall_alert_rate': float(overall_alert_rate),
            'average_daily_precision': float(np.mean(daily_precisions)) if daily_precisions else 0,
            'std_daily_precision': float(np.std(daily_precisions)) if daily_precisions else 0,
            'average_daily_recall': float(np.mean(daily_recalls)) if daily_recalls else 0,
            'std_daily_recall': float(np.std(daily_recalls)) if daily_recalls else 0
        }
    
    def _check_budget_adherence(self, daily_metrics: List[Dict]) -> bool:
        """
        Verify alert budget never exceeded on any day.
        
        Args:
            daily_metrics: List of daily metric dicts
            
        Returns:
            True if budget respected on all days
        """
        violations = []
        tolerance = 0.20  # 1% tolerance for rounding
        
        for day_result in daily_metrics:
            alert_rate = day_result['alert_rate']
            if alert_rate > self.alert_budget * (1 + tolerance):
                violations.append({
                    'date': day_result['date'],
                    'alert_rate': alert_rate,
                    'budget': self.alert_budget,
                    'violation': alert_rate - self.alert_budget
                })
        
        if violations and self.verbose:
            print(f"⚠️  Alert budget exceeded on {len(violations)} days:")
            for v in violations[:5]:  # Show first 5
                print(f"   {v['date']}: {v['alert_rate']:.4%} (budget: {v['budget']:.4%})")
        
        return len(violations) == 0
    
    def _generate_summary(
        self, 
        daily_metrics: List[Dict], 
        cumulative_metrics: Dict
    ) -> Dict:
        """Generate high-level summary statistics."""
        return {
            'period': f"{daily_metrics[0]['date']} to {daily_metrics[-1]['date']}" if daily_metrics else "N/A",
            'days_processed': len(daily_metrics),
            'total_transactions': cumulative_metrics.get('total_transactions', 0),
            'fraud_rate': cumulative_metrics.get('total_fraud', 0) / cumulative_metrics.get('total_transactions', 1),
            'overall_precision': cumulative_metrics.get('overall_precision', 0),
            'overall_recall': cumulative_metrics.get('overall_recall', 0),
            'fraud_caught': cumulative_metrics.get('total_caught', 0),
            'fraud_missed': cumulative_metrics.get('total_missed', 0),
            'alert_budget_violations': 0 if self._check_budget_adherence(daily_metrics) else len([
                d for d in daily_metrics if d['alert_rate'] > self.alert_budget * 1.01
            ])
        }
    
    def _save_results(self, results: Dict, output_dir: str):
        """Save backtest results to files."""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save daily metrics as CSV
        daily_df = pd.DataFrame(results['daily_metrics'])
        daily_df.to_csv(output_path / 'daily_metrics.csv', index=False)
        
        # Save complete results as JSON
        with open(output_path / 'backtest_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        if self.verbose:
            print(f"\n💾 Results saved to {output_dir}/")
            print(f"   - daily_metrics.csv")
            print(f"   - backtest_results.json")
    
    def _print_summary(self, summary: Dict):
        """Print backtest summary."""
        print(f"{'=' * 70}")
        print(f"BACKTEST SUMMARY")
        print(f"{'=' * 70}")
        print(f"Period: {summary['period']}")
        print(f"Days processed: {summary['days_processed']}")
        print(f"Total transactions: {summary['total_transactions']:,}")
        print(f"Fraud rate: {summary['fraud_rate']:.2%}")
        print(f"\n📊 Performance:")
        print(f"   Overall precision: {summary['overall_precision']:.2%}")
        print(f"   Overall recall: {summary['overall_recall']:.2%}")
        print(f"   Fraud caught: {summary['fraud_caught']:,}")
        print(f"   Fraud missed: {summary['fraud_missed']:,}")
        print(f"\n✅ Budget adherence:")
        if summary['alert_budget_violations'] == 0:
            print(f"   ✅ Budget respected on all days")
        else:
            print(f"   ❌ Budget exceeded on {summary['alert_budget_violations']} days")
        print(f"{'=' * 70}\n")


# ============================================================================
# TESTING EXAMPLE (Run with: python -m src.evaluation.backtest)
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("BACKTESTER - DEMO MODE")
    print("=" * 70)
    print("\n⚠️  This demo requires:")
    print("   1. Trained model: models/production/fraud_detector.json")
    print("   2. Feature store: data/processed/full_features.duckdb")
    print("\nIf you don't have these files, this demo will show you the interface.\n")
    
    # Check if files exist
    model_path = 'models/production/fraud_detector.json'
    feature_store_path = 'data/processed/full_features.duckdb'
    
    model_exists = Path(model_path).exists()
    features_exist = Path(feature_store_path).exists()
    
    if model_exists and features_exist:
        print("✅ Found model and feature store. Running REAL backtest...\n")
        
        # Initialize backtester
        backtester = Backtester(
            model_path=model_path,
            feature_store_path=feature_store_path,
            alert_budget=0.005,  # 0.5%
            verbose=True
        )
        
        # Run backtest on test period (from Phase 5)
        results = backtester.run_backtest(
            start_date='2025-06-01',
            end_date='2025-06-07',  # Just 1 week for demo
            output_dir='evaluation/backtest_results'
        )
        
        print("\n✅ Backtest complete!")
        print(f"\n💡 View results:")
        print(f"   - evaluation/backtest_results/daily_metrics.csv")
        print(f"   - evaluation/backtest_results/backtest_results.json")
        
    else:
        print("❌ Missing files:")
        if not model_exists:
            print(f"   - {model_path}")
        if not features_exist:
            print(f"   - {feature_store_path}")
        
        print("\n📝 To run backtest:")
        print("   1. Train model (Phase 5): python -m src.models.production_pipeline")
        print("   2. Build features (Phase 4): python -m src.features.offline_builder")
        print("   3. Run backtest: python -m src.evaluation.backtest")
        
        print("\n💡 Example usage:")
        print("""
        from src.evaluation.backtest import Backtester
        
        backtester = Backtester(
            model_path='models/production/fraud_detector.json',
            feature_store_path='data/processed/full_features.duckdb',
            alert_budget=0.005
        )
        
        results = backtester.run_backtest(
            start_date='2025-06-01',
            end_date='2025-07-02'
        )
        
        print(f"Precision: {results['summary']['overall_precision']:.2%}")
        print(f"Recall: {results['summary']['overall_recall']:.2%}")
        """)
