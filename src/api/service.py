"""
Fraud Scoring Service for FastAPI Integration.

Wraps the existing FraudPredictor from single_predict.py and adds:
- Alert budget enforcement (0.5% daily limit)
- Performance metrics tracking
- Health checks
- State management across requests

Author: Your Name
Date: January 24, 2026
"""
import time
import numpy as np
from typing import Dict, Optional
from datetime import datetime, date
from collections import deque
import logging


from src.inference.single_predict import FraudPredictor
from src.api.models import TransactionRequest, FraudScoreResponse


logger = logging.getLogger(__name__)



class ServiceMetrics:
    """
    Tracks API performance metrics across requests.
    
    Metrics:
    - Total requests, alerts, errors
    - Latency percentiles (p50, p95, p99)
    - Requests per second
    - Alert budget utilization
    """
    
    def __init__(self):
        self.total_requests = 0
        self.total_alerts = 0
        self.error_count = 0
        self.latencies = deque(maxlen=10000)  # Keep last 10K latencies
        self.start_time = time.time()
        self.last_request_time = time.time()
        
        # Daily tracking (resets at midnight)
        self.current_date = date.today()
        self.daily_transaction_count = 0
        self.daily_alert_count = 0
    
    def record_request_without_count(self, latency_ms: float, alerted: bool):
        """
        Record a scored transaction WITHOUT incrementing daily count.
        
        Note: daily_transaction_count is incremented in score() BEFORE 
        budget calculation to avoid off-by-one error.
        """
        self.total_requests += 1
        self.latencies.append(latency_ms)
        self.last_request_time = time.time()
        
        # Check if new day (reset counters)
        today = date.today()
        if today != self.current_date:
            logger.info(f"New day detected. Resetting daily counters. "
                       f"Yesterday: {self.daily_transaction_count} txns, "
                       f"{self.daily_alert_count} alerts "
                       f"({self.daily_alert_count/max(self.daily_transaction_count,1)*100:.2f}%)")
            self.current_date = today
            self.daily_transaction_count = 0
            self.daily_alert_count = 0
        
        # DON'T increment daily_transaction_count here (already done in score())
        
        if alerted:
            self.total_alerts += 1
            self.daily_alert_count += 1

    def record_error(self):
        """Record a failed prediction."""
        self.error_count += 1
    
    def get_summary(self) -> Dict:
        """Get current metrics summary."""
        if not self.latencies:
            return {
                "total_requests": 0,
                "total_alerts": 0,
                "alert_rate": 0.0,
                "avg_latency_ms": 0.0,
                "p50_latency_ms": 0.0,
                "p95_latency_ms": 0.0,
                "p99_latency_ms": 0.0,
                "requests_per_second": 0.0,
                "error_count": 0,
                "daily_budget_utilization": 0.0
            }
        
        latencies = np.array(list(self.latencies))
        uptime_seconds = time.time() - self.start_time
        rps = self.total_requests / max(uptime_seconds, 1)
        
        # Calculate budget utilization (daily)
        budget_utilization = self.daily_alert_count / max(self.daily_transaction_count, 1)
        
        return {
            "total_requests": self.total_requests,
            "total_alerts": self.total_alerts,
            "alert_rate": self.total_alerts / max(self.total_requests, 1),
            "avg_latency_ms": float(np.mean(latencies)),
            "p50_latency_ms": float(np.percentile(latencies, 50)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "p99_latency_ms": float(np.percentile(latencies, 99)),
            "requests_per_second": float(rps),
            "error_count": self.error_count,
            "daily_budget_utilization": float(budget_utilization)
        }



class FraudScoringService:
    """
    Production fraud scoring service for FastAPI.
    
    Wraps FraudPredictor and adds:
    - Daily alert budget enforcement (0.5% of transactions)
    - Performance metrics tracking
    - Health checks
    
    Usage:
        service = FraudScoringService(
            model_path="models/production/fraud_detector.json",
            alert_budget_pct=0.005
        )
        
        result = service.score(transaction_request)
    """
    
    def __init__(
        self,
        model_path: str,
        encoders_path: Optional[str] = None,  # Not used (in model artifacts)
        features_path: Optional[str] = None,  # Not used (in model artifacts)
        feature_store_path: Optional[str] = None,  # Not used (in-memory only)
        alert_budget_pct: float = 0.005
    ):
        """
        Initialize fraud scoring service.
        
        Args:
            model_path: Path to fraud_detector.json
            alert_budget_pct: Daily alert budget (default 0.5%)
        """
        logger.info("Initializing FraudScoringService...")
        
        # REUSE existing FraudPredictor (don't duplicate logic!)
        self.predictor = FraudPredictor(model_path)
        
        # Alert budget configuration
        self.alert_budget_pct = alert_budget_pct
        
        # Metrics tracking
        self.metrics = ServiceMetrics()
        
        logger.info("✅ FraudScoringService ready")
        logger.info(f"   Alert budget: {alert_budget_pct*100}% per day")
        logger.info(f"   Model: {self.predictor.metadata.get('model_type', 'XGBoost')}")
        logger.info(f"   Features: {len(self.predictor.feature_names)}")
    
    def score(self, txn: TransactionRequest) -> FraudScoreResponse:
        """
        Score a single transaction for fraud.
        
        Pipeline:
        1. Increment daily transaction count (BEFORE budget calculation)
        2. Call predictor.predict_single() → get fraud_probability
        3. Decide alert based on probability + budget
        4. Track metrics
        5. Return response
        """
        start_time = time.time()
        
        try:
            # ✅ FIX: Increment count FIRST
            self.metrics.daily_transaction_count += 1
            
            # Convert Pydantic model to dict
            txn_dict = txn.model_dump()
            
            # Use existing FraudPredictor (returns probability only)
            result = self.predictor.predict_single(txn_dict)
            fraud_prob = result['fraud_probability']
            
            # ✅ NEW: Decide alert based on probability + budget
            # Define threshold (you control this!)
            alert_threshold = 0.5  # Alert on fraud_prob >= 50%
            
            # Calculate daily budget
            daily_budget = max(1, int(self.metrics.daily_transaction_count * self.alert_budget_pct))
            budget_exceeded = False
            
            # Alert decision logic
            if fraud_prob >= alert_threshold:  # Suspicious transaction
                if self.metrics.daily_alert_count < daily_budget:  # Budget available
                    should_alert = True
                else:  # Budget exhausted
                    should_alert = False
                    budget_exceeded = True
                    logger.warning(
                        f"⚠️  Alert budget exhausted: {self.metrics.daily_alert_count}/{daily_budget}. "
                        f"Suppressing alert for {txn.transaction_id}"
                    )
            else:  # Not suspicious
                should_alert = False
            
            # Calculate remaining budget
            alert_budget_remaining = max(0, daily_budget - self.metrics.daily_alert_count)
            
            # Total latency
            total_latency_ms = (time.time() - start_time) * 1000
            
            # ✅ Track metrics WITHOUT re-incrementing count
            self.metrics.record_request_without_count(total_latency_ms, should_alert)
            
            # Build response
            response = FraudScoreResponse(
                transaction_id=result['transaction_id'],
                fraud_probability=fraud_prob,
                should_alert=should_alert,
                stage2_score=fraud_prob,
                latency_ms=total_latency_ms,
                model_version=self.predictor.metadata.get('model_type', 'v1.0'),
                threshold_used=alert_threshold,  # ← Now set by service
                risk_tier=result['risk_tier'],
                alert_budget_remaining=alert_budget_remaining,
                feature_compute_ms=result.get('latency_ms', None),
                budget_exceeded=budget_exceeded
            )
            
            return response
            
        except Exception as e:
            self.metrics.record_error()
            logger.error(f"Scoring failed: {e}", exc_info=True)
            raise

    
    def health_check(self) -> Dict:
        """
        Check if service is healthy.
        
        Returns:
            Dict with health status and component checks
        """
        try:
            # Check if model is loaded
            model_ok = self.predictor.model is not None
            
            # Check if encoders are loaded
            encoders_ok = self.predictor.encoders is not None
            
            # Check if feature store is initialized
            feature_store_ok = self.predictor.feature_store is not None
            
            # Overall status
            if model_ok and encoders_ok and feature_store_ok:
                status = "healthy"
            else:
                status = "degraded"
            
            return {
                "status": status,
                "model_loaded": model_ok,
                "encoders_loaded": encoders_ok,
                "feature_store_ok": feature_store_ok,
                "last_prediction_ms": self.metrics.latencies[-1] if self.metrics.latencies else None
            }
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "down",
                "model_loaded": False,
                "encoders_loaded": False,
                "feature_store_ok": False,
                "last_prediction_ms": None
            }
    
    def get_metrics(self) -> Dict:
        """Get current performance metrics."""
        return self.metrics.get_summary()
    
    def close(self):
        """Cleanup resources on shutdown."""
        logger.info("Closing FraudScoringService...")
        logger.info(f"Final stats: {self.metrics.total_requests} requests, "
                   f"{self.metrics.total_alerts} alerts, "
                   f"{self.metrics.error_count} errors")
