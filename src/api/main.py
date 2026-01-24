"""
FastAPI REST API for UPI Fraud Detection
Real-time fraud scoring with <500ms latency

Architecture:
- POST /score: Score single transaction
- GET /health: System health check
- GET /metrics: Performance metrics
- Computes 11 engineered features in real-time
- Uses pre-trained XGBoost model (Stage 2 only)
- Respects 0.5% daily alert budget

Author: Your Name
Date: January 24, 2026
"""
from fastapi import FastAPI, HTTPException, status
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import time
from datetime import datetime
import logging
from typing import Dict

# Import our modules
from src.api.models import (
    TransactionRequest,
    FraudScoreResponse,
    HealthCheckResponse,
    MetricsResponse,
    ErrorResponse
)
from src.api.service import FraudScoringService
from src.api.config import settings

# Setup logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Global service instance (initialized at startup)
scoring_service: FraudScoringService = None
startup_time: float = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager: Initialize resources at startup, cleanup at shutdown.
    
    Startup:
    - Load XGBoost model
    - Load label encoders
    - Connect to DuckDB feature store
    - Verify system health
    
    Shutdown:
    - Close database connections
    - Save metrics
    """
    global scoring_service, startup_time
    
    logger.info("="*70)
    logger.info("🚀 STARTING UPI FRAUD DETECTION API")
    logger.info("="*70)
    
    startup_time = time.time()
    
    try:
        # Initialize fraud scoring service
        logger.info("Loading fraud detection models...")
        scoring_service = FraudScoringService(
            model_path=settings.MODEL_PATH,
            encoders_path=settings.ENCODERS_PATH,
            features_path=settings.FEATURES_PATH,
            feature_store_path=settings.FEATURE_STORE_PATH,
            alert_budget_pct=settings.ALERT_BUDGET_PCT
        )
        
        logger.info("✅ Models loaded successfully")
        logger.info(f"   - XGBoost model: {settings.MODEL_PATH}")
        logger.info(f"   - Label encoders: {settings.ENCODERS_PATH}")
        logger.info(f"   - Feature store: {settings.FEATURE_STORE_PATH}")
        logger.info(f"   - Alert budget: {settings.ALERT_BUDGET_PCT*100}%")
        
        # Perform health check
        health = scoring_service.health_check()
        if health["status"] != "healthy":
            raise RuntimeError(f"Service unhealthy: {health}")
        
        logger.info("✅ Health check passed")
        logger.info("="*70)
        logger.info(f"🎯 API ready at http://{settings.API_HOST}:{settings.API_PORT}")
        logger.info("="*70)
        
        yield  # API is now running
        
    except Exception as e:
        logger.error(f"❌ Startup failed: {e}")
        raise
    
    finally:
        # Cleanup on shutdown
        logger.info("Shutting down API...")
        if scoring_service:
            scoring_service.close()
        logger.info("✅ Shutdown complete")


# Initialize FastAPI app
app = FastAPI(
    title=settings.API_TITLE,
    version=settings.API_VERSION,
    description=(
        "Production-ready UPI Fraud Detection API\n\n"
        "Features:\n"
        "- Real-time fraud scoring (<500ms latency)\n"
        "- 482 engineered features\n"
        "- XGBoost classifier (0.8953 ROC-AUC)\n"
        "- 0.5% daily alert budget enforcement\n"
        "- Point-in-time feature correctness\n\n"
        "Architecture:\n"
        "- Phase 1-4: Synthetic UPI data + feature engineering\n"
        "- Phase 5: XGBoost training (Stage 2 only)\n"
        "- Phase 6: Backtesting with alert policy\n"
        "- Phase 7: FastAPI deployment (this API)\n"
    ),
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)


# Custom exception handler for cleaner error responses
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Handle all uncaught exceptions gracefully."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred. Please try again.",
            "timestamp": datetime.utcnow().isoformat()
        }
    )


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.post(
    "/score",
    response_model=FraudScoreResponse,
    status_code=status.HTTP_200_OK,
    summary="Score UPI Transaction for Fraud",
    description=(
        "Submit a UPI transaction for real-time fraud detection.\n\n"
        "**Process:**\n"
        "1. Validate transaction schema\n"
        "2. Compute 11 real-time features from DuckDB\n"
        "3. Score with XGBoost model (482 features)\n"
        "4. Apply alert policy (0.5% daily budget)\n"
        "5. Return fraud probability + alert decision\n\n"
        "**Latency SLA:** <500ms\n"
        "**Input:** Transaction with Vesta features (V1-V339, C1-C14, etc.)\n"
        "**Output:** Fraud probability [0,1] + should_alert boolean"
    ),
    responses={
        200: {"description": "Successfully scored transaction"},
        400: {"model": ErrorResponse, "description": "Invalid transaction data"},
        500: {"model": ErrorResponse, "description": "Scoring failed"}
    }
)
async def score_transaction(txn: TransactionRequest) -> FraudScoreResponse:
    """
    Score a single UPI transaction for fraud.
    
    Example:
    ```json
    {
        "transaction_id": "TXN20260124195600ABC",
        "payer_id": "123456",
        "payer_vpa": "user@paytm",
        "payee_id": "789012",
        "payee_vpa": "merchant@phonepe",
        "amount": 5000.0,
        "currency": "INR",
        "device_id": "device_xyz_123",
        "device_type": "mobile",
        "event_timestamp": "2026-01-24T19:56:00Z",
        "V258": 523.5,  // Vesta features already present
        "V294": 1.0,
        ...
    }
    ```
    
    Returns:
    ```json
    {
        "transaction_id": "TXN20260124195600ABC",
        "fraud_probability": 0.087,
        "should_alert": false,
        "stage2_score": 0.087,
        "latency_ms": 145.3,
        "model_version": "v1.0",
        "threshold_used": 0.994,
        "alert_budget_remaining": 342,
        "feature_compute_ms": 98.2
    }
    ```
    """
    start_time = time.time()
    
    try:
        logger.info(f"📨 Scoring transaction: {txn.transaction_id}")
        
        # Score the transaction
        result = scoring_service.score(txn)
        
        # Calculate total latency
        total_latency_ms = (time.time() - start_time) * 1000
        result.latency_ms = total_latency_ms
        
        # Log performance
        if total_latency_ms > settings.MAX_LATENCY_MS:
            logger.warning(
                f"⚠️  Latency exceeded SLA: {total_latency_ms:.1f}ms "
                f"(target: {settings.MAX_LATENCY_MS}ms)"
            )
        
        logger.info(
            f"✅ Scored {txn.transaction_id}: "
            f"P(fraud)={result.fraud_probability:.3f}, "
            f"alert={result.should_alert}, "
            f"latency={total_latency_ms:.1f}ms"
        )
        
        return result
        
    except ValueError as e:
        # Invalid input data
        logger.error(f"❌ Validation error for {txn.transaction_id}: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "ValidationError",
                "message": str(e),
                "transaction_id": txn.transaction_id
            }
        )
        
    except Exception as e:
        # Scoring failed
        logger.error(f"❌ Scoring failed for {txn.transaction_id}: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": "ScoringError",
                "message": "Failed to score transaction. Using safe default fallback.",
                "transaction_id": txn.transaction_id
            }
        )


@app.get(
    "/health",
    response_model=HealthCheckResponse,
    status_code=status.HTTP_200_OK,
    summary="Health Check",
    description=(
        "Check if the fraud detection service is healthy.\n\n"
        "**Checks:**\n"
        "- XGBoost model loaded\n"
        "- Label encoders loaded\n"
        "- DuckDB connection working\n"
        "- Last prediction latency\n\n"
        "**Use:** Monitor service availability, readiness probes (Kubernetes)"
    )
)
async def health_check() -> HealthCheckResponse:
    """
    System health check endpoint.
    
    Returns:
    - status: "healthy" | "degraded" | "down"
    - model_loaded: bool
    - encoders_loaded: bool
    - feature_store_ok: bool
    - uptime_seconds: float
    - last_prediction_ms: float (optional)
    """
    try:
        health = scoring_service.health_check()
        
        # Add uptime
        health["uptime_seconds"] = time.time() - startup_time
        
        return HealthCheckResponse(**health)
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthCheckResponse(
            status="down",
            model_loaded=False,
            encoders_loaded=False,
            feature_store_ok=False,
            uptime_seconds=time.time() - startup_time,
            last_prediction_ms=None
        )


@app.get(
    "/metrics",
    response_model=MetricsResponse,
    status_code=status.HTTP_200_OK,
    summary="Performance Metrics",
    description=(
        "Get API performance metrics.\n\n"
        "**Metrics:**\n"
        "- Total requests served\n"
        "- Total alerts issued\n"
        "- Alert rate (0-1)\n"
        "- Latency percentiles (p50, p95, p99)\n"
        "- Requests per second (RPS)\n"
        "- Error count\n\n"
        "**Use:** Monitoring dashboards, SLA tracking, performance debugging"
    )
)
async def get_metrics() -> MetricsResponse:
    """
    Retrieve API performance metrics.
    
    Returns:
    - total_requests: Total transactions scored
    - total_alerts: Total alerts issued
    - alert_rate: Fraction of transactions alerted (should be ~0.005)
    - avg_latency_ms: Average scoring time
    - p50/p95/p99_latency_ms: Latency percentiles
    - requests_per_second: Current RPS
    - error_count: Total errors
    """
    try:
        metrics = scoring_service.get_metrics()
        return MetricsResponse(**metrics)
        
    except Exception as e:
        logger.error(f"Failed to retrieve metrics: {e}")
        # Return empty metrics on failure
        return MetricsResponse(
            total_requests=0,
            total_alerts=0,
            alert_rate=0.0,
            avg_latency_ms=0.0,
            p50_latency_ms=0.0,
            p95_latency_ms=0.0,
            p99_latency_ms=0.0,
            requests_per_second=0.0,
            error_count=1
        )


@app.get(
    "/",
    summary="Root Endpoint",
    description="Welcome message with API information"
)
async def root() -> Dict:
    """Root endpoint with API info."""
    return {
        "service": "UPI Fraud Detection API",
        "version": settings.API_VERSION,
        "status": "running",
        "uptime_seconds": time.time() - startup_time,
        "endpoints": {
            "score": "POST /score - Score a transaction",
            "health": "GET /health - Health check",
            "metrics": "GET /metrics - Performance metrics",
            "docs": "GET /docs - Interactive API documentation"
        },
        "documentation": "/docs",
        "project": {
            "phases_completed": "1-6",
            "current_phase": "7 - FastAPI Deployment",
            "model": "XGBoost (Stage 2 only)",
            "roc_auc": 0.8953,
            "features": 482,
            "alert_budget": f"{settings.ALERT_BUDGET_PCT*100}%"
        }
    }


# ============================================================================
# STARTUP MESSAGE
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    
    print("\n" + "="*70)
    print("🚀 STARTING UPI FRAUD DETECTION API")
    print("="*70)
    print(f"📍 Host: {settings.API_HOST}")
    print(f"🔌 Port: {settings.API_PORT}")
    print(f"📊 Model: {settings.MODEL_PATH}")
    print(f"💾 Feature Store: {settings.FEATURE_STORE_PATH}")
    print(f"🚨 Alert Budget: {settings.ALERT_BUDGET_PCT*100}%")
    print("="*70)
    print("\n💡 Tips:")
    print("   - API Docs: http://localhost:8000/docs")
    print("   - Health Check: http://localhost:8000/health")
    print("   - Test scoring: curl -X POST http://localhost:8000/score -d '{...}'")
    print("\n" + "="*70 + "\n")
    
    uvicorn.run(
        "src.api.main:app",
        host=settings.API_HOST,
        port=settings.API_PORT,
        reload=True,  # Auto-reload on code changes (dev only)
        log_level=settings.LOG_LEVEL.lower()
    )
