"""
Pydantic models for API request/response validation.
Enforces strict type checking at API boundary.
"""
from pydantic import BaseModel, Field, validator
from datetime import datetime
from typing import Optional


class TransactionRequest(BaseModel):
    """
    Input: Raw UPI transaction for fraud scoring.
    
    Contains:
    - Core transaction fields (required)
    - All Vesta features (optional - will use NaN if missing)
    
    The 'extra = "allow"' config accepts all 476 Vesta features
    without defining each one explicitly.
    """
    # === Required Core Fields ===
    transaction_id: str = Field(..., description="Unique transaction ID")
    event_timestamp: datetime = Field(..., description="Transaction timestamp (ISO 8601)")
    amount: float = Field(..., gt=0, description="Transaction amount in INR")
    payer_vpa: str = Field(..., description="Payer UPI address (e.g., user_abc123@upi)")
    payee_vpa: str = Field(..., description="Payee UPI address")
    device_id: str = Field(..., description="Device identifier")
    currency: str = Field(default="INR", description="Currency code")
    
    # === Optional Vesta Features (from Kaggle IEEE-CIS) ===
    # V1-V339: Vesta anonymized features
    V1: Optional[float] = None
    V2: Optional[float] = None
    V3: Optional[float] = None
    # ... (define all V1-V339 if needed, or rely on extra="allow")
    
    # C1-C14: Categorical aggregates
    C1: Optional[float] = None
    C2: Optional[float] = None
    # ... (C3-C14)
    
    # D1-D15: Timedelta features
    D1: Optional[float] = None
    D2: Optional[float] = None
    # ... (D3-D15)
    
    # M1-M9: Match features
    M1: Optional[bool] = None
    M2: Optional[bool] = None
    M3: Optional[bool] = None
    M4: Optional[str] = None
    # ... (M5-M9)
    
    # id_01-id_38: Identity features
    id_01: Optional[float] = None
    id_02: Optional[float] = None
    # ... (id_03 to id_38)
    
    # Card features
    card1: Optional[int] = None
    card2: Optional[float] = None
    card3: Optional[float] = None
    card4: Optional[str] = None
    card5: Optional[float] = None
    card6: Optional[str] = None
    
    # Address/distance features
    addr1: Optional[float] = None
    addr2: Optional[float] = None
    dist1: Optional[float] = None
    dist2: Optional[float] = None
    
    # Email domains
    P_emaildomain: Optional[str] = None
    R_emaildomain: Optional[str] = None
    
    # Device info
    DeviceType: Optional[str] = None
    DeviceInfo: Optional[str] = None
    
    # Product code
    ProductCD: Optional[str] = None
    
    @validator('amount')
    def amount_must_be_positive(cls, v):
        if v <= 0:
            raise ValueError("Amount must be positive")
        return v
    
    @validator('currency')
    def currency_must_be_inr(cls, v):
        if v != "INR":
            raise ValueError("Only INR currency supported")
        return v
    
    class Config:
        extra = "allow"  # Accept all Vesta features without defining each
        json_schema_extra = {  # ✅ FIX: Renamed from schema_extra
            "example": {
                "transaction_id": "TXN20260124200600ABC",
                "event_timestamp": "2026-01-24T20:06:00Z",
                "amount": 5000.0,
                "payer_vpa": "user_abc123@upi",
                "payee_vpa": "merchant_xyz@upi",
                "device_id": "device_xyz_123",
                "currency": "INR",
                "V1": 1.23,
                "V2": 0.45,
                "V258": 523.5,
                "C1": 12.0,
                "C2": 5.0
            }
        }


class FraudScoreResponse(BaseModel):
    """
    Output: Fraud detection result with metadata.
    """
    transaction_id: str
    fraud_probability: float = Field(..., ge=0, le=1, description="Fraud probability [0,1]")
    should_alert: bool = Field(..., description="Alert decision (respects budget)")
    stage2_score: float = Field(..., description="XGBoost fraud probability")
    latency_ms: float = Field(..., description="Total scoring latency")
    model_version: str = Field(default="v1.0", description="Model version")
    threshold_used: float = Field(..., description="Alert threshold applied")
    risk_tier: str = Field(..., description="Risk tier: low/medium/high/critical")
    alert_budget_remaining: Optional[int] = Field(None, description="Remaining alerts today")
    feature_compute_ms: Optional[float] = Field(None, description="Feature computation time")
    budget_exceeded: Optional[bool] = Field(False, description="True if budget exhausted")
    
    class Config:
        json_schema_extra = {  # ✅ FIX: Renamed from schema_extra
            "example": {
                "transaction_id": "TXN20260124200600ABC",
                "fraud_probability": 0.087,
                "should_alert": False,
                "stage2_score": 0.087,
                "latency_ms": 145.3,
                "model_version": "v1.0",
                "threshold_used": 0.994,
                "risk_tier": "low",
                "alert_budget_remaining": 342,
                "feature_compute_ms": 98.2,
                "budget_exceeded": False
            }
        }


class HealthCheckResponse(BaseModel):
    """System health status."""
    status: str = Field(..., description="healthy | degraded | down")
    model_loaded: bool
    encoders_loaded: bool
    feature_store_ok: bool
    uptime_seconds: float
    last_prediction_ms: Optional[float] = None


class MetricsResponse(BaseModel):
    """API performance metrics."""
    total_requests: int
    total_alerts: int
    alert_rate: float = Field(..., description="Fraction of txns alerted")
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    p99_latency_ms: float
    requests_per_second: float
    error_count: int = 0
    daily_budget_utilization: float = Field(..., description="0-1, how much budget used")


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    message: str
    transaction_id: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
