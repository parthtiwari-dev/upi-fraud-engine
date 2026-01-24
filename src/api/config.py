"""
Configuration management using Pydantic Settings.
Loads from environment variables or .env file.
"""
from pydantic_settings import BaseSettings
from typing import Optional


class Settings(BaseSettings):
    """
    API configuration loaded from environment variables.
    
    Usage:
        # .env file
        MODEL_PATH=models/production/fraud_detector.json
        ALERT_BUDGET_PCT=0.005
        API_PORT=8000
        
        # In code
        from src.api.config import settings
        print(settings.MODEL_PATH)
    """
    # Model artifacts
    MODEL_PATH: str = "models/production/fraud_detector.json"
    ENCODERS_PATH: Optional[str] = None  # Not used (bundled in model)
    FEATURES_PATH: Optional[str] = None  # Not used (bundled in model)
    FEATURE_STORE_PATH: Optional[str] = None  # Not used (in-memory)
    
    # Alert policy
    ALERT_BUDGET_PCT: float = 0.005  # 0.5% daily budget
    
    # Performance
    MAX_LATENCY_MS: float = 500.0  # SLA target
    
    # API settings
    API_TITLE: str = "UPI Fraud Detection API"
    API_VERSION: str = "1.0.0"
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FILE: Optional[str] = None
    
    # Safe defaults (Phase 7 monitoring - not implemented yet)
    DEVICE_RING_THRESHOLD: int = 3
    VELOCITY_BURST_THRESHOLD: int = 100
    FEATURE_NULL_PERCENTILE: float = 99.0
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"  # ✅ FIX: Ignore extra env vars like PYTHONPATH


# Global settings instance
settings = Settings()
