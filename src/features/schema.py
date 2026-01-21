from datetime import datetime
from pydantic import BaseModel, Field

class FeatureVector(BaseModel):
    """
    Defines the strict schema for the Feature Store output.
    Represents the world state reconstructed exactly at T - 1ms.
    """

    transaction_id: str = Field(..., description="Unique ID to join back to raw data")
    event_timestamp: datetime = Field(..., description="The reference time T. Used to verify time order.")

    payer_txn_count_5min: int = Field(..., ge=0)
    payer_txn_sum_5min: float = Field(..., ge=0.0)

    
    payer_txn_count_1h: int = Field(..., ge=0)
    payer_txn_sum_1h: float = Field(..., ge=0.0)

    payer_txn_count_24h: int = Field(..., ge=0)
    payer_txn_sum_24h: float = Field(..., ge=0.0)

    device_txn_count_1h: int = Field(..., ge=0)
    device_txn_count_24h: int = Field(..., ge=0)

    device_distinct_payers_7d: int = Field(..., ge=0, description="Unique payers in last ~1000 device txns (Fraud Ring signal)")
    payer_distinct_payees_7d: int = Field(..., ge=0, description="Unique payees in last ~1000 payer txns")

   
    payer_past_fraud_count_30d: int = Field(..., ge=0)

    class Config:
        frozen = True  
        extra = "forbid"