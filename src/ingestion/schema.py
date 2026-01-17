from datetime import datetime
from typing import Optional, Any
from pydantic import BaseModel, Field, validator

class Transaction(BaseModel):
    # --- 1. THE VIPS (CRITICAL COLUMNS) ---
    # We define these strictly. If "amount" is missing, the system crashes.
    # If "amount" is a string "100.50", Pydantic converts it to float 100.5.
    
    transaction_id: str
    event_timestamp: datetime  
    
    # Financials
    amount: float
    currency: str = "INR"  # Default to INR if missing
    
    # UPI Identities (The "Enriched" parts)
    payer_vpa: str
    payee_vpa: str
    device_id: str
    
    # Labels (Optional because live data won't have them yet)
    is_fraud: Optional[float] = None  # Float because your dataset might have 1.0/0.0
    fraud_pattern: Optional[str] = None
    label_available_timestamp: Optional[datetime] = None
    
    # --- 2. THE CROWD (RAW FEATURES) ---
    # We won't type out V1..V339 manually. 
    # Instead, we tell Pydantic: "If you see other columns, just let them in."
    # This acts as a catch-all bucket for V1, C1, D1, etc.
    
    class Config:
        extra = "allow"  # <--- THIS IS THE MAGIC
        # "allow" means: "If the input has columns I didn't define (like V1, V2), 
        # add them to the object anyway so I can use them later."

    # --- 3. DATA CLEANING (VALIDATORS) ---
    # Sometimes data is messy. We use validators to clean it on the way in.
    
    @validator("amount")
    def amount_must_be_positive(cls, v):
        if v < 0:
            raise ValueError("Amount cannot be negative")
        return v

    @validator("payer_vpa", "payee_vpa")
    def vpa_must_be_lowercase(cls, v):
        if v:
            return v.lower() # Normalize all VPAs to lowercase
        return v

    @validator("transaction_id", pre=True)
    def force_string_id(cls, v):
        return str(v)
