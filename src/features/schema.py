"""
Goal: Create a strict definition of the "Output" of this phase.

Logic to Implement:
Define a class FeatureVector. It must have these specific fields (and types):

1. Identity (The Keys):
• transaction_id (String): To join back to raw data.
• event_timestamp (Datetime): To verify time order.

2. Payer Velocity (The "Is this user manic?" signals):
• payer_txn_count_5min (Int): Count of txns by this payer in [T-5m, T).
• payer_txn_sum_5min (Float): Sum of amounts.
• payer_txn_count_1h (Int): Count in [T-1h, T).
• payer_txn_sum_1h (Float): Sum of amounts.
• payer_txn_count_24h (Int): Count in [T-24h, T).
• payer_txn_sum_24h (Float): Sum of amounts.

3. Device Velocity (The "Is this phone hot?" signals):
• device_txn_count_1h (Int): Count of txns from this device_id.
• device_txn_count_24h (Int): Count of txns from this device_id.

4. Graph Features (The "Criminal Network" signals):
• device_distinct_payers_7d (Int): How many unique payer_ids used this device in the last 7 days? (High = Fraud Ring).
• payer_distinct_payees_7d (Int): How many unique merchants did this user pay?

5. Risk History (The "Repeat Offender" signals):
• payer_past_fraud_count_30d (Int): How many confirmed fraud transactions did this payer have in the last 30 days?
• Constraint: Only count fraud where label_available_timestamp < T.
"""


from datetime import datetime
from pydantic import BaseModel, Field

class FeatureVector(BaseModel):
    """
    Defines the strict schema for the Feature Store output.
    Represents the world state reconstructed exactly at T - 1ms.
    """

    # --- 1. Identity (The Keys) ---
    transaction_id: str = Field(..., description="Unique ID to join back to raw data")
    event_timestamp: datetime = Field(..., description="The reference time T. Used to verify time order.")

    # --- 2. Payer Velocity (Manic User Signals) ---
    # Window: [T-5m, T)
    payer_txn_count_5min: int = Field(..., ge=0)
    payer_txn_sum_5min: float = Field(..., ge=0.0)

    # Window: [T-1h, T)
    payer_txn_count_1h: int = Field(..., ge=0)
    payer_txn_sum_1h: float = Field(..., ge=0.0)

    # Window: [T-24h, T)
    payer_txn_count_24h: int = Field(..., ge=0)
    payer_txn_sum_24h: float = Field(..., ge=0.0)

    # --- 3. Device Velocity (Hot Device Signals) ---
    # Note: Device velocity usually tracks usage intensity
    device_txn_count_1h: int = Field(..., ge=0)
    device_txn_count_24h: int = Field(..., ge=0)

    # --- 4. Graph Features (Criminal Network Signals) ---
    # Window: 7 days lookback
    device_distinct_payers_7d: int = Field(..., ge=0, description="Unique payer_ids on this device (Fraud Ring signal)")
    payer_distinct_payees_7d: int = Field(..., ge=0, description="Unique merchants paid by this user")

    # --- 5. Risk History (Repeat Offender Signals) ---
    # Window: 30 days lookback
    # CONSTRAINT: Only counts fraud where label_available_timestamp < T
    payer_past_fraud_count_30d: int = Field(..., ge=0)

    class Config:
        frozen = True  # Makes instances immutable to prevent accidental modification