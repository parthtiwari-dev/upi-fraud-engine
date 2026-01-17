from src.ingestion.schema import Transaction

# Mock data (mimics one row from your DuckDB)
raw_data = {
    "transaction_id": "TXN_12345",
    "event_timestamp": "2025-01-01T12:00:00", # String input
    "amount": "500.00",                       # String input
    "payer_vpa": "USER@UPI",                  # Uppercase input
    "payee_vpa": "SHOP@UPI",
    "device_id": "DEV_999",
    "V1": 1.0,                                # Extra column
    "id-01": 55                               # Weird column name
}

# Try to load it into your Law
txn = Transaction(**raw_data)

print(f"ID: {txn.transaction_id}")
print(f"Time: {txn.event_timestamp} (Type: {type(txn.event_timestamp)})")
print(f"Amount: {txn.amount} (Type: {type(txn.amount)})")
print(f"VPA: {txn.payer_vpa}")  # Should be lowercase
print(f"Extra V1: {txn.V1}")    # Pydantic let this in!
