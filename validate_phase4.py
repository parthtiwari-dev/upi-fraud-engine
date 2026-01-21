import duckdb
con = duckdb.connect("data/processed/full_features.duckdb", read_only=True)
columns = con.execute("SELECT * FROM training_data LIMIT 0").df().columns.tolist()
con.close()

# Count engineered
engineered = ['payer_txn_count_5min', 'payer_txn_sum_5min', 'payer_txn_count_1h', 
              'payer_txn_sum_1h', 'payer_txn_count_24h', 'payer_txn_sum_24h',
              'device_txn_count_1h', 'device_txn_count_24h',
              'device_distinct_payers_7d', 'payer_distinct_payees_7d',
              'payer_past_fraud_count_30d']

# Count identifiers
identifiers = ['transaction_id', 'event_timestamp', 'is_fraud', 'payer_id']

# Raw = Total - Engineered - Identifiers
raw_count = len(columns) - len(engineered) - len(identifiers)

print(f"Total: {len(columns)}")
print(f"Engineered: {len(engineered)}")
print(f"Identifiers: {len(identifiers)}")
print(f"Raw Vesta: {raw_count}")
