import duckdb

con = duckdb.connect("data/processed/transactions.duckdb")
con.execute("""
    COPY (
        SELECT
            CAST(transaction_id AS VARCHAR) AS transaction_id,
            *
        FROM transactions
        ORDER BY event_timestamp
        LIMIT 15000
    )
    TO 'data/processed/parity_sample.parquet'
""")
con.close()

print("parity_sample.parquet created")
