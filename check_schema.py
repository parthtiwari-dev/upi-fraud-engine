import duckdb

conn = duckdb.connect('data/processed/transactions.duckdb', read_only=True)
result = conn.execute("SELECT * FROM transactions LIMIT 0").df()
print("Columns in transactions table:")
print(list(result.columns))
