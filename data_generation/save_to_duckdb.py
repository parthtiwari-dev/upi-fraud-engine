import duckdb
from pathlib import Path

def save_to_duckdb():
    project_root = Path(__file__).parent.parent
    csv_path = project_root / "data" / "processed" / "full_upi_dataset_injected.csv"
    db_path = project_root / "data" / "processed" / "transactions.duckdb"
    
    print(f"🚀 Converting to DuckDB...")
    print(f"Input:  {csv_path}")
    print(f"Output: {db_path}")
    
    # Connect to (or create) the DuckDB file
    con = duckdb.connect(str(db_path))
    
    # 1. Drop table if exists (so you can re-run this script safely)
    con.execute("DROP TABLE IF EXISTS transactions")
    
    # 2. Create table directly from CSV (Super fast "Bulk Load")
    # read_csv_auto infers types automatically
    print("Loading data (this is fast)...")
    con.execute(f"""
        CREATE TABLE transactions AS 
        SELECT * FROM read_csv_auto('{csv_path}')
    """)
    
    # 3. Verify
    count = con.execute("SELECT count(*) FROM transactions").fetchone()[0]
    print(f"✅ Success! Saved {count} rows into 'transactions' table.")
    
    # Optional: Preview
    print("\nSample Data:")
    print(con.execute("SELECT transaction_id, is_fraud, fraud_pattern FROM transactions WHERE fraud_pattern IS NOT NULL LIMIT 5").df())
    
    con.close()

if __name__ == "__main__":
    save_to_duckdb()
