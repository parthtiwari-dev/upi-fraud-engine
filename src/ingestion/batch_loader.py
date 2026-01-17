import duckdb
import pandas as pd
from typing import List
from src.ingestion.schema import Transaction

def load_enriched_data(duckdb_path: str) -> pd.DataFrame:
    """
    Reads the raw data from DuckDB, sorted by time.
    Sorting is CRITICAL for preventing time-travel leakage later.
    """
    print(f"Connecting to {duckdb_path}...")
    con = duckdb.connect(duckdb_path)
    
    # We SELECT * to get all columns (including V1..V339)
    # We ORDER BY event_timestamp to simulate the timeline
    query = "SELECT * FROM transactions ORDER BY event_timestamp"
    
    df = con.execute(query).df()
    con.close()
    
    print(f"Loaded {len(df)} rows from DuckDB.")
    return df

def validate_and_convert(df: pd.DataFrame) -> List[Transaction]:
    """
    Converts a raw DataFrame into a list of strict Transaction objects.
    This runs every row against your 'Law' (schema.py).
    """
    print("Converting DataFrame to Transaction objects (Validation)...")
    
    # Convert DataFrame to a list of dictionaries (one dict per row)
    # 'records' mode gives us [{'id': 1, 'v1': 5}, {'id': 2, 'v1': 6}]
    raw_records = df.to_dict(orient="records")
    
    valid_transactions = []
    for record in raw_records:
        try:
            # This is where the magic happens.
            # We unpack the dict (**record) into the Transaction class.
            # The schema.py logic executes here for every single row.
            txn = Transaction(**record)
            valid_transactions.append(txn)
        except Exception as e:
            # In a real system, you would log this error to a file
            print(f"Failed to parse row: {e}")
            
    print(f"Successfully validated {len(valid_transactions)} transactions.")
    return valid_transactions

if __name__ == "__main__":
    # Point this to your actual file
    DB_PATH = "data/processed/transactions.duckdb" 
    
    # 1. Load raw data
    df = load_enriched_data(DB_PATH)
    
    # 2. Validate a small sample (first 1000) to be fast
    # We don't need to validate 1 million rows just to test the code.
    sample_df = df.head(1000)
    transactions = validate_and_convert(sample_df)
    
    # 3. Prove it worked
    print(f"\nSample Transaction 0:")
    print(transactions[0])
