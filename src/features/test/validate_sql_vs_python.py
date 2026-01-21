import duckdb
import pandas as pd
from src.features.offline_builder import build_features_batch

def validate_sql_vs_python():
    print("🔍 Validating SQL vs Python Feature Logic...")

    con = duckdb.connect("data/processed/transactions.duckdb")
    df = con.execute("SELECT * FROM transactions ORDER BY event_timestamp LIMIT 500").df()
    con.close()

    df["transaction_id"] = df["transaction_id"].astype(str)
    df["event_timestamp"] = pd.to_datetime(df["event_timestamp"])
    df["label_available_timestamp"] = pd.to_datetime(df["label_available_timestamp"])

    # Python truth
    python_df = build_features_batch(df)

    # SQL truth
    con = duckdb.connect("data/processed/full_features.duckdb")
    sql_df = con.execute("""
        SELECT * FROM features 
        ORDER BY event_timestamp 
        LIMIT 500
    """).df()
    con.close()

    pd.testing.assert_frame_equal(
        python_df.sort_index(axis=1),
        sql_df.sort_index(axis=1),
        check_dtype=False
    )

    print("✅ SQL matches Python truth engine")

if __name__ == "__main__":
    validate_sql_vs_python()
