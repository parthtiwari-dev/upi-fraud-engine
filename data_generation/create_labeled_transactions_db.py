import duckdb
import os

def create_labeled_transactions_db():
    src_db = "data/processed/transactions.duckdb"
    out_db = "data/processed/transactions_labeled.duckdb"

    os.makedirs("data/processed", exist_ok=True)

    print("🚀 Creating labeled-only transactions database...")
    print(f"Source: {src_db}")
    print(f"Output: {out_db}")

    con = duckdb.connect(src_db)

    # Sanity check
    total = con.execute("SELECT COUNT(*) FROM transactions").fetchone()[0]
    labeled = con.execute("""
        SELECT COUNT(*) 
        FROM transactions 
        WHERE is_fraud IN (0, 1)
    """).fetchone()[0]

    print(f"Total rows in source: {total:,}")
    print(f"Labeled rows (0/1): {labeled:,}")

    assert labeled > 500_000, "Something is wrong, too few labeled rows"

    # Create labeled-only table, STRICTLY ordered by time
    query = """
    CREATE TABLE labeled_transactions AS
    SELECT *
    FROM transactions
    WHERE is_fraud IN (0, 1)
    ORDER BY event_timestamp;
    """

    print("⏳ Writing labeled transactions...")
    con.execute(query)

    # Write to a new DuckDB file
    con.execute(f"ATTACH '{out_db}' AS out_db")
    con.execute("""
        CREATE TABLE out_db.transactions AS
        SELECT * FROM labeled_transactions
    """)

    rows = con.execute("SELECT COUNT(*) FROM labeled_transactions").fetchone()[0]
    print(f"✅ Created labeled transaction DB with {rows:,} rows")

    con.close()

if __name__ == "__main__":
    create_labeled_transactions_db()
