"""
Quick diagnostic to inspect DuckDB databases
"""

import duckdb
from pathlib import Path

print("=" * 70)
print("DATABASE INSPECTION")
print("=" * 70)

# Check both databases
databases = [
    'data/processed/full_features.duckdb',
    'data/processed/features.duckdb'
]

for db_path in databases:
    if not Path(db_path).exists():
        print(f"\n❌ {db_path} - NOT FOUND")
        continue
    
    print(f"\n{'=' * 70}")
    print(f"📂 DATABASE: {db_path}")
    print(f"{'=' * 70}")
    
    con = duckdb.connect(db_path, read_only=True)
    
    # List all tables
    tables = con.execute("SHOW TABLES").fetchall()
    print(f"\n📊 Tables found: {len(tables)}")
    
    for table in tables:
        table_name = table[0]
        print(f"\n  ✅ Table: {table_name}")
        
        # Get schema
        schema = con.execute(f"DESCRIBE {table_name}").fetchdf()
        print(f"     Columns ({len(schema)} total):")
        for idx, row in schema.iterrows():
            print(f"       - {row['column_name']:30} {row['column_type']}")
        
        # Get row count
        count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
        print(f"     Rows: {count:,}")
        
        # Show sample (first row)
        sample = con.execute(f"SELECT * FROM {table_name} LIMIT 1").fetchdf()
        print(f"     Sample columns: {list(sample.columns)[:10]}...")
    
    con.close()

print("\n" + "=" * 70)
print("INSPECTION COMPLETE")
print("=" * 70)
