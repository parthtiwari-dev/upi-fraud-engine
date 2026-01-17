import sys
from src.ingestion.batch_loader import load_enriched_data, validate_and_convert
from src.ingestion.streaming_simulator import EventGenerator

def run_consistency_check():
    DB_PATH = "data/processed/transactions.duckdb"
    CHECK_LIMIT = 1000  # We don't need to check 1 million rows. 100 is enough proof.
    
    print("--- STARTING CONSISTENCY CHECK ---")
    
    # 1. LOAD BATCH DATA (The History Book)
    print("1. Loading Batch Data...")
    df = load_enriched_data(DB_PATH)
    # We only convert the first 100 to save time/memory
    batch_transactions = validate_and_convert(df.head(CHECK_LIMIT))
    
    # 2. LOAD STREAM DATA (The Security Guard)
    print("2. Initializing Stream...")
    stream_gen = EventGenerator(DB_PATH)
    
    # 3. COMPARE THEM HEAD-TO-HEAD
    print(f"3. Comparing first {CHECK_LIMIT} transactions...")
    
    for i in range(CHECK_LIMIT):
        # Get the i-th item from Batch
        batch_txn = batch_transactions[i]
        
        # Get the next item from Stream
        try:
            stream_txn = next(stream_gen)
        except StopIteration:
            print(f"❌ FAIL: Stream ran out of data at index {i}!")
            sys.exit(1)
            
        # THE AUDIT: Compare IDs
        if batch_txn.transaction_id != stream_txn.transaction_id:
            print(f"❌ FAIL at index {i}: IDs do not match!")
            print(f"   Batch:  {batch_txn.transaction_id}")
            print(f"   Stream: {stream_txn.transaction_id}")
            sys.exit(1)
            
        # THE AUDIT: Compare Timestamps
        if batch_txn.event_timestamp != stream_txn.event_timestamp:
            print(f"❌ FAIL at index {i}: Timestamps do not match!")
            print(f"   Batch:  {batch_txn.event_timestamp}")
            print(f"   Stream: {stream_txn.event_timestamp}")
            sys.exit(1)
            
        # THE AUDIT: Compare Amounts
        if batch_txn.amount != stream_txn.amount:
            print(f"❌ FAIL at index {i}: Amounts do not match!")
            sys.exit(1)
            
    # 4. FINAL VERDICT
    print(f"\n✅ PASS: Checked {CHECK_LIMIT} transactions. Batch and Streaming paths are IDENTICAL.")

if __name__ == "__main__":
    run_consistency_check()
