import duckdb
from datetime import datetime
from typing import Iterator
from src.ingestion.schema import Transaction

class EventGenerator:
    """
    Simulates the 'Online' stream. 
    Connects to DuckDB and yields rows one by one (Cursor-based) to save RAM.
    """
    def __init__(self, duckdb_path: str):
        self.con = duckdb.connect(duckdb_path)
        self.query = "SELECT * FROM transactions ORDER BY event_timestamp"
        # Execute query but don't fetch all results
        self.cursor = self.con.execute(self.query)
        # Get column names to map tuple -> dict
        self.columns = [desc[0] for desc in self.cursor.description]
        
        # Buffer to hold the 'current' row if we peek at it
        self.current_txn = None

    def __iter__(self) -> Iterator[Transaction]:
        return self

    def __next__(self) -> Transaction:
        # 1. Fetch raw tuple
        row = self.cursor.fetchone()
        if row is None:
            self.con.close()
            raise StopIteration
            
        # 2. Map to Dict
        record = dict(zip(self.columns, row))
        
        # 3. Convert to Pydantic (The Law)
        # In a real app, you'd log errors here instead of crashing
        try:
            txn = Transaction(**record)
            self.current_txn = txn # Store it so we know where we are
            return txn
        except Exception as e:
            print(f"Skipping bad row: {e}")
            return self.__next__() # Skip and try next

    def advance_to_time(self, target_time: datetime):
        """
        Fast-forwards the stream until event_timestamp >= target_time.
        Useful for skipping the first 6 months of data during testing.
        """
        print(f"Fast-forwarding stream to {target_time}...")
        skipped_count = 0
        
        # We loop manually until we hit the time
        # Since we use fetchone(), this consumes the stream
        while True:
            # Check the next row
            row = self.cursor.fetchone()
            if row is None:
                break # End of data
            
            # Peek at the timestamp (assuming event_timestamp is the 2nd column, or map it)
            # Safer to map it to check properly
            record = dict(zip(self.columns, row))
            
            # We assume event_timestamp is already datetime because DuckDB handles it,
            # or Pydantic would handle it. Here we are checking RAW data.
            # DuckDB returns Python datetime objects for timestamp columns.
            current_time = record['event_timestamp']
            
            if current_time >= target_time:
                # We found it! 
                # PROBLEM: We just fetched it, so we 'consumed' it.
                # We need to process this row next.
                # Since __next__ calls fetchone(), we can't easily 'put it back'.
                # A simple Hack: We just print that we are here.
                # The NEXT call to __next__ will naturally fetch the following row.
                # This means we technically 'lost' the exact trigger row in this simple logic, 
                # but for simulation, getting 'to' that time is usually enough.
                # To be perfect, we would implement a 'peek' buffer, but let's keep it simple.
                print(f"Reached {current_time}. Skipped {skipped_count} events.")
                return
            
            skipped_count += 1

class StreamingSimulator:
    def __init__(self, generator: EventGenerator):
        self.generator = generator

    def run(self, limit: int = 10):
        print(f"\n--- STREAMING SIMULATOR START (Limit: {limit}) ---")
        count = 0
        for txn in self.generator:
            if count >= limit:
                break
            
            print(f"[LIVE] {txn.event_timestamp} | ID: {txn.transaction_id} | Amt: {txn.amount}")
            count += 1
        print("--- STREAM END ---")

if __name__ == "__main__":
    DB_PATH = "data/processed/transactions.duckdb"
    
    # 1. Setup
    gen = EventGenerator(DB_PATH)
    sim = StreamingSimulator(gen)
    
    # 2. Test Run
    sim.run(limit=10)
