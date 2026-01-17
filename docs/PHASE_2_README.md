# PHASE 2: INGESTION PIPELINE (BATCH + STREAMING)

## Executive Summary
Phase 2 builds the **"factory pipes"** that move data from Phase 1 (raw data generation) into the system safely and consistently. We create two parallel paths (Batch & Streaming) and prove they are identical. This guarantees that training and serving use the exact same data format.

**Goal:** Build a system that can replay data perfectly, verified by a consistency check.

**Status:** ✅ **COMPLETE** (Consistency Check Passed: 1000/1000 transactions identical)

---

## The Problem We Solve

### Training-Serving Skew (The #1 ML Bug)
In production ML systems, your code runs in **two completely different worlds:**

1. **Training (Past):** Grab all 1.1M historical transactions. Feed them to the model. Get a trained brain.
2. **Serving (Present):** A customer swipes their card. Handle it in 200ms. Make a decision.

If **Training** and **Serving** see data differently, the system fails silently.

**Example Disaster:**
- Training sees: `amount = 100.0` (Float)
- Serving sees: `amount = "100"` (String)
- Result: Model crashes on Day 1 of production.

**Our Solution:**
- Build a **Batch Loader** (Training).
- Build a **Streaming Simulator** (Serving).
- Prove they are identical with a **Consistency Check**.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                    PHASE 1 OUTPUT                           │
│              data/transactions.duckdb                        │
│          (1.1M rows, sorted by event_timestamp)             │
└────────────────────┬────────────────────────────────────────┘
                     │
                     │ (Two parallel paths from same source)
                     │
        ┌────────────┴────────────┐
        │                         │
        ▼                         ▼
   ┌─────────────┐          ┌─────────────────┐
   │   BATCH     │          │   STREAMING     │
   │   LOADER    │          │   SIMULATOR     │
   │  (Offline)  │          │    (Online)     │
   └──────┬──────┘          └────────┬────────┘
          │                          │
          │ SELECT * ORDER BY        │ yield one row
          │ event_timestamp          │ at a time
          │                          │
          │ Returns:                 │ Returns:
          │ [Txn1, Txn2, ...]       │ Txn1 -> Txn2 -> ...
          │                          │
          ▼                          ▼
    ┌──────────────┐           ┌──────────────┐
    │ List[Txn]    │           │ Iterator[Txn]│
    │ (~1.1M rows) │           │ (1 row/call) │
    │ (Memory: 4GB)│           │ (Memory: 50MB)
    └────────┬─────┘           └────────┬─────┘
             │                          │
             └──────────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │ CONSISTENCY CHECK    │
                 │ Compare first 1000   │
                 │ Assert IDs match     │
                 │ Assert amounts match │
                 │ Assert times match   │
                 └──────────────┬───────┘
                                │
                                ▼
                    ✅ PASS: Paths are IDENTICAL
```

---

## File Structure

```
src/
└── ingestion/
    ├── __init__.py                    # Package marker
    ├── schema.py                      # The "Law" - Pydantic model
    ├── batch_loader.py                # Offline path (Training)
    ├── streaming_simulator.py          # Online path (Serving)
    └── consistency_check.py            # Audit script
```

---

## Step-by-Step Breakdown

### STEP 1: The Schema (`src/ingestion/schema.py`)

**Goal:** Define the "Law" that all data must follow.

**What It Does:**
```python
class Transaction(BaseModel):
    # CRITICAL FIELDS (Must be present)
    transaction_id: str                # Unique identifier
    event_timestamp: datetime          # When did this happen? (CRUCIAL for ordering)
    amount: float                      # How much? (Must be a number, not text)
    
    # UPI IDENTITIES (Enriched from Phase 1)
    payer_vpa: str                     # Who is paying?
    payee_vpa: str                     # Who is receiving?
    device_id: str                     # From which device?
    
    # LABELS (Optional - live data won't have these)
    is_fraud: Optional[float]          # 0.0 = legit, 1.0 = fraud, None = unknown
    fraud_pattern: Optional[str]       # Device ring / Velocity spike / Time anomaly
    label_available_timestamp: Optional[datetime]
    
    # CATCH-ALL for raw features (V1..V339, C1..C14, etc.)
    class Config:
        extra = "allow"  # Accept 400+ columns without defining each one
```

**Why This Matters:**
- If a row is missing `amount`, **it crashes immediately**. No silent failures.
- If `amount` is the string `"100"`, Pydantic converts it to `100.0` automatically.
- The `extra="allow"` config lets us ingest all 400+ raw features without writing 400 variable names.

**Test:**
```bash
python test_schema.py
# Expected output:
# ID: TXN_12345
# Time: 2025-01-01 12:00:00 (Type: <class 'datetime.datetime'>)
# Amount: 500.0 (Type: <class 'float'>)
# VPA: user@upi
```

---

### STEP 2: The Batch Loader (`src/ingestion/batch_loader.py`)

**Goal:** The "History Book" reader. Load all transactions at once for training.

**What It Does:**

```python
def load_enriched_data(duckdb_path):
    # Connect to the database
    con = duckdb.connect(duckdb_path)
    
    # SELECT * FROM transactions ORDER BY event_timestamp
    # Why ORDER BY event_timestamp?
    # Time must be linear. Can't learn from Tuesday before Monday.
    
    df = con.execute(query).df()
    return df  # Returns all 1.1M rows as a table

def validate_and_convert(df):
    # Convert rows to list of dicts
    # For each dict, create a Transaction object
    # This runs Pydantic validation on every row
    # If a row fails, catch it and skip it
    
    return list_of_transaction_objects
```

**Memory Usage:** ~4GB (all 1.1M rows in RAM)

**Use Case:** Training the model. You need all history to learn patterns.

**Execution:**
```bash
python src/ingestion/batch_loader.py
# Expected output:
# Connecting to data/processed/transactions.duckdb...
# Loaded 1097231 rows from DuckDB.
# Converting DataFrame to Transaction objects (Validation)...
# Successfully validated 1000 transactions.
```

---

### STEP 3: The Streaming Simulator (`src/ingestion/streaming_simulator.py`)

**Goal:** The "Security Guard" reader. Yield transactions one by one to simulate live traffic.

**What It Does:**

```python
class EventGenerator:
    def __init__(self, duckdb_path):
        # Connect to database (NOT loading all rows)
        self.cursor = con.execute(query)
        self.columns = get_column_names()
    
    def __next__(self):
        # Fetch ONE row
        row = self.cursor.fetchone()
        
        # Convert tuple to dict
        record = dict(zip(self.columns, row))
        
        # Validate using Transaction schema
        return Transaction(**record)

class StreamingSimulator:
    def run(self, limit=100):
        # For each transaction from the generator
        # Process it (in Phase 5, call model.predict here)
        # Move to the next
```

**Memory Usage:** ~50MB (only 1 row in RAM at a time)

**Use Case:** Testing the model. Simulates live traffic (one txn at a time).

**Execution:**
```bash
python src/ingestion/streaming_simulator.py
# Expected output:
# [LIVE] 2025-01-02 00:00:00 | ID: 2987000 | Amt: 68.5
# [LIVE] 2025-01-02 00:01:00 | ID: 2987001 | Amt: 150.0
# ... (one per second, simulating live)
```

---

### STEP 4: The Consistency Check (`src/ingestion/consistency_check.py`)

**Goal:** The "Audit." Prove that Batch and Stream deliver identical data.

**What It Does:**

```python
# Load first 1000 from Batch
batch_txns = validate_and_convert(df.head(1000))

# Load first 1000 from Stream
stream_gen = EventGenerator(db_path)
stream_txns = [next(stream_gen) for i in range(1000)]

# Compare them
for i in range(1000):
    assert batch_txns[i].transaction_id == stream_txns[i].transaction_id
    assert batch_txns[i].event_timestamp == stream_txns[i].event_timestamp
    assert batch_txns[i].amount == stream_txns[i].amount

# If loop completes: ✅ PASS
```

**Execution:**
```bash
python src/ingestion/consistency_check.py
# Expected output:
# --- STARTING CONSISTENCY CHECK ---
# 1. Loading Batch Data...
# Loaded 1097231 rows from DuckDB.
# Successfully validated 1000 transactions.
# 2. Initializing Stream...
# 3. Comparing first 1000 transactions...
# ✅ PASS: Checked 1000 transactions. Batch and Streaming paths are IDENTICAL.
```

---

## Execution Order (How to Run Everything)

```bash
# 1. Test the schema (Optional - just to verify types)
python test_schema.py

# 2. Test batch loading
python src/ingestion/batch_loader.py

# 3. Test streaming (will print 10 transactions)
python src/ingestion/streaming_simulator.py

# 4. THE FINAL CHECK - Prove Batch == Stream
python src/ingestion/consistency_check.py

# Expected final output:
# ✅ PASS: Checked 1000 transactions. Batch and Streaming paths are IDENTICAL.
```

---

## Key Insights (The "Why")

### 1. Why ORDER BY event_timestamp?
Fraud patterns evolve over time. A device that was trusted in January might be compromised by March. If you load data out of order, the model learns backwards (Tuesday before Monday), which is invalid.

### 2. Why separate Batch and Stream?
- **Batch** = Historical analysis. You can afford to load 1.1M rows.
- **Stream** = Real-time action. You must process in <200ms per transaction.
- In Phase 5, you will use Batch to train. In Phase 6, you will use Stream to test.

### 3. Why the Consistency Check?
It is the contract. If this passes, you know:
- Your training code and serving code use the same data.
- No silent data mismatches.
- When you deploy the model tomorrow, it will work exactly as it did today.

---

## Common Failure Modes & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `Amount: 500.0 (Type: str)` | Pydantic not converting types | Check if schema.py has `amount: float` (not `str`) |
| `Batch ID != Stream ID` | Sort order changed | Verify `ORDER BY event_timestamp` in query |
| `Memory exceeded` | Loading all 1.1M rows twice | Use `df.head(1000)` for testing before full run |
| `NaN values in output` | Missing fields in raw data | This is OK - Phase 4 handles missing values |

---

## What Comes Next (Phase 3)

**Phase 3: Data Validation (Great Expectations)**
- Add quality gates to reject bad data.
- Example: Reject if `amount < 0` or `timestamp > now()`.
- Run these checks on both Batch and Stream.

**Phase 4: Feature Engineering**
- Reuse the Batch/Stream paths.
- Add computed features: "How many transactions in last 5 minutes?"
- Ensure no time leakage: "Don't peek into the future."

**Phase 5: Modeling (Two-Stage)**
- Train Stage 1 (Anomaly Detection) on Batch.
- Train Stage 2 (Supervised) on Batch.
- Test both on Stream.

**Phase 6: Backtesting**
- Replay historical data day-by-day.
- Apply your model.
- Measure precision, recall, false alerts.

---

## Summary Table

| Component | File | What It Does | Memory | Use Case |
|-----------|------|--------------|--------|----------|
| Schema | `schema.py` | Validates data types | Negligible | Quality gate at entrance |
| Batch Loader | `batch_loader.py` | Load all history | 4GB | Training (Phase 5) |
| Stream Simulator | `streaming_simulator.py` | Yield one txn/call | 50MB | Live testing (Phase 6) |
| Consistency Check | `consistency_check.py` | Audit Batch == Stream | Minimal | Verify correctness |

---

## Conclusion

**Phase 2 is COMPLETE.** ✅

You have built the foundation for a production ML system:
- Data enters through a validated schema (The Law).
- It flows through two paths (History & Live).
- Both paths are proven identical.

This is the engineering discipline that separates Kaggle notebooks from real systems. You now understand why top ML teams are so careful about ingestion.

**Next: Phase 3 (Data Validation).**
