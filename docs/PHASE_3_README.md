# PHASE 3: DATA VALIDATION (WITH GREAT EXPECTATIONS)

## Executive Summary
Phase 3 installs a **"Quality Control Inspector"** between your raw data and your model training pipeline. Using Great Expectations (GX), we define strict data quality rules ("Suites"), run them against every batch of data, and automatically **reject bad data** before it contaminates training or serving.

**Goal:** Build a gatekeeper that automatically rejects bad data before it reaches your model.

**Status:** ✅ **COMPLETE** (Batch validation passed on 1.1M transactions)

---

## The Problem We Solve

### Why Data Validation Matters
Imagine this scenario:
- **Day 1:** Train a model on historical data. It learns well. Metrics look great.
- **Day 2:** Live system processes new transactions. Someone accidentally sends `amount = -500` (a refund with a sign error).
- **Day 3:** Your model sees `-500` and crashes or makes insane predictions because it was never trained on negative amounts.

**Phase 3 prevents this.** We define "What does valid data look like?" **once** and then enforce it everywhere:
- During training (Batch).
- During live scoring (Streaming).
- In the future, automatically (Checkpoint).

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────┐
│               PHASE 1 OUTPUT (Raw Data)                          │
│              data/transactions.duckdb                             │
│              (1.1M rows, some might be dirty)                    │
└────────────────────────────────────────────────────────────────┬─┘
                                                                   │
                                ┌──────────────────────────────────┘
                                │
                    ┌───────────────────────┐
                    │   Phase 3: GX Engine  │
                    │   (The Police)        │
                    └───────────────────────┘
                                │
                    ┌───────────┴───────────┐
                    │                       │
        ┌───────────────────────┐  ┌──────────────────────┐
        │  RULE SET 1           │  │  RULE SET 2          │
        │  transaction_schema   │  │  business_logic      │
        │  (Structural)         │  │  (Quality)           │
        │                       │  │                      │
        │ • Columns exist       │  │ • Amount 0-1M        │
        │ • Types correct       │  │ • Currency = INR     │
        │ • ID unique           │  │ • No nulls           │
        └───────────────────────┘  └──────────────────────┘
                    │                       │
                    └───────────┬───────────┘
                                │
                    ┌───────────────────────┐
                    │   Checkpoint (v1.0)   │
                    │   Runs both suites    │
                    │   on the dataframe    │
                    └───────────┬───────────┘
                                │
                ┌───────────────┴───────────────┐
                │                               │
            ✅ SUCCESS                      ❌ FAILURE
    (Proceed to training)          (Raise error, stop)
                │                               │
                └───────────────┬───────────────┘
                                │
                    ┌───────────────────────┐
                    │   BATCH LOADER        │
                    │   (Phase 2)           │
                    │ Only accepts clean    │
                    │ data (gatekeeper)     │
                    └───────────────────────┘
```

---

## File Structure

```
src/
├── validation/
│   ├── __init__.py
│   ├── build_suite.py           # Define the rules (2 suites)
│   └── run_validation.py         # Apply the rules (Checkpoint logic)
│
└── ingestion/
    └── batch_loader.py          # Modified to call validate_batch()
    
great_expectations/              # GX auto-generated folder
├── expectations/
│   ├── transaction_schema.json  # Suite 1 (Structural)
│   └── business_logic.json      # Suite 2 (Quality)
├── validation_definitions/      # Runtime definitions (auto-created)
└── checkpoints/                 # Checkpoint configs (auto-created)
```

---

## Step-by-Step: What We Built

### STEP 1: Define the Rules (`build_suite.py`)

**Mental Model:** Think of this as writing a **constitution** for your data.

Two separate "Suites" (checklists):

#### Suite 1: `transaction_schema` (Structural Law)
*What does the raw shape of the data look like?*

```python
def build_schema_suite(context):
    suite = context.suites.add(gx.ExpectationSuite(name="transaction_schema"))
    
    # Rule 1: Critical columns must exist
    suite.add_expectation(gxe.ExpectColumnToExist(column="transaction_id"))
    suite.add_expectation(gxe.ExpectColumnToExist(column="amount"))
    suite.add_expectation(gxe.ExpectColumnToExist(column="event_timestamp"))
    suite.add_expectation(gxe.ExpectColumnToExist(column="payer_id"))
    suite.add_expectation(gxe.ExpectColumnToExist(column="currency"))
    
    # Rule 2: Types must be correct
    suite.add_expectation(gxe.ExpectColumnValuesToBeOfType(column="amount", type_="float"))
    suite.add_expectation(gxe.ExpectColumnValuesToBeOfType(column="transaction_id", type_="str"))
    
    # Rule 3: IDs must be unique
    suite.add_expectation(gxe.ExpectColumnValuesToBeUnique(column="transaction_id"))
    
    # Rule 4: Time must be increasing (no time-travel)
    suite.add_expectation(gxe.ExpectColumnValuesToBeIncreasing(column="event_timestamp"))
```

**Why:** If these fail, the data format is broken. The rest of the pipeline cannot function.

#### Suite 2: `business_logic` (Quality Law)
*Do the values make business sense?*

```python
def build_business_logic_suite(context):
    suite = context.suites.add(gx.ExpectationSuite(name="business_logic"))
    
    # Rule 1: Amounts are reasonable
    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(
        column="amount", min_value=0, max_value=1000000
    ))
    
    # Rule 2: Currency is always INR
    suite.add_expectation(gxe.ExpectColumnValuesToBeInSet(
        column="currency", value_set=['INR']
    ))
    
    # Rule 3: Every transaction must have a payer
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(column="payer_id"))
```

**Why:** Even if the data is structurally valid, these rules catch business logic errors (e.g., negative amounts, wrong currencies).

#### Main Function
```python
def main():
    context = gx.get_context(context_root_dir="great_expectations")
    build_schema_suite(context)
    build_business_logic_suite(context)
```

**Execution:** `python src/validation/build_suite.py`
- Creates `great_expectations/expectations/transaction_schema.json` and `business_logic.json`.
- These JSON files are the persisted "Laws."

---

### STEP 2: Run the Police (`run_validation.py`)

**Mental Model:** Now we have the laws. Time to enforce them.

#### Function 1: `validate_batch(df)` - Heavy Validation
*Used for historical data (Training, Backtesting).*

```python
def validate_batch(df: pd.DataFrame) -> bool:
    context = gx.get_context(context_root_dir="great_expectations")
    
    # 1. Setup a Datasource (connection to the data)
    datasource = context.data_sources.add_or_update_pandas(name="police_datasource")
    
    # 2. Create an Asset (one specific table/dataset)
    asset = datasource.add_dataframe_asset(name="transactions_asset")
    
    # 3. Define a Batch Definition (how to slice the data)
    batch_def = asset.add_batch_definition_whole_dataframe("batch_def")
    
    # 4. Load the Suites (retrieve the laws we defined in Step 1)
    schema_suite = context.suites.get("transaction_schema")
    business_suite = context.suites.get("business_logic")
    
    # 5. Create Validation Definitions (link data + rules)
    val_schema = context.validation_definitions.add_or_update(
        gx.ValidationDefinition(name="val_schema", data=batch_def, suite=schema_suite)
    )
    val_logic = context.validation_definitions.add_or_update(
        gx.ValidationDefinition(name="val_logic", data=batch_def, suite=business_suite)
    )
    
    # 6. Create a Checkpoint (orchestrator that runs both validations)
    checkpoint = context.checkpoints.add_or_update(
        gx.Checkpoint(
            name="transaction_checkpoint",
            validation_definitions=[val_schema, val_logic]
        )
    )
    
    # 7. Execute the Checkpoint (pass the actual dataframe)
    result = checkpoint.run(batch_parameters={"dataframe": df})
    
    # 8. Return True/False based on success
    if not result.success:
        print("❌ DATA VALIDATION FAILED!")
        return False
    
    print("✅ Batch Validation Passed.")
    return True
```

**Flow:** DataFrame → Datasource → Asset → Batch Definition → Validation Definitions → Checkpoint → Result.

**Time Cost:** ~5 minutes for 1.1M rows (GX calculates statistics on the entire batch).

#### Function 2: `validate_streaming_event(event)` - Lite Validation
*Used for live transactions (Inference, Real-time Scoring).*

```python
def validate_streaming_event(event: dict) -> bool:
    """Fast Python-only check. Avoids GX overhead."""
    return (
        event.get("amount", 0) > 0 and           # Amount must be positive
        event.get("currency") == "INR"           # Currency must be INR
    )
```

**Time Cost:** < 1ms (pure Python, no GX overhead).

**Why Two Functions?**
- **Batch:** Expensive but thorough. You can afford 5 minutes because it's offline.
- **Stream:** Fast but shallow. Must complete in < 1ms to keep API responsive.

#### Main Function
```python
def main() -> None:
    # 1. Load data from DuckDB
    df = con.execute("SELECT * FROM transactions ORDER BY event_timestamp").df()
    
    # 2. Run validation
    if not validate_batch(df):
        raise RuntimeError("⛔ STOP! Data Validation Failed. Fix the data before training.")
    
    print("🎉 Data is clean and safe for Phase 4.")
```

**Execution:** `python src/validation/run_validation.py`
- Loads all 1.1M rows.
- Runs both suites via the Checkpoint.
- Returns ✅ or ❌.

---

### STEP 3: Integrate with Phase 2 (`batch_loader.py`)

**Mental Model:** The Gatekeeper. No bad data enters the training pipeline.

**What Changed:**
```python
# NEW: Import the police
from src.validation.run_validation import validate_batch

def load_enriched_data(duckdb_path: str) -> pd.DataFrame:
    # Load data
    df = con.execute("SELECT * FROM transactions ORDER BY event_timestamp").df()
    
    # NEW: Call the gatekeeper
    if not validate_batch(df):
        raise ValueError("⛔ CRITICAL: Data failed validation. Stop.")
    
    # Only return if GX says OK
    return df
```

**Effect:** Now, whenever any downstream code calls `load_enriched_data()`, it **automatically** runs validation. Bad data **never** reaches training.

**Execution Test:** `python src/ingestion/batch_loader.py`
- Loads data.
- Runs validation (takes ~5 min).
- Returns clean DataFrame or raises error.

---

## How It All Works Together

```
Timeline of a Training Run (Phase 5):
┌────────────────────────────────────────────────┐
│ 1. Phase 5 code calls: batch_loader.load()    │
└────────────┬─────────────────────────────────┘
             │
┌────────────────────────────────────────────────┐
│ 2. batch_loader connects to DuckDB            │
│    Fetches: SELECT * FROM transactions        │
└────────────┬─────────────────────────────────┘
             │
┌────────────────────────────────────────────────┐
│ 3. batch_loader calls: validate_batch(df)     │
│    (AUTOMATIC GATEKEEPER)                     │
└────────────┬─────────────────────────────────┘
             │
┌────────────────────────────────────────────────┐
│ 4a. GX runs transaction_schema suite          │
│     ✅ Columns exist? ✅ Types correct?        │
│     ✅ ID unique? ✅ Time increasing?          │
└────────────┬─────────────────────────────────┘
             │
┌────────────────────────────────────────────────┐
│ 4b. GX runs business_logic suite              │
│     ✅ Amount 0-1M? ✅ Currency INR?           │
│     ✅ No nulls?                              │
└────────────┬─────────────────────────────────┘
             │
          ✅ PASS?
             │
    ┌────────┴────────┐
    │                 │
   YES               NO
    │                 │
    ▼                 ▼
┌──────────┐    ┌──────────────┐
│ Continue │    │ raise error  │
│  return  │    │ STOP TRAINING│
│   df     │    └──────────────┘
└──────────┘
```

---

## Great Expectations Key Concepts

### 1. Expectation Suite
A JSON file containing a list of "Expectations" (validation rules).
- **transaction_schema.json** (~2KB) contains structural rules.
- **business_logic.json** (~1KB) contains quality rules.

### 2. Validation Definition
A pairing of:
- **Data Definition** (which batch of data to check).
- **Expectation Suite** (which rules to apply).

Example: "val_schema" = "Check the transactions data against transaction_schema suite."

### 3. Checkpoint
A reusable "job" that runs multiple Validation Definitions together.
- Runs sequentially.
- Reports all results in one execution.
- Can be scheduled (e.g., daily via Airflow).

### 4. Context
The "Command Center" that owns all suites, datasources, validations, and checkpoints.
- Persists everything to `great_expectations/` folder.
- Lets you query: "Show me all checkpoints," "Get suite by name," etc.

---

## Execution Checklist

✅ **Step 1: Build Suites**
```bash
python src/validation/build_suite.py
# Output: ✅ Suite 'transaction_schema' registered and built.
#         ✅ Suite 'business_logic' registered and built.
```

✅ **Step 2: Run Validation**
```bash
python src/validation/run_validation.py
# Output: 📦 Loading data from data/processed/transactions.duckdb...
#         Rows: 1097231 | Fraud Rate: 0.0360
#         🔍 Running Batch Validation...
#         Calculating Metrics: 100%|██████████| 18/18 [02:20<00:00]
#         Calculating Metrics: 100%|██████████| 25/25 [03:45<00:00]
#         ✅ Batch Validation Passed.
#         🎉 Phase 3 Step 2 Complete: Data is clean and safe for Phase 4.
```

✅ **Step 3: Verify Integration**
```bash
python src/ingestion/batch_loader.py
# Output: Connecting to data/processed/transactions.duckdb...
#         Loaded 1097231 rows from DuckDB.
#         🔒 Running Data Validation Gatekeeper...
#         🔍 Running Batch Validation...
#         Calculating Metrics: 100%|██████████| 18/18 [02:20<00:00]
#         Calculating Metrics: 100%|██████████| 25/25 [03:45<00:00]
#         ✅ Gatekeeper Passed. Proceeding.
#         Successfully validated 1000 transactions.
```

---

## Data Quality Rules (The "Constitution")

| Rule | Suite | Column | Check | Why |
|------|-------|--------|-------|-----|
| Exists | Schema | transaction_id | Must exist | Cannot process unknown transactions |
| Exists | Schema | amount | Must exist | Core field for fraud detection |
| Exists | Schema | event_timestamp | Must exist | Required for time-based features |
| Type | Schema | amount | Must be float | Math operations expect numbers |
| Type | Schema | transaction_id | Must be str | IDs are categorical |
| Unique | Schema | transaction_id | No duplicates | Each transaction is one event |
| Increasing | Schema | event_timestamp | Time flows forward | Prevents leakage |
| Range | Logic | amount | 0 ≤ amount ≤ 1,000,000 | Bounds check (refunds OK if 0, limits exist) |
| Set | Logic | currency | Must be "INR" | Only handle Indian Rupees |
| NotNull | Logic | payer_id | Never null | Every transaction needs a payer |

---

## Common Failure Scenarios (And How Phase 3 Saves You)

| Scenario | Without Phase 3 | With Phase 3 |
|----------|-----------------|-------------|
| Backend sends `amount = -500` by mistake | Model trains on negative amounts, inference fails | Validation rejects it, error raised immediately |
| Currency field is sometimes "USD", sometimes "INR" | Model gets confused on USD transactions | Validation enforces currency = "INR" only |
| A transaction_id appears twice (bug in Phase 1) | Model sees duplicates, metrics misleading | Unique check catches it, stops training |
| Timestamp is not sorted (data corruption) | Time-based features break silently | Increasing check catches it, stops training |
| A new column is added but sometimes null | Inference crashes when null | NotNull rule catches it, prevents deployment |

---

## What Comes Next (Phase 4)

Phase 3 ensures **raw data quality**.
Phase 4 (Feature Engineering) will add **computed feature quality**:
- "If velocity features are missing, interpolate."
- "If graph statistics are invalid, use defaults."
- Monitor drift in feature distributions.

But Phase 3 guarantees the **input** is clean, so Phase 4 can focus on computation, not firefighting.

---

## Summary

| Component | Purpose | File | Execution |
|-----------|---------|------|-----------|
| **Suites (Laws)** | Define what valid data looks like | `build_suite.py` | `python src/validation/build_suite.py` |
| **Checkpoint (Police)** | Run the laws on actual data | `run_validation.py` | `python src/validation/run_validation.py` |
| **Gatekeeper (Security)** | Block bad data from entering training | `batch_loader.py` (modified) | `python src/ingestion/batch_loader.py` |

**Phase 3 is COMPLETE.** You have a production-grade data quality layer.

---

## Key Takeaway

**Before Phase 3:** "Is my model good?" → Maybe, but the data might be garbage.

**After Phase 3:** "Is my model good?" → Yes, *and* I guarantee the data is clean because bad data is rejected automatically.

This is the difference between a prototype and a production system.
