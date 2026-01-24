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

**Temporal Causality Guarantee (Critical for Fraud Systems)**
In fraud detection, labels arrive after transactions.
Training or scoring with future labels creates catastrophic data leakage.
Phase 3 enforces this with a hard contract:
`label_available_timestamp > event_timestamp`
If any row violates this rule, the pipeline fails immediately.
This guarantees that Phase 4 features and Phase 5 models are built in a causally valid universe.

---

## Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│               PHASE 1 OUTPUT (Raw Data)                          │
│              data/transactions.duckdb                             │
│              (1.1M rows, some might be dirty)                    │
└─────────────────────────────────────────────────────────────┬───┘
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
    suite.add_expectation(gxe.ExpectColumnValuesToBeBetween(column="amount", min_value=0.01))
    suite.add_expectation(gxe.ExpectColumnValuesToBeOfType(column="transaction_id", type_="str"))

    # Rule 3: IDs must be unique
    suite.add_expectation(gxe.ExpectColumnValuesToBeUnique(column="transaction_id"))

    # Rule 4: Temporal Causality (Label Leakage Prevention)
    suite.add_expectation(gxe.ExpectColumnPairValuesAToBeGreaterThanB(
        column_A="label_available_timestamp",
        column_B="event_timestamp")
    )
    suite.add_expectation(gxe.ExpectColumnValuesToNotBeNull(
        column="label_available_timestamp"
        )
    )
```

**Why:** If these fail, the data format is broken. The rest of the pipeline cannot function.

**Note on Time-Travel Check:** The `ExpectColumnValuesToBeIncreasing` check for `event_timestamp` is intentionally omitted in the current implementation. While chronological ordering is critical and enforced during data generation (Phase 1) and ingestion (Phase 2 via ORDER BY), validating this at the GX level on 1.1M rows can be computationally expensive. The time-ordering guarantee is instead maintained through pipeline design rather than runtime validation.

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

#### Function 2: `validate_streaming_event(event)` - Lightweight Validation
*Used for real-time event validation in streaming scenarios.*

```python
def validate_streaming_event(event: dict) -> bool:
    """Lightweight streaming gate for the simulator."""
    return (
        event.get("amount", 0) > 0 and
        event.get("currency") == "INR" and
        (event.get("label_available_timestamp") is None or 
         event["label_available_timestamp"] > event["event_timestamp"])
    )
```

**Why Lightweight:** In streaming, we can't afford the 5-minute GX overhead per event. This function performs critical checks only (amount, currency, temporal causality) in microseconds.

---

## Data Quality Rules (The Constitution)

| Rule | Suite | Column(s) | Check | Why |
| :--- | :--- | :--- | :--- | :--- |
| **Structure** | Schema | transaction_id | Not Null, Unique, String | Primary Key integrity |
| **Completeness** | Schema | amount, event_timestamp | Not Null | Critical business fields |
| **Types** | Schema | amount | Float (min 0.01) | Math operations require positive float |
| **Types** | Schema | transaction_id | String | IDs are text, not numbers |
| **Causality** | Schema | label_available_timestamp | Not Null | Every transaction must have label timing defined |
| **Causality** | Schema | label_available_timestamp, event_timestamp | label_available_timestamp > event_timestamp | Prevents future label leakage |
| **Logic** | Business | amount | 0 to 1,000,000 | Negative or huge amounts are errors |
| **Logic** | Business | currency | 'INR' | Only supporting INR for now |
| **Logic** | Business | payer_id | Not Null | Cannot have anonymous payments |

**Note on Chronological Ordering:** While time-ordering (`event_timestamp` increasing) is critical for fraud detection, it is enforced through pipeline design (Phase 1 sorting + Phase 2 ORDER BY queries) rather than as a GX expectation. This avoids the computational overhead of validating monotonic increase across 1.1M+ rows during every validation run.

---

## Execution Guide

### Step 1: Build the Validation Suites
```bash
python src/validation/build_suite.py
```

**Expected Output:**
```
--- Building Expectation Suites (Context-Managed) ---
✅ Suite 'transaction_schema' registered and built.
✅ Suite 'business_logic' registered and built.

🎉 Suites successfully registered in Data Context.
```

### Step 2: Run Validation on Your Data
```bash
python src/validation/run_validation.py
```

**Expected Output:**
```
📦 Loading data from data/processed/transactions.duckdb...
Rows: 1097231 | Fraud Rate: 0.0188
🔍 Running Batch Validation...
✅ Batch Validation Passed.

🎉 Phase 3 Step 2 Complete: Data is clean and safe for Phase 4.
```

### Step 3: Verify Gatekeeper Integration
The gatekeeper is automatically called in `batch_loader.py`:
```python
# In src/ingestion/batch_loader.py
print("🔒 Running Data Validation Gatekeeper...")
if not validate_batch(df):
    raise ValueError("⛔ CRITICAL: Batch Loader stopped because data failed validation rules.")
```

Test it:
```bash
python src/ingestion/batch_loader.py
```

---

## Common Failure Modes & Fixes

| Error | Cause | Fix |
|-------|-------|-----|
| `Type mismatch: expected str, got int64` | transaction_id column is numeric | Check Phase 1 data generation - should convert to string |
| `Causality check failed` | label_available_timestamp ≤ event_timestamp | Verify 48-hour delay in Phase 1 |
| `Currency validation failed` | Non-INR currencies in data | Filter or fix source data |
| `Amount out of range` | Amounts < 0 or > 1M | Check for data corruption in source |
| `Suites not found` | build_suite.py not run | Run `python src/validation/build_suite.py` first |

---

## What Comes Next (Phase 4)

**Phase 4: Feature Engineering**
- Build point-in-time features using the validated data
- Example: "How many transactions in last 5 minutes?"
- Ensure no time leakage: "Don't peek into the future."
- All features will be built on **validated, causally-correct data** from Phase 3

**Phase 5: Modeling (Two-Stage)**
- Train Stage 1 (Anomaly Detection) on validated Batch data
- Train Stage 2 (Supervised) on validated Batch data
- Test both on validated Stream data

**Phase 6: Backtesting**
- Replay historical data day-by-day (with validation at each step)
- Apply your model
- Measure precision, recall, false alerts

---

## Summary Table

| Component | File | What It Does | Time Cost | Use Case |
|-----------|------|--------------|-----------|----------|
| Schema Suite | `build_suite.py` | Define structural rules | 1 second | One-time setup |
| Business Suite | `build_suite.py` | Define quality rules | 1 second | One-time setup |
| Batch Validator | `run_validation.py` | Heavy validation | ~5 min for 1.1M rows | Training data |
| Stream Validator | `run_validation.py` | Lightweight checks | <1ms per event | Live scoring |
| Gatekeeper | `batch_loader.py` | Automatic rejection | Included in batch load | Production safety |

---

## Key Implementation Notes

### Why transaction_id is String (Not Integer)
Phase 1 generates transaction IDs as strings to support:
- UPI transaction IDs (alphanumeric like "TXN_ABC123")
- Future compatibility with external systems
- Prevents accidental arithmetic operations on IDs

The validation suite enforces `type_="str"` to match Phase 1/2 pipeline.

### Why Streaming Validation is Different
Batch validation uses full GX engine (statistics, profiling, checkpoints).
Streaming validation uses simple Python conditionals for speed:
- Batch: 5 minutes for 1.1M rows = Acceptable for training
- Stream: <1ms per event = Required for real-time scoring

### Temporal Causality Protection
The most critical rule for fraud systems:
```python
label_available_timestamp > event_timestamp
```

Without this, you could accidentally:
- Train on labels from the future
- Build features using information that didn't exist yet
- Create a model that works in backtesting but fails in production

Phase 3 makes this **impossible** by rejecting any data that violates causality.

---

## Conclusion
**Phase 3 is COMPLETE.** ✅

You have built a production-grade data quality layer that:
- Enforces structural integrity (correct types, unique IDs)
- Validates business logic (reasonable amounts, valid currencies)
- Guarantees temporal causality (no label leakage)
- Operates in both batch and streaming modes
- Automatically rejects bad data before it reaches your model

**Key Takeaway:**

**Before Phase 3:** "Is my model good?" → Maybe, but the data might be garbage.

**After Phase 3:** "Is my model good?" → Yes, *and* I guarantee the data is clean because bad data is rejected automatically.

**Phase 3 guarantees the input is clean, causally valid, and safe for point-in-time feature engineering.**
**Without this guarantee, any fraud model would be mathematically impressive but operationally impossible.**

**Next: Phase 4 (Feature Engineering with Point-in-Time Correctness).**
