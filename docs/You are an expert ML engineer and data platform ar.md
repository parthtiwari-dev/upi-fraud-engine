<img src="https://r2cdn.perplexity.ai/pplx-full-logo-primary-dark%402x.png" style="height:64px;margin-right:32px"/>

# You are an expert ML engineer and data platform architect.

Your only job in this chat is to design a complete, self-contained roadmap for a single project:
Real-Time UPI-Style Fraud Decision System built on top of the IEEE-CIS fraud dataset, with streaming simulation but WITHOUT requiring Kafka for a first implementation.
I am not going to use LLMs while building this project.
So whatever you output here must be:
Detailed enough that I can follow it step by step.
Ordered correctly so I never have to guess “what next?”.
Focused on code and architecture I can fully understand, not vibe-code or magic tools.
Treat this like a one-shot design doc + roadmap that I will follow blindly.
Reality framing
First, accept this:
This is not a “build a model” project.
This is a real-time decision system under uncertainty.
If you treat it like a Kaggle modeling exercise, the project fails.
If you treat it as data + decisions + constraints, it will get me hired.
Keep that frame in your head as you design the plan.
Constraints \& tech choices
Hard constraints:
Base data: IEEE-CIS Fraud Detection dataset (from Kaggle).
Domain: UPI-style / real-time payment fraud, but implemented by “UPI-ifying” IEEE-CIS.
First version: NO Kafka.
Use a simple Python-based “event generator” + in-memory/Redis/Postgres to simulate streaming.
The design must keep the ingestion boundary clean so Kafka/Redpanda can be plugged in later.
No Kubernetes, no heavy distributed systems.
Allowed stack (pick minimal, justified components):
Python
FastAPI (or similar) for the scoring API
Postgres or DuckDB for offline features and backtesting
Redis (or just in-memory Python dicts for v1) for online state
MLflow for experiments and model registry
Great Expectations for data validation
Optional but nice:
NetworkX (or similar) for graph features
A thin Streamlit UI as a control panel on top of the backend
Important:
If you recommend a tool or library, you must explain why it’s needed in this project (not just “because it’s cool”).
Concepts I must actually understand (not buzzwords)
You must structure the roadmap so that I learn and implement these properly:
Label delay and why fraud labels arrive late
Temporal leakage and point-in-time correctness
Precision at K vs ROC-AUC, and why alert budget matters more than accuracy
Concept drift vs data drift
Why unsupervised first, supervised second is used in fraud
Batch vs streaming ingestion and how to keep them consistent
If any step in your roadmap depends on one of these ideas, you must explicitly say:
“At this step, you should study X and implement Y because of Z.”
Project definition (you must keep enforcing this)
Before anything else, define and repeat this constraint:
At transaction time T, using only information available before T, decide whether to raise a fraud alert under a fixed daily alert budget.
Any feature, join, or evaluation that violates this time rule or the alert budget concept is wrong.
You must design the plan so that I do not accidentally violate this.
What you must output
I want a single, structured roadmap for the whole project, with:
A clear phase breakdown in the only correct order.
For each phase:
Goal in 1–2 sentences.
Exact tasks, written as a checklist I can follow.
What files/folders to create (e.g., data_generation/generate_upi_data.py, features/build_features.py, etc.).
Any minimal theory I must learn before or during that phase (with search keywords, not links).
What counts as “done” for that phase (a concrete deliverable).
The phases must include at least:
Phase 0: Decision definition
Write the decision sentence in the README and keep referring back to it.
Phase 1: Data simulation and UPI enrichment
Take IEEE-CIS transactions and:
Map to a UPI-ish schema:
user_id, payer_vpa, payee_vpa, device_id, ip_region, app_version, etc.
Add synthetic entity relationships:
A device used by multiple VPAs
A VPA used by multiple devices over time
Inject controlled fraud patterns:
Shared device fraud rings
Sudden velocity spikes (many tx in short time)
Time-of-day anomalies
Deliverable:
A single reproducible data generator script or module (e.g., data_generation/generate_upi_like_data.py) that:
Reads IEEE-CIS raw data.
Outputs enriched, UPI-like transactions with labels.
Is config-driven (so I can tweak fraud rates and pattern strength).
Phase 2: Ingestion pipeline (batch + streaming sim)
Batch path:
Load enriched data into an offline store (e.g., DuckDB/Postgres).
Partition by event time (not load time).
Streaming simulation path:
A Python script that:
Reads the same data in event-time order.
Sends one transaction at a time to a scoring service (via HTTP) or pushes events into an in-memory queue / Redis stream.
A consumer (inside the scoring service) that:
Processes each event exactly once.
Deliverable:
Guarantee that a given transaction is represented identically in batch and stream flows (same IDs, same timestamps, same payload).
You must specify how I will test “batch and stream agree”.
Phase 3: Data validation with Great Expectations
Design validation for both:
Batch backfills
Streaming micro-batches (or per-event validations aggregated)
Checks:
Schema correctness (types, required columns).
Value ranges (amount >= 0, timestamps increasing, categorical domains).
Basic distribution checks (amount distribution, device usage, etc.).
Deliverable:
Great Expectations suite(s) wired into both ingestion paths.
Clear distinction between hard failures (reject data) and soft warnings (log + monitor).
Phase 4: Feature engineering with strict time discipline
Features to build:
Velocity features:
Rolling count and sum per user/device/merchant over multiple windows (e.g., 5 min, 1 hour, 1 day).
Time features:
Time-of-day, day-of-week.
Graph-ish features:
Number of VPAs per device.
Number of devices per VPA.
Recent fraud density in local neighborhood (last N transactions on connected nodes).
Requirements:
Every feature must use only past data relative to the current transaction’s event time.
No future joins, no peeking at later events.
Deliverables:
Offline feature builder (batch) that can regenerate features from raw enriched data.
Online feature computation path that updates state incrementally as events arrive.
At least a couple of unit tests that would fail if I accidentally introduce leakage.
Phase 5: Two-stage modeling
Stage 1: Anomaly detector
Isolation Forest (or similar) trained on non-fraud / early-time data only.
Outputs a score (how weird the behavior is), no hard “fraud/not fraud” decision.
Stage 2: Supervised fraud classifier
Gradient boosting model (e.g., XGBoost/LightGBM) using:
All engineered features
The anomaly score from Stage 1 as an input.
Trained with cost-sensitive or class-weighted loss.
Deliverables:
Separate, reproducible training pipelines for Stage 1 and Stage 2 (scripts or modules, not notebooks).
Models and metrics tracked in MLflow.
A short, explicit explanation in the plan of why two stages (cheap anomaly filter + targeted classifier, robustness to new patterns).
Phase 6: Evaluation \& backtesting
No accuracy metric. Instead:
Precision/recall at a fixed alert budget (e.g., “we can only alert on 0.5% of transactions per day”).
Rolling evaluation over time (e.g., weekly or daily windows).
Latency per transaction (even if measured locally).
Scenarios to simulate:
Normal period.
Sudden spike in fraud rate (e.g., new attack pattern).
Behavior shifts (e.g., more night-time traffic).
Deliverables:
Backtesting script(s) that:
Replays historical data day by day.
Applies the model + alert policy.
Logs per-period metrics.
Plots over time (fraud caught, false alerts, alert rate, etc.).
Phase 7: Monitoring \& failure modes
Monitoring to implement (even if partially mocked):
Score distribution drift.
Feature drift for key features (velocity stats, graph degrees).
Observed fraud rate vs expected baseline.
Alert volume vs configured budget.
Failure modes to document:
Known blind spots (e.g., completely new devices + very small amounts).
Failure-safe modes (e.g., if features are missing, default to safer decision).
Dangerous failure modes (e.g., ingest lag meaning decisions are made on stale history).
Deliverables:
Minimal monitoring hooks (could be logs + small scripts or a very simple dashboard).
A “Failure Modes” section in the README, written like a design doc for reviewers.
Final repo structure
You must design the roadmap so that I end up with a repo like:
data_generation/
ingestion/
validation/
features/
models/
evaluation/
monitoring/
configs/
api/ (FastAPI service)
ui/ (optional Streamlit control panel)
README.md that tells a coherent, end-to-end story
No Jupyter notebook is allowed as the core artifact.
Notebooks are allowed only for exploration; the real pipeline must be runnable via scripts/CLI.
VERY IMPORTANT: Guidance style
When you answer:
Give me the phases in order.
Inside each phase, give me a linear checklist of tasks like:
[ ] Create file X with responsibility Y
[ ] Implement function Z that does A, B, C
Do not generate actual code, just structure and responsibilities.
Assume I won’t call an LLM again; this one document must be enough.
Your output should feel like a senior engineer’s project plan handed to a motivated junior who is good at Python/ML basics but weak on systems design and discipline.

I'm going to structure this as a complete, executable roadmap for a real-time fraud decision system. This is not a modeling exercise—it's an engineering project. I'll treat you like a capable junior who needs explicit ordering, time discipline rules, and a clear mental model of why each piece exists.

Let me start by anchoring the frame, then give you the phases.

***

## FRAME: THE DECISION PROBLEM (READ THIS FIRST)

Before phase 0, internalize this constraint. Repeat it back to yourself every time you're confused:

**At transaction time T, using ONLY information available strictly before T, decide whether to raise a fraud alert under a fixed daily alert budget.**

This sentence is your truth detector. If you find yourself:

- Using the transaction's label to engineer features → you've leaked.
- Aggregating statistics up to T+1 hour and calling it "history at T" → you've peeked.
- Raising alerts based on "how many we've already alerted today" from a batch report → you've violated the budget.

Then you've broken the problem.

**The alert budget constraint is hard.** You cannot alert on 100% of suspected fraud. You alert on at most 0.5% of transactions per day (example). This is NOT "low precision is okay." It's "choose which 0.5% to alert on, and miss the rest."

This changes everything about evaluation, modeling, and alerting. Most fraud teams get this wrong.

***

# PROJECT ROADMAP: REAL-TIME UPI FRAUD DECISION SYSTEM

## PHASE 0: Decision Definition \& Project Charter

**Goal:** Write down the decision rule, constraints, and success criteria in plain language. Make it impossible to accidentally build the wrong thing.

**Tasks:**

- [ ] Create `README.md` at repo root. Add a "Decision Problem" section:

```
At transaction time T, using only features computed from all transactions with timestamp < T,
decide: Alert or Not Alert.

Constraint 1: We have a fixed alert budget of K alerts per day (e.g., 0.5% of daily volume).
Constraint 2: Label delay is ~48 hours; we train on labels available now but must decide on live data with no label.
Constraint 3: New device + small amount = hardest case (legitimate or sophisticated fraud?).

Success: Catch X% of fraud within budget while keeping false alert rate < Y%.
```

- [ ] Create `DESIGN.md` with:
    - **Data flow diagram (text-based):**

```
Raw IEEE-CIS → Enrichment (UPI schema) → Batch store (DuckDB/Postgres)
                                       ↓
                                    Validation (GX)
                                       ↓
                                    Offline Features (point-in-time)
                                       ↓
                                    Model training (Stage 1 & 2)
                                       ↓
                                    Backtesting (budget + precision@K)

Streaming simulation:
Same enriched data → Event generator (one tx at a time, ordered by event_time)
                        ↓
                    HTTP to scoring API (or Redis queue)
                        ↓
                    Feature computation (online state)
                        ↓
                    Model scoring
                        ↓
                    Alert decision (respecting budget)
                        ↓
                    Logging + monitoring
```

    - **Label delay diagram:** Show that labels arrive ~48h late, so test set must be partitioned by "when we knew the label", not "when the transaction happened".
    - **Alert budget math:** If we process 100k tx/day and budget is 0.5%, we alert on 500. If fraud rate is 1%, we catch 500/1000 = 50% of fraud.
- [ ] Create `configs/project.yaml`:

```yaml
project_name: "upi_fraud_system"
alert_budget_pct: 0.005  # 0.5% of daily volume
label_delay_hours: 48

fraud_simulation:
  device_ring_rate: 0.02  # 2% of txs involve device sharing rings
  velocity_spike_rate: 0.01
  time_anomaly_rate: 0.015

features:
  velocity_windows_minutes: [5, 60, 1440]  # 5min, 1h, 1day
  graph_lookback_txs: 100  # recent N txs for graph features

models:
  stage1_type: "isolation_forest"
  stage1_contamination: 0.05  # assume ~5% anomalies in unlabeled data
  stage2_type: "xgboost"
  stage2_class_weight: "balanced"

backtesting:
  test_window_days: 7
  retest_frequency_days: 1
```

- [ ] Create a `LEARNINGS.md` file for yourself. Before each phase, add a section:

```
## Concept: Label Delay & Training Set Definition

**Why it matters:**
- You cannot use labels for events that happen today (they arrive tomorrow/day-after).
- Your training set is: old events + their true labels (now available).
- Your test set is: recent events + whatever labels are available now (partial/missing).
- Backtesting must account for this: replay day-by-day, train on fully-labeled history, test on next day.

**How to avoid mistakes:**
- Never mix "training date" with "event date".
- Always partition on event_time, not load_time.
- When backtesting, mark each event with "label available?" flag.
```


**Definition of "Done":**

- README clearly states the decision constraint and alert budget.
- DESIGN.md has text diagrams and label delay explanation.
- You can read these and explain the project to someone without speaking.
- LEARNINGS.md has Concept sections that you update as you go.

***

## PHASE 1: Data Simulation \& UPI Enrichment

**Goal:** Transform IEEE-CIS raw data into a UPI-like schema with controlled fraud patterns, such that you can replicate it deterministically and reason about where fraud patterns come from.

**Why this is not a "load data and train" step:**

- IEEE-CIS is payment card data, not UPI (which is real-time, person-to-person, device-centric).
- You need to know your data's structure intimately. If you can't generate it, you don't understand it.
- Controlled fraud patterns will teach you what features matter.

**Theory to learn before this phase:**

- Search: *IEEE-CIS fraud dataset structure, columns, target definition*
- Search: *UPI payment system, VPA, device binding, real-time settlement*
- Concept: Entity-event-time data model (transactions are events, users/devices/merchants are entities, relationships evolve).
- Concept: Synthetic data and fraud pattern injection (ring fraud, velocity spikes are real UPI attacks; this is not toy data).

**Tasks:**

- [ ] Create `data_generation/` folder structure:

```
data_generation/
  __init__.py
  download_raw.py          # fetch IEEE-CIS from Kaggle (you may need to do this manually)
  enrich_to_upi_schema.py  # main transformation script
  fraud_injector.py        # add controlled patterns
  validator.py             # sanity checks on generated data
  README.md                # data schema documentation
```

- [ ] Create `data_generation/README.md` documenting the UPI-like schema:

```
## Generated Schema

transaction_id: str (unique, deterministic from IEEE-CIS event_id)
event_timestamp: datetime (parsed from IEEE-CIS, used for all time-based logic)

### Payer & Payee (UPI VPA style)
payer_id: int (user, mapped from IEEE-CIS card_id)
payer_vpa: str (e.g., "user_123@bank")
payee_id: int (merchant, mapped from IEEE-CIS merchant_id)
payee_vpa: str (e.g., "merchant_456@bank")

### Device & Network
device_id: str (hashed, tied to a payer but can be used by multiple payers in fraud rings)
device_type: str (enum: mobile, web, desktop)
ip_region: str (geo, from IEEE-CIS or synthetic)
app_version: str (synthetic, to create app-version-specific patterns)

### Transaction Details
amount: float (from IEEE-CIS, in some normalized currency)
currency: str (hardcoded to INR for UPI)
txn_type: str (enum: p2p_transfer, bill_pay, merchant_payment)

### Labels & Timing
is_fraud: bool (IEEE-CIS target, or injected)
label_available_timestamp: datetime (event_timestamp + ~48h)
fraud_pattern: str or null (e.g., "device_ring", "velocity_spike", "time_anomaly")

### Metadata
load_timestamp: datetime (when this row was generated; useful for replay)
```

- [ ] Implement `data_generation/enrich_to_upi_schema.py`:

```
Module responsibilities:

1. read_ieee_cis_raw(filepath) -> pd.DataFrame
   - Load IEEE-CIS Parquet/CSV
   - Parse timestamp columns correctly
   - Return DataFrame with: event_id, card_id, merchant_id, amount, V1..V39 (features), Target, timestamps

2. map_to_upi_schema(ieee_cis_df, config) -> pd.DataFrame
   - Create transaction_id = hash(event_id + salt)
   - Create payer_vpa from card_id (e.g., "user_{card_id % 10000}@testbank")
   - Create payee_vpa from merchant_id (e.g., "merchant_{merchant_id}@testbank")
   - Assign device_id: 
       * Most txs: one device per user (deterministic from payer_id)
       * Some txs (controlled %): device used by multiple users (fraud ring setup)
   - Parse event_timestamp from IEEE-CIS
   - Compute label_available_timestamp = event_timestamp + 48h (or config)
   - Keep IEEE-CIS Target as is_fraud (initially)
   - Add fraud_pattern = null (to be filled by fraud_injector)
   - Return new DF

3. validate_schema(df) -> bool
   - Check all required columns exist
   - Check types (timestamp is datetime64, amounts are float, etc.)
   - Check no nulls in critical columns
   - Raise ValueError if invalid

4. main(ieee_cis_path, output_path, config_path)
   - Load config from YAML
   - Call read_ieee_cis_raw
   - Call map_to_upi_schema
   - Call validate_schema
   - Save to DuckDB or Parquet (deterministic order: sort by event_timestamp, then transaction_id)
```

- [ ] Implement `data_generation/fraud_injector.py`:

```
Module responsibilities:

Inject controlled fraud patterns into the enriched data (do this AFTER base mapping).

1. inject_device_ring_fraud(df, config) -> df
   - Select a fraction of is_fraud=True rows (controlled by config.fraud_simulation.device_ring_rate)
   - For each selected row, modify: assign a device_id used by 2-5 other users
     (i.e., make the device shared across multiple payers)
   - Set fraud_pattern = "device_ring"
   - Return modified df

2. inject_velocity_spikes(df, config) -> df
   - Within each day, select a fraction of non-fraud rows
   - For each, cluster nearby txs by (payer_id, payee_vpa) in a 1-hour window
   - If the cluster has >= 10 txs: mark as fraud, set fraud_pattern = "velocity_spike"
   - Only mark if is_fraud was originally False (don't double-count)
   - Return modified df

3. inject_time_anomalies(df, config) -> df
   - Select a fraction of non-fraud rows
   - If (hour(event_timestamp) in [2, 3, 4]) AND (payer_id typical_hour is 9-17):
     Mark as fraud, set fraud_pattern = "time_anomaly"
   - Return modified df

4. main(enriched_df, config) -> df
   - Apply all three injections sequentially
   - Verify: no row is marked twice (if so, pick one pattern, log warning)
   - Return modified df
```

- [ ] Implement `data_generation/validator.py`:

```
Module responsibilities:

Sanity checks on generated data (run before saving).

1. check_duplicates(df) -> bool
   - Verify transaction_id is unique
   - Raise if duplicates exist

2. check_timestamp_order(df) -> bool
   - Verify event_timestamp is non-decreasing
   - If not, sort and warn

3. check_label_delay(df) -> bool
   - Verify label_available_timestamp = event_timestamp + ~48h (within tolerance)
   - Flag if outliers exist

4. check_entity_consistency(df) -> bool
   - For each (payer_id, device_id) pair, verify device is used consistently over time
     (i.e., don't have device_id flip between two payers in adjacent txs; that's implausible)
   - Log warnings for suspicious patterns

5. check_fraud_distribution(df) -> dict
   - Return: total_fraud_count, fraud_rate (%), fraud_patterns breakdown
   - If fraud_rate > 10% or < 0.1%, warn (check if injection was misconfigured)

6. main(df, config) -> bool
   - Call all checks
   - Return True if all pass; raise if critical failures
```

- [ ] Create `data_generation/generate_data.py` (main entry point):

```
Script responsibilities:

if __name__ == "__main__":
    import argparse
    import yaml
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--ieee_cis_path", required=True, help="Path to IEEE-CIS raw data")
    parser.add_argument("--output_path", required=True, help="Where to save enriched data")
    parser.add_argument("--config_path", default="configs/project.yaml")
    args = parser.parse_args()
    
    config = yaml.safe_load(open(args.config_path))
    
    # Step 1: Load & map
    raw_df = read_ieee_cis_raw(args.ieee_cis_path)
    enriched_df = map_to_upi_schema(raw_df, config)
    validate_schema(enriched_df)
    print(f"[OK] Mapped to UPI schema: {len(enriched_df)} rows")
    
    # Step 2: Inject fraud patterns
    enriched_df = inject_device_ring_fraud(enriched_df, config)
    enriched_df = inject_velocity_spikes(enriched_df, config)
    enriched_df = inject_time_anomalies(enriched_df, config)
    print(f"[OK] Injected fraud patterns")
    
    # Step 3: Validate
    checks = validate_data(enriched_df, config)
    print(f"[VALIDATION] {checks}")
    
    # Step 4: Save
    save_to_duckdb(enriched_df, args.output_path)
    print(f"[OK] Saved {len(enriched_df)} rows to {args.output_path}")
```


**Definition of "Done":**

```
- Running `python data_generation/generate_data.py --ieee_cis_path <path> --output_path <output>` produces a deterministic, sorted DuckDB file.
```

- You can inspect the output and explain: "This device is used by 3 users (fraud ring)", "This user has 15 txs in 10 minutes (velocity spike)".
- Fraud patterns are injected and labeled in `fraud_pattern` column.
- Schema matches documentation.

***

## PHASE 2: Ingestion Pipeline (Batch + Streaming Simulation)

**Goal:** Build two parallel paths—batch and streaming—that process the same data identically, such that features and decisions are reproducible across both paths. This is the foundation for later Kafka integration.

**Why this matters:**

- If batch and streaming differ, you have a production bug.
- You will test: "Given transaction X, does it have the same payload and features in both paths?"
- Streaming is simulated (Python event generator), but the API it calls is real.

**Theory to learn:**

- Search: *Event time vs processing time in streaming systems*
- Search: *Exactly-once semantics and idempotency in event processing*
- Concept: Watermarks and triggering (when do you decide that no more events will arrive for time T?).
- Concept: Backpressure and batching (how do you avoid being overwhelmed when events arrive faster than you can process?).

**Tasks:**

- [ ] Create `ingestion/` folder:

```
ingestion/
  __init__.py
  batch_loader.py         # load enriched data into offline store
  streaming_simulator.py  # event generator (reads enriched data in order, sends to API)
  schema.py               # shared transaction schema (dataclass or Pydantic)
  validators.py           # Schema validation (integrates with Great Expectations in Phase 3)
  README.md
```

- [ ] Implement `ingestion/schema.py` (Pydantic models):

```
Module responsibility: Single source of truth for transaction schema.
Shared by batch loader, event generator, API, feature builder, and monitoring.

class Transaction(BaseModel):
    transaction_id: str
    event_timestamp: datetime
    payer_id: int
    payer_vpa: str
    payee_id: int
    payee_vpa: str
    device_id: str
    device_type: str
    ip_region: str
    app_version: str
    amount: float
    currency: str
    txn_type: str
    is_fraud: bool  # label (only populated after label delay)
    label_available_timestamp: datetime
    fraud_pattern: Optional[str]
    load_timestamp: datetime

class TransactionBatch(BaseModel):
    transactions: List[Transaction]
    batch_id: str
    processed_at: datetime
```

- [ ] Implement `ingestion/batch_loader.py`:

```
Module responsibilities:

1. load_enriched_data(duckdb_path, config) -> pd.DataFrame
   - Open DuckDB connection
   - Read all rows from enriched data table
   - Sort by event_timestamp (enforce)
   - Return as DataFrame

2. partition_by_event_time(df, date_column="event_timestamp") -> Dict[date, pd.DataFrame]
   - Group df by calendar date of event_timestamp
   - Return dict {date: df_for_that_date}
   - This is critical: partitions are by event time, not load time

3. write_to_offline_store(df, postgres_conn_or_duckdb_path, table_name) -> int
   - Write df to table (if table exists, append; else create)
   - Ensure event_timestamp is indexed
   - Ensure transaction_id is unique (or has unique index)
   - Return number of rows written
   - IMPORTANT: Do NOT dedup by transaction_id silently; fail if you see a duplicate

4. main(enriched_duckdb_path, offline_store_path, config)
   - Load enriched data
   - Partition by event_time
   - Write each partition to offline store (simulating "daily backfill")
   - Print summary: total rows, date range, fraud count
```

- [ ] Implement `ingestion/streaming_simulator.py`:

```
Module responsibilities:

1. EventGenerator class:
   - __init__(enriched_duckdb_path, config)
     * Load enriched data sorted by event_timestamp
     * Initialize pointer to first row
   - __iter__() -> Iterator[Transaction]
     * Yield transactions one by one in event_timestamp order
     * Return Pydantic Transaction objects
   - advance_to_time(target_datetime) -> List[Transaction]
     * If called with a specific time, skip to that time and yield all txs up to it
     * Useful for "fast-forward" replay in backtesting

2. StreamingSimulator class:
   - __init__(event_generator, api_endpoint_url, batch_size=1, delay_ms=0)
     * Takes an EventGenerator
     * Knows the scoring API endpoint
     * Can optionally batch events (batch_size > 1) or add artificial delay
   - run(end_time=None, max_events=None)
     * Call event_generator and for each transaction:
       - POST to API endpoint as JSON (using Transaction.dict())
       - Receive decision response (alert or not)
       - Log decision + metadata
     * If end_time provided, stop when generator reaches it
     * If max_events provided, stop after N events
     * Return list of (transaction, decision) tuples

3. main(enriched_duckdb_path, api_endpoint, config, output_log_path)
   - Create EventGenerator
   - Create StreamingSimulator
   - Run simulation, collect results
   - Save results to output_log_path (JSON or Parquet)
```

- [ ] Implement `ingestion/consistency_check.py` (test script, not core path):

```
Script responsibility:

Test that batch and streaming paths are identical.

if __name__ == "__main__":
    # 1. Load batch data from offline store
    batch_df = load_from_offline_store(offline_store_path, date="2024-01-15")
    
    # 2. Run streaming simulator for same date
    event_gen = EventGenerator(enriched_duckdb_path)
    streaming_results = run_streaming_sim_for_date("2024-01-15", event_gen, api_endpoint)
    
    # 3. Compare transaction-by-transaction
    for batch_txn, stream_txn in zip(batch_df.iterrows(), streaming_results):
        assert batch_txn['transaction_id'] == stream_txn.transaction_id
        assert batch_txn['event_timestamp'] == stream_txn.event_timestamp
        assert batch_txn['amount'] == stream_txn.amount
        # ... all fields
    
    print("[OK] Batch and streaming paths are identical")
```

- [ ] Create `ingestion/README.md`:

```
## Ingestion: Batch & Streaming Paths

Both paths consume the same enriched data (from Phase 1).

### Batch Path
- Entry: `ingestion/batch_loader.py main()`
- Reads all enriched data, partitions by event_time
- Writes to offline store (DuckDB or Postgres) with event_timestamp index
- Used for: training, backtesting, offline feature engineering

### Streaming Path
- Entry: `ingestion/streaming_simulator.py main()`
- Reads enriched data in event_time order
- Sends transactions one-by-one to scoring API
- Collects decisions and logs them
- Used for: live testing, latency measurement, alert validation

### Consistency Guarantee
- Run `python ingestion/consistency_check.py` after each change
- If batch and streaming differ, the test will fail
- Both paths use the same Transaction schema (Pydantic)
- Both paths sort by event_timestamp, never by load_timestamp
```


**Definition of "Done":**

- `python ingestion/batch_loader.py` populates an offline store with all enriched data, indexed by event_timestamp.
- `python ingestion/streaming_simulator.py` can replay the same data one transaction at a time to an HTTP endpoint.
- `python ingestion/consistency_check.py` passes: batch and streaming transactions are identical.

***

## PHASE 3: Data Validation with Great Expectations

**Goal:** Catch data quality issues early in both batch and streaming paths, before they reach feature engineering or scoring.

**Why this matters:**

- You will have bad data (missing timestamps, impossible amounts, new device + new merchant with 10k INR).
- You need to decide: reject it (hard failure), or log + continue (soft warning).
- Validation rules are your contract with the data.

**Theory to learn:**

- Search: *Great Expectations framework, checkpoints, actions*
- Concept: Deterministic vs statistical validation (schema is deterministic; "amount distribution looks normal" is statistical).

**Tasks:**

- [ ] Create `validation/` folder:

```
validation/
  __init__.py
  build_suite.py          # Define GX suites (schema, ranges, distributions)
  run_validation.py       # Execute validation, decide hard vs soft fail
  expectations/           # Auto-generated by GX (git-ignore if desired)
  README.md
```

- [ ] Implement `validation/build_suite.py`:

```
Module responsibilities:

1. build_schema_suite(context, suite_name="transaction_schema") -> ExpectationSuite
   - Expect columns: transaction_id, event_timestamp, payer_id, payer_vpa, ...
   - Expect types: str, datetime, int, str, ...
   - Expect no duplicates in transaction_id
   - Expect no nulls in critical columns
   - Return suite

2. build_value_ranges_suite(context, suite_name="value_ranges") -> ExpectationSuite
   - Expect amount > 0
   - Expect amount < 1,000,000 (reasonable UPI limit)
   - Expect currency = "INR"
   - Expect txn_type in ["p2p_transfer", "bill_pay", "merchant_payment"]
   - Expect device_type in ["mobile", "web", "desktop"]
   - Expect device_id is non-empty string
   - Return suite

3. build_temporal_suite(context, suite_name="temporal_consistency") -> ExpectationSuite
   - Expect event_timestamp < label_available_timestamp
   - Expect label_available_timestamp = event_timestamp + ~48h (within +/- 2h tolerance)
   - Expect event_timestamp is increasing (monotonic when sorted by transaction_id)
   - Return suite

4. build_distribution_suite(context, suite_name="distributions") -> ExpectationSuite
   - This suite is built from a reference dataset (training data)
   - Expect amount distribution to match reference (e.g., histogram KL-divergence < threshold)
   - Expect payer_id cardinality to be within reference range (soft check for new users)
   - Expect fraud_rate to be within baseline +/- 2% (soft check for sudden spikes)
   - Return suite

5. main(reference_data_df, config)
   - Create GX context
   - Build all suites
   - Save to configs/
```

- [ ] Implement `validation/run_validation.py`:

```
Module responsibilities:

1. validate_batch(df, suite, fail_level="hard") -> ValidationResult
   - Run suite against df
   - Collect failures
   - If fail_level == "hard": raise if ANY expectation fails
   - If fail_level == "soft": log warnings but continue
   - Return result object with summary

2. validate_streaming(transaction, suite, fail_level="soft") -> ValidationResult
   - Run suite against single transaction (converted to 1-row DataFrame)
   - If fail_level == "soft": log warning per-transaction (don't slow down API)
   - If fail_level == "hard": raise exception (API will return error)
   - Return result

3. main_batch(offline_store_df, config)
   - Load schema + value_ranges + temporal suites
   - Run validate_batch(df, suite, fail_level="hard")
   - If passes, proceed to feature engineering
   - If fails, print errors and halt

4. main_streaming(transaction_dict, config)
   - Load suites
   - Convert dict to Pydantic Transaction, then to DataFrame
   - Run validate_streaming(txn_df, suite, fail_level="soft")
   - Return decision: proceed or reject
```

- [ ] Create `validation/README.md`:

```
## Data Validation with Great Expectations

### Suites

1. **schema** (hard failure): Types, required columns, uniqueness
2. **value_ranges** (hard failure): Amount bounds, enum values
3. **temporal_consistency** (hard failure): Timestamp ordering, label delay correctness
4. **distributions** (soft warning): Statistical drift detection

### Integration Points

**Batch path:**
- Before writing to offline store, run schema + value_ranges + temporal suites
- If any fail, halt and investigate

**Streaming path:**
- For each incoming transaction, run schema + value_ranges (soft)
- Log any warnings to monitoring system
- Do NOT block API response on soft failures

### False Positives

- New devices will naturally have low transaction count (expected, not a failure)
- Overnight txs are expected (don't mark as anomaly just because they're rare)
- Adjust distribution suite thresholds to tolerate normal variance
```


**Definition of "Done":**

- `python validation/build_suite.py` generates suites in `configs/`.
- Before batch loading, `python validation/run_validation.py --mode batch` passes on enriched data.
- Streaming validation integrated into Phase 5 (API) such that soft warnings are logged but don't block responses.

***

## PHASE 4: Feature Engineering with Strict Time Discipline

**Goal:** Build features from transactions and entity history such that every feature uses ONLY past data relative to the current transaction's event_time. No leakage. No future peeking.

**Critical discipline:**
At transaction time T, your feature set includes:

- Counts/sums of past transactions (T-5min to T-1sec)
- Graph properties computed from past transactions
- Time-of-day / day-of-week of T itself
- **NOT:** Future transactions, future labels, information about T+1second

**Theory to learn:**

- Search: *Point-in-time correctness, as-of joins, feature engineering for time series*
- Search: *Windowed aggregations, watermarks, state stores*
- Concept: Point-in-time feature snapshot (given event_time T, what did the world look like just before T?).
- Concept: Offline vs online feature stores (offline = batch recomputation from raw data; online = stateful updates as events arrive).

**Tasks:**

- [ ] Create `features/` folder:

```
features/
  __init__.py
  schema.py               # FeatureVector dataclass
  offline_builder.py      # Batch feature computation (from raw, time-correct)
  online_builder.py       # Streaming feature computation (incremental state updates)
  feature_definitions.py  # Functions for each feature (to avoid copy-paste)
  time_utils.py           # Helper: get_point_in_time_data(), window queries
  tests/
    test_time_correctness.py  # Unit tests that verify no leakage
  README.md
```

- [ ] Implement `features/schema.py`:

```
Module responsibility: Define the feature vector.

class FeatureVector(BaseModel):
    transaction_id: str
    event_timestamp: datetime
    
    # Velocity: per payer
    payer_txn_count_5min: int       # N txs by this payer in [T-5min, T)
    payer_txn_sum_5min: float
    payer_txn_count_1h: int
    payer_txn_sum_1h: float
    payer_txn_count_1d: int
    payer_txn_sum_1d: float
    
    # Velocity: per device
    device_txn_count_5min: int
    device_txn_sum_5min: float
    device_txn_count_1h: int
    device_txn_sum_1h: float
    
    # Velocity: per (payer, payee) pair
    payer_payee_txn_count_1h: int   # How many times has this payer paid this payee in last hour?
    
    # Graph: entity cardinality
    device_unique_payers: int       # How many distinct payers have used this device (recent history)?
    payer_unique_devices: int       # How many distinct devices has this payer used?
    payer_unique_payees_1d: int
    
    # Graph: fraud density
    payer_recent_fraud_density: float  # Of last N txs by payer, how many were fraud?
    device_recent_fraud_density: float
    
    # Temporal
    txn_hour_of_day: int            # hour(event_timestamp)
    txn_day_of_week: int            # day(event_timestamp)
    txn_is_night_time: bool         # hour in [22, 23, 0, 1, 2, 3, 4]?
    
    # Categorical encoding (one-hot handled downstream)
    device_type: str
    ip_region: str
    app_version: str
    
    # Amount (normalized)
    amount_normalized: float        # amount / (payer mean + epsilon)
```

- [ ] Implement `features/time_utils.py`:

```
Module responsibilities: Time-correct queries.

1. get_past_transactions_in_window(
       offline_store_conn,
       entity_id: str,
       entity_type: str,  # "payer", "device", etc.
       event_timestamp: datetime,
       window_minutes: int,
       exclude_current_txn_id: str
   ) -> pd.DataFrame
   - Query offline store for all rows where:
     * (entity_column == entity_id)  # e.g., payer_id == entity_id if entity_type == "payer"
     * (event_timestamp >= event_timestamp - window_minutes)
     * (event_timestamp < event_timestamp)  # STRICT: no current txn, no future
     * (transaction_id != exclude_current_txn_id)  # don't include the current txn itself
   - Sort by event_timestamp ascending
   - Return DataFrame

2. compute_rolling_stats(past_txns_df, column: str) -> Dict[str, float]
   - Return: {"count": len, "sum": sum(column), "mean": mean, "max": max, "min": min}

3. get_graph_neighbors(
       offline_store_conn,
       node_id: str,
       node_type: str,  # "payer", "device"
       neighbor_type: str,  # "device", "payer"
       lookback_txns: int
   ) -> set
   - Query recent lookback_txns ordered by event_timestamp desc
   - Extract all unique neighbor_type values
   - Return as set

4. get_entity_cardinality(
       offline_store_conn,
       entity_id: str,
       entity_type: str,
       neighbor_type: str,
       event_timestamp: datetime,
       lookback_hours: int
   ) -> int
   - Count distinct neighbor entities that interacted with entity_id before event_timestamp
   - Use window of lookback_hours
```

- [ ] Implement `features/feature_definitions.py`:

```
Module responsibilities: Individual feature computation logic.

1. compute_velocity_features(
       transaction: Transaction,
       offline_store_conn,
       config
   ) -> Dict[str, float]
   - For each window in config.features.velocity_windows_minutes:
     * Call get_past_transactions_in_window(..., "payer", ...)
     * Compute count, sum
     * Store as {payer_txn_count_{window}min, payer_txn_sum_{window}min}
   - Similarly for device
   - Similarly for (payer, payee) pair
   - Return dict

2. compute_graph_features(
       transaction: Transaction,
       offline_store_conn,
       config
   ) -> Dict[str, float]
   - device_unique_payers = get_entity_cardinality(..., "device", "payer", ...)
   - payer_unique_devices = get_entity_cardinality(..., "payer", "device", ...)
   - payer_unique_payees_1d = ...
   - Return dict

3. compute_fraud_density_features(
       transaction: Transaction,
       offline_store_conn,
       config
   ) -> Dict[str, float]
   - past_payer_txns = get_past_transactions_in_window(..., "payer", ..., lookback_txns=100)
   - fraud_count = sum(is_fraud) among past_payer_txns
   - payer_recent_fraud_density = fraud_count / len(past_payer_txns)
   - Similarly for device
   - Return dict

4. compute_temporal_features(
       transaction: Transaction
   ) -> Dict[str, float]
   - hour = transaction.event_timestamp.hour
   - day_of_week = transaction.event_timestamp.weekday()
   - is_night = hour in [22, 23, 0, 1, 2, 3, 4]
   - Return dict

5. compute_amount_normalization(
       transaction: Transaction,
       offline_store_conn
   ) -> float
   - payer_mean_amount = mean(amount for all payer txns)
   - normalized = transaction.amount / (payer_mean_amount + epsilon)
   - Return float
```

- [ ] Implement `features/offline_builder.py`:

```
Module responsibilities: Batch feature computation.

1. build_feature_vectors_for_date(
       offline_store_conn,
       date_str: str,  # "2024-01-15"
       config
   ) -> pd.DataFrame
   - Query offline_store for all txns on date_str
   - Sort by event_timestamp
   - For each transaction:
     * Call compute_velocity_features(txn, ...)
     * Call compute_graph_features(txn, ...)
     * Call compute_fraud_density_features(txn, ...)
     * Call compute_temporal_features(txn)
     * Call compute_amount_normalization(txn, ...)
     * Combine into FeatureVector
   - Return DataFrame of FeatureVectors (or Parquet)

2. main(offline_store_path, output_feature_store_path, config)
   - For each date in offline_store:
     * build_feature_vectors_for_date(...)
     * Write to feature_store
   - Print summary
```

- [ ] Implement `features/online_builder.py`:

```
Module responsibilities: Streaming feature computation (stateful).

class OnlineFeatureStore:
    def __init__(self, config):
        self.payer_state = {}  # payer_id -> {recent_txns: deque, stats: dict}
        self.device_state = {}
        self.pair_state = {}  # (payer, payee) -> {recent_txns, stats}
        self.config = config
    
    def ingest_transaction(self, txn: Transaction) -> None
        # Called AFTER scoring decision, to update state for next txn
        # This is NON-BLOCKING: happens asynchronously
        payer_id = txn.payer_id
        if payer_id not in self.payer_state:
            self.payer_state[payer_id] = {
                "recent_txns_5min": deque(maxlen=1000),
                "recent_txns_1h": deque(maxlen=10000),
                "recent_txns_1d": deque(maxlen=100000),
            }
        self.payer_state[payer_id]["recent_txns_5min"].append(txn)
        # ... append to 1h, 1d windows too
        # Clean old entries outside window
    
    def compute_features_for_scoring(self, txn: Transaction) -> FeatureVector
        # Called DURING scoring (blocking)
        # Use in-memory state + optional fallback to offline store if state is missing
        
        velocity_features = {}
        for window_name, window_deque in self.payer_state[txn.payer_id].items():
            # Compute stats from deque
            past_txns = [t for t in window_deque if t.event_timestamp < txn.event_timestamp]
            count = len(past_txns)
            sum_amount = sum(t.amount for t in past_txns)
            velocity_features[f"payer_txn_count_{window_name}"] = count
            velocity_features[f"payer_txn_sum_{window_name}"] = sum_amount
        
        # ... similar for graph, temporal features
        
        return FeatureVector(
            transaction_id=txn.transaction_id,
            event_timestamp=txn.event_timestamp,
            **velocity_features,
            **graph_features,
            ...
        )
```

- [ ] Implement `features/tests/test_time_correctness.py`:

```
Unit tests to catch leakage. These are CRITICAL.

def test_no_future_features():
    # Given: txn at time T
    txn = Transaction(transaction_id="123", event_timestamp=datetime(2024, 1, 15, 10, 0, 0), ...)
    
    # Create offline store with txns:
    # - txn at 09:55 (5 min before) -> should be included
    # - txn at 10:00 (exact same time) -> should be EXCLUDED
    # - txn at 10:05 (5 min after) -> should be EXCLUDED
    
    # Compute features for txn
    features = compute_velocity_features(txn, store, config)
    
    # Assert: only the 09:55 txn is counted
    assert features['payer_txn_count_5min'] == 1

def test_point_in_time_correctness():
    # Scenario: payer has X txns up to T-1sec, Y txns after T+1sec
    # Features at T should reflect only X, not Y
    
    # Verify by running offline builder and checking specific transactions

def test_graph_cardinality_no_future_edges():
    # Scenario: a device will be used by 5 payers total, but 3 of them use it AFTER current txn
    # Graph feature at T should only count the 2 past users
    
    # Verify

def test_online_state_matches_offline_snapshot():
    # Scenario: run offline builder for date D, then stream same date through online store
    # For the last txn of day D, offline and online feature vectors should match (except for
    # features that depend on state updates that happen after scoring)
    
    # Verify
```

- [ ] Create `features/README.md`:

```
## Feature Engineering

### Time Discipline

At transaction time T, ONLY use transactions with event_timestamp < T.
Do not use labels (they arrive ~48h late).
Do not use future transactions.

### Features

**Velocity (rolling windows):**
- Count & sum of transactions per payer in [T-5min, T), [T-1h, T), [T-1d, T)
- Count & sum per device
- Count per (payer, payee) pair in last 1h

**Graph:**
- Unique devices per payer (recent history)
- Unique payers per device (recent history)
- Fraud density: % fraud among recent payer/device txns

**Temporal:**
- Hour of day, day of week
- Is night time?

**Normalization:**
- Amount scaled by payer's historical mean

### Offline vs Online

**Offline:** Batch-recompute features from raw enriched data, sorted by event_timestamp.
Used for training and backtesting.

**Online:** Maintain in-memory state (recent transactions), update after each decision.
Used for live scoring (but may have stale state if deployment is new).

### Tests

Run `pytest features/tests/test_time_correctness.py` before committing changes.
If tests pass, your features are time-correct.
```


**Definition of "Done":**

- `python features/offline_builder.py` produces a feature store (Parquet or DuckDB) with no time leakage.
- `pytest features/tests/test_time_correctness.py -v` passes (all tests green).
- You can read the code and trace through a transaction from T-5min to T and verify: "only past data is used".

***

## PHASE 5: Two-Stage Modeling

**Goal:** Build and train two models in sequence—an unsupervised anomaly detector (cheap, broad filter) and a supervised fraud classifier (targeted, precise)—such that you understand why two stages and can explain the trade-off.

**Why two stages:**

1. **Stage 1 (Anomaly):** Cheap, catches obvious outliers early. Works even with limited fraud labels.
2. **Stage 2 (Supervised):** Expensive (needs labels), precise. Catches sophisticated fraud with context.

- Think: Stage 1 is "alert on weird", Stage 2 is "alert on likely fraud".

**Theory to learn:**

- Search: *Isolation Forest, one-class SVM, unsupervised anomaly detection*
- Search: *Class imbalance, cost-sensitive learning, focal loss*
- Search: *Feature importance in tree models, SHAP values*
- Concept: Why label delay matters for train/test split (you cannot use today's labels to train today's model).
- Concept: Backtesting setup (replay day-by-day; train on old data, test on next day, never mix).

**Tasks:**

- [ ] Create `models/` folder:

```
models/
  __init__.py
  stage1_anomaly.py        # Isolation Forest pipeline
  stage2_supervised.py      # XGBoost pipeline
  training_pipeline.py      # Orchestrate both stages, handle time splits
  evaluation.py             # Metrics at fixed alert budget
  mlflow_utils.py           # Model registry integration
  model_schemas.py          # Pydantic for model metadata
  tests/
    test_no_label_leakage.py
  README.md
```

- [ ] Implement `models/stage1_anomaly.py`:

```
Module responsibilities: Train and score with Isolation Forest.

1. prepare_stage1_data(
       feature_vectors_df: pd.DataFrame,
       config,
       cutoff_date: datetime
   ) -> Tuple[pd.DataFrame, str]
   - Filter feature_vectors for event_timestamp < cutoff_date
   - Select columns: velocity features, graph features, temporal features, amount_normalized
   - Drop is_fraud (we don't use labels for unsupervised)
   - Standardize numerics (StandardScaler)
   - One-hot encode categoricals (device_type, ip_region, app_version)
   - Return X, feature_names

2. train_isolation_forest(
       X: pd.DataFrame,
       config
   ) -> IsolationForest
   - model = IsolationForest(
       contamination=config.models.stage1_contamination,  # e.g., 0.05
       random_state=42,
       n_estimators=100
     )
   - model.fit(X)
   - Return model

3. score_batch(
       model: IsolationForest,
       X: pd.DataFrame,
       feature_names: List[str]
   ) -> np.ndarray
   - Scores = model.decision_function(X)  # negative = more anomalous
   - Normalize to [0, 1]: score_01 = 1 / (1 + exp(score))
   - Return np.array of scores

4. main(
       feature_vectors_path,
       config,
       cutoff_date,
       output_model_path
   ) -> None
   - X, feature_names = prepare_stage1_data(...)
   - model = train_isolation_forest(X, config)
   - Save model to output_model_path (joblib)
   - Log to MLflow: model, params, hyperparams
```

- [ ] Implement `models/stage2_supervised.py`:

```
Module responsibilities: Train and score with XGBoost.

1. prepare_stage2_data(
       feature_vectors_df: pd.DataFrame,
       stage1_scores_df: pd.DataFrame,  # Scores from stage 1 inference on feature_vectors_df
       config,
       cutoff_date: datetime
   ) -> Tuple[pd.DataFrame, pd.Series]
   - Filter feature_vectors for event_timestamp < cutoff_date
   - Merge with stage1_scores (add as feature: "anomaly_score")
   - Select same features as stage 1 + anomaly_score
   - Extract target: is_fraud column
   - Check: is_fraud is not null (only use fully-labeled data)
   - Handle class imbalance: if imbalance > 10:1, use scale_pos_weight or class_weight
   - Return X, y

2. train_xgboost(
       X: pd.DataFrame,
       y: pd.Series,
       config
   ) -> XGBClassifier
   - model = XGBClassifier(
       n_estimators=200,
       max_depth=6,
       learning_rate=0.1,
       scale_pos_weight=compute_class_weight(y),  # handle imbalance
       random_state=42,
       eval_metric='aucpr'  # Precision-recall is better for imbalanced
     )
   - evals = [(X_val, y_val)]  # if we have holdout
   - model.fit(X, y, eval_set=evals, verbose=10)
   - Return model

3. score_batch(
       model: XGBClassifier,
       X: pd.DataFrame,
       feature_names
   ) -> np.ndarray
   - Scores = model.predict_proba(X)[:, 1]  # class 1 probability
   - Return np.array

4. main(
       feature_vectors_path,
       stage1_model_path,
       stage1_scores_path,
       config,
       cutoff_date,
       output_model_path
   ) -> None
   - Load stage 1 model, compute scores on feature_vectors
   - X, y = prepare_stage2_data(feature_vectors, stage1_scores, config, cutoff_date)
   - model = train_xgboost(X, y, config)
   - Save model to output_model_path
   - Log to MLflow
```

- [ ] Implement `models/training_pipeline.py`:

```
Module responsibilities: Orchestrate training with strict time splits.

1. get_time_split(
       all_feature_vectors_df: pd.DataFrame,
       config
   ) -> Tuple[datetime, datetime, datetime]
   - Find cutoff for training data (e.g., "7 days before max event_timestamp")
   - Return train_cutoff_date, test_start_date, test_end_date
   - CONSTRAINT: train_cutoff_date must account for label delay
     * If today is 2024-01-15, labels are ready for events up to 2024-01-13 (48h ago)
     * So train_cutoff_date = 2024-01-13

2. main(
       enriched_data_path,  # raw enriched transactions
       features_path,       # pre-computed feature vectors
       config,
       output_dir
   ) -> Dict
   - Load feature_vectors
   - Call get_time_split() -> train_cutoff, test_start, test_end
   - Stage 1: prepare_stage1_data(feature_vectors, train_cutoff)
   - Stage 1: train_isolation_forest(), save
   - Stage 1: score all feature_vectors (for stage 2 input)
   - Stage 2: prepare_stage2_data(feature_vectors, stage1_scores, train_cutoff)
   - Stage 2: train_xgboost(), save
   - Stage 2: score test set
   - Save both models to output_dir
   - Log to MLflow with run_id, train_cutoff, test dates
   - Return: {"stage1_model": path, "stage2_model": path, "metrics": {...}}
```

- [ ] Implement `models/evaluation.py`:

```
Module responsibilities: Evaluate at fixed alert budget (not accuracy metrics).

1. get_decisions_at_budget(
       scores: np.ndarray,
       budget_pct: float,  # e.g., 0.005 for 0.5%
       inverse: bool = True  # True if higher score = more likely fraud
   ) -> np.ndarray
   - Sort scores descending (or ascending if inverse=False)
   - Select top budget_pct% as fraud (1), rest as non-fraud (0)
   - Return binary decisions

2. compute_metrics(
       y_true: np.ndarray,
       decisions: np.ndarray,
       scores: np.ndarray
   ) -> Dict
   - TP = sum((y_true == 1) & (decisions == 1))
   - FP = sum((y_true == 0) & (decisions == 1))
   - FN = sum((y_true == 1) & (decisions == 0))
   - TN = sum((y_true == 0) & (decisions == 0))
   - precision = TP / (TP + FP) if TP + FP > 0 else 0
   - recall = TP / (TP + FN) if TP + FN > 0 else 0
   - false_alert_rate = FP / (FP + TN) if FP + TN > 0 else 0
   - auc_pr = auc(recall, precision)  # area under PR curve
   - Return {"TP": TP, "FP": FP, "FN": FN, "TN": TN, "precision": precision, "recall": recall, ...}

3. main(
       y_test,
       scores_test,
       config
   ) -> Dict
   - decisions = get_decisions_at_budget(scores_test, config.alert_budget_pct)
   - metrics = compute_metrics(y_test, decisions, scores_test)
   - Print summary table
   - Return metrics
```

- [ ] Implement `models/tests/test_no_label_leakage.py`:

```
Unit tests for label leakage.

def test_stage1_uses_no_labels():
    # Verify that stage 1 training does not use is_fraud
    # Read source code; if "is_fraud" appears in stage1_anomaly.py (except in prepare_stage2), fail

def test_train_test_time_disjoint():
    # Given train_cutoff_date = 2024-01-13, test_start = 2024-01-14
    # Verify: no feature_vector in training set has event_timestamp >= train_cutoff_date
    # Verify: all test feature_vectors have event_timestamp >= test_start

def test_stage2_trains_only_on_labeled_data():
    # Verify: stage 2 training set has no null is_fraud values
```

- [ ] Create `models/README.md`:

```
## Two-Stage Modeling

### Stage 1: Anomaly Detection (Isolation Forest)
- Input: Feature vectors (no labels used)
- Output: Anomaly score (0-1, 1 = most anomalous)
- Purpose: Cheap, broad filter. Catches obvious outliers.
- Assumption: Most historical data is clean (low contamination)

### Stage 2: Supervised Fraud Classifier (XGBoost)
- Input: Feature vectors + stage 1 anomaly scores
- Output: Fraud probability (0-1)
- Purpose: Targeted, precise detection using labeled examples
- Assumption: Labels are available and representative of fraud patterns

### Training Pipeline

1. Load feature vectors + enforce time split (train_cutoff_date)
2. Prepare stage 1 data (no is_fraud)
3. Train stage 1 model
4. Score all data with stage 1
5. Prepare stage 2 data (with is_fraud, only for event_timestamp < train_cutoff)
6. Train stage 2 model
7. Save both models

### Time Split

train_cutoff_date = max(event_timestamp) - label_delay_hours - buffer

Example:
- If max event in data is 2024-01-15 10:00 and label delay is 48h:
- Labels available for events up to 2024-01-13 10:00
- train_cutoff = 2024-01-13 09:00 (add 1h buffer for safety)
- test_start = 2024-01-14 00:00

Do NOT train on recent data; you'd use future labels.

### Evaluation

At fixed alert budget (e.g., 0.5% of transactions):
- Precision: of alerts we made, how many were correct?
- Recall: of all fraud, how much did we catch?
- False alert rate: of non-fraud txs, how many did we falsely alert?

Do NOT use accuracy; it's meaningless (fraud rate is low).
```


**Definition of "Done":**

- `python models/training_pipeline.py --config configs/project.yaml` trains both stage 1 and stage 2, saves models to `models/stage1_model.joblib` and `models/stage2_model.joblib`.
- `pytest models/tests/test_no_label_leakage.py -v` passes.
- You can explain: "Stage 1 is cheap anomaly filtering; stage 2 is precise fraud detection using labeled examples. We train on old data, test on new data, respecting label delay."

***

## PHASE 6: Evaluation \& Backtesting

**Goal:** Replay historical transactions day-by-day, apply trained models + alert policy, and measure performance under the fixed alert budget. Simulate scenarios (fraud spike, behavior shift) to understand system robustness.

**Why this matters:**

- You cannot evaluate offline and assume online will work identically.
- Real-time systems drift; backtesting reveals blind spots.
- Alert budget forces hard choices: "which fraud is most important to catch?"

**Theory to learn:**

- Search: *Backtesting financial systems, walk-forward validation*
- Search: *Concept drift in fraud detection, seasonal patterns*

**Tasks:**

- [ ] Create `evaluation/` folder:

```
evaluation/
  __init__.py
  backtest.py             # Day-by-day replay engine
  alert_policy.py         # Decision rule: given scores, which txs to alert?
  metrics.py              # Compute precision/recall/false_alert_rate
  scenarios.py            # Inject fraud spike, behavior shift, etc.
  visualize.py            # Plot metrics over time
  tests/
    test_alert_budget_respected.py
  README.md
```

- [ ] Implement `evaluation/alert_policy.py`:

```
Module responsibilities: Convert scores to alert decisions under budget constraint.

1. AlertPolicy (abstract base class):
    - Method: decide(scores: np.ndarray, budget_count: int) -> np.ndarray[bool]
    - Takes model scores and max number of alerts allowed
    - Returns binary array: True = alert, False = no alert
    - Contract: sum(decisions) <= budget_count

2. TopKPolicy(AlertPolicy):
    - Select the top K highest fraud scores and alert
    - If ties at boundary, use deterministic tiebreaker (e.g., lexicographic on transaction_id)

3. ThresholdPolicy(AlertPolicy):
    - Set a threshold T; alert if score >= T
    - Adjust T such that alert count ~= budget_count
    - Useful for A/B testing (compare to TopKPolicy)

4. main_select_policy(config) -> AlertPolicy
    - Read policy_type from config
    - Return appropriate instance
```

- [ ] Implement `evaluation/backtest.py`:

```
Module responsibilities: Day-by-day replay.

1. BacktestEngine:
    - __init__(
          offline_store_conn,
          feature_vectors_path,
          stage1_model_path,
          stage2_model_path,
          alert_policy: AlertPolicy,
          config
      )
    - Load models, feature vectors, offline store
    - Initialize results tracker
    
    - run(start_date: datetime, end_date: datetime, output_log_path: str) -> Dict
      * For each day in [start_date, end_date]:
        - Get all transactions for that day (from offline_store)
        - Get corresponding feature vectors
        - Score with stage 1 model
        - Score with stage 2 model
        - Apply alert_policy(stage2_scores, daily_budget)
        - Log: transaction_id, event_timestamp, true_label, stage1_score, stage2_score, alert_decision
        - Compute daily metrics (TP, FP, FN, precision, recall, false_alert_rate)
      * Save all logs to output_log_path
      * Return summary across all days

2. compute_cumulative_metrics(results_log: pd.DataFrame) -> Dict
    - Aggregate TP, FP, FN across all days
    - Compute overall precision, recall, false_alert_rate
    - Compute average daily alert count (verify budget respected)
    - Return dict

3. main(config, start_date, end_date, output_dir)
    - engine = BacktestEngine(...)
    - results = engine.run(start_date, end_date, output_dir + "/backtest_log.csv")
    - metrics = compute_cumulative_metrics(results)
    - Save metrics to output_dir + "/backtest_metrics.json"
    - Print summary table
```

- [ ] Implement `evaluation/scenarios.py`:

```
Module responsibilities: Inject scenario perturbations to test robustness.

1. inject_fraud_spike(
       backtest_log: pd.DataFrame,
       start_date: datetime,
       duration_days: int,
       spike_fraud_rate: float  # e.g., 0.05 (fraud rate increases to 5%)
   ) -> pd.DataFrame
   - For each txn in [start_date, start_date + duration_days]:
     * Randomly mark (1 - spike_fraud_rate) of txs as non-fraud
     * Mark spike_fraud_rate as fraud (overwrite true_label)
   - Return modified log

2. inject_behavior_shift(
       backtest_log: pd.DataFrame,
       start_date: datetime,
       duration_days: int,
       shift_type: str  # "more_night_txs", "higher_amounts", "new_merchants"
   ) -> pd.DataFrame
   - Modify feature values to reflect behavior shift
   - Example: more_night_txs -> increase txn_is_night_time percentage
   - Recompute model scores (may differ from baseline)
   - Return modified log

3. inject_model_degradation(
       backtest_log: pd.DataFrame,
       degradation_type: str  # "stale_features", "label_delay_increase"
   ) -> pd.DataFrame
   - stale_features: add 3-day delay to all feature computations
   - label_delay_increase: increase label availability to +72h (longer delay)
   - Simulate errors in feature pipelines, data quality issues

4. main(
       base_backtest_log_path,
       config,
       output_dir
   ) -> None
   - For each scenario:
     * Load base backtest log
     * Apply scenario injection
     * Re-compute metrics
     * Save scenario-specific results to output_dir/scenario_{name}.csv
   - Print comparison table (baseline vs each scenario)
```

- [ ] Implement `evaluation/visualize.py`:

```
Module responsibilities: Plot metrics over time.

import matplotlib.pyplot as plt

1. plot_metrics_over_time(backtest_log: pd.DataFrame, output_path: str) -> None
   - Extract daily precision, recall, false_alert_rate, daily_alert_count
   - Create 4-panel subplot:
     * Panel 1: Precision + Recall over time (line plot)
     * Panel 2: Daily alert count vs budget (bar chart, threshold line at budget)
     * Panel 3: False alert rate over time
     * Panel 4: Cumulative fraud caught (staircase plot)
   - Save to output_path (e.g., "evaluation/backtest_metrics.png")

2. plot_scenario_comparison(
       base_metrics: Dict,
       scenario_metrics: Dict[str, Dict],
       output_path: str
   ) -> None
   - Bar chart: precision, recall, false_alert_rate across [baseline, spike, shift, degradation]
   - Save to output_path

3. main(backtest_log_path, scenario_results_dir, output_dir)
    - plot_metrics_over_time(...)
    - For each scenario file in scenario_results_dir:
       * Load scenario metrics
       * plot_scenario_comparison(...)
```

- [ ] Implement `evaluation/tests/test_alert_budget_respected.py`:

```
def test_alert_budget_respected():
    # Run backtest for 1 day
    # Verify: daily_alert_count <= daily_budget for every day
    # If violated, fail

def test_alert_decisions_are_deterministic():
    # Run backtest twice with same config
    # Verify: same transactions are alerted both times

def test_scenario_spike_increases_false_alerts():
    # Run baseline backtest
    # Run with fraud spike scenario
    # Verify: spike scenario has higher false_alert_rate (more innocent txs alerted)
```

- [ ] Create `evaluation/README.md`:

```
## Backtesting & Evaluation

### Backtest Flow

1. Load models, features, offline store
2. For each day in test window:
   - Get transactions
   - Score with stage 1 + stage 2
   - Apply alert policy (respecting daily budget)
   - Log decisions + true labels
   - Compute daily metrics
3. Aggregate metrics across all days
4. Save log + metrics

### Metrics (at fixed alert budget)

- **Precision:** TP / (TP + FP)
  * Of alerts we made, how many were correct?

- **Recall:** TP / (TP + FN)
  * Of all fraud, how much did we catch?

- **False Alert Rate:** FP / (FP + TN)
  * Of innocent txs, what % did we falsely alert?

- **Daily Alert Count:** sum(decisions)
  * Must be <= budget for every day

### Scenarios

- **Fraud Spike:** Sudden increase in fraud rate (new attack)
- **Behavior Shift:** Normal users change patterns (more night txs, higher amounts)
- **Model Degradation:** Features become stale (3-day delay in ingestion)

Simulate each to understand system limits.

### Output

- `backtest_log.csv`: Per-transaction decisions, scores, labels
- `backtest_metrics.json`: Aggregated metrics
- `backtest_metrics.png`: Charts over time
- `scenario_*.csv`: Results for each scenario
```


**Definition of "Done":**

- `python evaluation/backtest.py --config configs/project.yaml --start-date 2024-01-01 --end-date 2024-01-31` produces `evaluation/backtest_log.csv` and `backtest_metrics.json`.
- `evaluation/backtest_metrics.json` shows precision, recall, false_alert_rate for each day, and confirms alert budget is respected.
- `pytest evaluation/tests/test_alert_budget_respected.py -v` passes.

***

## PHASE 7: Monitoring \& Failure Modes

**Goal:** Document failure modes explicitly. Implement minimal monitoring (logging + optional dashboard) to detect when the system is about to break. Design failure-safe defaults.

**Why this matters:**

- Production systems fail in unexpected ways.
- You must have explicit answers to: "What is the blind spot?", "What happens if X breaks?", "What's the safe default?"

**Tasks:**

- [ ] Create `monitoring/` folder:

```
monitoring/
  __init__.py
  drift_detection.py        # Score distribution, feature drift, fraud rate shifts
  failure_modes.md           # Document known blind spots and failure modes
  monitoring_logger.py       # Logging hooks for deployment
  alert_quality_checks.py    # Verify alerts make sense (sanity checks)
  README.md
```

- [ ] Create `monitoring/failure_modes.md`:

```
# Failure Modes & Blind Spots

## 1. New Device + Small Amount (Legitimate)

**Description:**
- User gets a new phone, tries first UPI txn with small amount (e.g., ₹10 to a verified merchant)
- Features: device_unique_payers = 1, payer_unique_devices = 1 (new), amount_normalized is high
- Isolation Forest may flag as anomaly (new device)
- If merchant is also new to payer: stage 2 might flag as fraud risk
- System may false-alert

**Blind Spot:**
- Graph features (device cardinality, etc.) don't distinguish "new phone (legitimate)" from "stolen phone (fraud)"

**Mitigation:**
- Add feature: is_device_new_but_amount_small (heuristic: device age < 24h AND amount < 50 INR)
- Downstream alert policy: lower stage 2 threshold for this case
- Monitor false_alert_rate on new-device transactions specifically

## 2. Sudden Fraud Rate Spike (New Attack Pattern)

**Description:**
- New attack emerges (e.g., coordinated SIM-swap targeting specific merchants)
- Actual fraud rate jumps from 1% to 3%
- System was trained on 1% baseline
- Alert budget is 0.5% of volume (fixed)
- System can catch at most 0.5%, so misses 2.5% of attack

**Blind Spot:**
- Model was trained on historical patterns; new pattern is out-of-distribution
- Stage 1 (anomaly) may catch some, but stage 2 (supervised) has no labeled examples

**Mitigation:**
- Stage 1 anomaly scores should rise when attack begins (detected as "weird")
- Monitoring: alert if daily_anomaly_scores drift upward (moving average)
- Human review: if anomaly score distribution shifts, retrain stage 1 urgently
- Escalation: if fraud catch rate drops below threshold, notify ops to adjust policy

## 3. Feature Computation Lag (Stale History)

**Description:**
- Ingestion pipeline is delayed by 3+ hours
- Velocity features (last 5 min, 1 hour) are stale
- Decision made on 3-hour-old history
- Fraudster makes 2 txs in the stale window; system only sees first txn's velocity

**Blind Spot:**
- Real-time velocity check fails
- High-speed attack (10 txs in 5 min) appears as 2–3 txs in delayed history

**Mitigation:**
- Monitoring: track ingestion lag (current_time - latest_event_timestamp)
- Alert if lag > configured threshold (e.g., 5 min)
- Fallback: if lag too high, raise stage 2 threshold (fewer alerts, lower risk of bad decisions)
- Post-mortem: investigate why ingestion is delayed

## 4. Label Delay Exceeds 48h (Labeling System Broken)

**Description:**
- Fraud label backfill is delayed (labels not arriving for 72+ hours)
- You can only train on old data (event_timestamp < 3 days ago)
- Training lag increases; model is older

**Blind Spot:**
- Cannot retrain frequently; cannot adapt to recent patterns

**Mitigation:**
- Monitoring: track max(event_timestamp where label is available)
- Alert if label_max_age > 60 hours
- Escalation: alert SRE to investigate labeling pipeline
- Fallback: use stage 1 anomaly model only (no retraining needed)

## 5. Model Performance Degrades (Data Drift)

**Description:**
- Test set precision was 80%; production precision is 60%
- Feature distributions changed (e.g., more nighttime txs, different merchants)
- Model is no longer calibrated for live data

**Blind Spot:**
- Detection: model metrics are correct (we compute them), but don't match reality
- Cause: train set is not representative of production

**Mitigation:**
- Monitoring: compare feature distributions (train vs live) using statistical tests
- Feature-specific: track rolling mean of key features (velocity, graph cardinality)
- Alert if any key feature distribution drifts > threshold (e.g., KL-div > 0.1)
- Trigger retraining if drift detected

## 6. Alert Policy is Unjust (Cheating Budget)

**Description:**
- Alert policy is supposed to respect daily budget (0.5% of volume)
- Implementation has a bug: alerts top K by score, but K is computed wrong
- Actual alert volume is 1% (double budget)
- System appears high-performing (catches more fraud) but violates policy

**Blind Spot:**
- Not a blind spot; it's a bug
- Reveals: contract between alert policy and monitoring is unclear

**Mitigation:**
- Test: `pytest evaluation/tests/test_alert_budget_respected.py`
- Must pass every day
- Monitoring: daily_alert_count must be <= budget_count
- Alert if violated

## 7. Completely New VPA (Out-of-Distribution)

**Description:**
- Fraudster creates a new merchant VPA and immediately receives 10k txs from stolen accounts
- Payee_id + payee_vpa is completely new (never seen before)
- payer_unique_payees_1d = 1 (payer has only paid this merchant today)
- Graph features (if they exist) are 0 or undefined

**Blind Spot:**
- New merchants are expected (legitimate); can't flag all as fraud
- Stage 2 classifier: new merchant is unusual but not necessarily fraud feature

**Mitigation:**
- If payee is < 1 hour old, use different threshold (lower tolerance for fraud)
- Stage 1 anomaly should catch (new payee + high volume = anomaly)
- Downstream: monitor payee_creation_rate; if spike, investigate

---

## Safe Defaults

If any of the following are true:
- Ingestion lag > 5 min: use Stage 1 only, raise stage 2 threshold
- Label availability age > 60 h: use Stage 1 only
- Feature computation fails: reject transaction (fail-safe)
- Model inference fails: reject transaction
- Alert budget is exhausted: reject remaining transactions (do not alert)

Reject = log detailed error, alert monitoring, return "no alert" to user.
```

- [ ] Implement `monitoring/drift_detection.py`:

```
Module responsibilities: Detect when distributions change.

1. FeatureDriftDetector:
    - __init__(reference_df, config)
      * Store reference feature distributions (mean, std, quantiles, histogram)
    
    - detect_drift(current_df, features_to_check: List[str]) -> Dict
      * For each feature:
        - Compute KL-divergence between reference and current histograms
        - Compute t-test if numeric (mu, sigma changed?)
        - Return: {feature: {kl_div, t_stat, p_value, drifted: bool}}

2. FraudRateDriftDetector:
    - __init__(reference_fraud_rate, tolerance_pct)
    - detect_drift(current_labels, current_period_name) -> Dict
      * Compute current fraud rate
      * If |current - reference| > tolerance_pct, mark as drifted
      * Return: {fraud_rate, reference, delta, drifted}

3. AnomalyScoreDriftDetector:
    - Similar: monitor stage 1 anomaly scores
    - If mean score rises, new anomalies appearing
    
4. main(reference_data, current_batch, config, output_path)
    - Create detectors
    - Run all checks
    - Save results to output_path
    - Alert if any drift detected
```

- [ ] Implement `monitoring/monitoring_logger.py`:

```
Module responsibilities: Structured logging for production.

class MonitoringLogger:
    def __init__(self, log_file_path, config):
        self.file = open(log_file_path, "a")
        self.config = config
    
    def log_transaction_decision(
        self,
        transaction_id,
        event_timestamp,
        stage1_score,
        stage2_score,
        alert_decision,
        latency_ms
    ):
        # Structured log (JSON)
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "transaction_id": transaction_id,
            "event_timestamp": event_timestamp.isoformat(),
            "stage1_score": stage1_score,
            "stage2_score": stage2_score,
            "alert": alert_decision,
            "latency_ms": latency_ms
        }
        self.file.write(json.dumps(entry) + "\n")
        self.file.flush()
    
    def log_drift_detected(self, feature_name, kl_div, threshold):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "drift_detected",
            "feature": feature_name,
            "kl_divergence": kl_div,
            "threshold": threshold
        }
        self.file.write(json.dumps(entry) + "\n")
        self.file.flush()
    
    def log_error(self, error_type, message):
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": "error",
            "error_type": error_type,
            "message": message
        }
        self.file.write(json.dumps(entry) + "\n")
        self.file.flush()
```

- [ ] Create `monitoring/README.md`:

```
## Monitoring & Observability

### What to Monitor

1. **Ingestion Health**
   - Lag: current_time - latest_event_timestamp (should be < 5 min)
   - Data quality: Great Expectations soft warnings per micro-batch

2. **Model Performance**
   - Daily precision, recall, false_alert_rate
   - Anomaly score distribution (mean, std)

3. **Feature Drift**
   - Key features: payer_txn_count_5min, device_unique_payers, txn_is_night_time, etc.
   - KL-divergence vs reference (alert if > 0.1)

4. **Fraud Rate Drift**
   - Observed fraud rate vs historical baseline
   - Alert if deviation > 2%

5. **Alert Budget**
   - Daily alert count vs daily budget
   - Alert if violated

6. **Latency**
   - Per-transaction scoring latency (should be < 100 ms)
   - Alert if P95 > 500 ms

### Dashboards (Optional Streamlit)

Could build a simple Streamlit app that reads monitoring logs and displays:
- Real-time transaction decisions
- Daily metrics (precision, recall, alert count)
- Feature drift time series
- Anomaly score distribution

Not required for Phase 7, but nice to have.

### Failure Modes (See failure_modes.md)

Document all known blind spots and safe defaults.
```


**Definition of "Done":**

- `monitoring/failure_modes.md` is written and reviewed (identifies >= 5 failure modes).
- `monitoring/drift_detection.py` can detect feature drift (KL-div) and fraud rate drift.
- `monitoring/monitoring_logger.py` logs decisions + drift signals to a file in structured JSON format.

***

## PHASE 8: API Service \& Integration

**Goal:** Build a FastAPI service that exposes the scoring logic, integrates with the feature store, and can be called by the streaming simulator (Phase 2) or a production system.

**Tasks:**

- [ ] Create `api/` folder:

```
api/
  __init__.py
  main.py                 # FastAPI app
  models.py               # Pydantic request/response schemas
  service.py              # Scoring logic (orchestrates stage 1, stage 2, decision)
  feature_lookup.py       # Query online feature store or offline store
  health.py               # Health check endpoints
  README.md
```

- [ ] Implement `api/models.py`:

```
from pydantic import BaseModel
from datetime import datetime

class ScoringRequest(BaseModel):
    transaction: Transaction  # From ingestion/schema.py

class ScoringResponse(BaseModel):
    transaction_id: str
    event_timestamp: datetime
    stage1_score: float
    stage2_score: float
    alert_decision: bool
    decision_reasoning: str  # e.g., "score in top 0.5%", "anomaly detected", "no alert"
    latency_ms: float
```

- [ ] Implement `api/service.py`:

```
class ScoringService:
    def __init__(self, config, stage1_model_path, stage2_model_path):
        self.config = config
        self.stage1_model = load_model(stage1_model_path)
        self.stage2_model = load_model(stage2_model_path)
        self.feature_store = OnlineFeatureStore(config)
        self.alert_policy = TopKPolicy()
        self.monitoring_logger = MonitoringLogger(...)
    
    def score(self, transaction: Transaction) -> ScoringResponse:
        start_time = time.time()
        
        try:
            # 1. Compute features
            features = self.feature_store.compute_features_for_scoring(transaction)
            
            # 2. Stage 1 (anomaly)
            X_1 = features.to_feature_vector()  # Convert to array
            stage1_score = self.stage1_model.predict(X_1)
            
            # 3. Stage 2 (supervised)
            X_2 = append_anomaly_score(X_1, stage1_score)
            stage2_score = self.stage2_model.predict_proba(X_2)[0, 1]
            
            # 4. Alert decision
            # (This is tricky: you don't know the daily budget utilization yet)
            # Fallback: use threshold-based policy for now
            threshold = self.config.models.stage2_threshold
            alert = stage2_score >= threshold
            
            # 5. Log
            latency_ms = (time.time() - start_time) * 1000
            self.monitoring_logger.log_transaction_decision(...)
            
            # 6. Respond
            return ScoringResponse(
                transaction_id=transaction.transaction_id,
                event_timestamp=transaction.event_timestamp,
                stage1_score=stage1_score,
                stage2_score=stage2_score,
                alert_decision=alert,
                decision_reasoning="...",
                latency_ms=latency_ms
            )
        
        except Exception as e:
            self.monitoring_logger.log_error(...)
            # Fail-safe: no alert
            return ScoringResponse(..., alert_decision=False, decision_reasoning="error")
```

- [ ] Implement `api/main.py`:

```
from fastapi import FastAPI, HTTPException
import uvicorn

app = FastAPI()

@app.on_event("startup")
def startup():
    global service
    service = ScoringService(config, stage1_model_path, stage2_model_path)

@app.post("/score", response_model=ScoringResponse)
def score(req: ScoringRequest):
    return service.score(req.transaction)

@app.get("/health")
def health():
    return {"status": "ok", "version": "1.0"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
```


**Definition of "Done":**

- `python api/main.py` starts FastAPI on localhost:8000.
- `curl -X POST http://localhost:8000/score -H "Content-Type: application/json" -d '{"transaction": {...}}'` returns a ScoringResponse.

***

## PHASE 9: Repo Structure \& Documentation

**Goal:** Organize all code, tests, and docs into a cohesive, navigable repository. Write README that tells the full story.

**Tasks:**

- [ ] Finalize folder structure:

```
upi_fraud_system/
  README.md                       # Main entry point; links to DESIGN.md, LEARNINGS.md
  DESIGN.md                       # System architecture, data flow, label delay
  LEARNINGS.md                    # Concept explanations (updated each phase)
  
  configs/
    project.yaml                  # All hyperparameters
    
  data_generation/
    __init__.py
    generate_data.py
    enrich_to_upi_schema.py
    fraud_injector.py
    validator.py
    README.md
    
  ingestion/
    __init__.py
    schema.py
    batch_loader.py
    streaming_simulator.py
    consistency_check.py
    README.md
    
  validation/
    __init__.py
    build_suite.py
    run_validation.py
    README.md
    
  features/
    __init__.py
    schema.py
    offline_builder.py
    online_builder.py
    feature_definitions.py
    time_utils.py
    tests/
      __init__.py
      test_time_correctness.py
    README.md
    
  models/
    __init__.py
    stage1_anomaly.py
    stage2_supervised.py
    training_pipeline.py
    evaluation.py
    mlflow_utils.py
    model_schemas.py
    tests/
      __init__.py
      test_no_label_leakage.py
    README.md
    
  evaluation/
    __init__.py
    backtest.py
    alert_policy.py
    metrics.py
    scenarios.py
    visualize.py
    tests/
      __init__.py
      test_alert_budget_respected.py
    README.md
    
  monitoring/
    __init__.py
    drift_detection.py
    failure_modes.md
    monitoring_logger.py
    alert_quality_checks.py
    README.md
    
  api/
    __init__.py
    main.py
    models.py
    service.py
    feature_lookup.py
    health.py
    README.md
    
  ui/ (optional)
    streamlit_app.py             # If you build a dashboard
    
  tests/
    __init__.py
    conftest.py                  # Pytest fixtures
    test_integration.py           # End-to-end tests
    
  scripts/
    run_pipeline.sh              # Orchestrate data gen → training → backtest
    setup_env.sh                 # Create DuckDB, set env vars
    
  .gitignore
  requirements.txt
  pytest.ini
```

- [ ] Create top-level `README.md`:

```
# UPI-Style Real-Time Fraud Decision System

A production-grade fraud detection system built to make real-time decisions on payment transactions under a fixed alert budget.

## Quick Start

```bash
# 1. Setup
python -m pip install -r requirements.txt
bash scripts/setup_env.sh

# 2. Generate data
python data_generation/generate_data.py \
    --ieee_cis_path /path/to/ieee_cis.csv \
    --output_path data/enriched_transactions.duckdb \
    --config_path configs/project.yaml

# 3. Load data (batch ingestion)
python ingestion/batch_loader.py \
    --enriched_duckdb_path data/enriched_transactions.duckdb \
    --offline_store_path data/offline_store.duckdb \
    --config_path configs/project.yaml

# 4. Build features (offline)
python features/offline_builder.py \
    --offline_store_path data/offline_store.duckdb \
    --output_feature_store_path data/feature_vectors.parquet \
    --config_path configs/project.yaml

# 5. Train models
python models/training_pipeline.py \
    --enriched_data_path data/enriched_transactions.duckdb \
    --features_path data/feature_vectors.parquet \
    --config_path configs/project.yaml \
    --output_dir models/

# 6. Backtest
python evaluation/backtest.py \
    --config configs/project.yaml \
    --start-date 2024-01-01 \
    --end-date 2024-01-31 \
    --output-dir evaluation/results/

# 7. Run API (live scoring)
python api/main.py

# 8. Stream data (in another terminal)
python ingestion/streaming_simulator.py \
    --enriched_duckdb_path data/enriched_transactions.duckdb \
    --api_endpoint http://localhost:8000/score \
    --config_path configs/project.yaml \
    --output_log_path evaluation/streaming_results.json
```


## Architecture

See [DESIGN.md](DESIGN.md) for data flow diagrams and label delay explanation.

## Key Concepts

See [LEARNINGS.md](LEARNINGS.md) for deep dives on:
    - Label delay and why it matters
    - Temporal leakage and point-in-time correctness
    - Alert budget and precision@K
    - Concept drift vs data drift


## Decision Problem (The North Star)

> At transaction time T, using ONLY information available strictly before T, decide: Alert or Not Alert.
> We have a fixed daily alert budget of K alerts (e.g., 0.5% of volume).

Everything in this project serves this constraint.

## Phases

1. **Data Simulation** (Phase 1): Enrich IEEE-CIS → UPI-like schema
2. **Ingestion** (Phase 2): Batch + streaming paths, consistency guaranteed
3. **Validation** (Phase 3): Great Expectations suites for quality gates
4. **Feature Engineering** (Phase 4): Time-correct offline + online features
5. **Modeling** (Phase 5): Two-stage (anomaly + supervised)
6. **Backtesting** (Phase 6): Day-by-day replay, alert budget respected
7. **Monitoring** (Phase 7): Drift detection, failure modes, safe defaults
8. **API** (Phase 8): FastAPI service for live scoring

## Testing

```bash
# Run all tests
pytest tests/ features/tests/ models/tests/ evaluation/tests/ -v
```


## Configuration

Edit [configs/project.yaml](configs/project.yaml) to tune:
    - Alert budget (%)
    - Fraud pattern injection rates
    - Feature window sizes
    - Model hyperparameters


## Production Deployment

(Future: Kafka ingestion, model serving, A/B testing framework)

For now: all-in-memory simulation. No Kafka, no Kubernetes.

## Authors

Built as a learning project on fraud systems architecture.

```

```

- [ ] Update `LEARNINGS.md` with final summary:

```
# Learning Summary

## Concepts Covered

1. **Label Delay** (Phase 0–1)
2. **Temporal Leakage** (Phase 4)
3. **Alert Budget as Primary Constraint** (Phase 6)
4. **Concept vs Data Drift** (Phase 7)
5. **Two-Stage Modeling** (Phase 5)
6. **Point-in-Time Correctness** (Phase 4)
7. **Batch-Streaming Consistency** (Phase 2)
8. **Failure Modes & Safe Defaults** (Phase 7)

## Common Mistakes (and How We Avoided Them)

### Mistake 1: Training on Future Labels
- **Problem:** Use today's labels (from yesterday's events) to train on today's events. Leakage.
- **Solution:** time_split() enforces train_cutoff_date respects label_delay.
- **Verification:** test_no_label_leakage.py catches this.

### Mistake 2: Feature Leakage (Using Future Data)
- **Problem:** Compute "max amount user sent in last 5 min" at T, but include T itself or T+1.
- **Solution:** get_past_transactions_in_window() uses strict < not <=.
- **Verification:** test_time_correctness.py catches this.

### Mistake 3: Ignoring Alert Budget
- **Problem:** Maximize accuracy; alert on top 2% of scores. But budget is 0.5%.
- **Solution:** AlertPolicy enforces budget; evaluation metrics use precision@K.
- **Verification:** test_alert_budget_respected.py catches this.

### Mistake 4: Batch ≠ Streaming
- **Problem:** Batch pipeline computes different features than streaming API.
- **Solution:** Both use same Transaction schema, same feature definitions, same time logic.
- **Verification:** consistency_check.py compares side-by-side.

### Mistake 5: No Blind Spot Analysis
- **Problem:** Model works in backtest; fails in production (new attack).
- **Solution:** Document failure modes explicitly. Design safe defaults.
- **Result:** failure_modes.md + monitoring/ catches issues early.

## If You Wanted to Deploy This

1. Replace Python event generator with Kafka consumer
2. Replace in-memory feature store with Redis or DuckDB with fast indexing
3. Implement proper alert budget tracking (across all instances, not per-process)
4. Add retraining pipeline (schedule models to retrain weekly)
5. Add A/B testing framework (compare alert policies)
6. Wrap API with authentication, rate limiting, observability
7. Build ops dashboard (monitoring, drift alerts, manual review queue)

But the core logic is unchanged. Phase 1–7 principles scale to production.
```


**Definition of "Done":**

- Repo is organized, navigable, fully documented.
- `README.md` tells a cohesive story and has a quick-start section.
- Every folder has a `README.md` explaining its responsibility.
- `pytest` runs all tests and they pass.

***

## FINAL CHECKLIST: "DONE WITH PROJECT"

**You can call this project "done" when:**

- [ ] You can explain the decision problem (constraint) from memory
- [ ] You can run the entire pipeline end-to-end: `data_generation → ingestion → features → training → backtesting`
- [ ] Backtesting shows: precision, recall, false_alert_rate all computed correctly, and alert budget is respected every day
- [ ] Streaming simulator can replay data to the API and get identical feature vectors as batch
- [ ] All tests pass: `pytest tests/ features/tests/ models/tests/ evaluation/tests/ -v`
- [ ] You've documented >= 5 failure modes and for each one, specified a safe default or mitigation
- [ ] You can point to the code and explain why temporal leakage is prevented
- [ ] You've built a two-stage model and can explain why (anomaly filter + supervised classifier)
- [ ] You have a working API that scores transactions

***

## REPO STRUCTURE SUMMARY

```
upi_fraud_system/
├── README.md                          ← START HERE
├── DESIGN.md                          ← Data flow, label delay, alert budget math
├── LEARNINGS.md                       ← Concept deep dives, common mistakes
├── requirements.txt
├── pytest.ini
├── .gitignore
│
├── configs/
│   └── project.yaml                   ← All hyperparameters
│
├── data_generation/                   ← Phase 1
│   ├── generate_data.py
│   ├── enrich_to_upi_schema.py
│   ├── fraud_injector.py
│   ├── validator.py
│   └── README.md
│
├── ingestion/                         ← Phase 2
│   ├── schema.py
│   ├── batch_loader.py
│   ├── streaming_simulator.py
│   ├── consistency_check.py
│   └── README.md
│
├── validation/                        ← Phase 3
│   ├── build_suite.py
│   ├── run_validation.py
│   └── README.md
│
├── features/                          ← Phase 4
│   ├── schema.py
│   ├── offline_builder.py
│   ├── online_builder.py
│   ├── feature_definitions.py
│   ├── time_utils.py
│   ├── tests/
│   │   └── test_time_correctness.py
│   └── README.md
│
├── models/                            ← Phase 5
│   ├── stage1_anomaly.py
│   ├── stage2_supervised.py
│   ├── training_pipeline.py
│   ├── evaluation.py
│   ├── mlflow_utils.py
│   ├── model_schemas.py
│   ├── tests/
│   │   └── test_no_label_leakage.py
│   └── README.md
│
├── evaluation/                        ← Phase 6
│   ├── backtest.py
│   ├── alert_policy.py
│   ├── metrics.py
│   ├── scenarios.py
│   ├── visualize.py
│   ├── tests/
│   │   └── test_alert_budget_respected.py
│   └── README.md
│
├── monitoring/                        ← Phase 7
│   ├── drift_detection.py
│   ├── failure_modes.md
│   ├── monitoring_logger.py
│   ├── alert_quality_checks.py
│   └── README.md
│
├── api/                               ← Phase 8
│   ├── main.py
│   ├── models.py
│   ├── service.py
│   ├── feature_lookup.py
│   ├── health.py
│   └── README.md
│
├── scripts/
│   ├── run_pipeline.sh
│   └── setup_env.sh
│
└── tests/
    ├── conftest.py
    └── test_integration.py
```


***

## WHY THIS ROADMAP WORKS

1. **Phases are strictly ordered.** You cannot jump to Phase 5 (modeling) without Phase 1–4 (data, ingestion, validation, features). Each phase builds on the previous.
2. **Time discipline is enforced everywhere.** Every file that touches timestamps has a comment explaining the rule. Tests verify it.
3. **Two paths (batch + streaming) stay in sync.** If they ever diverge, `consistency_check.py` catches it.
4. **Alert budget is a first-class constraint.** Not an afterthought. Backtest will fail if you violate it.
5. **Failure modes are documented.** You know what can go wrong and what to do (safe defaults).
6. **Code is testable.** Every phase includes unit tests that catch leakage, budget violations, and schema mismatches.
7. **No notebooks as source of truth.** Everything is scripts/modules. Notebooks are exploratory only.
8. **Scaling is possible.** Replace Python event generator with Kafka, in-memory state with Redis, DuckDB with Postgres. The interfaces don't change.

***

## FINAL ADVICE

**Before you start coding:**

1. Read DESIGN.md carefully. Understand the data flow.
2. Write down the decision constraint in your own words.
3. Skim LEARNINGS.md. Know the concepts before implementing.

**While coding:**

1. Respect the order. Do Phase 1 before Phase 2.
2. If you feel lost, re-read the phase goal and tasks.
3. Run tests frequently. They catch mistakes early.
4. Document as you go. Update LEARNINGS.md after each phase.

**After coding:**

1. Can you explain each phase in 2–3 sentences?
2. Can you run the pipeline end-to-end?
3. Can you read a transaction and trace it through features → models → decision?
4. Can you explain why batch and streaming are identical?
5. Can you describe your blind spots?

If yes to all: you've built a real system. You can talk about it in interviews.

This is not just a project. It's a portfolio piece that shows you understand:

- Real-time systems design
- Temporal correctness
- Constraints as first-class concepts
- Failure analysis
- Production thinking

Good luck. Build it. Ship it. Learn from it.

