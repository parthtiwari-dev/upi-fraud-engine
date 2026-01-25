# Project Navigation Guide

This guide helps you understand where everything is and how to navigate the project.

---

## 🗺️ High-Level Map

```
Real-Time UPI Fraud Detection System
│
├─ 📊 UNDERSTAND THE PROBLEM
│  └─ README.md (this folder)
│     • Problem statement
│     • Architecture diagram
│     • Tech stack choices
│     • Key results
│
├─ 📚 DEEP DIVE INTO DESIGN
│  └─ docs/
│     ├─ design.md              (Overall system architecture + data flow)
│     ├─ learning.md            (Key concepts: label delay, leakage, budget)
│     └─ PHASE_X_README.md      (1-8, detailed walkthrough of each phase)
│
├─ 💾 DATA PIPELINE
│  ├─ data/                      (IEEE-CIS fraud data - you add this)
│  ├─ data_generation/           (Transform card data → UPI schema)
│  └─ config/project.yaml        (Fraud injection rates, feature windows)
│
├─ 🔧 FEATURE ENGINEERING
│  └─ src/features/
│     ├─ offline_builder.py      (Batch: compute all features)
│     ├─ online_builder.py       (Streaming: stateful updates)
│     ├─ feature_definitions.py  (Individual feature logic)
│     ├─ time_utils.py           (Point-in-time correctness)
│     └─ test_time_correctness.py (Prevent leakage - CRITICAL)
│
├─ 🤖 MODEL TRAINING
│  └─ src/models/
│     ├─ stage1_anomaly.py       (Isolation Forest: catches anomalies)
│     ├─ stage2_supervised.py    (XGBoost: precise fraud detection)
│     ├─ training_pipeline.py    (Orchestrate both stages)
│     └─ tests/                  (Verify no label leakage)
│
├─ 📈 EVALUATION & BACKTESTING
│  ├─ src/evaluation/
│  │  ├─ backtest.py            (Day-by-day replay with alert budget)
│  │  ├─ alert_policy.py        (Decision rules: which txns to alert?)
│  │  ├─ metrics.py             (Precision, recall, false_alert_rate)
│  │  └─ visualize.py           (Plot results over time)
│  │
│  └─ evaluation/
│     ├─ backtest_results/      (Daily metrics, alert compliance)
│     └─ visualizations/        (Precision/recall trends, confusion matrix)
│
├─ 🚀 PRODUCTION DEPLOYMENT
│  ├─ src/api/                   (FastAPI service)
│  ├─ app.py                     (Streamlit UI)
│  ├─ dockerfile                 (Docker config for Render)
│  └─ docs/PHASE_8_README.md     (Step-by-step deployment)
│
├─ ✅ TESTING
│  ├─ src/features/test/         (Feature leakage tests)
│  ├─ src/models/tests/          (Label leakage tests)
│  ├─ src/evaluation/tests/      (Alert budget tests)
│  └─ tests/                     (Integration tests)
│
└─ 📋 CONFIGURATION
   └─ config/project.yaml        (All hyperparameters in one place)
```

---

## 🎯 For Different Audiences

### **For a Recruiter (5 min read)**
1. Start with **README.md** (this folder)
2. Skim **Problem Statement** + **Solution Architecture**
3. Check **Key Results** + **Interview Talking Points**
4. Visit live URLs: API + UI

**Time:** 5 minutes, understand the whole project

---

### **For an Interviewer (15 min deep dive)**
1. Read **README.md** (architecture, choices)
2. Skim **docs/design.md** (data flow, label delay)
3. Trace through **Phase 4: Feature Engineering** (time correctness)
4. Look at **Phase 6: Backtesting** (alert budget constraint)
5. Ask follow-up questions on trade-offs

**Time:** 15 minutes, understand design decisions

---

### **For a Junior Engineer Building Similar Systems**
1. Start with **docs/learning.md** (key concepts)
2. Read **Phase 1-8 READMEs** in sequence
3. Look at actual code:
   - `src/features/test_time_correctness.py` (how to verify no leakage)
   - `src/evaluation/backtest.py` (how to enforce budget)
   - `src/models/training_pipeline.py` (time-correct train/test split)
4. Run locally following **How to Use** section in README.md

**Time:** 4-6 hours, fully understand the system

---

### **For Production ML/Data Engineers**
Focus on:
- **Phase 3:** Great Expectations (data validation)
- **Phase 4:** Feature store design (batch + online parity)
- **Phase 6:** Backtesting framework (alert budget enforcement)
- **Phase 7:** API design (error handling, monitoring)
- **Phase 8:** Docker + Render (minimal ops overhead)

Ask: "How would you scale to 100k txn/sec?" → Answer: Replace Python event generator with Kafka, in-memory features with Redis, local DuckDB with Postgres.

**Time:** 30 min focused read

---

## 🔑 Key Files to Understand the System

### **#1: src/features/test_time_correctness.py**
**Why:** Shows how to verify features don't leak future data  
**What to look for:** Test cases that would FAIL if you accidentally included T+1 in a T window

```python
# Example: Feature at time T must not see events at T+1
def test_no_future_features():
    # Given: txn at time 10:00
    # Only txns before 10:00 should be counted
    # Txns at/after 10:00 should NOT be counted
```

---

### **#2: src/models/training_pipeline.py**
**Why:** Shows how to respect label delay when splitting train/test  
**What to look for:** Logic that says "train on events up to 2 days ago, test on next day"

```python
# Example: train_cutoff = max_event_time - 48h - buffer
# Never train on recent events (labels not available yet)
```

---

### **#3: src/evaluation/backtest.py**
**Why:** Shows alert budget enforcement in practice  
**What to look for:** Day-by-day loop that:
  1. Gets all txns for day D
  2. Scores them
  3. Applies alert policy (select top K to alert)
  4. Computes metrics

```python
# Example: if alert_budget_pct=0.5%, and daily_volume=100k
# Then we alert on exactly 500 txns (not 600, not 400)
```

---

### **#4: docs/design.md**
**Why:** System-level data flow and label delay diagram  
**What to look for:** Shows how batch and streaming are identical, where leakage can happen

---

### **#5: config/project.yaml**
**Why:** Single source of truth for all hyperparameters  
**What to look for:** 
  - Alert budget percentage
  - Fraud injection rates (testing)
  - Feature windows
  - Model hyperparameters

---

## 📊 Folder Responsibility Matrix

| Folder | Responsibility | Key File |
|--------|-----------------|----------|
| `data_generation/` | Transform IEEE-CIS → UPI schema | `generate_data.py` |
| `src/features/` | Build 482 features (time-correct) | `offline_builder.py` |
| `src/models/` | Stage 1 (anomaly) + Stage 2 (supervised) | `training_pipeline.py` |
| `src/ingestion/` | Batch + streaming paths | `batch_loader.py` |
| `src/validation/` | Great Expectations suites | `run_validation.py` |
| `src/evaluation/` | Backtest + metrics | `backtest.py` |
| `src/api/` | FastAPI service | `main.py` |
| `src/inference/` | Single transaction scoring | `single_predict.py` |
| `models/` | Trained artifacts (pickle, JSON) | `production/fraud_detector.json` |
| `evaluation/` | Backtest results + visualizations | `backtest_results/daily_metrics.csv` |
| `great_expectations/` | Data quality expectations | `expectations/transaction_schema.json` |
| `docs/` | Phase-by-phase detailed docs | `PHASE_X_README.md` |

---

## 🚀 Quick Start (Step-by-Step)

### Step 1: Get Data
```bash
# Download from Kaggle (you do this manually)
# https://www.kaggle.com/c/ieee-fraud-detection/data
# Place in data/ folder
```

### Step 2: Generate UPI Data
```bash
python data_generation/generate_data.py \
    --ieee_cis_path data/train_transaction.csv \
    --output_path data/enriched_transactions.duckdb
```

### Step 3: Build Features
```bash
python src/features/offline_builder.py \
    --offline_store data/enriched_transactions.duckdb \
    --output_feature_store data/feature_vectors.parquet
```

### Step 4: Train Models
```bash
python src/models/training_pipeline.py \
    --features_path data/feature_vectors.parquet
```

### Step 5: Backtest
```bash
python src/evaluation/backtest.py \
    --start_date 2024-01-01 \
    --end_date 2024-01-31
```

### Step 6: Run Locally
```bash
# Terminal 1
python src/api/main.py

# Terminal 2
streamlit run app.py
```

### Step 7: Deploy (Optional)
See `docs/PHASE_8_README.md`

---

## 🧪 Running Tests

```bash
# All tests
pytest tests/ src/ -v

# Specific suites
pytest src/features/test_time_correctness.py -v      # No leakage
pytest src/models/tests/test_no_label_leakage.py -v   # Time split OK
pytest src/evaluation/tests/ -v                       # Alert budget OK
```

**Expected:** 24/24 passing ✅

---

## 🎓 What Each Phase Teaches

| Phase | Focus | Key Lesson |
|-------|-------|-----------|
| **Phase 1** | Data generation | Understand your data deeply; synthetic fraud is real |
| **Phase 2** | Batch + streaming | Consistency is hard; test both paths |
| **Phase 3** | Data validation | Quality gates prevent bad data downstream |
| **Phase 4** | Feature engineering | Temporal correctness is not optional |
| **Phase 5** | Two-stage modeling | Cheap filter + precise classifier = robust |
| **Phase 6** | Backtesting | Alert budget is a first-class constraint |
| **Phase 7** | API service | Packaging matters; docs help users |
| **Phase 8** | Deployment | Docker + simple PaaS = accessible production |

---

## ❓ FAQ

### Q: Where do I add the IEEE-CIS data?
A: Download from Kaggle, place in `data/` folder (it's git-ignored).

### Q: How do I know features don't leak?
A: Run `pytest src/features/test_time_correctness.py`. Tests would fail if you accidentally included future data.

### Q: Why two models, not one?
A: Stage 1 is fast (catches obvious anomalies), Stage 2 is precise (uses labeled examples). Together they're robust to new fraud patterns.

### Q: Can I change the alert budget?
A: Yes, edit `config/project.yaml` and rerun backtest. System is designed to be budget-flexible.

### Q: How do I deploy to production?
A: See `docs/PHASE_8_README.md`. Docker + Render = 10 minutes.

### Q: What if I want to retrain weekly?
A: See Phase 9 (future). Requires: scheduling (cron/Airflow), label backfill pipeline, A/B testing framework.

---

## 📞 Contact

Questions about this project? Check the docs first:
- **Architecture questions?** → `docs/design.md`
- **Why did you do X?** → Relevant `PHASE_X_README.md`
- **How do I run it?** → `README.md` (this folder)
- **I'm rebuilding this.** → `docs/learning.md` (key concepts to avoid traps)

---

## 🎯 Next Steps

1. **Read this README** (you're here!)
2. **Understand the problem** by reading Phase 1-2 docs
3. **Run locally** following quickstart steps
4. **Backtest** and see alert budget in action
5. **Deploy** to Render if you want live demo
6. **Share** the live URLs in your portfolio

---

**Good luck! This is a real-world ML system. Build it, understand it, own it.**

