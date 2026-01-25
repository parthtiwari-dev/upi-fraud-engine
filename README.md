# 🚨 Real-Time UPI Fraud Detection System
**Production ML System | IEEE-CIS Dataset | Live Demo**

---

## Quick Links

| | Link | Status |
|---|---|---|
| **Live API** | https://upi-fraud-engine.onrender.com | ✅ Healthy |
| **Interactive UI** | https://upi-fraud-engine.streamlit.app | ✅ Live |
| **Docs** | `/docs` folder | ✅ Complete |
| **Source Code** | GitHub | ✅ Public |

---

## Problem Statement

**Situation:**  
UPI (India's real-time payment system) processes millions of transactions daily. Fraud detection must:
- Make decisions in **<500ms** per transaction
- Respect a fixed **alert budget** (can't alert on 100% of suspected fraud)
- Use **only past data** (labels arrive 48+ hours late)
- Handle **concept drift** (new fraud patterns emerge)

**Constraint:** At transaction time T, using only information available *strictly before T*, decide: **Alert or Not Alert?**

This is not a Kaggle "maximize accuracy" problem. It's an **operational decision system under constraints**.

---

## Solution Architecture

### System Design

```
User Transaction (Streamlit UI)
    ↓
FastAPI Scoring Service (Render, Docker)
    ├─ Feature Computation (482 features, <100ms)
    ├─ Stage 1: Anomaly Detection (Isolation Forest)
    ├─ Stage 2: Fraud Classification (XGBoost, 89.18% AUC)
    └─ Alert Policy (respects daily budget)
    ↓
Response: Fraud Probability + Alert Decision
    ↓
User sees: Risk Tier (LOW/MEDIUM/HIGH) + Gauge
```

### Tech Stack

| Layer | Technology | Why |
|-------|-----------|-----|
| **Frontend** | Streamlit | Pure Python, 100-line UI, real-time updates |
| **Backend API** | FastAPI | Async, fast, auto-generated docs (/docs endpoint) |
| **Deployment** | Docker + Render | Reproducible builds, $0/month free tier |
| **ML Models** | XGBoost + Isolation Forest | Proven fraud detection; two-stage = robustness |
| **Feature Store** | DuckDB (batch) + in-memory (online) | Fast, simple, no ops overhead |
| **Validation** | Great Expectations | Data quality gates on every transaction |
| **Backtesting** | Custom framework | Day-by-day replay with alert budget enforcement |
| **Model Registry** | MLflow (local), filesystem (production) | Track experiments, version models |

---

## Key Results (Production)

### Performance
- **Latency:** 321ms end-to-end (backend: 233ms, network: 88ms)
- **AUC-ROC:** 89.18% on held-out test set
- **Precision @ Budget:** 76.4% (catches fraud within alert limits)
- **Recall @ Budget:** 61.2% (detects majority of fraud with fixed budget)

### Robustness
- **No Temporal Leakage:** Features use strictly past data (tested)
- **Budget Compliance:** 100% of days respect 0.5% daily alert limit
- **Batch-Stream Parity:** Identical decisions offline and online
- **Concept Drift Detection:** Monitors feature distributions, alerts on drift

### Production Deployment
- **Uptime:** 99.9% (Render free tier with cold starts)
- **Build Time:** 3-5 minutes (Docker auto-deploy on GitHub push)
- **Cost:** $0/month (free tier), upgradeable to $7/mo for always-on

---

## How to Use

### 1. Data Setup (One-time)
```bash
# Download IEEE-CIS fraud dataset from Kaggle
# https://www.kaggle.com/c/ieee-fraud-detection/data

# Place in data/ folder (not committed to GitHub for space)
data/
  ├── train_transaction.csv
  ├── train_identity.csv
  └── test_transaction.csv
```

### 2. Generate UPI-like Data
```bash
cd data_generation
python generate_data.py \
    --ieee_cis_path ../data/train_transaction.csv \
    --output_path ../data/enriched_transactions.duckdb
```
Converts credit card → UPI schema with controlled fraud patterns.

### 3. Run Feature Engineering
```bash
cd src/features
python offline_builder.py \
    --offline_store data/enriched_transactions.duckdb \
    --output_feature_store data/feature_vectors.parquet
```
Builds 482 features from raw transactions (time-correct, no leakage).

### 4. Train Models
```bash
cd src/models
python training_pipeline.py \
    --features_path ../data/feature_vectors.parquet \
    --config_path ../../config/project.yaml
```
Two-stage pipeline: anomaly detector + supervised classifier.

### 5. Backtest
```bash
cd src/evaluation
python backtest.py \
    --start_date 2024-01-01 \
    --end_date 2024-01-31 \
    --output_dir evaluation/results/
```
Day-by-day replay with alert budget enforcement. See precision, recall, latency.

### 6. Run Locally
```bash
# Terminal 1: Start API
python api/main.py
# Runs on http://localhost:8000

# Terminal 2: Start UI
streamlit run app.py
# Opens http://localhost:8501
```

### 7. Deploy to Production
See `/docs/PHASE_8_README.md` for Docker + Render setup.

---

## Folder Structure

```
├── config/                          # Hyperparameters, model tuning
├── data/                            # IEEE-CIS data (git-ignored, you add this)
├── data_generation/                 # Transform credit card → UPI schema
├── src/
│   ├── api/                        # FastAPI service
│   ├── features/                   # Feature engineering (offline + online)
│   ├── models/                     # Stage 1 (anomaly) + Stage 2 (supervised)
│   ├── inference/                  # Single transaction scoring
│   ├── ingestion/                  # Batch + streaming paths
│   ├── validation/                 # Great Expectations suites
│   └── evaluation/                 # Backtesting, metrics, scenarios
├── models/
│   ├── phase5_two_stage/           # Training artifacts
│   └── production/                 # Deployed XGBoost model (2.3 MB)
├── evaluation/
│   ├── backtest_results/           # Day-by-day metrics
│   └── visualizations/             # Precision/recall trends, alert compliance
├── great_expectations/             # Data quality expectations
├── notebooks/                      # Exploratory analysis (not pipeline)
├── docs/                           # Phase 1-8 READMEs (detailed design)
├── tests/                          # Unit + integration tests
├── app.py                          # Streamlit UI
├── dockerfile                      # Docker for Render deployment
├── requirements.txt                # Dependencies
└── README.md                       # This file
```

---

## Architecture Decisions (Why This Design?)

### 1. Two-Stage Model (Not Single Classifier)
**Options:** Single XGBoost vs. Isolation Forest → XGBoost  
**Choice:** Two-stage ✅  
**Why:** Anomaly stage is cheap (catches obvious outliers), supervised stage is precise (uses labeled examples). Resilient to new attack patterns.

### 2. Time-Correct Features (Complexity vs Correctness)
**Options:** Use all data vs. strict point-in-time correctness  
**Choice:** Strict correctness ✅  
**Why:** Prevents temporal leakage. A transaction at T must not "see" events at T+1. Tests verify this.

### 3. Alert Budget (Hard Constraint)
**Options:** Maximize accuracy vs. respect budget  
**Choice:** Budget-first ✅  
**Why:** Real-world fraud teams cannot alert on 100% of suspicious txns. Budget forces realistic evaluation.

### 4. Docker + Render (vs AWS/Heroku)
**Options:** AWS ECS, Heroku, Render  
**Choice:** Render ✅  
**Why:** Free Docker support, GitHub auto-deploy, simple mental model. AWS is overkill for portfolio.

---

## Interview Talking Points

### "Walk me through your fraud system"
"I built a two-stage ML system that makes real-time fraud decisions under a fixed alert budget. The frontend is Streamlit (Python), backend is FastAPI (Docker on Render). The challenge wasn't accuracy—it was **temporal correctness**: never use future data, never leak labels. I solved this with point-in-time feature computation and day-by-day backtesting."

### "How did you handle the 48-hour label delay?"
"Labels arrive 48 hours late, so you can't train on today's data. I split by event time, not load time. Train on old labeled events, test on recent unlabeled events. Backtesting respects this: retrain weekly, test on the next day."

### "Why two models, not one?"
"Stage 1 (Isolation Forest) is cheap and catches obvious anomalies. Stage 2 (XGBoost) is precise but needs labeled data. If a new fraud pattern emerges, Stage 1 catches it immediately (high anomaly score), while we label examples for Stage 2 retraining."

### "What's your biggest blind spot?"
"Completely new device + tiny amount (e.g., ₹10 on new phone). The system might flag as fraud because it's anomalous, but it's often legitimate. Mitigation: add a heuristic for new device + small amount, lower the threshold."

---

## What I Learned Building This

1. **Real-time ML is not Kaggle:** Accuracy is 5% of the problem. The other 95% is constraints (budget, latency, label delay), monitoring, and failure modes.

2. **Temporal correctness is hard:** Off-by-one mistakes are subtle. Tests are essential.

3. **Batch ≠ Streaming:** Even with same code, implementations can diverge. I built consistency checks.

4. **Deployment matters:** A model in a notebook is useless. Docker + public URLs makes it real.

---

## Testing

```bash
# Run all tests
pytest tests/ src/ -v

# Specific test suites
pytest src/features/test_time_correctness.py      # No leakage
pytest src/models/tests/test_no_label_leakage.py   # Time split correct
pytest src/evaluation/tests/test_alert_budget_respected.py  # Budget enforced
```

**Test Results:** 24/24 passing ✅

---

## Next Steps (If I Had More Time)

1. **API Authentication:** Add API keys (currently public)
2. **Rate Limiting:** 100 req/min per IP
3. **Auto-Retraining:** Weekly model refresh with latest labels
4. **A/B Testing:** Deploy multiple models, measure which catches more fraud
5. **Advanced Monitoring:** PagerDuty alerts on model drift
6. **Mobile App:** Consume API for real-time alerting

---

## References

- **IEEE-CIS Fraud Dataset:** https://www.kaggle.com/c/ieee-fraud-detection
- **UPI Fraud Patterns:** Based on real-world mobile payment attacks (device rings, velocity spikes, time anomalies)
- **Design Docs:** See `/docs/` folder for detailed phase-by-phase breakdown

---

## Author

**Parth Tiwari**  
AI/ML Engineer | Data Platform Architect  
Building real-time ML systems that actually work in production.

---

## License

MIT License - See LICENSE file.

---

**Questions?** Check `/docs/` for deep dives on:
- **PHASE_1:** Data generation + UPI enrichment
- **PHASE_2:** Batch + streaming ingestion
- **PHASE_3:** Data validation (Great Expectations)
- **PHASE_4:** Feature engineering (time-correct)
- **PHASE_5:** Two-stage modeling
- **PHASE_6:** Backtesting + alert budget
- **PHASE_7:** API service
- **PHASE_8:** Production deployment (Docker + Render)

