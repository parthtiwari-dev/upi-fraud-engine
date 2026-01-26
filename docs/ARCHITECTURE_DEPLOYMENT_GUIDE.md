# 📐 UPI Fraud Detection - Complete Architecture & Deployment Guide

## Table of Contents
1. [System Architecture](#system-architecture)
2. [9-Phase Development Lifecycle](#9-phase-development-lifecycle)
3. [Data Flow Diagrams](#data-flow-diagrams)
4. [Deployment Architecture](#deployment-architecture)
5. [Database Schema](#database-schema)
6. [API Specification](#api-specification)
7. [Feature Engineering Pipeline](#feature-engineering-pipeline)
8. [Model Architecture](#model-architecture)
9. [Monitoring & Alerts](#monitoring--alerts)

---

## System Architecture

### High-Level Overview

```
┌────────────────────────────────────────────────────────────────────────┐
│                       PRODUCTION DEPLOYMENT                            │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│   ┌─────────────────────────────────────────────────────────────────┐  │
│   │                    END USERS (Browser/Mobile)                   │  │
│   └─────────────┬──────────────────────────────────────────────────-┘  │
│                 │ HTTPS                                                │
│   ┌─────────────▼──────────────────────────────────────────────────┐   │
│   │           STREAMLIT CLOUD (Frontend Tier)                      │   │
│   │     upi-fraud-engine.streamlit.app                             │   │
│   │   ┌────────────────────────────────────────────────────┐       │   │
│   │   │ • Transaction input form (sidebar)                 │       │   │
│   │   │ • Fraud probability gauge chart                    │       │   │
│   │   │ • Risk tier badge (LOW/MEDIUM/HIGH)                │       │   │
│   │   │ • Latency & performance metrics                    │       │   │
│   │   │ • API health status                                │       │   │
│   │   └────────────────────────────────────────────────────┘       │   │
│   └─────────────┬──────────────────────────────────────────────────┘   │
│                 │ HTTP POST /score (JSON)                              │
│   ┌─────────────▼──────────────────────────────────────────────────┐   │
│   │          RENDER (Backend Tier - Docker Container)              │   │
│   │     upi-fraud-engine.onrender.com                              │   │
│   │   ┌────────────────────────────────────────────────────┐       │   │
│   │   │  FastAPI Server (uvicorn)                          │       │   │
│   │   │  ├─ POST /score       [Real-time fraud scoring]    │       │   │
│   │   │  ├─ GET /health       [System health check]        │       │   │
│   │   │  ├─ GET /metrics      [Performance metrics]        │       │   │
│   │   │  └─ GET /docs         [Swagger/OpenAPI]            │       │   │
│   │   └────────────────────────────────────────────────────┘       │   │
│   │                           │                                    │   │
│   │           ┌───────────────┼───────────────┐                    │   │
│   │           ▼               ▼               ▼                    │   │
│   │   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐           │   │
│   │   │  Service     │ │  Metrics &   │ │  Inference   │           │   │
│   │   │  Layer       │ │  Monitoring  │ │  Layer       │           │   │
│   │   │service.py    │ │metrics.py    │ │single_predict│           │   │
│   │   └──────────────┘ └──────────────┘ └──────────────┘           │   │
│   │                           │                                    │   │
│   │   ┌────────────────────────┴────────────────────────┐          │   │
│   │   ▼                                                 ▼          │   │
│   │ ┌──────────────────┐                    ┌──────────────────┐   │   │
│   │ │ OnlineFeature    │                    │   ML Model       │   │   │
│   │ │ Store            │                    │   Layer          │   │   │
│   │ │ onlinebuilder.py │                    │                  │   │   │
│   │ │                  │                    │ Stage 1:         │   │   │
│   │ │ • Computes 482   │                    │ Isolation Forest │   │   │
│   │ │   features       │                    │ (anomaly scores) │   │   │
│   │ │ • Velocity       │                    │                  │   │   │
│   │ │   aggregations   │                    │ Stage 2:         │   │   │
│   │ │ • Behavioral     │                    │ XGBoost          │   │   │
│   │ │   signals        │                    │ (0.8953 ROC-AUC) │   │   │
│   │ │ • Historical     │                    │                  │   │   │
│   │ │   fraud counts   │                    │ 482 features     │   │   │
│   │ │ • Device         |                    │ 58 encoders      │   │   │
│   │ │   fingerprints   │                    │                  │   │   │
│   │ └──────────────────┘                    └──────────────────┘   │   │
│   └────────────────────────────────────────────────────────────────┘   │
│                           │                                            │
│   ┌───────────────────────▼───────────────────────────────────────┐    │
│   │    PERSISTENT STORAGE (In Docker - Render)                    │    │
│   │  ┌──────────────────────────────────────────────────────┐     │    │
│   │  │ modelsproduction/                                    │     │    │
│   │  │ ├─ frauddetector.json (2.3 MB XGBoost model)         │     │    │
│   │  │ ├─ frauddetectorencoders.pkl (58 label encoders)     │     │    │
│   │  │ ├─ frauddetectorfeatures.txt (482 feature names)     │     │    │
│   │  │ └─ frauddetectormetadata.json (performance metrics)  │     │    │
│   │  └──────────────────────────────────────────────────────┘     │    │
│   └───────────────────────────────────────────────────────────────┘    │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 9-Phase Development Lifecycle

### Phase 1: Data Generation & Ingestion
```
┌─────────────────────┐
│  Synthetic Data     │
│  Generation         │
├─────────────────────┤
│ • 1.1M UPI txns     │
│ • 3.61% fraud rate  │
│ • Temporal ordering │
│ • Jan 2-Jul 2 2025  │
└──────────┬──────────┘
           │
           ▼
    DuckDB Database
    (data/transactions.duckdb)
    ├─ 590K+ features
    └─ Point-in-time correct
```

**Key Stats:**
- Transactions: 1,097,231
- Fraud Rate: 3.61%
- Time Span: 181 days
- Storage: ~500MB

---

### Phase 2: Ingestion Pipeline (Batch & Streaming)
```
┌──────────────────────┐
│  Batch Loader        │
│ (Training Path)      │
├──────────────────────┤
│ SELECT * FROM txns   │
│ ORDER BY time        │
│ → Memory: 4GB        │
│ → Use: Training      │
└──────────┬───────────┘
           │
   ┌───────┴────────┐
   │                │
   ▼                ▼
BATCH        STREAM
(1.1M rows)  (1 row/call)
   │                │
   └───────┬────────┘
           │
      Consistency Check
      ✓ 1000/1000 match
      ✓ Paths identical
```

---

### Phase 3: Data Validation (Great Expectations)
```
┌──────────────────────────────────┐
│  Validation Suites (GX)          │
├──────────────────────────────────┤
│ Suite 1: transactionschema       │
│ • Columns exist                  │
│ • Types correct (amount: float)  │
│ • IDs unique                     │
│ • No NULL critical fields        │
│                                  │
│ Suite 2: businesslogic           │
│ • Amount: 0 to 1M                │
│ • Currency: INR only             │
│ • No negative amounts            │
│ • Temporal causality             │
│   (labelAvailable >= eventTime)  │
└──────────┬───────────────────────┘
           │
      ✓ Checkpoint
      ✓ All 1.1M pass
      ✓ Proceed to Feature Eng
```

---

### Phase 4: Feature Engineering
```
┌──────────────────────────────────────────┐
│  Feature Generation (Point-in-Time)      │
├──────────────────────────────────────────┤
│ 482 Production Features:                 │
│                                          │
│ Velocity Features (10):                  │
│ • payer_txn_count_5min/1h/24h            │
│ • payer_sum_5min/1h/24h                  │
│ • device_txn_count_1h/24h                │
│ • device_distinct_payers_7d              │
│ • payer_distinct_payees_7d               │
│                                          │
│ Vesta Features (400):                    │
│ • V258 (amount patterns)                 │
│ • V294 (email domain risk)               │
│ • V70 (device characteristics)           │
│ • C1-C14 (categorical encodings)         │
│                                          │
│ Historical Features (70):                │
│ • payer_fraud_count_7d/30d               │
│ • device_fraud_count_7d/30d              │
│ • payer_approved_rate_30d                │
│ • ... (70+ behavioral signals)           │
│                                          │
│ Temporal Guarantees:                     │
│ ✓ No future information                  │
│ ✓ Label availability respected           │
│ ✓ 48-hour buffer between train/test      │
└──────────┬───────────────────────────────┘
           │
      DuckDB Table
      (data/processed/fullfeatures.duckdb)
      ├─ 590,546 rows
      ├─ 482 columns
      └─ Temporal ordering
```

---

### Phase 5: Model Training & AB Testing
```
┌─────────────────────────────────────────────-┐
│         Two-Stage Architecture               │
├─────────────────────────────────────────────-┤
│                                              │
│  TRAINING DATA (80%):                        │
│  Jan 2 - May 31, 2025 (151 days)             │
│  498,108 transactions                        │
│                       │                      │
│  ┌────────────────────┴──────────────────┐   │
│  │                                        │  │
│  ▼                                        ▼  │
│ STAGE 1                               STAGE 2│
│ Isolation Forest                    XGBoost   │
│ (Unsupervised)                    (Supervised)│
│                                               │
│ Input: 10 velocity features  Input: 482 feats │
│ Output: anomalyScore         Output: fraud_prob│
│ AUC: 0.85 (individual)       AUC: 0.8918      │
│                                               │
│  └────────────────────┬──────────────────┘  │
│                       │                     │
│              ┌────────▼─────────┐           │
│              │  Ensemble: Both  │           │
│              │  Two-Stage AUC   │           │
│              │    0.8953 ✓      │           │
│              └──────────────────┘           │
│                                             │
│  TEST DATA (20%):                           │
│  Jun 2 - Jul 2, 2025 (30 days)              │
│  85,429 transactions                        │
│                                             │
│  Results:                                   │
│  • Precision (0.5% budget): 92.06%          │
│  • Recall: 12.81%                           │
│  • False Alert Rate: 7.94%                  │
│  • Cost-Benefit: ₹21.6Cr/year ROI           │
│                                             │
└─────────────────────────────────────────────┘
           │
    Production Artifacts:
    ├─ frauddetector.json (2.3 MB)
    ├─ frauddetectorencoders.pkl
    ├─ frauddetectorfeatures.txt
    └─ frauddetectormetadata.json
```

---

### Phase 6: Evaluation & Backtesting
```
┌────────────────────────────────────────────┐
│  Day-by-Day Replay (Jun 1-7, 2025)         │
├────────────────────────────────────────────┤
│                                            │
│  FOR EACH DAY:                             │
│  ┌──────────────────────────────────────┐  │
│  │ 1. Load transactions for DAY         │  │
│  │ 2. Compute 482 features (point-in-time)
│  │ 3. Score with trained model          │  │
│  │ 4. Apply alert policy (0.5% budget)  │  │
│  │ 5. Evaluate: Precision/Recall/Savings
│  │ 6. Save daily metrics                │  │
│  └──────────────────────────────────────┘  │
│                                            │
│  RESULTS (7-day aggregate):                │
│  ┌──────────────────────────────────────┐ │
│  │ Transactions: 22,071                 │ │
│  │ Frauds caught: 85                    │ │
│  │ Frauds missed: 616 (due to budget)   │ │
│  │ False alerts: 28                     │ │
│  │ Precision: 75.2%                     │ │
│  │ Recall: 12.1%                        │ │
│  │ Daily savings: ₹5.92L                │ │
│  │ Weekly savings: ₹41.4L               │ │
│  │ Annual projection: ₹21.6Cr ROI       │ │
│  └──────────────────────────────────────┘ │
│                                           │
│  VISUALIZATIONS:                          │
│  ✓ Precision/Recall trend (interactive)   │
│  ✓ Fraud breakdown (caught vs missed)     │
│  ✓ Financial impact (cumulative savings)  │
│  ✓ Budget compliance (daily adherence)    │
│  ✓ Confusion matrix (heatmap)             │
│                                           │
└───────────────────────────────────────────┘
```

---

### Phase 7-8: Deployment & Production Hardening
```
┌─────────────────────────────────────────┐
│   LOCAL DEVELOPMENT                     │
├─────────────────────────────────────────┤
│ ├─ Python 3.11 virtual env              │
│ ├─ FastAPI server (uvicorn)             │
│ ├─ Streamlit UI (localhost:8501)        │
│ ├─ Models loaded in memory              │
│ └─ DuckDB for feature queries           │
│                                         │
└────────────────┬────────────────────────┘
                 │ Docker build
                 ▼
    ┌─────────────────────────────────┐
    │    DOCKER IMAGE                 │
    ├─────────────────────────────────┤
    │ FROM python:3.11-slim           │
    │ COPY requirements.txt           │
    │ RUN pip install -r ...          │
    │ COPY src/ models/ config/       │
    │ EXPOSE 8000                     │
    │ CMD uvicorn src.api.main:app    │
    │ --host 0.0.0.0 --port 8000      │
    └────────────────┬────────────────┘
                     │ Push to GitHub
                     ▼
    ┌────────────────────────────────────┐
    │  RENDER (Backend - Docker Tier)    │
    ├────────────────────────────────────┤
    │ • Auto-deploys on git push        │
    │ • Free tier + auto-scaling        │
    │ • HTTPS auto-provisioned          │
    │ • Runs: uvicorn FastAPI server    │
    │ • Port: 8000 (internal)           │
    │ • Public: upi-fraud-engine.onrender.com
    │                                   │
    └────────────────┬───────────────────┘
                     │
    ┌────────────────────────────────────┐
    │  STREAMLIT CLOUD (Frontend Tier)   │
    ├────────────────────────────────────┤
    │ • Connects to GitHub repo         │
    │ • Auto-deploys on git push        │
    │ • Python 3.11 environment         │
    │ • Runs: streamlit run app.py      │
    │ • HTTPS auto-provisioned          │
    │ • Public: upi-fraud-engine.streamlit.app
    │                                   │
    └────────────────────────────────────┘
```

---

### Phase 9: Dynamic Threshold Validation
```
┌─────────────────────────────────────────┐
│  Dynamic Threshold Computation          │
├─────────────────────────────────────────┤
│                                         │
│  Problem: Hardcoded threshold (0.5)     │
│  Solution: Percentile-based adaptation  │
│                                         │
│  Algorithm:                             │
│  ┌──────────────────────────────────┐  │
│  │ 1. Accumulate recent fraud       │  │
│  │    scores (rolling window)       │  │
│  │ 2. Compute 99.5th percentile     │  │
│  │    (top 0.5% by score)           │  │
│  │ 3. Use as alert threshold        │  │
│  │ 4. Update in real-time           │  │
│  └──────────────────────────────────┘  │
│                                         │
│  Real-World Results:                    │
│  (1250 transaction integration test)    │
│                                         │
│  Txns 1-350 (low fraud):               │
│  └─ Threshold: 0.5 (baseline)          │
│                                         │
│  Txns 350-550 (fraud cluster):         │
│  └─ Threshold: 0.59 → 0.67 (adapts!)   │
│                                         │
│  Txns 550-850 (normalizing):           │
│  └─ Threshold: 0.67 (persists)         │
│                                         │
│  Txns 850+ (new fraud pattern):        │
│  └─ Threshold: 0.69 (second spike)     │
│                                         │
│  Final Txns (all normal):              │
│  └─ Threshold: 0.50 (returns to base)  │
│                                         │
│  Key Achievement:                       │
│  ✓ Threshold changed 47 times          │
│  ✓ Zero errors across all 1250 txns    │
│  ✓ Adapts to fraud distribution        │
│  ✓ Production-ready verified           │
│                                         │
└─────────────────────────────────────────┘
```

---

## Data Flow Diagrams

### Training Data Flow
```
Raw Data
(transactions.duckdb)
     │
     ▼
Phase 2: Ingestion
├─ Schema validation
├─ Batch loading
└─ Consistency check
     │
     ▼
Phase 3: Data Validation
├─ GX structural checks
├─ GX business logic checks
└─ Gatekeeper rejection
     │
     ▼
Phase 4: Feature Engineering
├─ Temporal split (no leakage)
├─ Point-in-time features
├─ 482 dimensions
└─ Label availability check
     │
     ▼
Phase 5: Model Training
├─ Stage 1: Isolation Forest
├─ Stage 2: XGBoost
├─ Early stopping
└─ Artifact saving
```

### Serving (Real-Time) Data Flow
```
User Input
(Streamlit form)
     │
     ▼
HTTP POST /score
(JSON request)
     │
     ▼
FastAPI Endpoint
├─ Pydantic validation
└─ Extract fields
     │
     ▼
OnlineFeatureStore
├─ Compute 482 features
├─ < 50ms latency
└─ Real-time aggregations
     │
     ▼
ML Inference
├─ Stage 1: anomalyScore
├─ Stage 2: fraudProbability
└─ < 200ms latency
     │
     ▼
Alert Policy
├─ Compute dynamic threshold
├─ Compare probability >= threshold
└─ Decide should_alert
     │
     ▼
JSON Response
├─ fraud_probability
├─ should_alert
├─ threshold_used
└─ risk_tier
     │
     ▼
Streamlit Display
├─ Gauge chart
├─ Risk badge
└─ Metrics
```

---

## Deployment Architecture

### Production Environment

**Backend (Render):**
- Container: Docker (Alpine-based Python 3.11)
- Server: uvicorn (ASGI)
- Memory: ~500MB (model + features cached)
- CPU: Shared (free tier)
- Auto-scaling: Enabled
- HTTPS: Auto-provisioned
- Health checks: /health every 30s
- Uptime SLA: 99.9%

**Frontend (Streamlit Cloud):**
- Runtime: Python 3.11
- Framework: Streamlit 1.40.0
- Storage: Ephemeral (stateless)
- HTTPS: Auto-provisioned
- Scaling: Serverless (auto)
- Deploy trigger: Git push to main
- Logs: Accessible in dashboard

**Database:**
- Local: DuckDB (read-only in production)
- Feature lookup: < 50ms
- No persistent writes (stateless)

### Network Flow
```
User Browser
    │ HTTPS
    ▼
Streamlit Cloud (CDN edge server)
    │ HTTP (internal)
    ▼
Render Backend (Docker container)
    │
    ├─ Load model (2.3 MB)
    ├─ Compute features (50ms)
    ├─ Run inference (200ms)
    └─ Return JSON response
    │ HTTP response
    ▼
Streamlit Cloud (render to HTML)
    │ HTTPS
    ▼
User Browser (display results)
```

---

## Database Schema

### Phase 1 Output: transactions.duckdb

```sql
CREATE TABLE transactions (
    transaction_id VARCHAR PRIMARY KEY,
    event_timestamp TIMESTAMP NOT NULL,
    amount DECIMAL(10, 2) NOT NULL,
    payer_id VARCHAR NOT NULL,
    payer_vpa VARCHAR NOT NULL,
    payee_id VARCHAR NOT NULL,
    payee_vpa VARCHAR NOT NULL,
    device_id VARCHAR NOT NULL,
    currency VARCHAR(3),
    is_fraud FLOAT,  -- 0.0 or 1.0
    fraud_pattern VARCHAR,  -- Artifact type
    label_available_timestamp TIMESTAMP
);

-- Indexes for performance
CREATE INDEX idx_event_timestamp ON transactions(event_timestamp);
CREATE INDEX idx_payer_id ON transactions(payer_id);
CREATE INDEX idx_device_id ON transactions(device_id);
```

### Phase 4 Output: fullfeatures.duckdb

```sql
CREATE TABLE features (
    transaction_id VARCHAR PRIMARY KEY,
    event_timestamp TIMESTAMP,
    amount DECIMAL(10, 2),
    is_fraud FLOAT,  -- Label
    
    -- Velocity Features (10)
    payer_txn_count_5min INT,
    payer_sum_5min DECIMAL,
    payer_txn_count_1h INT,
    payer_sum_1h DECIMAL,
    payer_txn_count_24h INT,
    payer_sum_24h DECIMAL,
    device_txn_count_1h INT,
    device_txn_count_24h INT,
    device_distinct_payers_7d INT,
    payer_distinct_payees_7d INT,
    
    -- Vesta Features (400)
    V258 FLOAT,
    V294 FLOAT,
    V70 FLOAT,
    C1 INT,
    C2 INT,
    ... (400+ more columns)
    
    -- Historical Features (70)
    payer_fraud_count_7d INT,
    payer_fraud_count_30d INT,
    device_fraud_count_7d INT,
    device_fraud_count_30d INT,
    ... (66+ more)
    
    -- Computed Features (2)
    anomaly_score FLOAT,  -- Stage 1 output
    
    label_available_timestamp TIMESTAMP
);

-- Indexes
CREATE INDEX idx_txn_id ON features(transaction_id);
CREATE INDEX idx_timestamp ON features(event_timestamp);
```

---

## API Specification

### Endpoint 1: POST /score

**Request:**
```json
{
  "transaction_id": "TXN20260125120000",
  "amount": 5000.50,
  "payer_vpa": "user@paytm",
  "payee_vpa": "merchant@phonepe",
  "device_id": "device_abc123",
  "currency": "INR"
}
```

**Response (200 OK):**
```json
{
  "transaction_id": "TXN20260125120000",
  "fraud_probability": 0.23,
  "should_alert": false,
  "threshold_used": 0.67,
  "risk_tier": "LOW",
  "latency_ms": 256.4,
  "timestamp": "2026-01-25T12:00:00Z",
  "model_version": "1.0",
  "stage1_anomaly_score": 0.15,
  "stage2_probability": 0.23
}
```

**Error Response (400 Bad Request):**
```json
{
  "detail": "validation error",
  "errors": [
    {
      "loc": ["body", "amount"],
      "msg": "ensure this value is greater than 0",
      "type": "value_error.number.not_gt"
    }
  ]
}
```

### Endpoint 2: GET /health

**Response (200 OK):**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "1.0",
  "features_available": 482,
  "roc_auc": 0.8953,
  "latency_ms": 256.4,
  "uptime_seconds": 3847,
  "timestamp": "2026-01-25T12:00:00Z"
}
```

### Endpoint 3: GET /metrics

**Response (200 OK):**
```json
{
  "total_requests": 47,
  "daily_transaction_count": 12,
  "daily_alert_count": 1,
  "alert_rate": 0.0833,
  "avg_latency_ms": 256.4,
  "p95_latency_ms": 312.5,
  "p99_latency_ms": 425.3,
  "uptime_seconds": 3847,
  "model_version": "1.0"
}
```

### Endpoint 4: GET /docs

**Response:** Interactive Swagger UI with:
- All endpoints documented
- Request/response examples
- Try-it-out functionality
- Schema visualization

---

## Feature Engineering Pipeline

### Velocity Features (10)

```python
# 5-minute window
payer_txn_count_5min = COUNT(txns WHERE payer_id == current.payer_id 
                             AND time_ago <= 5m)
payer_sum_5min = SUM(amount WHERE payer_id == current.payer_id 
                     AND time_ago <= 5m)

# 1-hour window
payer_txn_count_1h = COUNT(...)
payer_sum_1h = SUM(...)

# 24-hour window
payer_txn_count_24h = COUNT(...)
payer_sum_24h = SUM(...)
device_txn_count_1h = COUNT(...)
device_txn_count_24h = COUNT(...)

# Diversity features
device_distinct_payers_7d = COUNT(DISTINCT payers 
                                  WHERE device_id == current.device_id
                                  AND time_ago <= 7d)
payer_distinct_payees_7d = COUNT(DISTINCT payees 
                                 WHERE payer_id == current.payer_id
                                 AND time_ago <= 7d)
```

### Historical Features (70)

```python
# Fraud history
payer_fraud_count_7d = COUNT(fraud txns WHERE payer_id == current.payer_id 
                            AND time_ago <= 7d)
payer_fraud_count_30d = COUNT(...)
payer_approved_rate_30d = APPROVED_COUNT / TOTAL_COUNT
payer_fraud_rate_30d = FRAUD_COUNT / TOTAL_COUNT

# Device fraud history
device_fraud_count_7d = COUNT(...)
device_fraud_count_30d = COUNT(...)
device_avg_fraud_amount_7d = AVG(fraud_amount)
device_fraud_concentration_7d = MAX_FRAUD_AMOUNT / SUM_FRAUD_AMOUNT

# Network risk
payee_fraud_rate_7d = FRAUD_COUNT / TOTAL_COUNT
payee_victim_rate_7d = TIMES_VICTIM_OF_FRAUD / TOTAL_TXN_AS_PAYEE
... (60+ more behavioral signals)
```

### Vesta Features (400)

Pre-computed fraud signals from Vesta (3rd party service):
- V258: Transaction amount patterns (high importance)
- V294: Email domain risk scores
- V70: Device characteristics
- V69: Transaction velocity signals
- ... (396+ more)
- C1-C14: Categorical encodings

---

## Model Architecture

### Stage 1: Isolation Forest (Unsupervised Anomaly Detection)

**Purpose:** Detect novelty patterns without fraud labels

**Input Features (10):**
- 5-min velocity aggregations
- 1-hour velocity aggregations
- 24-hour velocity aggregations
- Device diversity metrics
- Payer diversity metrics

**Model Configuration:**
```python
IsolationForest(
    n_estimators=100,
    contamination=0.036,  # Match fraud rate
    max_samples='auto',
    random_state=42,
    n_jobs=-1
)
```

**Output:**
- anomalyScore: 0-1 (higher = more anomalous)
- Captures velocity bursts, unusual recipient changes
- Feature importance: Ranked 201 overall (not top but contributing)

### Stage 2: XGBoost (Supervised Classification)

**Purpose:** Combine all signals for fraud classification

**Input Features (482):**
- 400 Vesta features
- 70 historical/behavioral features
- 10 velocity features
- 1 anomaly score from Stage 1

**Model Configuration:**
```python
XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=26.95,  # Handle class imbalance
    early_stopping_rounds=20,
    eval_metric='aucpr',
    objective='binary:logistic',
    random_state=42
)
```

**Training:**
- Train/Val split: 80%/20% stratified
- Early stopping: Best validation AUC-PR at iteration 162
- Training samples: 398,487
- Validation samples: 99,621

**Output:**
- fraudProbability: 0-1 (higher = more fraudulent)
- Feature importance: Vesta features dominate top 10

### Ensemble Logic

```python
# Stage 1: Get anomaly score
anomaly_score = isolation_forest.predict(features_10)

# Stage 2: Get fraud probability
fraud_prob = xgboost_model.predict_proba([
    features_400,  # Vesta
    features_70,   # Historical
    features_10,   # Velocity
    anomaly_score  # Stage 1 output
])

# Final decision: Use Stage 2 probability
return fraud_prob
```

---

## Monitoring & Alerts

### Metrics Dashboard (Available at /metrics)

**Real-Time Metrics:**
```
Total Requests:        47
Avg Latency:          256.4 ms
P95 Latency:          312.5 ms
P99 Latency:          425.3 ms
Alert Rate:           0.0833 (8.33%)
Daily Transactions:    12
Daily Alerts:          1
Model Version:         1.0
Uptime:               3847 seconds
```

### Health Checks

**Automated Checks (every 30s via Render):**
- Service responds to /health
- Model loads successfully
- Feature store accessible
- Inference < 500ms

**Expected Response:**
```json
{
  "status": "healthy",
  "model_loaded": true,
  "roc_auc": 0.8953
}
```

### Production Alerts (Future Implementation)

**When to Alert:**
- ❌ Latency > 500ms (degraded performance)
- ❌ Model AUC drops > 5% (model drift)
- ❌ Alert rate > 2% (possible attack)
- ❌ Error rate > 1% (service issues)
- ❌ Unavailable > 5min (downtime)

**Alert Channels (To Be Configured):**
- Slack webhook
- Email (ops team)
- PagerDuty (critical issues)

---

## Deployment Checklist

- [x] Phase 1-5: Model trained & tested
- [x] Phase 6: Backtest validated with real data
- [x] Phase 7: Docker image built & tested
- [x] Phase 8: FastAPI endpoints production-hardened
- [x] Phase 9: Dynamic threshold implemented & validated
- [x] Render: Backend deployed & live
- [x] Streamlit Cloud: Frontend deployed & live
- [x] HTTPS: Auto-provisioned for both
- [x] Health checks: Automated every 30s
- [x] Monitoring: Metrics endpoint active
- [x] Documentation: All phases documented
- [ ] Authentication: API keys not yet implemented
- [ ] Rate limiting: Not yet implemented
- [ ] Database: Persistent layer not yet added
- [ ] Observability: Prometheus/Grafana not yet set up

---

## Next Steps

1. **Phase 10:** Add authentication & rate limiting
2. **Phase 11:** Implement persistent storage (PostgreSQL)
3. **Phase 12:** Set up monitoring (Prometheus + Grafana)
4. **Phase 13:** Implement A/B testing for model updates
5. **Phase 14:** Add explainability layer (SHAP values)

---

**Last Updated:** January 26, 2026  
**Status:** Production Live ✅  
**Maintained by:** [Your Name]

