# 🚨 Real-Time UPI Fraud Detection System

> A production-grade machine learning system that simulates how real fintech companies detect fraud — built with strict temporal correctness, alert budgets, and end-to-end deployment.

**This is not a Kaggle project.**
It is a full ML system: data → features → models → business logic → API → deployment.

---

## ⚡ One-Minute Overview

**Problem:**
At transaction time **T**, using only past data (labels arrive late), decide whether to raise a fraud alert under a fixed daily alert budget — in under 500ms.

**Solution:**
A two-stage fraud detection system with point-in-time features, leakage-free training, real-time inference, and production deployment.

**Core Capabilities:**
- Realistic UPI transaction simulation (1.1M+ transactions)
- Batch + streaming pipelines with parity checks
- Point-in-time feature engineering (no future leakage)
- Two-stage ML models (Isolation Forest + XGBoost)
- Backtesting under alert budget constraints
- FastAPI backend + Streamlit UI + Docker deployment

---

## 🧠 System Architecture (Big Picture)

```
User / Client
    │
    ▼
Streamlit UI (Frontend)
    │
    ▼
FastAPI Scoring Service (Docker, Render)
    │
    ├─ Online Feature Store (Stateful)
    ├─ Stage 1: Isolation Forest (Anomaly Detection)
    ├─ Stage 2: XGBoost (Fraud Classification)
    └─ Alert Policy Engine (0.5% Budget)
    │
    ▼
Fraud Probability + Alert Decision + Business Metrics
```

---

## 🏗️ End-to-End ML Pipeline

```
PHASE 1 ─ Data Generation (UPI Simulation)
    ↓
PHASE 2 ─ Ingestion (Batch + Streaming)
    ↓
PHASE 3 ─ Data Validation (Great Expectations)
    ↓
PHASE 4 ─ Feature Engineering (Point-in-Time Safe)
    ↓
PHASE 5 ─ Model Training & Leakage Audit
    ↓
PHASE 6 ─ Backtesting & Business Evaluation
    ↓
PHASE 7 ─ Real-Time API
    ↓
PHASE 8 ─ Production Deployment
```

---

## 📊 Key Results

### Dataset
- Transactions: **1,097,231**
- Fraud rate: **3.6% (labeled data)**
- Features: **482 production-safe features**

### Model Performance (Leakage-Free)
- ROC-AUC: **0.8918**
- Precision @ 0.5% alert budget: **~90%+**
- Recall @ 0.5% alert budget: **~12%**

### Production Metrics
- Latency: **~233ms avg (<500ms target)**
- Deployment: Render + Streamlit Cloud
- Architecture: Docker + FastAPI + Stateful Features

---

# 🧩 Phase-by-Phase System Design

---

## PHASE 1 — Realistic UPI Data Generation

### Objective
Simulate a real UPI ecosystem with realistic fraud patterns.

### Fraud Patterns
- Device rings
- Velocity bursts
- Time anomalies
- Label delays (fraud discovered hours/days later)

### Pipeline
```
IEEE-CIS Data → UPI Schema Mapping → Fraud Injection → Validation → DuckDB
```

Outcome: A synthetic but realistic fintech dataset suitable for system-level ML design.

---

## PHASE 2 — Ingestion Pipeline (Batch + Streaming)

### Problem Solved
**Training–serving skew** — the most common production ML failure.

### Architecture
```
DuckDB
  ├─ Batch Loader (Training)
  └─ Streaming Simulator (Serving)
         ↓
   Consistency Check (Identical Outputs)
```

Guarantee: Offline and online pipelines see identical data formats and semantics.

---

## PHASE 3 — Data Validation & Temporal Guarantees

### Tooling
- Great Expectations

### Enforced Constraints
- Schema correctness
- Type safety
- Unique IDs
- Temporal causality (no future data)
- Label delay constraints

Key Insight:
> If your data is temporally wrong, your model is meaningless.

---

## PHASE 4 — Feature Engineering (Point-in-Time Safe)

### Feature Families
1. Velocity Features (5min, 1h, 24h)
2. Graph Features (device ↔ users)
3. Risk History (label-aware)

### Core Rule
For every transaction T:
> Features must use only data from time < T.

### Output
- 487 total features
- Strictly leakage-free feature store

---

## PHASE 5 — Model Training & Leakage Audit

### Two-Stage Architecture

```
Stage 1: Isolation Forest (unsupervised anomalies)
Stage 2: XGBoost (supervised fraud classification)
```

### Critical Discovery
A synthetic column (`fraud_pattern`) caused label leakage.

```
AUC with leakage: 0.9106
AUC without leakage: 0.8918
```

Decision:
Deploy the leakage-free model despite lower metrics — prioritizing correctness over vanity scores.

---

## PHASE 6 — Backtesting & Business Evaluation

### Why Standard ML Metrics Fail
Real systems operate under operational constraints.

### Implemented
- Day-by-day replay
- Alert budget enforcement (0.5%)
- Cost–benefit analysis
- ROI estimation

Example Insight:
```
High accuracy ≠ useful system
Useful system = accuracy under budget constraints
```

---

## PHASE 7 — Real-Time Fraud Detection API

### Architecture

```
FastAPI API
   ↓
Online Feature Store (stateful)
   ↓
Two-Stage Model
   ↓
Alert Policy Engine
```

Capabilities:
- Real-time scoring (<500ms)
- Stateful feature updates
- Business-layer decision logic
- Production-ready inference pipeline

---

## PHASE 8 — Production Deployment

### Stack
- Backend: FastAPI + Docker + Render
- Frontend: Streamlit Cloud

### Live System Flow

```
User → Streamlit UI → FastAPI API → Model → Decision
```

---

# 🧱 Repository Structure

```
upi-fraud-engine/
├── src/
│   ├── api/            # FastAPI backend
│   ├── ingestion/      # Batch & streaming loaders
│   ├── validation/     # Great Expectations
│   ├── features/       # Feature engineering
│   ├── evaluation/     # Backtesting & metrics
│   └── inference/      # Model inference
├── models/             # Trained models & encoders
├── data_generation/    # Synthetic UPI data pipeline
├── evaluation/         # Reports & visualizations
├── docs/               # Phase-wise documentation
├── notebooks/          # Experiments
├── dockerfile          # Deployment
├── app.py              # Streamlit UI
└── README.md           # (this file)
```

---

# 🎯 Core Design Principles

### 1) Temporal Correctness
No future information is used in training or inference.

### 2) Training–Serving Parity
Batch and streaming pipelines are provably identical.

### 3) Business-First Evaluation
Metrics reflect operational constraints, not just accuracy.

### 4) Production Realism
System design mirrors real fintech fraud engines.

---

# 💡 Why This Project Is Different

Most ML projects answer:
> “Can I train a model?”

This project answers:
> “Can I build a system that would actually work in production?”

It demonstrates:
- ML engineering + data engineering + backend integration
- real-world fraud constraints (latency, budget, label delay)
- system-level thinking beyond algorithms

---

# 🚀 Future Extensions

- Kafka-based streaming pipeline
- Redis-backed online feature store
- Automated model retraining
- Drift detection & monitoring
- Online A/B testing
- Real UPI-scale simulation

---

# 👤 Author

**Parth Tiwari**  
Aspiring ML / AI Engineer focused on building production-grade ML systems.

---

# 🧠 Final Thought

Accuracy is easy.
Correctness is hard.
Production realism is harder.

This project was built to solve the hardest one.

