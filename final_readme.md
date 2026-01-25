# 🚨 Real-Time UPI Fraud Detection System

> A production-grade, end-to-end machine learning system that simulates real-world fintech fraud detection — from raw data generation to live deployment with strict temporal correctness, alert budgets, and business metrics.

---

## 🌍 What This Project Actually Is (Not Just Another ML Model)

Most ML projects stop at:
> dataset → model → accuracy → done

This project simulates a **real fintech fraud detection system**, including:

- realistic UPI transaction generation
- ingestion pipelines (batch + streaming)
- data validation & leakage prevention
- point-in-time feature engineering
- two-stage fraud modeling
- A/B testing & backtesting
- alert budget & business metrics
- real-time API + UI deployment

This is closer to how Stripe / Paytm / PhonePe systems work than Kaggle notebooks.

---

## 🧠 System Architecture (High-Level)

```
USER / CLIENT
    │
    ▼
Streamlit UI (Frontend)
    │
    ▼
FastAPI Backend (Dockerized)
    │
    ▼
Online Feature Store (Stateful)
    │
    ▼
Two-Stage Fraud Model (Isolation Forest + XGBoost)
    │
    ▼
Alert Policy Engine (0.5% Budget)
    │
    ▼
Fraud Decision + Business Metrics
```

---

## 🏗️ End-to-End Pipeline Architecture

```
PHASE 1 ─ Data Generation
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
PHASE 7 ─ Real-Time Fraud API
    ↓
PHASE 8 ─ Production Deployment
```

---

# 📊 Key Results

### Dataset
- Total transactions: **1,097,231**
- Fraud rate: **3.6% (labeled data)**
- Features: **482 production-safe features**

### Model Performance
- ROC-AUC: **0.8918** (leakage-free)
- Precision @ 0.5% alert budget: **~92%**
- Recall @ 0.5% alert budget: **~12%**

### Production Metrics
- Latency: **~233ms avg (<500ms target)**
- Deployment: Render + Streamlit Cloud
- Architecture: Docker + FastAPI + Stateful Features

---

# 🧩 Phase-by-Phase Breakdown

---

## PHASE 1 — Realistic UPI Data Generation

### Goal
Simulate a real UPI transaction ecosystem with fraud patterns.

### Key Features
- Device rings
- Velocity spikes
- Time anomalies
- Label delays (realistic fraud discovery)

### Output
- DuckDB database with 1.1M+ transactions

```
RAW DATA → UPI SCHEMA → FRAUD INJECTION → VALIDATION → DUCKDB
```

---

## PHASE 2 — Ingestion Pipeline (Batch + Streaming)

### Problem Solved
Training-serving skew (the #1 ML production bug).

### Architecture
```
DuckDB
  ├─ Batch Loader (Training)
  └─ Streaming Simulator (Serving)
         ↓
   Consistency Check (100% match)
```

### Result
- Guaranteed identical data formats across offline & online paths.

---

## PHASE 3 — Data Validation & Leakage Prevention

### Tools
- Great Expectations

### Rules Enforced
- schema correctness
- type safety
- unique IDs
- temporal causality
- label delay constraints

### Key Insight
> No model can be trusted if the data is not causally correct.

---

## PHASE 4 — Feature Engineering (Point-in-Time Safe)

### Feature Families
1. Velocity features (5min, 1h, 24h)
2. Graph features (device ↔ users)
3. Risk history (label-aware)

### Core Principle
For every transaction T:
> features must only use data from time < T

### Output
- 487 total features
- zero future leakage

---

## PHASE 5 — Model Training & Leakage Audit

### Two-Stage Architecture

```
Stage 1: Isolation Forest (Anomaly Detection)
Stage 2: XGBoost (Fraud Classification)
```

### Critical Discovery
Synthetic column `fraud_pattern` caused label leakage.

```
AUC with leakage: 0.9106
AUC without leakage: 0.8918
```

### Decision
Deploy leakage-free model despite lower metrics.

---

## PHASE 6 — Backtesting & Business Evaluation

### Why Normal ML Metrics Are Not Enough
Real systems operate under constraints.

### Implemented
- day-by-day replay
- alert budget enforcement (0.5%)
- cost-benefit analysis
- ROI estimation

### Example Result
```
Daily savings ≈ ₹6,00,000
Annual ROI ≈ 7400%
```

---

## PHASE 7 — Real-Time Fraud Detection API

### Architecture

```
FastAPI API
   ↓
Online Feature Store (stateful)
   ↓
XGBoost Model
   ↓
Alert Policy Engine
```

### Key Capabilities
- real-time scoring (<500ms)
- stateful feature updates
- alert budget logic
- business-layer decision making

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
Batch and streaming pipelines are identical.

### 3) Business-First Evaluation
Metrics reflect operational constraints, not just accuracy.

### 4) Production Realism
System designed like a fintech fraud engine, not a Kaggle project.

---

# 🚀 Why This Project Matters

This project demonstrates:

- ML system design, not just modeling
- data engineering + ML + backend integration
- real-world fraud detection constraints
- production deployment skills

It bridges the gap between:

> "I trained a model" → "I built a real ML system"

---

# 🧭 Future Work

- Kafka-based real streaming
- Redis-backed feature store
- model retraining pipeline
- drift detection & monitoring
- online A/B testing
- real UPI-like datasets

---

# 👤 Author

**Parth Tiwari**

Aspiring ML / AI Engineer focused on building production-grade ML systems.

---

# 🧠 If You Read This Far

This repository is not about maximizing accuracy.

It is about answering a harder question:

> "What does it actually take to build a real fraud detection system?"

