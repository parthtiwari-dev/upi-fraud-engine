```text
┌──────────────────────────────────────────────┐
│              OFFLINE PIPELINE                │
└──────────────────────────────────────────────┘

Raw IEEE-CIS Data
        │
        ▼
Enrichment (UPI Schema Normalization)
        │
        ▼
Batch Store (DuckDB / Postgres)
        │
        ▼
Validation (Great Expectations)
        │
        ▼
Offline Feature Engineering
(Point-in-Time Correct)
        │
        ▼
Model Training
(Stage 1: Candidate Models
 Stage 2: Calibrated / Filtered Models)
        │
        ▼
Backtesting & Evaluation
(Budget Constraints + Precision@K)


┌──────────────────────────────────────────────┐
│            STREAMING SIMULATION              │
└──────────────────────────────────────────────┘

Enriched Transaction Data
        │
        ▼
Event Generator
(One Transaction at a Time,
 Ordered by event_time)
        │
        ▼
Request Transport
(HTTP → Scoring API
 or Redis Queue)
        │
        ▼
Online Feature Computation
(Stateful, Low-Latency)
        │
        ▼
Model Scoring
        │
        ▼
Alert Decision Engine
(Budget-Aware Thresholding)
        │
        ▼
Logging, Metrics & Monitoring
```

