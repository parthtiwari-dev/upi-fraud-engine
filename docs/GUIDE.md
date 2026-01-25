# 📚 Documentation Guide

**Start here if you're new** → Pick your path below

---

## 🎯 Choose Your Path

### 👀 I Want a Quick Overview (10 min)
1. **[README.md](README.md)** - Project overview, live links, quick start
2. Try the API: https://upi-fraud-engine.streamlit.app/
3. Done ✓

### 💼 I'm Here for a Job Interview (1 hour)
1. **[README.md](README.md)** - Problem & solution (10 min)
2. **[Phase 5 README](docs/PHASE_5_README.md)** - Model training & two-stage architecture (25 min)
3. **[Phase 6 README](docs/PHASE_6_README.md)** - Backtesting & business metrics (15 min)
4. Practice the talking points (10 min)

**Key quote for interviews:**
> *"I discovered label leakage in the baseline model, which inflated ROC-AUC from 0.8953 to 0.9106. After fixing it, I confirmed the two-stage model was still the winner (+0.35%). I deployed the single-stage XGBoost in production for operational simplicity while keeping the same 0.8953 performance."*

### 🏗️ I Want to Understand the Architecture (2 hours)
1. **[README.md](README.md)** - Overview
2. **[Phase 4 README](docs/phase4_final_readme.md)** - Feature engineering (30 min)
3. **[Phase 5 README](docs/PHASE_5_README.md)** - Model training (30 min)
4. **[Phase 6 README](docs/PHASE_6_README.md)** - Backtesting (30 min)
5. Review code in `src/` folder

### 🚀 I Want to Deploy This (2 hours)
1. **[Phase 7 README](docs/phase7_readme.md)** - Deployment architecture (20 min)
2. **[Phase 8 README](docs/PHASE_8_README.md)** - Production hardening (20 min)
3. Run locally: `uvicorn src.api.main:app --reload`
4. Deploy to Render + Streamlit Cloud

### 📖 I Want to Learn Everything (4+ hours)
Read all 9 phase READMEs in order:
1. Phase 1: Data generation
2. Phase 2: Ingestion pipeline
3. Phase 3: Data validation
4. Phase 4: Feature engineering ⭐
5. Phase 5: Model training ⭐
6. Phase 6: Backtesting ⭐
7. Phase 7: Deployment
8. Phase 8: Production hardening
9. Phase 9: Dynamic threshold

---

## 📋 What Each Phase Covers

| Phase | Focus | Key Achievement | Read Time |
|-------|-------|-----------------|-----------|
| **1** | Data Generation | 1.1M synthetic transactions | 10 min |
| **2** | Ingestion Pipeline | Batch + streaming paths | 10 min |
| **3** | Data Validation | Great Expectations setup | 10 min |
| **4** | Feature Engineering | 482 production-safe features | **30 min** |
| **5** | Model Training | Two-stage A/B testing | **30 min** |
| **6** | Backtesting | Day-by-day replay, metrics | **30 min** |
| **7** | Deployment | Docker, FastAPI, APIs | 15 min |
| **8** | Production Hardening | Health checks, monitoring | 15 min |
| **9** | Dynamic Threshold | Adaptive percentile algorithm | 10 min |

**⭐ Critical for understanding:** Phases 4, 5, 6

---

## 🔍 Find Specific Topics

### I want to know about...

**The two-stage model**
→ [Phase 5 README - Architecture section](docs/PHASE_5_README.md#-architecture)

**Label leakage discovery**
→ [Phase 5 README - Key Finding section](docs/PHASE_5_README.md#-key-finding-leakage-audit)

**Business metrics & ROI**
→ [Phase 6 README - Business Impact section](docs/PHASE_6_README.md)

**Alert budget enforcement**
→ [Phase 6 README - Budget Policy section](docs/PHASE_6_README.md#-alert-budget-policy) or [Code](src/evaluation/alert_policy.py)

**Feature engineering**
→ [Phase 4 README - Features section](docs/phase4_final_readme.md)

**Production deployment**
→ [Phase 7 README](docs/phase7_readme.md)

**API endpoint details**
→ https://upi-fraud-engine.onrender.com/docs (live Swagger docs)

**Performance metrics**
→ [README.md - Performance section](README.md#-performance)

---

## 🎯 Quick Reference

### Problem
Fraud detection in UPI with:
- ✅ High precision (budget-limited alerts)
- ✅ Temporal correctness (no future information leak)
- ✅ Production guarantees (no label leakage)
- ✅ Adaptive thresholds (handle distribution shifts)

### Solution
- **Two-stage architecture:** Isolation Forest + XGBoost = 0.8953 ROC-AUC
- **Production model:** XGBoost alone (same performance, simpler)
- **Validation:** 55+ leakage tests, day-by-day backtesting
- **Business metric:** 92% precision @ 0.5% alert budget

### Key Files
- `src/api/main.py` - API endpoints
- `src/models/stage1_anomaly.py` - Isolation Forest
- `src/models/stage2_supervised.py` - XGBoost
- `src/evaluation/backtest.py` - Day-by-day validation
- `app.py` - Streamlit UI

### Live Systems
- API: https://upi-fraud-engine.onrender.com/docs
- UI: https://upi-fraud-engine.streamlit.app/
- Health: https://upi-fraud-engine.onrender.com/health

---

## 🚀 Getting Started (5 min)

```bash
# Clone
git clone https://github.com/yourusername/upi-fraud-engine.git
cd upi-fraud-engine

# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run API
uvicorn src.api.main:app --reload
# Open: http://localhost:8000/docs

# Run UI (new terminal)
streamlit run app.py
# Opens: http://localhost:8501
```

---

## 📊 Reading Time by Purpose

| Goal | Files | Time |
|------|-------|------|
| Quick overview | README.md | 10 min |
| Interview prep | README.md + Phase 5-6 | 1 hour |
| Architecture deep-dive | Phase 4-6 | 1.5 hours |
| Complete understanding | All 9 phases | 4 hours |
| Deployment | Phase 7-8 + README | 1 hour |

---

## ❓ FAQ

**Q: What's the difference between the two-stage model and production model?**
A: Two-stage (Isolation Forest + XGBoost) achieves 0.8953 ROC-AUC. Production uses XGBoost alone (same 0.8953 ROC-AUC) for operational simplicity. See Phase 5 README.

**Q: Why does the README mention 0.9106 ROC-AUC if it's 0.8953?**
A: 0.9106 was inflated by label leakage (synthetic-only `fraud_pattern` column). True performance is 0.8953. See Phase 5 - Key Finding.

**Q: Where's the model training code?**
A: `src/models/` folder:
- `stage1_anomaly.py` - Isolation Forest
- `stage2_supervised.py` - XGBoost
- `training_pipeline.py` - A/B testing framework

**Q: How do I score transactions in production?**
A: Two options:
1. **API:** POST to https://upi-fraud-engine.onrender.com/score
2. **Code:** `from src.inference.single_predict import score_transaction`

**Q: What does "0.5% alert budget" mean?**
A: You can alert on at most 0.5% of transactions daily (top 0.5% by fraud probability). See Phase 6 README - Alert Budget Policy.

**Q: Is this real or synthetic data?**
A: Synthetic data (1.1M generated UPI transactions). Real implementation would require actual UPI transaction data.

**Q: What are the 482 features?**
A: See README.md - 482 Features Breakdown:
- 400 Vesta features
- 70 historical (fraud counts, approval rates)
- 10 velocity (transaction counts/amounts)
- 1 anomaly score (from Stage 1)
- 1 temporal

---

## 🎓 Learning Path

**Week 1: Understand the Problem**
- Read: README.md
- Try: Live API at https://upi-fraud-engine.streamlit.app/
- Goal: Understand problem & solution

**Week 2: Deep Dive Model Training**
- Read: Phase 4 (features) + Phase 5 (model)
- Review: `src/models/` code
- Goal: Understand two-stage architecture & A/B testing

**Week 3: Learn Validation & Business**
- Read: Phase 6 (backtesting)
- Review: `src/evaluation/` code
- Goal: Understand budget constraints & metrics

**Week 4: Production Systems**
- Read: Phase 7-8 (deployment)
- Review: `src/api/` code
- Goal: Understand production setup & monitoring

---

## 🤝 Contributing

Want to improve something? Found a bug? Have ideas?
- Open a GitHub issue
- Submit a pull request
- Add more advanced features (see Next Steps in README.md)

---

**Status:** ✅ Production Live  
**Last Updated:** January 26, 2026  
**Version:** 1.0

**Start with:** [README.md](README.md) for quick overview

