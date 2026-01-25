"""
Test UPI Fraud Detection API with Real Transactions
Loads 100 normal + 30 fraud transactions from DuckDB
Sends to API and analyzes results
"""
import duckdb
import pandas as pd
import requests
import json
import time
from datetime import datetime
from typing import List, Dict
import numpy as np

# Configuration
API_BASE_URL = "http://localhost:8000"
DUCKDB_PATH = "data/processed/transactions.duckdb"
OUTPUT_FILE = "test_results.csv"

print("="*70)
print("  🧪 TESTING API WITH REAL TRANSACTIONS")
print("="*70)
print()

# ============================================================================
# STEP 1: Load Real Transactions from DuckDB
# ============================================================================
print("📊 Step 1: Loading transactions from DuckDB...")
print(f"   Database: {DUCKDB_PATH}")

try:
    con = duckdb.connect(DUCKDB_PATH)
    
    # Load first 100 transactions (sorted by time)
    print("   Loading first 1200 transactions (sorted by event_timestamp)...")
    query_normal = """
    SELECT *
    FROM transactions
    WHERE is_fraud = 0.0
    ORDER BY event_timestamp
    LIMIT 1200
    """
    df_normal = con.execute(query_normal).df()
    print(f"   ✅ Loaded {len(df_normal)} normal transactions")
    
    # Load 30 fraudulent transactions
    print("   Loading 30 fraudulent transactions...")
    query_fraud = """
    SELECT *
    FROM transactions
    WHERE is_fraud = 1.0
    ORDER BY event_timestamp
    LIMIT 50
    """
    df_fraud = con.execute(query_fraud).df()
    print(f"   ✅ Loaded {len(df_fraud)} fraudulent transactions")
    
    con.close()
    
    # Combine datasets
    df_all = pd.concat([df_normal, df_fraud], ignore_index=True)
    
    # Sort by time (to maintain temporal order)
    df_all = df_all.sort_values('event_timestamp').reset_index(drop=True)
    
    print(f"\n   📦 Total transactions to test: {len(df_all)}")
    print(f"   📅 Time range: {df_all['event_timestamp'].min()} to {df_all['event_timestamp'].max()}")
    print(f"   💰 Amount range: ₹{df_all['amount'].min():.2f} to ₹{df_all['amount'].max():.2f}")
    print()

except Exception as e:
    print(f"   ❌ Error loading data: {e}")
    exit(1)


# ============================================================================
# STEP 2: Prepare Transactions for API
# ============================================================================
print("🔧 Step 2: Preparing transactions for API...")

def prepare_transaction_for_api(row: pd.Series) -> Dict:
    """
    Convert DataFrame row to API request format.
    Includes all Vesta features.
    """
    # Start with required fields
    txn = {
        "transaction_id": str(row["transaction_id"]),
        "event_timestamp": row["event_timestamp"].isoformat() if pd.notna(row["event_timestamp"]) else datetime.now().isoformat(),
        "amount": float(row["amount"]),
        "payer_vpa": str(row["payer_vpa"]),
        "payee_vpa": str(row["payee_vpa"]),
        "device_id": str(row["device_id"]),
        "currency": "INR"
    }
    
    # Add all Vesta features (V1-V339, C1-C14, D1-D15, etc.)
    for col in row.index:
        if col not in txn and col not in ["is_fraud", "fraud_pattern", "label_available_timestamp"]:
            value = row[col]
            # Handle NaN values
            if pd.isna(value):
                txn[col] = None
            elif isinstance(value, (np.integer, np.floating)):
                txn[col] = float(value) if not np.isnan(value) else None
            else:
                txn[col] = value
    
    return txn

print(f"   ✅ Prepared {len(df_all)} transactions")
print()


# ============================================================================
# STEP 3: Send Transactions to API
# ============================================================================
print("🚀 Step 3: Sending transactions to API...")
print(f"   Endpoint: {API_BASE_URL}/score")
print()

results = []
start_time = time.time()

for idx, row in df_all.iterrows():
    txn = prepare_transaction_for_api(row)
    true_label = row["is_fraud"]
    
    try:
        # Send POST request
        response = requests.post(
            f"{API_BASE_URL}/score",
            json=txn,
            headers={"Content-Type": "application/json"}
        )
        
        if response.status_code == 200:
            result = response.json()
            
            # Store results
            results.append({
                "transaction_id": txn["transaction_id"],
                "true_label": true_label,
                "fraud_probability": result["fraud_probability"],
                "should_alert": result["should_alert"],
                "risk_tier": result["risk_tier"],
                "latency_ms": result["latency_ms"],
                "amount": txn["amount"],
                "event_timestamp": txn["event_timestamp"],
                "threshold_used": result["threshold_used"]
            })
            
            # Print progress every 10 transactions
            if (idx + 1) % 10 == 0:
                print(f"   ✅ Processed {idx + 1}/{len(df_all)} transactions")
        else:
            print(f"   ❌ Error for {txn['transaction_id']}: {response.status_code}")
            results.append({
                "transaction_id": txn["transaction_id"],
                "true_label": true_label,
                "fraud_probability": None,
                "should_alert": None,
                "risk_tier": "error",
                "latency_ms": None,
                "amount": txn["amount"],
                "event_timestamp": txn["event_timestamp"]
            })
    
    except Exception as e:
        print(f"   ❌ Exception for {txn['transaction_id']}: {e}")
        results.append({
            "transaction_id": txn["transaction_id"],
            "true_label": true_label,
            "fraud_probability": None,
            "should_alert": None,
            "risk_tier": "error",
            "latency_ms": None,
            "amount": txn["amount"],
            "event_timestamp": txn["event_timestamp"]
        })

total_time = time.time() - start_time
print()
print(f"   ✅ Completed {len(results)} transactions in {total_time:.2f}s")
print(f"   📊 Avg latency: {total_time / len(results) * 1000:.2f}ms per transaction")
print()


# ============================================================================
# STEP 4: Analyze Results
# ============================================================================
print("="*70)
print("  📊 ANALYSIS RESULTS")
print("="*70)
print()

df_results = pd.DataFrame(results)

print("\nThresholds_used:")
print(df_results["threshold_used"].to_list())



# Save results
df_results.to_csv(OUTPUT_FILE, index=False)
print(f"💾 Results saved to: {OUTPUT_FILE}")
print()

# Filter out errors
df_valid = df_results[df_results["fraud_probability"].notna()].copy()

if len(df_valid) == 0:
    print("❌ No valid results to analyze!")
    exit(1)

# Performance Metrics
print("⚡ PERFORMANCE METRICS")
print("─" * 50)
latencies = df_valid["latency_ms"].dropna()
print(f"Total Requests:    {len(df_valid)}")
print(f"Avg Latency:       {latencies.mean():.2f}ms")
print(f"P50 Latency:       {latencies.quantile(0.50):.2f}ms")
print(f"P95 Latency:       {latencies.quantile(0.95):.2f}ms")
print(f"P99 Latency:       {latencies.quantile(0.99):.2f}ms")
print(f"Max Latency:       {latencies.max():.2f}ms")
print(f"Throughput:        {len(df_valid) / total_time:.2f} req/s")
print()

# Fraud Detection Accuracy
print("🎯 FRAUD DETECTION ACCURACY")
print("─" * 50)

# Separate normal and fraud transactions
df_normal_results = df_valid[df_valid["true_label"] == 0.0]
df_fraud_results = df_valid[df_valid["true_label"] == 1.0]

print(f"Normal Transactions: {len(df_normal_results)}")
print(f"  Avg Fraud Prob:    {df_normal_results['fraud_probability'].mean():.4f}")
print(f"  Max Fraud Prob:    {df_normal_results['fraud_probability'].max():.4f}")
print(f"  Alerted:           {df_normal_results['should_alert'].sum()} ({df_normal_results['should_alert'].sum()/len(df_normal_results)*100:.2f}%)")
print()

print(f"Fraudulent Transactions: {len(df_fraud_results)}")
print(f"  Avg Fraud Prob:        {df_fraud_results['fraud_probability'].mean():.4f}")
print(f"  Min Fraud Prob:        {df_fraud_results['fraud_probability'].min():.4f}")
print(f"  Alerted:               {df_fraud_results['should_alert'].sum()} ({df_fraud_results['should_alert'].sum()/len(df_fraud_results)*100:.2f}%)")
print()

# Calculate metrics at threshold 0.5 (for analysis)
df_valid["predicted_fraud"] = df_valid["should_alert"]


tp = len(df_valid[(df_valid["true_label"] == 1.0) & (df_valid["predicted_fraud"] == True)])
tn = len(df_valid[(df_valid["true_label"] == 0.0) & (df_valid["predicted_fraud"] == False)])
fp = len(df_valid[(df_valid["true_label"] == 0.0) & (df_valid["predicted_fraud"] == True)])
fn = len(df_valid[(df_valid["true_label"] == 1.0) & (df_valid["predicted_fraud"] == False)])

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

# Calculate metrics at threshold 0.5 (for analysis)
df_valid["predicted_fraud"] = df_valid["should_alert"]

tp = len(df_valid[(df_valid["true_label"] == 1.0) & (df_valid["predicted_fraud"] == True)])
tn = len(df_valid[(df_valid["true_label"] == 0.0) & (df_valid["predicted_fraud"] == False)])
fp = len(df_valid[(df_valid["true_label"] == 0.0) & (df_valid["predicted_fraud"] == True)])
fn = len(df_valid[(df_valid["true_label"] == 1.0) & (df_valid["predicted_fraud"] == False)])

precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

# Get threshold stats
avg_threshold = df_valid["threshold_used"].mean()
min_threshold = df_valid["threshold_used"].min()
max_threshold = df_valid["threshold_used"].max()

print(f"🎯 CLASSIFICATION METRICS")
print("─" * 50)
print(f"Dynamic Threshold Range: {min_threshold:.4f} - {max_threshold:.4f}")
print(f"Average Threshold Used:  {avg_threshold:.4f}")
print()
print(f"Confusion Matrix:")
print(f"  True Positives:  {tp}")
print(f"  True Negatives:  {tn}")
print(f"  False Positives: {fp}")
print(f"  False Negatives: {fn}")
print()
print(f"Performance:")
print(f"  Precision:       {precision:.4f}")
print(f"  Recall:          {recall:.4f}")
print(f"  F1 Score:        {f1:.4f}")
print()

# Distribution of fraud probabilities
print("📈 FRAUD PROBABILITY DISTRIBUTION")
print("─" * 50)
print("Normal Transactions:")
print(df_normal_results["fraud_probability"].describe())
print()
print("Fraudulent Transactions:")
print(df_fraud_results["fraud_probability"].describe())
print()

# Top 10 highest fraud scores
print("🔴 TOP 10 HIGHEST FRAUD SCORES")
print("─" * 50)
top_10 = df_valid.nlargest(10, "fraud_probability")[["transaction_id", "true_label", "fraud_probability", "should_alert", "amount"]]
print(top_10.to_string(index=False))
print()

# Alert Budget Usage
print("🚨 ALERT POLICY")
print("─" * 50)
total_alerts = df_valid["should_alert"].sum()
alert_rate = total_alerts / len(df_valid)
print(f"Total Alerts:      {total_alerts}")
print(f"Alert Rate:        {alert_rate*100:.2f}%")
print(f"Target:            0.5%")
print(f"Status:            {'✅ Within budget' if alert_rate <= 0.005 else '⚠️ Over budget'}")
print()

# Final API metrics
print("🌐 FINAL API METRICS")
print("─" * 50)
try:
    metrics_response = requests.get(f"{API_BASE_URL}/metrics")
    if metrics_response.status_code == 200:
        metrics = metrics_response.json()
        print(f"Total API Requests:     {metrics['total_requests']}")
        print(f"Total Alerts Issued:    {metrics['total_alerts']}")
        print(f"API Alert Rate:         {metrics['alert_rate']*100:.2f}%")
        print(f"API Avg Latency:        {metrics['avg_latency_ms']:.2f}ms")
        print(f"API P95 Latency:        {metrics['p95_latency_ms']:.2f}ms")
        print(f"Budget Utilization:     {metrics['daily_budget_utilization']*100:.2f}%")
except:
    print("⚠️ Could not fetch API metrics")

print()
print("="*70)
print("  ✅ TEST COMPLETE")
print("="*70)
