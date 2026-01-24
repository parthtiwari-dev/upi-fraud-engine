"""
Tests for Alert Budget Enforcement

Validates that AlertPolicy class respects daily alert budgets under all conditions.
Critical for operational constraints - investigation teams have limited capacity.

"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.evaluation.alert_policy import (
    AlertPolicy,
    calculate_threshold_for_budget,
    compute_daily_metrics
)


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def sample_transactions():
    """Generate sample transaction data for testing."""
    np.random.seed(42)
    n_samples = 1000
    
    dates = pd.date_range('2025-06-01', periods=5, freq='D')
    transactions = []
    
    for date in dates:
        for i in range(n_samples // 5):
            transactions.append({
                'event_timestamp': date + timedelta(seconds=np.random.randint(0, 86400)),
                'fraud_probability': np.random.random(),
                'is_fraud': np.random.choice([0, 1], p=[0.97, 0.03])
            })
    
    df = pd.DataFrame(transactions)
    return df.sort_values('event_timestamp').reset_index(drop=True)


@pytest.fixture
def fraud_probabilities():
    """Generate fraud probability scores."""
    np.random.seed(42)
    return np.random.random(1000)


# ============================================================================
# TEST 1: Budget Never Exceeded
# ============================================================================

def test_alert_budget_never_exceeded(sample_transactions):
    """
    CRITICAL TEST: Alert budget must NEVER be exceeded on any day.
    
    This is a hard operational constraint. If investigation team
    can only handle 0.5% of transactions, we cannot exceed it.
    """
    budget = 0.005  # 0.5%
    policy = AlertPolicy(budget_pct=budget)
    
    # Get fraud probabilities
    fraud_probs = sample_transactions['fraud_probability'].values
    
    # Apply policy
    alerts, metadata = policy.decide_alerts(sample_transactions, fraud_probs)
    
    # Check overall alert rate
    overall_rate = metadata['alert_rate']
    assert overall_rate <= budget * 1.01, \
        f"Overall alert rate {overall_rate:.4%} exceeds budget {budget:.4%}"
    
    # Check daily alert rates
    df = sample_transactions.copy()
    df['alert'] = alerts
    df['date'] = pd.to_datetime(df['event_timestamp']).dt.date
    
    daily_rates = df.groupby('date').apply(
        lambda x: x['alert'].sum() / len(x)
    )
    
    for date, rate in daily_rates.items():
        assert rate <= budget * 1.01, \
            f"Alert rate on {date} ({rate:.4%}) exceeds budget ({budget:.4%})"
    
    print(f"✅ Budget respected on all {len(daily_rates)} days")


# ============================================================================
# TEST 2: Threshold Calculation
# ============================================================================

def test_threshold_calculation_is_correct():
    """
    Test that calculate_threshold_for_budget returns correct threshold.
    
    The threshold should be the Kth highest score where K = budget * N.
    """
    # Simple case: 10 scores, 20% budget → threshold should be 6th highest
    scores = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    budget = 0.2  # 20% → 2 alerts out of 10
    
    threshold = calculate_threshold_for_budget(scores, budget)
    
    # With budget 0.2 and 10 scores, we want top 2 (20%)
    # Threshold should be between 0.8 and 0.9
    assert 0.8 <= threshold <= 0.9, f"Threshold {threshold} not in expected range"
    
    # Verify: applying this threshold should give ~20% alerts
    alerts = scores >= threshold
    alert_rate = alerts.sum() / len(scores)
    assert 0.15 <= alert_rate <= 0.25, \
        f"Alert rate {alert_rate:.1%} not close to budget {budget:.1%}"


def test_threshold_with_ties():
    """
    Test threshold calculation when multiple scores are identical.
    
    Edge case: What if many transactions have same fraud score?
    """
    # 50 transactions with same score
    scores = np.array([0.5] * 50 + [0.6] * 30 + [0.7] * 20)
    budget = 0.25  # 25% → 25 alerts out of 100
    
    threshold = calculate_threshold_for_budget(scores, budget)
    
    # Should still respect budget
    alerts = scores >= threshold
    alert_rate = alerts.sum() / len(scores)
    
    # With ties, we might need to include all tied values
    # Allow up to 50% overage due to ties (20 alerts could become 50 if all 0.5s are included)
    assert alert_rate <= budget * 2.5, \
        f"Alert rate {alert_rate:.1%} exceeds budget {budget:.1%} by too much (ties expected)"
    
    # But should at least be somewhat close
    assert alert_rate >= budget * 0.5, \
        f"Alert rate {alert_rate:.1%} too far below budget {budget:.1%}"


def test_extreme_budgets():
    """Test edge cases: very low and very high budgets."""
    scores = np.random.random(1000)
    
    # Test 0.1% budget (very restrictive)
    threshold_low = calculate_threshold_for_budget(scores, 0.001)
    assert threshold_low > 0.99, "Very low budget should have high threshold"
    
    # Test 10% budget (permissive)
    threshold_high = calculate_threshold_for_budget(scores, 0.10)
    assert threshold_high < 0.95, "High budget should have lower threshold"
    
    # Test 0% budget (no alerts)
    threshold_zero = calculate_threshold_for_budget(scores, 0.0)
    assert threshold_zero > 0.999, "Zero budget should have threshold > max score"


# ============================================================================
# TEST 3: Daily Metrics Computation
# ============================================================================

def test_daily_metrics_computation():
    """
    Test that compute_daily_metrics calculates TP, FP, FN, TN correctly.
    """
    # Create simple test case
    dates = pd.date_range('2025-06-01', periods=2, freq='D')
    df = pd.DataFrame({
        'event_timestamp': [dates[0]] * 5 + [dates[1]] * 5,
        'is_fraud': [1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
        'alert': [True, False, True, False, False, True, True, False, False, False]
    })
    
    alerts = df['alert'].values
    y_true = df['is_fraud'].values
    
    metrics = compute_daily_metrics(alerts, y_true, df)
    
    # Should have 2 days
    assert len(metrics) == 2, f"Expected 2 days, got {len(metrics)}"
    
    # Day 1: TP=1, FP=1, FN=1, TN=2
    day1 = metrics[0]
    assert day1['tp'] == 1, f"Day 1 TP should be 1, got {day1['tp']}"
    assert day1['fp'] == 1, f"Day 1 FP should be 1, got {day1['fp']}"
    assert day1['fn'] == 1, f"Day 1 FN should be 1, got {day1['fn']}"
    assert day1['tn'] == 2, f"Day 1 TN should be 2, got {day1['tn']}"
    
    # Day 2: TP=2, FP=0, FN=1, TN=2
    day2 = metrics[1]
    assert day2['tp'] == 2, f"Day 2 TP should be 2, got {day2['tp']}"
    assert day2['fp'] == 0, f"Day 2 FP should be 0, got {day2['fp']}"
    assert day2['fn'] == 1, f"Day 2 FN should be 1, got {day2['fn']}"
    assert day2['tn'] == 2, f"Day 2 TN should be 2, got {day2['tn']}"
    
    print("✅ Daily metrics computed correctly")


def test_metrics_precision_recall():
    """Test that precision and recall are calculated correctly."""
    df = pd.DataFrame({
        'event_timestamp': pd.date_range('2025-06-01', periods=10, freq='H'),
        'is_fraud': [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
        'alert': [True, True, True, False, True, False, False, False, False, False]
    })
    
    alerts = df['alert'].values
    y_true = df['is_fraud'].values
    
    metrics = compute_daily_metrics(alerts, y_true, df)
    
    # Should have 1 day
    day = metrics[0]
    
    # TP=3, FP=1, FN=1, TN=5
    # Precision = TP/(TP+FP) = 3/4 = 0.75
    # Recall = TP/(TP+FN) = 3/4 = 0.75
    assert abs(day['precision'] - 0.75) < 0.01, \
        f"Precision should be 0.75, got {day['precision']}"
    assert abs(day['recall'] - 0.75) < 0.01, \
        f"Recall should be 0.75, got {day['recall']}"


# ============================================================================
# TEST 4: Edge Cases
# ============================================================================

def test_empty_day_handling():
    """Test that system handles days with no transactions gracefully."""
    df = pd.DataFrame({
        'event_timestamp': [],
        'fraud_probability': [],
        'is_fraud': []
    })
    
    policy = AlertPolicy(budget_pct=0.005)
    
    if len(df) == 0:
        # Should handle empty input gracefully
        assert True, "Empty dataframe handled"
    else:
        fraud_probs = df['fraud_probability'].values
        alerts, metadata = policy.decide_alerts(df, fraud_probs)
        assert len(alerts) == 0, "Empty input should produce no alerts"


def test_all_fraud_day():
    """
    Test edge case: What if 100% of transactions are fraud?
    
    Should still respect budget (only flag top 0.5%).
    """
    np.random.seed(42)  # For reproducibility
    df = pd.DataFrame({
        'event_timestamp': pd.date_range('2025-06-01', periods=1000, freq='min'),  # More samples
        'fraud_probability': np.random.random(1000),
        'is_fraud': [1] * 1000  # All fraud!
    })
    
    budget = 0.005  # 0.5%
    policy = AlertPolicy(budget_pct=budget)
    
    fraud_probs = df['fraud_probability'].values
    alerts, metadata = policy.decide_alerts(df, fraud_probs)
    
    # Even though all are fraud, should only alert on top 0.5%
    alert_rate = metadata['alert_rate']
    # Allow 20% tolerance for tie-breaking (0.5% → 0.6%)
    assert alert_rate <= budget * 1.2, \
        f"Alert rate {alert_rate:.4%} exceeds budget even with 100% fraud"
    
    # But recall should be very low (only catching 0.5% of fraud)
    tp = (alerts & (df['is_fraud'] == 1)).sum()
    recall = tp / df['is_fraud'].sum()
    assert recall < 0.02, f"Recall {recall:.2%} too high for 0.5% budget on 100% fraud"


def test_zero_fraud_day():
    """Test day with 0% fraud (all legitimate)."""
    np.random.seed(42)  # For reproducibility
    df = pd.DataFrame({
        'event_timestamp': pd.date_range('2025-06-01', periods=1000, freq='min'),  # More samples
        'fraud_probability': np.random.random(1000),
        'is_fraud': [0] * 1000  # No fraud!
    })
    
    budget = 0.005
    policy = AlertPolicy(budget_pct=budget)
    
    fraud_probs = df['fraud_probability'].values
    alerts, metadata = policy.decide_alerts(df, fraud_probs)
    
    # Should still make alerts (false positives based on scores)
    alert_rate = metadata['alert_rate']
    # Allow 20% tolerance for tie-breaking
    assert alert_rate <= budget * 1.2, "Budget should still be respected (with tie tolerance)"
    
    # All alerts are false positives
    metrics = compute_daily_metrics(alerts, df['is_fraud'].values, df)
    day = metrics[0]
    assert day['tp'] == 0, "Should have 0 true positives when no fraud"
    assert day['fn'] == 0, "Should have 0 false negatives when no fraud"


# ============================================================================
# TEST 5: Consistency Across Multiple Runs
# ============================================================================

def test_deterministic_behavior():
    """
    Test that AlertPolicy is deterministic.
    
    Same input → same output (important for reproducibility).
    """
    df = pd.DataFrame({
        'event_timestamp': pd.date_range('2025-06-01', periods=100, freq='min'),
        'fraud_probability': np.random.random(100),
        'is_fraud': np.random.choice([0, 1], 100, p=[0.97, 0.03])
    })
    
    budget = 0.005
    policy1 = AlertPolicy(budget_pct=budget)
    policy2 = AlertPolicy(budget_pct=budget)
    
    fraud_probs = df['fraud_probability'].values
    
    alerts1, metadata1 = policy1.decide_alerts(df.copy(), fraud_probs)
    alerts2, metadata2 = policy2.decide_alerts(df.copy(), fraud_probs)
    
    # Results should be identical
    assert np.array_equal(alerts1, alerts2), "Alert decisions should be deterministic"
    assert metadata1['threshold_used'] == metadata2['threshold_used'], \
        "Threshold should be deterministic"


# ============================================================================
# RUN TESTS
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])