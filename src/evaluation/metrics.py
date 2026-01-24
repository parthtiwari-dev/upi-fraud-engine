"""
Business-Focused Fraud Detection Metrics

Why accuracy is meaningless in fraud:
- 96.4% legitimate → predict all legitimate = 96.4% accuracy but 0% fraud caught!
- Real constraint: limited investigation capacity (e.g., can only review 0.5% of transactions)

Primary Metric: Precision at Fixed Alert Budget (e.g., 0.5% daily alert rate)
    "Of the 0.5% we flag, what % are actually fraud?"

Secondary Metrics:
- Recall at budget: "What % of all fraud did we catch?"
- False alert rate: "What % of alerts are false positives?"
- Cost-benefit analysis: fraud prevented vs investigation cost

This mirrors real operational constraints where:
- Investigation team has limited capacity
- Each alert costs money to review
- Missing fraud costs more than false alerts

Author: Your Name
Date: January 24, 2026
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from datetime import datetime, date
import warnings

from src.evaluation.alert_policy import (
    AlertPolicy, 
    calculate_threshold_for_budget,
    compute_daily_metrics
)


def precision_at_alert_budget(
    y_true: np.ndarray,
    fraud_probs: np.ndarray,
    transactions: pd.DataFrame,
    alert_budget: float = 0.005,
    verbose: bool = True
) -> Dict:
    """
    Calculate precision/recall at fixed daily alert budget.
    
    Real-world constraint: Investigation team can only review X% of transactions.
    We want to maximize fraud caught within this budget.
    
    Args:
        y_true: True labels (0 or 1), shape (N,)
        fraud_probs: Predicted fraud probabilities [0-1], shape (N,)
        transactions: DataFrame with 'event_timestamp' column
        alert_budget: Fraction of transactions to flag (default: 0.5%)
        verbose: Print detailed breakdown
        
    Returns:
        Dict with metrics at the budget threshold:
            - 'alert_budget': float (input budget)
            - 'threshold': float (fraud prob cutoff)
            - 'actual_alert_rate': float (realized alert rate)
            - 'precision': float (TP / (TP + FP))
            - 'recall': float (TP / (TP + FN))
            - 'false_alert_rate': float (FP / (FP + TN))
            - 'true_positives': int
            - 'false_positives': int
            - 'false_negatives': int
            - 'true_negatives': int
            - 'total_fraud': int
            - 'fraud_caught': int
            - 'fraud_missed': int
            - 'total_flagged': int
    
    Example:
        >>> metrics = precision_at_alert_budget(y_test, fraud_probs, df, alert_budget=0.005)
        >>> print(f"At 0.5% budget: {metrics['precision']:.1%} precision")
    """
    if len(y_true) != len(fraud_probs) or len(y_true) != len(transactions):
        raise ValueError(
            f"Length mismatch: y_true={len(y_true)}, "
            f"fraud_probs={len(fraud_probs)}, transactions={len(transactions)}"
        )
    
    # Apply alert policy to get decisions
    policy = AlertPolicy(budget_pct=alert_budget)
    alerts, metadata = policy.decide_alerts(transactions, fraud_probs)
    
    # Calculate confusion matrix
    tp = ((alerts == True) & (y_true == 1)).sum()
    fp = ((alerts == True) & (y_true == 0)).sum()
    fn = ((alerts == False) & (y_true == 1)).sum()
    tn = ((alerts == False) & (y_true == 0)).sum()
    
    # Calculate metrics
    total_flagged = tp + fp
    total_fraud = tp + fn
    fraud_caught = tp
    fraud_missed = fn
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    false_alert_rate = fp / (tn + fp) if (tn + fp) > 0 else 0.0
    
    results = {
        'alert_budget': alert_budget,
        'threshold': metadata['threshold_used'],
        'actual_alert_rate': metadata['alert_rate'],
        'precision': float(precision),
        'recall': float(recall),
        'false_alert_rate': float(false_alert_rate),
        'true_positives': int(tp),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_negatives': int(tn),
        'total_fraud': int(total_fraud),
        'fraud_caught': int(fraud_caught),
        'fraud_missed': int(fraud_missed),
        'total_flagged': int(total_flagged)
    }
    
    if verbose:
        print(f"{'=' * 70}")
        print(f"PRECISION AT {alert_budget:.1%} ALERT BUDGET")
        print(f"{'=' * 70}")
        print(f"\n📊 Constraints:")
        print(f"   Daily transaction volume: {len(y_true):,}")
        print(f"   Investigation capacity: {alert_budget:.1%} ({int(len(y_true) * alert_budget):,} alerts/day)")
        print(f"   Fraud rate in population: {total_fraud / len(y_true):.2%}")
        
        print(f"\n🎯 Performance at Threshold {metadata['threshold_used']:.4f}:")
        print(f"   Transactions flagged: {total_flagged:,} ({results['actual_alert_rate']:.2%})")
        print(f"   Precision (PPV): {precision:.2%} ({tp} frauds / {total_flagged} alerts)")
        print(f"   Recall (Sensitivity): {recall:.2%} (Caught {fraud_caught}/{total_fraud} frauds)")
        print(f"   False alert rate: {false_alert_rate:.4%} ({fp} false alarms)")
        
        print(f"\n💡 Impact:")
        print(f"   Fraud caught: {fraud_caught:,}/{total_fraud:,} ({recall:.1%})")
        print(f"   Fraud missed: {fraud_missed:,}/{total_fraud:,} ({fraud_missed/total_fraud:.1%})")
        print(f"   Wasted investigations: {fp:,}/{total_flagged:,} ({fp/total_flagged:.1%})")
        print(f"{'=' * 70}\n")
    
    return results


def alert_budget_curve(
    y_true: np.ndarray,
    fraud_probs: np.ndarray,
    transactions: pd.DataFrame,
    budgets: Optional[List[float]] = None,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Calculate precision/recall across different alert budgets.
    
    Shows trade-off: higher budget → more fraud caught but lower precision.
    
    Args:
        y_true: True labels
        fraud_probs: Predicted probabilities
        transactions: DataFrame with 'event_timestamp'
        budgets: List of alert budgets to test (default: [0.1%, 0.5%, 1%, 2%, 5%])
        verbose: Print summary table
        
    Returns:
        DataFrame with columns:
            - alert_budget: float
            - precision: float
            - recall: float
            - fraud_caught: int
            - false_positives: int
            - threshold: float
    
    Example:
        >>> curve_df = alert_budget_curve(y_test, fraud_probs, df)
        >>> # Find optimal budget for 80% precision target
        >>> optimal = curve_df[curve_df['precision'] >= 0.8].iloc[0]
    """
    if budgets is None:
        budgets = [0.001, 0.005, 0.01, 0.02, 0.05]  # 0.1%, 0.5%, 1%, 2%, 5%
    
    results = []
    
    for budget in budgets:
        metrics = precision_at_alert_budget(
            y_true, fraud_probs, transactions, budget, verbose=False
        )
        results.append(metrics)
    
    df = pd.DataFrame(results)
    
    if verbose:
        print(f"{'=' * 70}")
        print(f"ALERT BUDGET TRADE-OFF CURVE")
        print(f"{'=' * 70}")
        print(f"\nFinding optimal budget for your investigation team capacity...\n")
        
        # Display table
        display_df = df[['alert_budget', 'precision', 'recall', 'fraud_caught', 'false_positives']].copy()
        display_df['alert_budget'] = display_df['alert_budget'].apply(lambda x: f"{x:.1%}")
        display_df['precision'] = display_df['precision'].apply(lambda x: f"{x:.1%}")
        display_df['recall'] = display_df['recall'].apply(lambda x: f"{x:.1%}")
        
        print(display_df.to_string(index=False))
        print(f"{'=' * 70}")
        
        # Recommendation
        target_precision = 0.70  # 70% precision is common industry target
        meets_target = df[df['precision'] >= target_precision]
        
        if len(meets_target) > 0:
            optimal = meets_target.iloc[-1]  # Highest budget with ≥70% precision
            print(f"\n💡 RECOMMENDATION:")
            print(f"   For ≥70% precision target:")
            print(f"   Use {optimal['alert_budget']:.1%} budget")
            print(f"   Catch {optimal['recall']:.1%} of fraud ({optimal['fraud_caught']} frauds)")
            print(f"   {optimal['precision']:.1%} precision, {optimal['false_positives']} false alerts")
        else:
            print(f"\n⚠️  Cannot achieve 70% precision at these budgets")
            print(f"   Consider model improvements or lower precision target")
        
        print(f"{'=' * 70}\n")
    
    return df


def cost_benefit_analysis(
    y_true: np.ndarray,
    fraud_probs: np.ndarray,
    transactions: pd.DataFrame,
    alert_budget: float = 0.005,
    avg_fraud_loss: float = 50000,  # INR
    investigation_cost: float = 500,  # INR per alert
    verbose: bool = True
) -> Dict:
    """
    Calculate financial impact of fraud detection system.
    
    Compares cost of investigations vs fraud prevented.
    
    Args:
        y_true: True labels
        fraud_probs: Predicted probabilities
        transactions: DataFrame with 'event_timestamp'
        alert_budget: Alert budget to evaluate
        avg_fraud_loss: Average loss per fraud incident (INR)
        investigation_cost: Cost to investigate one alert (INR)
        verbose: Print detailed breakdown
        
    Returns:
        Dict with financial metrics:
            - All metrics from precision_at_alert_budget()
            - 'avg_fraud_loss': float
            - 'investigation_cost': float
            - 'fraud_prevented_value': float (₹)
            - 'fraud_losses': float (₹, missed frauds)
            - 'investigation_costs': float (₹)
            - 'net_savings': float (₹)
            - 'roi': float (return on investment, %)
    
    Example:
        >>> financials = cost_benefit_analysis(
        ...     y_test, fraud_probs, df,
        ...     avg_fraud_loss=50000,  # ₹50K INR average fraud
        ...     investigation_cost=500  # ₹500 INR per investigation
        ... )
        >>> print(f"Net savings: ₹{financials['net_savings']:,.0f}")
    """
    # Get precision/recall metrics
    metrics = precision_at_alert_budget(
        y_true, fraud_probs, transactions, alert_budget, verbose=False
    )
    
    # Calculate financial impact
    fraud_prevented_value = metrics['fraud_caught'] * avg_fraud_loss
    fraud_losses = metrics['fraud_missed'] * avg_fraud_loss
    investigation_costs = metrics['total_flagged'] * investigation_cost
    net_savings = fraud_prevented_value - investigation_costs
    roi = (fraud_prevented_value / investigation_costs - 1) if investigation_costs > 0 else 0
    
    results = {
        **metrics,  # Include all precision/recall metrics
        'avg_fraud_loss': avg_fraud_loss,
        'investigation_cost': investigation_cost,
        'fraud_prevented_value': fraud_prevented_value,
        'fraud_losses': fraud_losses,
        'investigation_costs': investigation_costs,
        'net_savings': net_savings,
        'roi': roi
    }
    
    if verbose:
        print(f"{'=' * 70}")
        print(f"COST-BENEFIT ANALYSIS")
        print(f"{'=' * 70}")
        print(f"\n💰 Assumptions:")
        print(f"   Average fraud loss: ₹{avg_fraud_loss:,}")
        print(f"   Investigation cost: ₹{investigation_cost:,} per alert")
        print(f"   Alert budget: {alert_budget:.1%}")
        
        print(f"\n📊 Impact:")
        print(f"   Fraud prevented: ₹{fraud_prevented_value:,.0f} ({metrics['fraud_caught']} frauds × ₹{avg_fraud_loss:,})")
        print(f"   Investigation costs: ₹{investigation_costs:,.0f} ({metrics['total_flagged']} alerts × ₹{investigation_cost:,})")
        print(f"   {'-' * 68}")
        print(f"   NET SAVINGS: ₹{net_savings:,.0f}")
        print(f"   ROI: {roi:.1%}")
        
        print(f"\n⚠️  Risk:")
        print(f"   Fraud losses (missed): ₹{fraud_losses:,.0f} ({metrics['fraud_missed']} frauds)")
        
        # Break-even analysis
        print(f"\n📈 Break-Even Analysis:")
        if net_savings > 0:
            print(f"   ✅ System is profitable!")
            print(f"   Saves ₹{net_savings:,.0f} compared to no fraud detection")
        else:
            print(f"   ❌ System costs more than it saves")
            print(f"   Need to reduce investigation costs or improve precision")
        
        print(f"{'=' * 70}\n")
    
    return results


def generate_evaluation_report(
    y_true: np.ndarray,
    fraud_probs: np.ndarray,
    transactions: pd.DataFrame,
    model_name: str = "Fraud Detection Model",
    save_path: Optional[str] = None
) -> Dict:
    """
    Generate complete business-focused evaluation report.
    
    Args:
        y_true: True labels
        fraud_probs: Predicted probabilities
        transactions: DataFrame with 'event_timestamp'
        model_name: Model identifier for report
        save_path: Optional path to save report (JSON)
        
    Returns:
        Dict with all metrics
    """
    print(f"{'=' * 70}")
    print(f"COMPREHENSIVE FRAUD DETECTION EVALUATION")
    print(f"Model: {model_name}")
    print(f"{'=' * 70}\n")
    
    # Standard ML metrics (for reference, but not primary focus)
    from sklearn.metrics import roc_auc_score, average_precision_score
    
    roc_auc = roc_auc_score(y_true, fraud_probs)
    pr_auc = average_precision_score(y_true, fraud_probs)
    
    print(f"📊 Standard ML Metrics (for reference):")
    print(f"   ROC-AUC: {roc_auc:.4f}")
    print(f"   PR-AUC: {pr_auc:.4f}")
    print(f"   Note: These don't reflect business constraints!\n")
    
    # Business metrics at typical alert budget
    print(f"🎯 Primary Business Metric:")
    budget_metrics = precision_at_alert_budget(
        y_true, fraud_probs, transactions, alert_budget=0.005, verbose=True
    )
    
    # Alert budget curve
    curve_df = alert_budget_curve(y_true, fraud_probs, transactions, verbose=True)
    
    # Cost-benefit analysis
    financials = cost_benefit_analysis(
        y_true, fraud_probs, transactions,
        alert_budget=0.005,
        avg_fraud_loss=50000,
        investigation_cost=500,
        verbose=True
    )
    
    report = {
        'model_name': model_name,
        'ml_metrics': {
            'roc_auc': float(roc_auc),
            'pr_auc': float(pr_auc)
        },
        'business_metrics_0.5pct': budget_metrics,
        'alert_budget_curve': curve_df.to_dict('records'),
        'cost_benefit_analysis': financials
    }
    
    if save_path:
        import json
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"📄 Report saved to {save_path}")
    
    return report


# ============================================================================
# TESTING EXAMPLE (Run with: python -m src.evaluation.metrics)
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("EVALUATION METRICS - TEST")
    print("=" * 70)
    
    # Generate example data (synthetic)
    np.random.seed(42)
    n_samples = 10000
    fraud_rate = 0.036
    
    # True labels
    y_true = np.random.binomial(1, fraud_rate, n_samples)
    
    # Simulate good model: frauds get higher scores
    fraud_probs = np.where(
        y_true == 1,
        np.random.beta(8, 2, n_samples),  # Frauds: high scores
        np.random.beta(2, 8, n_samples)   # Legit: low scores
    )
    
    # Create transactions DataFrame
    dates = pd.date_range('2025-01-01', periods=10, freq='D')
    transactions = []
    
    for i, prob in enumerate(fraud_probs):
        day_idx = i % len(dates)
        transactions.append({
            'event_timestamp': dates[day_idx] + pd.Timedelta(seconds=np.random.randint(0, 86400)),
            'fraud_probability': prob,
            'is_fraud': y_true[i]
        })
    
    df = pd.DataFrame(transactions)
    
    print(f"\n✅ Generated {len(df):,} synthetic transactions")
    print(f"   Fraud rate: {y_true.mean():.2%}")
    print(f"   Date range: {df['event_timestamp'].min().date()} to {df['event_timestamp'].max().date()}")
    
    # Test 1: Precision at alert budget
    print(f"\n{'=' * 70}")
    print("TEST 1: Precision at Alert Budget (0.5%)")
    print("=" * 70)
    
    metrics = precision_at_alert_budget(
        y_true, fraud_probs, df, alert_budget=0.005, verbose=True
    )
    
    # Test 2: Alert budget curve
    print(f"\n{'=' * 70}")
    print("TEST 2: Alert Budget Curve")
    print("=" * 70)
    
    curve_df = alert_budget_curve(y_true, fraud_probs, df, verbose=True)
    
    # Test 3: Cost-benefit analysis
    print(f"\n{'=' * 70}")
    print("TEST 3: Cost-Benefit Analysis")
    print("=" * 70)
    
    financials = cost_benefit_analysis(
        y_true, fraud_probs, df,
        alert_budget=0.005,
        avg_fraud_loss=50000,
        investigation_cost=500,
        verbose=True
    )
    
    # Test 4: Complete report
    print(f"\n{'=' * 70}")
    print("TEST 4: Complete Evaluation Report")
    print("=" * 70)
    
    report = generate_evaluation_report(
        y_true, fraud_probs, df,
        model_name="Example Model (Synthetic Data)",
        save_path=None  # Don't save for test
    )
    
    print(f"\n{'=' * 70}")
    print("ALL TESTS PASSED ✅")
    print("=" * 70)
    print(f"\n💡 Usage with real model predictions:")
    print(f"   from src.evaluation.metrics import precision_at_alert_budget")
    print(f"   metrics = precision_at_alert_budget(y_test, fraud_probs, df, alert_budget=0.005)")
