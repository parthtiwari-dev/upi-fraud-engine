"""
Fraud Detection Backtest Visualization

Creates professional, interactive visualizations of backtest results.
Generates both individual charts (PNG) and interactive dashboard (HTML).

Key Visualizations:
- Precision/Recall trends over time
- Alert budget compliance
- Fraud detection breakdown (caught vs missed)
- Financial impact (cost-benefit)
- Confusion matrix

Author: Your Name
Date: January 24, 2026
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Optional, List
import warnings

# Plotly for interactive charts
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Matplotlib for static charts (fallback)
import matplotlib.pyplot as plt
import seaborn as sns


def load_backtest_results(results_path: str) -> Dict:
    """
    Load backtest results from JSON file.
    
    Args:
        results_path: Path to backtest_results.json
        
    Returns:
        Dict with backtest results
    """
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results


def plot_precision_recall_over_time(
    daily_metrics: pd.DataFrame,
    output_path: Optional[str] = None,
    show: bool = True
) -> go.Figure:
    """
    Plot precision and recall trends over time.
    
    Args:
        daily_metrics: DataFrame with daily metrics
        output_path: Optional path to save chart
        show: Whether to display chart
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Precision line
    fig.add_trace(go.Scatter(
        x=daily_metrics['date'],
        y=daily_metrics['precision'] * 100,
        mode='lines+markers',
        name='Precision',
        line=dict(color='#2E86AB', width=3),
        marker=dict(size=8),
        hovertemplate='<b>%{x}</b><br>Precision: %{y:.1f}%<extra></extra>'
    ))
    
    # Recall line
    fig.add_trace(go.Scatter(
        x=daily_metrics['date'],
        y=daily_metrics['recall'] * 100,
        mode='lines+markers',
        name='Recall',
        line=dict(color='#A23B72', width=3),
        marker=dict(size=8),
        hovertemplate='<b>%{x}</b><br>Recall: %{y:.1f}%<extra></extra>'
    ))
    
    # Layout
    fig.update_layout(
        title={
            'text': 'Model Performance Over Time',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#333'}
        },
        xaxis_title='Date',
        yaxis_title='Percentage (%)',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01,
            bgcolor='rgba(255,255,255,0.8)'
        )
    )
    
    if output_path:
        fig.write_html(output_path.replace('.png', '.html'))
        fig.write_image(output_path)
    
    if show:
        fig.show()
    
    return fig


def plot_alert_budget_compliance(
    daily_metrics: pd.DataFrame,
    alert_budget: float = 0.005,
    output_path: Optional[str] = None,
    show: bool = True
) -> go.Figure:
    """
    Plot daily alert rates vs budget threshold.
    
    Args:
        daily_metrics: DataFrame with daily metrics
        alert_budget: Target alert budget (e.g., 0.005 for 0.5%)
        output_path: Optional path to save chart
        show: Whether to display chart
        
    Returns:
        Plotly figure
    """
    daily_metrics['alert_rate_pct'] = daily_metrics['alert_rate'] * 100
    budget_pct = alert_budget * 100
    
    # Color bars based on compliance
    colors = ['#06D6A0' if rate <= budget_pct * 1.01 else '#EF476F' 
              for rate in daily_metrics['alert_rate_pct']]
    
    fig = go.Figure()
    
    # Bar chart
    fig.add_trace(go.Bar(
        x=daily_metrics['date'],
        y=daily_metrics['alert_rate_pct'],
        marker_color=colors,
        name='Alert Rate',
        hovertemplate='<b>%{x}</b><br>Alert Rate: %{y:.2f}%<extra></extra>'
    ))
    
    # Budget threshold line
    fig.add_trace(go.Scatter(
        x=daily_metrics['date'],
        y=[budget_pct] * len(daily_metrics),
        mode='lines',
        name=f'Budget Target ({budget_pct:.1f}%)',
        line=dict(color='red', width=2, dash='dash'),
        hovertemplate='Budget: %{y:.2f}%<extra></extra>'
    ))
    
    # Layout
    fig.update_layout(
        title={
            'text': 'Alert Budget Compliance',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#333'}
        },
        xaxis_title='Date',
        yaxis_title='Alert Rate (%)',
        hovermode='x unified',
        template='plotly_white',
        height=500,
        showlegend=True
    )
    
    if output_path:
        fig.write_html(output_path.replace('.png', '.html'))
        fig.write_image(output_path)
    
    if show:
        fig.show()
    
    return fig


def plot_fraud_detection_breakdown(
    daily_metrics: pd.DataFrame,
    output_path: Optional[str] = None,
    show: bool = True
) -> go.Figure:
    """
    Stacked bar chart showing fraud caught vs missed daily.
    
    Args:
        daily_metrics: DataFrame with daily metrics
        output_path: Optional path to save chart
        show: Whether to display chart
        
    Returns:
        Plotly figure
    """
    fig = go.Figure()
    
    # Fraud caught (TP)
    fig.add_trace(go.Bar(
        x=daily_metrics['date'],
        y=daily_metrics['tp'],
        name='Fraud Caught',
        marker_color='#06D6A0',
        hovertemplate='<b>%{x}</b><br>Caught: %{y}<extra></extra>'
    ))
    
    # Fraud missed (FN)
    fig.add_trace(go.Bar(
        x=daily_metrics['date'],
        y=daily_metrics['fn'],
        name='Fraud Missed',
        marker_color='#EF476F',
        hovertemplate='<b>%{x}</b><br>Missed: %{y}<extra></extra>'
    ))
    
    # Layout
    fig.update_layout(
        title={
            'text': 'Fraud Detection Breakdown',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#333'}
        },
        xaxis_title='Date',
        yaxis_title='Number of Fraud Cases',
        barmode='stack',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    if output_path:
        fig.write_html(output_path.replace('.png', '.html'))
        fig.write_image(output_path)
    
    if show:
        fig.show()
    
    return fig


def plot_cumulative_financial_impact(
    daily_metrics: pd.DataFrame,
    avg_fraud_loss: float = 50000,
    investigation_cost: float = 500,
    output_path: Optional[str] = None,
    show: bool = True
) -> go.Figure:
    """
    Plot cumulative financial impact (savings) over time.
    
    Args:
        daily_metrics: DataFrame with daily metrics
        avg_fraud_loss: Average loss per fraud (INR)
        investigation_cost: Cost per investigation (INR)
        output_path: Optional path to save chart
        show: Whether to display chart
        
    Returns:
        Plotly figure
    """
    # Calculate daily financial impact
    daily_metrics['fraud_prevented'] = daily_metrics['tp'] * avg_fraud_loss
    daily_metrics['investigation_costs'] = daily_metrics['num_alerts'] * investigation_cost
    daily_metrics['daily_savings'] = daily_metrics['fraud_prevented'] - daily_metrics['investigation_costs']
    
    # Cumulative savings
    daily_metrics['cumulative_savings'] = daily_metrics['daily_savings'].cumsum()
    
    fig = go.Figure()
    
    # Cumulative savings line
    fig.add_trace(go.Scatter(
        x=daily_metrics['date'],
        y=daily_metrics['cumulative_savings'] / 1_000_000,  # Convert to millions
        mode='lines+markers',
        name='Cumulative Savings',
        line=dict(color='#06D6A0', width=3),
        marker=dict(size=10),
        fill='tozeroy',
        fillcolor='rgba(6, 214, 160, 0.1)',
        hovertemplate='<b>%{x}</b><br>Savings: ₹%{y:.2f}M<extra></extra>'
    ))
    
    # Layout
    fig.update_layout(
        title={
            'text': 'Cumulative Financial Impact',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#333'}
        },
        xaxis_title='Date',
        yaxis_title='Cumulative Savings (₹ Millions)',
        hovermode='x unified',
        template='plotly_white',
        height=500
    )
    
    if output_path:
        fig.write_html(output_path.replace('.png', '.html'))
        fig.write_image(output_path)
    
    if show:
        fig.show()
    
    return fig


def plot_confusion_matrix(
    cumulative_metrics: Dict,
    output_path: Optional[str] = None,
    show: bool = True
) -> go.Figure:
    """
    Plot confusion matrix heatmap.
    
    Args:
        cumulative_metrics: Dict with cumulative TP, FP, FN, TN
        output_path: Optional path to save chart
        show: Whether to display chart
        
    Returns:
        Plotly figure
    """
    # Extract metrics
    tp = cumulative_metrics['total_caught']
    fp = cumulative_metrics['total_false_positives']
    fn = cumulative_metrics['total_missed']
    tn = cumulative_metrics['total_transactions'] - tp - fp - fn
    
    # Create confusion matrix
    cm = np.array([[tn, fp], [fn, tp]])
    
    # Annotations
    annotations = []
    for i in range(2):
        for j in range(2):
            annotations.append(
                dict(
                    text=f"{cm[i, j]:,}",
                    x=['Predicted Legit', 'Predicted Fraud'][j],
                    y=['Actual Legit', 'Actual Fraud'][i],
                    showarrow=False,
                    font=dict(size=20, color='white' if cm[i, j] > cm.max() / 2 else 'black')
                )
            )
    
    fig = go.Figure(data=go.Heatmap(
        z=cm,
        x=['Predicted Legit', 'Predicted Fraud'],
        y=['Actual Legit', 'Actual Fraud'],
        colorscale='Blues',
        showscale=True,
        hovertemplate='%{y} & %{x}<br>Count: %{z:,}<extra></extra>'
    ))
    
    fig.update_layout(
        title={
            'text': 'Confusion Matrix',
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 20, 'color': '#333'}
        },
        annotations=annotations,
        template='plotly_white',
        height=500,
        width=600
    )
    
    if output_path:
        fig.write_html(output_path.replace('.png', '.html'))
        fig.write_image(output_path)
    
    if show:
        fig.show()
    
    return fig


def generate_full_report(
    results_path: str,
    output_dir: str = 'evaluation/visualizations',
    show_charts: bool = False
) -> Dict[str, str]:
    """
    Generate complete visualization report from backtest results.
    
    Args:
        results_path: Path to backtest_results.json
        output_dir: Directory to save visualizations
        show_charts: Whether to display charts interactively
        
    Returns:
        Dict with paths to generated files
    """
    print(f"{'=' * 70}")
    print(f"GENERATING VISUALIZATION REPORT")
    print(f"{'=' * 70}")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load results
    print(f"\n📂 Loading results from {results_path}...")
    results = load_backtest_results(results_path)
    
    daily_metrics = pd.DataFrame(results['daily_metrics'])
    cumulative_metrics = results['cumulative_metrics']
    config = results['config']
    
    print(f"✅ Loaded {len(daily_metrics)} days of data")
    
    # Generate charts
    generated_files = {}
    
    print(f"\n📊 Generating charts...")
    
    # 1. Precision/Recall over time
    print(f"   1. Precision/Recall trend...")
    fig1 = plot_precision_recall_over_time(
        daily_metrics,
        output_path=str(output_path / 'precision_recall_trend.png'),
        show=show_charts
    )
    generated_files['precision_recall'] = str(output_path / 'precision_recall_trend.html')
    
    # 2. Alert budget compliance
    print(f"   2. Alert budget compliance...")
    fig2 = plot_alert_budget_compliance(
        daily_metrics,
        alert_budget=config['alert_budget'],
        output_path=str(output_path / 'alert_budget_compliance.png'),
        show=show_charts
    )
    generated_files['budget_compliance'] = str(output_path / 'alert_budget_compliance.html')
    
    # 3. Fraud detection breakdown
    print(f"   3. Fraud detection breakdown...")
    fig3 = plot_fraud_detection_breakdown(
        daily_metrics,
        output_path=str(output_path / 'fraud_breakdown.png'),
        show=show_charts
    )
    generated_files['fraud_breakdown'] = str(output_path / 'fraud_breakdown.html')
    
    # 4. Financial impact
    print(f"   4. Financial impact...")
    fig4 = plot_cumulative_financial_impact(
        daily_metrics,
        output_path=str(output_path / 'financial_impact.png'),
        show=show_charts
    )
    generated_files['financial_impact'] = str(output_path / 'financial_impact.html')
    
    # 5. Confusion matrix
    print(f"   5. Confusion matrix...")
    fig5 = plot_confusion_matrix(
        cumulative_metrics,
        output_path=str(output_path / 'confusion_matrix.png'),
        show=show_charts
    )
    generated_files['confusion_matrix'] = str(output_path / 'confusion_matrix.html')
    
    # Create combined dashboard
    print(f"\n📈 Creating interactive dashboard...")
    create_dashboard([fig1, fig2, fig3, fig4, fig5], output_path / 'dashboard.html')
    generated_files['dashboard'] = str(output_path / 'dashboard.html')
    
    print(f"\n{'=' * 70}")
    print(f"✅ REPORT GENERATED SUCCESSFULLY")
    print(f"{'=' * 70}")
    print(f"\n📁 Output directory: {output_dir}/")
    print(f"\n📊 Generated files:")
    for name, path in generated_files.items():
        print(f"   - {name}: {path}")
    print(f"\n💡 Open dashboard.html in browser for interactive view!")
    print(f"{'=' * 70}\n")
    
    return generated_files


def create_dashboard(figures: List[go.Figure], output_path: Path):
    """
    Create combined HTML dashboard with all charts.
    
    Args:
        figures: List of Plotly figures
        output_path: Path to save dashboard
    """
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Fraud Detection Backtest Report</title>
        <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 20px;
                background-color: #f5f5f5;
            }
            h1 {
                text-align: center;
                color: #333;
            }
            .chart-container {
                background-color: white;
                padding: 20px;
                margin: 20px 0;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            }
        </style>
    </head>
    <body>
        <h1>🎯 Fraud Detection Backtest Report</h1>
    """
    
    for i, fig in enumerate(figures):
        html += f'<div class="chart-container" id="chart{i}"></div>\n'
        html += f'<script>\n'
        html += f'var data{i} = {fig.to_json()};\n'
        html += f'Plotly.newPlot("chart{i}", data{i}.data, data{i}.layout);\n'
        html += f'</script>\n'
    
    html += """
    </body>
    </html>
    """
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html)


# ============================================================================
# TESTING EXAMPLE (Run with: python -m src.evaluation.visualize)
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("VISUALIZATION MODULE - DEMO")
    print("=" * 70)
    
    # Check if backtest results exist
    results_path = 'evaluation/backtest_results/backtest_results.json'
    
    if Path(results_path).exists():
        print(f"\n✅ Found backtest results: {results_path}")
        print(f"\nGenerating visualization report...\n")
        
        # Generate full report
        generated_files = generate_full_report(
            results_path=results_path,
            output_dir='evaluation/visualizations',
            show_charts=False  # Set to True to display in browser
        )
        
        print(f"\n✅ Visualization complete!")
        print(f"\n💡 Next steps:")
        print(f"   1. Open evaluation/visualizations/dashboard.html in browser")
        print(f"   2. View individual charts as needed")
        print(f"   3. Use charts for portfolio/LinkedIn!")
        
    else:
        print(f"\n❌ Backtest results not found: {results_path}")
        print(f"\n💡 Run backtest first:")
        print(f"   python -m src.evaluation.backtest")
