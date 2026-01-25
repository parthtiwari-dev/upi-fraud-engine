import streamlit as st
import requests
import plotly.graph_objects as go
from datetime import datetime

# Configuration
API_URL = "https://upi-fraud-engine.onrender.com"  

st.set_page_config(page_title="UPI Fraud Detection", page_icon="🚨", layout="wide")

st.title("🚨 UPI Fraud Detection System")
st.markdown("Real-time fraud scoring with <500ms latency")
st.info(
    "This fraud detection system mirrors real fintech architectures: "
    "alerts are triggered only for the top-risk fraction of transactions. "
    "Low demo volume may result in fewer alerts by design."
)


# Sidebar - Transaction Input
st.sidebar.header("📝 Transaction Details")
txn = {
    "transaction_id": st.sidebar.text_input("Transaction ID", value=f"TXN_{int(datetime.now().timestamp())}"),
    "event_timestamp": datetime.now().isoformat(),
    "amount": st.sidebar.number_input("Amount (₹)", min_value=1.0, value=1000.0),
    "payer_vpa": st.sidebar.text_input("Payer UPI", value="user@paytm"),
    "payee_vpa": st.sidebar.text_input("Payee UPI", value="merchant@phonepe"),
    "device_id": st.sidebar.text_input("Device ID", value="device_123"),
    "currency": "INR"
}

# Score Button
if st.sidebar.button("🔍 Score Transaction", use_container_width=True):
    with st.spinner("Analyzing fraud risk..."):
        try:
            response = requests.post(f"{API_URL}/score", json=txn, timeout=2)
            result = response.json()
            
            # Display Results
            col1, col2, col3 = st.columns(3)
            col1.metric("Fraud Probability", f"{result['fraud_probability']*100:.2f}%")
            col2.metric("Risk Tier", result['risk_tier'].upper())
            col3.metric("Latency", f"{result['latency_ms']:.0f}ms")
            
            # Alert Decision
            if result['should_alert']:
                st.error("🚨 **HIGH RISK**: This transaction should be flagged for review!")
            else:
                st.success("✅ **LOW RISK**: Transaction appears legitimate")
            
            # Fraud Gauge
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=result['fraud_probability'] * 100,
                title={'text': "Fraud Risk Score"},
                gauge={
                    'axis': {'range': [0, 100]},
                    'bar': {'color': "darkred"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ]
                }
            ))
            st.plotly_chart(fig, use_container_width=True)
            
            # Details
            with st.expander("📊 Detailed Results"):
                st.json(result)
                
        except Exception as e:
            st.error(f"❌ Error: {e}")

# API Metrics
st.sidebar.markdown("---")
st.sidebar.header("📈 API Metrics")
try:
    metrics = requests.get(f"{API_URL}/metrics", timeout=2).json()
    st.sidebar.metric("Total Requests", metrics['total_requests'])
    st.sidebar.metric("Alert Rate", f"{metrics['alert_rate']*100:.2f}%")
    st.sidebar.metric("Avg Latency", f"{metrics['avg_latency_ms']:.0f}ms")
except:
    st.sidebar.warning("Could not fetch metrics")