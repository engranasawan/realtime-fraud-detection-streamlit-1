import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ==============================
# 1. Load artifacts (cached)
# ==============================


@st.cache_resource
def load_artifacts():
    models_dir = Path("models")

    def _load(name: str):
        path = models_dir / name
        try:
            return joblib.load(path)
        except Exception as e:
            st.error(f"❌ Error loading {name}")
            st.exception(e)
            raise

    supervised_pipeline = _load("supervised_lgbm_pipeline.joblib")
    iforest_pipeline = _load("iforest_pipeline.joblib")

    return supervised_pipeline, iforest_pipeline


supervised_pipeline, iforest_pipeline = load_artifacts()

# ==============================
# 2. Risk thresholds & logic
# ==============================

FRAUD_MED = 0.00005
FRAUD_HIGH = 0.00023328
FRAUD_CRIT = 0.01732857

ANOM_MED = 0.04
ANOM_HIGH = 0.05
ANOM_CRIT = 0.08


def risk_from_scores(fraud_prob: float, anomaly_score: float) -> str:
    if fraud_prob >= FRAUD_CRIT or anomaly_score >= ANOM_CRIT:
        return "CRITICAL"
    elif fraud_prob >= FRAUD_HIGH or (
        fraud_prob >= FRAUD_MED and anomaly_score >= ANOM_HIGH
    ):
        return "HIGH"
    elif fraud_prob >= FRAUD_MED or anomaly_score >= ANOM_MED:
        return "MEDIUM"
    else:
        return "LOW"


def score_transaction(input_dict: dict):
    """
    input_dict fields:
    Amount, TransactionType, Location, DeviceID, Channel, hour, day_of_week, month
    """
    df = pd.DataFrame([input_dict])

    # Supervised fraud probability
    fraud_prob = float(supervised_pipeline.predict_proba(df)[0, 1])

    # IsolationForest decision_function: larger → more normal
    raw_score = float(iforest_pipeline.decision_function(df)[0])
    anomaly_score = -raw_score  # invert: higher = more anomalous

    risk = risk_from_scores(fraud_prob, anomaly_score)
    return fraud_prob, anomaly_score, risk


def render_risk_badge(risk: str):
    color_map = {
        "LOW": "#2e7d32",
        "MEDIUM": "#f9a825",
        "HIGH": "#f57c00",
        "CRITICAL": "#c62828",
    }
    risk_color = color_map.get(risk, "#607d8b")

    st.markdown(
        f"""
        <div style="padding: 0.75rem 1rem; border-radius: 0.5rem;
                    background-color: {risk_color}22; border: 1px solid {risk_color};">
            <span style="font-size: 1.1rem; font-weight: 600; color: {risk_color};">
                Risk Level: {risk}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ==============================
# 3. Streamlit UI
# ==============================

st.set_page_config(
    page_title="Real-Time Fraud Detection Demo",
    page_icon="💳",
    layout="centered",
)

st.title("💳 Real-Time Fraud Detection Prototype")
st.write(
    "This demo uses a **supervised LightGBM model** and an **unsupervised Isolation Forest** "
    "to assess the fraud risk of a single transaction in real time."
)

st.markdown("---")

st.sidebar.header("⚙️ Configuration")
st.sidebar.markdown(
    """
**How this works:**

- A single **pipeline** handles preprocessing + the LightGBM classifier.
- A second pipeline handles preprocessing + Isolation Forest.
- The two outputs are combined into a final **risk level**:
  - LOW / MEDIUM / HIGH / CRITICAL
"""
)

st.header("🧾 Enter Transaction Details")

with st.form("txn_form"):
    col1, col2 = st.columns(2)

    with col1:
        amount = st.number_input(
            "Transaction Amount", min_value=0.0, value=1200.0, step=10.0
        )
        txn_type = st.selectbox(
            "Transaction Type",
            ["PAYMENT", "TRANSFER", "DEBIT", "CREDIT", "CASH_OUT", "OTHER"],
            index=0,
        )
        channel = st.selectbox(
            "Channel",
            ["Mobile", "NetBanking", "Online", "Card", "ATM", "Other"],
            index=2,
        )

    with col2:
        location = st.text_input("Location (City / Region)", value="Karachi")
        device = st.selectbox(
            "Device / OS",
            ["Android", "iOS", "Windows", "Linux", "Other"],
            index=0,
        )
        txn_date = st.date_input("Transaction Date", value=datetime.date.today())
        txn_time = st.time_input(
            "Transaction Time", value=datetime.datetime.now().time()
        )

    submitted = st.form_submit_button("🚀 Run Fraud Check")

if submitted:
    txn_datetime = datetime.datetime.combine(txn_date, txn_time)
    hour = txn_datetime.hour
    day_of_week = txn_datetime.weekday()
    month = txn_datetime.month

    input_payload = {
        "Amount": amount,
        "TransactionType": txn_type,
        "Location": location,
        "DeviceID": device,
        "Channel": channel,
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
    }

    with st.spinner("Scoring transaction..."):
        fraud_prob, anomaly_score, risk = score_transaction(input_payload)

    st.markdown("## 🔎 Results")
    render_risk_badge(risk)

    col_a, col_b = st.columns(2)
    with col_a:
        st.metric(
            "Fraud Probability (LightGBM)",
            f"{fraud_prob:.8f}",
            help="Direct output probability from the supervised model.",
        )
    with col_b:
        st.metric(
            "Anomaly Score (Isolation Forest)",
            f"{anomaly_score:.5f}",
            help="Higher = more unusual compared to 'normal' historical patterns.",
        )

    st.markdown("### 📦 Model Input Payload")
    st.json(input_payload)

    st.markdown(
        """
        ### 🧠 How to interpret this
        
        - **Fraud Probability** comes from a supervised LightGBM trained on labeled fraud data.
        - **Anomaly Score** comes from Isolation Forest trained on normal-only data.
        - The final **Risk Level** is determined using thresholds calibrated on a validation set.
        """
    )
else:
    st.info("Fill the form above and click **Run Fraud Check** to score a transaction.")
