# app.py
"""
Real-Time Fraud Detection Prototype (ML + Rules)
- Dynamic channel-specific UI (Bank, Mobile App, ATM, Credit Card, POS, Online Purchase, NetBanking)
- ML scoring using pre-saved supervised_pipeline & iforest_pipeline (joblib)
- Extended Rule Engine:
    * Velocity rules (1h, 24h, 7d)
    * Behavioural anomalies (device churn, new device + new location + high amount)
    * IP / geo mismatch, impossible travel (Haversine)
    * Spending spike vs monthly average & rolling average
    * Channel-specific rules (card, ATM, online shipping mismatch, new beneficiary)
- Rules produce structured outputs (name, severity, detail)
- Final risk = combination of ML risk and highest rule severity with clear logic
"""

import datetime
from math import radians, sin, cos, asin, sqrt
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------------
# Helpers
# ----------------------------

def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Return distance in km between two lat/lon points."""
    # If any missing, return None
    if None in (lat1, lon1, lat2, lon2):
        return None
    # convert decimal degrees to radians
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    km = 6371 * c
    return km

# Map severity order
SEVERITY_ORDER = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

def escalate(a: str, b: str) -> str:
    """Return the higher severity between a and b."""
    return a if SEVERITY_ORDER[a] >= SEVERITY_ORDER[b] else b

# ----------------------------
# Load ML artifacts (cached)
# ----------------------------
@st.cache_resource
def load_artifacts():
    models_dir = Path("models")
    def _load(name: str):
        path = models_dir / name
        try:
            return joblib.load(path)
        except Exception as e:
            # bubble up with clear messaging
            st.error(f"Error loading model artifact: {name}")
            st.exception(e)
            raise
    supervised_pipeline = _load("supervised_lgbm_pipeline.joblib")
    iforest_pipeline = _load("iforest_pipeline.joblib")
    return supervised_pipeline, iforest_pipeline

supervised_pipeline, iforest_pipeline = load_artifacts()

# ----------------------------
# ML risk thresholds
# ----------------------------
FRAUD_MED = 0.00005
FRAUD_HIGH = 0.00023328
FRAUD_CRIT = 0.01732857

ANOM_MED = 0.04
ANOM_HIGH = 0.05
ANOM_CRIT = 0.08

def ml_risk_label(fraud_prob: float, anomaly_score: float) -> str:
    if fraud_prob >= FRAUD_CRIT or anomaly_score >= ANOM_CRIT:
        return "CRITICAL"
    elif fraud_prob >= FRAUD_HIGH or (fraud_prob >= FRAUD_MED and anomaly_score >= ANOM_HIGH):
        return "HIGH"
    elif fraud_prob >= FRAUD_MED or anomaly_score >= ANOM_MED:
        return "MEDIUM"
    else:
        return "LOW"

# ----------------------------
# Rule engine
# ----------------------------
def evaluate_rules(payload: Dict) -> Tuple[List[Dict], str]:
    """
    Evaluate deterministic rules on the payload.
    Returns: (list_of_triggered_rules, highest_severity)
    Each rule: {"name": str, "severity": "LOW|MEDIUM|HIGH|CRITICAL", "detail": str}
    """
    rules: List[Dict] = []

    # convenience getters
    amt = float(payload.get("Amount", 0.0) or 0.0)
    channel = str(payload.get("Channel", "")).lower()
    hour = int(payload.get("hour", 0))
    monthly_avg = float(payload.get("monthly_avg", 0.0) or 0.0)
    rolling_avg_7d = float(payload.get("rolling_avg_7d", 0.0) or 0.0)
    txns_1h = int(payload.get("txns_last_1h", 0) or 0)
    txns_24h = int(payload.get("txns_last_24h", 0) or 0)
    txns_7d = int(payload.get("txns_last_7d", 0) or 0)
    failed_logins = int(payload.get("failed_login_attempts", 0) or 0)
    new_benef = bool(payload.get("new_beneficiary", False))
    ip_country = str(payload.get("ip_country", "")).lower()
    declared_country = str(payload.get("declared_country", "")).lower()
    last_device = str(payload.get("device_last_seen", "")).lower()
    curr_device = str(payload.get("DeviceID", "")).lower()
    last_lat = payload.get("last_known_lat")
    last_lon = payload.get("last_known_lon")
    txn_lat = payload.get("txn_lat")
    txn_lon = payload.get("txn_lon")
    atm_distance_km = float(payload.get("atm_distance_km", 0.0) or 0.0)
    card_country = str(payload.get("card_country", "")).lower()
    cvv_provided = payload.get("cvv_provided", True)
    card_masked = payload.get("card_masked", "")
    shipping_addr = payload.get("shipping_address", "")
    billing_addr = payload.get("billing_address", "")
    beneficiaries_added_24h = int(payload.get("beneficiaries_added_24h", 0) or 0)
    suspicious_ip_flag = payload.get("suspicious_ip_flag", False)

    # helper to add rule
    def add_rule(name: str, sev: str, detail: str):
        rules.append({"name": name, "severity": sev, "detail": detail})

    # -------- CRITICAL rules --------
    # huge absolute amount (tunable per product/currency)
    ABSOLUTE_CRIT_AMOUNT = 10_000_000  # tune
    if amt >= ABSOLUTE_CRIT_AMOUNT:
        add_rule("Absolute very large amount", "CRITICAL",
                 f"Amount {amt} >= critical threshold {ABSOLUTE_CRIT_AMOUNT}.")

    # new device + new high-amount + new location (behavioural anomaly)
    impossible_travel_distance = None
    if last_lat is not None and last_lon is not None and txn_lat is not None and txn_lon is not None:
        impossible_travel_distance = haversine_km(last_lat, last_lon, txn_lat, txn_lon)
    device_new = (not last_device) or (last_device == "")
    location_changed = False
    if impossible_travel_distance is not None and impossible_travel_distance > 500:
        # e.g., >500km since last known location in short time is suspicious
        location_changed = True

    if device_new and location_changed and amt > 1000:
        add_rule("New device + Impossible travel + High amount", "CRITICAL",
                 f"Device unseen before and travel {impossible_travel_distance:.1f} km since last known location; amount {amt}.")

    # multiple beneficiaries added recently + fund out
    if beneficiaries_added_24h >= 3 and amt > 2000:
        add_rule("Multiple beneficiaries added recently + high transfer", "CRITICAL",
                 f"{beneficiaries_added_24h} beneficiaries added in last 24h and transfer amount {amt}.")

    # -------- HIGH rules --------
    # Velocity: many transactions in short window
    if txns_1h >= 10:
        add_rule("High velocity (1h)", "HIGH", f"{txns_1h} transactions in the last 1 hour.")
    if txns_24h >= 50:
        add_rule("Very high velocity (24h)", "HIGH", f"{txns_24h} transactions in the last 24 hours.")

    # IP country mismatch especially for high amount
    if ip_country and declared_country and ip_country != declared_country:
        sev = "HIGH" if amt > 2000 else "MEDIUM"
        add_rule("IP / Declared country mismatch", sev,
                 f"IP country '{ip_country}' differs from declared country '{declared_country}'.")

    # multiple failed logins
    if failed_logins >= 5:
        add_rule("Multiple failed login attempts", "HIGH", f"{failed_logins} failed auth attempts.")

    # new beneficiary + large transfer
    if new_benef and amt >= 1000:
        add_rule("New beneficiary + significant amount", "HIGH",
                 "Transfer to newly added beneficiary with amount above threshold.")

    # suspicious IP flag (from threat intel)
    if suspicious_ip_flag and amt > 500:
        add_rule("IP flagged by intel", "HIGH", "IP address is flagged as suspicious and amount is non-trivial.")

    # ATM distance large
    if channel == "atm" and atm_distance_km and atm_distance_km > 300:
        add_rule("ATM distance from last location", "HIGH", f"ATM is {atm_distance_km:.1f} km from last known location.")

    # card country mismatch cross-border
    if card_country and declared_country and card_country != declared_country:
        add_rule("Card country mismatch", "HIGH", f"Card country {card_country} != declared country {declared_country}.")

    # -------- MEDIUM rules --------
    # Spending spike vs monthly avg or 7d rolling avg
    if monthly_avg > 0 and amt >= 5 * monthly_avg and amt > 1000:
        add_rule("Large spike vs monthly avg", "HIGH",
                 f"Amount {amt} >=5x monthly average {monthly_avg:.2f}.")
    elif rolling_avg_7d > 0 and amt >= 3 * rolling_avg_7d and amt > 500:
        add_rule("Spike vs 7-day average", "MEDIUM",
                 f"Amount {amt} >=3x 7-day rolling avg {rolling_avg_7d:.2f}.")
    elif monthly_avg > 0 and amt >= 2 * monthly_avg and amt > 500:
        add_rule("Above monthly usual", "MEDIUM",
                 f"Amount {amt} >=2x monthly average {monthly_avg:.2f}.")

    # Moderate velocity
    if txns_1h >= 5:
        add_rule("Elevated velocity (1h)", "MEDIUM", f"{txns_1h} txns in last 1 hour.")
    if txns_24h >= 10 and txns_24h < 50:
        add_rule("Elevated velocity (24h)", "MEDIUM", f"{txns_24h} txns in last 24 hours.")

    # Time-of-day anomalies for low-activity customers
    if 0 <= hour <= 5 and monthly_avg < 2000 and amt > 100:
        add_rule("Late-night transaction for low-activity customer", "MEDIUM",
                 f"Transaction at hour {hour} for a low-activity customer; amount {amt}.")

    # Device mismatch vs last seen
    if last_device and curr_device and last_device != curr_device:
        add_rule("Device mismatch from last seen", "MEDIUM",
                 f"Device changed from '{last_device}' to '{curr_device}'.")

    # billing vs shipping mismatch for online orders
    if channel == "online" or channel == "online purchase":
        if shipping_addr and billing_addr and shipping_addr.strip().lower() != billing_addr.strip().lower():
            add_rule("Billing vs shipping address mismatch", "MEDIUM",
                     "Billing address differs from shipping address for e-commerce transaction.")

    # card fields missing when required
    if channel in ("credit card", "online", "online purchase"):
        if not cvv_provided:
            add_rule("Missing CVV for card transaction", "MEDIUM", "CVV not provided for card e-commerce transaction.")

    # new device but low amount (low risk but notable)
    if device_new and amt < 200:
        add_rule("New device (low amount)", "LOW", "Transaction from new device but low amount.")

    # beneficiaries created but not high amount
    if beneficiaries_added_24h > 0 and beneficiaries_added_24h < 3:
        add_rule("Beneficiaries recently added", "LOW",
                 f"{beneficiaries_added_24h} beneficiaries added in last 24h.")

    # suspicious but not decisive IP info
    if ip_country and ip_country in {"nigeria", "romania", "ukraine", "russia"}:
        add_rule("IP from higher-risk country", "MEDIUM",
                 f"IP country flagged as higher-risk: {ip_country} (contextual).")

    # default highest severity
    highest = "LOW"
    for r in rules:
        highest = escalate(highest, r["severity"])

    return rules, highest

# ----------------------------
# Combine ML + Rules into final decision
# ----------------------------
def combine_final_risk(ml_risk: str, rule_highest: str) -> str:
    """
    Combine ML risk and rule-derived highest severity.
    Priority: CRITICAL > HIGH > MEDIUM > LOW
    But allow rules to escalate ML and vice versa.
    """
    # escalate to the worst of the two
    return escalate(ml_risk, rule_highest)

# ----------------------------
# ML scoring wrapper
# ----------------------------
def score_transaction_ml(model_pipeline, iforest_pipeline, model_payload: Dict) -> Tuple[float, float, str]:
    # Prepare minimal df for the pipeline (adapt if pipeline expects more)
    model_df = pd.DataFrame([{
        "Amount": model_payload.get("Amount", 0.0),
        "TransactionType": model_payload.get("TransactionType", "PAYMENT"),
        "Location": model_payload.get("Location", "Unknown"),
        "DeviceID": model_payload.get("DeviceID", "Unknown"),
        "Channel": model_payload.get("Channel", "Other"),
        "hour": model_payload.get("hour", 0),
        "day_of_week": model_payload.get("day_of_week", 0),
        "month": model_payload.get("month", 0),
    }])
    try:
        fraud_prob = float(model_pipeline.predict_proba(model_df)[0, 1])
    except Exception as e:
        st.error("Supervised model scoring error - check pipeline input schema")
        st.exception(e)
        fraud_prob = 0.0
    try:
        raw = float(iforest_pipeline.decision_function(model_df)[0])
        anomaly_score = -raw
    except Exception as e:
        st.error("IsolationForest scoring error - check pipeline input schema")
        st.exception(e)
        anomaly_score = 0.0
    ml_label = ml_risk_label(fraud_prob, anomaly_score)
    return fraud_prob, anomaly_score, ml_label

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="AI Powered Real-Time Fraud Detection in Banking Transactions", page_icon="💳", layout="centered")
st.title("💳 AI Powered Real-Time Fraud Detection in Banking Transactions")
st.write("Select channel and fill required fields. Optional historical fields enable velocity & behavioural rules.")

st.markdown("---")
st.sidebar.header("Configuration / Notes")
st.sidebar.markdown("""
- Provide historical/telemetry inputs when available (monthly avg, last device, last location coords, counts).
- Velocity rules (1h / 24h / 7d) and behavioural anomalies (device churn, impossible travel) are supported.
- Tune thresholds for your product and currency.
""")

# Channel selector (single choice)
channel = st.selectbox("Transaction Channel", ["Choose...", "Bank", "Mobile App", "ATM", "Credit Card", "POS", "Online Purchase", "NetBanking"])

# Render channel-specific inputs only after selection
if channel and channel != "Choose...":
    st.markdown(f"### Inputs for channel: **{channel}**")

    # Common fields
    col1, col2 = st.columns(2)
    with col1:
        amount = st.number_input("Transaction amount", min_value=0.0, value=1200.0, step=10.0)
        txn_type = st.selectbox("Transaction type", ["PAYMENT", "TRANSFER", "DEBIT", "CREDIT", "CASH_OUT", "OTHER"])
        location = st.text_input("City / Region (declared location)", value="Karachi")
        declared_country = st.text_input("Declared Country", value="Pakistan")
    with col2:
        txn_date = st.date_input("Transaction date", value=datetime.date.today())
        txn_time = st.time_input("Transaction time", value=datetime.datetime.now().time())
        device = st.text_input("Device / OS", value="Android")
        device_last_seen = st.text_input("Last known device (optional)", value="Android")

    txn_dt = datetime.datetime.combine(txn_date, txn_time)
    hour = txn_dt.hour
    day_of_week = txn_dt.weekday()
    month = txn_dt.month

    # Telemetry & history (optional but recommended)
    st.markdown("#### Optional account telemetry / recent activity (helps rules)")
    col3, col4 = st.columns(2)
    with col3:
        monthly_avg = st.number_input("Customer monthly average spend", min_value=0.0, value=10000.0, step=100.0)
        rolling_avg_7d = st.number_input("7-day rolling average", min_value=0.0, value=3000.0, step=50.0)
        txns_last_1h = st.number_input("Transactions in last 1 hour", min_value=0, value=0, step=1)
        txns_last_24h = st.number_input("Transactions in last 24 hours", min_value=0, value=1, step=1)
    with col4:
        txns_last_7d = st.number_input("Transactions in last 7 days", min_value=0, value=7, step=1)
        beneficiaries_added_24h = st.number_input("Beneficiaries added in last 24h", min_value=0, value=0, step=1)
        failed_login_attempts = st.number_input("Failed login attempts (recent)", min_value=0, value=0, step=1)
        beneficiaries_added_24h = int(beneficiaries_added_24h)

    # IP / geo / coords
    st.markdown("#### IP & Geo (optional but highly recommended)")
    col5, col6 = st.columns(2)
    with col5:
        client_ip = st.text_input("Client IP (optional)", value="")
        ip_country = st.text_input("IP-derived country (optional)", value="")
        suspicious_ip_flag = st.checkbox("IP flagged by threat intel?", value=False)
    with col6:
        # last known coords (from prior session) and current txn coords (if available)
        last_known_lat = st.number_input("Last known latitude (optional)", format="%.6f", value=0.0)
        last_known_lon = st.number_input("Last known longitude (optional)", format="%.6f", value=0.0)
        txn_lat = st.number_input("Transaction latitude (optional)", format="%.6f", value=0.0)
        txn_lon = st.number_input("Transaction longitude (optional)", format="%.6f", value=0.0)

    # Channel-specific fields (only visible when that channel chosen)
    card_masked = ""
    card_country = ""
    cvv_provided = True
    atm_distance_km = 0.0
    shipping_address = ""
    billing_address = ""
    new_beneficiary = False
    beneficiaries_added_24h = int(beneficiaries_added_24h)
    card_used_online = False

    if channel == "Credit Card":
        st.markdown("**Credit Card details**")
        card_masked = st.text_input("Card number (masked, e.g. 4111****1111)", value="")
        card_country = st.text_input("Card issuing country (optional)", value="")
        cvv_provided = st.checkbox("CVV provided/verified?", value=True)
        card_used_online = st.checkbox("Card used for e-commerce?", value=False)

    elif channel == "Online Purchase":
        st.markdown("**Online purchase details**")
        merchant = st.text_input("Merchant name / ID", value="")
        shipping_address = st.text_input("Shipping address", value="")
        billing_address = st.text_input("Billing address", value=shipping_address)
        card_used_online = st.checkbox("Payment by card?", value=False)
        if card_used_online:
            cvv_provided = st.checkbox("CVV provided/verified?", value=True)
            card_masked = st.text_input("Card masked", value="")

    elif channel == "ATM":
        st.markdown("**ATM details**")
        atm_id = st.text_input("ATM ID / Terminal", value="")
        atm_distance_km = st.number_input("ATM distance from last known location (km)", min_value=0.0, value=0.0, step=1.0)

    elif channel == "Mobile App":
        st.markdown("**Mobile app details**")
        app_version = st.text_input("App version", value="1.0.0")
        device_fingerprint = st.text_input("Device fingerprint ID (optional)", value="")

    elif channel == "POS":
        st.markdown("**POS details**")
        pos_merchant_id = st.text_input("POS Merchant ID", value="")
        store_name = st.text_input("Store name", value="")

    elif channel == "Bank" or channel == "NetBanking":
        st.markdown("**Bank / NetBanking details**")
        beneficiary = st.text_input("Beneficiary (if transfer)", value="")
        new_beneficiary = st.checkbox("Is this a newly added beneficiary?", value=False)

    # submit
    submit = st.button("🚀 Run Fraud Check")

    # When clicked: build payload, run ML scoring, run rules, combine
    if submit:
        # Prepare payload for rules & ML
        payload = {
            "Amount": amount,
            "TransactionType": txn_type,
            "Location": location,
            "DeviceID": device,
            "device_last_seen": device_last_seen,
            "Channel": channel,
            "hour": hour,
            "day_of_week": day_of_week,
            "month": month,
            # historical telemetry
            "monthly_avg": monthly_avg,
            "rolling_avg_7d": rolling_avg_7d,
            "txns_last_1h": int(txns_last_1h),
            "txns_last_24h": int(txns_last_24h),
            "txns_last_7d": int(txns_last_7d),
            "beneficiaries_added_24h": beneficiaries_added_24h,
            "failed_login_attempts": failed_login_attempts,
            "beneficiaries_added_24h": beneficiaries_added_24h,
            # ip / geo
            "client_ip": client_ip,
            "ip_country": ip_country,
            "declared_country": declared_country,
            "suspicious_ip_flag": suspicious_ip_flag,
            "last_known_lat": last_known_lat if last_known_lat != 0.0 else None,
            "last_known_lon": last_known_lon if last_known_lon != 0.0 else None,
            "txn_lat": txn_lat if txn_lat != 0.0 else None,
            "txn_lon": txn_lon if txn_lon != 0.0 else None,
            # channel specifics
            "card_masked": card_masked,
            "card_country": card_country,
            "cvv_provided": cvv_provided,
            "atm_distance_km": atm_distance_km,
            "shipping_address": shipping_address,
            "billing_address": billing_address,
            "new_beneficiary": new_beneficiary,
            "beneficiaries_added_24h": beneficiaries_added_24h,
            "suspicious_ip_flag": suspicious_ip_flag,
            "DeviceID": device,
            "device_last_seen": device_last_seen,
            "card_used_online": card_used_online,
        }

        # Score ML
        with st.spinner("Scoring with ML models..."):
            fraud_prob, anomaly_score, ml_label = score_transaction_ml(supervised_pipeline, iforest_pipeline, payload)

        # Evaluate rules (velocity + behavioural anomalies included)
        rules_triggered, rules_highest = evaluate_rules(payload)

        # Combine final risk
        final_risk = combine_final_risk(ml_label, rules_highest)

        # Present results
        st.markdown("## 🔎 Results")
        # nice badge
        color_map = {"LOW": "#2e7d32", "MEDIUM": "#f9a825", "HIGH": "#f57c00", "CRITICAL": "#c62828"}
        badge_color = color_map.get(final_risk, "#607d8b")
        st.markdown(
            f"""<div style="padding:0.75rem 1rem;border-radius:0.5rem;background-color:{badge_color}22;border:1px solid {badge_color};">
                <strong style="color:{badge_color};font-size:1.1rem;">Final Risk Level: {final_risk}</strong>
            </div>""",
            unsafe_allow_html=True,
        )

        colA, colB = st.columns(2)
        with colA:
            st.metric("Fraud Probability (supervised)", f"{fraud_prob:.8f}")
            st.metric("ML Risk Label", ml_label)
        with colB:
            st.metric("Anomaly Score (IsolationForest)", f"{anomaly_score:.5f}")
            st.metric("Rules-derived highest severity", rules_highest)

        st.markdown("### ⚠ Triggered Rules")
        if rules_triggered:
            for r in rules_triggered:
                sev = r["severity"]
                emoji = "🔴" if sev in ("HIGH", "CRITICAL") else "🟠" if sev == "MEDIUM" else "🟢"
                st.write(f"{emoji} **{r['name']}** — *{r['severity']}*")
                st.caption(r["detail"])
        else:
            st.success("No deterministic rules triggered.")

        # Diagnostic payload for debugging
        st.markdown("### 📦 Payload (debug)")
        st.json(payload)

        st.markdown(
            """
            ### Notes & tuning
            - Velocity thresholds (1h / 24h / 7d) and absolute amount thresholds are examples — tune for your product.
            - Provide real telemetry (last coords, last device, txns counts) for accurate behavioural checks.
            - Consider logging triggered rules + ML outputs to a datastore for periodic threshold tuning and model retraining.
            """
        )

else:
    st.info("Choose a transaction channel to show channel-specific fields.")

