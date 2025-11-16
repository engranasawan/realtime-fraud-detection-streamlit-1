# app.py
"""
Real-Time Fraud Detection Prototype (ML + Rules) — Currency-adaptive
- Adds a currency selector *before* the channel selection (Option A).
- Thresholds adapt automatically to the selected currency.
- Supports 6 currencies: USD, EUR, GBP, PKR, AED, AUD (example rates).
- Comprehensive inline comments explain each rule, threshold, and logic.
- Keeps ML scoring using supervised_pipeline & iforest_pipeline (joblib).
- Rule engine uses velocity, behavioural, IP/location, device, and channel-specific rules.
"""

import datetime
from math import radians, sin, cos, asin, sqrt
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# ===========================
# 0) CURRENCY CONFIGURATION
# ===========================
# Currency selector MUST appear at the very first page (before channel).
# We store base thresholds in PKR and convert them to the selected currency
# using PKR-per-unit exchange rates defined below.
#
# NOTE: These exchange rates are example constants — update to real rates
# for production (via API or admin panel). All thresholds are computed as:
#   threshold_in_currency = base_threshold_pkr / PKR_PER_UNIT[selected_currency]
#
# Explanation:
# - PKR_PER_UNIT['USD'] = how many PKR equals 1 USD (e.g., 280 PKR = 1 USD)
# - So an absolute PKR threshold of 100,000 PKR -> in USD = 100000 / PKR_PER_UNIT['USD'].

# Example exchange rates (PKR per 1 unit of currency). Replace with live rates as needed.
PKR_PER_UNIT = {
    "USD": 280.0,  # 1 USD = 280 PKR (example)
    "EUR": 300.0,  # 1 EUR = 300 PKR (example)
    "GBP": 350.0,  # 1 GBP = 350 PKR (example)
    "PKR": 1.0,    # native currency
    "AED": 76.0,   # 1 AED = 76 PKR (example)
    "AUD": 180.0,  # 1 AUD = 180 PKR (example)
}

# Currency list for the dropdown
CURRENCY_OPTIONS = list(PKR_PER_UNIT.keys())

# ----------------------------
# Helpers / Geospatial
# ----------------------------
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Return distance in km between two lat/lon points."""
    if None in (lat1, lon1, lat2, lon2):
        return None
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    km = 6371 * c
    return km

# ----------------------------
# Severity ordering helpers
# ----------------------------
SEVERITY_ORDER = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

def escalate(a: str, b: str) -> str:
    """Return the higher severity between a and b."""
    return a if SEVERITY_ORDER[a] >= SEVERITY_ORDER[b] else b

# ----------------------------
# 1) Load ML artifacts (cached)
# ----------------------------
@st.cache_resource
def load_artifacts():
    models_dir = Path("models")
    def _load(name: str):
        path = models_dir / name
        try:
            return joblib.load(path)
        except Exception as e:
            st.error(f"Error loading model artifact: {name}")
            st.exception(e)
            raise
    supervised_pipeline = _load("supervised_lgbm_pipeline.joblib")
    iforest_pipeline = _load("iforest_pipeline.joblib")
    return supervised_pipeline, iforest_pipeline

supervised_pipeline, iforest_pipeline = load_artifacts()

# ----------------------------
# 2) ML risk thresholds (constants based on prior configuration)
# ----------------------------
# These thresholds are model-derived and remain the same numerically,
# they are not currency-sensitive because they are probabilities/anomaly scores.
FRAUD_MED = 0.00005
FRAUD_HIGH = 0.00023328
FRAUD_CRIT = 0.01732857

ANOM_MED = 0.04
ANOM_HIGH = 0.05
ANOM_CRIT = 0.08

def ml_risk_label(fraud_prob: float, anomaly_score: float) -> str:
    """Return ML label using fixed thresholds (probability and anomaly score)."""
    if fraud_prob >= FRAUD_CRIT or anomaly_score >= ANOM_CRIT:
        return "CRITICAL"
    elif fraud_prob >= FRAUD_HIGH or (fraud_prob >= FRAUD_MED and anomaly_score >= ANOM_HIGH):
        return "HIGH"
    elif fraud_prob >= FRAUD_MED or anomaly_score >= ANOM_MED:
        return "MEDIUM"
    else:
        return "LOW"

# ----------------------------
# 3) Currency utilities (adaptive thresholds)
# ----------------------------
def pkr_to_currency(amount_in_pkr: float, currency: str) -> float:
    """
    Convert a PKR-denominated threshold to the selected currency unit.
    Example: If base=100000 PKR and currency='USD' (PKR_PER_UNIT['USD']=280),
    result = 100000 / 280 = 357.14 USD (threshold in USD).
    """
    if currency not in PKR_PER_UNIT:
        # fallback to PKR if unknown
        return amount_in_pkr
    pkr_per_unit = PKR_PER_UNIT[currency]
    adapted = amount_in_pkr / pkr_per_unit
    return adapted

# Define base thresholds in PKR (these are our 'canonical' numbers).
# We will convert them to the selected currency at runtime.
BASE_THRESHOLDS_PKR = {
    # absolute critical threshold: extremely large single transaction (CRITICAL)
    "absolute_crit_amount": 10_000_000,  # PKR 10 million by default
    # high thresholds used in many rules
    "high_amount_threshold": 2_000_000,  # PKR 2 million (example)
    "medium_amount_threshold": 100_000,  # PKR 100k
    # ATM-specific thresholds
    "atm_high_withdrawal": 300_000,  # PKR 300k
    # velocity-based thresholds (counts are currency-agnostic)
    "card_test_small_amount_pkr": 200,  # micro-test amount in PKR
}

# ----------------------------
# 4) Rule engine
# ----------------------------
def evaluate_rules(payload: Dict, currency: str) -> Tuple[List[Dict], str]:
    """
    Evaluate deterministic rules on the payload.
    Returns: (list_of_triggered_rules, highest_severity)
    Each rule: {"name": str, "severity": "LOW|MEDIUM|HIGH|CRITICAL", "detail": str}
    All monetary thresholds are adapted to the selected currency.
    """
    # Convert canonical PKR thresholds into selected currency units
    ABSOLUTE_CRIT_AMOUNT = pkr_to_currency(BASE_THRESHOLDS_PKR["absolute_crit_amount"], currency)
    HIGH_AMOUNT_THRESHOLD = pkr_to_currency(BASE_THRESHOLDS_PKR["high_amount_threshold"], currency)
    MEDIUM_AMOUNT_THRESHOLD = pkr_to_currency(BASE_THRESHOLDS_PKR["medium_amount_threshold"], currency)
    ATM_HIGH_WITHDRAWAL = pkr_to_currency(BASE_THRESHOLDS_PKR["atm_high_withdrawal"], currency)
    # The "micro test" threshold for card testing: show in selected currency
    CARD_TEST_SMALL_AMOUNT = pkr_to_currency(BASE_THRESHOLDS_PKR["card_test_small_amount_pkr"], currency)

    rules: List[Dict] = []

    # convenience getters & safe parsing
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
    beneficiaries_added_24h = int(payload.get("beneficiaries_added_24h", 0) or 0)

    # Helper to attach a rule
    def add_rule(name: str, sev: str, detail: str):
        rules.append({"name": name, "severity": sev, "detail": detail})

    # ------------------------------
    # CRITICAL rules (immediate escalation)
    # ------------------------------
    # CRIT-1: Absolute very large amount — regardless of history or channel.
    # Rationale: extremely large single transactions are immediate high risk.
    if amt >= ABSOLUTE_CRIT_AMOUNT:
        add_rule(
            "Absolute very large amount",
            "CRITICAL",
            f"Transaction amount {amt:.2f} {currency} >= critical threshold {ABSOLUTE_CRIT_AMOUNT:.2f} {currency}."
        )

    # CRIT-2: New device + impossible travel + high amount
    # Rationale: device not seen before + huge distance since last known location => classic ATO/clone signal.
    impossible_travel_distance = None
    if last_lat is not None and last_lon is not None and txn_lat is not None and txn_lon is not None:
        impossible_travel_distance = haversine_km(last_lat, last_lon, txn_lat, txn_lon)

    device_new = (not last_device) or (last_device == "")
    location_changed = False
    if impossible_travel_distance is not None and impossible_travel_distance > 500:
        # >500 km jump is considered suspicious rapid travel in our system (tunable)
        location_changed = True

    if device_new and location_changed and amt > MEDIUM_AMOUNT_THRESHOLD:
        add_rule(
            "New device + Impossible travel + High amount",
            "CRITICAL",
            f"Device unseen before and travel {impossible_travel_distance:.1f} km since last known location; amount {amt:.2f} {currency}."
        )

    # CRIT-3: Multiple beneficiaries added recently + fund out
    if beneficiaries_added_24h >= 3 and amt > HIGH_AMOUNT_THRESHOLD:
        add_rule(
            "Multiple beneficiaries added recently + high transfer",
            "CRITICAL",
            f"{beneficiaries_added_24h} beneficiaries added in last 24h and transfer amount {amt:.2f} {currency}."
        )

    # ------------------------------
    # HIGH rules (strong indicators)
    # ------------------------------
    # Velocity-based high rules: many transactions in short windows
    if txns_1h >= 10:
        add_rule("High velocity (1h)", "HIGH", f"{txns_1h} transactions in the last 1 hour.")
    if txns_24h >= 50:
        add_rule("Very high velocity (24h)", "HIGH", f"{txns_24h} transactions in the last 24 hours.")

    # IP vs declared country mismatch (higher severity if large amount)
    if ip_country and declared_country and ip_country != declared_country:
        sev = "HIGH" if amt > HIGH_AMOUNT_THRESHOLD else "MEDIUM"
        add_rule("IP / Declared country mismatch", sev,
                 f"IP country '{ip_country}' differs from declared country '{declared_country}'.")

    # Excessive failed logins → likely brute force / credential stuffing
    if failed_logins >= 5:
        add_rule("Multiple failed login attempts", "HIGH", f"{failed_logins} failed authentication attempts recently.")

    # New beneficiary + large transfer
    if new_benef and amt >= MEDIUM_AMOUNT_THRESHOLD:
        add_rule("New beneficiary + significant amount", "HIGH",
                 "Transfer to newly added beneficiary with amount above threshold.")

    # Suspicious IP flagged by intel + non-trivial amount
    if suspicious_ip_flag and amt > MEDIUM_AMOUNT_THRESHOLD / 4:
        add_rule("IP flagged by threat intelligence", "HIGH", "IP address is flagged as suspicious and amount is non-trivial.")

    # ATM distance large for ATM channel (tunable per currency)
    if channel == "atm" and atm_distance_km and atm_distance_km > 300:
        add_rule("ATM distance from last location", "HIGH", f"ATM is {atm_distance_km:.1f} km from last known location.")

    # Card country mismatch (attempt cross-border use for domestic card)
    if card_country and declared_country and card_country != declared_country and amt > MEDIUM_AMOUNT_THRESHOLD:
        add_rule("Card country mismatch (cross-border)", "HIGH",
                 f"Card country {card_country} != declared country {declared_country}.")

    # ------------------------------
    # MEDIUM rules (suspicious but contextual)
    # ------------------------------
    # Spending spike vs monthly avg or 7d rolling avg
    if monthly_avg > 0 and amt >= 5 * monthly_avg and amt > MEDIUM_AMOUNT_THRESHOLD:
        # This is severe enough to upgrade to HIGH because it's a strong spike
        add_rule("Large spike vs monthly average", "HIGH",
                 f"Amount {amt:.2f} {currency} >=5x monthly average {monthly_avg:.2f} {currency}.")
    elif rolling_avg_7d > 0 and amt >= 3 * rolling_avg_7d and amt > (MEDIUM_AMOUNT_THRESHOLD / 2):
        add_rule("Spike vs 7-day rolling average", "MEDIUM",
                 f"Amount {amt:.2f} {currency} >=3x 7-day rolling avg {rolling_avg_7d:.2f} {currency}.")
    elif monthly_avg > 0 and amt >= 2 * monthly_avg and amt > (MEDIUM_AMOUNT_THRESHOLD / 2):
        add_rule("Above monthly usual", "MEDIUM",
                 f"Amount {amt:.2f} {currency} >=2x monthly average {monthly_avg:.2f} {currency}.")

    # Moderate velocity
    if txns_1h >= 5:
        add_rule("Elevated velocity (1h)", "MEDIUM", f"{txns_1h} transactions in the last 1 hour.")
    if 10 <= txns_24h < 50:
        add_rule("Elevated velocity (24h)", "MEDIUM", f"{txns_24h} transactions in last 24 hours.")

    # Time-of-day anomalies for low-activity customers
    if 0 <= hour <= 5 and monthly_avg < (MEDIUM_AMOUNT_THRESHOLD * 2) and amt > (MEDIUM_AMOUNT_THRESHOLD / 10):
        add_rule("Late-night transaction for low-activity customer", "MEDIUM",
                 f"Transaction at hour {hour} for a low-activity customer; amount {amt:.2f} {currency}.")

    # Device mismatch vs last seen
    if last_device and curr_device and last_device != curr_device:
        add_rule("Device mismatch from last seen", "MEDIUM",
                 f"Device changed from '{last_device}' to '{curr_device}'.")

    # Billing vs shipping mismatch for online orders
    if channel in ("online", "online purchase"):
        if shipping_addr and billing_addr and shipping_addr.strip().lower() != billing_addr.strip().lower():
            add_rule("Billing vs shipping address mismatch", "MEDIUM",
                     "Billing address differs from shipping address for e-commerce transaction.")

    # Missing card verification for card transactions
    if channel in ("credit card", "online", "online purchase"):
        if not cvv_provided:
            add_rule("Missing CVV for card transaction", "MEDIUM", "CVV not provided for card e-commerce transaction.")

    # Low-dollar new device (low severity)
    if device_new and amt < (MEDIUM_AMOUNT_THRESHOLD / 10):
        add_rule("New device (low amount)", "LOW", "Transaction from new device but low amount.")

    # beneficiaries added recently but not high amount
    if 0 < beneficiaries_added_24h < 3:
        add_rule("Beneficiaries recently added", "LOW",
                 f"{beneficiaries_added_24h} beneficiaries added in last 24h.")

    # IP from higher-risk country (contextual)
    if ip_country and ip_country in {"nigeria", "romania", "ukraine", "russia"}:
        add_rule("IP from higher-risk country", "MEDIUM",
                 f"IP country flagged as higher-risk: {ip_country} (contextual).")

    # ------------------------------
    # Channel-specific micro rules (examples)
    # ------------------------------
    # Credit card micro-testing: many small transactions to different merchants
    # Count check is expected to be computed externally and passed in payload as 'card_small_attempts_in_5min'
    card_small_attempts = int(payload.get("card_small_attempts_in_5min", 0) or 0)
    if card_small_attempts >= 6 and CARD_TEST_SMALL_AMOUNT > 0:
        # Elevated severity because this is a classic card-testing pattern
        add_rule("Card testing / micro-charges detected", "HIGH",
                 f"{card_small_attempts} small attempts within short timeframe; threshold for micro amount ~{CARD_TEST_SMALL_AMOUNT:.2f} {currency}.")

    # ATM rules
    if channel == "atm" and amt >= ATM_HIGH_WITHDRAWAL:
        add_rule("Large ATM withdrawal", "HIGH", f"ATM withdrawal {amt:.2f} {currency} >= threshold {ATM_HIGH_WITHDRAWAL:.2f} {currency}.")

    # POS: many repeated transactions at same POS within a short window (payload expects pos_repeat_count)
    pos_repeat_count = int(payload.get("pos_repeat_count", 0) or 0)
    if pos_repeat_count >= 10:
        add_rule("POS repeat transactions (possible merchant abuse)", "HIGH",
                 f"{pos_repeat_count} rapid transactions at same POS terminal.")

    # NetBanking: beneficiary added recently and immediate transfer
    if channel in ("netbanking", "bank"):
        beneficiary_added_minutes = int(payload.get("beneficiary_added_minutes", 9999) or 9999)
        if beneficiary_added_minutes < 10 and amt >= MEDIUM_AMOUNT_THRESHOLD:
            add_rule("Immediate transfer to newly added beneficiary", "HIGH",
                     f"Beneficiary added {beneficiary_added_minutes} minutes ago and transfer amount {amt:.2f} {currency}.")

    # ------------------------------
    # compute aggregate highest severity
    # ------------------------------
    highest = "LOW"
    for r in rules:
        highest = escalate(highest, r["severity"])

    return rules, highest

# ----------------------------
# 5) Combine ML + Rules into final decision
# ----------------------------
def combine_final_risk(ml_risk: str, rule_highest: str) -> str:
    """
    Combine ML risk and rule-derived highest severity.
    The final label is the escalation (worst) of the two.
    """
    return escalate(ml_risk, rule_highest)

# ----------------------------
# 6) ML scoring wrapper
# ----------------------------
def score_transaction_ml(model_pipeline, iforest_pipeline, model_payload: Dict) -> Tuple[float, float, str]:
    """
    Score transaction with supervised and unsupervised models.
    The model thresholds (probabilities & anomaly score) are currency-agnostic.
    """
    # Prepare minimal df for pipeline (adjust if your pipeline expects more fields)
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
        anomaly_score = -raw  # invert (higher => more anomalous)
    except Exception as e:
        st.error("IsolationForest scoring error - check pipeline input schema")
        st.exception(e)
        anomaly_score = 0.0
    ml_label = ml_risk_label(fraud_prob, anomaly_score)
    return fraud_prob, anomaly_score, ml_label

# ----------------------------
# 7) Streamlit UI
# ----------------------------
st.set_page_config(page_title="AI Powered Real-Time Fraud Detection (Currency-adaptive)", page_icon="💳", layout="centered")
st.title("💳 AI Powered Real-Time Fraud Detection in Banking Transactions")
st.write("Choose currency first — thresholds adapt to the selected currency. Then select the channel and fill channel-specific fields.")

st.markdown("---")
st.sidebar.header("Configuration / Notes")
st.sidebar.markdown("""
- Select currency on the top of the page. Thresholds are converted from PKR to the chosen currency.
- Exchange rates shown in code are examples. Replace with live rates for production.
- Provide telemetry (last device, last coords, txns counts) when possible — this enables velocity & behavioural checks.
""")

# === Currency selector (very first page element) ===
st.markdown("### Global settings")
currency = st.selectbox("Select currency (affects thresholds)", CURRENCY_OPTIONS, index=CURRENCY_OPTIONS.index("PKR"))

# Show the approximate exchange rate used (transparency)
st.caption(f"Using example rate: 1 {currency} = {PKR_PER_UNIT.get(currency):,.2f} PKR. Replace with live rates in production.")

st.markdown("---")

# Channel selection is shown AFTER currency selection
channel = st.selectbox("Transaction Channel", ["Choose...", "Bank", "Mobile App", "ATM", "Credit Card", "POS", "Online Purchase", "NetBanking"])

if channel and channel != "Choose...":
    st.markdown(f"### Inputs for channel: **{channel}** (thresholds in {currency})")

    # --- Common fields ---
    col1, col2 = st.columns(2)
    with col1:
        # Display amount in selected currency
        amount = st.number_input(f"Transaction amount ({currency})", min_value=0.0, value=1200.0, step=10.0)
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

    # --- Telemetry & history (optional) ---
    st.markdown("#### Optional account telemetry / recent activity (helps rules)")
    col3, col4 = st.columns(2)
    with col3:
        monthly_avg = st.number_input(f"Customer monthly average spend ({currency})", min_value=0.0, value=10000.0, step=100.0)
        rolling_avg_7d = st.number_input(f"7-day rolling average ({currency})", min_value=0.0, value=3000.0, step=50.0)
        txns_last_1h = st.number_input("Transactions in last 1 hour", min_value=0, value=0, step=1)
        txns_last_24h = st.number_input("Transactions in last 24 hours", min_value=0, value=1, step=1)
    with col4:
        txns_last_7d = st.number_input("Transactions in last 7 days", min_value=0, value=7, step=1)
        beneficiaries_added_24h = st.number_input("Beneficiaries added in last 24h", min_value=0, value=0, step=1)
        failed_login_attempts = st.number_input("Failed login attempts (recent)", min_value=0, value=0, step=1)
        beneficiaries_added_24h = int(beneficiaries_added_24h)

    # --- IP / geo / coords ---
    st.markdown("#### IP & Geo (optional but highly recommended)")
    col5, col6 = st.columns(2)
    with col5:
        client_ip = st.text_input("Client IP (optional)", value="")
        ip_country = st.text_input("IP-derived country (optional)", value="")
        suspicious_ip_flag = st.checkbox("IP flagged by threat intel?", value=False)
    with col6:
        last_known_lat = st.number_input("Last known latitude (optional)", format="%.6f", value=0.0)
        last_known_lon = st.number_input("Last known longitude (optional)", format="%.6f", value=0.0)
        txn_lat = st.number_input("Transaction latitude (optional)", format="%.6f", value=0.0)
        txn_lon = st.number_input("Transaction longitude (optional)", format="%.6f", value=0.0)

    # --- Channel-specific fields (only displayed for selected channel) ---
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
        beneficiary_added_minutes = st.number_input("Minutes since beneficiary was added (if known)", min_value=0, value=9999, step=1)

    # === Submit button ===
    submit = st.button("🚀 Run Fraud Check")

    if submit:
        # Build the payload for ML and rules. All monetary values remain in the selected currency.
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
            # telemetry
            "monthly_avg": monthly_avg,
            "rolling_avg_7d": rolling_avg_7d,
            "txns_last_1h": int(txns_last_1h),
            "txns_last_24h": int(txns_last_24h),
            "txns_last_7d": int(txns_last_7d),
            "beneficiaries_added_24h": beneficiaries_added_24h,
            "beneficiary_added_minutes": int(beneficiary_added_minutes) if channel in ("Bank", "NetBanking") else 9999,
            "failed_login_attempts": failed_login_attempts,
            # ip / geo
            "client_ip": client_ip,
            "ip_country": ip_country,
            "declared_country": declared_country,
            "suspicious_ip_flag": suspicious_ip_flag,
            "last_known_lat": last_known_lat if last_known_lat != 0.0 else None,
            "last_known_lon": last_known_lon if last_known_lon != 0.0 else None,
            "txn_lat": txn_lat if txn_lat != 0.0 else None,
            "txn_lon": txn_lon if txn_lon != 0.0 else None,
            # channel-specific
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
            # additional micro counts (optional inputs, default 0)
            "card_small_attempts_in_5min": int(st.session_state.get("card_small_attempts_in_5min", 0) if "card_small_attempts_in_5min" in st.session_state else 0),
            "pos_repeat_count": int(st.session_state.get("pos_repeat_count", 0) if "pos_repeat_count" in st.session_state else 0),
            # currency selected so rule engine can present thresholds in the same currency in messages
            "selected_currency": currency,
        }

        # Score with ML (currency-agnostic models expect numeric amounts; if model was trained in PKR
        # you'll want to convert amounts back to PKR before passing to model. For now we assume
        # the models accept the amount in the same currency the pipeline expects.)
        # If your model was trained in PKR, convert: payload_for_model_amount = amount * PKR_PER_UNIT[currency]
        model_payload = payload.copy()
        # Example: if supervised_pipeline was trained on PKR amounts convert back to PKR:
        # model_payload["Amount"] = amount * PKR_PER_UNIT[currency]
        # For now we keep the passed amount as-is; adjust according to how your training data was scaled.
        with st.spinner("Scoring with ML models..."):
            fraud_prob, anomaly_score, ml_label = score_transaction_ml(supervised_pipeline, iforest_pipeline, model_payload)

        # Evaluate rules with currency-aware thresholds
        rules_triggered, rules_highest = evaluate_rules(payload, currency)

        # Combine ML + rules
        final_risk = combine_final_risk(ml_label, rules_highest)

        # Present results and detailed rule explanations
        st.markdown("## 🔎 Results")
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
            st.metric(f"Fraud Probability (supervised)", f"{fraud_prob:.8f}")
            st.metric("ML Risk Label", ml_label)
        with colB:
            st.metric("Anomaly Score (IsolationForest)", f"{anomaly_score:.5f}")
            st.metric("Rules-derived highest severity", rules_highest)

        st.markdown("### ⚠ Triggered Rules (detailed)")
        if rules_triggered:
            for r in rules_triggered:
                sev = r["severity"]
                emoji = "🔴" if sev in ("HIGH", "CRITICAL") else "🟠" if sev == "MEDIUM" else "🟢"
                st.write(f"{emoji} **{r['name']}** — *{r['severity']}*")
                st.caption(r["detail"])
        else:
            st.success("No deterministic rules triggered.")

        # Debug payload
        st.markdown("### 📦 Payload (debug)")
        st.json(payload)

        st.markdown(
            """
            ### Notes & tuning
            - Currency thresholds are converted from PKR to the selected currency using the `PKR_PER_UNIT` table.
            - If your ML model was trained using PKR amounts, convert the input `Amount` back to PKR (multiply by PKR_PER_UNIT[currency]) before scoring.
            - Replace the hard-coded exchange rates with live rates for production.
            - Many thresholds (distance km, counts, multipliers) are intentionally conservative — tune them with historical data.
            """
        )
else:
    st.info("Select a currency at the top, then choose a transaction channel to show channel-specific fields.")
