# app.py
# AI Powered Real-Time Fraud Detection
# - UI aligned by channel & transaction type
# - Home city/country vs transaction city/country rules
# - Time-of-day + amount rules
# - ML & Rules justification block
# - Response time per transaction
# - Example “good” & “fraud” transactions per channel
# - Hooks for model performance metrics
# - Inline comments for KT

import datetime
import time  # used for response-time measurement
from math import radians, sin, cos, asin, sqrt
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd
import streamlit as st

# -------------------------
# CONFIGURATION
# -------------------------
INR_PER_UNIT = {
    "INR": 1.0,
    "USD": 83.2,
    "EUR": 90.5,
    "GBP": 105.3,
    "AED": 22.7,
    "AUD": 61.0,
    "SGD": 61.5,
}
CURRENCIES = list(INR_PER_UNIT.keys())
DEFAULT_CURRENCY = "INR"

BASE_THRESHOLDS_INR = {
    "absolute_crit_amount": 10_000_000,
    "high_amount_threshold": 2_000_000,
    "medium_amount_threshold": 100_000,
    "atm_high_withdrawal": 300_000,
    "card_test_small_amount_inr": 200,
}

# Thresholds used by ML-risk labelling
FRAUD_MED = 0.00005
FRAUD_HIGH = 0.00023328
FRAUD_CRIT = 0.01732857

ANOM_MED = 0.04
ANOM_HIGH = 0.05
ANOM_CRIT = 0.08

SEVERITY_ORDER = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

CHANNEL_TXN_TYPES = {
    "atm": ["CASH_WITHDRAWAL", "TRANSFER"],
    "credit card": ["PAYMENT", "REFUND"],
    "mobile app": ["PAYMENT", "TRANSFER", "BILL_PAY"],
    "pos": ["PAYMENT"],
    "online purchase": ["PAYMENT"],
    "bank": ["DEPOSIT", "TRANSFER", "WITHDRAWAL"],
    "netbanking": ["TRANSFER", "BILL_PAY", "PAYMENT"],
}

# Placeholder model performance metrics
# Replace Nones with actual values from your training/evaluation reports.
MODEL_PERFORMANCE = {
    "overall": {
        "roc_auc": None,
        "accuracy": None,
        "precision": None,
        "recall": None,
    },
    "channels": {
        # keys must be lower-case channel names
        "atm": {"accuracy": None, "precision": None, "recall": None},
        "credit card": {"accuracy": None, "precision": None, "recall": None},
        "mobile app": {"accuracy": None, "precision": None, "recall": None},
        "pos": {"accuracy": None, "precision": None, "recall": None},
        "online purchase": {"accuracy": None, "precision": None, "recall": None},
        "bank": {"accuracy": None, "precision": None, "recall": None},
        "netbanking": {"accuracy": None, "precision": None, "recall": None},
    },
}


def build_example_transactions() -> pd.DataFrame:
    """
    Build a small synthetic table of example transactions per channel.
    These are illustrative only – not from the real model.
    """
    rows = []
    for channel_name, txn_types in CHANNEL_TXN_TYPES.items():
        # 5 good + 5 fraud examples per channel
        for i in range(5):
            rows.append(
                {
                    "channel": channel_name,
                    "example_type": "GOOD",
                    "transaction_type": txn_types[i % len(txn_types)],
                    "amount_in_inr": 5_000 + i * 2_000,
                    "fraud_confidence_ml_pct": 0.5 + i * 0.3,  # illustrative
                    "anomaly_score_ml_pct": 0.2 + i * 0.2,
                    "rules_risk": "LOW",
                    "final_risk": "LOW",
                }
            )
        for i in range(5):
            rows.append(
                {
                    "channel": channel_name,
                    "example_type": "FRAUD",
                    "transaction_type": txn_types[i % len(txn_types)],
                    "amount_in_inr": 200_000 + i * 500_000,
                    "fraud_confidence_ml_pct": 60 + i * 8,
                    "anomaly_score_ml_pct": 40 + i * 5,
                    "rules_risk": "HIGH" if i < 3 else "CRITICAL",
                    "final_risk": "HIGH" if i < 3 else "CRITICAL",
                }
            )
    df = pd.DataFrame(rows)
    return df


EXAMPLE_TXNS_DF = build_example_transactions()

# -------------------------
# HELPERS
# -------------------------


def inr_to_currency(amount_in_inr: float, currency: str) -> float:
    if currency not in INR_PER_UNIT or INR_PER_UNIT[currency] == 0:
        return amount_in_inr
    return amount_in_inr / INR_PER_UNIT[currency]


def normalize_score(x: float, min_val: float = 0.0, max_val: float = 0.02) -> float:
    """
    Normalize an ML score into a 0–100 range for business interpretability.

    For fraud probability:
      - We assume most interesting values are in [0, 0.02] (0%–2%),
      - 0 => 0, 0.02 => 100.
    For anomaly score you can pass a larger max_val, e.g. 0.10.
    """
    if x is None:
        return 0.0
    try:
        val = float(x)
    except Exception:
        return 0.0
    # Clamp within expected range
    if val < min_val:
        val = min_val
    if val > max_val:
        val = max_val
    if max_val == min_val:
        return 0.0
    # Map to 0-100
    return (val - min_val) / (max_val - min_val) * 100.0


def haversine_km(lat1, lon1, lat2, lon2) -> float:
    """Geo distance used for impossible-travel rules."""
    if None in (lat1, lon1, lat2, lon2):
        return None
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return 6371 * c


def escalate(a: str, b: str) -> str:
    """Return the higher risk between two severities."""
    return a if SEVERITY_ORDER[a] >= SEVERITY_ORDER[b] else b


# -------------------------
# ML MODEL LOADING
# -------------------------
@st.cache_resource
def load_models():
    models_dir = Path("models")

    def _load(name: str):
        path = models_dir / name
        try:
            return joblib.load(path)
        except Exception as e:
            st.error(f"Error loading model artifact: {name}")
            st.exception(e)
            raise

    supervised = None
    iforest = None
    try:
        supervised = _load("supervised_lgbm_pipeline.joblib")
    except Exception:
        supervised = None
    try:
        iforest = _load("iforest_pipeline.joblib")
    except Exception:
        iforest = None
    return supervised, iforest


supervised_pipeline, iforest_pipeline = load_models()

# -------------------------
# ML SCORING WRAPPER
# -------------------------


def ml_risk_label(fraud_prob: float, anomaly_score: float) -> str:
    """Map model probabilities into LOW / MEDIUM / HIGH / CRITICAL."""
    if fraud_prob >= FRAUD_CRIT or anomaly_score >= ANOM_CRIT:
        return "CRITICAL"
    if fraud_prob >= FRAUD_HIGH or (fraud_prob >= FRAUD_MED and anomaly_score >= ANOM_HIGH):
        return "HIGH"
    if fraud_prob >= FRAUD_MED or anomaly_score >= ANOM_MED:
        return "MEDIUM"
    return "LOW"


def score_transaction_ml(
    model_pipeline,
    iforest_pipeline,
    model_payload: Dict,
    convert_to_inr: bool = False,
    currency: str = "INR",
) -> Tuple[float, float, str]:
    """
    Core ML scoring wrapper used by the UI and the API.
    Returns (fraud_probability, anomaly_score, ml_risk_label).

    fraud_probability: output of supervised model (0–1).
    anomaly_score: transformed IsolationForest decision score (0–1-ish).
    """
    amt_for_model = model_payload.get("Amount", 0.0)
    if convert_to_inr:
        amt_for_model = amt_for_model * INR_PER_UNIT.get(currency, 1.0)

    model_df = pd.DataFrame(
        [
            {
                "Amount": amt_for_model,
                "TransactionType": model_payload.get("TransactionType", "PAYMENT"),
                "Location": model_payload.get("txn_city", "Unknown"),
                "DeviceID": model_payload.get("DeviceID", "Unknown"),
                "Channel": model_payload.get("Channel", "Other"),
                "hour": model_payload.get("hour", 0),
                "day_of_week": model_payload.get("day_of_week", 0),
                "month": model_payload.get("month", 0),
            }
        ]
    )

    fraud_prob = 0.0
    anomaly_score = 0.0
    try:
        if model_pipeline is not None:
            fraud_prob = float(model_pipeline.predict_proba(model_df)[0, 1])
    except Exception as e:
        st.error("Supervised model scoring error - check pipeline input schema")
        st.exception(e)
    try:
        if iforest_pipeline is not None:
            raw = float(iforest_pipeline.decision_function(model_df)[0])
            anomaly_score = -raw
    except Exception as e:
        st.error("IsolationForest scoring error - check pipeline input schema")
        st.exception(e)

    label = ml_risk_label(fraud_prob, anomaly_score)
    return fraud_prob, anomaly_score, label


# -------------------------
# RULE ENGINE
# -------------------------

def evaluate_rules(payload: Dict, currency: str) -> Tuple[List[Dict], str]:
    """
    Deterministic rule engine.
    Returns a list of triggered rules and the highest severity across rules.
    """
    ABS_CRIT = inr_to_currency(BASE_THRESHOLDS_INR["absolute_crit_amount"], currency)
    HIGH_AMT = inr_to_currency(BASE_THRESHOLDS_INR["high_amount_threshold"], currency)
    MED_AMT = inr_to_currency(BASE_THRESHOLDS_INR["medium_amount_threshold"], currency)
    ATM_HIGH = inr_to_currency(BASE_THRESHOLDS_INR["atm_high_withdrawal"], currency)
    CARD_TEST_SMALL = inr_to_currency(BASE_THRESHOLDS_INR["card_test_small_amount_inr"], currency)

    rules: List[Dict] = []

    amt = float(payload.get("Amount", 0.0) or 0.0)
    channel = str(payload.get("Channel", "")).lower()
    hour = int(payload.get("hour", 0) or 0)
    monthly_avg = float(payload.get("monthly_avg", 0.0) or 0.0)
    rolling_avg_7d = float(payload.get("rolling_avg_7d", 0.0) or 0.0)
    txns_1h = int(payload.get("txns_last_1h", 0) or 0)
    txns_24h = int(payload.get("txns_last_24h", 0) or 0)
    txns_7d = int(payload.get("txns_last_7d", 0) or 0)
    failed_logins = int(payload.get("failed_login_attempts", 0) or 0)
    new_benef = bool(payload.get("new_beneficiary", False))

    ip_country = str(payload.get("ip_country", "") or "").lower()
    declared_country = str(payload.get("declared_country", "") or "").lower()

    # Home vs transaction location
    home_city = str(payload.get("home_city", "") or "").lower()
    home_country = str(payload.get("home_country", "") or "").lower()
    txn_city = str(payload.get("txn_city", "") or "").lower()
    txn_country = str(payload.get("txn_country", "") or "").lower()

    last_device = str(payload.get("device_last_seen", "") or "").lower()
    curr_device = str(payload.get("DeviceID", "") or "").lower()

    last_lat = payload.get("last_known_lat")
    last_lon = payload.get("last_known_lon")
    txn_lat = payload.get("txn_lat")
    txn_lon = payload.get("txn_lon")
    atm_distance_km = float(payload.get("atm_distance_km", 0.0) or 0.0)

    card_country = str(payload.get("card_country", "") or "").lower()
    cvv_provided = payload.get("cvv_provided", True)
    shipping_addr = payload.get("shipping_address", "")
    billing_addr = payload.get("billing_address", "")
    beneficiaries_added_24h = int(payload.get("beneficiaries_added_24h", 0) or 0)
    suspicious_ip_flag = payload.get("suspicious_ip_flag", False)
    card_small_attempts = int(payload.get("card_small_attempts_in_5min", 0) or 0)
    pos_repeat_count = int(payload.get("pos_repeat_count", 0) or 0)
    beneficiary_added_minutes = int(payload.get("beneficiary_added_minutes", 9999) or 9999)

    def add_rule(name: str, sev: str, detail: str):
        rules.append({"name": name, "severity": sev, "detail": detail})

    # 1) Absolute large amount
    if amt >= ABS_CRIT:
        add_rule(
            "Absolute very large amount",
            "CRITICAL",
            f"Amount {amt:.2f} {currency} >= critical {ABS_CRIT:.2f} {currency}.",
        )

    # 2) Impossible travel based on geo-coordinates
    impossible_travel_distance = None
    if last_lat is not None and last_lon is not None and txn_lat is not None and txn_lon is not None:
        impossible_travel_distance = haversine_km(last_lat, last_lon, txn_lat, txn_lon)

    device_checks_enabled = channel not in ("bank", "atm")

    if device_checks_enabled:
        device_new = (not last_device) or last_device == "" or (curr_device and curr_device != last_device)
        location_changed = impossible_travel_distance is not None and impossible_travel_distance > 500
        if device_new and location_changed and amt > MED_AMT:
            add_rule(
                "New device + Impossible travel + High amount",
                "CRITICAL",
                f"New device + travel {impossible_travel_distance:.1f} km; amount {amt:.2f} {currency}.",
            )

    # 3) Multiple beneficiaries + high transfer
    if beneficiaries_added_24h >= 3 and amt > HIGH_AMT:
        add_rule(
            "Multiple beneficiaries added + high transfer",
            "CRITICAL",
            f"{beneficiaries_added_24h} beneficiaries added and amount {amt:.2f} {currency}.",
        )

    # 4) Velocity rules
    if txns_1h >= 10:
        add_rule("High velocity (1h)", "HIGH", f"{txns_1h} txns in last 1 hour.")
    if txns_24h >= 50:
        add_rule("Very high velocity (24h)", "HIGH", f"{txns_24h} txns in last 24h.")

    # 5) IP / declared mismatch (if you pass declared_country as KYC/home country)
    if ip_country and declared_country and ip_country != declared_country:
        sev = "HIGH" if amt > HIGH_AMT else "MEDIUM"
        add_rule(
            "IP / Declared country mismatch",
            sev,
            f"IP country '{ip_country}' differs from declared '{declared_country}'.",
        )

    # 6) Login security
    if failed_logins >= 5:
        add_rule("Multiple failed login attempts", "HIGH", f"{failed_logins} failed auth attempts.")

    # 7) New beneficiary + amount
    if new_benef and amt >= MED_AMT:
        add_rule(
            "New beneficiary + significant amount",
            "HIGH",
            "Transfer to newly added beneficiary with amount above threshold.",
        )

    # 8) IP flagged as risky
    if suspicious_ip_flag and amt > (MED_AMT / 4):
        add_rule("IP flagged by threat intelligence", "HIGH", "IP flagged and non-trivial amount.")

    # 9) ATM distance from last known location
    if channel == "atm" and atm_distance_km and atm_distance_km > 300:
        add_rule("ATM distance from last location", "HIGH", f"ATM is {atm_distance_km:.1f} km away.")

    # 10) Card issuing country mismatch vs home
    if card_country and home_country and card_country != home_country and amt > MED_AMT:
        add_rule(
            "Card country mismatch vs home country",
            "HIGH",
            f"Card country {card_country} != home country {home_country}.",
        )

    # 11) Amount vs historical spending patterns
    if monthly_avg > 0 and amt >= 5 * monthly_avg and amt > MED_AMT:
        add_rule(
            "Large spike vs monthly avg",
            "HIGH",
            f"Amount {amt:.2f} >= 5x monthly avg {monthly_avg:.2f}.",
        )
    elif rolling_avg_7d > 0 and amt >= 3 * rolling_avg_7d and amt > (MED_AMT / 2):
        add_rule(
            "Spike vs 7-day avg",
            "MEDIUM",
            f"Amount {amt:.2f} >= 3x 7-day avg {rolling_avg_7d:.2f}.",
        )
    elif monthly_avg > 0 and amt >= 2 * monthly_avg and amt > (MED_AMT / 2):
        add_rule(
            "Above monthly usual",
            "MEDIUM",
            f"Amount {amt:.2f} >= 2x monthly avg {monthly_avg:.2f}.",
        )

    # 12) Additional velocity
    if txns_1h >= 5:
        add_rule("Elevated velocity (1h)", "MEDIUM", f"{txns_1h} in last 1 hour.")
    if 10 <= txns_24h < 50:
        add_rule("Elevated velocity (24h)", "MEDIUM", f"{txns_24h} in last 24h.")

    # 13) Time-of-day rules (unusual times)
    if 0 <= hour <= 5 and monthly_avg < (MED_AMT * 2) and amt > (MED_AMT / 10):
        add_rule(
            "Late-night txn for low-activity customer",
            "MEDIUM",
            f"Txn at hour {hour} for low-activity customer; amt {amt:.2f}.",
        )

    if 0 <= hour <= 4 and amt >= HIGH_AMT:
        add_rule(
            "Very high amount during unusual time",
            "HIGH",
            f"Txn at hour {hour} with amount {amt:.2f} {currency} >= high threshold {HIGH_AMT:.2f}.",
        )

    # 14) Device new + low amount (benign)
    if device_checks_enabled and ((not last_device) or last_device == "") and amt < (MED_AMT / 10):
        add_rule("New device (low amount)", "LOW", "Transaction from new device but low amount.")

    # 15) Recently added beneficiaries but not extreme
    if 0 < beneficiaries_added_24h < 3:
        add_rule("Beneficiaries recently added", "LOW", f"{beneficiaries_added_24h} beneficiaries added.")

    # 16) Higher-risk countries based on transaction country
    high_risk_countries = {"nigeria", "romania", "ukraine", "russia"}
    if txn_country and txn_country in high_risk_countries:
        add_rule(
            "Transaction in higher-risk country",
            "MEDIUM",
            f"Transaction country flagged as higher-risk: {txn_country}.",
        )

    # 17) Card testing / micro-charges
    if card_small_attempts >= 6 and CARD_TEST_SMALL > 0:
        add_rule(
            "Card testing / micro-charges detected",
            "HIGH",
            f"{card_small_attempts} small attempts; micro amount {CARD_TEST_SMALL:.2f} {currency}.",
        )

    # 18) Large ATM withdrawal
    if channel == "atm" and amt >= ATM_HIGH:
        add_rule(
            "Large ATM withdrawal",
            "HIGH",
            f"ATM withdrawal {amt:.2f} {currency} >= {ATM_HIGH:.2f}",
        )

    # 19) POS repeat
    if pos_repeat_count >= 10:
        add_rule("POS repeat transactions", "HIGH", f"{pos_repeat_count} rapid transactions at same POS.")

    # 20) Immediate transfer to just-added beneficiary (bank / netbanking)
    if channel in ("netbanking", "bank") and beneficiary_added_minutes < 10 and amt >= MED_AMT:
        add_rule(
            "Immediate transfer to newly added beneficiary",
            "HIGH",
            f"Beneficiary added {beneficiary_added_minutes} minutes ago and transfer amount {amt:.2f} {currency}.",
        )

    # 21) Home vs transaction city/country
    if home_country and txn_country and home_country != txn_country:
        sev = "HIGH" if amt >= MED_AMT else "MEDIUM"
        add_rule(
            "Txn country differs from home country",
            sev,
            f"Home country '{home_country}' vs transaction country '{txn_country}'.",
        )

    if home_city and txn_city and home_city != txn_city and amt >= (MED_AMT / 2):
        add_rule(
            "Txn city differs from home city",
            "MEDIUM",
            f"Home city '{home_city}' vs transaction city '{txn_city}'.",
        )

    # 22) Transfer-specific structural checks
    if str(payload.get("TransactionType", "")).upper() == "TRANSFER":
        from_acc = payload.get("from_account_number")
        to_acc = payload.get("to_account_number")
        if not from_acc or not to_acc:
            add_rule(
                "Missing transfer account data",
                "HIGH",
                "Transfer missing source or destination account details.",
            )

    # Determine highest severity
    highest = "LOW"
    for r in rules:
        highest = escalate(highest, r["severity"])

    return rules, highest


# -------------------------
# Combine final risk
# -------------------------

def combine_final_risk(ml_risk: str, rule_highest: str) -> str:
    """Final risk is the maximum of ML label and rules label."""
    return escalate(ml_risk, rule_highest)


# -------------------------
# Explanation builder
# -------------------------

def build_explanation(
    payload: Dict,
    fraud_score: float,
    anomaly_score: float,
    ml_label: str,
    rules_triggered: List[Dict],
    final_risk: str,
) -> List[str]:
    """
    Human-readable explanation bullets combining ML & rules,
    used to justify scores for auditors / business.

    fraud_score, anomaly_score are expected to be in 0–100 normalized range.
    """
    reasons = []

    amt = float(payload.get("Amount", 0.0) or 0.0)
    currency = payload.get("Currency", "INR")
    channel = str(payload.get("Channel", "Unknown"))
    txn_type = str(payload.get("TransactionType", "Unknown"))
    hour = int(payload.get("hour", 0) or 0)
    monthly_avg = float(payload.get("monthly_avg", 0.0) or 0.0)
    txn_city = payload.get("txn_city", "")
    txn_country = payload.get("txn_country", "")
    home_city = payload.get("home_city", "")
    home_country = payload.get("home_country", "")

    reasons.append(
        f"ML model fraud risk score is {fraud_score:.1f} (0–100), anomaly risk score is {anomaly_score:.1f} (0–100), mapped to ML label **{ml_label}**."
    )
    reasons.append(
        f"Deterministic rules evaluated this as **{final_risk}** after combining ML and rules."
    )
    reasons.append(
        f"Transaction context: channel **{channel}**, type **{txn_type}**, amount **{amt:.2f} {currency}**, at hour **{hour}:00**."
    )

    if monthly_avg > 0 and amt > 2 * monthly_avg:
        reasons.append(
            f"Amount is significantly higher than customer's monthly average of ~{monthly_avg:.2f} {currency}."
        )

    if home_country and txn_country and home_country.lower() != txn_country.lower():
        reasons.append(
            f"Customer home country (**{home_country}**) is different from transaction country (**{txn_country}**)."
        )
    if home_city and txn_city and home_city.lower() != txn_city.lower():
        reasons.append(
            f"Customer home city (**{home_city}**) is different from transaction city (**{txn_city}**)."
        )

    if 0 <= hour <= 5:
        reasons.append(
            "Transaction took place during unusual hours (midnight to early morning), which increases fraud risk for large amounts."
        )

    if rules_triggered:
        # Summarise most important rules
        top_rules = sorted(
            rules_triggered,
            key=lambda r: SEVERITY_ORDER[r["severity"]],
            reverse=True,
        )[:3]
        for r in top_rules:
            reasons.append(f"Key rule fired: **{r['name']}** – {r['detail']}")

    return reasons


# -------------------------
# STREAMLIT UI
# -------------------------
st.set_page_config(
    page_title="AI Powered Real-Time Fraud Detection",
    page_icon="💳",
    layout="centered",
)
st.title("💳 AI Powered Real-Time Fraud Detection")



# Inline 4 fields: Currency, Amount, Date, Time
col1, col2, col3, col4 = st.columns(4)
with col1:
    currency = st.selectbox(
        "Select currency",
        CURRENCIES,
        index=CURRENCIES.index(DEFAULT_CURRENCY),
        key="currency_select",
        help="Currency in which the transaction is performed, e.g. INR / USD.",
    )
with col2:
    amount = st.number_input(
        f"Transaction amount",
        min_value=0.0,
        value=1200.0,
        step=10.0,
        key="amount_common",
        help="Total transaction amount in the selected currency.",
    )
with col3:
    txn_date = st.date_input(
        "Transaction date",
        value=datetime.date.today(),
        key="txn_date",
        help="Calendar date on which the transaction is initiated.",
    )
with col4:
    txn_time = st.time_input(
        "Transaction time",
        value=datetime.time(12, 0),
        key="txn_time",
        help="Local time of the transaction (24h clock).",
    )

# Combine for scoring
txn_dt = datetime.datetime.combine(txn_date, txn_time)
hour = txn_dt.hour
day_of_week = txn_dt.weekday()
month = txn_dt.month

st.markdown("---")
channel = st.selectbox(
    "Transaction Channel",
    ["Choose...", "Bank", "Mobile App", "ATM", "Credit Card", "POS", "Online Purchase", "NetBanking"],
    key="channel_select",
    help="Where the transaction was initiated from.",
)
if channel and channel != "Choose...":
    channel_lower = channel.lower()
    st.markdown(f"### Channel: {channel}")

    # Per-channel model metrics in sidebar
    with st.sidebar:
        st.markdown("### Channel Metrics")
        ch_metrics = MODEL_PERFORMANCE["channels"].get(channel_lower, {})
        if ch_metrics and any(v is not None for v in ch_metrics.values()):
            if ch_metrics["accuracy"] is not None:
                st.metric("Accuracy", f"{ch_metrics['accuracy'] * 100:.2f}%")
            if ch_metrics["precision"] is not None:
                st.metric("Precision", f"{ch_metrics['precision'] * 100:.2f}%")
            if ch_metrics["recall"] is not None:
                st.metric("Recall", f"{ch_metrics['recall'] * 100:.2f}%")
        else:
            st.caption("No per-channel metrics configured yet.")

    txn_options = CHANNEL_TXN_TYPES.get(channel_lower, ["OTHER"])
    txn_type = st.selectbox(
        "Transaction type",
        txn_options,
        key=f"txn_type_{channel_lower}",
        help="Functional nature of the transaction, e.g. TRANSFER, PAYMENT, BILL_PAY.",
    )

    # Transaction-type specific panels: TRANSFER / PAYMENT / BILL_PAY
    st.markdown("#### Transaction type details")

    transfer_fields = {}
    payment_fields = {}
    billpay_fields = {}

    # TRANSFER UI
    if str(txn_type).upper() == "TRANSFER":
        st.subheader("TRANSFER — source and destination account details")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            from_account_number = st.text_input(
                "From account number",
                key=f"from_acc_{channel_lower}",
                help="Source account number for the transfer, e.g. 001234567890.",
            )
            from_account_holder_name = st.text_input(
                "From account holder name",
                key=f"from_name_{channel_lower}",
                help="Name of the source account holder.",
            )
        with col_f2:
            to_account_number = st.text_input(
                "To account number",
                key=f"to_acc_{channel_lower}",
                help="Destination account number for the transfer.",
            )
            to_account_holder_name = st.text_input(
                "To account holder name",
                key=f"to_name_{channel_lower}",
                help="Name of the destination account holder.",
            )
        st.checkbox(
            "Is this to a known beneficiary?",
            value=False,
            key=f"benef_flag_{channel_lower}",
            help="Check if the beneficiary has been used previously.",
        )
        new_beneficiary = st.checkbox(
            "Is this a newly added beneficiary?",
            value=False,
            key=f"new_benef_{channel_lower}",
            help="Check if the beneficiary was added recently.",
        )
        beneficiary_added_minutes = st.number_input(
            "Minutes since beneficiary added (if applicable)",
            min_value=0,
            value=9999,
            step=1,
            key=f"ben_min_{channel_lower}",
            help="Use 9999 if beneficiary was not recently added.",
        )
        reason = st.text_area(
            "Reason / notes (optional)",
            key=f"transfer_reason_{channel_lower}",
            help="Free-text comments, e.g. 'rent payment' or 'family support'.",
        )
        transfer_fields.update(
            {
                "from_account_number": from_account_number,
                "from_account_holder_name": from_account_holder_name,
                "to_account_number": to_account_number,
                "to_account_holder_name": to_account_holder_name,
                "new_beneficiary": new_beneficiary,
                "beneficiary_added_minutes": int(beneficiary_added_minutes),
                "reason": reason,
            }
        )

    # PAYMENT UI
    if str(txn_type).upper() == "PAYMENT":
        st.subheader("PAYMENT — merchant / payment details")
        payment_category = st.selectbox(
            "Payment category",
            ["ecommerce", "utilities", "subscription", "pos", "other"],
            key=f"pay_cat_{channel_lower}",
            help="High-level category of the payment.",
        )
        merchant_id = st.text_input(
            "Merchant name / ID",
            key=f"merchant_{channel_lower}",
            help="Merchant name or identifier, e.g. Amazon, Store123.",
        )

        card_used = False
        card_masked = ""
        cvv_provided = True
        device_for_payment = ""

        # Do NOT ask 'Paid using card?' for Credit Card channel (it's implicit)
        if channel_lower != "credit card":
            card_used = st.checkbox(
                "Paid using card?",
                value=False,
                key=f"card_used_{channel_lower}",
                help="Check if a card (debit/credit) was used for this payment.",
            )
        else:
            card_used = True  # implicit for credit-card channel

        if card_used and channel_lower != "credit card":
            card_masked = st.text_input(
                "Card masked (e.g., 4111****1111)",
                key=f"pay_card_{channel_lower}",
                help="Masked card number used for the payment.",
            )
            cvv_provided = st.checkbox(
                "CVV provided (checked if present)",
                value=True,
                key=f"pay_cvv_{channel_lower}",
                help="Uncheck only if card was used without CVV (e.g. stored card).",
            )

        # Device info optional for digital channels
        if channel_lower in ("mobile app", "online purchase", "netbanking", "credit card"):
            device_for_payment = st.text_input(
                "Device / Browser (optional)",
                key=f"pay_device_{channel_lower}",
                help="Device or browser string, e.g. 'Android Chrome'.",
            )

        payment_fields.update(
            {
                "payment_category": payment_category,
                "merchant_id": merchant_id,
                "card_used": bool(card_used),
                "card_masked": card_masked,
                "cvv_provided": bool(cvv_provided),
                "device_info": device_for_payment,
            }
        )

    # BILL PAY UI
    if str(txn_type).upper() == "BILL_PAY":
        st.subheader("BILL PAY — structured biller details")
        biller_category = st.selectbox(
            "Biller category",
            ["electricity", "water", "telecom", "internet", "insurance", "other"],
            key=f"biller_cat_{channel_lower}",
            help="Type of bill being paid.",
        )
        biller_id = st.text_input(
            "Biller ID",
            key=f"biller_id_{channel_lower}",
            help="Identifier of the biller, e.g. utility account number.",
        )
        bill_reference_number = st.text_input(
            "Bill/reference number",
            key=f"bill_ref_{channel_lower}",
            help="Bill reference or invoice number.",
        )
        due_date = st.date_input(
            "Bill due date (optional)",
            value=None,
            key=f"bill_due_{channel_lower}",
            help="Due date printed on the bill, if available.",
        )
        bill_period_start = st.date_input(
            "Bill period start (optional)",
            value=None,
            key=f"bill_start_{channel_lower}",
            help="Start date of the billing period.",
        )
        bill_period_end = st.date_input(
            "Bill period end (optional)",
            value=None,
            key=f"bill_end_{channel_lower}",
            help="End date of the billing period.",
        )
        billpay_fields.update(
            {
                "biller_category": biller_category,
                "biller_id": biller_id,
                "bill_reference_number": bill_reference_number,
                "due_date": str(due_date) if due_date else "",
                "bill_period_start": str(bill_period_start) if bill_period_start else "",
                "bill_period_end": str(bill_period_end) if bill_period_end else "",
            }
        )

    st.markdown("---")
    st.markdown("#### Channel-specific fields")

    # Bank
    bank_fields = {}
    if channel_lower == "bank":
        st.subheader("In-branch (Bank) fields — Identity only")
        id_type = st.selectbox(
            "ID Document Type",
            ["Passport", "Driver License", "Government ID", "Other"],
            key="bank_id_type",
            help="Type of identity document presented.",
        )
        id_number = st.text_input(
            "ID Document Number",
            key="bank_id_number",
            help="Identifier printed on the ID document.",
        )
        branch = st.text_input(
            "Branch Name / Code",
            value="",
            key="bank_branch",
            help="Name or internal code of the physical branch.",
        )
        teller_id = st.text_input(
            "Teller ID (optional)",
            value="",
            key="bank_teller",
            help="Internal teller identifier if available.",
        )
        bank_fields.update(
            {"id_type": id_type, "id_number": id_number, "branch": branch, "teller_id": teller_id}
        )

    # ATM
    atm_fields = {}
    if channel_lower == "atm":
        st.subheader("ATM fields — card + ATM info (no IP/device)")
        atm_id = st.text_input(
            "ATM ID / Terminal",
            key="atm_id",
            help="Unique ATM or terminal identifier.",
        )
        atm_location = st.text_input(
            "ATM Location (free text)",
            key="atm_location",
            help="Human-readable ATM location, e.g. 'Mumbai - Bandra West'.",
        )
        atm_distance_km = st.number_input(
            "ATM distance from last known location (km)",
            min_value=0.0,
            value=0.0,
            step=1.0,
            key="atm_distance",
            help="Approximate distance between previous known location and this ATM.",
        )
        card_masked_atm = st.text_input(
            "Card masked (e.g., 4111****1111)",
            key="atm_card_masked",
            help="Masked card number used at the ATM.",
        )
        atm_fields.update(
            {
                "atm_id": atm_id,
                "atm_location": atm_location,
                "atm_distance_km": atm_distance_km,
                "card_masked": card_masked_atm,
            }
        )

    # Mobile App
    mobile_fields = {}
    if channel_lower == "mobile app":
        st.subheader("Mobile App fields — device + app telemetry")
        device = st.text_input(
            "Device / OS (e.g., Android)",
            value="Android",
            key="mobile_device",
            help="Device family, e.g. 'Android', 'iOS'.",
        )
        device_fingerprint = st.text_input(
            "Device fingerprint (optional)",
            key="mobile_device_fp",
            help="Hashed device fingerprint or identifier.",
        )
        app_version = st.text_input(
            "App version",
            value="1.0.0",
            key="mobile_app_ver",
            help="Version of the mobile application.",
        )
        last_device = st.text_input(
            "Last known device (optional)",
            key="mobile_last_device",
            help="Device ID used in previous sessions, if recorded.",
        )
        mobile_fields.update(
            {
                "DeviceID": device,
                "device_fingerprint": device_fingerprint,
                "app_version": app_version,
                "device_last_seen": last_device,
            }
        )

    # Credit Card
    cc_fields = {}
    if channel_lower == "credit card":
        st.subheader("Credit Card: choose mode")
        cc_mode = st.radio(
            "Credit Card mode",
            ["POS (physical)", "Mobile/Web (app or web)"],
            key="cc_mode",
            help="Was the card used physically at POS, or via app/web?",
        )
        if cc_mode == "POS (physical)":
            card_masked_cc = st.text_input(
                "Card masked (4111****1111)",
                key="cc_pos_card",
                help="Masked card number used at POS.",
            )
            card_country = st.text_input(
                "Card issuing country",
                key="cc_pos_country",
                help="Country where the card was issued, e.g. 'India'.",
            )
            cvv_provided = st.checkbox(
                "CVV provided (checked if present)",
                value=True,
                key="cc_pos_cvv",
                help="Was the CVV available for this transaction?",
            )
            pos_merchant_id = st.text_input(
                "POS Merchant ID (optional)",
                key="cc_pos_mid",
                help="Merchant ID of the POS terminal.",
            )
            cc_fields.update(
                {
                    "card_masked": card_masked_cc,
                    "card_country": card_country,
                    "cvv_provided": cvv_provided,
                    "pos_merchant_id": pos_merchant_id,
                }
            )
        else:
            card_masked_cc = st.text_input(
                "Card masked (4111****1111)",
                key="cc_web_card",
                help="Masked card number used online.",
            )
            card_country = st.text_input(
                "Card issuing country",
                key="cc_web_country",
                help="Country where the card was issued.",
            )
            cvv_provided = st.checkbox(
                "CVV provided (checked if present)",
                value=True,
                key="cc_web_cvv",
                help="Was CVV provided as part of the online transaction?",
            )
            device_cc = st.text_input(
                "Device / Browser (optional)",
                key="cc_web_device",
                help="Device or browser user-agent string.",
            )
            device_fingerprint_cc = st.text_input(
                "Device fingerprint (optional)",
                key="cc_web_fp",
                help="Hashed device fingerprint.",
            )
            last_device_cc = st.text_input(
                "Last known device (optional)",
                key="cc_web_last_device",
                help="Device identifier observed in previous sessions.",
            )
            cc_fields.update(
                {
                    "card_masked": card_masked_cc,
                    "card_country": card_country,
                    "cvv_provided": cvv_provided,
                    "DeviceID": device_cc,
                    "device_fingerprint": device_fingerprint_cc,
                    "device_last_seen": last_device_cc,
                }
            )

    # POS
    pos_fields = {}
    if channel_lower == "pos":
        st.subheader("POS fields")
        pos_merchant_id = st.text_input(
            "POS Merchant ID",
            key="pos_mid",
            help="Merchant ID or terminal ID of the POS device.",
        )
        store_name = st.text_input(
            "Store name",
            key="pos_store",
            help="Name of the physical store.",
        )
        pos_repeat_count = st.number_input(
            "Rapid repeat transactions at same POS",
            min_value=0,
            value=0,
            step=1,
            key="pos_repeat",
            help="Number of back-to-back transactions at the same POS.",
        )
        pos_fields.update(
            {
                "pos_merchant_id": pos_merchant_id,
                "store_name": store_name,
                "pos_repeat_count": pos_repeat_count,
            }
        )

    # Online Purchase
    online_fields = {}
    if channel_lower == "online purchase":
        st.subheader("Online Purchase fields (device + addresses)")
        merchant = st.text_input(
            "Merchant name / ID",
            key="online_merchant",
            help="Online merchant name or ID.",
        )
        shipping_address = st.text_input(
            "Shipping address",
            key="online_ship",
            help="Full shipping address, e.g. 'Flat 101, Mumbai, India'.",
        )
        billing_address = st.text_input(
            "Billing address",
            value=shipping_address,
            key="online_bill",
            help="Billing address; prefilled with shipping address by default.",
        )
        used_card_online = st.checkbox(
            "Paid by card online?",
            value=False,
            key="online_card_used",
            help="Check if a card was used to pay online.",
        )
        card_masked_online = ""
        cvv_provided_online = True
        if used_card_online:
            cvv_provided_online = st.checkbox(
                "CVV provided (checked if present)",
                value=True,
                key="online_cvv",
                help="Uncheck only if CVV was not required.",
            )
            card_masked_online = st.text_input(
                "Card masked",
                key="online_card_masked",
                help="Masked card number used online.",
            )
        device_online = st.text_input(
            "Device / Browser (optional)",
            key="online_device",
            help="Device or browser user-agent string.",
        )
        last_device_online = st.text_input(
            "Last known device (optional)",
            key="online_last_device",
            help="Device identifier observed in previous online sessions.",
        )
        online_fields.update(
            {
                "merchant": merchant,
                "shipping_address": shipping_address,
                "billing_address": billing_address,
                "card_masked": card_masked_online,
                "cvv_provided": cvv_provided_online,
                "DeviceID": device_online,
                "device_last_seen": last_device_online,
            }
        )

    # NetBanking
    netbanking_fields = {}
    if channel_lower == "netbanking":
        st.subheader("NetBanking fields (device-aware)")
        username = st.text_input(
            "User ID / Login",
            key="nb_user",
            help="NetBanking user login or customer ID.",
        )
        device = st.text_input(
            "Device / Browser (used to login)",
            key="nb_device",
            help="Device or browser used for NetBanking login.",
        )
        last_device = st.text_input(
            "Last known device (optional)",
            key="nb_last_device",
            help="Device used in previous NetBanking sessions.",
        )
        beneficiary = st.text_input(
            "Beneficiary (if transfer)",
            key="nb_beneficiary",
            help="Primary beneficiary name or identifier.",
        )
        new_beneficiary = st.checkbox(
            "Is beneficiary newly added?",
            value=False,
            key="nb_new_benef",
            help="True if beneficiary was created shortly before this transfer.",
        )
        beneficiary_added_minutes = st.number_input(
            "Minutes since beneficiary was added (if known)",
            min_value=0,
            value=9999,
            step=1,
            key="nb_benef_minutes",
            help="Use actual minutes when available; 9999 means 'unknown/old'.",
        )
        netbanking_fields.update(
            {
                "username": username,
                "DeviceID": device,
                "device_last_seen": last_device,
                "beneficiary": beneficiary,
                "new_beneficiary": new_beneficiary,
                "beneficiary_added_minutes": int(beneficiary_added_minutes),
            }
        )

    # Optional telemetry panel
    st.markdown("#### Optional telemetry (helps rules; provide if available)")
    colT1, colT2 = st.columns(2)
    with colT1:
        monthly_avg = st.number_input(
            f"Customer monthly average spend ({currency})",
            min_value=0.0,
            value=10000.0,
            step=100.0,
            key=f"monthly_avg_{channel_lower}",
            help="Average monthly outgoing amount across all channels.",
        )
        rolling_avg_7d = st.number_input(
            f"7-day rolling average ({currency})",
            min_value=0.0,
            value=3000.0,
            step=50.0,
            key=f"rolling_avg_{channel_lower}",
            help="Average outgoing amount in the last 7 days.",
        )
        txns_last_1h = st.number_input(
            "Transactions in last 1 hour",
            min_value=0,
            value=0,
            step=1,
            key=f"txns1h_{channel_lower}",
            help="Total number of transactions across all channels in last 60 minutes.",
        )
        txns_last_24h = st.number_input(
            "Transactions in last 24 hours",
            min_value=0,
            value=0,
            step=1,
            key=f"txns24h_{channel_lower}",
            help="Total number of transactions in the last 24 hours.",
        )
    with colT2:
        txns_last_7d = st.number_input(
            "Transactions in last 7 days",
            min_value=0,
            value=7,
            step=1,
            key=f"txns7d_{channel_lower}",
            help="Total number of transactions in the last 7 days.",
        )
        beneficiaries_added_24h = st.number_input(
            "Beneficiaries added in last 24h",
            min_value=0,
            value=0,
            step=1,
            key=f"ben24_{channel_lower}",
            help="Number of new beneficiaries created in last 24 hours.",
        )
        beneficiaries_added_24h = int(beneficiaries_added_24h)
        failed_login_attempts = st.number_input(
            "Failed login attempts (recent)",
            min_value=0,
            value=0,
            step=1,
            key=f"failed_{channel_lower}",
            help="Count of failed authentication attempts in the recent window.",
        )

    # IP & Geo telemetry (centralized)
    st.markdown("#### IP / Geo (centralized fields)")

    home_city = st.text_input(
        "Customer home city",
        key=f"home_city_{channel_lower}",
        help="Customer’s usual residential city, e.g. 'Bangalore'.",
    )
    home_country = st.text_input(
        "Customer home country",
        key=f"home_country_{channel_lower}",
        help="Customer’s KYC country of residence, e.g. 'India'.",
    )

    if channel_lower in ("bank", "atm"):
        st.info("Bank/ATM: client IP is not collected by design for in-branch and ATM flows.")
        client_ip = ""
        ip_country = ""
        suspicious_ip_flag = False
    else:
        client_ip = st.text_input(
            "Client IP (optional)",
            key=f"client_ip_{channel_lower}",
            help="IP address as seen by the front-end channel.",
        )
       
        )
        suspicious_ip_flag = st.checkbox(
            "IP flagged by threat intel?",
            value=False,
            key=f"suspicious_{channel_lower}",
            help="Check if IP appears in any threat intelligence list.",
        )

    txn_location_ip = st.text_input(
        "Transaction origin IP (txn_location_ip) (optional)",
        key=f"txn_loc_ip_{channel_lower}",
        help="IP used to initiate the transaction.",
    )
    txn_city = st.text_input(
        "Transaction city (txn_city)",
        key=f"txn_city_{channel_lower}",
        help="City where the transaction originates (terminal/app).",
    )
    txn_country = st.text_input(
        "Transaction country (txn_country)",
        key=f"txn_country_{channel_lower}",
        help="Country where the transaction originates.",
    )

    # Lat/long optional for distance/impossible-travel checks
    last_known_lat = st.number_input(
        "Last known latitude (optional)",
        format="%.6f",
        value=0.0,
        key=f"lastlat_{channel_lower}",
        help="Latitude from previous session or transaction, if available.",
    )
    last_known_lon = st.number_input(
        "Last known longitude (optional)",
        format="%.6f",
        value=0.0,
        key=f"lastlon_{channel_lower}",
        help="Longitude from previous session or transaction, if available.",
    )
    txn_lat = st.number_input(
        "Transaction latitude (optional)",
        format="%.6f",
        value=0.0,
        key=f"txnlat_{channel_lower}",
        help="Latitude where current transaction originates.",
    )
    txn_lon = st.number_input(
        "Transaction longitude (optional)",
        format="%.6f",
        value=0.0,
        key=f"txnon_{channel_lower}",
        help="Longitude where current transaction originates.",
    )

    last_known_lat = last_known_lat if last_known_lat != 0.0 else None
    last_known_lon = last_known_lon if last_known_lon != 0.0 else None
    txn_lat = txn_lat if txn_lat != 0.0 else None
    txn_lon = txn_lon if txn_lon != 0.0 else None

    # Submit
    submit = st.button("🚀 Run Fraud Check", key=f"submit_{channel_lower}")

    if submit:
        # Measure response time from here (client req 9)
        start_time = time.perf_counter()

        payload: Dict = {
            "Amount": amount,
            "Currency": currency,
            "TransactionType": txn_type,
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
            "beneficiaries_added_24h": int(beneficiaries_added_24h),
            "failed_login_attempts": int(failed_login_attempts),
            # centralized ip/geo
            "client_ip": client_ip,
            "ip_country": ip_country,
            "txn_location_ip": txn_location_ip,
            "txn_city": txn_city,
            "txn_country": txn_country,
            "home_city": home_city,
            "home_country": home_country,
            "declared_country": home_country,  # treat KYC/home as declared country
            "suspicious_ip_flag": suspicious_ip_flag,
            "last_known_lat": last_known_lat,
            "last_known_lon": last_known_lon,
            "txn_lat": txn_lat,
            "txn_lon": txn_lon,
        }

        # attach transaction-type specific details
        if str(txn_type).upper() == "TRANSFER":
            payload.update(transfer_fields)
        elif str(txn_type).upper() == "PAYMENT":
            payload.update(payment_fields)
        elif str(txn_type).upper() == "BILL_PAY":
            payload.update(billpay_fields)

        # attach channel-specific details
        if channel_lower == "bank":
            payload.update(bank_fields)
        elif channel_lower == "atm":
            payload.update(atm_fields)
        elif channel_lower == "mobile app":
            payload.update(mobile_fields)
        elif channel_lower == "credit card":
            payload.update(cc_fields)
        elif channel_lower == "pos":
            payload.update(pos_fields)
        elif channel_lower == "online purchase":
            payload.update(online_fields)
        elif channel_lower == "netbanking":
            payload.update(netbanking_fields)

        convert_to_inr_for_model = False

        with st.spinner("Scoring with ML models..."):
            fraud_prob_raw, anomaly_raw, ml_label = score_transaction_ml(
                supervised_pipeline,
                iforest_pipeline,
                payload,
                convert_to_inr=convert_to_inr_for_model,
                currency=currency,
            )

        # Normalize for interpretability (0–100)
        fraud_score = normalize_score(fraud_prob_raw, min_val=0.0, max_val=0.02)
        anomaly_score = normalize_score(anomaly_raw, min_val=0.0, max_val=0.10)

        rules_triggered, rules_highest = evaluate_rules(payload, currency)

        final_risk = combine_final_risk(ml_label, rules_highest)

        end_time = time.perf_counter()
        response_time_s = end_time - start_time

        # ---------------- Results UI ----------------
        st.markdown("## 🔎 Results")
        color_map = {
            "LOW": "#2e7d32",
            "MEDIUM": "#f9a825",
            "HIGH": "#f57c00",
            "CRITICAL": "#c62828",
        }
        badge_color = color_map.get(final_risk, "#607d8b")
        st.markdown(
            f"""<div style="padding:0.75rem 1rem;border-radius:0.5rem;background-color:{badge_color}22;border:1px solid {badge_color};">
                <strong style="color:{badge_color};font-size:1.1rem;">Final Risk Level: {final_risk}</strong>
            </div>""",
            unsafe_allow_html=True,
        )

        colA, colB, colC = st.columns(3)
        with colA:
            st.metric(
                "Fraud Risk Score (0–100)",
                f"{fraud_score:.1f}",
                help="Calibrated fraud risk score derived from supervised ML probability.",
            )
            st.metric("ML Risk Label", ml_label)
        with colB:
            st.metric(
                "Anomaly Risk Score (0–100)",
                f"{anomaly_score:.1f}",
                help="Calibrated anomaly risk score derived from IsolationForest.",
            )
            st.metric("Rules-derived highest severity", rules_highest)
        with colC:
            st.metric(
                "Response time (seconds)",
                f"{response_time_s:.3f}",
                help="End-to-end scoring time for this transaction.",
            )

        st.markdown("### 🧠 ML & Rules Justification")
        explanation_bullets = build_explanation(
            payload,
            fraud_score,
            anomaly_score,
            ml_label,
            rules_triggered,
            final_risk,
        )
        for line in explanation_bullets:
            st.markdown(f"- {line}")

        st.markdown("### ⚠ Triggered Rules (detailed)")
        if rules_triggered:
            for r in rules_triggered:
                sev = r["severity"]
                emoji = "🔴" if sev in ("HIGH", "CRITICAL") else "🟠" if sev == "MEDIUM" else "🟢"
                st.write(f"{emoji} **{r['name']}** — *{r['severity']}*")
                st.caption(r["detail"])
        else:
            st.success("No deterministic rules triggered.")

        st.markdown("### 📦 Payload (debug)")
        st.json(payload)

        # Examples section for KT
        with st.expander("📚 Example good & fraud transactions for this channel"):
            ch_df = EXAMPLE_TXNS_DF[EXAMPLE_TXNS_DF["channel"] == channel_lower]
            good_df = ch_df[ch_df["example_type"] == "GOOD"]
            fraud_df = ch_df[ch_df["example_type"] == "FRAUD"]

            st.markdown("**Good / genuine transactions (examples)**")
            st.dataframe(
                good_df[
                    [
                        "transaction_type",
                        "amount_in_inr",
                        "fraud_confidence_ml_pct",
                        "anomaly_score_ml_pct",
                        "final_risk",
                    ]
                ]
            )
            st.markdown("**Fraud / suspicious transactions (examples)**")
            st.dataframe(
                fraud_df[
                    [
                        "transaction_type",
                        "amount_in_inr",
                        "fraud_confidence_ml_pct",
                        "anomaly_score_ml_pct",
                        "final_risk",
                    ]
                ]
            )

else:
    st.info(
        "Select currency, enter amount/date/time, then pick a channel to show channel-specific inputs."
    )
