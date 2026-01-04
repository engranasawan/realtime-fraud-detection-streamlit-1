# app.py
# AI Powered Real-Time Fraud Detection (Optimized + Human-Readable Scoring)
#
# Goals achieved:
# ✅ ML Fraud Risk Score is ALWAYS human readable on a 1–100 scale (monotonic: higher risk => higher score)
# ✅ Anomaly Risk Score is ALSO human readable on a 1–100 scale (monotonic)
# ✅ LOW / MEDIUM / HIGH / CRITICAL labels are bound to the 1–100 score bands
# ✅ Rule engine kept intact (logic preserved)
# ✅ UI separates inputs that affect ML from inputs that affect Rules (two tabs)
#
# NOTE:
# - ML uses ONLY these 8 features:
#   Amount, TransactionType, Location, DeviceID, Channel, hour, day_of_week, month
# - Everything else is used for rules/explanations only.

import datetime
import json
import time
from math import radians, sin, cos, asin, sqrt
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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

SEVERITY_ORDER = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

# Canonical channel transaction types (keys are lower-case canonical names)
CHANNEL_TXN_TYPES = {
    "atm": ["CASH_WITHDRAWAL", "TRANSFER"],
    "credit card": ["PAYMENT", "REFUND"],
    "debit card": ["PAYMENT", "REFUND"],
    "mobile app": ["PAYMENT", "TRANSFER", "BILL_PAY"],
    "pos": ["PAYMENT"],
    "online purchase": ["PAYMENT"],
    "bank": ["DEPOSIT", "TRANSFER", "WITHDRAWAL"],
    "netbanking": ["TRANSFER", "BILL_PAY", "PAYMENT"],
    "upi": ["PAYMENT", "TRANSFER"],
    "other": ["PAYMENT", "TRANSFER", "BILL_PAY", "OTHER"],
}

# Mapping UI display names -> canonical Channel values (must match training categories)
CHANNEL_DISPLAY_TO_CANONICAL = {
    "Onsite Branch Transaction (Bank)": "Bank",
    "Mobile App": "Mobile App",
    "ATM": "ATM",
    "Credit Card": "Credit Card",
    "Debit Card": "Debit Card",
    "POS": "POS",
    "Online Purchase": "Online Purchase",
    "NetBanking": "NetBanking",
    "UPI": "UPI",
    "Other": "Other",
}

# -------------------------
# HELPERS
# -------------------------
def currency_to_inr(amount: float, currency: str) -> float:
    """Convert an input amount in selected currency into INR for ML consistency."""
    return float(amount) * float(INR_PER_UNIT.get(currency, 1.0))


def inr_to_currency(amount_in_inr: float, currency: str) -> float:
    """Convert INR thresholds to a selected currency for rules consistency."""
    denom = INR_PER_UNIT.get(currency, 1.0)
    if denom == 0:
        return amount_in_inr
    return amount_in_inr / denom


def haversine_km(lat1, lon1, lat2, lon2) -> Optional[float]:
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
# HUMAN-READABLE SCORING (FIXED)
# -------------------------
def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def fraud_prob_to_score_1_100(p: float) -> float:
    """
    Map model fraud probability p (0..1) -> 1..100 in a monotonic, human-friendly way.

    Why this mapping?
    - Linear mapping often makes most real-world probabilities look like "0–5".
    - This uses a log curve to spread low probabilities while still being monotonic.

    score = 1 + 99 * log1p(k*p) / log1p(k)
    with k=100 makes p=1% feel meaningful while keeping p=100% near 100.
    """
    try:
        p = float(p)
    except Exception:
        p = 0.0
    p = clamp(p, 0.0, 1.0)
    k = 100.0
    import math
    score = 1.0 + 99.0 * (math.log1p(k * p) / math.log1p(k))
    return float(clamp(score, 1.0, 100.0))


def anomaly_raw_to_score_1_100(anom: float, lo: float = -0.10, hi: float = 0.10) -> float:
    """
    Map IsolationForest anomaly score (higher => more anomalous) into 1..100.

    In your training runs, anomaly_score tended to live roughly in ~[-0.08, 0.09].
    We use a safe default lo/hi and clamp.
    """
    try:
        anom = float(anom)
    except Exception:
        anom = 0.0
    anom = clamp(anom, lo, hi)
    if hi == lo:
        return 1.0
    score = 1.0 + 99.0 * ((anom - lo) / (hi - lo))
    return float(clamp(score, 1.0, 100.0))


def score_to_label(score_1_100: float) -> str:
    """
    Bind 1–100 score to labels (simple, stable, human understandable).
    """
    s = clamp(float(score_1_100), 1.0, 100.0)
    if s >= 75:
        return "CRITICAL"
    if s >= 50:
        return "HIGH"
    if s >= 25:
        return "MEDIUM"
    return "LOW"


# -------------------------
# ML MODEL LOADING
# -------------------------
@st.cache_resource
def load_models_and_thresholds():
    models_dir = Path("models")

    def _load_joblib(name: str):
        path = models_dir / name
        return joblib.load(path)

    supervised = None
    iforest = None
    thresholds = {}

    try:
        supervised = _load_joblib("supervised_lgbm_pipeline.joblib")
    except Exception as e:
        supervised = None
        st.warning("Supervised model could not be loaded (models/supervised_lgbm_pipeline.joblib).")
        st.caption(str(e))

    try:
        iforest = _load_joblib("iforest_pipeline.joblib")
    except Exception as e:
        iforest = None
        st.warning("IsolationForest model could not be loaded (models/iforest_pipeline.joblib).")
        st.caption(str(e))

    try:
        th_path = models_dir / "model_thresholds.json"
        if th_path.exists():
            thresholds = json.loads(th_path.read_text())
    except Exception as e:
        thresholds = {}
        st.caption(f"Could not read model_thresholds.json: {e}")

    return supervised, iforest, thresholds


supervised_pipeline, iforest_pipeline, MODEL_THRESHOLDS = load_models_and_thresholds()


# -------------------------
# ML SCORING WRAPPER (ONLY 8 FEATURES)
# -------------------------
ML_FEATURES = ["Amount", "TransactionType", "Location", "DeviceID", "Channel", "hour", "day_of_week", "month"]


def score_transaction_ml(
    model_pipeline,
    iforest_pipeline,
    ml_payload: Dict,
    currency: str = "INR",
    convert_amount_to_inr: bool = True,
) -> Tuple[float, float, float, float, str, str]:
    """
    Returns:
      fraud_prob (0..1),
      anomaly_raw (higher => more anomalous),
      fraud_score_1_100,
      anomaly_score_1_100,
      fraud_label,
      anomaly_label
    """
    amt = float(ml_payload.get("Amount", 0.0) or 0.0)
    if convert_amount_to_inr:
        amt = currency_to_inr(amt, currency)

    model_df = pd.DataFrame(
        [
            {
                "Amount": amt,
                "TransactionType": ml_payload.get("TransactionType", "PAYMENT"),
                "Location": ml_payload.get("Location", "Unknown"),
                "DeviceID": ml_payload.get("DeviceID", "Unknown"),
                "Channel": ml_payload.get("Channel", "Other"),
                "hour": int(ml_payload.get("hour", 0) or 0),
                "day_of_week": int(ml_payload.get("day_of_week", 0) or 0),
                "month": int(ml_payload.get("month", 1) or 1),
            }
        ]
    )

    # Reindex to trained feature list if available (prevents feature-name/order quirks)
    try:
        if model_pipeline is not None and hasattr(model_pipeline, "feature_names_in_"):
            model_df = model_df.reindex(columns=list(model_pipeline.feature_names_in_))
    except Exception:
        pass

    fraud_prob = 0.0
    anomaly_raw = 0.0

    if model_pipeline is not None:
        try:
            fraud_prob = float(model_pipeline.predict_proba(model_df)[0, 1])
        except Exception:
            fraud_prob = 0.0

    if iforest_pipeline is not None:
        try:
            raw = float(iforest_pipeline.decision_function(model_df)[0])
            anomaly_raw = -raw  # higher => more anomalous
        except Exception:
            anomaly_raw = 0.0

    fraud_score = fraud_prob_to_score_1_100(fraud_prob)
    anom_score = anomaly_raw_to_score_1_100(anomaly_raw, lo=-0.10, hi=0.10)

    fraud_label = score_to_label(fraud_score)
    anom_label = score_to_label(anom_score)

    return fraud_prob, anomaly_raw, fraud_score, anom_score, fraud_label, anom_label


# -------------------------
# RULE ENGINE (LOGIC PRESERVED)
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
    channel_val = str(payload.get("Channel", "") or "")
    channel = channel_val.lower()
    hour = int(payload.get("hour", 0) or 0)
    monthly_avg = float(payload.get("monthly_avg", 0.0) or 0.0)
    rolling_avg_7d = float(payload.get("rolling_avg_7d", 0.0) or 0.0)
    txns_1h = int(payload.get("txns_last_1h", 0) or 0)
    txns_24h = int(payload.get("txns_last_24h", 0) or 0)
    txns_7d = int(payload.get("txns_last_7d", 0) or 0)
    failed_logins = int(payload.get("failed_login_attempts", 0) or 0)

    new_benef = bool(payload.get("new_beneficiary", False))
    existing_benef = bool(payload.get("existing_beneficiary", False))
    beneficiaries_added_24h = int(payload.get("beneficiaries_added_24h", 0) or 0)
    beneficiary_added_minutes = int(payload.get("beneficiary_added_minutes", 9999) or 9999)

    declared_country = str(payload.get("declared_country", "") or "").lower()
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
    cvv_provided = bool(payload.get("cvv_provided", True))

    suspicious_ip_flag = bool(payload.get("suspicious_ip_flag", False))
    card_small_attempts = int(payload.get("card_small_attempts_in_5min", 0) or 0)
    pos_repeat_count = int(payload.get("pos_repeat_count", 0) or 0)

    id_type = str(payload.get("id_type", "") or "").strip()
    id_number = str(payload.get("id_number", "") or "").strip()

    vpn_detected = bool(payload.get("vpn_detected", False))
    vpn_provider = str(payload.get("vpn_provider", "") or "")
    tor_exit_node = bool(payload.get("tor_exit_node", False))
    cloud_host_ip = bool(payload.get("cloud_host_ip", False))
    ip_risk_score = int(payload.get("ip_risk_score", 0) or 0)

    def add_rule(name: str, sev: str, detail: str):
        rules.append({"name": name, "severity": sev, "detail": detail})

    # UPI-specific rules
    if channel == "upi":
        if amt >= MED_AMT and new_benef:
            add_rule(
                "UPI payment to newly added payee",
                "HIGH",
                "UPI transfer to an untrusted/new VPA with significant amount.",
            )
        if txns_1h >= 5:
            add_rule("UPI rapid transactions", "MEDIUM", f"{txns_1h} UPI transactions within 1 hour.")

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

    # 5) Transaction / declared country mismatch
    if txn_country and declared_country and txn_country != declared_country and channel not in ("bank", "atm"):
        sev = "HIGH" if amt > HIGH_AMT else "MEDIUM"
        add_rule(
            "Txn / Declared country mismatch",
            sev,
            f"Transaction country '{txn_country}' differs from declared '{declared_country}'.",
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
        add_rule("Large spike vs monthly avg", "HIGH", f"Amount {amt:.2f} >= 5x monthly avg {monthly_avg:.2f}.")
    elif rolling_avg_7d > 0 and amt >= 3 * rolling_avg_7d and amt > (MED_AMT / 2):
        add_rule("Spike vs 7-day avg", "MEDIUM", f"Amount {amt:.2f} >= 3x 7-day avg {rolling_avg_7d:.2f}.")
    elif monthly_avg > 0 and amt >= 2 * monthly_avg and amt > (MED_AMT / 2):
        add_rule("Above monthly usual", "MEDIUM", f"Amount {amt:.2f} >= 2x monthly avg {monthly_avg:.2f}.")

    # 12) Additional velocity
    if txns_1h >= 5:
        add_rule("Elevated velocity (1h)", "MEDIUM", f"{txns_1h} in last 1 hour.")
    if 10 <= txns_24h < 50:
        add_rule("Elevated velocity (24h)", "MEDIUM", f"{txns_24h} in last 24 hours.")

    # 13) Time-of-day rules
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

    # 16) Higher-risk countries (txn_country)
    high_risk_countries = {"nigeria", "romania", "ukraine", "russia"}
    if txn_country and txn_country in high_risk_countries:
        add_rule("Transaction in higher-risk country", "MEDIUM", f"Transaction country flagged as higher-risk: {txn_country}.")

    # 17) Card testing / micro-charges
    if card_small_attempts >= 6 and CARD_TEST_SMALL > 0:
        add_rule(
            "Card testing / micro-charges detected",
            "HIGH",
            f"{card_small_attempts} small attempts; micro amount {CARD_TEST_SMALL:.2f} {currency}.",
        )

    # 18) Large ATM withdrawal
    if channel == "atm" and amt >= ATM_HIGH:
        add_rule("Large ATM withdrawal", "HIGH", f"ATM withdrawal {amt:.2f} {currency} >= {ATM_HIGH:.2f}")

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
            add_rule("Missing transfer account data", "HIGH", "Transfer missing source or destination account details.")

    # 23) Onsite branch transactions without identity
    if channel == "bank":
        if not id_type or not id_number:
            add_rule("Onsite branch transaction without identity", "HIGH", "No identity document captured for onsite branch transaction.")

    # 24) VPN / TOR / cloud-host IP rules (manual flags)
    if tor_exit_node:
        add_rule("Connection via TOR exit node", "CRITICAL", "Traffic is coming from a TOR exit node – highly anonymized.")
    if vpn_detected:
        sev = "HIGH" if amt >= MED_AMT else "MEDIUM"
        add_rule("VPN usage detected", sev, f"Upstream systems flagged VPN usage ({vpn_provider or 'provider not specified'}).")
    if cloud_host_ip and channel not in ("atm", "bank"):
        add_rule("Connection from cloud hosting provider", "MEDIUM", "IP appears to belong to a cloud hosting provider, not a residential ISP.")

    if ip_risk_score >= 80:
        add_rule("High IP reputation risk score", "HIGH", f"IP reputation risk score {ip_risk_score} (>= 80).")
    elif ip_risk_score >= 50:
        add_rule("Elevated IP reputation risk score", "MEDIUM", f"IP reputation risk score {ip_risk_score} (>= 50).")

    # Highest severity
    highest = "LOW"
    for r in rules:
        highest = escalate(highest, r["severity"])

    return rules, highest


# -------------------------
# Combine final risk
# -------------------------
def combine_final_risk(ml_risk: str, rule_highest: str) -> str:
    return escalate(ml_risk, rule_highest)


# -------------------------
# Explanation builder (updated to reflect 1–100 scores clearly)
# -------------------------
def build_explanation(
    payload: Dict,
    fraud_prob: float,
    anomaly_raw: float,
    fraud_score_1_100: float,
    anomaly_score_1_100: float,
    ml_label: str,
    rules_triggered: List[Dict],
    final_risk: str,
) -> List[str]:
    reasons: List[str] = []

    amt = float(payload.get("Amount", 0.0) or 0.0)
    currency = payload.get("Currency", "INR")
    channel_val = str(payload.get("Channel", "Unknown"))
    txn_type = str(payload.get("TransactionType", "Unknown"))
    hour = int(payload.get("hour", 0) or 0)
    monthly_avg = float(payload.get("monthly_avg", 0.0) or 0.0)
    txn_city = payload.get("txn_city", "")
    txn_country = payload.get("txn_country", "")
    home_city = payload.get("home_city", "")
    home_country = payload.get("home_country", "")

    vpn_detected = bool(payload.get("vpn_detected", False))
    tor_exit_node = bool(payload.get("tor_exit_node", False))
    cloud_host_ip = bool(payload.get("cloud_host_ip", False))

    existing_benef = bool(payload.get("existing_beneficiary", False))
    new_benef = bool(payload.get("new_beneficiary", False))

    reasons.append(
        f"**ML Fraud Risk Score:** {fraud_score_1_100:.1f}/100 (probability={fraud_prob:.6f}) → **{ml_label}**."
    )
    reasons.append(
        f"**ML Anomaly Risk Score:** {anomaly_score_1_100:.1f}/100 (raw anomaly={anomaly_raw:.6f})."
    )
    reasons.append(
        f"**Final Risk Level:** **{final_risk}** (max of ML label and Rules severity)."
    )
    reasons.append(
        f"Context: channel **{channel_val}**, type **{txn_type}**, amount **{amt:.2f} {currency}**, time **{hour}:00**."
    )

    if monthly_avg > 0 and amt > 2 * monthly_avg:
        reasons.append(f"Amount is significantly higher than customer's monthly average of ~{monthly_avg:.2f} {currency}.")

    if home_country and txn_country and home_country.lower() != txn_country.lower():
        reasons.append(f"Home country (**{home_country}**) differs from transaction country (**{txn_country}**).")
    if home_city and txn_city and home_city.lower() != txn_city.lower():
        reasons.append(f"Home city (**{home_city}**) differs from transaction city (**{txn_city}**).")

    if 0 <= hour <= 5:
        reasons.append("Transaction occurred during unusual hours (midnight–early morning).")

    if vpn_detected:
        reasons.append("VPN usage flagged — reduces traceability and increases risk.")
    if tor_exit_node:
        reasons.append("TOR exit node detected — strongly associated with anonymized/high-risk activity.")
    if cloud_host_ip:
        reasons.append("IP resembles a cloud-hosting provider (AWS/GCP/Azure) rather than a consumer ISP.")

    if new_benef:
        reasons.append("Transfer to a newly added beneficiary — common fraud pattern.")
    elif existing_benef:
        reasons.append("Transfer to an existing beneficiary — typically lower risk than a new payee.")

    if rules_triggered:
        top_rules = sorted(rules_triggered, key=lambda r: SEVERITY_ORDER[r["severity"]], reverse=True)[:3]
        for r in top_rules:
            reasons.append(f"Key rule fired: **{r['name']}** — {r['detail']}")

    return reasons


# -------------------------
# STREAMLIT UI
# -------------------------
st.set_page_config(page_title="AI Powered Real-Time Fraud Detection", page_icon="💳", layout="centered")
st.title("💳 AI Powered Real-Time Fraud Detection (Optimized Scoring)")

st.caption(
    "ML scores are human-readable (1–100) and monotonic: higher risk always means a higher score. "
    "ML inputs are separated from Rules/Telemetry inputs."
)

# TOP: Currency/Amount/Date/Time (Amount impacts ML; date/time drive hour/day_of_week/month)
col1, col2, col3, col4 = st.columns(4)
with col1:
    currency = st.selectbox("Select currency", CURRENCIES, index=CURRENCIES.index(DEFAULT_CURRENCY))
with col2:
    amount = st.number_input("Transaction amount", min_value=0.0, value=1200.0, step=10.0)
with col3:
    txn_date = st.date_input("Transaction date", value=datetime.date.today())
with col4:
    txn_time = st.time_input("Transaction time", value=datetime.time(12, 0))

txn_dt = datetime.datetime.combine(txn_date, txn_time)
hour = txn_dt.hour
day_of_week = txn_dt.weekday()
month = txn_dt.month

st.markdown("---")
channel_display = st.selectbox("Transaction Channel", ["Choose..."] + list(CHANNEL_DISPLAY_TO_CANONICAL.keys()))
if not channel_display or channel_display == "Choose...":
    st.info("Select currency/amount/date/time, then choose a channel to show channel-specific inputs.")
    st.stop()

canonical_channel = CHANNEL_DISPLAY_TO_CANONICAL[channel_display]
channel_lower = canonical_channel.lower()

st.markdown(f"### Channel: {channel_display}")
txn_options = CHANNEL_TXN_TYPES.get(channel_lower, ["OTHER"])
txn_type = st.selectbox("Transaction type", txn_options)

# Tabs: ML Inputs vs Rule Inputs
tab_ml, tab_rules = st.tabs(["🤖 ML Inputs (affect ML score)", "📏 Rules & Telemetry (affect rules)"])

# -------------------------
# ML INPUTS TAB (ONLY THINGS USED BY MODEL)
# -------------------------
with tab_ml:
    st.subheader("ML feature inputs (only these change ML score)")
    st.caption("Model uses: Amount, TransactionType, Location, DeviceID, Channel, hour, day_of_week, month.")

    c1, c2 = st.columns(2)
    with c1:
        txn_city_ml = st.text_input(
            "Location (Transaction city) — ML feature",
            help="Used as model's Location feature.",
            value="Mumbai" if canonical_channel != "Other" else "Unknown",
        )
    with c2:
        device_id_ml = st.text_input(
            "Device / Terminal ID — ML feature",
            help="Used as model's DeviceID feature.",
            value="WEB-CHROME",
        )

    st.info(
        f"Derived ML time-features from date/time: hour={hour}, day_of_week={day_of_week}, month={month}."
    )

# -------------------------
# RULES TAB (TELEMETRY + CHANNEL DETAILS)
# -------------------------
# We keep your rule logic & inputs, but move them here to avoid confusion.
with tab_rules:
    st.subheader("Rules & Telemetry inputs (do NOT change ML score)")
    st.caption("These fields affect rule triggers and explanations only.")

    st.markdown("#### Beneficiary flags (Mobile App & NetBanking only)")
    existing_beneficiary = False
    new_beneficiary = False
    if channel_lower in ("mobile app", "netbanking"):
        beneficiary_type = st.radio(
            "Beneficiary type (if applicable)",
            ["Not a beneficiary", "Existing / known beneficiary", "Newly added beneficiary"],
            index=0,
        )
        existing_beneficiary = beneficiary_type == "Existing / known beneficiary"
        new_beneficiary = beneficiary_type == "Newly added beneficiary"
    else:
        st.caption("Not applicable for this channel.")

    st.markdown("#### Transaction type details")

    transfer_fields: Dict = {}
    payment_fields: Dict = {}
    billpay_fields: Dict = {}

    if str(txn_type).upper() == "TRANSFER":
        st.subheader("TRANSFER — source and destination account details")
        col_f1, col_f2 = st.columns(2)
        with col_f1:
            from_account_number = st.text_input("From account number")
            from_account_holder_name = st.text_input("From account holder name")
        with col_f2:
            to_account_number = st.text_input("To account number")
            to_account_holder_name = st.text_input("To account holder name")

        beneficiary_added_minutes = st.number_input("Minutes since beneficiary added (if applicable)", min_value=0, value=9999, step=1)
        reason = st.text_area("Reason / notes (optional)")

        transfer_fields.update(
            {
                "from_account_number": from_account_number,
                "from_account_holder_name": from_account_holder_name,
                "to_account_number": to_account_number,
                "to_account_holder_name": to_account_holder_name,
                "beneficiary_added_minutes": int(beneficiary_added_minutes),
                "reason": reason,
            }
        )

    if str(txn_type).upper() == "PAYMENT":
        st.subheader("PAYMENT — merchant / payment details")
        payment_category = st.selectbox("Payment category", ["ecommerce", "utilities", "subscription", "pos", "other"])
        merchant_id = st.text_input("Merchant name / ID")

        card_used = True if channel_lower in ("credit card", "debit card") else st.checkbox("Paid using card?", value=False)
        card_masked = ""
        cvv_provided = True

        if card_used and channel_lower not in ("credit card", "debit card"):
            card_masked = st.text_input("Card masked (e.g., 4111****1111)")
            cvv_provided = st.checkbox("CVV provided (checked if present)", value=True)

        payment_fields.update(
            {
                "payment_category": payment_category,
                "merchant_id": merchant_id,
                "card_used": bool(card_used),
                "card_masked": card_masked,
                "cvv_provided": bool(cvv_provided),
            }
        )

    if str(txn_type).upper() == "BILL_PAY":
        st.subheader("BILL PAY — structured biller details")
        biller_category = st.selectbox("Biller category", ["electricity", "water", "telecom", "internet", "insurance", "other"])
        biller_id = st.text_input("Biller ID")
        bill_reference_number = st.text_input("Bill/reference number")
        due_date = st.date_input("Bill due date (optional)", value=None)
        bill_period_start = st.date_input("Bill period start (optional)", value=None)
        bill_period_end = st.date_input("Bill period end (optional)", value=None)

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
    st.markdown("#### Optional telemetry (helps rules; provide if available)")
    colT1, colT2 = st.columns(2)
    with colT1:
        monthly_avg = st.number_input(f"Customer monthly average spend ({currency})", min_value=0.0, value=10000.0, step=100.0)
        rolling_avg_7d = st.number_input(f"7-day rolling average ({currency})", min_value=0.0, value=3000.0, step=50.0)
        txns_last_1h = st.number_input("Transactions in last 1 hour", min_value=0, value=0, step=1)
        txns_last_24h = st.number_input("Transactions in last 24 hours", min_value=0, value=0, step=1)
    with colT2:
        txns_last_7d = st.number_input("Transactions in last 7 days", min_value=0, value=7, step=1)
        beneficiaries_added_24h = st.number_input("Beneficiaries added in last 24h", min_value=0, value=0, step=1)
        failed_login_attempts = st.number_input("Failed login attempts (recent)", min_value=0, value=0, step=1)

    st.markdown("#### IP / Geo")
    home_city = st.text_input("Customer home city", value="Bangalore")
    home_country = st.text_input("Customer home country", value="India")
    txn_country = st.text_input("Transaction country (txn_country)", value="India")

    if channel_lower in ("bank", "atm"):
        st.info("Bank/ATM: client IP is not collected by design for in-branch and ATM flows.")
        client_ip = ""
        suspicious_ip_flag = False
    else:
        client_ip = st.text_input("Client IP (optional)")
        suspicious_ip_flag = st.checkbox("IP flagged by threat intel?", value=False)

    txn_location_ip = st.text_input("Transaction origin IP (txn_location_ip) (optional)")

    st.markdown("#### VPN / Anonymization (if known)")
    vpn_detected = st.checkbox("VPN detected?", value=False)
    vpn_provider = st.text_input("VPN provider (optional)")
    tor_exit_node = st.checkbox("TOR exit node?", value=False)
    cloud_host_ip = st.checkbox("Cloud hosting IP?", value=False)
    ip_risk_score = st.slider("IP reputation risk score (0–100)", min_value=0, max_value=100, value=0, step=1)

    st.markdown("#### Location coordinates (optional)")
    last_known_lat = st.number_input("Last known latitude (optional)", format="%.6f", value=0.0)
    last_known_lon = st.number_input("Last known longitude (optional)", format="%.6f", value=0.0)
    txn_lat = st.number_input("Transaction latitude (optional)", format="%.6f", value=0.0)
    txn_lon = st.number_input("Transaction longitude (optional)", format="%.6f", value=0.0)

    last_known_lat = last_known_lat if last_known_lat != 0.0 else None
    last_known_lon = last_known_lon if last_known_lon != 0.0 else None
    txn_lat = txn_lat if txn_lat != 0.0 else None
    txn_lon = txn_lon if txn_lon != 0.0 else None

    st.markdown("#### Channel-specific fields")
    bank_fields: Dict = {}
    atm_fields: Dict = {}
    pos_fields: Dict = {}
    online_fields: Dict = {}
    netbanking_fields: Dict = {}
    upi_fields: Dict = {}
    cc_fields: Dict = {}
    debit_fields: Dict = {}
    mobile_fields: Dict = {}
    other_fields: Dict = {}

    if channel_lower == "bank":
        st.subheader("Onsite Branch (Bank) — Identity & branch details")
        id_type = st.selectbox("ID Document Type", ["", "Aadhaar Card", "Passport", "Driver License", "Government ID", "Other"])
        id_number = st.text_input("ID Document Number")
        branch = st.text_input("Branch Name / Code", value="")
        teller_id = st.text_input("Teller ID (optional)", value="")
        bank_fields.update({"id_type": id_type, "id_number": id_number, "branch": branch, "teller_id": teller_id})

    if channel_lower == "atm":
        st.subheader("ATM fields — card + ATM info (no IP/device rules inputs)")
        atm_id = st.text_input("ATM ID / Terminal")
        atm_location = st.text_input("ATM Location (free text)")
        atm_distance_km = st.number_input("ATM distance from last known location (km)", min_value=0.0, value=0.0, step=1.0)
        card_masked_atm = st.text_input("Card masked (e.g., 4111****1111)")
        atm_fields.update({"atm_id": atm_id, "atm_location": atm_location, "atm_distance_km": atm_distance_km, "card_masked": card_masked_atm})

    if channel_lower == "pos":
        st.subheader("POS fields")
        pos_merchant_id = st.text_input("POS Merchant ID")
        store_name = st.text_input("Store name")
        pos_repeat_count = st.number_input("Rapid repeat transactions at same POS", min_value=0, value=0, step=1)
        pos_fields.update({"pos_merchant_id": pos_merchant_id, "store_name": store_name, "pos_repeat_count": pos_repeat_count})

    if channel_lower == "online purchase":
        st.subheader("Online Purchase fields (addresses, card checks)")
        merchant = st.text_input("Merchant name / ID")
        shipping_address = st.text_input("Shipping address")
        billing_address = st.text_input("Billing address", value=shipping_address)
        used_card_online = st.checkbox("Paid by card online?", value=False)
        card_masked_online = ""
        cvv_provided_online = True
        if used_card_online:
            cvv_provided_online = st.checkbox("CVV provided (checked if present)", value=True)
            card_masked_online = st.text_input("Card masked")
        online_fields.update(
            {
                "merchant": merchant,
                "shipping_address": shipping_address,
                "billing_address": billing_address,
                "card_masked": card_masked_online,
                "cvv_provided": cvv_provided_online,
            }
        )

    if channel_lower == "netbanking":
        st.subheader("NetBanking fields")
        username = st.text_input("User ID / Login")
        device_last_seen = st.text_input("Last known device (optional)")
        beneficiary = st.text_input("Beneficiary (if transfer)")
        netbanking_fields.update({"username": username, "device_last_seen": device_last_seen, "beneficiary": beneficiary})

    if channel_lower == "upi":
        st.subheader("UPI fields")
        upi_id = st.text_input("UPI ID (e.g., username@bank)")
        merchant_upi = st.text_input("Merchant / Recipient UPI ID")
        upi_fields.update({"upi_id": upi_id, "merchant_upi": merchant_upi})

    if channel_lower == "mobile app":
        st.subheader("Mobile App fields")
        device_last_seen = st.text_input("Last known device (optional)")
        mobile_fields.update({"device_last_seen": device_last_seen})

    if channel_lower == "credit card":
        st.subheader("Credit Card fields")
        card_country = st.text_input("Card issuing country")
        cvv_provided_cc = st.checkbox("CVV provided", value=True)
        cc_fields.update({"card_country": card_country, "cvv_provided": cvv_provided_cc})

    if channel_lower == "debit card":
        st.subheader("Debit Card fields")
        card_country_dc = st.text_input("Card issuing country")
        cvv_provided_dc = st.checkbox("CVV provided", value=True)
        debit_fields.update({"card_country": card_country_dc, "cvv_provided": cvv_provided_dc})

    if channel_lower == "other":
        st.subheader("Other channel fields")
        origin_description = st.text_input("Origin / Channel description")
        other_fields.update({"origin_description": origin_description})

# -------------------------
# SUBMIT / SCORING
# -------------------------
st.markdown("---")
submit = st.button("🚀 Run Fraud Check")

if submit:
    start_time = time.perf_counter()

    # --- ML payload (ONLY 8 FEATURES) ---
    ml_payload: Dict = {
        "Amount": amount,
        "TransactionType": txn_type,
        "Location": txn_city_ml,
        "DeviceID": device_id_ml,
        "Channel": canonical_channel,
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
    }

    # --- Rules payload (everything else; can include ML fields for convenience) ---
    rules_payload: Dict = {
        "Amount": amount,
        "Currency": currency,
        "TransactionType": txn_type,
        "Channel": canonical_channel,
        "DeviceID": device_id_ml,  # used for device rules
        "hour": hour,
        "day_of_week": day_of_week,
        "month": month,
        "txn_city": txn_city_ml,
        "monthly_avg": float(locals().get("monthly_avg", 0.0) or 0.0),
        "rolling_avg_7d": float(locals().get("rolling_avg_7d", 0.0) or 0.0),
        "txns_last_1h": int(locals().get("txns_last_1h", 0) or 0),
        "txns_last_24h": int(locals().get("txns_last_24h", 0) or 0),
        "txns_last_7d": int(locals().get("txns_last_7d", 0) or 0),
        "beneficiaries_added_24h": int(locals().get("beneficiaries_added_24h", 0) or 0),
        "failed_login_attempts": int(locals().get("failed_login_attempts", 0) or 0),
        "client_ip": locals().get("client_ip", ""),
        "txn_location_ip": locals().get("txn_location_ip", ""),
        "txn_country": locals().get("txn_country", ""),
        "home_city": locals().get("home_city", ""),
        "home_country": locals().get("home_country", ""),
        "declared_country": locals().get("home_country", ""),  # treat KYC/home as declared
        "suspicious_ip_flag": bool(locals().get("suspicious_ip_flag", False)),
        "last_known_lat": locals().get("last_known_lat", None),
        "last_known_lon": locals().get("last_known_lon", None),
        "txn_lat": locals().get("txn_lat", None),
        "txn_lon": locals().get("txn_lon", None),
        "vpn_detected": bool(locals().get("vpn_detected", False)),
        "vpn_provider": locals().get("vpn_provider", ""),
        "tor_exit_node": bool(locals().get("tor_exit_node", False)),
        "cloud_host_ip": bool(locals().get("cloud_host_ip", False)),
        "ip_risk_score": int(locals().get("ip_risk_score", 0) or 0),
        "existing_beneficiary": bool(locals().get("existing_beneficiary", False)),
        "new_beneficiary": bool(locals().get("new_beneficiary", False)),
    }

    # Attach txn-type specific rule fields
    rules_payload.update(locals().get("transfer_fields", {}))
    rules_payload.update(locals().get("payment_fields", {}))
    rules_payload.update(locals().get("billpay_fields", {}))

    # Attach channel-specific rule fields
    rules_payload.update(locals().get("bank_fields", {}))
    rules_payload.update(locals().get("atm_fields", {}))
    rules_payload.update(locals().get("pos_fields", {}))
    rules_payload.update(locals().get("online_fields", {}))
    rules_payload.update(locals().get("netbanking_fields", {}))
    rules_payload.update(locals().get("upi_fields", {}))
    rules_payload.update(locals().get("cc_fields", {}))
    rules_payload.update(locals().get("debit_fields", {}))
    rules_payload.update(locals().get("mobile_fields", {}))
    rules_payload.update(locals().get("other_fields", {}))

    with st.spinner("Scoring with ML models..."):
        fraud_prob, anomaly_raw, fraud_score_1_100, anomaly_score_1_100, ml_label, anomaly_label = score_transaction_ml(
            supervised_pipeline,
            iforest_pipeline,
            ml_payload,
            currency=currency,
            convert_amount_to_inr=True,  # ✅ make ML stable across currencies
        )

    rules_triggered, rules_highest = evaluate_rules(rules_payload, currency)
    final_risk = combine_final_risk(ml_label, rules_highest)

    end_time = time.perf_counter()
    response_time_s = end_time - start_time

    # ---------------- Results UI ----------------
    st.markdown("## 🔎 Results")

    color_map = {"LOW": "#2e7d32", "MEDIUM": "#f9a825", "HIGH": "#f57c00", "CRITICAL": "#c62828"}
    badge_color = color_map.get(final_risk, "#607d8b")

    st.markdown(
        f"""
        <div style="padding:0.75rem 1rem;border-radius:0.5rem;background-color:{badge_color}22;border:1px solid {badge_color};">
            <strong style="color:{badge_color};font-size:1.1rem;">Final Risk Level: {final_risk}</strong>
        </div>
        """,
        unsafe_allow_html=True,
    )

    colA, colB, colC = st.columns(3)
    with colA:
        st.metric("ML Fraud Risk Score (1–100)", f"{fraud_score_1_100:.1f}")
        st.metric("ML Fraud Label", ml_label)
    with colB:
        st.metric("ML Anomaly Risk Score (1–100)", f"{anomaly_score_1_100:.1f}")
        st.metric("Rules Highest Severity", rules_highest)
    with colC:
        st.metric("Response time (seconds)", f"{response_time_s:.3f}")

    st.markdown("### 🧠 ML & Rules Justification")
    explanation_bullets = build_explanation(
        rules_payload,
        fraud_prob,
        anomaly_raw,
        fraud_score_1_100,
        anomaly_score_1_100,
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

    st.markdown("### 🔧 Debug (optional)")
    with st.expander("Show ML payload (only fields used by ML)"):
        st.json(ml_payload)
    with st.expander("Show rules payload (fields used by rules)"):
        st.json(rules_payload)
