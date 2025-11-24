"""
fraud_logic.py
Aligned 100% with main Streamlit app.py logic.
"""

import datetime
from math import radians, sin, cos, asin, sqrt
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import pandas as pd


# ===========================================================
# Currency table aligned with app.py
# ===========================================================
INR_PER_UNIT = {
    "INR": 1.0,
    "USD": 83.2,
    "EUR": 90.5,
    "GBP": 105.3,
    "AED": 22.7,
    "AUD": 61.0,
    "SGD": 61.5,
}


def inr_to_currency(amount_in_inr: float, currency: str) -> float:
    if currency not in INR_PER_UNIT or INR_PER_UNIT[currency] == 0:
        return amount_in_inr
    return amount_in_inr / INR_PER_UNIT[currency]


# ===========================================================
# Score Normalization (0–100)
# ===========================================================
def normalize_score(x: float, min_val: float = 0.0, max_val: float = 0.02) -> float:
    if x is None:
        return 0.0
    try:
        val = float(x)
    except:
        return 0.0
    if val < min_val:
        val = min_val
    if val > max_val:
        val = max_val
    if max_val == min_val:
        return 0.0
    return (val - min_val) / (max_val - min_val) * 100.0


# ===========================================================
# Distance helper
# ===========================================================
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    if None in (lat1, lon1, lat2, lon2):
        return None
    lat1, lon1, lat2, lon2 = map(radians, (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return 6371 * c


SEVERITY_ORDER = {"LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}


def escalate(a: str, b: str) -> str:
    return a if SEVERITY_ORDER[a] >= SEVERITY_ORDER[b] else b


# ===========================================================
# Load ML artifacts
# ===========================================================
def load_artifacts(models_dir: str = "models"):
    models_dir = Path(models_dir)

    def _load(name):
        return joblib.load(models_dir / name)

    try:
        supervised = _load("supervised_lgbm_pipeline.joblib")
    except:
        supervised = None

    try:
        iforest = _load("iforest_pipeline.joblib")
    except:
        iforest = None

    return supervised, iforest


# ===========================================================
# ML Thresholds
# ===========================================================
FRAUD_MED = 0.00005
FRAUD_HIGH = 0.00023328
FRAUD_CRIT = 0.01732857

ANOM_MED = 0.04
ANOM_HIGH = 0.05
ANOM_CRIT = 0.08


def ml_risk_label(fraud_prob: float, anomaly_score: float) -> str:
    if fraud_prob >= FRAUD_CRIT or anomaly_score >= ANOM_CRIT:
        return "CRITICAL"
    if fraud_prob >= FRAUD_HIGH or (fraud_prob >= FRAUD_MED and anomaly_score >= ANOM_HIGH):
        return "HIGH"
    if fraud_prob >= FRAUD_MED or anomaly_score >= ANOM_MED:
        return "MEDIUM"
    return "LOW"


# ===========================================================
# INR-based thresholds (aligned with UI)
# ===========================================================
BASE_THRESHOLDS_INR = {
    "absolute_crit_amount": 10_000_000,
    "high_amount_threshold": 2_000_000,
    "medium_amount_threshold": 100_000,
    "atm_high_withdrawal": 300_000,
    "card_test_small_amount_inr": 200,
}


# ===========================================================
# Rule Engine (EXACT COPY of Streamlit logic)
# ===========================================================
def evaluate_rules(payload: Dict, currency: str):
    ABS_CRIT = inr_to_currency(BASE_THRESHOLDS_INR["absolute_crit_amount"], currency)
    HIGH_AMT = inr_to_currency(BASE_THRESHOLDS_INR["high_amount_threshold"], currency)
    MED_AMT = inr_to_currency(BASE_THRESHOLDS_INR["medium_amount_threshold"], currency)
    ATM_HIGH = inr_to_currency(BASE_THRESHOLDS_INR["atm_high_withdrawal"], currency)
    CARD_TEST_SMALL = inr_to_currency(BASE_THRESHOLDS_INR["card_test_small_amount_inr"], currency)

    rules = []

    amt = float(payload.get("Amount", 0.0) or 0.0)
    channel = str(payload.get("Channel", "")).lower()
    hour = int(payload.get("hour", 0))
    monthly_avg = float(payload.get("monthly_avg", 0.0) or 0.0)
    rolling_avg_7d = float(payload.get("rolling_avg_7d", 0.0) or 0.0)
    txns_1h = int(payload.get("txns_last_1h", 0) or 0)
    txns_24h = int(payload.get("txns_last_24h", 0) or 0)
    new_benef = bool(payload.get("new_beneficiary", False))
    failed_logins = int(payload.get("failed_login_attempts", 0) or 0)
    beneficiaries_added_24h = int(payload.get("beneficiaries_added_24h", 0) or 0)

    # Geo fields
    ip_country = (payload.get("ip_country") or "").lower()
    declared_country = (payload.get("declared_country") or "").lower()
    home_country = (payload.get("home_country") or "").lower()
    home_city = (payload.get("home_city") or "").lower()
    txn_city = (payload.get("txn_city") or "").lower()
    txn_country = (payload.get("txn_country") or "").lower()

    last_lat = payload.get("last_known_lat")
    last_lon = payload.get("last_known_lon")
    txn_lat = payload.get("txn_lat")
    txn_lon = payload.get("txn_lon")

    atm_distance_km = float(payload.get("atm_distance_km", 0.0) or 0.0)

    card_country = (payload.get("card_country") or "").lower()
    cvv_provided = payload.get("cvv_provided", True)
    suspicious_ip_flag = payload.get("suspicious_ip_flag", False)
    card_small_attempts = int(payload.get("card_small_attempts_in_5min", 0) or 0)
    pos_repeat_count = int(payload.get("pos_repeat_count", 0) or 0)
    beneficiary_added_minutes = int(payload.get("beneficiary_added_minutes", 9999) or 9999)

    last_device = (payload.get("device_last_seen") or "").lower()
    curr_device = (payload.get("DeviceID") or "").lower()

    def add(name, sev, detail):
        rules.append({"name": name, "severity": sev, "detail": detail})

    # ---------------- CRITICAL ----------------
    if amt >= ABS_CRIT:
        add("Absolute very large amount", "CRITICAL", f"{amt:.2f} >= {ABS_CRIT:.2f}")

    # Impossible travel
    impossible_distance = haversine_km(last_lat, last_lon, txn_lat, txn_lon)

    device_checks_enabled = channel not in ("bank", "atm")
    if device_checks_enabled:
        device_new = (not last_device) or (curr_device and curr_device != last_device)
        location_changed = impossible_distance and impossible_distance > 500
        if device_new and location_changed and amt > MED_AMT:
            add("New device + Impossible travel + High amount", "CRITICAL",
                f"Travel {impossible_distance:.1f} km; amount {amt:.2f}")

    if beneficiaries_added_24h >= 3 and amt > HIGH_AMT:
        add("Multiple beneficiaries added + high transfer", "CRITICAL",
            f"{beneficiaries_added_24h} beneficiaries + amount {amt}")

    # ---------------- HIGH ----------------
    if txns_1h >= 10:
        add("High velocity (1h)", "HIGH", f"{txns_1h} txns")

    if txns_24h >= 50:
        add("Very high velocity (24h)", "HIGH", f"{txns_24h} txns")

    if ip_country and declared_country and ip_country != declared_country:
        sev = "HIGH" if amt > HIGH_AMT else "MEDIUM"
        add("IP / Declared country mismatch", sev, f"{ip_country} != {declared_country}")

    if failed_logins >= 5:
        add("Multiple failed login attempts", "HIGH", f"{failed_logins} fails")

    if new_benef and amt >= MED_AMT:
        add("New beneficiary + significant amount", "HIGH", "New beneficiary with large amount")

    if suspicious_ip_flag and amt > MED_AMT / 4:
        add("IP flagged by threat intelligence", "HIGH", "Threat intel flagged IP")

    if channel == "atm" and atm_distance_km > 300:
        add("ATM distance from last location", "HIGH", f"{atm_distance_km} km away")

    if card_country and home_country and card_country != home_country and amt > MED_AMT:
        add("Card country mismatch vs home country", "HIGH", f"{card_country} != {home_country}")

    # ---------------- MEDIUM ----------------
    if monthly_avg > 0 and amt >= 5 * monthly_avg and amt > MED_AMT:
        add("Large spike vs monthly avg", "HIGH", f"{amt} >= 5x monthly avg")
    elif rolling_avg_7d > 0 and amt >= 3 * rolling_avg_7d and amt > (MED_AMT / 2):
        add("Spike vs 7-day avg", "MEDIUM", f"{amt} >= 3x avg")
    elif monthly_avg > 0 and amt >= 2 * monthly_avg and amt > (MED_AMT / 2):
        add("Above monthly usual", "MEDIUM", f"{amt} >= 2x monthly avg")

    if 0 <= hour <= 5 and monthly_avg < MED_AMT * 2 and amt > (MED_AMT / 10):
        add("Late-night txn for low-activity customer", "MEDIUM", f"hour={hour}")

    if card_small_attempts >= 6:
        add("Card testing / micro-charges detected", "HIGH", f"{card_small_attempts} attempts")

    if channel == "atm" and amt >= ATM_HIGH:
        add("Large ATM withdrawal", "HIGH", f"{amt}")

    if pos_repeat_count >= 10:
        add("POS repeat transactions", "HIGH", f"{pos_repeat_count} repeats")

    if channel in ("bank", "netbanking") and beneficiary_added_minutes < 10 and amt >= MED_AMT:
        add("Immediate transfer to newly added beneficiary", "HIGH",
            f"added {beneficiary_added_minutes} mins ago")

    # Home vs transaction
    if home_country and txn_country and home_country != txn_country:
        sev = "HIGH" if amt >= MED_AMT else "MEDIUM"
        add("Txn country differs from home country", sev, f"{home_country} vs {txn_country}")

    if home_city and txn_city and home_city != txn_city and amt >= (MED_AMT / 2):
        add("Txn city differs from home city", "MEDIUM", f"{home_city} vs {txn_city}")

    # ---------------- FINAL ----------------
    highest = "LOW"
    for r in rules:
        highest = escalate(highest, r["severity"])

    return rules, highest


# ===========================================================
# ML Scoring (aligned with UI)
# ===========================================================
def score_transaction_ml(supervised_pipeline, iforest_pipeline, payload):
    model_df = pd.DataFrame([{
        "Amount": payload.get("Amount", 0.0),
        "TransactionType": payload.get("TransactionType", "PAYMENT"),
        "Location": payload.get("txn_city", "Unknown"),
        "DeviceID": payload.get("DeviceID", "Unknown"),
        "Channel": payload.get("Channel", "Other"),
        "hour": payload.get("hour", 0),
        "day_of_week": payload.get("day_of_week", 0),
        "month": payload.get("month", 0),
    }])

    # supervised model
    try:
        fraud_prob = float(supervised_pipeline.predict_proba(model_df)[0, 1])
    except:
        fraud_prob = 0.0

    # isolation forest
    try:
        anomaly_raw = -float(iforest_pipeline.decision_function(model_df)[0])
    except:
        anomaly_raw = 0.0

    label = ml_risk_label(fraud_prob, anomaly_raw)
    return fraud_prob, anomaly_raw, label


# ===========================================================
# Combine ML + Rule risk
# ===========================================================
def combine_final_risk(ml_risk, rule_risk):
    return escalate(ml_risk, rule_risk)
