import os
import csv
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from flask import Flask, Blueprint, request, jsonify
from collections import deque

# =========================================================
# APP + BLUEPRINT
# =========================================================
app = Flask(__name__)
fraud_api = Blueprint("fraud_api", __name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Login model directory
MODELS_DIR = os.path.join(BASE_DIR, "user_models")

# Transaction model directory (A1)
TXN_MODELS_DIR = os.path.join(BASE_DIR, "transaction_models")
os.makedirs(TXN_MODELS_DIR, exist_ok=True)

# Login biometrics dataset
DATA_FILE = os.path.join(BASE_DIR, "user_biometrics.csv")

# Transaction biometrics dataset
TXN_DATA_FILE = os.path.join(BASE_DIR, "transaction_biometrics.csv")

FEATURE_COLS = [
    "typing_speed_wpm",
    "avg_key_hold_ms",
    "avg_key_interval_ms",
    "mouse_distance_px",
    "session_duration_s",
    "text_length",
]

TIME_STEPS = len(FEATURE_COLS)

# Separate caches for login and transaction
MODEL_CACHE = {}
MODEL_CACHE_TXN = {}

ROLLING_RISK = {}

# =========================================================
# UTILITIES
# =========================================================
def mahalanobis(x, mu, cov_inv):
    d = x - mu
    return float(np.sqrt(d @ cov_inv @ d.T))


def build_vector(b):
    vals = [
        b.get("typing_speed_wpm"),
        b.get("avg_key_hold_ms"),
        b.get("avg_key_interval_ms"),
        b.get("mouse_distance_px"),
        b.get("session_duration_s"),
        b.get("text_length"),
    ]

    if any(v is None for v in vals):
        raise ValueError("Corrupt behavioural vector")

    return np.array(vals, dtype=np.float32)


def smooth(username, risk):
    if username not in ROLLING_RISK:
        ROLLING_RISK[username] = deque(maxlen=3)
    ROLLING_RISK[username].append(risk)
    return float(np.mean(ROLLING_RISK[username]))

# =========================================================
# LOGIN MODEL LOADING
# =========================================================
def load_user(username):
    """Loads login model for a specific user."""
    if username in MODEL_CACHE:
        return MODEL_CACHE[username]

    enc = os.path.join(MODELS_DIR, f"encoder_{username}.keras")
    sc  = os.path.join(MODELS_DIR, f"scaler_{username}.pkl")
    mh  = os.path.join(MODELS_DIR, f"mahalanobis_{username}.pkl")
    fs  = os.path.join(MODELS_DIR, f"feature_stats_{username}.pkl")
    gm  = os.path.join(MODELS_DIR, "global_mahalanobis.pkl")

    if not all(map(os.path.exists, [enc, sc, mh, fs, gm])):
        return None

    encoder      = tf.keras.models.load_model(enc, compile=False)
    scaler       = joblib.load(sc)
    maha_stats   = joblib.load(mh)
    feats        = joblib.load(fs)
    global_maha  = joblib.load(gm)

    MODEL_CACHE[username] = (encoder, scaler, maha_stats, feats, global_maha)
    return MODEL_CACHE[username]

# =========================================================
# TRANSACTION MODEL LOADING
# =========================================================
def load_txn_models(owner):
    """Loads transaction models separate from login."""
    if owner in MODEL_CACHE_TXN:
        return MODEL_CACHE_TXN[owner]

    enc = os.path.join(TXN_MODELS_DIR, f"txn_encoder_{owner}.keras")
    sc  = os.path.join(TXN_MODELS_DIR, f"txn_scaler_{owner}.pkl")
    mh  = os.path.join(TXN_MODELS_DIR, f"txn_mahalanobis_{owner}.pkl")
    fs  = os.path.join(TXN_MODELS_DIR, f"txn_feature_stats_{owner}.pkl")
    gm  = os.path.join(TXN_MODELS_DIR, "txn_global_mahalanobis.pkl")

    if not all(os.path.exists(x) for x in [enc, sc, mh, fs, gm]):
        return None

    encoder      = tf.keras.models.load_model(enc, compile=False)
    scaler       = joblib.load(sc)
    maha_stats   = joblib.load(mh)
    feats        = joblib.load(fs)
    global_maha  = joblib.load(gm)

    MODEL_CACHE_TXN[owner] = (encoder, scaler, maha_stats, feats, global_maha)
    return MODEL_CACHE_TXN[owner]

# =========================================================
# HARD SECURITY GATES
# =========================================================
def security_gates(b):
    """Hand-crafted rules to block bots/hard anomalies."""
    reasons = []

    if b.get("illegal_edit_detected"):
        return False, ["Illegal backspace/enter detected"]

    if b.get("text_length", 0) < 6:
        reasons.append("Extremely short typing pattern")

    if b.get("session_duration_s", 0) < 1.0:
        reasons.append("Impossible session speed (likely bot)")

    if b.get("avg_key_hold_ms", 9999) < 35 or b.get("avg_key_hold_ms", 0) > 600:
        reasons.append("Unnatural key hold duration")

    return True, reasons

# =========================================================
# LOGIN FRAUD ENGINE
# =========================================================
def check(username, behaviour):
    """Same fraud engine for login."""
    models = load_user(username)

    # Cold start
    if models is None:
        return True, 0.25, "ALLOW", "cold_start", ["No behavioural model"], None

    ok, reasons = security_gates(behaviour)
    if not ok:
        return False, 0.99, "BLOCK", "security_gate", reasons, None

    encoder, scaler, maha_u, feats, maha_g = models

    x = build_vector(behaviour).reshape(1, -1)

    z = np.abs((x - feats["mean"]) / feats["std"])
    feature_z = float(np.clip(np.mean(z), 0, 10))

    x_scaled = scaler.transform(x)
    x_cnn = x_scaled.reshape(1, TIME_STEPS, 1)
    emb = encoder.predict(x_cnn, verbose=0)[0]

    d_user   = mahalanobis(emb, maha_u["mu"], maha_u["cov_inv"])
    d_global = mahalanobis(emb, maha_g["mu"], maha_g["cov_inv"])

    r_user   = min(d_user / maha_u["threshold"], 2.0)
    r_global = min(d_global / maha_g["threshold"], 2.0)

    risk = (
        0.65 * r_user +
        0.15 * r_global +
        0.15 * (feature_z / 6.0) +
        0.05 * r_user
    )

    smoothed = smooth(username, risk)

    if risk > 0.92:
        return False, risk, "BLOCK", "outside_human", ["Severe anomaly"], emb

    if smoothed < 0.55:
        return True, smoothed, "ALLOW", "user_space", ["Behaviour trusted"], emb
    elif smoothed < 0.75:
        return True, smoothed, "MONITOR", "cohort_space", ["Suspicious but human"], emb
    else:
        return False, smoothed, "BLOCK", "outside_human", ["Not genuine"], emb

# =========================================================
# TRANSACTION FRAUD ENGINE
# =========================================================
def check_transaction(owner, behaviour):
    """Fraud engine for transaction models."""
    models = load_txn_models(owner)

    if models is None:
        return True, 0.30, "ALLOW", "cold_start", ["No transaction model"], None

    ok, reasons = security_gates(behaviour)
    if not ok:
        return False, 0.99, "BLOCK", "security_gate", reasons, None

    encoder, scaler, maha_u, feats, maha_g = models

    x = build_vector(behaviour).reshape(1, -1)

    z = np.abs((x - feats["mean"]) / feats["std"])
    z_score = float(min(np.mean(z), 10))

    x_scaled = scaler.transform(x)
    emb = encoder.predict(x_scaled.reshape(1, TIME_STEPS, 1), verbose=0)[0]

    du = mahalanobis(emb, maha_u["mu"], maha_u["cov_inv"])
    dg = mahalanobis(emb, maha_g["mu"], maha_g["cov_inv"])

    ru = min(du / maha_u["threshold"], 2.0)
    rg = min(dg / maha_g["threshold"], 2.0)

    risk = 0.65 * ru + 0.20 * rg + 0.15 * (z_score / 6.0)
    smoothed = smooth(owner, risk)

    if risk > 0.92:
        return False, risk, "BLOCK", "outside_human", ["Severe anomaly"], emb

    if smoothed < 0.55:
        return True, smoothed, "ALLOW", "user_space", ["Behaviour trusted"], emb
    elif smoothed < 0.75:
        return True, smoothed, "MONITOR", "cohort_space", ["Suspicious"], emb

    return False, smoothed, "BLOCK", "outside_human", ["Anomalous"], emb

# =========================================================
# ADAPTIVE LEARNING (LOGIN)
# =========================================================
def adaptive_learn(username, embedding):
    """Updates LOGIN Mahalanobis centroid."""
    try:
        path = os.path.join(MODELS_DIR, f"mahalanobis_{username}.pkl")
        stats = joblib.load(path)

        mu = stats["mu"]
        alpha = 0.08
        stats["mu"] = (1 - alpha) * mu + alpha * embedding

        joblib.dump(stats, path)
    except:
        pass

# =========================================================
# ADAPTIVE LEARNING (TRANSACTION)
# =========================================================
def adaptive_learn_txn(owner, embedding):
    """Updates TRANSACTION Mahalanobis centroid."""
    try:
        path = os.path.join(TXN_MODELS_DIR, f"txn_mahalanobis_{owner}.pkl")
        stats = joblib.load(path)

        mu = stats["mu"]
        alpha = 0.08
        stats["mu"] = (1 - alpha) * mu + alpha * embedding

        joblib.dump(stats, path)
    except:
        pass

# =========================================================
# LOGIN BIOMETRIC CHECK ENDPOINTS
# =========================================================
@fraud_api.route("/api/hmog-login", methods=["POST"])
@fraud_api.route("/api/check-fraud", methods=["POST"])
def fraud_login():
    try:
        data = request.get_json()
        username = data.get("username").strip().lower()
        behaviour = data.get("behaviour")

        allowed, risk, decision, space, reasons, emb = check(username, behaviour)

        if allowed and decision == "ALLOW" and risk < 0.55 and emb is not None:
            adaptive_learn(username, emb)

        return jsonify({
            "allowed": allowed,
            "riskScore": round(risk, 3),
            "decision": decision,
            "space_position": space,
            "reasons": reasons,
            "model": f"encoder_{username} | {space}"
        })

    except Exception as e:
        print("LOGIN API ERROR:", e)
        return jsonify({"allowed": True, "riskScore": 0.0})

# =========================================================
# SAVE LOGIN TRAINING SAMPLES
# =========================================================
@fraud_api.route("/api/save-training-data", methods=["POST"])
def save_training_data():
    try:
        data     = request.get_json()
        username = data.get("username").strip().lower()
        b        = data.get("behaviour")

        row = {
            "subject": username,
            "typing_speed_wpm": b.get("typing_speed_wpm", 0.0),
            "avg_key_hold_ms": b.get("avg_key_hold_ms", 0.0),
            "avg_key_interval_ms": b.get("avg_key_interval_ms", 0.0),
            "mouse_distance_px": b.get("mouse_distance_px", 0.0),
            "session_duration_s": b.get("session_duration_s", 0.0),
            "text_length": b.get("text_length", 0),
        }

        exists = os.path.exists(DATA_FILE)
        with open(DATA_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not exists:
                writer.writeheader()
            writer.writerow(row)

        df = pd.read_csv(DATA_FILE)
        total = len(df[df["subject"] == username])

        return jsonify({"ok": True, "total_count": int(total)})

    except Exception as e:
        print("❌ LOGIN TRAIN SAVE ERROR:", e)
        return jsonify({"ok": False}), 500

# =========================================================
# SAVE TRANSACTION TRAINING SAMPLES
# =========================================================
@fraud_api.route("/api/save-transaction-data", methods=["POST"])
def save_txn_data():
    try:
        data  = request.get_json()
        owner = data.get("owner").strip().lower()
        b     = data.get("behaviour")

        row = {
            "owner": owner,
            "typing_speed_wpm": b.get("typing_speed_wpm", 0.0),
            "avg_key_hold_ms": b.get("avg_key_hold_ms", 0.0),
            "avg_key_interval_ms": b.get("avg_key_interval_ms", 0.0),
            "mouse_distance_px": b.get("mouse_distance_px", 0.0),
            "session_duration_s": b.get("session_duration_s", 0.0),
            "text_length": b.get("text_length", 0),
        }

        exists = os.path.exists(TXN_DATA_FILE)
        with open(TXN_DATA_FILE, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=row.keys())
            if not exists:
                writer.writeheader()
            writer.writerow(row)

        df = pd.read_csv(TXN_DATA_FILE)
        total = len(df[df["owner"] == owner])

        return jsonify({"ok": True, "total": int(total)})

    except Exception as e:
        print("❌ TRANSACTION TRAIN SAVE ERROR:", e)
        return jsonify({"ok": False}), 500

# =========================================================
# TRANSACTION FRAUD CHECK
# =========================================================
@fraud_api.route("/api/transaction-check", methods=["POST"])
def transaction_check():
    try:
        data  = request.get_json()
        owner = data.get("owner").strip().lower()
        behaviour = data.get("behaviour")

        allowed, risk, decision, space, reasons, emb = check_transaction(owner, behaviour)

        if allowed and decision == "ALLOW" and risk < 0.55 and emb is not None:
            adaptive_learn_txn(owner, emb)

        return jsonify({
            "allowed": allowed,
            "riskScore": round(risk, 3),
            "decision": decision,
            "reasons": reasons,
            "model": f"txn_encoder_{owner}"
        })

    except Exception as e:
        print("TRANSACTION API ERROR:", e)
        return jsonify({"allowed": True, "riskScore": 0.0})

# =========================================================
# REGISTER + RUN
# =========================================================
app.register_blueprint(fraud_api)

if __name__ == "__main__":
    print("🚀 Behavioural Fraud AI Running at 127.0.0.1:5001")
    app.run(host="127.0.0.1", port=5001, debug=False)
