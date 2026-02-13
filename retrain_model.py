import os
import numpy as np
import pandas as pd
import joblib
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from keras.layers import Input, SeparableConv1D, GlobalAveragePooling1D, Dense, Dropout
from keras.callbacks import EarlyStopping
from keras import Model
import tensorflow as tf


# =============================================
# CONFIG
# =============================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "user_biometrics.csv")
MODELS_DIR = os.path.join(BASE_DIR, "user_models")
os.makedirs(MODELS_DIR, exist_ok=True)

FEATURE_COLS = [
    "typing_speed_wpm",
    "avg_key_hold_ms",
    "avg_key_interval_ms",
    "mouse_distance_px",
    "session_duration_s",
    "text_length",
]

TIME_STEPS = len(FEATURE_COLS)
EMBED_DIM = 32

# =============================================
# CNN ENCODER
# =============================================
def build_cnn_encoder():
    inputs = Input(shape=(TIME_STEPS, 1))
    x = SeparableConv1D(32, 3, activation="relu")(inputs)
    x = SeparableConv1D(64, 3, activation="relu")(x)
    x = GlobalAveragePooling1D()(x)
    x = Dense(EMBED_DIM, activation="relu")(x)
    x = Dropout(0.25)(x)
    return Model(inputs, x)


# =============================================
# MAHALANOBIS UTILS
# =============================================
def compute_stats(embeddings):
    mu = np.mean(embeddings, axis=0)
    cov = np.cov(embeddings, rowvar=False)
    cov += np.eye(cov.shape[0]) * 0.03
    cov_inv = np.linalg.inv(cov)
    distances = np.sqrt(np.sum((embeddings - mu) @ cov_inv * (embeddings - mu), axis=1))
    return mu, cov_inv, distances


# =============================================
# TRAIN USER MODEL
# =============================================
def train_user(user, df):
    df_user = df[df["subject"] == user]

    if len(df_user) < 8:
        print(f"⚠️ Skipping {user} — Not enough samples")
        return

    X = df_user[FEATURE_COLS].values.astype(float)

    feature_stats = {
        "mean": np.mean(X, axis=0),
        "std": np.std(X, axis=0) + 1e-6
    }

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_cnn = X_scaled.reshape(-1, TIME_STEPS, 1)

    encoder = build_cnn_encoder()
    decoder = Dense(TIME_STEPS)(encoder.output)
    autoencoder = Model(encoder.input, decoder)

    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.fit(
        X_cnn,
        X_scaled,
        epochs=60,
        batch_size=8,
        verbose=0,
        callbacks=[EarlyStopping(monitor="loss", patience=8, restore_best_weights=True)]
    )

    embeddings = encoder.predict(X_cnn, verbose=0)

    mu_user, cov_user, d_user = compute_stats(embeddings)

    variability = np.std(d_user)
    base = np.mean(d_user)

    if len(d_user) < 15:
        threshold_user = np.percentile(d_user, 97)
    else:
        alpha = 2.0 if variability < 0.25 else 3.0
        threshold_user = base + alpha * variability

    print(f"Trained {user} | threshold={threshold_user:.3f}")

    encoder.save(os.path.join(MODELS_DIR, f"encoder_{user}.keras"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, f"scaler_{user}.pkl"))
    joblib.dump(
        {"mu": mu_user, "cov_inv": cov_user, "threshold": float(threshold_user)},
        os.path.join(MODELS_DIR, f"mahalanobis_{user}.pkl")
    )
    joblib.dump(feature_stats, os.path.join(MODELS_DIR, f"feature_stats_{user}.pkl"))


# =============================================
# GLOBAL HUMAN COHORT MODEL
# =============================================
def train_global(df):
    X = df[FEATURE_COLS].values.astype(float)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_cnn = X_scaled.reshape(-1, TIME_STEPS, 1)

    encoder = build_cnn_encoder()
    decoder = Dense(TIME_STEPS)(encoder.output)
    autoencoder = Model(encoder.input, decoder)

    autoencoder.compile(optimizer="adam", loss="mse")
    autoencoder.fit(X_cnn, X_scaled, epochs=40, batch_size=10, verbose=0)

    embeddings = encoder.predict(X_cnn, verbose=0)

    mu_g, cov_g, d_g = compute_stats(embeddings)
    threshold_g = np.percentile(d_g, 98)

    joblib.dump({"mu": mu_g, "cov_inv": cov_g, "threshold": float(threshold_g)},
                os.path.join(MODELS_DIR, "global_mahalanobis.pkl"))

    print(f"🌍 Global cohort trained | threshold={threshold_g:.3f}")


# =============================================
# MAIN TRAIN
# =============================================
def retrain_all():
    df = pd.read_csv(DATA_FILE)
    df["subject"] = df["subject"].astype(str).str.strip()

    unique_users = df["subject"].unique()
    print("Training users:", unique_users)

    for u in unique_users:
        train_user(u, df)

    train_global(df)
    print("🎯 Training complete")


if __name__ == "__main__":
    retrain_all()
