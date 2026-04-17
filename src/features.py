import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib, os

def encode_categoricals(df, cat_cols):
    df = df.copy()
    encoders = {}
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    return df, encoders

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc  = scaler.transform(X_test)
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.pkl")
    return X_train_sc, X_test_sc, scaler

def engineer_features(df):
    df = df.copy()
    # Ratio engagement YouTube
    df["engagement_rate"] = (df["Likes"] + df["Comments"]) / (df["Views"] + 1)
    # Log-transformación de variables con sesgo
    for col in ["Views", "Likes", "Comments", "Stream"]:
        if col in df.columns:
            df[f"log_{col}"] = np.log1p(df[col])
    # Popularidad combinada
    df["popularity_score"] = (
        df["log_Stream"] * 0.6 + df["log_Views"] * 0.4
    )
    return df

def get_feature_matrix(df, config, use_log=False):
    feats = list(config["features_numericas"]) + list(config["features_categoricas"])
    
    # Solo columnas que existen
    feats = [f for f in feats if f in df.columns]
    
    # Excluir targets
    targets = [config["target_classification"], "log_Stream"]
    feats = [f for f in feats if f not in targets]
    
    return df[feats]