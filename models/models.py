import numpy as np
import pandas as pd
import joblib
import os

from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, r2_score, mean_absolute_error,
    accuracy_score, precision_score, recall_score, f1_score
)

from xgboost import XGBRegressor


# ─────────────────────────────────────────────
# REGRESIÓN
# ─────────────────────────────────────────────
def train_regression_models(X_train, y_train, random_state=42):
    models = {
        "Ridge": Ridge(),
        "RandomForest": RandomForestRegressor(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=100,
            random_state=random_state,
            verbosity=0,
            n_jobs=-1
        ),
    }

    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model

    return trained


def evaluate_regression(models, X_test, y_test):
    results = []

    for name, model in models.items():
        y_pred = model.predict(X_test)

        results.append({
            "Modelo": name,
            "R2": round(r2_score(y_test, y_pred), 4),
            "RMSE": round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
            "MAE": round(mean_absolute_error(y_test, y_pred), 4),
        })

    return pd.DataFrame(results).sort_values("R2", ascending=False)


# ─────────────────────────────────────────────
# CLASIFICACIÓN
# ─────────────────────────────────────────────
def train_classification_models(X_train, y_train, random_state=42):
    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            random_state=random_state
        ),
        "DecisionTree": DecisionTreeClassifier(
            random_state=random_state
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            random_state=random_state,
            n_jobs=-1
        ),
    }

    trained = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained[name] = model

    return trained


def evaluate_classification(models, X_test, y_test):
    results = []

    for name, model in models.items():
        y_pred = model.predict(X_test)

        results.append({
            "Modelo": name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "Precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred, zero_division=0), 4),
            "F1": round(f1_score(y_test, y_pred, zero_division=0), 4),
        })

    return pd.DataFrame(results).sort_values("F1", ascending=False)


# ─────────────────────────────────────────────
# GUARDADO DE MODELOS
# ─────────────────────────────────────────────
def save_model_local(model, name):
    os.makedirs("models", exist_ok=True)
    path = f"models/{name}.pkl"
    joblib.dump(model, path)
    return path