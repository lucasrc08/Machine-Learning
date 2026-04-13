import numpy as np
import pandas as pd
import joblib, os
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (mean_squared_error, r2_score, mean_absolute_error,
                             accuracy_score, f1_score, roc_auc_score,
                             classification_report, confusion_matrix)
from sklearn.model_selection import cross_val_score, GridSearchCV
from xgboost import XGBClassifier, XGBRegressor

# ─── REGRESIÓN ────────────────────────────────────────────────────
def train_regression_models(X_train, y_train, random_state=42):
    models = {
        "Ridge":         Ridge(),
        "RandomForest":  RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1),
        "XGBoost":       XGBRegressor(n_estimators=100, random_state=random_state,
                                      verbosity=0, n_jobs=-1),
    }
    trained = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        trained[name] = m
        print(f"{name} entrenado")
    return trained

def evaluate_regression(models, X_test, y_test):
    results = []
    for name, m in models.items():
        y_pred = m.predict(X_test)
        results.append({
            "Modelo": name,
            "R²":     round(r2_score(y_test, y_pred), 4),
            "RMSE":   round(np.sqrt(mean_squared_error(y_test, y_pred)), 4),
            "MAE":    round(mean_absolute_error(y_test, y_pred), 4),
        })
    return pd.DataFrame(results).sort_values("R²", ascending=False)

# ─── CLASIFICACIÓN ────────────────────────────────────────────────
def train_classification_models(X_train, y_train, random_state=42):
    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=random_state),
        "RandomForest":       RandomForestClassifier(n_estimators=100, random_state=random_state, n_jobs=-1),
        "XGBoost":            XGBClassifier(n_estimators=100, random_state=random_state,
                                            verbosity=0, n_jobs=-1, eval_metric="logloss"),
    }
    trained = {}
    for name, m in models.items():
        m.fit(X_train, y_train)
        trained[name] = m
        print(f"{name} entrenado")
    return trained

def evaluate_classification(models, X_test, y_test):
    results = []
    for name, m in models.items():
        y_pred  = m.predict(X_test)
        y_proba = m.predict_proba(X_test)[:, 1] if hasattr(m, "predict_proba") else None
        results.append({
            "Modelo":   name,
            "Accuracy": round(accuracy_score(y_test, y_pred), 4),
            "F1-Score": round(f1_score(y_test, y_pred), 4),
            "ROC-AUC":  round(roc_auc_score(y_test, y_proba), 4) if y_proba is not None else "N/A",
        })
    return pd.DataFrame(results).sort_values("F1-Score", ascending=False)

def save_model(model, name):
    os.makedirs("models", exist_ok=True)
    path = f"models/{name}.pkl"
    joblib.dump(model, path)
    print(f"Modelo guardado: {path}")