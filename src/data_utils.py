import pandas as pd
import numpy as np
import yaml
import os

def load_config(path="config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_raw_data(config):
    df = pd.read_csv(config["data"]["raw_path"], index_col=0)
    print(f"Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    return df

def get_missing_summary(df):
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    return pd.DataFrame({
        "Missing": missing,
        "Porcentaje (%)": missing_pct.round(2)
    }).query("Missing > 0").sort_values("Porcentaje (%)", ascending=False)

def clean_data(df, config):
    df = df.copy()
    
    # Eliminar duplicados
    n_before = len(df)
    df.drop_duplicates(subset=["Artist", "Track"], keep="first", inplace=True)
    print(f"Duplicados eliminados: {n_before - len(df)}")
    
    # Eliminar filas sin Stream (necesario para crear viral)
    df.dropna(subset=["Stream"], inplace=True)
    
    # Imputar numéricos con mediana
    for col in config["features_numericas"]:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].median())
    
    # Imputar categóricos con moda
    for col in config["features_categoricas"]:
        if col in df.columns and df[col].isnull().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    
    # Crear variable objetivo
    threshold = config["viral_threshold"]
    df[config["target_classification"]] = (df["Stream"] >= threshold).astype(int)
    
    print(f"Canciones virales (>= {threshold:,}): {df[config['target_classification']].sum()}")
    print(df[config["target_classification"]].value_counts(normalize=True))
    
    return df

def save_processed(df, config):
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(config["data"]["processed_path"], index=False)
    print(f" Datos guardados en {config['data']['processed_path']}")
