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
    print(f" Duplicados eliminados: {n_before - len(df)}")
    
    # Eliminar filas con target nulo
    df.dropna(subset=[config["target_regression"]], inplace=True)
    
    # Imputar numéricos con mediana
    for col in config["features_numericas"]:
        if col in df.columns and df[col].isnull().any():
            df[col].fillna(df[col].median(), inplace=True)
    
    # Imputar categóricos con moda
    for col in config["features_categoricas"]:
        if col in df.columns and df[col].isnull().any():
            df[col].fillna(df[col].mode()[0], inplace=True)
    
    # Crear variable objetivo de clasificación
    threshold = config["viral_threshold"]
    df[config["target_classification"]] = (df[config["target_regression"]] >= threshold).astype(int)
    print(f" Canciones virales (>= {threshold:,} streams): {df[config['target_classification']].sum()}")
    
    return df

def save_processed(df, config):
    os.makedirs("data/processed", exist_ok=True)
    df.to_csv(config["data"]["processed_path"], index=False)
    print(f" Datos guardados en {config['data']['processed_path']}")
