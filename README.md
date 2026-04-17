#  Predicción de Popularidad Musical: Spotify & YouTube

## Descripción del Problema

En la industria musical actual, los artistas y sellos discográficos necesitan anticipar qué canciones tendrán mayor éxito en plataformas de streaming y redes sociales. Este proyecto aplica técnicas de aprendizaje de máquina sobre un dataset combinado de Spotify y YouTube para **predecir el número de reproducciones (Streams) en Spotify** a partir de características acústicas y métricas de YouTube.

La pregunta central es: **¿Las características sonoras de una canción y su desempeño en YouTube permiten predecir su popularidad en Spotify?**

---

## Objetivos

- Analizar y preparar un dataset de 20.718 canciones con características de audio (Spotify) y métricas de engagement (YouTube).
- Construir y comparar modelos de regresión (Regresión Lineal, Random Forest, Gradient Boosting) para predecir el número de Streams.
- Evaluar los modelos con métricas estándar (RMSE, MAE, R²) e interpretar los resultados.
- Comunicar los hallazgos a través de un dashboard interactivo en Power BI / Tableau.

---

## Metodología: CRISP-DM

```
1. Comprensión del Negocio  →  notebooks/01_business_understanding.ipynb
2. Comprensión de los Datos →  notebooks/02_data_understanding.ipynb
3. Preparación de los Datos →  notebooks/03_data_preparation.ipynb
4. Modelado                 →  notebooks/04_modeling.ipynb
5. Evaluación               →  notebooks/05_evaluation.ipynb
6. Despliegue / Dashboard   →  reports/
```

---

## Estructura del Repositorio

```
Machine-Learning/
│
├── config/
│   └── config.yaml              # Parámetros globales del proyecto
│
├── data/
│   ├── raw/                     # Datos originales sin modificar
│   │   └── Datos_proyecto_C1_Spotify_Youtube.csv
│   └── processed/               # Datos limpios y transformados (generados)
│
├── models/                      # Modelos entrenados guardados (.pkl)
│
├── notebooks/
│   ├── 01_business_understanding.ipynb
│   ├── 02_data_understanding.ipynb
│   ├── 03_data_preparation.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_evaluation.ipynb
│
├── reports/
│   ├── figures/                 # Gráficas generadas
│   └── dashboard_guide.md       # Guía del dashboard en Power BI / Tableau
│
├── src/
│   ├── __init__.py
│   ├── data_preprocessing.py    # Funciones de limpieza y transformación
│   ├── feature_engineering.py   # Ingeniería de características
│   ├── train.py                 # Script de entrenamiento de modelos
│   └── evaluate.py              # Métricas y evaluación
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

## Dataset

- **Fuente:** Datos combinados de Spotify Web API y YouTube Data API
- **Registros:** 20.718 canciones
- **Variables:** 28 columnas (características acústicas, métricas de YouTube, metadatos)
- **Variable objetivo:** `Stream` (reproducciones en Spotify)

---

## Autores

Proyecto Final — Máquina de Aprendizaje 1  
Universidad de La Sabana · Prof. Jesús Antonio Villarraga P.
Juan Portocarrero, lucas Reales, Juan peña