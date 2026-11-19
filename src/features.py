"""Feature engineering y preprocesamiento.

Responsable: Juan David.

- `crear_variables`: nuevas variables derivadas (fila a fila).
- `Winsorizador`: recorta valores atípicos en los percentiles 1-99
  aprendidos SOLO del conjunto de entrenamiento.
- `construir_preprocesador`: imputación + escalado + one-hot encoding
  dentro de un ColumnTransformer, para que todo quede en un único
  Pipeline guardable con joblib (lo usa el dashboard).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src import config

# Agrupación de fuentes de tráfico en canales de marketing
CANALES = {
    "Google": "Búsqueda",
    "Organic Search": "Búsqueda",
    "Bing": "Búsqueda",
    "Direct Traffic": "Directo",
    "Olark Chat": "Chat",
    "Reference": "Referidos",
    "Referral Sites": "Referidos",
    "Welingak Website": "Referidos",
    "Facebook": "Redes sociales",
    "Social Media": "Redes sociales",
}

ACTIVIDADES_ALTO_INTERES = {
    "SMS Sent",
    "Had a Phone Conversation",
    "Email Link Clicked",
    "Form Submitted on Website",
    "Approached upfront",
}


def crear_variables(df: pd.DataFrame) -> pd.DataFrame:
    """Variables nuevas a partir de las originales (no usan el target)."""
    df = df.copy()
    visitas = df.get("TotalVisits")
    tiempo = df.get("Total Time Spent on Website")
    paginas = df.get("Page Views Per Visit")

    if visitas is not None and tiempo is not None:
        # Tiempo promedio por visita (segundos). Si no hubo visitas -> 0
        df["fe_tiempo_por_visita"] = np.where(visitas > 0, tiempo / visitas.where(visitas > 0), 0.0)
        # Índice de engagement: combina frecuencia y profundidad de navegación
        df["fe_engagement"] = np.log1p(tiempo) * np.log1p(visitas)
        df["fe_tiene_visitas"] = (visitas > 0).astype(float).where(visitas.notna())
    if visitas is not None and paginas is not None:
        df["fe_paginas_totales"] = visitas * paginas
    if tiempo is not None:
        df["fe_log_tiempo"] = np.log1p(tiempo)

    if "What is your current occupation" in df.columns:
        df["fe_profesional"] = (df["What is your current occupation"] == "Working Professional").astype(int)
    if "Lead Source" in df.columns:
        df["fe_canal"] = df["Lead Source"].map(CANALES).fillna("Otro")
    if "Last Activity" in df.columns:
        df["fe_actividad_alto_interes"] = df["Last Activity"].isin(ACTIVIDADES_ALTO_INTERES).astype(int)
    return df


class Winsorizador(BaseEstimator, TransformerMixin):
    """Recorta cada columna numérica entre los percentiles [inferior, superior]
    calculados en el entrenamiento (tratamiento de outliers sin borrar filas)."""

    def __init__(self, inferior: float = 0.01, superior: float = 0.99):
        self.inferior = inferior
        self.superior = superior

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lim_inf_ = np.nanquantile(X, self.inferior, axis=0)
        self.lim_sup_ = np.nanquantile(X, self.superior, axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lim_inf_, self.lim_sup_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def tipos_de_columnas(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Separa columnas numéricas y categóricas (excluye target e IDs)."""
    excluir = {config.TARGET, *config.COLS_IDENTIFICADORES}
    cols = [c for c in df.columns if c not in excluir]
    numericas = [c for c in cols if pd.api.types.is_numeric_dtype(df[c])]
    categoricas = [c for c in cols if c not in numericas]
    return numericas, categoricas


def construir_preprocesador(numericas: list[str], categoricas: list[str]) -> ColumnTransformer:
    """Imputación + winsorización + escalado (numéricas) y one-hot (categóricas).

    - Numéricas: mediana (robusta a outliers) -> recorte p1-p99 -> estandarización z.
      El escalado es indispensable para KNN, regresión logística y la red neuronal.
    - Categóricas: 'Desconocido' para nulos -> one-hot. Las categorías con
      menos de 1% de frecuencia se agrupan como 'infrequent' (evita columnas
      casi vacías y categorías nuevas no vistas).
    """
    pipe_num = Pipeline(
        [
            ("imputar", SimpleImputer(strategy="median")),
            ("outliers", Winsorizador()),
            ("escalar", StandardScaler()),
        ]
    )
    pipe_cat = Pipeline(
        [
            ("imputar", SimpleImputer(strategy="constant", fill_value="Desconocido")),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=0.01,
                    sparse_output=False,
                ),
            ),
        ]
    )
    return ColumnTransformer(
        [("num", pipe_num, numericas), ("cat", pipe_cat, categoricas)],
        verbose_feature_names_out=False,
    )
