"""Carga del modelo de producción y predicción sobre leads nuevos.

Responsable: Jack (lo usa el dashboard).
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from src import config
from src.data_prep import limpiar
from src.features import crear_variables


def preparar_X(df: pd.DataFrame, cols_eliminar: list[str]) -> pd.DataFrame:
    """Limpieza + feature engineering; quita target e identificadores."""
    X = crear_variables(limpiar(df, cols_eliminar))
    return X.drop(columns=[c for c in (config.TARGET, *config.COLS_IDENTIFICADORES) if c in X.columns])


def cargar_artefacto(ruta=config.MODELO_PRODUCCION) -> dict:
    """Diccionario con: pipeline, modelo, umbral, cols_eliminar, columnas_entrada."""
    return joblib.load(ruta)


def predecir_probabilidad(artefacto: dict, df: pd.DataFrame):
    X = preparar_X(df, artefacto["cols_eliminar"])
    # Columnas faltantes (p. ej. un CSV incompleto) se crean vacías: el imputador las rellena
    for c in artefacto["columnas_entrada"]:
        if c not in X.columns:
            X[c] = np.nan
    X = X[artefacto["columnas_entrada"]]
    return artefacto["pipeline"].predict_proba(X)[:, 1]
