"""Carga y limpieza del dataset Leads.csv (X Education).

Responsable: Juan David.

Todas las operaciones de `limpiar` son fila a fila (no dependen del resto
del dataset), así que la misma función sirve para entrenar y para puntuar
un lead nuevo en el dashboard sin fuga de información.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config

# Normalización de categorías escritas de varias formas en el dataset
MAPEO_LEAD_SOURCE = {
    "google": "Google",
    "Facebook": "Facebook",
    "bing": "Bing",
    "blog": "Blog",
    "welearnblog_Home": "WeLearn",
    "youtubechannel": "Social Media",
    "Pay per Click Ads": "Google",
    "Live Chat": "Olark Chat",
}


def cargar_datos(ruta=config.DATA_RAW) -> pd.DataFrame:
    """Lee el CSV original."""
    df = pd.read_csv(ruta)
    print(f"[datos] {df.shape[0]:,} filas x {df.shape[1]} columnas cargadas")
    return df


def columnas_a_eliminar(df: pd.DataFrame) -> list[str]:
    """Decide qué columnas eliminar usando SOLO el conjunto de entrenamiento.

    Reglas:
      1. Variables con fuga de información (llenadas por ventas).
      2. Más de MAX_PORC_NULOS de valores nulos (tras tratar 'Select').
      3. Una sola categoría ocupa más de MAX_PORC_CLASE_DOMINANTE
         (varianza casi nula, no aportan información).
    """
    df = df.replace("Select", np.nan)
    eliminar = [c for c in config.COLS_FUGA_INFORMACION if c in df.columns]
    for col in df.columns:
        if col in (config.TARGET, *config.COLS_IDENTIFICADORES) or col in eliminar:
            continue
        if col in config.COLS_NUMERICAS_BASE:
            continue
        porc_nulos = df[col].isna().mean()
        porc_dominante = df[col].value_counts(normalize=True, dropna=True).max()
        if porc_nulos > config.MAX_PORC_NULOS or porc_dominante > config.MAX_PORC_CLASE_DOMINANTE:
            eliminar.append(col)
    return eliminar


def limpiar(df: pd.DataFrame, cols_eliminar: list[str] | None = None) -> pd.DataFrame:
    """Limpieza fila a fila. Devuelve una copia."""
    df = df.copy()
    # 'Select' es el valor por defecto del formulario web = el usuario no eligió nada
    df = df.replace("Select", np.nan)

    if cols_eliminar:
        df = df.drop(columns=[c for c in cols_eliminar if c in df.columns])

    if "Lead Source" in df.columns:
        df["Lead Source"] = df["Lead Source"].replace(MAPEO_LEAD_SOURCE)

    for col in config.COLS_BINARIAS:
        if col in df.columns:
            df[col] = df[col].map({"Yes": 1, "No": 0})

    for col in config.COLS_NUMERICAS_BASE:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def resumen_calidad(df: pd.DataFrame) -> pd.DataFrame:
    """Tabla de calidad de datos: tipo, % nulos, # categorías (para el informe)."""
    tmp = df.replace("Select", np.nan)
    return (
        pd.DataFrame(
            {
                "tipo": tmp.dtypes.astype(str),
                "porc_nulos": (tmp.isna().mean() * 100).round(2),
                "n_valores_unicos": tmp.nunique(),
            }
        )
        .sort_values("porc_nulos", ascending=False)
    )
