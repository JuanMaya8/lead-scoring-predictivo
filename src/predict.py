"""Production model loading and prediction on new leads.

Owner: Jack. Used by ``src/pipeline.py`` and the Streamlit dashboard.

The production artifact is a dictionary saved by the pipeline with keys:
``pipeline``, ``modelo``, ``umbral``, ``cols_eliminar``, ``columnas_entrada``,
``sklearn_version`` and ``fecha``.
"""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src import config
from src.data_prep import limpiar
from src.features import crear_variables

REQUIRED_ARTIFACT_KEYS = ("pipeline", "cols_eliminar", "columnas_entrada")


class ModelArtifactError(RuntimeError):
    """Raised when the production model cannot be loaded or is malformed."""


def prepare_features(leads: pd.DataFrame, columns_to_drop: list[str]) -> pd.DataFrame:
    """Clean + feature engineering, removing the target and identifier columns."""
    features = crear_variables(limpiar(leads, columns_to_drop))
    excluded = (config.TARGET, *config.COLS_IDENTIFICADORES)
    return features.drop(columns=[c for c in excluded if c in features.columns])


def load_artifact(path: str | Path = config.MODELO_PRODUCCION) -> dict:
    """Load the production artifact and check it has the expected structure."""
    path = Path(path)
    if not path.exists():
        raise ModelArtifactError(f"Model file not found: {path}")
    try:
        artifact = joblib.load(path)
    except Exception as exc:  # e.g. pickled with a different scikit-learn version
        raise ModelArtifactError(f"Could not load {path.name}: {exc}") from exc

    missing = [key for key in REQUIRED_ARTIFACT_KEYS if key not in artifact]
    if missing:
        raise ModelArtifactError(f"Artifact is missing keys: {missing}")
    return artifact


def align_columns(features: pd.DataFrame, expected: list[str]) -> pd.DataFrame:
    """Reorder columns to the training layout.

    Missing columns (e.g. an incomplete CSV) are created empty so the
    pipeline's imputers fill them; unexpected columns are dropped.
    """
    aligned = features.copy()
    for column in expected:
        if column not in aligned.columns:
            aligned[column] = np.nan
    return aligned[expected]


def predict_proba(artifact: dict, leads: pd.DataFrame) -> np.ndarray:
    """Conversion probability (0-1) for each row of ``leads``."""
    if leads.empty:
        return np.array([], dtype=float)
    features = prepare_features(leads, artifact["cols_eliminar"])
    features = align_columns(features, artifact["columnas_entrada"])
    return artifact["pipeline"].predict_proba(features)[:, 1]


# Backward-compatible names used by src/pipeline.py (shared file).
preparar_X = prepare_features
cargar_artefacto = load_artifact
predecir_probabilidad = predict_proba
