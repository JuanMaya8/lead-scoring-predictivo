"""Cached loading of the artifacts produced by ``python -m src.pipeline``."""
from __future__ import annotations

import json
from dataclasses import dataclass

import pandas as pd
import streamlit as st

from src import config
from src.predict import ModelArtifactError, load_artifact

REQUIRED_FILES = (config.LEADS_SCORED, config.METRICS_JSON, config.PRED_TEST_CSV, config.IMPORTANCIA_CSV)


@dataclass(frozen=True)
class DashboardData:
    """Everything the views need, loaded once per session."""

    scored: pd.DataFrame          # every historical lead with its out-of-fold score
    metrics: dict                 # reports/metrics.json
    test_predictions: pd.DataFrame  # y_real + one probability column per model
    importance: pd.DataFrame      # permutation importance of the best model
    artifact: dict | None         # production model (None if it failed to load)
    artifact_error: str = ""

    @property
    def best_model(self) -> str:
        return self.metrics["mejor_modelo"]

    @property
    def youden_threshold(self) -> float:
        return float(self.metrics["umbrales"]["youden"])


def missing_files() -> list[str]:
    """Relative paths of required artifacts that do not exist yet."""
    return [str(p.relative_to(config.RAIZ)) for p in REQUIRED_FILES if not p.exists()]


@st.cache_data(show_spinner=False)
def _read_csv(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def _read_json(path: str) -> dict:
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


@st.cache_resource(show_spinner=False)
def _load_model() -> tuple[dict | None, str]:
    try:
        return load_artifact(), ""
    except ModelArtifactError as exc:
        return None, str(exc)


def load_dashboard_data() -> DashboardData:
    artifact, error = _load_model()
    return DashboardData(
        scored=_read_csv(str(config.LEADS_SCORED)),
        metrics=_read_json(str(config.METRICS_JSON)),
        test_predictions=_read_csv(str(config.PRED_TEST_CSV)),
        importance=_read_csv(str(config.IMPORTANCIA_CSV)),
        artifact=artifact,
        artifact_error=error,
    )
