"""Ranking de leads y sistema de priorización comercial.

Responsable: Jack.

Lead Score = probabilidad de conversión x 100 (0 a 100).
El ranking ordena los leads de mayor a menor score; la prioridad asigna
un nivel A/B/C/D con una acción comercial y un tiempo de respuesta (SLA).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from src import config


def asignar_prioridad(score: float) -> str:
    for minimo, etiqueta, _, _ in config.NIVELES_PRIORIDAD:
        if score >= minimo:
            return etiqueta
    return config.NIVELES_PRIORIDAD[-1][1]


def accion_por_prioridad() -> pd.DataFrame:
    return pd.DataFrame(
        [(e, m, a, s) for m, e, a, s in config.NIVELES_PRIORIDAD],
        columns=["prioridad", "score_minimo", "accion_recomendada", "sla"],
    )


def puntuar_leads(df_original: pd.DataFrame, probabilidades) -> pd.DataFrame:
    """Agrega score, ranking, prioridad y acción a una tabla de leads."""
    out = df_original.copy()
    out["probabilidad"] = np.asarray(probabilidades, dtype=float)
    out["lead_score"] = (out["probabilidad"] * 100).round().astype(int)
    out["prioridad"] = out["lead_score"].apply(asignar_prioridad)
    acciones = {e: a for _, e, a, _ in config.NIVELES_PRIORIDAD}
    slas = {e: s for _, e, _, s in config.NIVELES_PRIORIDAD}
    out["accion_recomendada"] = out["prioridad"].map(acciones)
    out["sla"] = out["prioridad"].map(slas)
    out = out.sort_values("probabilidad", ascending=False).reset_index(drop=True)
    out.insert(0, "ranking", np.arange(1, len(out) + 1))
    return out


def resumen_por_prioridad(scored: pd.DataFrame) -> pd.DataFrame:
    """Leads y conversión real por nivel (requiere la columna Converted)."""
    orden = [e for _, e, _, _ in config.NIVELES_PRIORIDAD]
    g = scored.groupby("prioridad").agg(
        leads=("lead_score", "size"),
        score_promedio=("lead_score", "mean"),
        convertidos=(config.TARGET, "sum"),
    )
    g = g.reindex(orden).fillna(0)
    g["tasa_conversion"] = np.where(g["leads"] > 0, g["convertidos"] / g["leads"].clip(lower=1), 0)
    g["porc_leads"] = g["leads"] / g["leads"].sum()
    g["porc_conversiones"] = g["convertidos"] / max(g["convertidos"].sum(), 1)
    return g.reset_index()
