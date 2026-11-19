"""Lead ranking and commercial prioritization system.

Owner: Jack.

* **Lead Score** = conversion probability x 100, an integer from 0 to 100.
* **Ranking** sorts leads from the highest to the lowest probability.
* **Priority** maps the score to a tier (A/B/C/D) with a recommended sales
  action and a response-time target (SLA).

The tiers are defined in ``config.NIVELES_PRIORIDAD`` so the business rules
live in a single place. Output column names (``lead_score``, ``prioridad``...)
are part of the contract with the pipeline and the dashboard, so they stay
in Spanish.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from src import config

# Output column names (shared contract with src/pipeline.py and the dashboard).
COL_RANK = "ranking"
COL_PROBABILITY = "probabilidad"
COL_SCORE = "lead_score"
COL_PRIORITY = "prioridad"
COL_ACTION = "accion_recomendada"
COL_SLA = "sla"


@dataclass(frozen=True)
class PriorityTier:
    """One level of the prioritization system."""

    min_score: int
    label: str
    action: str
    sla: str

    @property
    def code(self) -> str:
        """Single-letter tier code, e.g. ``"A"``."""
        return self.label.split(" ")[0]


# Sorted from the highest to the lowest threshold.
PRIORITY_TIERS: tuple[PriorityTier, ...] = tuple(
    sorted(
        (PriorityTier(*row) for row in config.NIVELES_PRIORIDAD),
        key=lambda tier: tier.min_score,
        reverse=True,
    )
)
PRIORITY_ORDER: list[str] = [tier.label for tier in PRIORITY_TIERS]


def assign_priority(score: float) -> str:
    """Return the tier label for a 0-100 score."""
    for tier in PRIORITY_TIERS:
        if score >= tier.min_score:
            return tier.label
    return PRIORITY_TIERS[-1].label


def priority_table() -> pd.DataFrame:
    """Prioritization rules as a table: tier, minimum score, action and SLA."""
    return pd.DataFrame(
        [(t.label, t.min_score, t.action, t.sla) for t in PRIORITY_TIERS],
        columns=[COL_PRIORITY, "score_minimo", COL_ACTION, COL_SLA],
    )


def score_leads(leads: pd.DataFrame, probabilities: Iterable[float]) -> pd.DataFrame:
    """Add probability, score, priority, action, SLA and ranking to ``leads``.

    The result is sorted by probability (best leads first) and the ``ranking``
    column is inserted as the first column, starting at 1.
    """
    scored = leads.copy()
    scored[COL_PROBABILITY] = np.asarray(list(probabilities), dtype=float)
    scored[COL_SCORE] = (scored[COL_PROBABILITY] * 100).round().astype(int)
    scored[COL_PRIORITY] = scored[COL_SCORE].map(assign_priority)

    by_label = {tier.label: tier for tier in PRIORITY_TIERS}
    scored[COL_ACTION] = scored[COL_PRIORITY].map(lambda label: by_label[label].action)
    scored[COL_SLA] = scored[COL_PRIORITY].map(lambda label: by_label[label].sla)

    scored = scored.sort_values(COL_PROBABILITY, ascending=False, kind="mergesort").reset_index(drop=True)
    scored.insert(0, COL_RANK, np.arange(1, len(scored) + 1))
    return scored


def summarize_by_priority(scored: pd.DataFrame, target: str = config.TARGET) -> pd.DataFrame:
    """Leads, average score and real conversion per tier.

    Requires the ground-truth column ``target`` (historical leads only).
    """
    summary = (
        scored.groupby(COL_PRIORITY)
        .agg(leads=(COL_SCORE, "size"), score_promedio=(COL_SCORE, "mean"), convertidos=(target, "sum"))
        .reindex(PRIORITY_ORDER)
        .fillna(0)
    )
    total_leads = max(summary["leads"].sum(), 1)
    total_conversions = max(summary["convertidos"].sum(), 1)
    summary["tasa_conversion"] = summary["convertidos"] / summary["leads"].clip(lower=1)
    summary["porc_leads"] = summary["leads"] / total_leads
    summary["porc_conversiones"] = summary["convertidos"] / total_conversions
    return summary.reset_index()


# Backward-compatible names used by src/pipeline.py (shared file).
puntuar_leads = score_leads
resumen_por_prioridad = summarize_by_priority
asignar_prioridad = assign_priority
accion_por_prioridad = priority_table
