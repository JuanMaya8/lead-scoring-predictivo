"""Sidebar filters applied to the scored leads table."""
from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd
import streamlit as st

from src.scoring import COL_PRIORITY, COL_SCORE, PRIORITY_ORDER

# (column in leads_scored.csv, label shown in the sidebar)
CATEGORICAL_FILTERS: list[tuple[str, str]] = [
    ("Lead Origin", "Origen del lead"),
    ("Lead Source", "Fuente"),
    ("What is your current occupation", "Ocupación"),
    ("Last Activity", "Última actividad"),
]


@dataclass
class LeadFilters:
    score_range: tuple[int, int] = (0, 100)
    priorities: list[str] = field(default_factory=list)
    categories: dict[str, list[str]] = field(default_factory=dict)

    @property
    def active_count(self) -> int:
        count = len(self.priorities) > 0 or self.score_range != (0, 100)
        return int(count) + sum(bool(values) for values in self.categories.values())

    def apply(self, leads: pd.DataFrame) -> pd.DataFrame:
        low, high = self.score_range
        mask = leads[COL_SCORE].between(low, high)
        if self.priorities:
            mask &= leads[COL_PRIORITY].isin(self.priorities)
        for column, values in self.categories.items():
            if values and column in leads.columns:
                mask &= leads[column].astype(str).isin(values)
        return leads[mask]


def _options(leads: pd.DataFrame, column: str) -> list[str]:
    return sorted(leads[column].dropna().astype(str).unique())


def render_sidebar_filters(leads: pd.DataFrame) -> LeadFilters:
    """Draw the filter widgets and return the current selection."""
    st.sidebar.markdown("#### Filtros")
    filters = LeadFilters(
        priorities=st.sidebar.multiselect("Prioridad", PRIORITY_ORDER, placeholder="Todas"),
        score_range=st.sidebar.slider("Rango de Lead Score", 0, 100, (0, 100)),
    )
    for column, label in CATEGORICAL_FILTERS:
        if column in leads.columns:
            filters.categories[column] = st.sidebar.multiselect(label, _options(leads, column), placeholder="Todos")
    return filters
