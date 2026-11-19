"""Tab 2 - Ranked list of leads ready to hand to the sales team."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.data import DashboardData
from app.ui import components
from src import config
from src.scoring import COL_ACTION, COL_PRIORITY, COL_RANK, COL_SCORE, COL_SLA

TABLE_COLUMNS = [
    COL_RANK, "Lead Number", COL_SCORE, COL_PRIORITY, COL_ACTION, COL_SLA, "Lead Origin", "Lead Source",
    "Last Activity", "What is your current occupation", "TotalVisits", "Total Time Spent on Website",
    config.TARGET,
]
PAGE_SIZES = [25, 50, 100, 250, 500, 1000]


def _search(leads: pd.DataFrame, query: str) -> pd.DataFrame:
    query = query.strip()
    if not query:
        return leads
    return leads[leads["Lead Number"].astype(str).str.contains(query, regex=False)]


def render(data: DashboardData, leads: pd.DataFrame) -> None:
    components.section(
        "Ranking de leads por probabilidad de conversión",
        "Ordenados de mayor a menor score. Los scores son out-of-fold: cada lead fue puntuado por un "
        "modelo que no lo vio durante el entrenamiento.",
    )
    search_col, size_col = st.columns([3, 1])
    query = search_col.text_input("Buscar por número de lead", placeholder="Ej. 660737")
    page_size = size_col.selectbox("Filas a mostrar", PAGE_SIZES, index=2)

    result = _search(leads, query)
    view = result[[c for c in TABLE_COLUMNS if c in result.columns]].head(page_size)
    st.dataframe(
        view, use_container_width=True, hide_index=True, height=540,
        column_config={
            COL_RANK: st.column_config.NumberColumn("#", width="small"),
            "Lead Number": st.column_config.TextColumn("Lead"),
            COL_SCORE: st.column_config.ProgressColumn("Lead Score", min_value=0, max_value=100, format="%d"),
            COL_PRIORITY: "Prioridad",
            COL_ACTION: st.column_config.TextColumn("Acción recomendada", width="large"),
            COL_SLA: "SLA",
            "What is your current occupation": "Ocupación",
            "TotalVisits": st.column_config.NumberColumn("Visitas", format="%d"),
            "Total Time Spent on Website": st.column_config.NumberColumn("Tiempo en web (s)", format="%d"),
            config.TARGET: st.column_config.CheckboxColumn("¿Convirtió?"),
        },
    )
    st.caption(f"Mostrando {len(view):,} de {len(result):,} leads.")
    st.download_button(
        "⬇️ Descargar ranking filtrado (CSV)", result.to_csv(index=False).encode("utf-8"),
        file_name="ranking_leads.csv", mime="text/csv", type="primary",
    )
