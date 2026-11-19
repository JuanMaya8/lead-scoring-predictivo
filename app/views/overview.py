"""Tab 1 - Executive overview of the filtered leads."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app.data import DashboardData
from app.ui import components, theme
from src import config
from src.scoring import COL_PRIORITY, COL_PROBABILITY, COL_SCORE, PRIORITY_ORDER

MIN_GROUP_SIZE = 20  # hide tiny groups whose conversion rate is just noise


def _kpis(leads: pd.DataFrame, total: int) -> None:
    has_rows = len(leads) > 0
    hot = int((leads[COL_PRIORITY] == PRIORITY_ORDER[0]).sum())
    components.kpi_row([
        ("Leads", f"{len(leads):,}", f"de {total:,} en la base"),
        ("Conversión histórica", f"{leads[config.TARGET].mean():.1%}" if has_rows else "–", "tasa real observada"),
        ("Score promedio", f"{leads[COL_SCORE].mean():.0f}" if has_rows else "–", "escala 0 a 100"),
        ("Leads calientes (A)", f"{hot:,}", f"{hot / max(len(leads), 1):.0%} del filtro"),
        ("Conversiones esperadas", f"{leads[COL_PROBABILITY].sum():,.0f}", "suma de probabilidades"),
    ])


def _score_distribution(leads: pd.DataFrame) -> None:
    fig = px.histogram(
        leads, x=COL_SCORE, color=COL_PRIORITY, nbins=40,
        color_discrete_map=theme.PRIORITY_COLORS, category_orders={COL_PRIORITY: PRIORITY_ORDER},
        title="Distribución del Lead Score",
    )
    fig.update_layout(bargap=0.08, xaxis_title="Lead Score", yaxis_title="Leads", legend_title_text="")
    st.plotly_chart(fig, use_container_width=True)


def _conversion_by_band(leads: pd.DataFrame) -> None:
    bands = pd.cut(leads[COL_SCORE], bins=range(0, 101, 10), include_lowest=True)
    grouped = (leads.groupby(bands, observed=True)[config.TARGET]
               .agg(conversion="mean", leads="size").reset_index())
    grouped["band"] = grouped[COL_SCORE].map(lambda i: f"{int(i.left) + (i.left > 0)}–{int(i.right)}")
    fig = px.bar(grouped, x="band", y="conversion", text=grouped["conversion"].map("{:.0%}".format),
                 hover_data={"leads": ":,", "band": False},
                 title="Conversión real por rango de score",
                 labels={"conversion": "Tasa de conversión", "band": "Lead Score"})
    fig.update_traces(marker_color=theme.PRIMARY, textposition="outside", cliponaxis=False)
    fig.update_yaxes(tickformat=".0%", range=[0, 1.08])
    st.plotly_chart(fig, use_container_width=True)


def _score_by_group(leads: pd.DataFrame, column: str, title: str) -> None:
    grouped = (leads.groupby(column)
               .agg(leads=(COL_SCORE, "size"), score=(COL_SCORE, "mean"), conversion=(config.TARGET, "mean"))
               .query("leads >= @MIN_GROUP_SIZE").sort_values("score").reset_index())
    if grouped.empty:
        st.info(f"No hay grupos con al menos {MIN_GROUP_SIZE} leads para «{column}» con los filtros actuales.")
        return
    fig = px.bar(grouped, x="score", y=column, orientation="h", color="conversion",
                 color_continuous_scale=theme.SEQUENTIAL_SCALE, range_color=(0, 1),
                 hover_data={"leads": ":,", "conversion": ":.1%", "score": ":.1f"},
                 title=title, labels={"score": "Score promedio", column: "", "conversion": "Conversión"})
    fig.update_layout(coloraxis_colorbar=dict(tickformat=".0%", thickness=10, title=""),
                      height=max(320, 34 * len(grouped) + 90))
    st.plotly_chart(fig, use_container_width=True)


def render(data: DashboardData, leads: pd.DataFrame) -> None:
    _kpis(leads, total=len(data.scored))
    if leads.empty:
        st.warning("Ningún lead cumple los filtros seleccionados.")
        return

    left, right = st.columns(2)
    with left:
        _score_distribution(leads)
    with right:
        _conversion_by_band(leads)
    components.callout(
        "El score está **bien ordenado**: a mayor score, mayor conversión real. "
        "Eso es lo que permite priorizar llamadas con confianza."
    )

    left, right = st.columns(2)
    with left:
        _score_by_group(leads, "Lead Source", "Score promedio por fuente (color = conversión real)")
    with right:
        _score_by_group(leads, "Lead Origin", "Score promedio por origen (color = conversión real)")
