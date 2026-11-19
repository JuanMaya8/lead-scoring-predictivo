"""Tab 5 - Which variables drive conversion."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from app.data import DashboardData
from app.ui import components, theme
from src import config
from src.scoring import COL_SCORE

TOP_FEATURES = 15
MIN_GROUP_SIZE = 15
EXPLORABLE = ["Last Activity", "Lead Source", "Lead Origin", "What is your current occupation",
              "Specialization", "City"]


def _importance_chart(data: DashboardData) -> None:
    top = data.importance.head(TOP_FEATURES).iloc[::-1]
    fig = px.bar(top, x="importancia", y="variable", orientation="h", error_x="desviacion",
                 title=f"Importancia por permutación · {data.best_model}",
                 labels={"importancia": "Caída del ROC-AUC al desordenar la variable", "variable": ""})
    fig.update_traces(marker_color=theme.PRIMARY, error_x=dict(color=theme.TEXT_MUTED, thickness=1))
    fig.update_layout(height=540)
    st.plotly_chart(fig, use_container_width=True)


def _conversion_explorer(data: DashboardData, leads: pd.DataFrame) -> None:
    options = [c for c in EXPLORABLE if c in data.scored.columns]
    variable = st.selectbox("Explorar conversión por variable", options)
    grouped = (leads.groupby(variable)
               .agg(leads=(COL_SCORE, "size"), conversion=(config.TARGET, "mean"), score=(COL_SCORE, "mean"))
               .query("leads >= @MIN_GROUP_SIZE").sort_values("conversion").reset_index())
    if grouped.empty:
        st.info("No hay suficientes leads con los filtros actuales.")
        return
    overall = leads[config.TARGET].mean()
    fig = px.bar(grouped, x="conversion", y=variable, orientation="h", color="score",
                 color_continuous_scale=theme.SEQUENTIAL_SCALE, range_color=(0, 100),
                 hover_data={"leads": ":,", "conversion": ":.1%", "score": ":.0f"},
                 labels={"conversion": "Tasa de conversión real", variable: "", "score": "Score prom."},
                 title=f"Conversión real por «{variable}»")
    fig.add_vline(x=overall, line_dash="dash", line_color=theme.TEXT_MUTED,
                  annotation_text=f"Promedio {overall:.0%}", annotation_position="top")
    fig.update_layout(xaxis_tickformat=".0%", height=max(360, 30 * len(grouped) + 100),
                      coloraxis_colorbar=dict(thickness=10, title=""))
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Solo se muestran categorías con al menos {MIN_GROUP_SIZE} leads.")


def render(data: DashboardData, leads: pd.DataFrame) -> None:
    components.section(
        "¿Qué variables influyen más en la conversión?",
        "Importancia por permutación: cuánto empeora el modelo si se desordena cada variable.",
    )
    left, right = st.columns([1, 1])
    with left:
        _importance_chart(data)
    with right:
        _conversion_explorer(data, leads)
