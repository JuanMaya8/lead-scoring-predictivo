"""Tab 3 - Commercial prioritization: tiers, effort vs. result and capacity planning."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from app.data import DashboardData
from app.ui import components, theme
from src import config
from src.scoring import (COL_PRIORITY, PRIORITY_ORDER, PRIORITY_TIERS, gain_curve, plan_capacity,
                         summarize_by_priority)


def _score_range(index: int) -> str:
    tier = PRIORITY_TIERS[index]
    if index == 0:
        return f"≥ {tier.min_score}"
    upper = PRIORITY_TIERS[index - 1].min_score - 1
    return f"{tier.min_score} – {upper}" if tier.min_score > 0 else f"< {upper + 1}"


def _tier_cards(summary: pd.DataFrame) -> None:
    stats = summary.set_index(COL_PRIORITY)
    for index, (column, tier) in enumerate(zip(st.columns(len(PRIORITY_TIERS)), PRIORITY_TIERS)):
        with column:
            components.tier_card(
                tier.label, _score_range(index), tier.action, tier.sla,
                leads=int(stats.loc[tier.label, "leads"]),
                conversion=float(stats.loc[tier.label, "tasa_conversion"]),
            )


def _conversion_by_tier(summary: pd.DataFrame) -> None:
    fig = px.bar(summary, x=COL_PRIORITY, y="tasa_conversion", color=COL_PRIORITY,
                 color_discrete_map=theme.PRIORITY_COLORS, category_orders={COL_PRIORITY: PRIORITY_ORDER},
                 text=summary["tasa_conversion"].map("{:.0%}".format),
                 title="Tasa de conversión real por prioridad")
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(showlegend=False, yaxis_tickformat=".0%", yaxis_range=[0, 1.08],
                      yaxis_title="Conversión", xaxis_title="")
    st.plotly_chart(fig, use_container_width=True)


def _effort_vs_result(summary: pd.DataFrame) -> None:
    fig = go.Figure()
    fig.add_bar(x=summary[COL_PRIORITY], y=summary["porc_leads"], name="% de leads (esfuerzo)",
                marker_color=theme.PRIMARY, hovertemplate="%{x}<br>%{y:.1%} de los leads<extra></extra>")
    fig.add_bar(x=summary[COL_PRIORITY], y=summary["porc_conversiones"], name="% de conversiones (resultado)",
                marker_color=theme.ACCENT, hovertemplate="%{x}<br>%{y:.1%} de las conversiones<extra></extra>")
    fig.update_layout(barmode="group", title="Esfuerzo vs. resultado por prioridad",
                      yaxis_tickformat=".0%", bargroupgap=0.08)
    st.plotly_chart(fig, use_container_width=True)


def _gain_chart(leads: pd.DataFrame) -> None:
    curve = gain_curve(leads)
    # Downsample for a lighter chart; the shape is unchanged.
    step = max(len(curve) // 400, 1)
    curve = pd.concat([curve.iloc[::step], curve.tail(1)])
    fig = go.Figure()
    fig.add_scatter(x=curve["porc_contactado"], y=curve["porc_capturado"], mode="lines", name="Con el modelo",
                    line=dict(color=theme.PRIMARY, width=2.5), fill="tozeroy", fillcolor="rgba(42,120,214,0.08)",
                    hovertemplate="Contactando el %{x:.0%} → capturo el %{y:.0%}<extra></extra>")
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Sin modelo (al azar)",
                    line=dict(color=theme.TEXT_MUTED, dash="dash", width=1.5), hoverinfo="skip")
    fig.update_layout(title="Curva de ganancia acumulada", xaxis_title="% de leads contactados (de mayor a menor score)",
                      yaxis_title="% de conversiones capturadas", xaxis_tickformat=".0%", yaxis_tickformat=".0%",
                      hovermode="x", height=420)
    st.plotly_chart(fig, use_container_width=True)


def _capacity_planner(leads: pd.DataFrame) -> None:
    components.section(
        "Planificador de capacidad comercial",
        "Si el equipo solo puede contactar N leads, ¿cuántas conversiones obtiene siguiendo el ranking "
        "frente a llamar al azar?",
    )
    max_capacity = len(leads)
    if max_capacity < 2:
        st.info("Se necesitan al menos 2 leads en el filtro para planificar la capacidad.")
        return
    default = min(max(max_capacity // 5, 1), max_capacity)
    capacity = st.slider("Leads que el equipo puede contactar", 1, max_capacity, default,
                         step=max(max_capacity // 200, 1))
    plan = plan_capacity(leads, capacity)
    components.kpi_row([
        ("Conversiones esperadas", f"{plan.expected_conversions:,.0f}", "siguiendo el ranking"),
        ("Si se llama al azar", f"{plan.random_conversions:,.0f}", "mismo número de llamadas"),
        ("Mejora", f"{plan.uplift:.1f}×", "más conversiones que el azar"),
        ("Tasa esperada", f"{plan.conversion_rate:.0%}", f"score mínimo contactado: {plan.min_score_reached}"),
    ])


def render(data: DashboardData, leads: pd.DataFrame) -> None:
    if leads.empty:
        st.warning("Ningún lead cumple los filtros seleccionados.")
        return

    summary = summarize_by_priority(leads)
    components.section("Sistema de priorización", "Cada nivel define una acción comercial y un tiempo de respuesta.")
    _tier_cards(summary)

    top = summary[summary[COL_PRIORITY].isin(PRIORITY_ORDER[:2])]
    top_leads, top_conv = top["leads"].sum(), top["convertidos"].sum()
    if summary["convertidos"].sum() > 0 and top_leads > 0:
        components.callout(
            f"Contactando solo los leads **A y B** ({top['porc_leads'].sum():.0%} de la base) se capturan "
            f"**{top['porc_conversiones'].sum():.0%} de las conversiones**, con una tasa de "
            f"**{top_conv / top_leads:.0%}** frente al {leads[config.TARGET].mean():.0%} de llamar a todos."
        )

    left, right = st.columns(2)
    with left:
        _conversion_by_tier(summary)
    with right:
        _effort_vs_result(summary)

    _gain_chart(leads)
    _capacity_planner(leads)
