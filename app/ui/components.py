"""Reusable presentational components (HTML cards rendered with Streamlit)."""
from __future__ import annotations

from html import escape

import plotly.graph_objects as go
import streamlit as st

from app.ui import theme


def hero(title: str, subtitle: str, chips: list[str] | None = None) -> None:
    """Gradient banner at the top of the page."""
    chips_html = "".join(f'<span class="ls-chip">{escape(c)}</span>' for c in chips or [])
    st.markdown(
        f'<div class="ls-hero"><h1>{escape(title)}</h1><p>{escape(subtitle)}</p>'
        f'<div class="ls-chips">{chips_html}</div></div>',
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, hint: str = "") -> None:
    """Single headline number with a caption."""
    hint_html = f'<div class="hint">{escape(hint)}</div>' if hint else ""
    st.markdown(
        f'<div class="ls-kpi"><div class="label">{escape(label)}</div>'
        f'<div class="value">{escape(value)}</div>{hint_html}</div>',
        unsafe_allow_html=True,
    )


def kpi_row(items: list[tuple[str, str, str]]) -> None:
    """Row of KPI cards from ``(label, value, hint)`` tuples."""
    for column, (label, value, hint) in zip(st.columns(len(items)), items):
        with column:
            kpi_card(label, value, hint)


def metric_strip(items: list[tuple[str, str]]) -> None:
    """Compact responsive row of ``(label, value)`` pairs for narrow columns."""
    cells = "".join(f"<div><span>{escape(label)}</span><b>{escape(value)}</b></div>" for label, value in items)
    st.markdown(f'<div class="ls-mini">{cells}</div>', unsafe_allow_html=True)


def section(title: str, description: str = "") -> None:
    """Section heading with an optional one-line explanation."""
    description_html = f"<p>{escape(description)}</p>" if description else ""
    st.markdown(f'<div class="ls-section"><h3>{escape(title)}</h3>{description_html}</div>',
                unsafe_allow_html=True)


def callout(markdown_text: str) -> None:
    """Highlighted insight box. Accepts **bold** markdown."""
    html = escape(markdown_text)
    # Minimal bold support without enabling arbitrary HTML.
    parts = html.split("**")
    html = "".join(f"<b>{p}</b>" if i % 2 else p for i, p in enumerate(parts))
    st.markdown(f'<div class="ls-callout">{html}</div>', unsafe_allow_html=True)


def tier_card(label: str, score_range: str, action: str, sla: str,
              leads: int | None = None, conversion: float | None = None) -> None:
    """Card describing one priority tier, colored with the ordinal ramp."""
    color = theme.PRIORITY_COLORS.get(label, theme.PRIMARY)
    stats = ""
    if leads is not None and conversion is not None:
        stats = (f'<div class="tier-stats"><div><b>{leads:,}</b><span>leads</span></div>'
                 f'<div><b>{conversion:.0%}</b><span>conversión real</span></div></div>')
    st.markdown(
        f'<div class="ls-tier" style="--tier-color:{color}">'
        f'<div class="tier-title">{escape(label)}</div>'
        f'<div class="tier-range">Score {escape(score_range)}</div>'
        f'<div class="tier-action">{escape(action)}</div>'
        f'<span class="tier-sla">⏱ {escape(sla)}</span>{stats}</div>',
        unsafe_allow_html=True,
    )


def score_gauge(score: int, color: str) -> go.Figure:
    """Semicircular gauge for a single lead score (0-100)."""
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number=dict(suffix=" / 100", font=dict(size=34, color=theme.TEXT_PRIMARY)),
        gauge=dict(
            axis=dict(range=[0, 100], tickvals=[0, 25, 50, 75, 100], tickcolor=theme.TEXT_MUTED),
            bar=dict(color=color, thickness=0.32),
            bgcolor=theme.SURFACE_ALT,
            borderwidth=0,
            steps=[dict(range=[0, 100], color=theme.SURFACE_ALT)],
            threshold=dict(line=dict(color=theme.TEXT_PRIMARY, width=2), thickness=0.8, value=score),
        ),
    ))
    fig.update_layout(height=230, margin=dict(l=20, r=20, t=20, b=0))
    return fig


def footer(text: str) -> None:
    st.markdown(f'<div class="ls-footer">{escape(text)}</div>', unsafe_allow_html=True)
