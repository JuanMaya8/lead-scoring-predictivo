"""Visual identity of the dashboard: color tokens, global CSS and Plotly template.

Colors were validated for color-vision deficiency and contrast:
  * Priority tiers are ordinal (A > B > C > D), so they use a single-hue blue
    ramp from dark to light instead of a rainbow of unrelated colors.
  * Two-series comparisons use blue vs. orange.
  * Green/red are reserved for status messages (good/bad), never for series.
"""
from __future__ import annotations

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ---------------------------------------------------------------------------
# Design tokens
# ---------------------------------------------------------------------------
SURFACE = "#fcfcfb"
SURFACE_ALT = "#f4f6fa"
BORDER = "#e3e7ee"
TEXT_PRIMARY = "#0f1b2d"
TEXT_SECONDARY = "#52607a"
TEXT_MUTED = "#8a94a6"
GRID = "#eceff4"

PRIMARY = "#2a78d6"
PRIMARY_DARK = "#104281"
ACCENT = "#eb6834"          # second series (comparisons)
NEUTRAL_SERIES = "#b4bccb"  # de-emphasized series (e.g. non-selected models)

STATUS_GOOD = "#0ca30c"
STATUS_CRITICAL = "#d03b3b"

# Ordinal ramp for priority tiers: darkest = hottest lead.
PRIORITY_COLORS: dict[str, str] = {
    "A - Caliente": "#104281",
    "B - Tibio": "#256abf",
    "C - Templado": "#5598e7",
    "D - Frío": "#86b6ef",
}

# Sequential single-hue scale for continuous magnitudes (heatmaps, gradients).
SEQUENTIAL_SCALE = [
    [0.0, "#e8f1fd"],
    [0.25, "#9ec5f4"],
    [0.5, "#5598e7"],
    [0.75, "#256abf"],
    [1.0, "#0d366b"],
]

FONT_FAMILY = "Inter, 'Segoe UI', system-ui, -apple-system, sans-serif"


def _register_plotly_template() -> None:
    """Register and activate a clean Plotly template shared by every chart."""
    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(family=FONT_FAMILY, color=TEXT_PRIMARY, size=13),
        title=dict(font=dict(size=15, color=TEXT_PRIMARY), x=0, xanchor="left", pad=dict(l=4),
                   y=1, yref="container", yanchor="top", automargin=False),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=[PRIMARY, ACCENT, "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
        margin=dict(l=8, r=8, t=84, b=8),
        hoverlabel=dict(bgcolor="white", bordercolor=BORDER, font=dict(color=TEXT_PRIMARY, family=FONT_FAMILY)),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    font=dict(color=TEXT_SECONDARY, size=12), bgcolor="rgba(0,0,0,0)"),
        xaxis=dict(gridcolor=GRID, linecolor=BORDER, zeroline=False, tickfont=dict(color=TEXT_SECONDARY),
                   title=dict(font=dict(color=TEXT_SECONDARY, size=12))),
        yaxis=dict(gridcolor=GRID, linecolor=BORDER, zeroline=False, tickfont=dict(color=TEXT_SECONDARY),
                   title=dict(font=dict(color=TEXT_SECONDARY, size=12))),
        bargap=0.25,
    )
    template.data.bar = [go.Bar(marker=dict(line=dict(color=SURFACE, width=1.5)))]
    template.data.scatter = [go.Scatter(line=dict(width=2))]
    pio.templates["lead_scoring"] = template
    pio.templates.default = "plotly_white+lead_scoring"


_CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{ font-family: {FONT_FAMILY}; }}
.block-container {{ padding-top: 1.6rem; padding-bottom: 3rem; max-width: 1320px; }}
h1, h2, h3, h4 {{ color: {TEXT_PRIMARY}; letter-spacing: -0.01em; }}

/* Hero banner */
.ls-hero {{
  background: linear-gradient(120deg, {PRIMARY_DARK} 0%, {PRIMARY} 100%);
  border-radius: 18px; padding: 1.6rem 1.9rem; color: #fff; margin-bottom: 1.2rem;
  box-shadow: 0 8px 24px rgba(16, 66, 129, 0.18);
}}
.ls-hero h1 {{ color: #fff; font-size: 1.85rem; font-weight: 700; margin: 0 0 .25rem 0; padding: 0; }}
.ls-hero p {{ color: rgba(255,255,255,.86); margin: 0; font-size: .98rem; }}
.ls-hero .ls-chips {{ margin-top: .9rem; display: flex; gap: .5rem; flex-wrap: wrap; }}
.ls-chip {{
  background: rgba(255,255,255,.14); border: 1px solid rgba(255,255,255,.25);
  border-radius: 999px; padding: .2rem .75rem; font-size: .8rem; color: #fff;
}}

/* KPI cards */
.ls-kpi {{
  background: #fff; border: 1px solid {BORDER}; border-radius: 14px;
  padding: 1rem 1.1rem; height: 100%; box-shadow: 0 1px 2px rgba(15,27,45,.04);
}}
.ls-kpi .label {{ color: {TEXT_SECONDARY}; font-size: .8rem; font-weight: 500;
  text-transform: uppercase; letter-spacing: .04em; }}
.ls-kpi .value {{ color: {TEXT_PRIMARY}; font-size: 1.7rem; font-weight: 700; line-height: 1.25; margin-top: .2rem; }}
.ls-kpi .hint {{ color: {TEXT_MUTED}; font-size: .8rem; margin-top: .15rem; }}

/* Compact metric strip */
.ls-mini {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(68px, 1fr)); gap: .5rem; margin: .4rem 0; }}
.ls-mini div {{ background: #fff; border: 1px solid {BORDER}; border-radius: 10px; padding: .5rem .6rem; }}
.ls-mini span {{ display: block; color: {TEXT_SECONDARY}; font-size: .72rem; text-transform: uppercase; letter-spacing: .03em; }}
.ls-mini b {{ color: {TEXT_PRIMARY}; font-size: 1.15rem; }}

/* Section header */
.ls-section {{ margin: 1.4rem 0 .6rem 0; }}
.ls-section h3 {{ font-size: 1.15rem; font-weight: 650; margin: 0; padding: 0; }}
.ls-section p {{ color: {TEXT_SECONDARY}; font-size: .9rem; margin: .2rem 0 0 0; }}

/* Priority tier cards */
.ls-tier {{
  background: #fff; border: 1px solid {BORDER}; border-radius: 14px; padding: 1rem 1.1rem;
  border-top: 5px solid var(--tier-color); height: 100%;
}}
.ls-tier .tier-title {{ font-weight: 700; color: {TEXT_PRIMARY}; font-size: 1rem; }}
.ls-tier .tier-range {{ color: {TEXT_SECONDARY}; font-size: .82rem; margin-bottom: .55rem; }}
.ls-tier .tier-action {{ color: {TEXT_PRIMARY}; font-size: .88rem; min-height: 2.6rem; }}
.ls-tier .tier-sla {{
  display: inline-block; margin-top: .55rem; background: {SURFACE_ALT}; color: {TEXT_SECONDARY};
  border-radius: 999px; padding: .1rem .6rem; font-size: .78rem; font-weight: 500;
}}
.ls-tier .tier-stats {{ margin-top: .7rem; display: flex; gap: 1.2rem; }}
.ls-tier .tier-stats b {{ display: block; font-size: 1.15rem; color: {TEXT_PRIMARY}; }}
.ls-tier .tier-stats span {{ color: {TEXT_MUTED}; font-size: .75rem; }}

/* Callout */
.ls-callout {{
  background: #eef5fe; border: 1px solid #cde2fb; border-left: 4px solid {PRIMARY};
  border-radius: 10px; padding: .85rem 1rem; color: {TEXT_PRIMARY}; font-size: .93rem; margin: .6rem 0 1rem 0;
}}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {{ gap: .25rem; border-bottom: 1px solid {BORDER}; }}
.stTabs [data-baseweb="tab"] {{ padding: .55rem .95rem; border-radius: 10px 10px 0 0; font-weight: 500; }}
.stTabs [aria-selected="true"] {{ background: {SURFACE_ALT}; }}

/* Sidebar */
section[data-testid="stSidebar"] {{ background: {SURFACE_ALT}; border-right: 1px solid {BORDER}; }}
.ls-brand {{ font-weight: 700; font-size: 1.15rem; color: {TEXT_PRIMARY}; }}
.ls-brand small {{ display: block; font-weight: 400; color: {TEXT_SECONDARY}; font-size: .8rem; }}

/* Footer */
.ls-footer {{ color: {TEXT_MUTED}; font-size: .8rem; text-align: center; margin-top: 2rem; }}

@media (max-width: 640px) {{
  .ls-hero {{ padding: 1.2rem; }}
  .ls-hero h1 {{ font-size: 1.4rem; }}
  .ls-kpi .value {{ font-size: 1.35rem; }}
}}
</style>
"""


def apply_theme() -> None:
    """Inject global CSS and activate the Plotly template. Call once per run."""
    _register_plotly_template()
    st.markdown(_CSS, unsafe_allow_html=True)
