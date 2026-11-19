"""Sales dashboard - Predictive Lead Scoring (X Education).

Owner: Jack.

Run locally from the repository root:
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.data import load_dashboard_data, missing_files  # noqa: E402
from app.filters import render_sidebar_filters  # noqa: E402
from app.ui import components  # noqa: E402
from app.ui.theme import apply_theme  # noqa: E402
from app.views import about, drivers, models, overview, prioritization, ranking, scorer  # noqa: E402

st.set_page_config(
    page_title="Lead Scoring | X Education",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

TABS = [
    ("📊 Resumen", overview),
    ("🏆 Ranking", ranking),
    ("🚦 Priorización", prioritization),
    ("🤖 Modelos", models),
    ("🔍 Factores clave", drivers),
    ("🧮 Puntuar leads", scorer),
    ("📘 Metodología", about),
]


def main() -> None:
    missing = missing_files()
    if missing:
        st.error("No se encontraron los resultados del modelo. Ejecute primero `python -m src.pipeline`.")
        st.code("\n".join(missing))
        st.stop()

    data = load_dashboard_data()

    st.sidebar.markdown('<div class="ls-brand">🎯 Lead Scoring<small>X Education</small></div>',
                        unsafe_allow_html=True)
    st.sidebar.caption(f"Modelo en producción: **{data.best_model}**")
    st.sidebar.divider()
    filters = render_sidebar_filters(data.scored)
    leads = filters.apply(data.scored)
    st.sidebar.divider()
    st.sidebar.caption(f"**{len(leads):,}** de {len(data.scored):,} leads seleccionados"
                       + (f" · {filters.active_count} filtro(s) activo(s)" if filters.active_count else ""))

    best = data.metrics["metricas_mejor_umbral_optimo"]
    components.hero(
        "Dashboard comercial de Lead Scoring",
        "Probabilidad de conversión de cada lead estimada con machine learning, "
        "convertida en un ranking y una prioridad de contacto.",
        chips=[f"Modelo: {data.best_model}", f"ROC-AUC {best['roc_auc']:.3f}",
               f"{len(data.scored):,} leads puntuados", f"Umbral óptimo {data.youden_threshold:.2f}"],
    )

    for tab, (_, view) in zip(st.tabs([label for label, _ in TABS]), TABS):
        with tab:
            view.render(data, leads)

    components.footer("Proyecto final de Minería de Datos · Juan David Maya & Jack · "
                      "Dataset: X Education Leads (Kaggle)")


main()
