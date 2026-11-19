"""Tab 6 - Score new leads: single-lead simulator and batch CSV upload."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from app.data import DashboardData
from app.ui import components, theme
from src.predict import predict_proba
from src.scoring import COL_PRIORITY, COL_SCORE, PRIORITY_ORDER, score_leads

NOT_PROVIDED = "(sin dato)"
MAX_UPLOAD_ROWS = 50_000
PREVIEW_ROWS = 500
TEMPLATE_COLUMNS = [
    "Lead Number", "Lead Origin", "Lead Source", "Do Not Email", "TotalVisits", "Total Time Spent on Website",
    "Page Views Per Visit", "Last Activity", "Specialization", "What is your current occupation", "City",
    "A free copy of Mastering The Interview",
]


def _options(data: DashboardData, column: str) -> list[str]:
    if column not in data.scored.columns:
        return []
    return data.scored[column].dropna().astype(str).value_counts().index.tolist()  # most frequent first


def _optional(value: str):
    return np.nan if value == NOT_PROVIDED else value


def _single_lead_form(data: DashboardData) -> pd.DataFrame | None:
    with st.form("lead_simulator"):
        first, second, third = st.columns(3)
        with first:
            origin = st.selectbox("Origen del lead", _options(data, "Lead Origin"))
            source = st.selectbox("Fuente", _options(data, "Lead Source"))
            activity = st.selectbox("Última actividad", _options(data, "Last Activity"))
        with second:
            occupation = st.selectbox("Ocupación", _options(data, "What is your current occupation"))
            specialization = st.selectbox("Especialización", [NOT_PROVIDED, *_options(data, "Specialization")])
            city = st.selectbox("Ciudad", [NOT_PROVIDED, *_options(data, "City")])
        with third:
            visits = st.number_input("Total de visitas", 0, 300, 3)
            seconds = st.number_input("Tiempo total en el sitio (segundos)", 0, 3000, 300, step=30)
            pages = st.number_input("Páginas por visita", 0.0, 60.0, 2.5, 0.5)
        flag_left, flag_right = st.columns(2)
        no_email = flag_left.checkbox("Pidió no recibir correos")
        ebook = flag_right.checkbox("Descargó «Mastering The Interview»")
        submitted = st.form_submit_button("Calcular score", type="primary", use_container_width=True)

    if not submitted:
        return None
    return pd.DataFrame([{
        "Lead Origin": origin, "Lead Source": source, "Last Activity": activity,
        "What is your current occupation": occupation,
        "Specialization": _optional(specialization), "City": _optional(city),
        "TotalVisits": visits, "Total Time Spent on Website": seconds, "Page Views Per Visit": pages,
        "Do Not Email": "Yes" if no_email else "No",
        "A free copy of Mastering The Interview": "Yes" if ebook else "No",
    }])


def _show_single_result(data: DashboardData, lead: pd.DataFrame) -> None:
    result = score_leads(lead, predict_proba(data.artifact, lead)).iloc[0]
    color = theme.PRIORITY_COLORS.get(result[COL_PRIORITY], theme.PRIMARY)
    percentile = (data.scored[COL_SCORE] < result[COL_SCORE]).mean()

    gauge_col, detail_col = st.columns([1, 1.3])
    with gauge_col:
        st.plotly_chart(components.score_gauge(int(result[COL_SCORE]), color), use_container_width=True)
    with detail_col:
        components.kpi_row([
            ("Prioridad", result[COL_PRIORITY], f"supera al {percentile:.0%} de la base"),
            ("Responder en", result["sla"], "SLA del nivel"),
        ])
        components.callout(f"**Acción recomendada:** {result['accion_recomendada']}")


def _batch_upload(data: DashboardData) -> None:
    components.section("Carga masiva", "Suba un CSV con las columnas de Leads.csv; las que falten se imputan.")
    template = pd.DataFrame(columns=TEMPLATE_COLUMNS).to_csv(index=False).encode("utf-8")
    st.download_button("📄 Descargar plantilla CSV", template, "plantilla_leads.csv", "text/csv")

    uploaded = st.file_uploader("CSV de leads", type="csv")
    if uploaded is None:
        return
    try:
        new_leads = pd.read_csv(uploaded)
    except (pd.errors.ParserError, UnicodeDecodeError, ValueError) as exc:
        st.error(f"No se pudo leer el archivo: {exc}")
        return
    if new_leads.empty:
        st.warning("El archivo no tiene filas.")
        return
    if len(new_leads) > MAX_UPLOAD_ROWS:
        st.error(f"El archivo supera el máximo de {MAX_UPLOAD_ROWS:,} filas.")
        return
    known = set(TEMPLATE_COLUMNS) & set(new_leads.columns)
    if not known:
        st.error("Ninguna columna coincide con Leads.csv. Use la plantilla como guía.")
        return

    with st.spinner("Puntuando leads…"):
        result = score_leads(new_leads, predict_proba(data.artifact, new_leads))

    counts = result[COL_PRIORITY].value_counts().reindex(PRIORITY_ORDER, fill_value=0).reset_index()
    counts.columns = [COL_PRIORITY, "leads"]
    left, right = st.columns([1, 2])
    with left:
        fig = px.bar(counts, x=COL_PRIORITY, y="leads", color=COL_PRIORITY, text="leads",
                     color_discrete_map=theme.PRIORITY_COLORS, title=f"{len(result):,} leads puntuados")
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(showlegend=False, xaxis_title="", yaxis_title="Leads", height=360)
        st.plotly_chart(fig, use_container_width=True)
    with right:
        st.dataframe(result.head(PREVIEW_ROWS), hide_index=True, use_container_width=True, height=360,
                     column_config={COL_SCORE: st.column_config.ProgressColumn(
                         "Lead Score", min_value=0, max_value=100, format="%d")})
    st.download_button("⬇️ Descargar leads puntuados", result.to_csv(index=False).encode("utf-8"),
                       "leads_puntuados.csv", "text/csv", type="primary")


def render(data: DashboardData, leads: pd.DataFrame) -> None:
    if data.artifact is None:
        st.warning(
            "No se pudo cargar `models/modelo_produccion.joblib`. Ejecute `python -m src.pipeline` con las "
            f"versiones de requirements.txt.\n\nDetalle: {data.artifact_error}"
        )
        return

    components.section("Simulador: puntuar un lead nuevo",
                       "Complete los datos que se conocen al momento de llegar el lead.")
    lead = _single_lead_form(data)
    if lead is not None:
        _show_single_result(data, lead)

    st.divider()
    _batch_upload(data)
