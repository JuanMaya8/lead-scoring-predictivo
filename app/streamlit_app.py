"""Dashboard comercial - Lead Scoring Predictivo (X Education).

Responsable: Jack.

Ejecutar localmente (desde la raíz del repo):
    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from src import config  # noqa: E402
from src.evaluate import calcular_metricas  # noqa: E402
from src.predict import cargar_artefacto, predecir_probabilidad  # noqa: E402
from src.scoring import accion_por_prioridad, puntuar_leads  # noqa: E402

st.set_page_config(page_title="Lead Scoring | X Education", page_icon="🎯", layout="wide")

COLORES_PRIORIDAD = {
    "A - Caliente": "#d62728",
    "B - Tibio": "#ff7f0e",
    "C - Templado": "#f2c14e",
    "D - Frío": "#1f77b4",
}
ORDEN_PRIORIDAD = list(COLORES_PRIORIDAD)


# ---------------------------------------------------------------------------
# Carga de datos (cacheada)
# ---------------------------------------------------------------------------
@st.cache_data
def cargar_scored() -> pd.DataFrame:
    return pd.read_csv(config.LEADS_SCORED)


@st.cache_data
def cargar_metricas() -> dict:
    with open(config.METRICS_JSON, encoding="utf-8") as f:
        return json.load(f)


@st.cache_data
def cargar_pred_test() -> pd.DataFrame:
    return pd.read_csv(config.PRED_TEST_CSV)


@st.cache_data
def cargar_importancia() -> pd.DataFrame:
    return pd.read_csv(config.IMPORTANCIA_CSV)


@st.cache_resource
def cargar_modelo():
    try:
        return cargar_artefacto()
    except Exception as e:  # versión de sklearn distinta, archivo faltante, etc.
        st.session_state["error_modelo"] = str(e)
        return None


if not config.LEADS_SCORED.exists() or not config.METRICS_JSON.exists():
    st.error("No se encontraron los resultados del modelo. Ejecute primero: `python -m src.pipeline`")
    st.stop()

scored = cargar_scored()
met = cargar_metricas()
pred_test = cargar_pred_test()
importancia = cargar_importancia()
artefacto = cargar_modelo()
mejor = met["mejor_modelo"]

# ---------------------------------------------------------------------------
# Barra lateral: filtros
# ---------------------------------------------------------------------------
st.sidebar.title("🎯 Lead Scoring")
st.sidebar.caption(f"Modelo en producción: **{mejor}**")
st.sidebar.divider()
st.sidebar.subheader("Filtros")


def filtro_multi(col: str, etiqueta: str):
    if col not in scored.columns:
        return None
    opciones = sorted(scored[col].dropna().astype(str).unique())
    return st.sidebar.multiselect(etiqueta, opciones)


f_prio = st.sidebar.multiselect("Prioridad", ORDEN_PRIORIDAD)
f_origen = filtro_multi("Lead Origin", "Origen del lead")
f_fuente = filtro_multi("Lead Source", "Fuente")
f_ocup = filtro_multi("What is your current occupation", "Ocupación")
rango = st.sidebar.slider("Rango de Lead Score", 0, 100, (0, 100))

df = scored[(scored["lead_score"] >= rango[0]) & (scored["lead_score"] <= rango[1])]
for col, sel in [("prioridad", f_prio), ("Lead Origin", f_origen), ("Lead Source", f_fuente),
                 ("What is your current occupation", f_ocup)]:
    if sel:
        df = df[df[col].astype(str).isin(sel)]

st.sidebar.divider()
st.sidebar.caption(f"{len(df):,} de {len(scored):,} leads seleccionados")

# ---------------------------------------------------------------------------
# Encabezado y pestañas
# ---------------------------------------------------------------------------
st.title("Dashboard comercial de Lead Scoring")
st.caption("X Education · Probabilidad de conversión de cada lead estimada con machine learning")

tabs = st.tabs(["📊 Resumen", "🏆 Ranking", "🚦 Priorización", "🤖 Modelos", "🔍 Factores clave", "🧮 Puntuar leads"])

# ---- 1. Resumen ------------------------------------------------------------
with tabs[0]:
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Leads", f"{len(df):,}")
    c2.metric("Conversión histórica", f"{df[config.TARGET].mean():.1%}" if len(df) else "-")
    c3.metric("Score promedio", f"{df['lead_score'].mean():.0f}" if len(df) else "-")
    calientes = (df["prioridad"] == "A - Caliente").sum()
    c4.metric("Leads calientes (A)", f"{calientes:,}")
    c5.metric("Conversiones esperadas", f"{df['probabilidad'].sum():,.0f}")

    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.histogram(df, x="lead_score", color="prioridad", nbins=40,
                           color_discrete_map=COLORES_PRIORIDAD, category_orders={"prioridad": ORDEN_PRIORIDAD},
                           title="Distribución del Lead Score")
        fig.update_layout(bargap=0.05, xaxis_title="Lead Score", yaxis_title="Leads")
        st.plotly_chart(fig, use_container_width=True)
    with col_b:
        tmp = df.assign(rango_score=pd.cut(df["lead_score"], bins=range(0, 101, 10), include_lowest=True).astype(str))
        g = tmp.groupby("rango_score", observed=True)[config.TARGET].agg(["mean", "size"]).reset_index()
        fig = px.bar(g, x="rango_score", y="mean", text=g["mean"].map("{:.0%}".format),
                     title="Conversión real por rango de score", labels={"mean": "Tasa de conversión", "rango_score": "Score"})
        fig.update_yaxes(tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    col_c, col_d = st.columns(2)
    for col_ui, var, titulo in [(col_c, "Lead Source", "Score promedio y conversión por fuente"),
                                (col_d, "Lead Origin", "Score promedio y conversión por origen")]:
        with col_ui:
            g = (df.groupby(var).agg(leads=("lead_score", "size"), score=("lead_score", "mean"),
                                     conversion=(config.TARGET, "mean"))
                 .query("leads >= 20").sort_values("score", ascending=False).reset_index())
            fig = px.bar(g, x=var, y="score", color="conversion", color_continuous_scale="RdYlGn",
                         hover_data={"leads": True, "conversion": ":.1%"}, title=titulo)
            fig.update_layout(yaxis_title="Score promedio", coloraxis_colorbar_tickformat=".0%")
            st.plotly_chart(fig, use_container_width=True)

# ---- 2. Ranking ------------------------------------------------------------
with tabs[1]:
    st.subheader("Ranking de leads por probabilidad de conversión")
    st.caption("Ordenados de mayor a menor score. Los scores son *out-of-fold*: cada lead fue puntuado por un "
               "modelo que no lo vio durante el entrenamiento.")
    top_n = st.select_slider("Mostrar", options=[25, 50, 100, 250, 500, 1000, len(df) or 1], value=100)
    cols = ["ranking", "Lead Number", "lead_score", "prioridad", "accion_recomendada", "sla", "Lead Origin",
            "Lead Source", "Last Activity", "What is your current occupation", "TotalVisits",
            "Total Time Spent on Website", config.TARGET]
    vista = df[[c for c in cols if c in df.columns]].head(top_n)
    st.dataframe(
        vista, use_container_width=True, hide_index=True, height=520,
        column_config={
            "lead_score": st.column_config.ProgressColumn("Lead Score", min_value=0, max_value=100, format="%d"),
            config.TARGET: st.column_config.CheckboxColumn("¿Convirtió?"),
            "accion_recomendada": "Acción recomendada",
            "sla": "SLA",
        },
    )
    st.download_button("⬇️ Descargar ranking filtrado (CSV)", df.to_csv(index=False).encode("utf-8"),
                       "ranking_leads.csv", "text/csv")

# ---- 3. Priorización -------------------------------------------------------
with tabs[2]:
    st.subheader("Sistema de priorización comercial")
    st.dataframe(accion_por_prioridad(), hide_index=True, use_container_width=True)

    g = (df.groupby("prioridad").agg(leads=("lead_score", "size"), convertidos=(config.TARGET, "sum"))
         .reindex(ORDEN_PRIORIDAD).fillna(0).reset_index())
    g["tasa"] = g["convertidos"] / g["leads"].clip(lower=1)
    g["porc_leads"] = g["leads"] / max(g["leads"].sum(), 1)
    g["porc_conv"] = g["convertidos"] / max(g["convertidos"].sum(), 1)

    c1, c2 = st.columns(2)
    with c1:
        fig = px.bar(g, x="prioridad", y="tasa", color="prioridad", color_discrete_map=COLORES_PRIORIDAD,
                     text=g["tasa"].map("{:.0%}".format), title="Tasa de conversión real por prioridad")
        fig.update_layout(showlegend=False, yaxis_tickformat=".0%", yaxis_title="Conversión")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fig = go.Figure()
        fig.add_bar(x=g["prioridad"], y=g["porc_leads"], name="% de leads")
        fig.add_bar(x=g["prioridad"], y=g["porc_conv"], name="% de conversiones")
        fig.update_layout(barmode="group", title="Esfuerzo vs. resultado", yaxis_tickformat=".0%")
        st.plotly_chart(fig, use_container_width=True)

    ab = g[g["prioridad"].isin(ORDEN_PRIORIDAD[:2])]
    if g["convertidos"].sum() > 0:
        st.success(
            f"Contactando solo los leads **A y B** ({ab['porc_leads'].sum():.0%} de la base) se capturan "
            f"**{ab['porc_conv'].sum():.0%} de las conversiones**, con una tasa de conversión de "
            f"**{ab['convertidos'].sum() / max(ab['leads'].sum(), 1):.0%}** frente al "
            f"{df[config.TARGET].mean():.0%} de llamar a todos."
        )

    st.markdown("#### Curva de ganancia: ¿cuántas conversiones capturo si contacto el top X%?")
    orden = df.sort_values("probabilidad", ascending=False)
    total_conv = max(orden[config.TARGET].sum(), 1)
    curva = pd.DataFrame({
        "porc_contactado": np.arange(1, len(orden) + 1) / max(len(orden), 1),
        "porc_capturado": orden[config.TARGET].cumsum().values / total_conv,
    })
    fig = px.line(curva, x="porc_contactado", y="porc_capturado")
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line_dash="dash", name="Sin modelo (azar)")
    fig.update_layout(xaxis_tickformat=".0%", yaxis_tickformat=".0%", xaxis_title="% de leads contactados",
                      yaxis_title="% de conversiones capturadas")
    st.plotly_chart(fig, use_container_width=True)

# ---- 4. Modelos ------------------------------------------------------------
with tabs[3]:
    st.subheader("Comparación de modelos (conjunto de prueba, 20%)")
    tabla = pd.DataFrame(met["metricas_umbral_05"]).T
    cols_m = ["cv_roc_auc", "roc_auc", "accuracy", "precision", "recall", "especificidad", "f1", "pr_auc", "ks", "mcc"]
    tabla = tabla[cols_m].astype(float).sort_values("cv_roc_auc", ascending=False)
    st.dataframe(
        tabla.style.format("{:.3f}").highlight_max(axis=0, color="#c7e9c0"),
        use_container_width=True,
    )
    st.caption("cv_roc_auc = AUC promedio en validación cruzada 5-fold (criterio de selección). "
               "Demás métricas en test con umbral 0.5.")

    c1, c2 = st.columns(2)
    with c1:
        fig = go.Figure()
        y_real = pred_test["y_real"]
        for m_nombre in [c for c in pred_test.columns if c != "y_real"]:
            fpr, tpr, _ = roc_curve(y_real, pred_test[m_nombre])
            auc = met["metricas_umbral_05"][m_nombre]["roc_auc"]
            fig.add_scatter(x=fpr, y=tpr, mode="lines", name=f"{m_nombre} ({auc:.3f})")
        fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", line_dash="dash", name="Azar (0.5)")
        fig.update_layout(title="Curvas ROC", xaxis_title="FPR (1 - especificidad)", yaxis_title="TPR (recall)",
                          height=480, legend_font_size=10)
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.markdown(f"**Matriz de confusión interactiva – {mejor}**")
        umbral = st.slider("Umbral de decisión", 0.05, 0.95, float(round(met["umbrales"]["youden"], 2)), 0.01,
                           help="Umbral óptimo (Youden) calculado con validación cruzada en el entrenamiento.")
        m = calcular_metricas(y_real, pred_test[mejor], umbral)
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        fig = px.imshow(cm, text_auto=True, color_continuous_scale="Blues",
                        x=["Pred: No convierte", "Pred: Convierte"], y=["Real: No convierte", "Real: Convierte"])
        fig.update_layout(coloraxis_showscale=False, height=300, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)
        k = st.columns(5)
        for col_ui, (nom, clave) in zip(k, [("Accuracy", "accuracy"), ("Precision", "precision"),
                                             ("Recall", "recall"), ("Especif.", "especificidad"), ("F1", "f1")]):
            col_ui.metric(nom, f"{m[clave]:.3f}")
        st.caption("Subir el umbral ⇒ más precision, menos recall (menos llamadas perdidas en leads fríos). "
                   "Bajarlo ⇒ más recall (no dejar escapar conversiones).")

    lift = pd.DataFrame(met["lift"])
    fig = px.bar(lift, x="decil", y="lift", text=lift["tasa_conversion"].map("{:.0%}".format),
                 title=f"Lift por decil – {mejor} (texto = conversión real del decil)")
    fig.add_hline(y=1, line_dash="dash")
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("Hiperparámetros óptimos por modelo"):
        st.json({k: v["mejores_parametros"] for k, v in met["metricas_umbral_05"].items()})

# ---- 5. Factores clave -----------------------------------------------------
with tabs[4]:
    st.subheader("¿Qué variables influyen más en la conversión?")
    top = importancia.head(15).iloc[::-1]
    fig = px.bar(top, x="importancia", y="variable", orientation="h", error_x="desviacion",
                 title=f"Importancia por permutación – {mejor}",
                 labels={"importancia": "Caída del ROC-AUC al desordenar la variable", "variable": ""})
    fig.update_layout(height=520)
    st.plotly_chart(fig, use_container_width=True)

    var = st.selectbox("Explorar conversión por variable",
                       [c for c in ["Last Activity", "Lead Source", "Lead Origin", "What is your current occupation",
                                    "Specialization", "City"] if c in scored.columns])
    g = (df.groupby(var).agg(leads=("lead_score", "size"), conversion=(config.TARGET, "mean"),
                             score=("lead_score", "mean")).query("leads >= 15").sort_values("conversion"))
    fig = px.bar(g.reset_index(), x="conversion", y=var, orientation="h", color="score",
                 color_continuous_scale="RdYlGn", hover_data=["leads"], title=f"Conversión real por {var}")
    fig.update_layout(xaxis_tickformat=".0%", height=max(350, 28 * len(g)))
    st.plotly_chart(fig, use_container_width=True)

# ---- 6. Puntuar leads nuevos ----------------------------------------------
with tabs[5]:
    if artefacto is None:
        st.warning("No se pudo cargar `models/modelo_produccion.joblib`. Ejecute `python -m src.pipeline` "
                   f"con las versiones de requirements.txt. Detalle: {st.session_state.get('error_modelo', '')}")
    else:
        st.subheader("Simulador: puntuar un lead nuevo")

        def opciones(col):
            return sorted(scored[col].dropna().astype(str).unique()) if col in scored.columns else []

        with st.form("simulador"):
            a, b, c = st.columns(3)
            origen = a.selectbox("Lead Origin", opciones("Lead Origin"))
            fuente = a.selectbox("Lead Source", opciones("Lead Source"))
            actividad = a.selectbox("Last Activity", opciones("Last Activity"))
            ocupacion = b.selectbox("Ocupación", opciones("What is your current occupation"))
            espec = b.selectbox("Especialización", ["(sin dato)"] + opciones("Specialization"))
            ciudad = b.selectbox("Ciudad", ["(sin dato)"] + opciones("City"))
            visitas = c.number_input("Total de visitas", 0, 300, 3)
            tiempo = c.number_input("Tiempo total en el sitio (s)", 0, 3000, 300)
            paginas = c.number_input("Páginas por visita", 0.0, 60.0, 2.5, 0.5)
            no_email = a.checkbox("Pidió no recibir correos")
            libro = b.checkbox("Descargó 'Mastering The Interview'")
            enviar = st.form_submit_button("Calcular score", type="primary")

        if enviar:
            lead = pd.DataFrame([{
                "Lead Origin": origen, "Lead Source": fuente, "Last Activity": actividad,
                "What is your current occupation": ocupacion,
                "Specialization": np.nan if espec == "(sin dato)" else espec,
                "City": np.nan if ciudad == "(sin dato)" else ciudad,
                "TotalVisits": visitas, "Total Time Spent on Website": tiempo, "Page Views Per Visit": paginas,
                "Do Not Email": "Yes" if no_email else "No",
                "A free copy of Mastering The Interview": "Yes" if libro else "No",
            }])
            res = puntuar_leads(lead, predecir_probabilidad(artefacto, lead)).iloc[0]
            k1, k2, k3 = st.columns(3)
            k1.metric("Lead Score", f"{res['lead_score']} / 100")
            k2.metric("Prioridad", res["prioridad"])
            k3.metric("SLA", res["sla"])
            st.info(f"**Acción recomendada:** {res['accion_recomendada']}")
            percentil = (scored["lead_score"] < res["lead_score"]).mean()
            st.caption(f"Este lead supera al {percentil:.0%} de los leads históricos.")

        st.divider()
        st.subheader("Carga masiva: puntuar un archivo CSV")
        st.caption("Suba un CSV con las mismas columnas de Leads.csv (las faltantes se imputan).")
        archivo = st.file_uploader("CSV de leads", type="csv")
        if archivo is not None:
            nuevos = pd.read_csv(archivo)
            res = puntuar_leads(nuevos, predecir_probabilidad(artefacto, nuevos))
            st.dataframe(res.head(500), hide_index=True, use_container_width=True)
            st.bar_chart(res["prioridad"].value_counts().reindex(ORDEN_PRIORIDAD).fillna(0))
            st.download_button("⬇️ Descargar leads puntuados", res.to_csv(index=False).encode("utf-8"),
                               "leads_puntuados.csv", "text/csv")

st.divider()
st.caption("Proyecto final de Minería de Datos · Juan David Maya & Jack · Dataset: X Education Leads (Kaggle)")
