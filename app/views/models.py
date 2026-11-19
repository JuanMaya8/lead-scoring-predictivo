"""Tab 4 - Model comparison, ROC curves, interactive confusion matrix and lift."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.metrics import roc_curve

from app.data import DashboardData
from app.ui import components, theme
from src.evaluate import calcular_metricas

METRIC_LABELS = {
    "cv_roc_auc": "CV ROC-AUC",
    "roc_auc": "ROC-AUC",
    "accuracy": "Accuracy",
    "precision": "Precision",
    "recall": "Recall",
    "especificidad": "Especificidad",
    "f1": "F1",
    "pr_auc": "PR-AUC",
    "ks": "KS",
    "mcc": "MCC",
}


def _comparison_table(data: DashboardData) -> None:
    table = pd.DataFrame(data.metrics["metricas_umbral_05"]).T
    table = (table[list(METRIC_LABELS)].astype(float)
             .sort_values("cv_roc_auc", ascending=False)
             .rename(columns=METRIC_LABELS))
    table.index = [f"⭐ {name}" if name == data.best_model else name for name in table.index]
    st.dataframe(
        table.style.format("{:.3f}").highlight_max(axis=0, props="background-color:#dbe9fb; font-weight:600;"),
        use_container_width=True,
    )
    st.caption("CV ROC-AUC = promedio en validación cruzada 5-fold (criterio de selección del modelo). "
               "Las demás métricas se calculan en el conjunto de prueba con umbral 0,5. "
               "Resaltado = mejor valor de cada columna.")


def _roc_chart(data: DashboardData) -> None:
    y_true = data.test_predictions["y_real"]
    models = [c for c in data.test_predictions.columns if c != "y_real"]
    # Draw the production model last so it sits on top.
    models.sort(key=lambda name: name == data.best_model)
    fig = go.Figure()
    for name in models:
        fpr, tpr, _ = roc_curve(y_true, data.test_predictions[name])
        auc = data.metrics["metricas_umbral_05"][name]["roc_auc"]
        is_best = name == data.best_model
        fig.add_scatter(
            x=fpr, y=tpr, mode="lines", name=f"{name} ({auc:.3f})",
            line=dict(color=theme.PRIMARY if is_best else theme.NEUTRAL_SERIES, width=3 if is_best else 1.5),
            hovertemplate=f"<b>{name}</b><br>FPR %{{x:.2f}} · TPR %{{y:.2f}}<extra></extra>",
        )
    fig.add_scatter(x=[0, 1], y=[0, 1], mode="lines", name="Azar (0,500)",
                    line=dict(color=theme.TEXT_MUTED, dash="dash", width=1), hoverinfo="skip")
    fig.update_layout(title=f"Curvas ROC (resaltado: {data.best_model})", height=500,
                      xaxis_title="Tasa de falsos positivos (1 − especificidad)",
                      yaxis_title="Tasa de verdaderos positivos (recall)",
                      legend=dict(orientation="v", y=0.02, yanchor="bottom", x=0.98, xanchor="right",
                                  bgcolor="rgba(255,255,255,0.85)", font=dict(size=11)))
    st.plotly_chart(fig, use_container_width=True)


def _confusion_matrix(data: DashboardData) -> None:
    st.markdown(f"**Matriz de confusión interactiva · {data.best_model}**")
    threshold = st.slider(
        "Umbral de decisión", 0.05, 0.95, float(round(data.youden_threshold, 2)), 0.01,
        help="Por defecto, el umbral óptimo de Youden calculado con validación cruzada en el entrenamiento.",
    )
    m = calcular_metricas(data.test_predictions["y_real"], data.test_predictions[data.best_model], threshold)
    matrix = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig = px.imshow(matrix, text_auto=True, color_continuous_scale=theme.SEQUENTIAL_SCALE, aspect="auto",
                    x=["Predice: no convierte", "Predice: convierte"],
                    y=["Real: no convierte", "Real: convierte"])
    fig.update_traces(textfont_size=18, hovertemplate="%{y}<br>%{x}<br><b>%{z:,} leads</b><extra></extra>")
    fig.update_layout(coloraxis_showscale=False, height=290, margin=dict(t=10, l=8, r=8, b=8))
    st.plotly_chart(fig, use_container_width=True)

    components.metric_strip([
        (label, f"{m[key]:.3f}")
        for label, key in [("Accuracy", "accuracy"), ("Precision", "precision"), ("Recall", "recall"),
                           ("Especif.", "especificidad"), ("F1", "f1")]
    ])
    st.caption("Subir el umbral → más precision y menos llamadas a leads fríos. "
               "Bajarlo → más recall: se escapan menos conversiones.")


def _lift_chart(data: DashboardData) -> None:
    lift = pd.DataFrame(data.metrics["lift"])
    fig = px.bar(lift, x="decil", y="lift", text=lift["tasa_conversion"].map("{:.0%}".format),
                 hover_data={"leads": ":,", "convertidos": ":,", "tasa_conversion": ":.1%"},
                 labels={"decil": "Decil de score (1 = mejores leads)", "lift": "Lift",
                         "tasa_conversion": "Conversión"},
                 title=f"Lift por decil · {data.best_model} (etiqueta = conversión real del decil)")
    fig.update_traces(marker_color=theme.PRIMARY, textposition="outside", cliponaxis=False)
    fig.add_hline(y=1, line_dash="dash", line_color=theme.TEXT_MUTED,
                  annotation_text="Sin modelo", annotation_position="top right")
    fig.update_xaxes(dtick=1)
    st.plotly_chart(fig, use_container_width=True)


def render(data: DashboardData, leads: pd.DataFrame) -> None:
    components.section("Comparación de modelos", "Conjunto de prueba (20 %), nunca usado para entrenar ni elegir.")
    _comparison_table(data)

    left, right = st.columns([1.15, 1])
    with left:
        _roc_chart(data)
    with right:
        _confusion_matrix(data)

    _lift_chart(data)
    with st.expander("Hiperparámetros óptimos por modelo"):
        st.json({name: values["mejores_parametros"] for name, values in data.metrics["metricas_umbral_05"].items()})
