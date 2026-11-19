"""Tab 7 - Project methodology (CRISP-DM) for the class presentation."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.data import DashboardData
from app.ui import components

CRISP_DM = [
    ("1. Comprensión del negocio", "Priorizar los leads con mayor probabilidad de compra para que ventas "
                                   "enfoque su esfuerzo. Meta: ~80 % de conversión en los leads priorizados."),
    ("2. Comprensión de los datos", "9.240 leads y 37 variables de X Education. Análisis de nulos, "
                                    "distribución del target y conversión por variable."),
    ("3. Preparación", "'Select' → nulo, eliminación de variables con fuga de información, imputación, "
                       "winsorización p1–p99, escalado y one-hot encoding dentro de un Pipeline."),
    ("4. Modelado", "Siete modelos (Regresión Logística, KNN, Árbol, Random Forest, Gradient Boosting, "
                    "XGBoost y Red Neuronal MLP) con búsqueda aleatoria de hiperparámetros y CV 5-fold."),
    ("5. Evaluación", "Métricas en un 20 % de prueba, curvas ROC/PR, lift, calibración y umbral óptimo de Youden."),
    ("6. Despliegue", "Score 0–100, ranking, sistema de priorización A/B/C/D y este dashboard en Streamlit Cloud."),
]


def render(data: DashboardData, leads: pd.DataFrame) -> None:
    components.section("Metodología CRISP-DM", "Cómo se construyó el modelo, de principio a fin.")
    for row_start in range(0, len(CRISP_DM), 3):
        for column, (title, text) in zip(st.columns(3), CRISP_DM[row_start:row_start + 3]):
            with column, st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(text)

    components.section("¿Por qué se excluyen algunas variables?")
    components.callout(
        "**Tags**, **Lead Quality**, **Lead Profile** y los índices **Asymmetrique** los llena el equipo de "
        "ventas **después** de hablar con el lead. Con ellas el AUC sube artificialmente a ~0,98, pero un lead "
        "nuevo todavía no las tiene. El modelo usa solo lo que se sabe cuando el lead llega."
    )

    best = data.metrics["metricas_mejor_umbral_optimo"]
    components.section(f"Modelo en producción: {data.best_model}",
                       f"Umbral óptimo de Youden = {data.youden_threshold:.3f}, métricas en el conjunto de prueba.")
    components.kpi_row([
        ("ROC-AUC", f"{best['roc_auc']:.3f}", "capacidad de ordenar"),
        ("Recall", f"{best['recall']:.3f}", "conversiones detectadas"),
        ("Precision", f"{best['precision']:.3f}", "aciertos al predecir"),
        ("F1", f"{best['f1']:.3f}", "balance P/R"),
        ("KS", f"{best['ks']:.3f}", "separación de clases"),
    ])
    st.caption(f"Entrenamiento: {data.metrics['n_train']:,} leads · Prueba: {data.metrics['n_test']:,} leads · "
               f"Conversión base: {data.metrics['tasa_conversion']:.1%}")
