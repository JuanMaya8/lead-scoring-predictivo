# 🎯 Lead Scoring Predictivo: X Education

Proyecto final de **Minería de Datos**. Modelo predictivo que clasifica leads según su
**probabilidad de conversión**, los ordena en un **ranking**, les asigna una **prioridad comercial**
y los muestra en un **dashboard web** hecho con Streamlit.

**Equipo:** Juan David Maya · Jack
**Dataset:** [Lead Scoring, X Education (Kaggle)](https://www.kaggle.com/code/kratikulhara/lead-scoring-case-study): 9.240 leads, 37 variables, 38,5% de conversión.
**Dashboard en línea:** <https://lead-scoring-predictivo-gxznqeerygyejm9egbebgs.streamlit.app/>

---

## Objetivo

X Education convierte solo ~30-38% de sus leads. El equipo comercial pierde tiempo llamando a
todos por igual. Se busca identificar los **"hot leads"** para enfocar el esfuerzo en ellos.
La meta del caso de negocio es una conversión de ~80% en los leads priorizados.

## Metodología (CRISP-DM)

| Fase | Qué se hizo | Archivo |
|---|---|---|
| 1. Comprensión del negocio | Priorizar leads con mayor probabilidad de compra | este README |
| 2. Comprensión de los datos | EDA: nulos, distribución del target, conversión por variable | `src/eda.py` |
| 3. Preparación | `Select` → nulo, eliminación de variables con fuga, >45% nulos o >95% de una sola clase; imputación, winsorización p1-p99, escalado z, one-hot | `src/data_prep.py`, `src/features.py` |
| 4. Modelado | 7 modelos con `RandomizedSearchCV` (5-fold, ROC-AUC) | `src/models.py`, `src/ann.py` |
| 5. Evaluación | 13 métricas + ROC, PR, lift, calibración y umbral óptimo (Youden) | `src/evaluate.py` |
| 6. Despliegue | Ranking, priorización y dashboard Streamlit | `src/scoring.py`, `app/` |

### Feature engineering

| Variable nueva | Fórmula / lógica |
|---|---|
| `fe_tiempo_por_visita` | Tiempo total en el sitio / número de visitas |
| `fe_engagement` | log(1+tiempo) × log(1+visitas) |
| `fe_paginas_totales` | Visitas × páginas por visita |
| `fe_log_tiempo` | log(1+tiempo): reduce la asimetría |
| `fe_tiene_visitas` | 1 si visitó el sitio al menos una vez |
| `fe_profesional` | 1 si es "Working Professional" |
| `fe_canal` | Agrupa 20 fuentes en 6 canales (búsqueda, directo, chat, referidos, redes, otro) |
| `fe_actividad_alto_interes` | 1 si la última actividad fue SMS, llamada, clic en email o formulario |

### ⚠️ Fuga de información (data leakage)

Se excluyen `Tags`, `Lead Quality`, `Lead Profile` y los índices `Asymmetrique`. Estas variables las
llena el equipo de ventas **después** de hablar con el lead. Con ellas el AUC sube artificialmente
a ~0,98, pero un lead nuevo todavía no las tiene. El modelo se entrena solo con lo que se sabe
al momento de llegar el lead.

## Resultados (conjunto de prueba, 20% = 1.848 leads)

| Modelo | CV ROC-AUC | Test ROC-AUC | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| **Gradient Boosting** ⭐ | **0,907** | 0,887 | 0,816 | 0,763 | 0,758 | 0,761 |
| XGBoost | 0,907 | 0,891 | 0,826 | 0,767 | 0,789 | 0,778 |
| Red Neuronal (MLP 64-32) | 0,905 | 0,887 | 0,823 | 0,765 | 0,781 | 0,773 |
| Random Forest | 0,904 | 0,886 | 0,824 | 0,765 | 0,782 | 0,774 |
| Regresión Logística | 0,900 | 0,879 | 0,803 | 0,760 | 0,715 | 0,737 |
| KNN (k=37, Manhattan) | 0,897 | 0,884 | 0,822 | 0,782 | 0,746 | 0,763 |
| Árbol de Decisión | 0,890 | 0,877 | 0,812 | 0,739 | 0,789 | 0,764 |

*Métricas con umbral 0,5. El mejor modelo se elige por AUC de validación cruzada, no por test
(Gradient Boosting y XGBoost empatan en 0,907; Gradient Boosting queda por una diferencia mínima).*

**Modelo final: Gradient Boosting con umbral óptimo de Youden de 0,351.**
Accuracy 0,816 · Precision 0,733 · **Recall 0,822** · Especificidad 0,812 · F1 0,775 ·
ROC-AUC 0,887 · PR-AUC 0,827 · KS 0,643 · MCC 0,623.

**Impacto comercial:**
- Los leads con prioridad **A y B** son el 37% de la base y concentran el **76% de las conversiones**, con una **tasa de conversión de 80%** (meta del caso), frente al 38,5% de llamar a todos.
- El **10% de leads con mayor score convierte al 91%** (lift de 2,4×).

**Variables más importantes:** última actividad, ocupación, origen del lead, tiempo en el sitio web
y páginas por visita.

## Sistema de priorización

| Prioridad | Score | Acción | SLA | Conversión real |
|---|---|---|---|---|
| 🔴 A - Caliente | ≥ 75 | Llamada inmediata de asesor senior | < 1 hora | 88% |
| 🟠 B - Tibio | 50-74 | Llamada + correo personalizado | < 24 horas | 64% |
| 🟡 C - Templado | 25-49 | Nurturing por correo/WhatsApp | Semanal | 38% |
| 🔵 D - Frío | < 25 | Campañas automáticas | Mensual | 8% |

## Estructura

```
lead-scoring-predictivo/
├── app/                      # Dashboard Streamlit (Jack)
│   ├── streamlit_app.py      # Punto de entrada
│   ├── data.py, filters.py   # Carga cacheada de artefactos y filtros de la barra lateral
│   ├── ui/                   # Tema y componentes reutilizables
│   └── views/                # Una vista por pestaña (resumen, ranking, priorización, ...)
├── src/
│   ├── config.py             # Configuración (Juan David)
│   ├── data_prep.py          # Limpieza (Juan David)
│   ├── eda.py                # Análisis exploratorio (Juan David)
│   ├── features.py           # Feature engineering (Juan David)
│   ├── models.py             # LogReg, KNN, Árbol, RF, GB, XGBoost (Juan David)
│   ├── ann.py                # Red neuronal MLP (Jack)
│   ├── evaluate.py           # Métricas y gráficas (Juan David)
│   ├── scoring.py            # Ranking y priorización (Jack)
│   ├── predict.py            # Predicción para el dashboard (Jack)
│   └── pipeline.py           # Orquestador (compartido)
├── data/raw/Leads.csv
├── data/processed/leads_scored.csv
├── models/modelo_produccion.joblib
├── reports/                  # metrics.json, figuras, importancia
└── docs/                     # Colaboración, prompts de git, despliegue y guía del dashboard
```

Guía de uso de cada pestaña: [docs/GUIA_DASHBOARD.md](docs/GUIA_DASHBOARD.md).

## Cómo ejecutarlo

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.eda                  # figuras del análisis exploratorio
python -m src.pipeline             # entrena los 7 modelos, unos 5-8 minutos
streamlit run app/streamlit_app.py # abre el dashboard en http://localhost:8501
```

## Despliegue en Streamlit Community Cloud

1. Subir el repo a GitHub con `models/`, `reports/` y `data/` ya generados.
2. Entrar a <https://share.streamlit.io> con la cuenta de GitHub → **Create app**.
3. Repositorio: `lead-scoring-predictivo` · Rama: `main` · Archivo: `app/streamlit_app.py`.
4. En *Advanced settings*, Python **3.12** → **Deploy**.

> El modelo `.joblib` solo carga con la misma versión de scikit-learn con la que se entrenó.
> Por eso hay que ejecutar `python -m src.pipeline` **con el `requirements.txt` del repo** antes de subir `models/`.
