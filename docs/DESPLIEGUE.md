# Despliegue del dashboard en Streamlit Community Cloud

Responsable: Jack.

## Requisitos previos

- El repositorio `lead-scoring-predictivo` es **público** en GitHub.
- En `main` ya están los artefactos que genera `python -m src.pipeline` (los sube Juan David):

| Archivo | Para qué lo usa el dashboard |
|---|---|
| `models/modelo_produccion.joblib` | Puntuar leads nuevos (pestaña *Puntuar leads*) |
| `reports/metrics.json` | Métricas, umbral óptimo, lift y resumen por prioridad |
| `reports/predicciones_test.csv` | Curvas ROC y matriz de confusión interactiva |
| `reports/importancia_variables.csv` | Pestaña *Factores clave* |
| `data/processed/leads_scored.csv` | Ranking, filtros y todos los gráficos de negocio |

Si falta alguno, el dashboard muestra exactamente qué archivo falta en lugar de fallar.

## 1. Probar en local

```bash
python -m venv .venv
.venv\Scripts\activate             # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py # http://localhost:8501
```

Revisar las 7 pestañas: Resumen, Ranking, Priorización, Modelos, Factores clave, Puntuar leads y Metodología.

## 2. Publicar

1. Ir a <https://share.streamlit.io> → **Continue with GitHub** → autorizar.
2. **Create app** → *Deploy a public app from GitHub*.
3. Completar:
   - Repository: `JuanMaya8/lead-scoring-predictivo`
   - Branch: `main`
   - Main file path: `app/streamlit_app.py`
   - App URL: `lead-scoring-xeducation` (o el nombre que esté libre)
4. **Advanced settings** → Python version **3.12** → Save.
5. **Deploy**. La primera instalación tarda de 2 a 4 minutos.
6. Copiar la URL pública (`https://<nombre>.streamlit.app`) al README.

Cada merge a `main` vuelve a desplegar la app automáticamente.

## Estructura del dashboard

```
app/
├── streamlit_app.py   # Punto de entrada: configuración, barra lateral y pestañas
├── data.py            # Carga cacheada de artefactos (CSV, JSON, modelo)
├── filters.py         # Filtros de la barra lateral
├── ui/
│   ├── theme.py       # Paleta validada, CSS global y plantilla de Plotly
│   └── components.py  # Tarjetas KPI, hero, tarjetas de prioridad, medidor
└── views/             # Una vista por pestaña, cada una con render(data, leads)
.streamlit/config.toml # Tema claro y límites del servidor
```

## Errores comunes

| Síntoma | Causa | Solución |
|---|---|---|
| "No se encontraron los resultados del modelo" + lista de archivos | Faltan artefactos en `reports/`, `models/` o `data/processed/` | Ejecutar el pipeline y que Juan David suba esas carpetas |
| Aviso "No se pudo cargar modelo_produccion.joblib" | El modelo se entrenó con otra versión de scikit-learn | Reentrenar con `pip install -r requirements.txt` y volver a subir `models/` |
| `ModuleNotFoundError: app` o `src` | Ruta del archivo principal mal configurada | Main file path debe ser exactamente `app/streamlit_app.py` |
| La app "se duerme" | Plan gratuito tras días sin uso | Abrirla y pulsar **Wake up** antes de la exposición |
