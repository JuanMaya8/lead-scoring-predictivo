# Despliegue del dashboard en Streamlit Community Cloud

Responsable: Jack.

## Requisitos previos
- El repo `lead-scoring-predictivo` es **público** en GitHub (el plan gratuito lo exige o pide permisos).
- En `main` ya están: `models/modelo_produccion.joblib`, `reports/metrics.json`,
  `reports/predicciones_test.csv`, `reports/importancia_variables.csv` y `data/processed/leads_scored.csv`.
  Los genera `python -m src.pipeline` y los sube Juan David.

## Pasos
1. Probar en local: `streamlit run app/streamlit_app.py` y revisar las 6 pestañas.
2. Ir a <https://share.streamlit.io> → **Continue with GitHub** → autorizar.
3. **Create app** → *Deploy a public app from GitHub*.
4. Completar:
   - Repository: `<usuario>/lead-scoring-predictivo`
   - Branch: `main`
   - Main file path: `app/streamlit_app.py`
   - App URL: `lead-scoring-xeducation` (o el nombre que esté libre)
5. **Advanced settings** → Python version **3.12** → Save.
6. **Deploy**. La primera instalación tarda 2-4 minutos.
7. Copiar la URL pública (`https://<nombre>.streamlit.app`) al README.

## Errores comunes
| Síntoma | Causa | Solución |
|---|---|---|
| "No se encontraron los resultados del modelo" | Faltan archivos en `reports/` o `data/processed/` | Ejecutar el pipeline y hacer commit de esas carpetas |
| Aviso "No se pudo cargar modelo_produccion.joblib" | El modelo se entrenó con otra versión de scikit-learn | Reentrenar en local con `pip install -r requirements.txt` y volver a subir `models/` |
| `ModuleNotFoundError: src` | Archivo principal mal configurado | Main file path debe ser exactamente `app/streamlit_app.py` |
| La app se "duerme" | Plan gratuito tras días sin uso | Abrirla y pulsar "Wake up" antes de la exposición |
