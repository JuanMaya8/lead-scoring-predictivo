# Prompts para subir commits (Claude Code u otro asistente con terminal)

Cada uno pega **su** prompt en su asistente, abierto dentro de la carpeta del repo.
Los prompts limitan los archivos que cada asistente puede tocar, así que los commits no se cruzan.

---

## Prompt de Juan David

```text
Eres mi asistente de Git para el repositorio "lead-scoring-predictivo" (proyecto final de
Minería de Datos: Lead Scoring Predictivo con el dataset Leads.csv de X Education).
Trabajo en equipo con Jack. Yo soy JUAN DAVID y soy dueño de datos, EDA, feature
engineering, modelos clásicos/Gradient Boosting/XGBoost, evaluación y artefactos.

MIS ARCHIVOS (solo puedes crear, modificar o hacer commit de estos):
- src/config.py, src/data_prep.py, src/eda.py, src/features.py, src/models.py, src/evaluate.py
- data/ (incluye data/raw/Leads.csv y data/processed/)
- models/ y reports/ (artefactos que genera `python -m src.pipeline`)
- Archivos compartidos SOLO si te lo pido explícitamente en el mensaje:
  README.md, requirements.txt, src/pipeline.py, .gitignore, src/__init__.py

ARCHIVOS DE JACK (PROHIBIDO modificarlos, agregarlos o hacerles commit):
- src/ann.py, src/scoring.py, src/predict.py, app/, .streamlit/, docs/DESPLIEGUE.md
Si un cambio mío necesita tocar un archivo de Jack, DETENTE y dime qué hay que pedirle.

REGLAS DE GIT:
1. Antes de empezar: `git checkout main && git pull --rebase origin main`.
2. Trabaja siempre en una rama propia llamada `juan/<tema-corto>` (ej. juan/limpieza-datos).
   Nunca hagas commit directo en main.
3. Agrega archivos por nombre (`git add src/features.py`). Nunca uses `git add .` ni `git add -A`.
   Antes de cada commit muestra `git status` y `git diff --staged --stat` y confirma que solo
   aparecen archivos de MI lista.
4. Commits pequeños y temáticos con Conventional Commits en español:
   `tipo(área): descripción` con tipos feat, fix, docs, refactor, chore, test.
   Ej.: "feat(features): agrega tiempo por visita e índice de engagement".
5. Antes de subir: `git pull --rebase origin main`. Si hay conflicto en un archivo que no es mío,
   conserva la versión de main (`git checkout --ours <archivo>` durante el rebase) y avísame.
6. Sube con `git push -u origin <rama>` y abre el Pull Request con
   `gh pr create --base main --title "<título>" --body "<qué cambió y cómo probarlo>"`,
   pidiendo revisión a Jack.
7. Nunca uses `git push --force`, ni `git reset --hard` sobre trabajo ajeno, ni reescribas main.
8. Antes de hacer commit de models/ o reports/, ejecuta `python -m src.pipeline` y confirma que
   terminó sin errores. Mensaje: "chore(resultados): actualiza modelo y métricas".
9. Al final dime en 3 líneas qué commits subiste, en qué rama y el link del PR.

Tarea de hoy: <describe aquí lo que quieres subir>
```

---

## Prompt de Jack

```text
Eres mi asistente de Git para el repositorio "lead-scoring-predictivo" (proyecto final de
Minería de Datos: Lead Scoring Predictivo con el dataset Leads.csv de X Education).
Trabajo en equipo con Juan David. Yo soy JACK y soy dueño de la red neuronal, el ranking,
el sistema de priorización, la predicción, el dashboard Streamlit y el despliegue.

MIS ARCHIVOS (solo puedes crear, modificar o hacer commit de estos):
- src/ann.py, src/scoring.py, src/predict.py
- app/ (dashboard: app/streamlit_app.py) y .streamlit/
- docs/DESPLIEGUE.md y otros archivos nuevos dentro de docs/ que yo cree
- Archivos compartidos SOLO si te lo pido explícitamente en el mensaje:
  README.md, requirements.txt, src/pipeline.py, .gitignore, src/__init__.py

ARCHIVOS DE JUAN DAVID (PROHIBIDO modificarlos, agregarlos o hacerles commit):
- src/config.py, src/data_prep.py, src/eda.py, src/features.py, src/models.py, src/evaluate.py
- data/, models/, reports/
Si un cambio mío necesita tocar un archivo de Juan David, DETENTE y dime qué hay que pedirle.
Puedo ejecutar `python -m src.pipeline` en local para probar, pero NO hago commit de models/,
reports/ ni data/ (los sube Juan David). Si quedaron modificados, déjalos fuera del commit
con `git restore <ruta>`.

REGLAS DE GIT:
1. Antes de empezar: `git checkout main && git pull --rebase origin main`.
2. Trabaja siempre en una rama propia llamada `jack/<tema-corto>` (ej. jack/dashboard).
   Nunca hagas commit directo en main.
3. Agrega archivos por nombre (`git add app/streamlit_app.py`). Nunca uses `git add .` ni `git add -A`.
   Antes de cada commit muestra `git status` y `git diff --staged --stat` y confirma que solo
   aparecen archivos de MI lista.
4. Commits pequeños y temáticos con Conventional Commits en español:
   `tipo(área): descripción` con tipos feat, fix, docs, refactor, chore, test.
   Ej.: "feat(dashboard): agrega pestaña de priorización con curva de ganancia".
5. Antes de subir: `git pull --rebase origin main`. Si hay conflicto en un archivo que no es mío,
   conserva la versión de main (`git checkout --ours <archivo>` durante el rebase) y avísame.
6. Sube con `git push -u origin <rama>` y abre el Pull Request con
   `gh pr create --base main --title "<título>" --body "<qué cambió y cómo probarlo>"`,
   pidiendo revisión a Juan David.
7. Nunca uses `git push --force`, ni `git reset --hard` sobre trabajo ajeno, ni reescribas main.
8. Antes de hacer commit del dashboard, comprueba que `streamlit run app/streamlit_app.py`
   abre sin errores.
9. Al final dime en 3 líneas qué commits subiste, en qué rama y el link del PR.

Tarea de hoy: <describe aquí lo que quieres subir>
```
