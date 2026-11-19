# Flujo de trabajo en Git: Juan David y Jack

Cada persona trabaja en **su propia rama** y solo modifica **sus propios archivos**.
Los cambios llegan a `main` por Pull Request (PR). Así los commits nunca se cruzan.

## 1. Dueños de cada archivo

| Archivo / carpeta | Dueño | Qué contiene |
|---|---|---|
| `src/config.py` | Juan David | Rutas, variables y parámetros |
| `src/data_prep.py` | Juan David | Carga y limpieza de datos |
| `src/eda.py` | Juan David | Análisis exploratorio |
| `src/features.py` | Juan David | Feature engineering y preprocesador |
| `src/models.py` | Juan David | Modelos clásicos, Gradient Boosting y XGBoost |
| `src/evaluate.py` | Juan David | Métricas y gráficas |
| `data/`, `models/`, `reports/` | Juan David | Dataset y artefactos generados por el pipeline |
| `src/ann.py` | Jack | Red neuronal (MLP) |
| `src/scoring.py` | Jack | Ranking y priorización |
| `src/predict.py` | Jack | Carga del modelo y predicción |
| `app/`, `.streamlit/` | Jack | Dashboard Streamlit |
| `docs/DESPLIEGUE.md` | Jack | Guía de despliegue |
| `README.md`, `requirements.txt`, `src/pipeline.py`, `.gitignore`, `src/__init__.py`, `docs/COLABORACION.md`, `docs/PROMPTS_GIT.md` | **Compartido** | Se cambian solo avisando al otro |

**Regla de oro:** si necesitas cambiar un archivo del otro, no lo toques. Avísale o abre un *issue*.

## 2. Configuración inicial (una sola vez)

**Juan David** crea el repo `lead-scoring-predictivo` en GitHub (público, sin README) y agrega a Jack:
*Settings → Collaborators → Add people*.

Cada uno en su computador:

```bash
git clone https://github.com/<usuario-juan>/lead-scoring-predictivo.git
cd lead-scoring-predictivo
git config user.name  "Tu Nombre"
git config user.email "tu-correo-de-github@ejemplo.com"
python -m venv .venv
# Windows: .venv\Scripts\activate    Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
```

## 3. Ciclo de trabajo diario

```bash
git checkout main && git pull --rebase origin main     # 1. traer lo último
git checkout -b juan/feature-engineering               # 2. rama propia (jack/... para Jack)
# ... trabajar SOLO en tus archivos ...
git add src/features.py                                # 3. agregar archivos por nombre (nunca "git add .")
git commit -m "feat(features): variables de engagement y canal"
git pull --rebase origin main                          # 4. sincronizar antes de subir
git push -u origin juan/feature-engineering            # 5. subir la rama
# 6. abrir el Pull Request en GitHub → el otro lo revisa → Merge → borrar la rama
```

## 4. Formato de commits (Conventional Commits)

`tipo(área): descripción en presente y en español`

| Tipo | Uso |
|---|---|
| `feat` | Funcionalidad nueva |
| `fix` | Corrección de un error |
| `docs` | Documentación |
| `refactor` | Reorganizar código sin cambiar el resultado |
| `chore` | Configuración, dependencias, artefactos |
| `test` | Pruebas |

Ejemplos: `feat(modelos): agrega XGBoost con búsqueda aleatoria`, `fix(dashboard): corrige filtro de prioridad`.

## 5. Orden sugerido de entregas

| Ronda | Juan David (`juan/...`) | Jack (`jack/...`) |
|---|---|---|
| 0 | `chore: estructura inicial` (archivos compartidos) | Clona el repo |
| 1 | `src/config.py`, `src/data_prep.py`, `data/raw/Leads.csv` | `src/ann.py` |
| 2 | `src/eda.py` y sus figuras | `src/scoring.py`, `src/predict.py` |
| 3 | `src/features.py`, `src/evaluate.py` | `app/streamlit_app.py`, `.streamlit/` |
| 4 | `src/models.py`, `src/pipeline.py` (después del merge de `ann.py`) | Revisa el PR de Juan |
| 5 | Ejecuta `python -m src.pipeline` y sube `models/`, `reports/`, `data/processed/` | Despliega en Streamlit Cloud y escribe `docs/DESPLIEGUE.md` |

## 6. Si aparece un conflicto

```bash
git pull --rebase origin main
# Git marca el archivo en conflicto. Si NO es tuyo: conserva la versión de main
git checkout --ours ruta/del/archivo     # en un rebase: "ours" = main, "theirs" = tus commits
git add ruta/del/archivo
git rebase --continue
```

Si dudas, **no uses `git push --force`**. Escríbele al compañero y resuélvanlo juntos.
