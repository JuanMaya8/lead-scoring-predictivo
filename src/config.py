"""Configuración central del proyecto Lead Scoring Predictivo.

Responsable: Juan David (datos y modelado).
"""
from pathlib import Path

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[1]
DATA_RAW = RAIZ / "data" / "raw" / "Leads.csv"
DATA_PROCESSED = RAIZ / "data" / "processed"
MODELS_DIR = RAIZ / "models"
REPORTS_DIR = RAIZ / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

LEADS_SCORED = DATA_PROCESSED / "leads_scored.csv"
METRICS_JSON = REPORTS_DIR / "metrics.json"
PRED_TEST_CSV = REPORTS_DIR / "predicciones_test.csv"
IMPORTANCIA_CSV = REPORTS_DIR / "importancia_variables.csv"
MODELO_PRODUCCION = MODELS_DIR / "modelo_produccion.joblib"

# ---------------------------------------------------------------------------
# Variables
# ---------------------------------------------------------------------------
TARGET = "Converted"
ID_COL = "Lead Number"
COLS_IDENTIFICADORES = ["Prospect ID", "Lead Number"]

# Variables que el equipo comercial llena DESPUÉS de contactar al lead.
# Usarlas produce "data leakage" (fuga de información): el modelo vería
# parte de la respuesta y el AUC saldría artificialmente alto (~0.98).
# Un lead nuevo todavía no tiene estos datos, así que se excluyen.
COLS_FUGA_INFORMACION = [
    "Tags",
    "Lead Quality",
    "Lead Profile",
    "Last Notable Activity",  # redundante con Last Activity
    "Asymmetrique Activity Index",
    "Asymmetrique Profile Index",
    "Asymmetrique Activity Score",
    "Asymmetrique Profile Score",
]

# Umbrales de limpieza automática
MAX_PORC_NULOS = 0.45        # se elimina la columna si tiene más de 45% nulos
MAX_PORC_CLASE_DOMINANTE = 0.95  # se elimina si una categoría ocupa > 95%

# Columnas binarias Yes/No que se convierten a 1/0
COLS_BINARIAS = [
    "Do Not Email",
    "A free copy of Mastering The Interview",
]

COLS_NUMERICAS_BASE = [
    "TotalVisits",
    "Total Time Spent on Website",
    "Page Views Per Visit",
]

# ---------------------------------------------------------------------------
# Modelado
# ---------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20      # división 80/20 vista en clase
CV_FOLDS = 5
N_ITER_BUSQUEDA = 12  # iteraciones de RandomizedSearchCV por modelo

# ---------------------------------------------------------------------------
# Sistema de priorización (score 0-100)
# ---------------------------------------------------------------------------
NIVELES_PRIORIDAD = [
    # (score mínimo, etiqueta, acción comercial recomendada, SLA)
    (75, "A - Caliente", "Llamada inmediata de un asesor senior", "< 1 hora"),
    (50, "B - Tibio", "Llamada + correo personalizado", "< 24 horas"),
    (25, "C - Templado", "Nurturing: correo/WhatsApp con contenido del curso", "Semanal"),
    (0, "D - Frío", "Campañas automáticas de bajo costo", "Mensual"),
]
