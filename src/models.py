"""Catálogo de modelos de clasificación y sus espacios de hiperparámetros.

Responsable: Juan David (la red neuronal vive en src/ann.py, de Jack).

Modelos:
  1. Regresión Logística   (línea base, interpretable)
  2. KNN                   (visto en clase, basado en distancias)
  3. Árbol de Decisión
  4. Random Forest         (bagging de árboles)
  5. Gradient Boosting     (boosting de scikit-learn)
  6. XGBoost               (boosting optimizado con regularización)
  7. Red Neuronal (MLP)    (src/ann.py)
"""
from __future__ import annotations

import warnings

from scipy.stats import loguniform, randint, uniform
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from src import config
from src.ann import ESPACIO_RED_NEURONAL, crear_red_neuronal

try:
    from xgboost import XGBClassifier

    XGBOOST_DISPONIBLE = True
except ImportError:  # pragma: no cover
    XGBOOST_DISPONIBLE = False
    warnings.warn("xgboost no está instalado: pip install xgboost")

RS = config.RANDOM_STATE


def catalogo_modelos() -> dict:
    """Devuelve {nombre: (estimador, espacio_de_busqueda)}."""
    modelos = {
        "Regresión Logística": (
            LogisticRegression(max_iter=2000, random_state=RS),
            {"model__C": loguniform(1e-3, 1e2)},
        ),
        "KNN": (
            KNeighborsClassifier(),
            {
                "model__n_neighbors": list(range(5, 52, 2)),  # k impar
                "model__weights": ["uniform", "distance"],
                "model__p": [1, 2],  # 1 = Manhattan, 2 = Euclidiana
            },
        ),
        "Árbol de Decisión": (
            DecisionTreeClassifier(random_state=RS),
            {
                "model__max_depth": randint(3, 15),
                "model__min_samples_leaf": randint(5, 60),
                "model__criterion": ["gini", "entropy"],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(n_estimators=300, n_jobs=-1, random_state=RS),
            {
                "model__max_depth": [None, 8, 12, 16, 20],
                "model__min_samples_leaf": randint(1, 20),
                "model__max_features": ["sqrt", 0.3, 0.5],
            },
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=RS),
            {
                "model__n_estimators": randint(100, 500),
                "model__learning_rate": loguniform(0.01, 0.3),
                "model__max_depth": randint(2, 6),
                "model__subsample": uniform(0.6, 0.4),
            },
        ),
    }
    if XGBOOST_DISPONIBLE:
        modelos["XGBoost"] = (
            XGBClassifier(
                eval_metric="logloss",
                tree_method="hist",
                device="cpu",
                n_jobs=1,
                random_state=RS,
            ),
            {
                "model__n_estimators": randint(150, 600),
                "model__learning_rate": loguniform(0.01, 0.3),
                "model__max_depth": randint(2, 8),
                "model__subsample": uniform(0.6, 0.4),
                "model__colsample_bytree": uniform(0.5, 0.5),
                "model__min_child_weight": randint(1, 10),
                "model__reg_lambda": loguniform(1e-2, 10),
            },
        )
    modelos["Red Neuronal (MLP)"] = (crear_red_neuronal(), ESPACIO_RED_NEURONAL)
    return modelos
