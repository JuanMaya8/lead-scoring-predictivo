"""Red neuronal artificial (perceptrón multicapa) para lead scoring.

Responsable: Jack.

Se usa MLPClassifier de scikit-learn porque:
  - es una red neuronal feed-forward real (capas ocultas, ReLU, Adam,
    retropropagación y early stopping),
  - comparte la misma interfaz que los demás modelos (Pipeline + joblib),
  - es liviana y se despliega sin problemas en Streamlit Cloud
    (TensorFlow pesa más de 500 MB y suele fallar en el plan gratuito).

Arquitectura base:
    entrada (~60 variables tras one-hot)
      -> capa oculta 1 (64 neuronas, ReLU)
      -> capa oculta 2 (32 neuronas, ReLU)
      -> salida (1 neurona, sigmoide = probabilidad de conversión)
"""
from __future__ import annotations

from scipy.stats import loguniform
from sklearn.neural_network import MLPClassifier

from src import config


def crear_red_neuronal() -> MLPClassifier:
    return MLPClassifier(
        hidden_layer_sizes=(64, 32),
        activation="relu",
        solver="adam",
        alpha=1e-3,                # regularización L2 (reduce sobreajuste)
        learning_rate_init=1e-3,
        batch_size=64,
        max_iter=400,
        early_stopping=True,       # separa 10% del train como validación
        validation_fraction=0.1,
        n_iter_no_change=20,
        random_state=config.RANDOM_STATE,
    )


# Espacio de búsqueda de hiperparámetros (RandomizedSearchCV)
ESPACIO_RED_NEURONAL = {
    "model__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64), (64, 32, 16)],
    "model__alpha": loguniform(1e-5, 1e-1),
    "model__learning_rate_init": loguniform(1e-4, 1e-2),
    "model__activation": ["relu", "tanh"],
}
