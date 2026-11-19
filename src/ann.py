"""Artificial neural network (multilayer perceptron) for lead scoring.

Owner: Jack.

Why scikit-learn's ``MLPClassifier``:
  * It is a genuine feed-forward neural network: hidden layers, ReLU/tanh
    activations, the Adam optimizer, backpropagation and early stopping.
  * It shares the scikit-learn estimator interface with every other model,
    so it plugs into the same ``Pipeline`` and is persisted with ``joblib``.
  * It is lightweight and deploys cleanly on Streamlit Community Cloud
    (TensorFlow weighs >500 MB and often exceeds the free tier limits).

Baseline architecture::

    input (~60 features after one-hot encoding)
      -> hidden layer 1 (64 neurons, ReLU)
      -> hidden layer 2 (32 neurons, ReLU)
      -> output (1 neuron, logistic = conversion probability)
"""
from __future__ import annotations

from scipy.stats import loguniform
from sklearn.neural_network import MLPClassifier

from src import config

# Baseline hyperparameters (the randomized search explores around them).
HIDDEN_LAYERS: tuple[int, ...] = (64, 32)
L2_PENALTY = 1e-3          # alpha: L2 regularization against overfitting
LEARNING_RATE = 1e-3
BATCH_SIZE = 64
MAX_EPOCHS = 400
VALIDATION_FRACTION = 0.10  # share of the training fold held out for early stopping
PATIENCE = 20               # epochs without improvement before stopping


def build_mlp_classifier() -> MLPClassifier:
    """Return an untrained MLP with the baseline architecture."""
    return MLPClassifier(
        hidden_layer_sizes=HIDDEN_LAYERS,
        activation="relu",
        solver="adam",
        alpha=L2_PENALTY,
        learning_rate_init=LEARNING_RATE,
        batch_size=BATCH_SIZE,
        max_iter=MAX_EPOCHS,
        early_stopping=True,
        validation_fraction=VALIDATION_FRACTION,
        n_iter_no_change=PATIENCE,
        random_state=config.RANDOM_STATE,
    )


# Search space for RandomizedSearchCV. Keys are prefixed with ``model__``
# because the estimator lives inside a Pipeline step named "model".
MLP_PARAM_SPACE: dict[str, object] = {
    "model__hidden_layer_sizes": [(32,), (64,), (64, 32), (128, 64), (64, 32, 16)],
    "model__alpha": loguniform(1e-5, 1e-1),
    "model__learning_rate_init": loguniform(1e-4, 1e-2),
    "model__activation": ["relu", "tanh"],
}


# Backward-compatible names used by src/models.py (owned by Juan David).
crear_red_neuronal = build_mlp_classifier
ESPACIO_RED_NEURONAL = MLP_PARAM_SPACE

__all__ = [
    "build_mlp_classifier",
    "MLP_PARAM_SPACE",
    "crear_red_neuronal",
    "ESPACIO_RED_NEURONAL",
]
