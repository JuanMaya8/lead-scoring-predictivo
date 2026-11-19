"""Métricas de evaluación y gráficas.

Responsable: Juan David.

Métricas (clase positiva = 1 = lead convertido):
  Accuracy, Precision, Recall (sensibilidad), Especificidad, F1,
  ROC-AUC, PR-AUC (Average Precision), Log Loss, Brier, MCC,
  Kappa de Cohen, estadístico KS y matriz de confusión.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    log_loss,
    matthews_corrcoef,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src import config


def calcular_metricas(y_true, y_prob, umbral: float = 0.5) -> dict:
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= umbral).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return {
        "umbral": round(float(umbral), 4),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "especificidad": tn / (tn + fp) if (tn + fp) else 0.0,
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "log_loss": log_loss(y_true, np.clip(y_prob, 1e-7, 1 - 1e-7)),
        "brier": brier_score_loss(y_true, y_prob),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "ks": float(np.max(tpr - fpr)),
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
    }


def umbrales_optimos(y_true, y_prob) -> dict:
    """Umbrales calculados sobre predicciones out-of-fold del ENTRENAMIENTO
    (nunca sobre el test, para no sobreajustar el punto de corte)."""
    fpr, tpr, thr_roc = roc_curve(y_true, y_prob)
    j = tpr - fpr  # índice de Youden = sensibilidad + especificidad - 1
    umbral_youden = float(thr_roc[np.argmax(j)])

    prec, rec, thr_pr = precision_recall_curve(y_true, y_prob)
    f1 = 2 * prec[:-1] * rec[:-1] / np.clip(prec[:-1] + rec[:-1], 1e-12, None)
    umbral_f1 = float(thr_pr[np.argmax(f1)])
    return {"youden": min(umbral_youden, 1.0), "f1": umbral_f1}


def tabla_lift(y_true, y_prob, n_grupos: int = 10) -> pd.DataFrame:
    """Tabla de ganancia/lift por deciles (decil 1 = leads con mayor score)."""
    df = pd.DataFrame({"y": np.asarray(y_true), "p": np.asarray(y_prob)})
    df = df.sort_values("p", ascending=False).reset_index(drop=True)
    df["decil"] = pd.qcut(np.arange(len(df)), n_grupos, labels=range(1, n_grupos + 1)).astype(int)
    tasa_base = df["y"].mean()
    t = df.groupby("decil").agg(leads=("y", "size"), convertidos=("y", "sum"), score_min=("p", "min"))
    t["tasa_conversion"] = t["convertidos"] / t["leads"]
    t["lift"] = t["tasa_conversion"] / tasa_base
    t["porc_leads_acum"] = t["leads"].cumsum() / t["leads"].sum()
    t["ganancia_acum"] = t["convertidos"].cumsum() / t["convertidos"].sum()
    return t.reset_index()


# ---------------------------------------------------------------------------
# Gráficas para el informe (reports/figures)
# ---------------------------------------------------------------------------
def _guardar(fig, nombre):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / nombre, dpi=130)
    plt.close(fig)


def graficar_roc(y_true, probs: dict):
    fig, ax = plt.subplots(figsize=(7, 6))
    for nombre, p in probs.items():
        fpr, tpr, _ = roc_curve(y_true, p)
        ax.plot(fpr, tpr, label=f"{nombre} (AUC={roc_auc_score(y_true, p):.3f})")
    ax.plot([0, 1], [0, 1], "k--", label="Azar (AUC=0.5)")
    ax.set(xlabel="Tasa de falsos positivos (FPR)", ylabel="Tasa de verdaderos positivos (TPR / Recall)",
           title="Curvas ROC - conjunto de prueba")
    ax.legend(loc="lower right", fontsize=8)
    _guardar(fig, "curvas_roc.png")


def graficar_pr(y_true, probs: dict):
    fig, ax = plt.subplots(figsize=(7, 6))
    for nombre, p in probs.items():
        prec, rec, _ = precision_recall_curve(y_true, p)
        ax.plot(rec, prec, label=f"{nombre} (AP={average_precision_score(y_true, p):.3f})")
    ax.axhline(np.mean(y_true), color="k", ls="--", label="Tasa base")
    ax.set(xlabel="Recall", ylabel="Precision", title="Curvas Precision-Recall - prueba")
    ax.legend(fontsize=8)
    _guardar(fig, "curvas_precision_recall.png")


def graficar_matriz_confusion(m: dict, nombre_modelo: str):
    cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.imshow(cm, cmap="Blues")
    etiquetas = [["TN", "FP"], ["FN", "TP"]]
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{etiquetas[i][j]}\n{cm[i, j]}", ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1], ["Pred. No convierte", "Pred. Convierte"])
    ax.set_yticks([0, 1], ["Real No convierte", "Real Convierte"])
    ax.set_title(f"Matriz de confusión - {nombre_modelo}\n(umbral = {m['umbral']:.2f})")
    _guardar(fig, "matriz_confusion.png")


def graficar_lift(t: pd.DataFrame):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.5))
    a1.bar(t["decil"], t["lift"], color="#2a6fdb")
    a1.axhline(1, color="k", ls="--")
    a1.set(xlabel="Decil (1 = score más alto)", ylabel="Lift", title="Lift por decil")
    a2.plot([0, *t["porc_leads_acum"]], [0, *t["ganancia_acum"]], marker="o", label="Modelo")
    a2.plot([0, 1], [0, 1], "k--", label="Azar")
    a2.set(xlabel="% de leads contactados", ylabel="% de conversiones capturadas", title="Curva de ganancia acumulada")
    a2.legend()
    _guardar(fig, "lift_ganancia.png")


def graficar_calibracion(y_true, probs: dict):
    fig, ax = plt.subplots(figsize=(6, 5.5))
    for nombre, p in probs.items():
        frac, media = calibration_curve(y_true, p, n_bins=10)
        ax.plot(media, frac, marker="o", label=nombre)
    ax.plot([0, 1], [0, 1], "k--", label="Calibración perfecta")
    ax.set(xlabel="Probabilidad predicha", ylabel="Fracción real de convertidos", title="Curva de calibración")
    ax.legend(fontsize=8)
    _guardar(fig, "calibracion.png")


def graficar_importancia(imp: pd.DataFrame, top: int = 15):
    d = imp.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(d["variable"], d["importancia"], xerr=d["desviacion"], color="#2a6fdb")
    ax.set(xlabel="Caída del ROC-AUC al permutar la variable", title="Importancia de variables (permutación)")
    _guardar(fig, "importancia_variables.png")


def graficar_comparacion(tabla: pd.DataFrame):
    cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    fig, ax = plt.subplots(figsize=(11, 5))
    x = np.arange(len(tabla))
    w = 0.16
    for i, c in enumerate(cols):
        ax.bar(x + (i - 2) * w, tabla[c], w, label=c)
    ax.set_xticks(x, tabla.index, rotation=20, ha="right")
    ax.set_ylim(0.5, 1)
    ax.set_title("Comparación de modelos - conjunto de prueba")
    ax.legend(ncol=5, fontsize=8)
    _guardar(fig, "comparacion_modelos.png")
