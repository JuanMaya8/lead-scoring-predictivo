"""Análisis exploratorio de datos (EDA) -> reports/figures/eda_*.png

Responsable: Juan David.

Uso:  python -m src.eda
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src import config
from src.data_prep import cargar_datos, resumen_calidad


def _guardar(fig, nombre):
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(config.FIGURES_DIR / nombre, dpi=120)
    plt.close(fig)


def conversion_por(df, col, top=12):
    g = (df.groupby(col)[config.TARGET].agg(["mean", "size"]).query("size >= 20")
         .sort_values("size", ascending=False).head(top).sort_values("mean"))
    fig, ax = plt.subplots(figsize=(8, 0.45 * len(g) + 1.5))
    ax.barh(g.index.astype(str), g["mean"], color="#2a6fdb")
    ax.axvline(df[config.TARGET].mean(), color="k", ls="--", label="Promedio")
    for i, (m, n) in enumerate(zip(g["mean"], g["size"])):
        ax.text(m + 0.01, i, f"{m:.0%} (n={n})", va="center", fontsize=8)
    ax.set(xlabel="Tasa de conversión", title=f"Conversión por {col}", xlim=(0, 1.15))
    ax.legend()
    _guardar(fig, f"eda_conversion_{col.replace(' ', '_').lower()[:30]}.png")


def main():
    df = cargar_datos().replace("Select", np.nan)
    calidad = resumen_calidad(df)
    calidad.to_csv(config.REPORTS_DIR / "calidad_datos.csv")
    print(calidad.head(15).to_string())
    print(f"\nTasa de conversión: {df[config.TARGET].mean():.2%}")

    # Valores nulos
    nulos = calidad["porc_nulos"][calidad["porc_nulos"] > 0]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(nulos.index[::-1], nulos.values[::-1], color="#d62728")
    ax.axvline(config.MAX_PORC_NULOS * 100, color="k", ls="--", label="Umbral de eliminación")
    ax.set(xlabel="% de nulos (incluye 'Select')", title="Valores faltantes por variable")
    ax.legend()
    _guardar(fig, "eda_nulos.png")

    # Variable objetivo
    fig, ax = plt.subplots(figsize=(5, 4))
    conteo = df[config.TARGET].value_counts().sort_index()
    ax.bar(["No convirtió (0)", "Convirtió (1)"], conteo.values, color=["#1f77b4", "#d62728"])
    for i, v in enumerate(conteo.values):
        ax.text(i, v, f"{v:,} ({v / conteo.sum():.0%})", ha="center", va="bottom")
    ax.set_title("Distribución de la variable objetivo")
    _guardar(fig, "eda_target.png")

    # Numéricas vs conversión
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    for ax, col in zip(axes, config.COLS_NUMERICAS_BASE):
        datos = [df.loc[df[config.TARGET] == k, col].dropna().clip(upper=df[col].quantile(0.99)) for k in (0, 1)]
        ax.boxplot(datos, tick_labels=["No convirtió", "Convirtió"])
        ax.set_title(col)
    _guardar(fig, "eda_numericas_vs_target.png")

    for col in ["Lead Origin", "Lead Source", "Last Activity", "What is your current occupation", "Specialization"]:
        conversion_por(df, col)

    # Correlación entre numéricas
    corr = df[config.COLS_NUMERICAS_BASE + [config.TARGET]].corr()
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
    ax.set_xticks(range(len(corr)), corr.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(corr)), corr.columns, fontsize=8)
    for i in range(len(corr)):
        for j in range(len(corr)):
            ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center", va="center", fontsize=8)
    fig.colorbar(im)
    ax.set_title("Correlación de Pearson")
    _guardar(fig, "eda_correlacion.png")
    print("[eda] figuras guardadas en reports/figures/")


if __name__ == "__main__":
    main()
