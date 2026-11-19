"""Pipeline completo: datos -> modelos -> evaluación -> scoring -> artefactos.

Uso (desde la raíz del repo):
    python -m src.pipeline            # búsqueda completa de hiperparámetros
    python -m src.pipeline --rapido   # pocas iteraciones (para probar)

Archivo compartido: cualquier cambio se acuerda entre Juan David y Jack.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.inspection import permutation_importance
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline

from src import config, evaluate
from src.data_prep import cargar_datos, columnas_a_eliminar
from src.features import construir_preprocesador, tipos_de_columnas
from src.models import catalogo_modelos
from src.predict import preparar_X
from src.scoring import puntuar_leads, resumen_por_prioridad


def main(rapido: bool = False):
    t0 = time.time()
    for d in (config.DATA_PROCESSED, config.MODELS_DIR, config.FIGURES_DIR):
        d.mkdir(parents=True, exist_ok=True)

    # 1. Datos y división 80/20 estratificada (antes de cualquier ajuste)
    df = cargar_datos()
    df = df.drop_duplicates(subset=[config.ID_COL])
    y = df[config.TARGET].astype(int)
    df_train, df_test, y_train, y_test = train_test_split(
        df, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_STATE
    )

    # 2. Limpieza y feature engineering (reglas aprendidas solo con train)
    cols_eliminar = columnas_a_eliminar(df_train)
    print(f"[limpieza] columnas eliminadas ({len(cols_eliminar)}): {cols_eliminar}")
    X_train = preparar_X(df_train, cols_eliminar)
    X_test = preparar_X(df_test, cols_eliminar)
    numericas, categoricas = tipos_de_columnas(X_train)
    print(f"[features] {len(numericas)} numéricas, {len(categoricas)} categóricas")

    cv = StratifiedKFold(config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_STATE)
    n_iter = 3 if rapido else config.N_ITER_BUSQUEDA

    # 3. Entrenamiento con búsqueda de hiperparámetros (CV 5-fold, ROC-AUC)
    resultados, probs_test, mejores = {}, {}, {}
    for nombre, (estimador, espacio) in catalogo_modelos().items():
        ti = time.time()
        pipe = Pipeline([("prep", construir_preprocesador(numericas, categoricas)), ("model", estimador)])
        busqueda = RandomizedSearchCV(
            pipe, espacio, n_iter=n_iter, scoring="roc_auc", cv=cv,
            n_jobs=1 if nombre == "XGBoost" else -1,
            random_state=config.RANDOM_STATE, refit=True,
        )
        busqueda.fit(X_train, y_train)
        p_test = busqueda.predict_proba(X_test)[:, 1]
        p_train = busqueda.predict_proba(X_train)[:, 1]
        m = evaluate.calcular_metricas(y_test, p_test, 0.5)
        m["cv_roc_auc"] = busqueda.best_score_
        m["cv_roc_auc_std"] = float(busqueda.cv_results_["std_test_score"][busqueda.best_index_])
        m["train_roc_auc"] = evaluate.roc_auc_score(y_train, p_train)
        m["mejores_parametros"] = {k.replace("model__", ""): (v if isinstance(v, (int, float, str)) else str(v))
                                   for k, v in busqueda.best_params_.items()}
        m["segundos"] = round(time.time() - ti, 1)
        resultados[nombre], probs_test[nombre], mejores[nombre] = m, p_test, busqueda.best_estimator_
        print(f"[modelo] {nombre:22s} CV-AUC={m['cv_roc_auc']:.4f}  Test-AUC={m['roc_auc']:.4f}  "
              f"F1={m['f1']:.4f}  ({m['segundos']}s)")

    # 4. Selección del mejor modelo por AUC de validación cruzada (no por test)
    mejor = max(resultados, key=lambda k: resultados[k]["cv_roc_auc"])
    print(f"[selección] mejor modelo: {mejor}")

    # 5. Umbral óptimo con predicciones out-of-fold del entrenamiento
    oof_train = cross_val_predict(clone(mejores[mejor]), X_train, y_train, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    umbrales = evaluate.umbrales_optimos(y_train, oof_train)
    umbral = umbrales["youden"]
    m_opt = evaluate.calcular_metricas(y_test, probs_test[mejor], umbral)
    print(f"[umbral] Youden={umbral:.3f}  F1-óptimo={umbrales['f1']:.3f}")

    # 6. Importancia de variables por permutación (sobre columnas originales)
    imp = permutation_importance(mejores[mejor], X_test, y_test, scoring="roc_auc",
                                 n_repeats=5, random_state=config.RANDOM_STATE, n_jobs=-1)
    imp_df = (pd.DataFrame({"variable": X_test.columns, "importancia": imp.importances_mean,
                            "desviacion": imp.importances_std})
              .sort_values("importancia", ascending=False))
    imp_df.to_csv(config.IMPORTANCIA_CSV, index=False)

    # 7. Lift / ganancia del mejor modelo en test
    lift = evaluate.tabla_lift(y_test, probs_test[mejor])

    # 8. Scores honestos para TODOS los leads (out-of-fold, 5 particiones)
    X_all = preparar_X(df, cols_eliminar)
    oof_all = cross_val_predict(clone(mejores[mejor]), X_all, y, cv=cv, method="predict_proba", n_jobs=-1)[:, 1]
    cols_mostrar = ["Prospect ID", "Lead Number", "Lead Origin", "Lead Source", "Last Activity",
                    "What is your current occupation", "Specialization", "City", "TotalVisits",
                    "Total Time Spent on Website", "Page Views Per Visit", config.TARGET]
    scored = puntuar_leads(df[[c for c in cols_mostrar if c in df.columns]].reset_index(drop=True), oof_all)
    scored.to_csv(config.LEADS_SCORED, index=False)
    resumen_prio = resumen_por_prioridad(scored)
    print(resumen_prio.to_string(index=False))

    # 9. Modelo de producción: mejores hiperparámetros reentrenados con el 100% de los datos
    final = clone(mejores[mejor]).fit(X_all, y)
    joblib.dump(
        {"pipeline": final, "modelo": mejor, "umbral": umbral, "cols_eliminar": cols_eliminar,
         "columnas_entrada": list(X_all.columns), "sklearn_version": sklearn.__version__,
         "fecha": datetime.now().isoformat(timespec="seconds")},
        config.MODELO_PRODUCCION, compress=3,
    )

    # 10. Reportes
    pd.DataFrame({"y_real": y_test.values, **{k: v for k, v in probs_test.items()}}).to_csv(config.PRED_TEST_CSV, index=False)
    tabla = pd.DataFrame(resultados).T
    with open(config.METRICS_JSON, "w", encoding="utf-8") as f:
        json.dump({
            "mejor_modelo": mejor,
            "umbrales": umbrales,
            "metricas_umbral_05": resultados,
            "metricas_mejor_umbral_optimo": m_opt,
            "lift": lift.to_dict(orient="records"),
            "resumen_prioridad": resumen_prio.to_dict(orient="records"),
            "columnas_eliminadas": cols_eliminar,
            "n_train": int(len(X_train)), "n_test": int(len(X_test)),
            "tasa_conversion": float(y.mean()),
        }, f, ensure_ascii=False, indent=2, default=float)

    evaluate.graficar_roc(y_test, probs_test)
    evaluate.graficar_pr(y_test, probs_test)
    evaluate.graficar_calibracion(y_test, {k: probs_test[k] for k in list(probs_test)[:4] + [mejor]})
    evaluate.graficar_matriz_confusion(m_opt, mejor)
    evaluate.graficar_lift(lift)
    evaluate.graficar_importancia(imp_df)
    evaluate.graficar_comparacion(tabla[["accuracy", "precision", "recall", "f1", "roc_auc"]].astype(float))

    cols = ["cv_roc_auc", "roc_auc", "accuracy", "precision", "recall", "especificidad", "f1", "pr_auc", "ks"]
    print("\n", tabla[cols].astype(float).round(4).sort_values("cv_roc_auc", ascending=False).to_string())
    print(f"\n[listo] {time.time() - t0:.0f}s. Artefactos en models/, reports/ y data/processed/")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rapido", action="store_true", help="búsqueda corta para probar")
    main(**vars(ap.parse_args()))
