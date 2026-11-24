"""
tune_final.py
=============

Búsqueda conjunta (grid) de hiperparámetros para elegir la configuración final
de DT y RF, en lugar de optimizar cada hiperparámetro por separado. Cada fold
ajusta su propio preprocesador sobre los datos de entrenamiento.

Motivación: los barridos univariados eligen `max_depth` y `min_samples_leaf`
de forma independiente, pero ambos interactúan (un árbol profundo con hojas
grandes se comporta distinto que uno superficial con hojas chicas). Lo mismo
pasa con `max_depth` y `max_features` en el bosque. Este script evalúa la
grilla cruzada con validación cruzada estratificada 5-fold y guarda:

- results/grilla_dt.csv
- results/grilla_rf.csv

y actualiza `results/resumen.json` con la configuración elegida (sin
reentrenar los modelos finales; eso lo hace `finalize.py`).

    python tune_final.py
"""

from __future__ import annotations

import itertools
import json
import time

import numpy as np
import pandas as pd

import experiments as E
from dt_rf import DecisionTree, RandomForest
from preprocessing import Preprocessor, TARGET

CV = 5
SEED = 42


def grid_dt(df_raw) -> pd.DataFrame:
    rows = []
    for depth, leaf in itertools.product([3, 4, 5, 6, 8], [1, 5, 10, 20, 40]):
        make = lambda d=depth, l=leaf: DecisionTree(
            max_depth=d, min_samples_leaf=l, criterion="gini",
            random_state=SEED,
        )
        res = E.cross_validate_with_preprocessing(
            df_raw, TARGET, lambda: Preprocessor(), make, n_splits=CV
        )
        rows.append(
            {"max_depth": depth, "min_samples_leaf": leaf,
             **{k: res[k] for k in ("accuracy", "f1_opt", "roc_auc", "pr_auc",
                                    "leaves", "fit_time")},
             "roc_auc_std": res["roc_auc_std"]}
        )
        print(f"  DT depth={depth} leaf={leaf:2d}  auc={res['roc_auc']:.4f}")
    return pd.DataFrame(rows)


def grid_rf(df_raw) -> pd.DataFrame:
    rows = []
    for depth, mf in itertools.product([3, 5, 8], ["sqrt", 0.3, 0.6]):
        make = lambda d=depth, m=mf: RandomForest(
            n_estimators=100, max_depth=d, max_features=m,
            criterion="gini", n_jobs=1, random_state=SEED,
        )
        res = E.cross_validate_with_preprocessing(
            df_raw, TARGET, lambda: Preprocessor(), make, n_splits=CV
        )
        rows.append(
            {"max_depth": depth, "max_features": str(mf),
             **{k: res[k] for k in ("accuracy", "f1_opt", "roc_auc", "pr_auc",
                                    "oob_accuracy", "fit_time")},
             "roc_auc_std": res["roc_auc_std"]}
        )
        print(f"  RF depth={depth} mf={str(mf):4s}  auc={res['roc_auc']:.4f}")
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    df_raw = pd.read_csv("data_cardiovascular_risk.csv")

    print("Grilla DT")
    dt_grid = grid_dt(df_raw)
    dt_grid.to_csv("results/grilla_dt.csv", index=False)
    best_dt = dt_grid.loc[dt_grid["roc_auc"].idxmax()]
    print("  mejor:", dict(best_dt[["max_depth", "min_samples_leaf", "roc_auc"]]))

    print("Grilla RF (100 árboles)")
    rf_grid = grid_rf(df_raw)
    rf_grid.to_csv("results/grilla_rf.csv", index=False)
    best_rf = rf_grid.loc[rf_grid["roc_auc"].idxmax()]
    print("  mejor:", dict(best_rf[["max_depth", "max_features", "roc_auc"]]))

    with open("results/resumen.json", encoding="utf-8") as f:
        resumen = json.load(f)
    resumen["final"]["config_dt"] = {
        "max_depth": int(best_dt["max_depth"]),
        "min_samples_leaf": int(best_dt["min_samples_leaf"]),
        "criterion": "gini",
    }
    resumen["final"]["config_rf"] = {
        "n_estimators": 300,
        "max_depth": int(best_rf["max_depth"]),
        "max_features": str(best_rf["max_features"]),
        "criterion": "gini",
    }
    resumen["final"]["grilla_dt_auc"] = float(best_dt["roc_auc"])
    resumen["final"]["grilla_rf_auc"] = float(best_rf["roc_auc"])
    with open("results/resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    print(f"Listo en {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
