"""
run_experiments.py
==================

Reproduce todos los experimentos del informe y guarda los resultados en
`results/`. Es el equivalente "de línea de comandos" de lo que hace el
notebook, útil para regenerar las tablas del informe sin abrir Colab.

    python run_experiments.py

Salidas
-------
- data/dataset_procesado.csv
- data/auditoria_inicial.md
- data/auditoria_preprocesamiento.md
- results/*.csv  (una tabla por experimento)
- results/resumen.json  (métricas puntuales usadas en el informe)
- results/protocolo.json  (semillas y reglas de evaluación)
"""

from __future__ import annotations

import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

import experiments as E
from dt_rf import DecisionTree, RandomForest
from preprocessing import Preprocessor, audit, audit_to_markdown, load_raw, TARGET

RAW_PATH = "data_cardiovascular_risk.csv"
PROCESSED_PATH = "data/dataset_procesado.csv"
RESULTS_DIR = "results"
CV = 5
SEED = 42


def _save(df: pd.DataFrame, name: str):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, name)
    df.to_csv(path, index=False)
    print(f"  -> {path}")
    return df


def main():
    t_start = time.time()
    os.makedirs("data", exist_ok=True)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    summary: dict = {}
    summary["protocol"] = {
        "seed": SEED,
        "cv_folds": CV,
        "holdout_test_size": 0.30,
        "preprocessing": "fit only on each training fold/train split",
        "threshold": "learned on train and applied to validation/test",
        "pr_auc": "average_precision",
    }

    # ------------------------------------------------------------------
    # 1. Auditoría y pre-procesamiento
    # ------------------------------------------------------------------
    print("\n[1/7] Auditoría y pre-procesamiento")
    df_raw = load_raw(RAW_PATH)
    info = audit(df_raw)
    with open("data/auditoria_inicial.md", "w", encoding="utf-8") as f:
        f.write(audit_to_markdown(info))

    prep = Preprocessor()
    clean = prep.fit_transform(df_raw)
    clean.to_csv(PROCESSED_PATH, index=False)
    with open("data/auditoria_preprocesamiento.md", "w", encoding="utf-8") as f:
        f.write(prep.report_markdown())

    summary["dataset"] = {
        "filas": int(len(df_raw)),
        "columnas": int(df_raw.shape[1]),
        "nulos_celdas": int(info["missing_total_cells"]),
        "filas_con_nulos": int(info["missing_rows_any"]),
        "duplicados": int(info["duplicates"]["exactos"]),
        "positivos": int((df_raw[TARGET] == 1).sum()),
        "pct_positivos": round(float((df_raw[TARGET] == 1).mean() * 100), 2),
        "imputados": int(sum(prep.report_["imputados"].values())),
        "winsorizados": int(sum(prep.report_["winsorizados"].values())),
        "columnas_salida": int(clean.shape[1]),
    }
    print("  ", summary["dataset"])

    # El CSV procesado completo se entrega como artefacto, pero no se usa
    # para medir rendimiento: los parámetros de limpieza deben aprenderse
    # sólo con el train de cada evaluación.
    y = df_raw[TARGET].to_numpy()

    # ------------------------------------------------------------------
    # 2. Línea base: hold-out y comparación con scikit-learn
    # ------------------------------------------------------------------
    print("\n[2/7] Comparación contra scikit-learn")
    raw_idx = np.arange(len(df_raw))
    idx_tr, idx_te = train_test_split(
        raw_idx, test_size=0.3, random_state=SEED, stratify=y
    )
    prep_holdout = Preprocessor()
    clean_tr = prep_holdout.fit_transform(
        df_raw.iloc[idx_tr].reset_index(drop=True)
    )
    clean_te = prep_holdout.transform(
        df_raw.iloc[idx_te].reset_index(drop=True)
    )
    X_tr, y_tr = clean_tr.drop(columns=[TARGET]), clean_tr[TARGET].to_numpy()
    X_te, y_te = clean_te.drop(columns=[TARGET]), clean_te[TARGET].to_numpy()
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.tree import DecisionTreeClassifier

    base = {}
    payload = {}
    for name, model in [
        ("DT propio (entropía, gain ratio)", DecisionTree(max_depth=6)),
        ("DT propio (entropía, gain)", DecisionTree(max_depth=6, split_metric="gain")),
        ("DT propio (gini)", DecisionTree(max_depth=6, criterion="gini")),
        ("DT sklearn (entropy)", DecisionTreeClassifier(max_depth=6, criterion="entropy", random_state=0)),
        ("DT sklearn (gini)", DecisionTreeClassifier(max_depth=6, criterion="gini", random_state=0)),
        ("RF propio", RandomForest(n_estimators=100, max_depth=10, n_jobs=-1, random_state=0)),
        ("RF sklearn", RandomForestClassifier(n_estimators=100, max_depth=10, n_jobs=-1, random_state=0)),
    ]:
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        fit_time = time.perf_counter() - t0
        proba = model.predict_proba(X_te)[:, 1]
        res = E.evaluate_with_train_threshold(
            y_tr,
            model.predict_proba(X_tr)[:, 1],
            y_te,
            proba,
            pred_test=model.predict(X_te),
        )
        res["tiempo (s)"] = round(fit_time, 3)
        base[name] = res
        payload[name] = proba
        print(f"  {name:36s} acc={res['accuracy']:.4f} auc={res['roc_auc']:.4f} "
              f"f1={res['f1']:.4f} t={fit_time:.2f}s")
    summary["sklearn_check"] = base
    base_table = pd.DataFrame([
        {"modelo": name, **{metric: result[metric] for metric in E.METRICS},
         "tiempo (s)": result["tiempo (s)"]}
        for name, result in base.items()
    ])
    _save(base_table, "comparacion_sklearn.csv")
    np.save(os.path.join(RESULTS_DIR, "holdout_probas.npy"),
            {"y": y_te, "probas": payload}, allow_pickle=True)

    # ------------------------------------------------------------------
    # 3. Barridos de Decision Tree
    # ------------------------------------------------------------------
    print("\n[3/7] Barridos de Decision Tree")
    prep_factory = lambda: Preprocessor()
    dt_fixed = {"criterion": "gini", "random_state": SEED}

    depth_df = E.sweep_with_preprocessing(
        E.make_dt, "max_depth",
        [1, 2, 3, 4, 5, 6, 8, 10, 12, 16, 20],
        df_raw, TARGET, prep_factory, fixed=dt_fixed,
    )
    _save(depth_df, "dt_barrido_profundidad.csv")

    leaf_df = E.sweep_with_preprocessing(
        E.make_dt, "min_samples_leaf", [1, 2, 5, 10, 20, 40, 80],
        df_raw, TARGET, prep_factory, fixed={**dt_fixed, "max_depth": 8},
    )
    _save(leaf_df, "dt_barrido_min_hoja.csv")

    crit_rows = []
    for criterion, split_metric, label in [
        ("entropy", "gain_ratio", "entropia + gain ratio"),
        ("entropy", "gain", "entropia + gain"),
        ("gini", None, "gini"),
    ]:
        res = E.cross_validate_with_preprocessing(
            df_raw,
            TARGET,
            prep_factory,
            lambda c=criterion, sm=split_metric: DecisionTree(
                max_depth=6, criterion=c, split_metric=sm, random_state=SEED
            ),
        )
        res["criterion"] = criterion
        res["split_metric"] = split_metric
        crit_rows.append(res)
        print(f"  criterio {label:22s} acc={res['accuracy']:.4f} auc={res['roc_auc']:.4f}")
    _save(pd.DataFrame(crit_rows), "dt_barrido_criterio.csv")

    prune_rows = []
    for label, prune in [("sin poda", False), ("pesimista (C4.5)", "pessimistic"),
                         ("REP", "rep")]:
        res = E.cross_validate_with_preprocessing(
            df_raw,
            TARGET,
            prep_factory,
            lambda: DecisionTree(max_depth=12, criterion="gini", random_state=SEED),
            prune=prune,
            verbose=False,
        )
        res["poda"] = label
        prune_rows.append(res)
        print(f"  poda {label:20s} acc={res['accuracy']:.4f} auc={res['roc_auc']:.4f} "
              f"hojas={res['leaves']:.1f}")
    _save(pd.DataFrame(prune_rows), "dt_barrido_poda.csv")

    weight_rows = []
    for label, cw in [("sin balanceo", None), ("balanced", "balanced")]:
        res = E.cross_validate_with_preprocessing(
            df_raw,
            TARGET,
            prep_factory,
            lambda: DecisionTree(
                max_depth=6, criterion="gini", class_weight=cw,
                random_state=SEED
            ),
        )
        res["class_weight"] = label
        weight_rows.append(res)
        print(f"  class_weight {label:14s} acc={res['accuracy']:.4f} "
              f"recall={res['recall']:.4f} auc={res['roc_auc']:.4f}")
    _save(pd.DataFrame(weight_rows), "dt_class_weight.csv")

    # ------------------------------------------------------------------
    # 4. Barridos de Random Forest
    # ------------------------------------------------------------------
    print("\n[4/7] Barridos de Random Forest")
    trees_df = E.sweep_with_preprocessing(
        E.make_rf, "n_estimators", [10, 25, 50, 100, 200, 300],
        df_raw, TARGET, prep_factory,
        fixed={"max_depth": 12, "criterion": "gini",
               "n_jobs": -1, "random_state": SEED},
    )
    _save(trees_df, "rf_barrido_arboles.csv")

    mf_df = E.sweep_with_preprocessing(
        E.make_rf, "max_features", ["sqrt", "log2", 0.3, 0.6],
        df_raw, TARGET, prep_factory,
        fixed={"n_estimators": 100, "max_depth": 12, "criterion": "gini",
               "n_jobs": -1, "random_state": SEED},
    )
    _save(mf_df, "rf_barrido_max_features.csv")

    rfdepth_df = E.sweep_with_preprocessing(
        E.make_rf, "max_depth", [3, 5, 8, 12, 20, None],
        df_raw, TARGET, prep_factory,
        fixed={"n_estimators": 100, "max_features": "sqrt",
               "criterion": "gini", "n_jobs": -1, "random_state": SEED},
    )
    _save(rfdepth_df, "rf_barrido_profundidad.csv")

    voting_rows = []
    for voting in ["soft", "hard"]:
        res = E.cross_validate_with_preprocessing(
            df_raw,
            TARGET,
            prep_factory,
            lambda: RandomForest(n_estimators=100, max_depth=12, criterion="gini",
                                 voting=voting, n_jobs=-1, random_state=SEED),
        )
        res["voting"] = voting
        voting_rows.append(res)
        print(f"  voting {voting:5s} acc={res['accuracy']:.4f} auc={res['roc_auc']:.4f}")
    _save(pd.DataFrame(voting_rows), "rf_voting.csv")

    # ------------------------------------------------------------------
    # 5. Ablaciones de pre-procesamiento
    # ------------------------------------------------------------------
    print("\n[5/7] Ablaciones de pre-procesamiento")
    ablations = [
        ("MICE + winsor + is_smoking (base)", dict(imputer="mice", clip_outliers=True)),
        ("mediana + winsor", dict(imputer="median", clip_outliers=True)),
        ("KNN + winsor", dict(imputer="knn", clip_outliers=True)),
        ("MICE sin winsor", dict(imputer="mice", clip_outliers=False)),
        ("MICE + sin is_smoking", dict(imputer="mice", clip_outliers=True,
                                       drop_is_smoking=True)),
        ("MICE sin pulsePressure", dict(imputer="mice", clip_outliers=True,
                                        add_pulse_pressure=False)),
    ]

    def rf_default():
        return RandomForest(
            n_estimators=80, max_depth=10, criterion="gini",
            n_jobs=-1, random_state=SEED
        )

    ablation_rows = []
    for label, params in ablations:
        res = E.cross_validate_with_preprocessing(
            df_raw, TARGET, lambda p=params: Preprocessor(**p), rf_default
        )
        res["configuracion"] = label
        ablation_rows.append(res)
        print(f"  {label:38s} acc={res['accuracy']:.4f} auc={res['roc_auc']:.4f}")
    _save(pd.DataFrame(ablation_rows), "ablaciones_preprocesamiento.csv")

    # ------------------------------------------------------------------
    # 6. Comparación final DT vs RF
    # ------------------------------------------------------------------
    print("\n[6/7] Selección conjunta de hiperparámetros y evaluación final")

    # Se guarda un resumen parcial para que tune_final.py / finalize.py lo lean
    # y lo vayan completando (así la lógica de selección final vive en un solo
    # lugar y no se duplica acá).
    summary["final"] = {"baseline_mayoritaria": round(float((y == 0).mean()), 4)}
    with open(os.path.join(RESULTS_DIR, "resumen.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with open(os.path.join(RESULTS_DIR, "protocolo.json"), "w", encoding="utf-8") as f:
        json.dump(summary["protocol"], f, indent=2, ensure_ascii=False)

    import finalize
    import tune_final

    tune_final.main()
    finalize.main()

    with open(os.path.join(RESULTS_DIR, "resumen.json"), encoding="utf-8") as f:
        summary = json.load(f)

    # ------------------------------------------------------------------
    # 7. Resumen
    # ------------------------------------------------------------------
    print("\n[7/7] Resumen")
    with open(os.path.join(RESULTS_DIR, "resumen.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"  Todo guardado en {RESULTS_DIR}/")
    print(f"  Tiempo total: {time.time() - t_start:.1f} s")
    return summary


if __name__ == "__main__":
    main()
