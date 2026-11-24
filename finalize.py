"""
finalize.py
===========

Con la configuración elegida por `tune_final.py`, ajusta el preprocesamiento
dentro de cada fold/train split, entrena los modelos finales, calcula la
comparación 5-fold, el hold-out (matrices de confusión y curvas), el test de
Wilcoxon y las importancias, y actualiza `results/resumen.json`.

    python finalize.py
"""

from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split

import experiments as E
from dt_rf import DecisionTree, RandomForest
from preprocessing import Preprocessor, TARGET

CV = 5
SEED = 42
TARGET = "TenYearCHD"


def per_fold_accuracies(make_model, df_raw, n_splits=CV, seed=SEED):
    """Exactitudes por fold, para el test de Wilcoxon."""
    y = df_raw[TARGET].to_numpy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    accs = []
    for tr, te in skf.split(np.zeros(len(y)), y):
        prep = Preprocessor()
        clean_tr = prep.fit_transform(df_raw.iloc[tr].reset_index(drop=True))
        clean_te = prep.transform(df_raw.iloc[te].reset_index(drop=True))
        X_tr = clean_tr.drop(columns=[TARGET])
        y_tr = clean_tr[TARGET].to_numpy()
        X_te = clean_te.drop(columns=[TARGET])
        y_te = clean_te[TARGET].to_numpy()
        model = make_model()
        model.fit(X_tr, y_tr)
        pred = model.predict(X_te)
        accs.append(float((pred == y_te).mean()))
    return np.array(accs)


def main():
    t0 = time.time()
    with open("results/resumen.json", encoding="utf-8") as f:
        resumen = json.load(f)

    cfg_dt = resumen["final"]["config_dt"]
    cfg_dt = {**cfg_dt, "random_state": SEED}
    cfg_rf = resumen["final"]["config_rf"]
    best_mf = cfg_rf["max_features"]
    if isinstance(best_mf, str) and best_mf.replace(".", "", 1).isdigit():
        best_mf = float(best_mf)
    cfg_rf = {**cfg_rf, "max_features": best_mf, "n_jobs": 1, "random_state": SEED}

    df_raw = pd.read_csv("data_cardiovascular_risk.csv")
    y = df_raw[TARGET].to_numpy()
    raw_idx = np.arange(len(df_raw))
    idx_tr, idx_te = train_test_split(
        raw_idx, test_size=0.30, random_state=SEED, stratify=y
    )
    prep_holdout = Preprocessor()
    clean_tr = prep_holdout.fit_transform(
        df_raw.iloc[idx_tr].reset_index(drop=True)
    )
    clean_te = prep_holdout.transform(
        df_raw.iloc[idx_te].reset_index(drop=True)
    )
    X_tr = clean_tr.drop(columns=[TARGET])
    y_tr = clean_tr[TARGET].to_numpy()
    X_te = clean_te.drop(columns=[TARGET])
    y_te = clean_te[TARGET].to_numpy()

    factories = {
        "DT": lambda: DecisionTree(**cfg_dt),
        "RF": lambda: RandomForest(**cfg_rf),
    }

    print("Configuración final")
    print("  DT:", cfg_dt)
    print("  RF:", cfg_rf)

    # ------------------------------------------------------ validación cruzada
    cv = {}
    for name, make in factories.items():
        res = E.cross_validate_with_preprocessing(
            df_raw, TARGET, lambda: Preprocessor(), make, n_splits=CV
        )
        cv[name] = res
        print(f"  CV {name}: acc={res['accuracy']:.4f}±{res['accuracy_std']:.4f} "
              f"auc={res['roc_auc']:.4f} f1={res['f1']:.4f}")

    acc_dt = per_fold_accuracies(factories["DT"], df_raw)
    acc_rf = per_fold_accuracies(factories["RF"], df_raw)
    try:
        from scipy.stats import wilcoxon

        stat, p_value = wilcoxon(acc_dt, acc_rf)
        stat, p_value = float(stat), float(p_value)
    except Exception:
        stat, p_value = float("nan"), float("nan")
    print(f"  Wilcoxon: p={p_value:.4f}")

    # ----------------------------------------------------------------- hold-out
    holdout, fitted = {}, {}
    for name, make in factories.items():
        model = make()
        t = time.perf_counter()
        model.fit(X_tr, y_tr)
        fit_time = time.perf_counter() - t
        proba = model.predict_proba(X_te)[:, 1]
        pred = model.predict(X_te)
        metrics = E.evaluate_with_train_threshold(
            y_tr,
            model.predict_proba(X_tr)[:, 1],
            y_te,
            proba,
            pred_test=pred,
        )
        pred_opt = (proba >= metrics["threshold_opt"]).astype(int)
        holdout[name] = {
            **metrics,
            "tiempo (s)": round(fit_time, 3),
            "matriz_confusion": E.confusion(y_te, pred).tolist(),
            "matriz_confusion_opt": E.confusion(y_te, pred_opt).tolist(),
        }
        fitted[name] = (model, proba, pred)
        print(f"  Hold-out {name}: acc={holdout[name]['accuracy']:.4f} "
              f"auc={holdout[name]['roc_auc']:.4f}")

    pd.DataFrame(cv).T.reset_index().rename(columns={"index": "modelo"}).to_csv(
        "results/comparacion_final_cv.csv", index=False
    )
    pd.DataFrame(holdout).T.reset_index().rename(columns={"index": "modelo"}).to_csv(
        "results/comparacion_final_holdout.csv", index=False
    )
    np.save(
        "results/final_probas.npy",
        {"y": y_te, "probas": {k: v[1] for k, v in fitted.items()}},
        allow_pickle=True,
    )

    imp = pd.DataFrame({
        "atributo": list(X_tr.columns),
        "DT": fitted["DT"][0].feature_importances_.round(4).tolist(),
        "RF": fitted["RF"][0].feature_importances_.round(4).tolist(),
    }).sort_values("RF", ascending=False)
    imp.to_csv("results/importancias.csv", index=False)

    resumen["final"]["cv"] = cv
    resumen["final"]["holdout"] = holdout
    resumen["final"]["wilcoxon_p"] = p_value
    resumen["final"]["wilcoxon_stat"] = stat
    resumen["final"]["baseline_mayoritaria"] = round(float((y == 0).mean()), 4)
    with open("results/resumen.json", "w", encoding="utf-8") as f:
        json.dump(resumen, f, indent=2, ensure_ascii=False)

    print(f"Listo en {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
