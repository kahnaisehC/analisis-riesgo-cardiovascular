"""
experiments.py
==============

Orquestación de los experimentos del TP: métricas, validación cruzada
estratificada y barridos de hiperparámetros. Se usa desde el notebook
(`TP_IAD_2026.ipynb`) y desde `run_experiments.py` para reproducir los
resultados del informe.

No contiene lógica de los algoritmos (eso vive en `dt_rf.py`) ni de limpieza
(`preprocessing.py`): sólo evaluación y comparación.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    average_precision_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

from dt_rf import DecisionTree, RandomForest

METRICS = [
    "accuracy",
    "balanced_accuracy",
    "precision",
    "recall",
    "f1",
    "f1_opt",
    "roc_auc",
    "pr_auc",
    "threshold_opt",
]


def optimal_threshold(y_true, proba) -> tuple[float, float]:
    """Umbral que maximiza F1 y el F1 resultante.

    Con clases desbalanceadas el umbral 0.5 suele predecir casi todo negativo;
    este análisis muestra cuánto se gana moviendo el punto de corte.
    """
    y_true = np.asarray(y_true).ravel()
    proba = np.asarray(proba).ravel()
    prec, rec, thresholds = precision_recall_curve(y_true, proba)
    denom = prec + rec
    f1 = np.divide(2 * prec * rec, denom, out=np.zeros_like(denom), where=denom > 0)
    best = int(np.nanargmax(f1))
    if len(thresholds) == 0:
        return 0.5, float(f1[best])
    thr = float(thresholds[min(best, len(thresholds) - 1)])
    return thr, float(f1[best])


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------
def evaluate_predictions(y_true, proba, pred=None, tune_threshold=False) -> dict:
    """Calcula el set de métricas usado en todo el TP.

    Parameters
    ----------
    y_true : array-like
    proba : array-like
        Probabilidad de la clase positiva.
    pred : array-like, optional
        Etiquetas predichas. Si es ``None`` se usa ``proba >= 0.5``.
    tune_threshold : bool, default=False
        Si es ``True``, calcula el mejor umbral sobre este mismo conjunto.
        Es una medida exploratoria optimista; para evaluar un test se debe
        usar ``evaluate_with_train_threshold``.
    """
    y_true = np.asarray(y_true).ravel()
    proba = np.asarray(proba).ravel()
    pred = (proba >= 0.5).astype(int) if pred is None else np.asarray(pred).ravel()
    out = {
        "accuracy": accuracy_score(y_true, pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }
    if tune_threshold:
        try:
            out["threshold_opt"], out["f1_opt"] = optimal_threshold(y_true, proba)
        except ValueError:
            out["threshold_opt"], out["f1_opt"] = np.nan, np.nan
    # AUC indefinido si el fold tiene una sola clase
    try:
        out["roc_auc"] = roc_auc_score(y_true, proba)
    except ValueError:
        out["roc_auc"] = np.nan
    try:
        out["pr_auc"] = average_precision_score(y_true, proba)
    except ValueError:
        out["pr_auc"] = np.nan
    return out


def evaluate_with_train_threshold(
    y_train,
    proba_train,
    y_test,
    proba_test,
    pred_test=None,
) -> dict:
    """Evalúa un conjunto de test con un umbral aprendido sólo en train.

    ``evaluate_predictions`` conserva la opción histórica de calcular el
    umbral óptimo sobre el mismo conjunto evaluado, útil para exploración pero
    optimista. Esta función es la que deben usar los resultados finales:
    aprende el umbral en train y lo aplica una sola vez en test.
    """
    threshold, _ = optimal_threshold(y_train, proba_train)
    y_test = np.asarray(y_test).ravel()
    proba_test = np.asarray(proba_test).ravel()
    pred_test = (
        (proba_test >= 0.5).astype(int)
        if pred_test is None
        else np.asarray(pred_test).ravel()
    )
    out = evaluate_predictions(
        y_test, proba_test, pred=pred_test, tune_threshold=False
    )
    pred_opt = (proba_test >= threshold).astype(int)
    out["threshold_opt"] = float(threshold)
    out["f1_opt"] = float(f1_score(y_test, pred_opt, zero_division=0))
    return out


def _take(X, idx):
    """Indexa X preservando el tipo (DataFrame o ndarray)."""
    if hasattr(X, "iloc"):
        return X.iloc[idx].reset_index(drop=True)
    return X[idx]


def _model_stats(model) -> dict:
    """Estadísticas estructurales del modelo (hojas, profundidad, OOB)."""
    stats = {}
    if isinstance(model, DecisionTree):
        stats["leaves"] = model.get_n_leaves()
        stats["depth"] = model.get_depth()
    elif isinstance(model, RandomForest):
        stats["leaves"] = float(np.mean([t.get_n_leaves() for t in model.estimators_]))
        stats["depth"] = float(np.mean([t.get_depth() for t in model.estimators_]))
        stats["oob_accuracy"] = getattr(model, "oob_score_", np.nan)
    return stats


# ---------------------------------------------------------------------------
# Validación cruzada
# ---------------------------------------------------------------------------
def cross_validate(
    make_model,
    X,
    y,
    n_splits: int = 5,
    random_state: int = 42,
    prune: bool = False,
    val_size: float = 0.2,
    verbose: bool = False,
) -> dict:
    """Validación cruzada estratificada con métricas de clasificación.

    Parameters
    ----------
    make_model : callable
        Devuelve un modelo sin entrenar en cada fold.
    X, y : array-like
    n_splits : int, default=5
    prune : bool or {'rep', 'pessimistic'}, default=False
        Poda a aplicar al árbol antes de evaluar. ``True`` equivale a
        ``'rep'``, que reserva un 20 % del train de cada fold como validación.
        ``'pessimistic'`` no necesita datos extra (poda de C4.5).
    verbose : bool

    Returns
    -------
    dict
        ``{metrica: media, metrica + '_std': desvio, ...}`` más ``fit_time``,
        ``leaves``, ``depth`` y ``oob_accuracy`` cuando aplica.
    """
    y = np.asarray(y).ravel()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    prune_method = "rep" if prune is True else prune
    rows = []
    for k, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        X_tr, y_tr = _take(X, tr), y[tr]
        X_te, y_te = _take(X, te), y[te]

        if prune_method == "rep":
            X_tr, X_val, y_tr, y_val = train_test_split(
                X_tr, y_tr, test_size=val_size, stratify=y_tr, random_state=random_state + k
            )

        model = make_model()
        # Reproducibilidad: si el modelo es estocástico (RF) y no se fijó semilla,
        # se usa una semilla distinta por fold, derivada de ``random_state``.
        if getattr(model, "random_state", 0) is None:
            model.set_params(random_state=random_state + k)
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        fit_time = time.perf_counter() - t0
        if prune_method == "rep":
            model.prune(X_val, y_val, method="rep")
        elif prune_method == "pessimistic":
            model.prune(method="pessimistic")

        proba = model.predict_proba(X_te)[:, 1]
        proba_train = model.predict_proba(X_tr)[:, 1]
        pred = model.predict(X_te)
        row = evaluate_with_train_threshold(
            y_tr, proba_train, y_te, proba, pred_test=pred
        )
        row["fit_time"] = fit_time
        row.update(_model_stats(model))
        rows.append(row)
        if verbose:
            print(f"  fold {k}/{n_splits}  acc={row['accuracy']:.4f}  "
                  f"f1={row['f1']:.4f}  auc={row['roc_auc']:.4f}")

    df = pd.DataFrame(rows)
    out = {}
    for col in df.columns:
        out[col] = float(df[col].mean())
        out[f"{col}_std"] = float(df[col].std(ddof=1))
    return out


# ---------------------------------------------------------------------------
# Barridos de hiperparámetros
# ---------------------------------------------------------------------------
def sweep(
    make_model_factory,
    param_name: str,
    values,
    X,
    y,
    n_splits: int = 5,
    prune: bool = False,
    fixed: dict | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Entrena y evalúa un modelo para cada valor de un hiperparámetro.

    Parameters
    ----------
    make_model_factory : callable
        Recibe un dict de parámetros y devuelve un modelo sin entrenar.
    param_name : str
    values : iterable
    fixed : dict, optional
        Parámetros fijos para todas las corridas.
    """
    fixed = fixed or {}
    rows = []
    for value in values:
        params = {**fixed, param_name: value}
        if verbose:
            print(f"— {param_name} = {value}")
        res = cross_validate(
            lambda p=params: make_model_factory(p),
            X, y, n_splits=n_splits, prune=prune, verbose=verbose,
        )
        res[param_name] = value
        rows.append(res)
    cols = [param_name] + [c for c in rows[0] if c != param_name]
    return pd.DataFrame(rows)[cols]


def make_dt(params: dict) -> DecisionTree:
    """Factory de `DecisionTree` a partir de un dict de hiperparámetros."""
    return DecisionTree(**params)


def make_rf(params: dict) -> RandomForest:
    """Factory de `RandomForest` a partir de un dict de hiperparámetros."""
    return RandomForest(**params)


# ---------------------------------------------------------------------------
# Ablaciones de pre-procesamiento
# ---------------------------------------------------------------------------
def cross_validate_with_preprocessing(
    df_raw: pd.DataFrame,
    target: str,
    prep_factory,
    model_factory,
    n_splits: int = 5,
    random_state: int = 42,
    prune: bool = False,
    val_size: float = 0.2,
    verbose: bool = False,
) -> dict:
    """CV donde el pre-procesamiento se ajusta **dentro** de cada fold.

    Evita fuga de información: el imputador y los percentiles de winsorización
    se aprenden sólo con el train del fold.

    Parameters
    ----------
    df_raw : DataFrame
        Dataset original (con nulos y outliers).
    prep_factory : callable
        Devuelve un `Preprocessor` nuevo (los parámetros de la ablación).
    model_factory : callable
        Devuelve un modelo nuevo.
    """
    y = df_raw[target].to_numpy()
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    rows = []
    prune_method = "rep" if prune is True else prune
    for k, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y), start=1):
        raw_tr = df_raw.iloc[tr].reset_index(drop=True)
        raw_te = df_raw.iloc[te].reset_index(drop=True)

        # Para REP, separar también la validación antes de ajustar el
        # preprocesador. Así ningún parámetro de imputación/clip ve esa
        # validación ni el test externo.
        if prune_method == "rep":
            inner_tr, inner_val = train_test_split(
                np.arange(len(raw_tr)),
                test_size=val_size,
                stratify=raw_tr[target].to_numpy(),
                random_state=random_state + k,
            )
            raw_fit = raw_tr.iloc[inner_tr].reset_index(drop=True)
            raw_val = raw_tr.iloc[inner_val].reset_index(drop=True)
        else:
            raw_fit = raw_tr
            raw_val = None

        prep = prep_factory()
        clean_tr = prep.fit_transform(raw_fit)
        clean_te = prep.transform(raw_te)
        X_tr = clean_tr.drop(columns=[target])
        y_tr = clean_tr[target].to_numpy()
        X_te = clean_te.drop(columns=[target])
        y_te = clean_te[target].to_numpy()
        if raw_val is not None:
            clean_val = prep.transform(raw_val)
            X_val = clean_val.drop(columns=[target])
            y_val = clean_val[target].to_numpy()

        model = model_factory()
        if getattr(model, "random_state", 0) is None:
            model.set_params(random_state=random_state + k)
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        fit_time = time.perf_counter() - t0
        if prune_method == "rep":
            model.prune(X_val, y_val, method="rep")
        elif prune_method == "pessimistic":
            model.prune(method="pessimistic")
        proba = model.predict_proba(X_te)[:, 1]
        proba_train = model.predict_proba(X_tr)[:, 1]
        row = evaluate_with_train_threshold(
            y_tr, proba_train, y_te, proba, pred_test=model.predict(X_te)
        )
        row["fit_time"] = fit_time
        row.update(_model_stats(model))
        rows.append(row)
        if verbose:
            print(
                f"  fold {k}/{n_splits}  acc={row['accuracy']:.4f}  "
                f"f1={row['f1']:.4f}  auc={row['roc_auc']:.4f}"
            )
    df = pd.DataFrame(rows)
    out = {}
    for col in df.columns:
        out[col] = float(df[col].mean())
        out[f"{col}_std"] = float(df[col].std(ddof=1))
    return out


def sweep_with_preprocessing(
    make_model_factory,
    param_name: str,
    values,
    df_raw: pd.DataFrame,
    target: str,
    prep_factory,
    n_splits: int = 5,
    prune: bool = False,
    fixed: dict | None = None,
    verbose: bool = True,
) -> pd.DataFrame:
    """Barrido con preprocesamiento ajustado dentro de cada fold."""
    fixed = fixed or {}
    rows = []
    for value in values:
        params = {**fixed, param_name: value}
        if verbose:
            print(f"— {param_name} = {value}")
        res = cross_validate_with_preprocessing(
            df_raw,
            target,
            prep_factory,
            lambda p=params: make_model_factory(p),
            n_splits=n_splits,
            prune=prune,
            verbose=verbose,
        )
        res[param_name] = value
        rows.append(res)
    cols = [param_name] + [c for c in rows[0] if c != param_name]
    return pd.DataFrame(rows)[cols]


# ---------------------------------------------------------------------------
# Presentación
# ---------------------------------------------------------------------------
def summary_table(results: dict, label: str = "modelo") -> pd.DataFrame:
    """Tabla de media ± desvío lista para imprimir."""
    rows = []
    for name, res in results.items():
        row = {"modelo": name}
        for m in METRICS:
            row[m] = f"{res[m]:.4f} ± {res[m + '_std']:.4f}"
        row["tiempo (s)"] = f"{res['fit_time']:.3f}"
        rows.append(row)
    return pd.DataFrame(rows)


def confusion(y_true, y_pred, labels=(0, 1)) -> np.ndarray:
    """Matriz de confusión con orden de etiquetas fijo."""
    return confusion_matrix(y_true, y_pred, labels=list(labels))
