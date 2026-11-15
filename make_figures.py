"""
make_figures.py
===============

Genera las figuras del informe a partir de los resultados ya calculados por
`run_experiments.py`. No reentrena nada: lee `results/resumen.json` y los CSV
de barridos, y escribe PNGs en `results/figuras/`.

    python make_figures.py
"""

from __future__ import annotations

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import experiments as E
from dt_rf import DecisionTree, RandomForest
from plots import plot_confusion_matrix, plot_curve, plot_importances, plot_roc_pr

OUT = "results/figuras"


def main():
    os.makedirs(OUT, exist_ok=True)
    with open("results/resumen.json", encoding="utf-8") as f:
        resumen = json.load(f)

    holdout = np.load("results/final_probas.npy", allow_pickle=True).item()
    y_te = holdout["y"]
    probas = holdout["probas"]

    # ---------------------------------------------------------------- ROC / PR
    fig, _ = plot_roc_pr(y_te, probas)
    fig.savefig(os.path.join(OUT, "roc_pr.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ------------------------------------------------------ matrices de confusión
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.6))
    for ax, (nombre, fila) in zip(axs, resumen["final"]["holdout"].items()):
        cm = np.asarray(fila["matriz_confusion"], dtype=int)
        plot_confusion_matrix(cm, labels=("0", "1"), ax=ax,
                              title=f"{nombre} (umbral 0,5)")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "matrices_confusion.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------------- importancias
    imp = pd.read_csv("results/importancias.csv")
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    plot_importances(imp["DT"].to_numpy(), list(imp["atributo"]), top_n=10,
                     ax=axs[0], title="Decision Tree")
    plot_importances(imp["RF"].to_numpy(), list(imp["atributo"]), top_n=10,
                     ax=axs[1], title="Random Forest")
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "importancias.png"), dpi=150)
    plt.close(fig)

    # ------------------------------------------------------- curvas de barridos
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    d = pd.read_csv("results/dt_barrido_profundidad.csv")
    plot_curve(d["max_depth"], d["roc_auc"], d["roc_auc_std"], "max_depth",
               "ROC-AUC", "DT: AUC vs profundidad", ax=axs[0])
    plot_curve(d["max_depth"], d["accuracy"], d["accuracy_std"], "max_depth",
               "Accuracy", "DT: accuracy vs profundidad", ax=axs[1])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "dt_profundidad.png"), dpi=150)
    plt.close(fig)

    fig, axs = plt.subplots(1, 2, figsize=(11, 3.8))
    r = pd.read_csv("results/rf_barrido_arboles.csv")
    plot_curve(r["n_estimators"], r["roc_auc"], r["roc_auc_std"], "n_estimators",
               "ROC-AUC", "RF: AUC vs cantidad de árboles", ax=axs[0])
    plot_curve(r["n_estimators"], r["oob_accuracy"], r["oob_accuracy_std"],
               "n_estimators", "OOB accuracy",
               "RF: error out-of-bag vs cantidad de árboles", ax=axs[1])
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "rf_arboles.png"), dpi=150)
    plt.close(fig)

    # --------------------------------------------- ablaciones de preprocesamiento
    ab = pd.read_csv("results/ablaciones_preprocesamiento.csv")
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    ax.barh(ab["configuracion"], ab["roc_auc"], color="#55A868")
    ax.set_xlim(ab["roc_auc"].min() - 0.005, ab["roc_auc"].max() + 0.005)
    ax.set_xlabel("ROC-AUC (CV 5-fold)")
    ax.set_title("Impacto de las decisiones de pre-procesamiento")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "ablaciones.png"), dpi=150)
    plt.close(fig)

    # -------------------------------------------------- árbol final (3 niveles)
    from plots import draw_tree
    from sklearn.model_selection import train_test_split

    df = pd.read_csv("data/dataset_procesado.csv")
    X = df.drop(columns=["TenYearCHD"])
    y = df["TenYearCHD"].to_numpy()
    X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.3, random_state=42,
                                        stratify=y)
    cfg = resumen["final"]["config_dt"]
    arbol = DecisionTree(**cfg).fit(X_tr, y_tr)
    fig, _ = draw_tree(arbol, list(X.columns), max_depth=3, figsize=(14, 7))
    fig.savefig(os.path.join(OUT, "arbol_dt.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)

    print("Figuras guardadas en", OUT)
    for nombre in sorted(os.listdir(OUT)):
        print("  ", nombre)


if __name__ == "__main__":
    main()
