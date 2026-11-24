"""
plots.py
========

Helpers de visualización para el TP. Se mantienen separados de los algoritmos
para que `dt_rf.py` no dependa de matplotlib (así el módulo es reutilizable y
los tests corren más rápido).

Funciones
---------
- `draw_tree`: dibuja el árbol de decisión (sin graphviz).
- `plot_confusion_matrix`: matriz de confusión anotada.
- `plot_roc_pr`: curvas ROC y Precision-Recall para varios modelos.
- `plot_curve`: curva métrica vs hiperparámetro, con banda de desvío.
- `plot_importances`: importancia de atributos.
- `plot_target_balance`: distribución de la clase objetivo.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

try:  # sklearn sólo se usa para las curvas, no para los modelos
    from sklearn.metrics import (
        roc_curve,
        precision_recall_curve,
        auc,
        average_precision_score,
    )
except Exception:  # pragma: no cover
    roc_curve = None
    average_precision_score = None


# ---------------------------------------------------------------------------
# Árbol
# ---------------------------------------------------------------------------
def _layout(node, x_counter, positions, edges, parent=None, label=""):
    """Asigna coordenadas (x, y) a cada nodo y registra las aristas."""
    if node.is_leaf:
        x = x_counter[0]
        x_counter[0] += 1
    else:
        child_xs = []
        for i, child in enumerate(node.children):
            if node.is_numeric:
                edge_label = "≤" if i == 0 else ">"
            else:
                edge_label = f"= {node.categories[i]}"
            child_xs.append(
                _layout(child, x_counter, positions, edges, node, edge_label)
            )
        x = float(np.mean(child_xs))
    positions[id(node)] = (x, -node.depth)
    if parent is not None:
        edges.append((id(parent), id(node), label))
    return x


def draw_tree(
    tree,
    feature_names=None,
    class_names=None,
    max_depth=None,
    figsize=None,
    fontsize=8,
    ax=None,
):
    """Dibuja un `DecisionTree` entrenado.

    Parameters
    ----------
    tree : DecisionTree
        Modelo ya ajustado (usa ``tree.tree_``).
    feature_names : list[str], optional
    class_names : list[str], optional
    max_depth : int, optional
        Si se indica, sólo dibuja hasta esa profundidad (útil para árboles
        grandes).
    """
    names = (
        list(feature_names)
        if feature_names is not None
        else [f"x{i}" for i in range(tree.n_features_in_)]
    )
    classes = (
        list(class_names)
        if class_names is not None
        else [str(c) for c in tree.classes_]
    )

    positions, edges = {}, []
    _layout(tree.tree_, [0.0], positions, edges)
    xs = [p[0] for p in positions.values()]
    ys = [p[1] for p in positions.values()]
    width = max(xs) + 1
    height = max(-min(ys) + 1, 2)

    if ax is None:
        figsize = figsize or (min(2.2 * width, 60), min(1.6 * height, 40))
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    # Colores por clase mayoritaria de la hoja
    cmap = plt.get_cmap("tab10")
    for node in tree._nodes():
        if max_depth is not None and node.depth > max_depth:
            continue
        x, y = positions[id(node)]
        if node.is_leaf:
            cls = int(np.argmax(node.value))
            color = cmap(cls % 10)
            p = node.value.max()
            text = f"{classes[cls]}\nn={node.n_samples}\np={p:.2f}"
            alpha = 0.85
        else:
            color = "white"
            if node.is_numeric:
                text = f"{names[node.feature]}\n≤ {node.threshold:.3g}"
            else:
                text = f"{names[node.feature]}\n(categórico)"
            alpha = 1.0
        ax.text(
            x,
            y,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            bbox=dict(
                boxstyle="round,pad=0.35",
                facecolor=color,
                alpha=alpha,
                edgecolor="black",
                linewidth=0.8,
            ),
            zorder=3,
        )

    for parent_id, child_id, label in edges:
        px, py = positions[parent_id]
        cx, cy = positions[child_id]
        if max_depth is not None:
            if -py > max_depth or -cy > max_depth:
                continue
        arrow = FancyArrowPatch(
            (px, py - 0.18),
            (cx, cy + 0.18),
            arrowstyle="-|>",
            mutation_scale=10,
            color="gray",
            linewidth=0.8,
            zorder=2,
        )
        ax.add_patch(arrow)
        if label:
            ax.text(
                (px + cx) / 2 + (0.06 if cx >= px else -0.06),
                (py + cy) / 2,
                label,
                fontsize=max(fontsize - 1, 5),
                color="dimgray",
                ha="center",
                va="center",
                zorder=4,
            )

    ax.set_xlim(-1, width)
    ax.set_ylim(min(ys) - 1, 1)
    ax.axis("off")
    ax.set_title(
        f"Árbol de decisión — hojas: {tree.get_n_leaves()}, "
        f"profundidad: {tree.get_depth()}"
    )
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Evaluación
# ---------------------------------------------------------------------------
def plot_confusion_matrix(cm, labels=("0", "1"), ax=None, title="Matriz de confusión"):
    """Dibuja una matriz de confusión ya calculada."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.2, 3.6))
    else:
        fig = ax.figure
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels)
    ax.set_yticks(range(len(labels)), labels)
    ax.set_xlabel("Predicho")
    ax.set_ylabel("Real")
    ax.set_title(title)
    vmax = cm.max() if cm.max() > 0 else 1
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                f"{cm[i, j]:,}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > vmax / 2 else "black",
                fontsize=11,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    return fig, ax


def plot_roc_pr(y_true, models, axs=None, title_suffix=""):
    """Curvas ROC y Precision-Recall.

    Parameters
    ----------
    y_true : array-like
    models : dict[str, array-like]
        Nombre del modelo -> probabilidad de la clase positiva.
    """
    if roc_curve is None:
        raise ImportError("scikit-learn es necesario para las curvas")
    if axs is None:
        fig, axs = plt.subplots(1, 2, figsize=(11, 4.2))
    else:
        fig = axs[0].figure

    for name, proba in models.items():
        fpr, tpr, _ = roc_curve(y_true, proba)
        axs[0].plot(fpr, tpr, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
        prec, rec, _ = precision_recall_curve(y_true, proba)
        ap = average_precision_score(y_true, proba)
        axs[1].plot(rec, prec, label=f"{name} (AP={ap:.3f})")

    axs[0].plot([0, 1], [0, 1], "--", color="gray", linewidth=0.8)
    axs[0].set_xlabel("Tasa de falsos positivos")
    axs[0].set_ylabel("Tasa de verdaderos positivos")
    axs[0].set_title("Curva ROC" + title_suffix)
    axs[0].legend(loc="lower right", fontsize=8)
    axs[0].grid(alpha=0.3)

    axs[1].set_xlabel("Recall")
    axs[1].set_ylabel("Precision")
    axs[1].set_title("Curva Precision-Recall" + title_suffix)
    axs[1].legend(loc="lower left", fontsize=8)
    axs[1].grid(alpha=0.3)
    fig.tight_layout()
    return fig, axs


def plot_curve(x, mean, std=None, xlabel="", ylabel="", title="", ax=None, label=None):
    """Curva métrica vs hiperparámetro, opcionalmente con banda ±1 desvío."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 3.8))
    else:
        fig = ax.figure
    ax.plot(x, mean, marker="o", label=label)
    if std is not None:
        ax.fill_between(x, np.array(mean) - np.array(std), np.array(mean) + np.array(std),
                        alpha=0.2)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if label:
        ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_importances(importances, names, top_n=10, ax=None, title="Importancia de atributos"):
    """Barras horizontales con las `top_n` importancias más altas."""
    importances = np.asarray(importances)
    order = np.argsort(importances)[::-1][:top_n]
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 0.35 * len(order) + 1.2))
    else:
        fig = ax.figure
    ax.barh(
        [names[i] for i in order][::-1],
        importances[order][::-1],
        color="#4C72B0",
    )
    ax.set_xlabel("Importancia (decremento de impureza)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_target_balance(y, labels=("0", "1"), ax=None):
    """Barras con el conteo de cada clase."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(4.5, 3.2))
    else:
        fig = ax.figure
    values, counts = np.unique(y, return_counts=True)
    ax.bar([str(v) for v in values], counts, color=["#4C72B0", "#C44E52"][: len(values)])
    for i, c in enumerate(counts):
        ax.text(i, c, f"{c}\n({c / len(y) * 100:.1f} %)", ha="center", va="bottom")
    ax.set_ylabel("Registros")
    ax.set_title("Balance de clase")
    ax.set_ylim(0, counts.max() * 1.18)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return fig, ax


def plot_missing(missing: dict, ax=None, title="Valores faltantes por columna"):
    """Barras con el porcentaje de nulos por columna (dict columna -> pct)."""
    items = sorted(missing.items(), key=lambda kv: kv[1], reverse=True)
    if not items:
        return None, None
    names = [k for k, _ in items]
    values = [v for _, v in items]
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 0.32 * len(items) + 1.3))
    else:
        fig = ax.figure
    ax.barh(names[::-1], values[::-1], color="#DD8452")
    ax.set_xlabel("% de nulos")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    return fig, ax
