"""
dt_rf.py
========

Implementación desde cero de **Decision Tree (C4.5)** y **Random Forest**
(Breiman, 2001) usando únicamente NumPy.

El módulo no depende de scikit-learn: los árboles, la selección de splits, la
poda y el bosque están implementados a mano. Se expone una API estilo
scikit-learn (`fit`, `predict`, `predict_proba`, `score`, `get_params`,
`set_params`) para que sea cómodo de usar en el notebook y de comparar contra
`DecisionTreeClassifier` / `RandomForestClassifier`.

Referencias
-----------
- Liu, Bing. *Web Data Mining*. Springer, 2nd ed., p. 67 (C4.5).
- Breiman, L. *Random Forests*. Machine Learning 45, 5-32 (2001).

Conceptos implementados
-----------------------
Decision Tree
    * Criterios de impureza: entropía (con *gain ratio*, como C4.5) y Gini.
    * Splits binarios para atributos numéricos (umbral = punto medio entre
      valores ordenados consecutivos).
    * Splits multi-vía para atributos categóricos (una rama por categoría).
    * Parada por profundidad, mínimo de muestras por nodo/hoja y ganancia
      mínima.
    * Poda por error reducido (*Reduced Error Pruning*, REP) sobre un conjunto
      de validación.
    * Soporte de `class_weight='balanced'` para datasets desbalanceados.
    * Importancia de atributos por decremento ponderado de impureza.

Random Forest
    * Bagging: cada árbol se entrena con una muestra bootstrap.
    * Selección aleatoria de un subconjunto de atributos en cada split.
    * Voto mayoritario (duro) o promedio de probabilidades (suave).
    * Estimación *out-of-bag* (OOB) del error de generalización.
    * Importancia de atributos promediada entre árboles.

Convenciones
------------
* Las etiquetas se codifican internamente a ``0..n_classes-1`` y se devuelven
  con su valor original en `predict`.
* Los atributos categóricos se detectan automáticamente (dtype no numérico,
  booleano, o entero con pocos valores únicos) o se fuerzan con
  ``categorical_features``. Los categóricos se parten de forma multi-vía.
* Los valores faltantes en `predict` se enrutan a la rama mayoritaria del nodo
  (no se usan para elegir el split durante el entrenamiento).
"""

from __future__ import annotations

import numpy as np

try:  # joblib es opcional: sólo se usa para paralelizar el bosque
    from joblib import Parallel, delayed

    _HAS_JOBLIB = True
except Exception:  # pragma: no cover - entorno sin joblib
    Parallel = None
    delayed = None
    _HAS_JOBLIB = False


# ---------------------------------------------------------------------------
# Utilidades numéricas
# ---------------------------------------------------------------------------
def _entropy(counts: np.ndarray) -> float:
    """Entropía en bits de un vector de conteos (o pesos) por clase."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log2(p)).sum())


def _gini(counts: np.ndarray) -> float:
    """Índice de Gini de un vector de conteos (o pesos) por clase."""
    total = counts.sum()
    if total <= 0:
        return 0.0
    p = counts / total
    return float(1.0 - (p * p).sum())


def _impurity(counts: np.ndarray, criterion: str) -> float:
    if criterion == "entropy":
        return _entropy(counts)
    if criterion == "gini":
        return _gini(counts)
    raise ValueError(f"Criterio desconocido: {criterion!r} (usar 'entropy' o 'gini')")


def _split_info(weights: np.ndarray) -> float:
    """Entropía de la partición (denominador del *gain ratio* de C4.5)."""
    total = weights.sum()
    if total <= 0:
        return 0.0
    p = weights[weights > 0] / total
    return float(-(p * np.log2(p)).sum())


def _isnan(arr: np.ndarray) -> np.ndarray:
    """Máscara de faltantes que funciona con dtypes numéricos, object y str."""
    arr = np.asarray(arr)
    if arr.dtype.kind in ("f", "c"):
        return np.isnan(arr)
    if arr.dtype.kind in ("O", "U", "S"):
        return np.array(
            [v is None or (isinstance(v, float) and np.isnan(v)) for v in arr],
            dtype=bool,
        )
    return np.zeros(len(arr), dtype=bool)


def _le(col, threshold) -> np.ndarray:
    """Comparación ``<=`` silenciando el warning de NaN (se maneja aparte)."""
    with np.errstate(invalid="ignore"):
        return col <= threshold


def _infer_categorical_from_frame(X, max_cat_levels: int = 12):
    """Detecta atributos categóricos usando los dtypes de un DataFrame.

    Es necesario porque ``np.asarray(df)`` homogeniza todo a ``float`` y se
    pierde la distinción entre, por ejemplo, ``education`` (ordinal 1-4) y
    ``age`` (numérica). Devuelve una lista de índices o ``None`` si ``X`` no
    es un DataFrame.
    """
    if not hasattr(X, "dtypes"):
        return None
    out = []
    for i, (name, dtype) in enumerate(zip(X.columns, X.dtypes)):
        # str(dtype) es robusto con dtypes extendidos de pandas (Int64, string)
        dt = str(dtype).lower()
        is_int = "int" in dt and "interval" not in dt
        is_float = "float" in dt
        if not (is_int or is_float):
            out.append(i)
        elif is_int and X[name].nunique(dropna=True) <= max_cat_levels:
            out.append(i)
    return out or None


def _map_categories(col, categories, default_child: int) -> np.ndarray:
    """Traduce valores de un atributo categórico a índice de hijo.

    Los valores no vistos (o faltantes) se enrutan a ``default_child``.
    """
    out = np.full(len(col), default_child, dtype=int)
    if len(categories) == 0:
        return out
    nan_mask = _isnan(col)
    if categories.dtype.kind in ("f", "i", "u", "b"):
        valid = ~nan_mask
        if not valid.any():
            return out
        pos = np.searchsorted(categories, col[valid])
        pos = np.clip(pos, 0, len(categories) - 1)
        match = categories[pos] == col[valid]
        out[valid] = np.where(match, pos, default_child)
    else:
        mapping = {c: i for i, c in enumerate(categories)}
        for i, value in enumerate(col):
            if isinstance(value, float) and np.isnan(value):
                continue
            if value in mapping:
                out[i] = mapping[value]
    return out


# ---------------------------------------------------------------------------
# Nodo del árbol
# ---------------------------------------------------------------------------
class Node:
    """Nodo de un árbol de decisión.

    Atributos
    ---------
    feature : int | None
        Índice del atributo usado para partir. ``None`` en hojas.
    threshold : float | None
        Umbral para atributos numéricos (``x <= threshold`` va a la izquierda).
    categories : np.ndarray | None
        Categorías ordenadas para atributos categóricos.
    children : list[Node] | None
        Hijos. Para numéricos: ``[izq, der]``. Para categóricos: un hijo por
        categoría, alineado con ``categories``.
    default_child : int
        Hijo al que se enrutan valores no vistos o faltantes.
    value : np.ndarray
        Distribución de clases (probabilidades) del nodo.
    n_samples, n_weight, impurity, depth : float | int
        Estadísticas del nodo, útiles para graficar e interpretar.
    """

    __slots__ = (
        "feature",
        "threshold",
        "categories",
        "children",
        "default_child",
        "is_numeric",
        "value",
        "n_samples",
        "n_weight",
        "impurity",
        "depth",
    )

    def __init__(self, value, n_samples, n_weight, impurity, depth):
        self.feature = None
        self.threshold = None
        self.categories = None
        self.children = None
        self.default_child = 0
        self.is_numeric = True
        self.value = value
        self.n_samples = n_samples
        self.n_weight = n_weight
        self.impurity = impurity
        self.depth = depth

    @property
    def is_leaf(self) -> bool:
        return self.children is None


# ---------------------------------------------------------------------------
# Decision Tree (C4.5)
# ---------------------------------------------------------------------------
class DecisionTree:
    """Árbol de decisión C4.5 implementado desde cero.

    Parameters
    ----------
    criterion : {'entropy', 'gini'}, default='entropy'
        Impureza. Con ``'entropy'`` se usa *gain ratio* (C4.5); con ``'gini'``
        se maximiza la reducción de Gini.
    split_metric : {'gain_ratio', 'gain'}, default='gain_ratio'
        Sólo aplica con ``criterion='entropy'``. ``'gain_ratio'`` es el
        criterio de C4.5 (ganancia dividida por la entropía de la partición,
        considerando sólo atributos con ganancia >= promedio). ``'gain'`` usa
        *information gain* puro, que en este dataset generaliza mejor (ver
        informe).
    max_depth : int or None, default=10
        Profundidad máxima. ``None`` = sin límite.
    min_samples_split : int, default=2
        Mínimo de muestras (ponderadas) para intentar partir un nodo.
    min_samples_leaf : int, default=1
        Mínimo de muestras (ponderadas) en cada hijo.
    min_impurity_decrease : float, default=0.0
        Ganancia mínima (en unidades de impureza) para aceptar un split.
    max_features : {'sqrt', 'log2', None} | int | float, default=None
        Atributos evaluados en cada split. ``None`` = todos. Un ``float`` es
        una fracción. Es lo que usa Random Forest.
    max_bins : int or None, default=64
        Si un atributo numérico tiene más valores únicos, se evalúan como
        máximo ``max_bins`` umbrales candidatos (cuantiles). ``None`` = todos.
    class_weight : dict | 'balanced' | None, default=None
        Pesos por clase. ``'balanced'`` replica ``n / (n_classes * n_c)``.
    categorical_features : array-like of int, default=None
        Índices forzados como categóricos. Si es ``None`` se detectan por dtype
        y cardinalidad (ver ``_detect_categorical``).
    max_cat_levels : int, default=12
        Un atributo numérico entero con a lo sumo esta cantidad de valores
        únicos se trata como categórico (p. ej. ``education`` 1-4).
    random_state : int or None, default=None
        Semilla para el muestreo de atributos.

    Attributes
    ----------
    classes_ : np.ndarray
        Etiquetas originales de las clases.
    n_classes_ : int
    feature_importances_ : np.ndarray
        Importancia normalizada (suma 1) por decremento de impureza.
    tree_ : Node
        Raíz del árbol entrenado.
    """

    def __init__(
        self,
        criterion="entropy",
        split_metric="gain_ratio",
        max_depth=10,
        min_samples_split=2,
        min_samples_leaf=1,
        min_impurity_decrease=0.0,
        max_features=None,
        max_bins=64,
        class_weight=None,
        categorical_features=None,
        max_cat_levels=12,
        random_state=None,
    ):
        self.criterion = criterion
        self.split_metric = split_metric
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.min_impurity_decrease = min_impurity_decrease
        self.max_features = max_features
        self.max_bins = max_bins
        self.class_weight = class_weight
        self.categorical_features = categorical_features
        self.max_cat_levels = max_cat_levels
        self.random_state = random_state

    # -- API estilo sklearn ------------------------------------------------
    def get_params(self, deep=True):
        return {
            "criterion": self.criterion,
            "split_metric": self.split_metric,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "min_impurity_decrease": self.min_impurity_decrease,
            "max_features": self.max_features,
            "max_bins": self.max_bins,
            "class_weight": self.class_weight,
            "categorical_features": self.categorical_features,
            "max_cat_levels": self.max_cat_levels,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        for key, value in params.items():
            if key not in self.get_params():
                raise ValueError(f"Hiperparámetro desconocido: {key!r}")
            setattr(self, key, value)
        return self

    # -- Entrenamiento -----------------------------------------------------
    def fit(self, X, y, sample_weight=None):
        """Ajusta el árbol.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
        y : array-like of shape (n_samples,)
        sample_weight : array-like of shape (n_samples,), optional
            Pesos por muestra (Random Forest los usa como conteos bootstrap).
        """
        cat_hint = _infer_categorical_from_frame(X, self.max_cat_levels)
        X = np.asarray(X)
        if X.dtype.kind in ("O", "U", "S"):
            X = X.astype(object)
        y = np.asarray(y).ravel()
        if X.shape[0] != y.shape[0]:
            raise ValueError("X e y tienen distinto número de filas")
        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n_features_in_ = X.shape[1]
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self._cat_hint = cat_hint
        # Pesos: sample_weight * class_weight
        if sample_weight is None:
            w = np.ones(len(y), dtype=float)
        else:
            w = np.asarray(sample_weight, dtype=float).ravel()
        self._class_weight_vec = self._resolve_class_weight(y_idx, w)
        w = w * self._class_weight_vec[y_idx]

        # Mascara de atributos categóricos
        self._cat_mask = self._detect_categorical(X)

        # Importancia de atributos (acumulador)
        self._impurity_decrease = np.zeros(self.n_features_in_, dtype=float)
        self._rng = np.random.default_rng(self.random_state)
        self._w_total = w.sum()
        if self.max_bins is not None and int(self.max_bins) < 1:
            raise ValueError("max_bins debe ser None o un entero positivo")

        idx = np.arange(len(y))
        self.tree_ = self._grow(X, y_idx, w, idx, depth=0)

        total = self._impurity_decrease.sum()
        self.feature_importances_ = (
            self._impurity_decrease / total if total > 0 else self._impurity_decrease
        )
        self._X_fit = X
        self._y_fit = y_idx
        return self

    def _resolve_class_weight(self, y_idx, w):
        n_classes = self.n_classes_
        if self.class_weight is None:
            return np.ones(n_classes)
        if self.class_weight == "balanced":
            counts = np.bincount(y_idx, weights=w, minlength=n_classes)
            counts[counts == 0] = 1.0
            return w.sum() / (n_classes * counts)
        if isinstance(self.class_weight, dict):
            vec = np.ones(n_classes)
            for cls, weight in self.class_weight.items():
                positions = np.where(self.classes_ == cls)[0]
                if len(positions):
                    vec[positions[0]] = weight
            return vec
        raise ValueError("class_weight debe ser None, 'balanced' o un dict")

    def _detect_categorical(self, X) -> np.ndarray:
        """Devuelve una máscara booleana de atributos categóricos."""
        n_features = X.shape[1]
        mask = np.zeros(n_features, dtype=bool)
        if self.categorical_features is not None:
            for f in np.atleast_1d(self.categorical_features):
                mask[int(f)] = True
            return mask
        if getattr(self, "_cat_hint", None):
            for f in self._cat_hint:
                mask[int(f)] = True
            return mask
        for f in range(n_features):
            col = X[:, f]
            if col.dtype.kind in ("O", "U", "S", "b"):
                mask[f] = True
                continue
            if col.dtype.kind in ("i", "u"):
                n_unique = len(np.unique(col[~_isnan(col)]))
                if n_unique <= self.max_cat_levels:
                    mask[f] = True
        return mask

    def _grow(self, X, y_idx, w, idx, depth: int) -> Node:
        """Construye el subárbol para las muestras ``idx``."""
        y_node = y_idx[idx]
        w_node = w[idx]
        counts = np.bincount(y_node, weights=w_node, minlength=self.n_classes_)
        total_w = w_node.sum()
        value = counts / total_w if total_w > 0 else np.full(self.n_classes_, 1 / self.n_classes_)
        impurity = _impurity(counts, self.criterion)
        node = Node(value, len(idx), total_w, impurity, depth)

        # --- Condiciones de parada ---
        if (
            impurity == 0.0
            or total_w < self.min_samples_split
            or (self.max_depth is not None and depth >= self.max_depth)
            or np.count_nonzero(counts > 0) <= 1
        ):
            return node

        split = self._best_split(X, y_idx, w_node, idx)
        if split is None:
            return node
        feature, is_numeric, threshold, categories, gain, left_idx, right_idx = split

        if gain <= self.min_impurity_decrease:
            return node

        node.feature = feature
        node.is_numeric = is_numeric
        node.impurity = impurity

        if is_numeric:
            node.threshold = threshold
            node.children = [
                self._grow(X, y_idx, w, left_idx, depth + 1),
                self._grow(X, y_idx, w, right_idx, depth + 1),
            ]
            # Rama mayoritaria para valores faltantes / no vistos
            node.default_child = 0 if len(left_idx) >= len(right_idx) else 1
        else:
            node.categories = np.sort(categories)
            child_of = _map_categories(X[idx, feature], node.categories, 0)
            node.children = []
            sizes = []
            for c in range(len(node.categories)):
                child_idx = idx[child_of == c]
                node.children.append(self._grow(X, y_idx, w, child_idx, depth + 1))
                sizes.append(len(child_idx))
            node.default_child = int(np.argmax(sizes))

        # Importancia: decremento de impureza ponderado por fracción de peso
        weighted_gain = gain * (total_w / self._w_total)
        self._impurity_decrease[feature] += weighted_gain
        return node

    def _feature_subset(self, n_features: int) -> np.ndarray:
        """Atributos a evaluar en un split (todos, o muestreados al azar)."""
        mf = self.max_features
        if mf is None:
            return np.arange(n_features)
        if mf == "sqrt":
            k = max(1, int(np.sqrt(n_features)))
        elif mf == "log2":
            k = max(1, int(np.log2(n_features)))
        elif isinstance(mf, float):
            k = max(1, int(mf * n_features))
        else:
            k = int(mf)
        k = min(k, n_features)
        if k == n_features:
            return np.arange(n_features)
        return self._rng.choice(n_features, size=k, replace=False)

    def _best_split(self, X, y_idx, w_node, idx):
        """Busca el mejor split entre los atributos considerados.

        Con ``criterion='entropy'`` se aplica el criterio de C4.5: se calcula
        *gain ratio* para todos los atributos, pero sólo se consideran aquellos
        cuyo *information gain* supera el promedio (si no, el gain ratio tiende
        a elegir splits muy desbalanceados que aíslan pocos casos).

        Returns
        -------
        tuple | None
            ``(feature, is_numeric, threshold, categories, gain, left_idx,
            right_idx)``. Para categóricos ``threshold`` es ``None`` y
            ``left_idx``/``right_idx`` son ``None`` (los hijos se arman en
            ``_grow``). ``None`` si no hay split válido.
        """
        candidates = []  # (gain, score, feature, is_numeric, threshold, categories)
        n_features = X.shape[1]
        parent_counts = np.bincount(
            y_idx[idx], weights=w_node, minlength=self.n_classes_
        )
        parent_impurity = _impurity(parent_counts, self.criterion)

        for feature in self._feature_subset(n_features):
            col = X[idx, feature]
            nan_mask = _isnan(col)
            if nan_mask.all():
                continue
            valid = ~nan_mask
            col_v = col[valid]
            y_v = y_idx[idx][valid]
            w_v = w_node[valid]

            if self._cat_mask[feature]:
                found = self._best_categorical_split(col_v, y_v, w_v, parent_impurity)
            else:
                found = self._best_numeric_split(
                    col_v,
                    y_v,
                    w_v,
                    parent_impurity,
                    missing_y=y_idx[idx][nan_mask],
                    missing_w=w_node[nan_mask],
                )
            if found is None:
                continue
            gain, score, extra = found
            candidates.append((gain, score, int(feature), extra))

        if not candidates:
            return None

        # Heurística de C4.5: filtrar por ganancia >= promedio y, entre esos,
        # quedarse con el mayor gain ratio.
        if self.criterion == "entropy" and self.split_metric == "gain_ratio":
            avg_gain = np.mean([c[0] for c in candidates])
            eligible = [c for c in candidates if c[0] >= avg_gain]
            if not eligible:
                eligible = candidates
        else:
            eligible = candidates
        eligible.sort(key=lambda c: (c[1], c[0]), reverse=True)
        gain, score, feature, extra = eligible[0]

        # Recalcular índices de los hijos para el split ganador
        col = X[idx, feature]
        if not self._cat_mask[feature]:
            threshold = extra
            nan_mask = _isnan(col)
            go_left = _le(col, threshold)
            go_left[nan_mask] = True  # faltantes a la rama izquierda
            left_idx = idx[go_left]
            right_idx = idx[~go_left]
            if (
                w_node[go_left].sum() < self.min_samples_leaf
                or w_node[~go_left].sum() < self.min_samples_leaf
            ):
                return None
            return feature, True, threshold, None, gain, left_idx, right_idx
        return feature, False, None, extra, gain, None, None

    def _best_numeric_split(
        self, col, y, w, parent_impurity, missing_y=None, missing_w=None
    ):
        """Mejor umbral ``<=`` para un atributo numérico (barrido ordenado)."""
        order = np.argsort(col, kind="mergesort")
        x_sorted = col[order]
        y_sorted = y[order]
        w_sorted = w[order]

        # Conteos acumulados ponderados por clase a la izquierda
        onehot = np.zeros((len(y_sorted), self.n_classes_))
        onehot[np.arange(len(y_sorted)), y_sorted] = w_sorted
        cum = np.cumsum(onehot, axis=0)
        total_valid = cum[-1]
        left_w = cum.sum(axis=1)
        if missing_y is None:
            missing_y = np.empty(0, dtype=int)
        if missing_w is None:
            missing_w = np.empty(0, dtype=float)
        missing_counts = np.bincount(
            np.asarray(missing_y, dtype=int),
            weights=np.asarray(missing_w, dtype=float),
            minlength=self.n_classes_,
        )
        missing_total = float(missing_counts.sum())
        total = total_valid + missing_counts

        # Sólo posiciones donde el valor cambia
        distinct = np.where(x_sorted[:-1] < x_sorted[1:])[0]
        if len(distinct) == 0:
            return None

        # Reducción de candidatos por cuantiles si hay demasiados valores.
        # No concatenar con ``distinct``: eso conservaría todos los
        # candidatos y haría que max_bins no tuviera ningún efecto.
        if self.max_bins is not None and len(distinct) > self.max_bins:
            quantiles = np.linspace(
                0, 1, int(self.max_bins) + 2, dtype=float
            )[1:-1]
            positions = np.rint(quantiles * (len(distinct) - 1)).astype(int)
            distinct = np.unique(distinct[positions])
            distinct = distinct[(x_sorted[distinct] < x_sorted[distinct + 1])]

        left_counts = cum[distinct] + missing_counts
        right_counts = total - left_counts
        # Los faltantes se enrutan a la izquierda, igual que en la predicción.
        wl = left_w[distinct] + missing_total
        wr = total_valid.sum() - left_w[distinct]
        if np.any(wl <= 0) or np.any(wr <= 0):
            keep = (wl > 0) & (wr > 0)
            if not keep.any():
                return None
            distinct, left_counts, right_counts = distinct[keep], left_counts[keep], right_counts[keep]
            wl, wr = wl[keep], wr[keep]

        # Impureza de cada hijo
        if self.criterion == "entropy":
            pl = left_counts / wl[:, None]
            pr = right_counts / wr[:, None]
            with np.errstate(divide="ignore", invalid="ignore"):
                il = -np.nansum(np.where(pl > 0, pl * np.log2(pl), 0.0), axis=1)
                ir = -np.nansum(np.where(pr > 0, pr * np.log2(pr), 0.0), axis=1)
        else:
            pl = left_counts / wl[:, None]
            pr = right_counts / wr[:, None]
            il = 1.0 - (pl * pl).sum(axis=1)
            ir = 1.0 - (pr * pr).sum(axis=1)

        w_total = total.sum()
        child_impurity = (wl * il + wr * ir) / w_total
        gain = parent_impurity - child_impurity

        if self.criterion == "entropy" and self.split_metric == "gain_ratio":
            # C4.5: gain ratio = ganancia / entropía de la partición
            pl_w = wl / w_total
            pr_w = wr / w_total
            with np.errstate(divide="ignore", invalid="ignore"):
                si = -(pl_w * np.log2(pl_w) + pr_w * np.log2(pr_w))
            score = np.where(si > 0, gain / si, 0.0)
        else:
            score = gain

        # Respetar min_samples_leaf (en muestras, no en peso)
        n_left = distinct + 1
        n_right = len(x_sorted) - n_left
        score = np.where(
            (n_left >= self.min_samples_leaf) & (n_right >= self.min_samples_leaf),
            score,
            -np.inf,
        )
        best_pos = int(np.argmax(score))
        if not np.isfinite(score[best_pos]) or gain[best_pos] <= 0:
            return None
        threshold = (x_sorted[distinct[best_pos]] + x_sorted[distinct[best_pos] + 1]) / 2.0
        return float(gain[best_pos]), float(score[best_pos]), float(threshold)

    def _best_categorical_split(self, col, y, w, parent_impurity):
        """Mejor partición multi-vía para un atributo categórico (C4.5)."""
        levels, inverse = np.unique(col, return_inverse=True)
        if len(levels) < 2:
            return None

        counts = np.zeros((len(levels), self.n_classes_))
        np.add.at(counts, (inverse, y), w)
        level_w = counts.sum(axis=1)

        keep = level_w >= self.min_samples_leaf
        if keep.sum() < 2:
            return None
        levels = levels[keep]
        counts = counts[keep]
        level_w = level_w[keep]

        total_w = level_w.sum()
        w_total = total_w
        child_impurity = 0.0
        for c, lw in zip(counts, level_w):
            child_impurity += (lw / w_total) * _impurity(c, self.criterion)
        gain = parent_impurity - child_impurity

        if self.criterion == "entropy" and self.split_metric == "gain_ratio":
            si = _split_info(level_w)
            score = gain / si if si > 0 else 0.0
        else:
            score = gain

        if gain <= 0 or score <= 0:
            return None
        return float(gain), float(score), levels

    # -- Poda --------------------------------------------------------------
    def prune(self, X_val=None, y_val=None, method="rep", cf=0.25):
        """Poda el árbol. Modifica el árbol *in place* y devuelve ``self``.

        Parameters
        ----------
        X_val, y_val : array-like, optional
            Conjunto de validación. Requerido con ``method='rep'``.
        method : {'rep', 'pessimistic'}, default='rep'
            * ``'rep'``: *Reduced Error Pruning*. Recorre el árbol de abajo
              hacia arriba y reemplaza un subárbol por una hoja si eso no
              empeora la exactitud en el conjunto de validación.
            * ``'pessimistic'``: poda pesimista de C4.5. No necesita datos
              extra: estima el error de cada subárbol con una cota superior
              binomial (nivel de confianza ``1 - cf``) y poda si la hoja
              estima menos error que el subárbol.
        cf : float, default=0.25
            Nivel de confianza de la poda pesimista (C4.5 usa 0.25).
        """
        if method == "pessimistic":
            self._prune_pessimistic(cf)
            self._refresh(self.tree_, 0)
            self._recompute_feature_importances()
            return self
        if method != "rep":
            raise ValueError("method debe ser 'rep' o 'pessimistic'")
        if X_val is None or y_val is None:
            raise ValueError("La poda REP requiere X_val e y_val")

        X_val = np.asarray(X_val)
        y_val = np.asarray(y_val).ravel()
        if not np.isin(np.unique(y_val), self.classes_).all():
            raise ValueError("y_val contiene clases no vistas en entrenamiento")
        y_val_idx = np.searchsorted(self.classes_, y_val)

        # Repartir las muestras de validación por nodo
        at_node: dict[int, list[int]] = {}

        def route(node, idx):
            at_node.setdefault(id(node), []).extend(idx.tolist())
            if node.is_leaf or len(idx) == 0:
                return
            col = X_val[idx, node.feature]
            if node.is_numeric:
                mask = _le(col, node.threshold)
                mask[_isnan(col)] = True
                route(node.children[0], idx[mask])
                route(node.children[1], idx[~mask])
            else:
                child_of = _map_categories(col, node.categories, 0)
                for c, child in enumerate(node.children):
                    route(child, idx[child_of == c])

        route(self.tree_, np.arange(len(X_val)))

        def majority(values):
            if len(values) == 0:
                return None
            return np.bincount(values, minlength=self.n_classes_).argmax()

        def post_order(node):
            if node.is_leaf:
                return
            for child in node.children:
                post_order(child)
            vals = np.array(at_node.get(id(node), []), dtype=int)
            if len(vals) == 0:
                return
            y_node = y_val_idx[vals]

            # Exactitud del subárbol en validación
            pred = self._predict_node(node, X_val[vals])
            acc_subtree = (pred == y_node).mean()

            # Exactitud si el nodo fuera hoja
            cls = majority(y_node)
            acc_leaf = (y_node == cls).mean() if cls is not None else 0.0

            if acc_leaf >= acc_subtree:
                node.children = None
                node.feature = None
                node.threshold = None
                node.categories = None

        post_order(self.tree_)
        # Recalcular profundidades/hojas
        self._refresh(self.tree_, 0)
        self._recompute_feature_importances()
        return self

    def _refresh(self, node, depth):
        node.depth = depth
        if not node.is_leaf:
            for child in node.children:
                self._refresh(child, depth + 1)

    def _recompute_feature_importances(self):
        """Recalcula las importancias después de modificar/podar el árbol."""
        imp = np.zeros(self.n_features_in_, dtype=float)

        def visit(node):
            if node.is_leaf:
                return
            total = float(node.n_weight)
            if total > 0:
                child_impurity = sum(
                    float(child.n_weight) * float(child.impurity)
                    for child in node.children
                ) / total
                gain = max(float(node.impurity) - child_impurity, 0.0)
                imp[node.feature] += gain * (total / self._w_total)
            for child in node.children:
                visit(child)

        visit(self.tree_)
        total = imp.sum()
        self._impurity_decrease = imp
        self.feature_importances_ = imp / total if total > 0 else imp

    @staticmethod
    def _binomial_upper(errors: float, n: float, cf: float) -> float:
        """Cota superior del error (C4.5, nivel de confianza ``1 - cf``)."""
        if n <= 0:
            return float("inf")
        from statistics import NormalDist

        z = NormalDist().inv_cdf(1 - cf)
        e = errors
        return (
            e + 0.5 + z * z / 2 + z * np.sqrt(max(e * (n - e) / n + z * z / 4, 0.0))
        ) / (n + z * z)

    def _prune_pessimistic(self, cf: float):
        """Poda pesimista: compara el error estimado de la hoja vs el subárbol.

        La comparación se hace en **cantidad estimada de errores** (tasa de la
        cota binomial × n), no en tasas: sumar tasas de hojas con distinto
        tamaño subestimaría el error del subárbol y podaría todo el árbol.
        """

        def post_order(node):
            n = max(node.n_samples, 1)
            errors = n * (1.0 - float(node.value.max()))
            leaf_error = self._binomial_upper(errors, n, cf) * n
            if node.is_leaf:
                return leaf_error, 1
            subtree_error, leaves = 0.0, 0
            for child in node.children:
                err, n_leaves = post_order(child)
                subtree_error += err
                leaves += n_leaves
            if leaf_error <= subtree_error:
                node.children = None
                node.feature = None
                node.threshold = None
                node.categories = None
                return leaf_error, 1
            return subtree_error, leaves

        post_order(self.tree_)

    def _predict_node(self, node, X):
        """Recorre el árbol para un lote de muestras (versión vectorizada)."""
        out = np.empty(len(X), dtype=int)
        stack = [(node, np.arange(len(X)))]
        while stack:
            current, idx = stack.pop()
            if len(idx) == 0:
                continue
            if current.is_leaf:
                out[idx] = int(np.argmax(current.value))
                continue
            col = X[idx, current.feature]
            if current.is_numeric:
                mask = _le(col, current.threshold)
                mask[_isnan(col)] = current.default_child == 0
                stack.append((current.children[0], idx[mask]))
                stack.append((current.children[1], idx[~mask]))
            else:
                child_of = _map_categories(col, current.categories, current.default_child)
                for c in range(len(current.children)):
                    stack.append((current.children[c], idx[child_of == c]))
        return out

    # -- Predicción --------------------------------------------------------
    def predict_proba(self, X):
        """Probabilidades de clase (frecuencias ponderadas de la hoja)."""
        X = np.asarray(X)
        n = len(X)
        out = np.zeros((n, self.n_classes_))
        stack = [(self.tree_, np.arange(n))]
        while stack:
            node, idx = stack.pop()
            if len(idx) == 0:
                continue
            if node.is_leaf:
                out[idx] = node.value
                continue
            col = X[idx, node.feature]
            if node.is_numeric:
                mask = _le(col, node.threshold)
                mask[_isnan(col)] = node.default_child == 0
                stack.append((node.children[0], idx[mask]))
                stack.append((node.children[1], idx[~mask]))
            else:
                child_of = _map_categories(col, node.categories, node.default_child)
                for c in range(len(node.children)):
                    stack.append((node.children[c], idx[child_of == c]))
        return out

    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[proba.argmax(axis=1)]

    def score(self, X, y):
        """Exactitud (accuracy)."""
        return float((self.predict(X) == np.asarray(y).ravel()).mean())

    # -- Utilidades --------------------------------------------------------
    def get_depth(self):
        return self._max_depth(self.tree_)

    def _max_depth(self, node):
        if node.is_leaf:
            return node.depth
        return max(self._max_depth(c) for c in node.children)

    def get_n_leaves(self):
        return self._count_leaves(self.tree_)

    def _count_leaves(self, node):
        if node.is_leaf:
            return 1
        return sum(self._count_leaves(c) for c in node.children)

    def export_text(self, feature_names=None):
        """Devuelve las reglas del árbol como texto indentado."""
        names = (
            list(feature_names)
            if feature_names is not None
            else [f"x{i}" for i in range(self.n_features_in_)]
        )
        lines: list[str] = []

        def walk(node, prefix=""):
            cls = self.classes_[int(np.argmax(node.value))]
            if node.is_leaf:
                lines.append(
                    f"{prefix}=> clase={cls}  (n={node.n_samples}, "
                    f"p={node.value.max():.3f})"
                )
                return
            name = names[node.feature]
            if node.is_numeric:
                lines.append(f"{prefix}si {name} <= {node.threshold:.4g}:")
                walk(node.children[0], prefix + "  ")
                lines.append(f"{prefix}sino:")
                walk(node.children[1], prefix + "  ")
            else:
                for cat, child in zip(node.categories, node.children):
                    lines.append(f"{prefix}si {name} == {cat!r}:")
                    walk(child, prefix + "  ")

        walk(self.tree_)
        return "\n".join(lines)

    def get_n_nodes(self):
        return len(self._nodes())

    def _nodes(self):
        out, stack = [], [self.tree_]
        while stack:
            n = stack.pop()
            out.append(n)
            if not n.is_leaf:
                stack.extend(n.children)
        return out


# ---------------------------------------------------------------------------
# Random Forest
# ---------------------------------------------------------------------------
class RandomForest:
    """Random Forest desde cero (bagging + subespacios aleatorios).

    Parameters
    ----------
    n_estimators : int, default=100
        Cantidad de árboles.
    criterion : {'entropy', 'gini'}, default='entropy'
    split_metric : {'gain_ratio', 'gain'}, default='gain_ratio'
        Criterio de selección del split (ver `DecisionTree`).
    max_depth : int or None, default=12
    min_samples_split : int, default=2
    min_samples_leaf : int, default=1
    max_features : {'sqrt', 'log2', None} | int | float, default='sqrt'
        Atributos evaluados por split. ``'sqrt'`` es el valor clásico de
        Breiman para clasificación.
    max_samples : int or float or None, default=None
        Tamaño del bootstrap. ``None`` = n_samples. Un ``float`` es fracción.
    bootstrap : bool, default=True
    voting : {'soft', 'hard'}, default='soft'
        ``'soft'`` promedia probabilidades; ``'hard'`` cuenta votos.
    class_weight : dict | 'balanced' | None, default=None
    categorical_features : array-like of int, default=None
    max_bins : int or None, default=32
    n_jobs : int or None, default=None
        Paralelización con joblib. ``None``/1 = serial, ``-1`` = todos los cores.
    random_state : int or None, default=None

    Attributes
    ----------
    estimators_ : list[DecisionTree]
    oob_score_ : float
        Exactitud out-of-bag (sólo si ``bootstrap=True``).
    oob_decision_function_ : np.ndarray
    feature_importances_ : np.ndarray
    """

    def __init__(
        self,
        n_estimators=100,
        criterion="entropy",
        split_metric="gain_ratio",
        max_depth=12,
        min_samples_split=2,
        min_samples_leaf=1,
        max_features="sqrt",
        max_samples=None,
        bootstrap=True,
        voting="soft",
        class_weight=None,
        categorical_features=None,
        max_bins=32,
        n_jobs=None,
        random_state=None,
    ):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.split_metric = split_metric
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.max_features = max_features
        self.max_samples = max_samples
        self.bootstrap = bootstrap
        self.voting = voting
        self.class_weight = class_weight
        self.categorical_features = categorical_features
        self.max_bins = max_bins
        self.n_jobs = n_jobs
        self.random_state = random_state

    def get_params(self, deep=True):
        return {
            "n_estimators": self.n_estimators,
            "criterion": self.criterion,
            "split_metric": self.split_metric,
            "max_depth": self.max_depth,
            "min_samples_split": self.min_samples_split,
            "min_samples_leaf": self.min_samples_leaf,
            "max_features": self.max_features,
            "max_samples": self.max_samples,
            "bootstrap": self.bootstrap,
            "voting": self.voting,
            "class_weight": self.class_weight,
            "categorical_features": self.categorical_features,
            "max_bins": self.max_bins,
            "n_jobs": self.n_jobs,
            "random_state": self.random_state,
        }

    def set_params(self, **params):
        for key, value in params.items():
            if key not in self.get_params():
                raise ValueError(f"Hiperparámetro desconocido: {key!r}")
            setattr(self, key, value)
        return self

    # -- Entrenamiento -----------------------------------------------------
    def fit(self, X, y):
        cat_hint = _infer_categorical_from_frame(X, 12)
        if self.categorical_features is not None:
            cat_features = self.categorical_features
        else:
            cat_features = cat_hint
        self._cat_features_resolved = cat_features
        X = np.asarray(X)
        if X.dtype.kind in ("O", "U", "S"):
            X = X.astype(object)
        y = np.asarray(y).ravel()
        n = len(y)
        if X.ndim == 1:
            X = X.reshape(-1, 1)
        if X.shape[0] != n:
            raise ValueError("X e y tienen distinto número de filas")
        if self.n_estimators < 1:
            raise ValueError("n_estimators debe ser un entero positivo")
        if self.voting not in ("soft", "hard"):
            raise ValueError("voting debe ser 'soft' o 'hard'")
        self.classes_, y_idx = np.unique(y, return_inverse=True)
        self.n_classes_ = len(self.classes_)
        self.n_features_in_ = X.shape[1]
        self._rng = np.random.default_rng(self.random_state)

        if self.max_samples is None:
            n_boot = n
        elif isinstance(self.max_samples, float):
            n_boot = max(1, int(self.max_samples * n))
        else:
            n_boot = int(self.max_samples)

        seeds = self._rng.integers(0, 2**31 - 1, size=self.n_estimators)

        def build(seed):
            rng = np.random.default_rng(seed)
            if self.bootstrap:
                sample_idx = rng.integers(0, n, size=n_boot)
                counts = np.bincount(sample_idx, minlength=n).astype(float)
                in_bag = counts > 0
                oob = np.where(~in_bag)[0]
            else:
                counts = np.ones(n)
                oob = np.arange(n)  # sin bootstrap no hay OOB real
            tree = DecisionTree(
                criterion=self.criterion,
                split_metric=self.split_metric,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                max_bins=self.max_bins,
                class_weight=self.class_weight,
                categorical_features=cat_features,
                random_state=int(seed),
            )
            tree.fit(X, y, sample_weight=counts)
            return tree, oob

        tasks = seeds
        if _HAS_JOBLIB and (self.n_jobs is None or self.n_jobs != 1):
            results = Parallel(n_jobs=self.n_jobs or 1)(
                delayed(build)(s) for s in tasks
            )
        else:
            results = [build(s) for s in tasks]

        self.estimators_ = [tree for tree, _ in results]
        self._oob_indices = [oob for _, oob in results]

        # Importancias promediadas
        imp = np.mean([t.feature_importances_ for t in self.estimators_], axis=0)
        self.feature_importances_ = imp / imp.sum() if imp.sum() > 0 else imp

        if self.bootstrap:
            self._compute_oob(X, y_idx)
        return self

    def _compute_oob(self, X, y_idx):
        """Exactitud out-of-bag: cada muestra se predice con los árboles que no
        la vieron durante el bootstrap."""
        n = len(X)
        proba_sum = np.zeros((n, self.n_classes_))
        counts = np.zeros(n)
        for tree, oob in zip(self.estimators_, self._oob_indices):
            if len(oob) == 0:
                continue
            if self.voting == "hard":
                votes = tree.predict(X[oob])
                vote_idx = np.searchsorted(self.classes_, votes)
                proba_sum[oob, vote_idx] += 1.0
            else:
                tree_proba = tree.predict_proba(X[oob])
                for j, cls in enumerate(tree.classes_):
                    global_j = int(np.searchsorted(self.classes_, cls))
                    proba_sum[oob, global_j] += tree_proba[:, j]
            counts[oob] += 1
        has_pred = counts > 0
        self.oob_decision_function_ = np.full((n, self.n_classes_), np.nan)
        self.oob_decision_function_[has_pred] = (
            proba_sum[has_pred] / counts[has_pred, None]
        )
        if has_pred.sum() == 0:
            self.oob_score_ = float("nan")
            return
        pred = self.oob_decision_function_[has_pred].argmax(axis=1)
        self.oob_score_ = float((pred == y_idx[has_pred]).mean())

    # -- Predicción --------------------------------------------------------
    def _aligned_tree_proba(self, tree, X):
        """Alinea las columnas de un árbol con ``self.classes_``."""
        raw = tree.predict_proba(X)
        out = np.zeros((len(X), self.n_classes_), dtype=float)
        for j, cls in enumerate(tree.classes_):
            global_j = int(np.searchsorted(self.classes_, cls))
            out[:, global_j] = raw[:, j]
        return out

    def predict_proba(self, X):
        X = np.asarray(X)
        if self.voting == "hard":
            # Fracción de árboles que votan cada clase: es la "probabilidad"
            # coherente con la predicción por mayoría (y permite calcular
            # ROC-AUC / PR-AUC del voto duro).
            votes = np.stack([t.predict(X) for t in self.estimators_], axis=0)
            proba = np.zeros((X.shape[0], self.n_classes_))
            for j, cls in enumerate(self.classes_):
                proba[:, j] = (votes == cls).mean(axis=0)
            return proba
        return np.mean(
            [self._aligned_tree_proba(t, X) for t in self.estimators_], axis=0
        )

    def predict(self, X):
        if self.voting == "soft":
            return self.classes_[self.predict_proba(X).argmax(axis=1)]
        # Voto duro
        votes = np.stack([t.predict(X) for t in self.estimators_])
        out = []
        for row in votes.T:
            vals, cnt = np.unique(row, return_counts=True)
            out.append(vals[cnt.argmax()])
        return np.asarray(out)

    def score(self, X, y):
        return float((self.predict(X) == np.asarray(y).ravel()).mean())
