"""
preprocessing.py
================

Pre-procesamiento y auditoría del dataset *Framingham Heart Study*
(`data_cardiovascular_risk.csv`), siguiendo los pasos vistos en clase.

El módulo hace dos cosas:

1. **Auditoría** (`audit`): radiografía del dataset *antes* de tocarlo
   (nulos, duplicados, outliers, tipos, inconsistencias, balance de clase).
   Se puede imprimir como Markdown para pegarla en el informe.
2. **Limpieza** (`Preprocessor`): pipeline con `fit`/`transform` (estilo
   scikit-learn) que imputa, winsoriza, codifica y agrega atributos. Lleva
   contadores de todo lo que modifica para poder reportar el "después".

Uso rápido
----------
>>> from preprocessing import load_raw, Preprocessor, audit, audit_to_markdown
>>> df = load_raw("data_cardiovascular_risk.csv")
>>> print(audit_to_markdown(audit(df)))
>>> prep = Preprocessor().fit(df)
>>> clean = prep.transform(df)
>>> prep.report_  # contadores de imputaciones, outliers, columnas eliminadas

Desde la terminal:

    python preprocessing.py            # genera data/dataset_procesado.csv + auditorías

Decisiones de limpieza (justificadas en el informe)
---------------------------------------------------
* `id` se elimina: es un identificador sin poder predictivo.
* No hay duplicados exactos (se verifica y se reporta).
* Los nulos se imputan (no se eliminan filas): borrar las 463 filas con algún
  nulo eliminaría 67 positivos de una clase que ya es minoritaria (15 %).
* `BPMeds` tiene nulos estructurales: si `prevalentHyp == 0` el valor correcto
  es 0 (no toma medicación para la presión). Se imputa por regla y se reporta.
* Los outliers se **winsorizan** (recorte en percentiles) en lugar de
  eliminarse, porque muchos son valores clínicamente reales (glucosa 394,
  sysBP 295) que pertenecen justamente a la clase de riesgo.
* `is_smoking` es redundante con `cigsPerDay` (0 inconsistencias). Se puede
  eliminar con `drop_is_smoking=True`; por defecto se conserva para comparar.
"""

from __future__ import annotations

import argparse
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuración del dataset
# ---------------------------------------------------------------------------
TARGET = "TenYearCHD"

# Columnas binarias (0/1) tal como vienen o codificadas desde texto
BINARY_COLS = [
    "sex",
    "is_smoking",
    "BPMeds",
    "prevalentStroke",
    "prevalentHyp",
    "diabetes",
]
# Ordinales de pocos niveles
ORDINAL_COLS = ["education"]
# Continuas
NUMERIC_COLS = [
    "age",
    "cigsPerDay",
    "totChol",
    "sysBP",
    "diaBP",
    "BMI",
    "heartRate",
    "glucose",
]
# Columnas que no son atributos predictivos
ID_COLS = ["id"]

# Rangos fisiológicamente imposibles (para detectar anomalías, no para borrar)
PLAUSIBLE_RANGES = {
    "age": (18, 100),
    "cigsPerDay": (0, 100),
    "totChol": (50, 800),
    "sysBP": (60, 300),
    "diaBP": (30, 200),
    "BMI": (10, 80),
    "heartRate": (30, 220),
    "glucose": (30, 600),
    "education": (1, 4),
}


# ---------------------------------------------------------------------------
# Carga
# ---------------------------------------------------------------------------
def load_raw(path: str = "data_cardiovascular_risk.csv") -> pd.DataFrame:
    """Carga el CSV original sin modificarlo."""
    df = pd.read_csv(path)
    return df


# ---------------------------------------------------------------------------
# Auditoría (antes del pre-procesamiento)
# ---------------------------------------------------------------------------
def audit(df: pd.DataFrame) -> dict:
    """Radiografía del dataset: nulos, duplicados, outliers y anomalías.

    Returns
    -------
    dict
        Con las claves ``shape``, ``dtypes``, ``missing``, ``duplicates``,
        ``outliers_iqr``, ``outliers_z``, ``class_balance``, ``constant``,
        ``cardinality``, ``inconsistencies`` y ``implausible``.
    """
    n = len(df)
    missing = pd.DataFrame(
        {
            "nulos": df.isna().sum(),
            "pct": df.isna().mean().mul(100).round(2),
            "tipo": df.dtypes.astype(str),
            "unicos": df.nunique(dropna=True),
        }
    ).sort_values("nulos", ascending=False)

    # Duplicados
    dup_full = int(df.duplicated().sum())
    drop_id = [c for c in ID_COLS if c in df.columns]
    dup_sin_id = int(df.drop(columns=drop_id).duplicated().sum())

    # Outliers por IQR y por z-score (sólo variables continuas: se excluyen
    # binarias, ordinales e identificadores)
    outliers_iqr, outliers_z = {}, {}
    for col in df.select_dtypes(include=[np.number]).columns:
        if col in ID_COLS or df[col].nunique(dropna=True) <= 10:
            continue
        s = df[col].dropna()
        if len(s) == 0:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers_iqr[col] = {
            "n": int(((s < lo) | (s > hi)).sum()),
            "pct": round(float(((s < lo) | (s > hi)).mean() * 100), 2),
            "limite_inf": round(float(lo), 2),
            "limite_sup": round(float(hi), 2),
        }
        if s.std(ddof=0) > 0:
            z = (s - s.mean()) / s.std(ddof=0)
            outliers_z[col] = {
                "n": int((z.abs() > 3).sum()),
                "pct": round(float((z.abs() > 3).mean() * 100), 2),
            }

    # Balance de clase
    class_balance = {}
    if TARGET in df.columns:
        vc = df[TARGET].value_counts(dropna=False)
        class_balance = {
            "conteo": {str(k): int(v) for k, v in vc.items()},
            "pct": {str(k): round(float(v / n * 100), 2) for k, v in vc.items()},
            "ratio": round(float(vc.max() / max(vc.min(), 1)), 2),
        }

    # Columnas constantes o con un solo valor
    constant = [c for c in df.columns if df[c].nunique(dropna=True) <= 1]

    # Inconsistencias lógicas específicas del dataset
    inconsistencies = {}
    if {"cigsPerDay", "is_smoking"} <= set(df.columns):
        fuma = df["is_smoking"].astype(str).str.upper().isin(["YES", "1", "1.0", "TRUE"])
        cig = pd.to_numeric(df["cigsPerDay"], errors="coerce")
        inconsistencies["cigs>0 con is_smoking=NO"] = int(((cig > 0) & ~fuma).sum())
        inconsistencies["cigs=0 con is_smoking=YES"] = int(((cig == 0) & fuma).sum())
    if {"BPMeds", "prevalentHyp"} <= set(df.columns):
        inconsistencies["BPMeds nulo con prevalentHyp=0"] = int(
            (df["BPMeds"].isna() & (df["prevalentHyp"] == 0)).sum()
        )
        inconsistencies["BPMeds nulo con prevalentHyp=1"] = int(
            (df["BPMeds"].isna() & (df["prevalentHyp"] == 1)).sum()
        )
    if {"sysBP", "diaBP"} <= set(df.columns):
        inconsistencies["diaBP >= sysBP"] = int((df["diaBP"] >= df["sysBP"]).sum())

    # Valores fuera de rango fisiológico
    implausible = {}
    for col, (lo, hi) in PLAUSIBLE_RANGES.items():
        if col in df.columns:
            s = pd.to_numeric(df[col], errors="coerce")
            n_out = int(((s < lo) | (s > hi)).sum())
            if n_out:
                implausible[col] = n_out

    return {
        "shape": (n, df.shape[1]),
        "dtypes": df.dtypes.astype(str).to_dict(),
        "missing": missing,
        "missing_total_cells": int(df.isna().sum().sum()),
        "missing_rows_any": int(df.isna().any(axis=1).sum()),
        "missing_pct_cells": round(float(df.isna().mean().mean() * 100), 2),
        "duplicates": {"exactos": dup_full, "sin_id": dup_sin_id},
        "outliers_iqr": outliers_iqr,
        "outliers_z": outliers_z,
        "class_balance": class_balance,
        "constant": constant,
        "inconsistencies": inconsistencies,
        "implausible": implausible,
    }


def audit_to_markdown(info: dict) -> str:
    """Convierte el dict de `audit` en un informe Markdown."""
    n, p = info["shape"]
    lines = [
        "# Auditoría inicial del dataset",
        "",
        f"- Registros: **{n}**",
        f"- Columnas: **{p}**",
        f"- Celdas nulas: **{info['missing_total_cells']}** "
        f"({info['missing_pct_cells']} % del total)",
        f"- Filas con al menos un nulo: **{info['missing_rows_any']}** "
        f"({info['missing_rows_any'] / n * 100:.2f} %)",
        f"- Duplicados exactos: **{info['duplicates']['exactos']}** "
        f"(sin `id`: {info['duplicates']['sin_id']})",
        "",
        "## Valores faltantes por columna",
        "",
        "| Columna | Tipo | Nulos | % | Únicos |",
        "|---|---|---:|---:|---:|",
    ]
    for col, row in info["missing"].iterrows():
        lines.append(
            f"| `{col}` | {row['tipo']} | {int(row['nulos'])} | {row['pct']:.2f} | "
            f"{int(row['unicos'])} |"
        )

    if info["class_balance"]:
        cb = info["class_balance"]
        lines += [
            "",
            "## Balance de clase",
            "",
            f"`{TARGET}`: "
            + ", ".join(f"{k} = {v} ({cb['pct'][k]} %)" for k, v in cb["conteo"].items())
            + f". Ratio mayoritaria:minoritaria = **{cb['ratio']}:1**.",
        ]

    lines += [
        "",
        "## Outliers",
        "",
        "| Columna | IQR (n) | IQR (%) | z > 3 (n) | z > 3 (%) | Límite inf | Límite sup |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for col, d in info["outliers_iqr"].items():
        z = info["outliers_z"].get(col, {"n": "-", "pct": "-"})
        lines.append(
            f"| `{col}` | {d['n']} | {d['pct']} | {z['n']} | {z['pct']} | "
            f"{d['limite_inf']} | {d['limite_sup']} |"
        )

    if info["inconsistencies"]:
        lines += ["", "## Inconsistencias detectadas", ""]
        for k, v in info["inconsistencies"].items():
            lines.append(f"- {k}: **{v}**")

    if info["implausible"]:
        lines += ["", "## Valores fuera de rango fisiológico", ""]
        for k, v in info["implausible"].items():
            lines.append(f"- `{k}`: **{v}**")

    if info["constant"]:
        lines += ["", f"Columnas constantes: {info['constant']}"]

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Pipeline de limpieza
# ---------------------------------------------------------------------------
@dataclass
class Preprocessor:
    """Pipeline de pre-procesamiento con contadores para el informe.

    Parameters
    ----------
    imputer : {'mice', 'knn', 'median', 'mean'}, default='mice'
        Estrategia para variables continuas. Las categóricas/ordinales usan la
        moda (o la regla de negocio para `BPMeds`).
    clip_outliers : bool, default=True
        Winsoriza las continuas en ``(clip_lower, clip_upper)``.
    clip_lower, clip_upper : float, default=0.01, 0.99
        Percentiles de recorte.
    drop_is_smoking : bool, default=False
        Elimina `is_smoking` (redundante con `cigsPerDay`).
    add_pulse_pressure : bool, default=True
        Agrega `pulsePressure = sysBP - diaBP`.
    drop_id : bool, default=True
    scale : bool, default=False
        Estandariza las continuas (no hace falta para árboles; útil para el
        imputador KNN y para comparar con otros modelos).
    """

    imputer: str = "mice"
    clip_outliers: bool = True
    clip_lower: float = 0.01
    clip_upper: float = 0.99
    # Columnas a winsorizar: se excluyen `age` y `cigsPerDay` a propósito,
    # porque sus colas son fisiológicamente válidas y clínicamente informativas
    # (los mayores y los fumadores empedernidos son justamente la población de
    # riesgo). Recortarlas destruiría señal.
    clip_cols: tuple = (
        "totChol",
        "sysBP",
        "diaBP",
        "BMI",
        "heartRate",
        "glucose",
    )
    drop_is_smoking: bool = False
    add_pulse_pressure: bool = True
    drop_id: bool = True
    scale: bool = False
    target: str = TARGET

    # Resultados del ajuste
    feature_names_: list = field(default_factory=list, init=False)
    report_: dict = field(default_factory=dict, init=False)

    # ------------------------------------------------------------------
    def fit(self, df: pd.DataFrame) -> "Preprocessor":
        """Aprende parámetros (medianas, modas, percentiles) del dataset."""
        df = df.copy()
        self.report_ = {
            "filas_entrada": int(len(df)),
            "columnas_entrada": int(df.shape[1]),
            "filas_eliminadas": 0,
            "columnas_eliminadas": [],
            "imputados": {},
            "winsorizados": {},
            "target_na": 0,
        }

        df = self._drop_columns(df, fit=True)
        df = self._coerce_types(df)

        # Columnas con nulos
        self._na_cols = [c for c in df.columns if df[c].isna().any() and c != self.target]

        # Parámetros de imputación
        self._impute_values = {}
        for col in self._na_cols:
            if col in NUMERIC_COLS:
                self._impute_values[col] = float(df[col].median())
            else:
                mode = df[col].mode(dropna=True)
                self._impute_values[col] = (
                    float(mode.iloc[0]) if len(mode) else np.nan
                )

        # Percentiles de winsorización (sólo continuas seleccionadas)
        self._clip_bounds = {}
        for col in NUMERIC_COLS:
            if col in df.columns and col in self.clip_cols:
                lo = float(df[col].quantile(self.clip_lower))
                hi = float(df[col].quantile(self.clip_upper))
                self._clip_bounds[col] = (lo, hi)

        # Medias/desvíos para escalado opcional
        self._scale_params = {}
        for col in NUMERIC_COLS:
            if col in df.columns:
                mu = float(df[col].mean())
                sd = float(df[col].std(ddof=0)) or 1.0
                self._scale_params[col] = (mu, sd)

        self.feature_names_ = [c for c in df.columns if c != self.target]
        self._is_fitted = True
        return self

    # ------------------------------------------------------------------
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Aplica limpieza e imputación. Devuelve un DataFrame nuevo."""
        if not getattr(self, "_is_fitted", False):
            raise RuntimeError("Llamar a fit() antes de transform()")

        df = df.copy()
        report = {
            "filas_entrada": int(len(df)),
            "filas_eliminadas": 0,
            "columnas_eliminadas": list(self.report_["columnas_eliminadas"]),
            "imputados": {},
            "winsorizados": {},
            "target_na": 0,
        }

        df = self._drop_columns(df, fit=False)
        df = self._coerce_types(df)

        if self.target in df.columns:
            before = len(df)
            df = df[df[self.target].notna()]
            report["target_na"] = before - len(df)
            report["filas_eliminadas"] += before - len(df)

        df = self._impute(df, report)
        if self.clip_outliers:
            df = self._clip(df, report)
        if self.add_pulse_pressure and {"sysBP", "diaBP"} <= set(df.columns):
            df["pulsePressure"] = df["sysBP"] - df["diaBP"]
        if self.scale:
            for col, (mu, sd) in self._scale_params.items():
                if col in df.columns:
                    df[col] = (df[col] - mu) / sd

        # Orden de columnas: atributos + target
        feats = [c for c in df.columns if c != self.target]
        if self.target in df.columns:
            df = df[feats + [self.target]]
        else:
            df = df[feats]

        self.report_ = {**self.report_, **report}
        self.report_["filas_salida"] = int(len(df))
        self.report_["columnas_salida"] = int(df.shape[1])
        self.report_["nulos_salida"] = int(df.isna().sum().sum())
        return df.reset_index(drop=True)

    # ------------------------------------------------------------------
    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    # ------------------------------------------------------------------
    # Pasos internos
    # ------------------------------------------------------------------
    def _drop_columns(self, df: pd.DataFrame, fit: bool) -> pd.DataFrame:
        to_drop = []
        if self.drop_id:
            to_drop += [c for c in ID_COLS if c in df.columns]
        if self.drop_is_smoking and "is_smoking" in df.columns:
            to_drop.append("is_smoking")
        if to_drop:
            df = df.drop(columns=to_drop)
            if fit:
                self.report_["columnas_eliminadas"] = to_drop
        return df

    def _coerce_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Normaliza tipos: binarias a 0/1, ordinales y continuas a numérico."""
        for col in BINARY_COLS:
            if col not in df.columns:
                continue
            if not pd.api.types.is_numeric_dtype(df[col]):
                mapped = (
                    df[col]
                    .astype(str)
                    .str.strip()
                    .str.upper()
                    .map({"M": 0, "F": 1, "NO": 0, "YES": 1, "0": 0, "1": 1,
                          "FALSE": 0, "TRUE": 1, "NAN": np.nan, "NONE": np.nan,
                          "": np.nan})
                )
                df[col] = mapped
            else:
                df[col] = pd.to_numeric(df[col], errors="coerce")
            # BPMeds con nulos -> Int64, el resto int
            df[col] = df[col].astype("Int64")

        for col in ORDINAL_COLS + NUMERIC_COLS:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        for col in ORDINAL_COLS:
            if col in df.columns:
                df[col] = df[col].astype("Int64")

        if self.target in df.columns:
            df[self.target] = pd.to_numeric(df[self.target], errors="coerce").astype("Int64")
        return df

    def _impute(self, df: pd.DataFrame, report: dict) -> pd.DataFrame:
        """Imputa nulos y registra cuántos valores tocó por columna."""
        # 1) Regla de negocio: BPMeds = 0 si no hay hipertensión prevalente
        if {"BPMeds", "prevalentHyp"} <= set(df.columns):
            rule = df["BPMeds"].isna() & (df["prevalentHyp"] == 0)
            n_rule = int(rule.sum())
            if n_rule:
                df.loc[rule, "BPMeds"] = 0
                report["imputados"]["BPMeds (regla prevalentHyp=0)"] = n_rule

        # 2) Columnas continuas: imputación elegida
        cont = [c for c in self._na_cols if c in df.columns and c in NUMERIC_COLS]
        if cont and df[cont].isna().any().any():
            n_before = df[cont].isna().sum().to_dict()
            if self.imputer == "mice":
                from sklearn.experimental import enable_iterative_imputer  # noqa: F401
                from sklearn.impute import IterativeImputer

                imp = IterativeImputer(
                    max_iter=10, random_state=0, sample_posterior=False
                )
                df[cont] = imp.fit_transform(df[cont])
            elif self.imputer == "knn":
                from sklearn.impute import KNNImputer

                imp = KNNImputer(n_neighbors=5)
                df[cont] = imp.fit_transform(df[cont])
            elif self.imputer in ("median", "mean"):
                for col in cont:
                    val = (
                        df[col].median() if self.imputer == "median" else df[col].mean()
                    )
                    df[col] = df[col].fillna(val)
            else:
                raise ValueError(f"imputer desconocido: {self.imputer!r}")
            for col in cont:
                if n_before.get(col, 0):
                    report["imputados"][col] = int(n_before[col])

        # 3) Resto de columnas (ordinales/binarias): moda o valor aprendido
        rest = [c for c in self._na_cols if c in df.columns and c not in cont]
        for col in rest:
            n_na = int(df[col].isna().sum())
            if n_na:
                df[col] = df[col].fillna(self._impute_values.get(col, 0))
                report["imputados"][col] = report["imputados"].get(col, 0) + n_na

        # 4) Cualquier nulo residual (p. ej. columna nueva)
        for col in df.columns:
            if col != self.target and df[col].isna().any():
                n_na = int(df[col].isna().sum())
                fill = self._impute_values.get(col, df[col].median())
                df[col] = df[col].fillna(fill)
                report["imputados"][col] = report["imputados"].get(col, 0) + n_na

        # Conversión final de enteros
        for col in BINARY_COLS + ORDINAL_COLS:
            if col in df.columns:
                df[col] = df[col].astype(int)
        return df

    def _clip(self, df: pd.DataFrame, report: dict) -> pd.DataFrame:
        """Winsoriza continuas y cuenta cuántos valores se recortaron."""
        for col, (lo, hi) in self._clip_bounds.items():
            if col not in df.columns:
                continue
            s = df[col]
            mask = (s < lo) | (s > hi)
            n = int(mask.sum())
            if n:
                df[col] = s.clip(lo, hi)
                report["winsorizados"][col] = n
        return df

    # ------------------------------------------------------------------
    def report_markdown(self) -> str:
        """Informe del 'después' del pre-procesamiento."""
        r = self.report_
        lines = [
            "# Informe del pre-procesamiento",
            "",
            f"- Filas de entrada: **{r['filas_entrada']}**",
            f"- Filas eliminadas: **{r['filas_eliminadas']}** "
            f"(target nulo: {r['target_na']})",
            f"- Filas de salida: **{r.get('filas_salida', '-')}**",
            f"- Columnas eliminadas: **{', '.join(r['columnas_eliminadas']) or 'ninguna'}**",
            f"- Columnas de salida: **{r.get('columnas_salida', '-')}**",
            f"- Nulos en la salida: **{r.get('nulos_salida', 0)}**",
            "",
            "## Valores imputados",
            "",
            "| Columna | Valores imputados |",
            "|---|---:|",
        ]
        for col, n in r["imputados"].items():
            lines.append(f"| `{col}` | {n} |")
        if not r["imputados"]:
            lines.append("| — | 0 |")

        lines += [
            "",
            "## Outliers winsorizados (p1-p99)",
            "",
            "| Columna | Valores recortados |",
            "|---|---:|",
        ]
        for col, n in r["winsorizados"].items():
            lines.append(f"| `{col}` | {n} |")
        if not r["winsorizados"]:
            lines.append("| — | 0 |")

        lines += [
            "",
            f"Total de celdas imputadas: **{sum(r['imputados'].values())}**",
            f"Total de celdas winsorizadas: **{sum(r['winsorizados'].values())}**",
        ]
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI: genera el dataset procesado y los dos informes
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Pre-procesa el dataset del TP")
    parser.add_argument("--input", default="data_cardiovascular_risk.csv")
    parser.add_argument("--output", default="data/dataset_procesado.csv")
    parser.add_argument("--imputer", default="mice",
                        choices=["mice", "knn", "median", "mean"])
    parser.add_argument("--no-clip", action="store_true")
    parser.add_argument("--drop-is-smoking", action="store_true")
    args = parser.parse_args()

    df = load_raw(args.input)
    info = audit(df)
    markdown_before = audit_to_markdown(info)

    prep = Preprocessor(
        imputer=args.imputer,
        clip_outliers=not args.no_clip,
        drop_is_smoking=args.drop_is_smoking,
    )
    clean = prep.fit_transform(df)
    markdown_after = prep.report_markdown()

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    clean.to_csv(args.output, index=False)

    with open("data/auditoria_inicial.md", "w", encoding="utf-8") as f:
        f.write(markdown_before)
    with open("data/auditoria_preprocesamiento.md", "w", encoding="utf-8") as f:
        f.write(markdown_after)

    print(markdown_before)
    print()
    print(markdown_after)
    print(f"\nDataset procesado: {args.output}  ->  {clean.shape}")


if __name__ == "__main__":
    main()
