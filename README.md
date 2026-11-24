# TP IAD 2026 — Decision Tree vs Random Forest

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/kahnaisehC/analisis-riesgo-cardiovascular/blob/main/TP_IAD_2026.ipynb)

Trabajo Práctico de **Inteligencia Artificial para Datos** (UTN F.R.Re., 2026):
pre-procesamiento del dataset *Framingham Heart Study* y comparación de
**Decision Tree (C4.5)** y **Random Forest**, ambos implementados desde cero.

Repositorio: <https://github.com/kahnaisehC/analisis-riesgo-cardiovascular>

## Archivos

| Archivo | Qué contiene |
|---|---|
| `dt_rf.py` | Implementación desde cero de `DecisionTree` (C4.5: gain ratio, Gini, splits numéricos y categóricos, poda pesimista y REP) y `RandomForest` (bagging, subespacios aleatorios, OOB, voto duro/suave). Sólo NumPy. |
| `preprocessing.py` | Auditoría del dataset (`audit`) y pipeline de limpieza (`Preprocessor`) con contadores de imputaciones, outliers y columnas eliminadas. |
| `experiments.py` | Métricas, validación cruzada estratificada, barridos de hiperparámetros y ablaciones. |
| `plots.py` | Gráficos: árbol, matriz de confusión, ROC/PR, curvas e importancias. |
| `TP_IAD_2026.ipynb` | Notebook para Google Colab: auditoría, pre-procesamiento, experimentos, comparación final e interfaz interactiva. |
| `run_experiments.py` | Reproduce todos los experimentos del informe por línea de comandos y guarda las tablas en `results/`. |
| `tune_final.py` | Grilla cruzada de hiperparámetros (DT y RF) para elegir la configuración final. |
| `finalize.py` | Entrena la configuración final, calcula CV, hold-out, Wilcoxon e importancias. |
| `make_figures.py` | Genera las figuras del informe (ROC/PR, matrices de confusión, importancias, curvas de barridos) en `results/figuras/`. |
| `tests/test_core.py` | Pruebas de regresión para imputación, `max_bins` y OOB hard voting. |
| `informe.md` | Informe completo del trabajo (fuente). |
| `informe.pdf` | Informe en PDF con tablas y figuras. |
| `TP_IAD_2026_ejecutado.ipynb` | Notebook ya ejecutado de punta a punta (evidencia de la corrida). |
| `data/dataset_procesado.csv` | Dataset limpio, imputado, codificado y con `pulsePressure`. |
| `data/auditoria_inicial.md` | Informe del dataset antes del pre-procesamiento. |
| `data/auditoria_preprocesamiento.md` | Informe del dataset después del pre-procesamiento. |
| `results/` | Tablas CSV de cada experimento, `resumen.json`, `resumen_notebook.json` y `protocolo.json`. |

## Cómo correrlo en Google Colab

**Opción rápida (recomendada):** abrir el notebook directo desde el repositorio:

<https://colab.research.google.com/github/kahnaisehC/analisis-riesgo-cardiovascular/blob/main/TP_IAD_2026.ipynb>

El notebook descarga desde GitHub los módulos y el CSV que falten en el
entorno. Sólo hay que hacer *Runtime → Run all*.

**Opción manual (subir solamente el notebook):**

1. Abrir <https://colab.research.google.com> → *Subir notebook* → `TP_IAD_2026.ipynb`.
2. Correr las celdas en orden. La primera celda (`bootstrap`) descarga desde
   GitHub, **sólo si faltan**, `dt_rf.py`, `preprocessing.py`, `experiments.py`,
   `plots.py` y `data_cardiovascular_risk.csv`. No hace falta subir nada a mano.
3. Si no hay acceso a la red, la celda lo avisa y la subida manual aparece
   cuando el notebook intenta importar los módulos o leer el CSV.
4. La sección 12 trae un panel con sliders para probar parametrizaciones sin
   editar código.

El bootstrap usa una versión fija del repositorio (constante `COMMIT` en la
primera celda) para que la ejecución sea reproducible. Si se modifican los
módulos o el CSV, actualizar ese hash (o usar `main` para seguir siempre la
última versión).

Todo funciona con las bibliotecas que Colab trae por defecto
(`numpy`, `pandas`, `matplotlib`, `scikit-learn`, `ipywidgets`). scikit-learn
se usa para métricas y validación; los imputadores, DT y RF propios están
implementados sin modelos de scikit-learn.

## Cómo correrlo localmente

```bash
pip install -r requirements.txt
python preprocessing.py            # genera data/dataset_procesado.csv + auditorías
python run_experiments.py          # reproduce las tablas del informe en results/
python make_figures.py             # genera las figuras en results/figuras/
jupyter notebook TP_IAD_2026.ipynb # o abrir el .ipynb en Colab
python -m unittest discover -s tests -v
```

`run_experiments.py` tarda varios minutos en una laptop de 4 núcleos
(la mayor parte, el barrido de Random Forest y el pre-procesamiento por fold).
Las semillas están fijas y el pre-procesamiento se ajusta dentro de cada fold,
así que las métricas se reproducen entre corridas.

## Uso rápido de los algoritmos

```python
from dt_rf import DecisionTree, RandomForest

arbol = DecisionTree(criterion="gini", max_depth=5, min_samples_leaf=20)
arbol.fit(X_train, y_train)
print(arbol.score(X_test, y_test))
print(arbol.export_text(feature_names))

bosque = RandomForest(n_estimators=200, max_features="sqrt", n_jobs=-1)
bosque.fit(X_train, y_train)
print(bosque.oob_score_)          # estimación out-of-bag
print(bosque.feature_importances_)
```

Cualquier `DataFrame` de pandas sirve como `X`: las columnas categóricas
(object, bool o enteras de pocos niveles) se parten de forma multi-vía
automáticamente.
