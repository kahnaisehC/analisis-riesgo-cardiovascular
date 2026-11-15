# Auditoría inicial del dataset

- Registros: **3390**
- Columnas: **17**
- Celdas nulas: **510** (0.88 % del total)
- Filas con al menos un nulo: **463** (13.66 %)
- Duplicados exactos: **0** (sin `id`: 0)

## Valores faltantes por columna

| Columna | Tipo | Nulos | % | Únicos |
|---|---|---:|---:|---:|
| `glucose` | float64 | 304 | 8.97 | 132 |
| `education` | float64 | 87 | 2.57 | 4 |
| `BPMeds` | float64 | 44 | 1.30 | 2 |
| `totChol` | float64 | 38 | 1.12 | 240 |
| `cigsPerDay` | float64 | 22 | 0.65 | 32 |
| `BMI` | float64 | 14 | 0.41 | 1259 |
| `heartRate` | float64 | 1 | 0.03 | 68 |
| `id` | int64 | 0 | 0.00 | 3390 |
| `age` | int64 | 0 | 0.00 | 39 |
| `prevalentHyp` | int64 | 0 | 0.00 | 2 |
| `prevalentStroke` | int64 | 0 | 0.00 | 2 |
| `sex` | str | 0 | 0.00 | 2 |
| `is_smoking` | str | 0 | 0.00 | 2 |
| `diaBP` | float64 | 0 | 0.00 | 142 |
| `sysBP` | float64 | 0 | 0.00 | 226 |
| `diabetes` | int64 | 0 | 0.00 | 2 |
| `TenYearCHD` | int64 | 0 | 0.00 | 2 |

## Balance de clase

`TenYearCHD`: 0 = 2879 (84.93 %), 1 = 511 (15.07 %). Ratio mayoritaria:minoritaria = **5.63:1**.

## Outliers

| Columna | IQR (n) | IQR (%) | z > 3 (n) | z > 3 (%) | Límite inf | Límite sup |
|---|---:|---:|---:|---:|---:|---:|
| `age` | 0 | 0.0 | 0 | 0.0 | 21.0 | 77.0 |
| `cigsPerDay` | 9 | 0.27 | 17 | 0.5 | -30.0 | 50.0 |
| `totChol` | 43 | 1.28 | 19 | 0.57 | 119.0 | 351.0 |
| `sysBP` | 105 | 3.1 | 39 | 1.15 | 76.5 | 184.5 |
| `diaBP` | 58 | 1.71 | 36 | 1.06 | 51.25 | 113.25 |
| `BMI` | 77 | 2.28 | 46 | 1.36 | 15.49 | 35.57 |
| `heartRate` | 64 | 1.89 | 19 | 0.56 | 45.5 | 105.5 |
| `glucose` | 158 | 5.12 | 42 | 1.36 | 47.0 | 111.0 |

## Inconsistencias detectadas

- cigs>0 con is_smoking=NO: **0**
- cigs=0 con is_smoking=YES: **0**
- BPMeds nulo con prevalentHyp=0: **25**
- BPMeds nulo con prevalentHyp=1: **19**
- diaBP >= sysBP: **0**