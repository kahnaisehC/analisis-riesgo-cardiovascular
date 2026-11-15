# Informe del pre-procesamiento

- Filas de entrada: **3390**
- Filas eliminadas: **0** (target nulo: 0)
- Filas de salida: **3390**
- Columnas eliminadas: **id**
- Columnas de salida: **17**
- Nulos en la salida: **0**

## Valores imputados

| Columna | Valores imputados |
|---|---:|
| `BPMeds (regla prevalentHyp=0)` | 25 |
| `cigsPerDay` | 22 |
| `totChol` | 38 |
| `BMI` | 14 |
| `heartRate` | 1 |
| `glucose` | 304 |
| `education` | 87 |
| `BPMeds` | 19 |

## Outliers winsorizados (p1-p99)

| Columna | Valores recortados |
|---|---:|
| `totChol` | 67 |
| `sysBP` | 68 |
| `diaBP` | 66 |
| `BMI` | 68 |
| `heartRate` | 53 |
| `glucose` | 53 |

Total de celdas imputadas: **510**
Total de celdas winsorizadas: **375**