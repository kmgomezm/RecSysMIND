# RecSysMIND: recomendación de noticias con Spark sobre MIND

Proyecto del curso Minería de Grandes Volúmenes de Datos (EAFIT). Compara cinco
recomendadores de noticias implementados en PySpark sobre el dataset MIND de
Microsoft: popularidad reciente (baseline), contenido con TF-IDF, contenido con
embeddings (MiniLM), filtrado colaborativo (ALS implícito) y un híbrido
CTR + embeddings.

## Resultados principales

| Modelo | AUC test (MIND-small) | AUC dev (MIND-large) |
|---|---|---|
| Híbrido (CTR + embeddings) | **0,6832** | **0,6828** |
| Popularidad (CTR 24 h) | 0,6526 | 0,6534 |
| Embeddings MiniLM | 0,6407 | 0,6401 |
| ALS implícito | 0,5305 | — |

El híbrido supera al baseline en +0,031 de AUC (IC 95 % [0,029; 0,032]).
MIND-large se procesó en AWS EMR. Métricas completas en `results/metrics/`.

## Estructura

```
configs/      Configuración por entorno
src/          Lectura, preprocesamiento, modelos y evaluación
notebooks/    Análisis y experimentos, numerados en orden de ejecución
scripts/      Script del híbrido para MIND-large en AWS EMR
results/      Métricas (CSV) y figuras
```

## Datos

No se incluyen en el repositorio. Descargar MINDsmall y MINDlarge desde
https://msnews.github.io y descomprimir en `data/raw/` (por ejemplo,
`data/raw/MINDsmall_train/`).

## Entorno

Python 3.11, Java 17 y PySpark 4.0.2. Los pasos completos, incluida la
configuración de Hadoop en Windows, están en [SETUP.md](SETUP.md).

```
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Orden de ejecución

| Paso | Notebooks |
|---|---|
| Datos | `01_eda`, `02_preprocesamiento`, `02b_validacion_metricas` |
| Modelos (validación) | `03_baseline_popularidad`, `04_contenido_tfidf`, `04b_contenido_embeddings`, `05_colaborativo_als`, `06_hibrido` |
| Evaluación en test | `07_evaluacion_final_test`, `10_analisis_errores` |
| MIND-large | `08a_embeddings_large`, `scripts/hybrid_large.py` (EMR), `08b_resultados_large` |
| Test oficial (CodaLab) | `09a_embeddings_test`, `09b_escenario_sin_clics`, `09c_submission_codalab` |