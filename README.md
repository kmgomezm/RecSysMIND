# RecSysMIND: recomendación de noticias con Spark sobre MIND

Proyecto del curso de Minería de Grandes Volúmenes de Datos.
Compara un baseline de popularidad reciente, filtrado colaborativo (ALS),
modelos de contenido (TF-IDF y embeddings de sentence-transformers) y un
modelo híbrido, usando PySpark sobre el dataset MIND de Microsoft.

## Estructura

```
configs/      Configuración por entorno (small / large)
src/          Código reutilizable: lectura, preprocesamiento, modelos, evaluación
notebooks/    EDA, preprocesamiento, modelos y evaluación final (en orden)
scripts/      Script para MIND-large en AWS EMR
results/      Métricas (CSV) y figuras
```

## Datos

Los datos no se incluyen en el repositorio. Descargar MINDsmall y MINDlarge
(train y dev) desde https://msnews.github.io y descomprimir en
`data/raw/MINDsmall_train/`, `data/raw/MINDsmall_dev/`, etc.

## Entorno

- Python 3.11, Java 17 (Temurin), PySpark 4.0.2
- En Windows: `winutils.exe` y `hadoop.dll` (Hadoop 3.3.6) con `HADOOP_HOME`

```
py -3.11 -m venv venv
.\venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Orden de ejecución

1. `01_eda` → 2. `02_preprocesamiento` → `02b_validacion_metricas`
3. `03_baseline_popularidad` → `04_contenido_tfidf` → `04b_contenido_embeddings`
   → `05_colaborativo_als` → `06_hibrido`
4. `07_evaluacion_final_test`
5. `08a_embeddings_large` + `scripts/hybrid_large.py` (EMR)