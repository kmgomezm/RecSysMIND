"""Baseline de popularidad reciente en streaming.

Para cada candidata de una impresión en el instante t, cuenta las exposiciones y
los clics que recibió esa noticia (en impresiones de cualquier usuario) entre
t - h y t, sin incluir t. Con esos conteos se calcula un CTR suavizado.
"""
from pyspark.sql import functions as F, Window


def add_recent_counts(imp, hours=(12, 24)):
    """Agrega clics_{h}h y expos_{h}h por cada ventana h (en horas)."""
    df = imp.withColumn("ts", F.unix_timestamp("time"))
    base = Window.partitionBy("news_id").orderBy("ts")
    for h in hours:
        # Marco [t - h, t - 1 s]: estrictamente antes de la impresión actual
        w = base.rangeBetween(-h * 3600, -1)
        df = (df
              .withColumn(f"clics_{h}h", F.coalesce(F.sum("label").over(w), F.lit(0)))
              .withColumn(f"expos_{h}h", F.count("label").over(w)))
    return df.drop("ts")


def smoothed_ctr(h, alpha, prior):
    """CTR suavizado: (clics + alpha·prior) / (exposiciones + alpha)."""
    return (F.col(f"clics_{h}h") + alpha * prior) / (F.col(f"expos_{h}h") + alpha)