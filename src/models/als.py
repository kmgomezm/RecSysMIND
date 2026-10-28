"""Filtrado colaborativo con ALS implícito (Spark MLlib).

Interacciones: historiales + clics en impresiones de entrenamiento (valor 1).
Las noticias o usuarios sin interacciones de entrenamiento no tienen vector:
ALS devuelve NaN y se asigna un puntaje neutro (promedio de la impresión) o al fondo.
"""
from pyspark.ml.feature import StringIndexer
from pyspark.ml.recommendation import ALS
from pyspark.sql import functions as F, Window


def build_interactions(beh, imp, hist_periods=("train", "val"), click_periods=("train",)):
    """Pares (usuario, noticia) observados antes del periodo a evaluar."""
    hist = (beh.filter(F.col("periodo").isin(*hist_periods))
               .select("user_id", "history").dropDuplicates(["user_id"])
               .select("user_id", F.explode("history").alias("news_id")))
    clicks = (imp.filter(F.col("periodo").isin(*click_periods) & (F.col("label") == 1))
                 .select("user_id", "news_id"))
    return hist.unionByName(clicks).distinct().withColumn("r", F.lit(1.0))


def fit_indexers(inter):
    """ALS necesita IDs enteros. Los IDs no vistos reciben el índice 'extra' (= n)."""
    u_idx = StringIndexer(inputCol="user_id", outputCol="u", handleInvalid="keep").fit(inter)
    i_idx = StringIndexer(inputCol="news_id", outputCol="i", handleInvalid="keep").fit(inter)
    return u_idx, i_idx


def add_indices(df, u_idx, i_idx):
    out = i_idx.transform(u_idx.transform(df))
    return out.withColumn("u", F.col("u").cast("int")).withColumn("i", F.col("i").cast("int"))


def fit_als(inter_idx, rank, alpha, reg=0.1, max_iter=10, seed=42):
    als = ALS(userCol="u", itemCol="i", ratingCol="r", implicitPrefs=True,
              rank=rank, alpha=alpha, regParam=reg, maxIter=max_iter,
              coldStartStrategy="nan", seed=seed)
    return als.fit(inter_idx)


def score_als(model, cands_idx, cold="neutral"):
    """Agrega 'score_als'. cold='neutral': promedio ALS de la impresión; 'bottom': al fondo."""
    pred = model.transform(cands_idx)
    pred = pred.withColumn("als", F.when(F.isnan("prediction"), F.lit(None))
                                   .otherwise(F.col("prediction")))
    if cold == "neutral":
        fill = F.coalesce(F.avg("als").over(Window.partitionBy("impression_id")), F.lit(0.0))
    else:
        fill = F.lit(-1e9)
    return pred.withColumn("score_als", F.coalesce(F.col("als"), fill))