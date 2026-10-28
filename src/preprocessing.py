"""Transformaciones de crudo a tablas listas para modelar."""
from pyspark.sql import functions as F


def build_news(news_tr, news_dev):
    """Catálogo unificado (sin duplicados) con texto título + abstract."""
    return (news_tr.unionByName(news_dev)
            .dropDuplicates(["news_id"])
            .withColumn("title", F.trim(F.coalesce("title", F.lit(""))))
            .withColumn("abstract", F.trim(F.coalesce("abstract", F.lit(""))))
            .withColumn("text", F.trim(F.concat_ws(" ", "title", "abstract")))
            .select("news_id", "category", "subcategory", "title", "abstract", "text",
                    "title_entities", "abstract_entities"))


def enrich_behaviors(beh, validation_date: str):
    """Historial como lista, longitudes, clics y etiqueta de periodo."""
    return (beh
            .withColumn("impression_id", F.col("impression_id").cast("int"))
            .withColumn("history",
                        F.when(F.col("history").isNull(), F.array().cast("array<string>"))
                         .otherwise(F.split("history", " ")))
            .withColumn("hist_len", F.size("history"))
            .withColumn("cands", F.split("impressions", " "))
            .withColumn("imp_len", F.size("cands"))
            .withColumn("n_clicks", F.size(F.filter("cands", lambda c: c.endswith("-1"))))
            .withColumn("periodo",
                        F.when(F.col("split") == "dev", "test")
                         .when(F.to_date("time") == F.lit(validation_date), "val")
                         .otherwise("train"))
            .drop("cands"))


def explode_impressions(beh):
    """Una fila por candidata: (impresión, noticia, etiqueta, posición)."""
    return (beh
            .select("impression_id", "user_id", "time", "split", "periodo",
                    F.posexplode(F.split("impressions", " ")).alias("pos", "item"))
            .withColumn("news_id", F.split("item", "-")[0])
            .withColumn("label", F.split("item", "-")[1].cast("int"))
            .drop("item"))