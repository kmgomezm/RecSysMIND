"""Lectura de los archivos crudos de MIND."""
from pyspark.sql import functions as F, types as T

NEWS_COLS = ["news_id", "category", "subcategory", "title", "abstract",
             "url", "title_entities", "abstract_entities"]
BEH_COLS = ["impression_id", "user_id", "time", "history", "impressions"]


def _schema(cols):
    return T.StructType([T.StructField(c, T.StringType(), True) for c in cols])


def read_tsv(spark, path, cols):
    # quote = \u0000 desactiva las comillas: los títulos contienen "
    return (spark.read
            .option("sep", "\t").option("header", False)
            .option("quote", "\u0000")
            .schema(_schema(cols))
            .csv(str(path)))


def load_split(spark, cfg, split: str):
    """Devuelve (news, behaviors) crudos de un split ('train' o 'dev')."""
    d = cfg["paths"]["raw"] / cfg["splits"][split]
    news = read_tsv(spark, d / "news.tsv", NEWS_COLS).withColumn("split", F.lit(split))
    beh = (read_tsv(spark, d / "behaviors.tsv", BEH_COLS)
           .withColumn("time", F.to_timestamp("time", "M/d/yyyy h:mm:ss a"))
           .withColumn("split", F.lit(split)))
    return news, beh