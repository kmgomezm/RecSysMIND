"""Modelo de contenido con embeddings densos (sentence-transformers).

- Noticias: embedding del texto (título + abstract), normalizado (norma 1).
- Usuario: promedio de los embeddings de su historial, normalizado.
- Puntaje: coseno = producto punto, calculado con funciones nativas de Spark SQL.
"""
from pyspark.ml.functions import array_to_vector, vector_to_array
from pyspark.ml.stat import Summarizer
from pyspark.sql import functions as F

from src.models.content import last_k


def encode_news(news_pdf, model_name, batch_size=128):
    """Codifica el texto de las noticias (pandas). Se ejecuta fuera de Spark."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name, device="cpu")
    emb = model.encode(news_pdf["text"].tolist(), batch_size=batch_size,
                       normalize_embeddings=True, show_progress_bar=True,
                       convert_to_numpy=True)
    return emb


def user_profiles_dense(users, news_emb, k=None):
    """Perfil = promedio de los embeddings del historial, normalizado a norma 1."""
    prof = (users.select("user_id", F.explode(last_k("history", k)).alias("news_id"))
                 .join(news_emb, "news_id")
                 .withColumn("v", array_to_vector("emb"))
                 .groupBy("user_id")
                 .agg(Summarizer.mean(F.col("v")).alias("m"))
                 .withColumn("u", vector_to_array("m")))
    norma = F.expr("sqrt(aggregate(transform(u, x -> x * x), 0D, (acc, x) -> acc + x))")
    return (prof.withColumn("norma", norma)
                .withColumn("u_emb", F.expr("transform(u, x -> x / norma)"))
                .select("user_id", "u_emb"))


def score_dense(cands, news_emb, profiles):
    """Agrega 'cos' = producto punto perfil–noticia (ambos normalizados)."""
    dot = F.expr("aggregate(zip_with(u_emb, emb, (x, y) -> x * y), 0D, (acc, z) -> acc + z)")
    return (cands
            .join(F.broadcast(news_emb), "news_id", "left")
            .join(F.broadcast(profiles), "user_id", "left")
            .withColumn("cos", F.when(F.col("u_emb").isNotNull(), dot))
            .drop("emb", "u_emb"))