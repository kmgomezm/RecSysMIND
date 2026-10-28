"""Modelo basado en contenido.

- Noticias: TF-IDF de título + abstract, normalizado (norma L2 = 1).
- Usuario: TF-IDF del texto concatenado de su historial (completo o últimas K noticias).
- Puntaje: similitud coseno = producto punto de vectores normalizados.
"""
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import RegexTokenizer, StopWordsRemover, CountVectorizer, IDF, Normalizer
from pyspark.sql import functions as F, types as T


def fit_tfidf(news, min_df=2):
    """Ajusta el pipeline sobre el catálogo. Devuelve el modelo completo
    (texto -> vector) y el 'vectorizador' (tokens -> vector) para los perfiles."""
    pipeline = Pipeline(stages=[
        RegexTokenizer(inputCol="text", outputCol="tokens_raw", pattern=r"\W+", minTokenLength=2),
        StopWordsRemover(inputCol="tokens_raw", outputCol="tokens"),
        CountVectorizer(inputCol="tokens", outputCol="tf", minDF=min_df),
        IDF(inputCol="tf", outputCol="tfidf"),
        Normalizer(inputCol="tfidf", outputCol="vec", p=2.0),
    ])
    model = pipeline.fit(news)
    vectorizer = PipelineModel(stages=model.stages[2:])   # CountVectorizer + IDF + Normalizer
    return model, vectorizer


def last_k(col, k=None):
    """Últimas k noticias del historial (está en orden cronológico). k=None: todo."""
    if k is None:
        return F.col(col)
    # slice con inicio protegido: si el historial tiene menos de k, se toma completo
    return F.slice(col, F.greatest(F.size(col) - k + 1, F.lit(1)), F.lit(k))


def user_profiles(users, news_tok, vectorizer, k=None):
    """Perfil = TF-IDF del texto concatenado de las noticias del historial.
    Los usuarios sin historial no aparecen en la salida."""
    docs = (users.select("user_id", F.explode(last_k("history", k)).alias("news_id"))
                 .join(news_tok.select("news_id", "tokens"), "news_id")
                 .groupBy("user_id")
                 .agg(F.flatten(F.collect_list("tokens")).alias("tokens")))
    return vectorizer.transform(docs).select("user_id", F.col("vec").alias("u_vec"))


@F.udf(T.DoubleType())
def cosine(u, v):
    """Producto punto de dos vectores normalizados (= coseno)."""
    if u is None or v is None:
        return None
    return float(u.dot(v))


def score_content(cands, news_vec, profiles):
    """Agrega la columna 'cos' a las candidatas (impresión, usuario, noticia)."""
    return (cands
            .join(F.broadcast(news_vec.select("news_id", "vec")), "news_id", "left")
            .join(F.broadcast(profiles), "user_id", "left")
            .withColumn("cos", cosine("u_vec", "vec"))
            .drop("vec", "u_vec"))