"""Métricas de ranking por impresión, como en MIND (Wu et al., 2020):
AUC, MRR, nDCG@5 y nDCG@10, promediadas sobre impresiones.

Implementación en Spark puro (funciones de ventana): todo se ejecuta en la JVM,
sin pasar filas a Python, por lo que escala igual a MIND-large.
"""
from pyspark.sql import functions as F, Window


def per_impression_metrics(df, score_col="score", label_col="label",
                           imp_col="impression_id", ks=(5, 10), seed=42):
    """Recibe una fila por candidata (impresión, etiqueta, puntaje) y
    devuelve una fila por impresión con sus métricas."""
    w = Window.partitionBy(imp_col)

    d = (df.select(imp_col,
                   F.col(label_col).cast("int").alias("y"),
                   F.col(score_col).cast("double").alias("s"))
           .withColumn("tb", F.rand(seed)))          # desempate aleatorio reproducible

    d = (d
         # Posición en el ranking del modelo (1 = primera), con desempate aleatorio
         .withColumn("r", F.row_number().over(w.orderBy(F.desc("s"), F.desc("tb"))))
         # Posición en el ranking ideal (clics primero), para el nDCG ideal
         .withColumn("r_ideal", F.row_number().over(w.orderBy(F.desc("y"), F.desc("tb"))))
         # Rango ascendente promediando empates (para AUC vía Mann–Whitney)
         .withColumn("r_min", F.rank().over(w.orderBy("s")))
         .withColumn("n_empate", F.count("*").over(Window.partitionBy(imp_col, "s")))
         .withColumn("r_asc", F.col("r_min") + (F.col("n_empate") - 1) / 2.0))

    aggs = [
        F.count("*").alias("n"),
        F.sum("y").alias("n_pos"),
        F.sum(F.when(F.col("y") == 1, F.col("r_asc")).otherwise(0)).alias("suma_rangos_pos"),
        F.sum(F.col("y") / F.col("r")).alias("suma_rr"),
    ]
    for k in ks:
        # Ganancia binaria: 2^y - 1 = y ; descuento log2(posición + 1)
        aggs.append(F.sum(F.when(F.col("r") <= k, F.col("y") / F.log(2.0, F.col("r") + 1))
                           .otherwise(0)).alias(f"dcg{k}"))
        aggs.append(F.sum(F.when(F.col("r_ideal") <= k, F.col("y") / F.log(2.0, F.col("r_ideal") + 1))
                           .otherwise(0)).alias(f"idcg{k}"))

    m = (d.groupBy(imp_col).agg(*aggs)
          .withColumn("n_neg", F.col("n") - F.col("n_pos")))

    valida = (F.col("n_pos") > 0) & (F.col("n_neg") > 0)
    m = (m
         .withColumn("auc", F.when(valida,
                     (F.col("suma_rangos_pos") - F.col("n_pos") * (F.col("n_pos") + 1) / 2.0)
                     / (F.col("n_pos") * F.col("n_neg"))))
         .withColumn("mrr", F.when(F.col("n_pos") > 0, F.col("suma_rr") / F.col("n_pos"))))
    for k in ks:
        m = m.withColumn(f"ndcg{k}", F.when(F.col(f"idcg{k}") > 0,
                                            F.col(f"dcg{k}") / F.col(f"idcg{k}")))

    return m.select(imp_col, "n", "n_pos", "auc", "mrr", *[f"ndcg{k}" for k in ks])


def summarize(per_imp, ks=(5, 10)) -> dict:
    """Promedio de cada métrica sobre las impresiones (como en el paper)."""
    cols = ["auc", "mrr"] + [f"ndcg{k}" for k in ks]
    r = per_imp.agg(
        F.count("*").alias("impresiones"),
        F.sum(F.col("auc").isNull().cast("int")).alias("impresiones_sin_auc"),
        *[F.mean(c).alias(c) for c in cols],
    ).first()
    return r.asDict()


def evaluate(df, score_col="score", label_col="label",
             imp_col="impression_id", ks=(5, 10), seed=42):
    """Atajo: métricas por impresión + resumen."""
    per_imp = per_impression_metrics(df, score_col, label_col, imp_col, ks, seed).cache()
    return per_imp, summarize(per_imp, ks)