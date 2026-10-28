"""Combinación de señales normalizadas dentro de cada impresión."""
from pyspark.sql import functions as F, Window


def pct_rank(col, imp_col="impression_id"):
    """Rango percentil dentro de la impresión: 0 = menor, 1 = mayor."""
    return F.percent_rank().over(Window.partitionBy(imp_col).orderBy(F.col(col)))


def zscore(col, imp_col="impression_id"):
    """(x - media) / desviación dentro de la impresión; 0 si no hay variación o es nulo."""
    w = Window.partitionBy(imp_col)
    sd = F.stddev_pop(col).over(w)
    return F.when(sd > 0, (F.col(col) - F.avg(col).over(w)) / sd).otherwise(F.lit(0.0))


def weighted(col_a, col_b, w):
    """w · a + (1 - w) · b"""
    return w * F.col(col_a) + (1 - w) * F.col(col_b)