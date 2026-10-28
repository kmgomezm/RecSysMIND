"""Crea la SparkSession a partir de la configuración."""
import os
import sys
from pyspark.sql import SparkSession


def get_spark(cfg: dict) -> SparkSession:
    # Python del venv para los workers (necesario en Windows)
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

    s = cfg["spark"]
    builder = (SparkSession.builder
               .appName(s["app_name"])
               .config("spark.driver.memory", s["driver_memory"])
               .config("spark.sql.shuffle.partitions", str(s["shuffle_partitions"]))
               .config("spark.sql.session.timeZone", "UTC"))
    if s.get("master"):          # en EMR no se fija: lo define YARN
        builder = builder.master(s["master"])

    spark = builder.getOrCreate()
    spark.sparkContext.setLogLevel("WARN")
    return spark