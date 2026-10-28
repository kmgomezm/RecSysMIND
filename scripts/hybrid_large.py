"""Evaluación del híbrido (CTR 24 h + embeddings MiniLM) en MIND. Dev = test.
Configuración fijada en MIND-small: CTR suavizado (h = 24, alpha = 20), puntaje z, w = 0.4.

EMR:    hybrid_large.py --bucket s3://TU-BUCKET
Local:  python scripts/hybrid_large.py --train ... --dev ... --emb ... --out ...
"""
import argparse, os, sys, time
from pyspark.sql import SparkSession, functions as F, types as T, Window

# ---------- Argumentos ----------
ap = argparse.ArgumentParser()
ap.add_argument("--bucket", help="s3://bucket con raw/train, raw/dev, emb/ y results/")
ap.add_argument("--train"); ap.add_argument("--dev"); ap.add_argument("--emb"); ap.add_argument("--out")
args = ap.parse_args()
if args.bucket:
    b = args.bucket.rstrip("/")
    args.train = args.train or f"{b}/raw/train/behaviors.tsv"
    args.dev = args.dev or f"{b}/raw/dev/behaviors.tsv"
    args.emb = args.emb or f"{b}/emb/news_emb_minilm.parquet"
    args.out = args.out or f"{b}/results"

W_HIBRIDO, H, ALPHA, DIM = 0.4, 24, 20, 384
INICIO_14NOV = "2019-11-14 00:00:00"

if os.name == "nt":                                   # solo para la prueba local en Windows
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable

builder = SparkSession.builder.appName("MIND-hibrido")
if os.name == "nt":          # prueba local en Windows; en EMR la memoria la define el clúster
    builder = (builder.master("local[*]")
                      .config("spark.driver.memory", "4g")
                      .config("spark.sql.shuffle.partitions", "32"))
spark = builder.getOrCreate()
spark.conf.set("spark.sql.session.timeZone", "UTC")
# Las tablas grandes (perfiles, embeddings) no deben copiarse enteras al driver:
spark.conf.set("spark.sql.autoBroadcastJoinThreshold", "-1")

T0 = time.time()
tiempos = []

def log(msg):
    print(f"[{time.time() - T0:7.0f} s] {msg}", flush=True)
    tiempos.append((msg, int(time.time() - T0)))

# ---------- Lectura ----------
BEH_COLS = ["impression_id", "user_id", "time", "history", "impressions"]

def read_beh(path, split):
    schema = T.StructType([T.StructField(c, T.StringType()) for c in BEH_COLS])
    return (spark.read.option("sep", "\t").option("header", False).option("quote", "\u0000")
                 .schema(schema).csv(path)
                 .withColumn("time", F.to_timestamp("time", "M/d/yyyy h:mm:ss a"))
                 .withColumn("impression_id", F.col("impression_id").cast("int"))
                 .withColumn("split", F.lit(split)))

beh_tr = read_beh(args.train, "train")
beh_dev = read_beh(args.dev, "dev")

# p0: CTR global de train, calculado sin explotar las impresiones
cands_txt = F.split("impressions", " ")
p0 = beh_tr.agg((F.sum(F.size(F.filter(cands_txt, lambda c: c.endswith("-1"))))
                 / F.sum(F.size(cands_txt))).alias("p0")).first()["p0"]
log(f"p0 = {p0:.4f}")

# ---------- Señal 1: CTR reciente en streaming ----------
def explode(beh):
    return (beh.select("impression_id", "user_id", "time", "split",
                       F.explode(F.split("impressions", " ")).alias("item"))
               .withColumn("news_id", F.split("item", "-")[0])
               .withColumn("label", F.split("item", "-")[1].cast("int"))
               .drop("item"))

base = explode(beh_tr.filter(F.col("time") >= F.to_timestamp(F.lit(INICIO_14NOV)))
               .unionByName(beh_dev))
w_ctr = Window.partitionBy("news_id").orderBy("ts").rangeBetween(-H * 3600, -1)
cands = (base.withColumn("ts", F.unix_timestamp("time"))
             .withColumn("clics", F.coalesce(F.sum("label").over(w_ctr), F.lit(0)))
             .withColumn("expos", F.count("label").over(w_ctr))
             .filter("split = 'dev'")
             .withColumn("ctr", (F.col("clics") + ALPHA * p0) / (F.col("expos") + ALPHA))
             .select("impression_id", "user_id", "news_id", "label", "ctr"))

# ---------- Señal 2: perfil = promedio de embeddings del historial (SQL puro) ----------
emb = spark.read.parquet(args.emb)
users = (beh_dev.select("user_id", "history").dropDuplicates(["user_id"])
                .filter(F.col("history").isNotNull()))
suma = (f"aggregate(embs, array_repeat(cast(0.0 as double), {DIM}), "
        f"(acc, x) -> zip_with(acc, x, (a, b) -> a + b))")
prof = (users.select("user_id", F.explode(F.split("history", " ")).alias("news_id"))
             .join(emb, "news_id")
             .groupBy("user_id").agg(F.collect_list("emb").alias("embs"))
             .withColumn("s", F.expr(suma))
             .withColumn("norma", F.expr("sqrt(aggregate(transform(s, x -> x * x), 0D, (a, x) -> a + x))"))
             .withColumn("u_emb", F.expr("transform(s, x -> x / norma)"))
             .select("user_id", "u_emb"))

# ---------- Híbrido ----------
dot = "aggregate(zip_with(u_emb, emb, (x, y) -> x * y), 0D, (a, z) -> a + z)"
wi = Window.partitionBy("impression_id")

def zscore(c):
    sd = F.stddev_pop(c).over(wi)
    return F.when(sd > 0, (F.col(c) - F.avg(c).over(wi)) / sd).otherwise(F.lit(0.0))

senales = (cands.join(emb, "news_id", "left").join(prof, "user_id", "left")
                .withColumn("cos", F.when(F.col("u_emb").isNotNull() & F.col("emb").isNotNull(),
                                          F.expr(dot)))
                .withColumn("emb_resp", F.coalesce("cos", "ctr"))
                .withColumn("hibrido", W_HIBRIDO * zscore("cos") + (1 - W_HIBRIDO) * zscore("ctr"))
                .withColumn("aleatorio", F.rand(1))
                .select("impression_id", "label", "ctr", "emb_resp", "hibrido", "aleatorio")
                .cache())
log(f"Candidatas de dev: {senales.count():,}")

# ---------- Métricas por impresión (misma lógica que src/evaluation/metrics.py) ----------
def per_impression(df, p, ks=(5, 10), seed=42):
    w = Window.partitionBy("impression_id")
    d = (df.select("impression_id", F.col("label").alias("y"), F.col(p).cast("double").alias("s"))
           .withColumn("tb", F.rand(seed))
           .withColumn("r", F.row_number().over(w.orderBy(F.desc("s"), F.desc("tb"))))
           .withColumn("r_ideal", F.row_number().over(w.orderBy(F.desc("y"), F.desc("tb"))))
           .withColumn("r_min", F.rank().over(w.orderBy("s")))
           .withColumn("n_emp", F.count("*").over(Window.partitionBy("impression_id", "s")))
           .withColumn("r_asc", F.col("r_min") + (F.col("n_emp") - 1) / 2.0))
    aggs = [F.count("*").alias("n"), F.sum("y").alias("n_pos"),
            F.sum(F.when(F.col("y") == 1, F.col("r_asc")).otherwise(0)).alias("srp"),
            F.sum(F.col("y") / F.col("r")).alias("srr")]
    for k in ks:
        aggs += [F.sum(F.when(F.col("r") <= k, F.col("y") / F.log(2.0, F.col("r") + 1)).otherwise(0)).alias(f"dcg{k}"),
                 F.sum(F.when(F.col("r_ideal") <= k, F.col("y") / F.log(2.0, F.col("r_ideal") + 1)).otherwise(0)).alias(f"idcg{k}")]
    m = d.groupBy("impression_id").agg(*aggs).withColumn("n_neg", F.col("n") - F.col("n_pos"))
    m = (m.withColumn(f"auc_{p}", F.when((F.col("n_pos") > 0) & (F.col("n_neg") > 0),
                      (F.col("srp") - F.col("n_pos") * (F.col("n_pos") + 1) / 2.0) / (F.col("n_pos") * F.col("n_neg"))))
          .withColumn(f"mrr_{p}", F.when(F.col("n_pos") > 0, F.col("srr") / F.col("n_pos"))))
    for k in ks:
        m = m.withColumn(f"ndcg{k}_{p}", F.when(F.col(f"idcg{k}") > 0, F.col(f"dcg{k}") / F.col(f"idcg{k}")))
    return m.select("impression_id", *[c for c in m.columns if c.endswith(f"_{p}")])

modelos = ["aleatorio", "ctr", "emb_resp", "hibrido"]
por_imp = None
for mo in modelos:
    pm = per_impression(senales, mo)
    por_imp = pm if por_imp is None else por_imp.join(pm, "impression_id")
por_imp = por_imp.cache()
log(f"Impresiones evaluadas: {por_imp.count():,}")

# ---------- Resultados ----------
METS = ("auc", "mrr", "ndcg5", "ndcg10")
res = por_imp.agg(*[F.avg(f"{m}_{mo}").alias(f"{m}_{mo}") for mo in modelos for m in METS]).first()
filas = [(mo, *[float(res[f"{m}_{mo}"]) for m in METS]) for mo in modelos]
for f in filas:
    log(f"{f[0]:10s} AUC = {f[1]:.4f} | MRR = {f[2]:.4f} | nDCG@5 = {f[3]:.4f} | nDCG@10 = {f[4]:.4f}")

(spark.createDataFrame(filas, ["modelo", *METS])
      .coalesce(1).write.mode("overwrite").option("header", True).csv(f"{args.out}/resumen"))
por_imp.coalesce(1).write.mode("overwrite").parquet(f"{args.out}/por_impresion")
log("Resultados guardados")
(spark.createDataFrame(tiempos, ["etapa", "segundos"])
      .coalesce(1).write.mode("overwrite").option("header", True).csv(f"{args.out}/tiempos"))
spark.stop()