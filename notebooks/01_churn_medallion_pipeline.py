# Databricks notebook source
# MAGIC %md
# MAGIC # Pipeline de Análise de Churn — Arquitetura Medallion
# MAGIC **Autor:** Petrus Sampaio
# MAGIC **Objetivo:** Construir um pipeline de dados em camadas (Bronze → Silver → Gold) usando PySpark no Databricks,
# MAGIC para análise de churn de clientes, culminando em um dashboard no Data Studio.
# MAGIC
# MAGIC **Dataset:** Telco Customer Churn (IBM/Kaggle)
# MAGIC
# MAGIC **Etapas:**
# MAGIC 1. Bronze — ingestão bruta
# MAGIC 2. Silver — limpeza e padronização
# MAGIC 3. Gold — features de negócio e agregações
# MAGIC 4. Análises com Spark SQL
# MAGIC 5. Modelo preditivo simples (regressão logística)
# MAGIC 6. Exportação para Data Studio

# COMMAND ----------

# MAGIC %md
# MAGIC ## 0. Setup

# COMMAND ----------

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *

spark = SparkSession.builder.appName("churn_medallion").getOrCreate()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Fonte do dataset
# MAGIC O CSV já foi carregado como tabela no Unity Catalog via **Catalog > Add data > Create or modify table**,
# MAGIC disponível em `workspace.default.telco_customer_churn`.
# MAGIC Vamos ler essa tabela diretamente como ponto de partida da camada Bronze.

# COMMAND ----------

SOURCE_TABLE = "workspace.default.telco_customer_churn"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. BRONZE — Ingestão bruta
# MAGIC Nenhuma transformação aqui. Só lemos o dado como veio e salvamos como Delta Table,
# MAGIC preservando a origem (rastreabilidade é um princípio chave da arquitetura medallion).

# COMMAND ----------

df_bronze = spark.table(SOURCE_TABLE)

df_bronze = df_bronze.withColumn("ingestion_timestamp", F.current_timestamp())

df_bronze.write.format("delta").mode("overwrite").saveAsTable("bronze_telco_churn")

print(f"Linhas na camada Bronze: {df_bronze.count()}")
df_bronze.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. SILVER — Limpeza e padronização
# MAGIC Aqui tratamos:
# MAGIC - `TotalCharges` que vem como string (tem valores em branco para clientes novos)
# MAGIC - Padronização de colunas Yes/No para boolean
# MAGIC - Nomes de colunas em snake_case

# COMMAND ----------

df_silver = spark.table("bronze_telco_churn")

# TotalCharges / TotalRevenue podem vir com nulos, tratamos com 0 (cliente sem cobrança fechada ainda)
numeric_fix_cols = ["TotalCharges", "TotalRefunds", "TotalExtraDataCharges", "TotalLongDistanceCharges", "TotalRevenue"]
for c in numeric_fix_cols:
    if c in df_silver.columns:
        df_silver = df_silver.withColumn(c, F.col(c).cast("double"))
df_silver = df_silver.fillna({c: 0 for c in numeric_fix_cols if c in df_silver.columns})

# Padroniza colunas Yes/No para boolean (0/1), esse dataset usa ChurnLabel (Yes/No) em vez de Churn
yes_no_cols = ["Married", "Dependents", "PhoneService", "PaperlessBilling", "ChurnLabel", "ReferredaFriend"]
for c in yes_no_cols:
    if c in df_silver.columns:
        df_silver = df_silver.withColumn(c, F.when(F.col(c) == "Yes", 1).otherwise(0))

# SeniorCitizen já vem como texto Yes/No neste dataset
if "SeniorCitizen" in df_silver.columns:
    df_silver = df_silver.withColumn("SeniorCitizen", F.when(F.col("SeniorCitizen") == "Yes", 1).otherwise(0))

# Renomeia colunas para snake_case 
rename_map = {
    "CustomerID": "customer_id",
    "SeniorCitizen": "senior_citizen",
    "MonthlyCharge": "monthly_charges",
    "TotalCharges": "total_charges",
    "TotalRevenue": "total_revenue",
    "Contract": "contract_type",
    "PaymentMethod": "payment_method",
    "InternetService": "internet_service",
    "TenureinMonths": "tenure_months",
    "ChurnLabel": "churn",
    "ChurnScore": "churn_score",
    "ChurnCategory": "churn_category",
    "ChurnReason": "churn_reason",
    "CustomerStatus": "customer_status",
}
for old, new in rename_map.items():
    if old in df_silver.columns:
        df_silver = df_silver.withColumnRenamed(old, new)

df_silver.write.format("delta").mode("overwrite").saveAsTable("silver_telco_churn")

print(f"Linhas na camada Silver: {df_silver.count()}")
df_silver.select("customer_id", "tenure_months", "monthly_charges", "total_charges", "churn").show(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. GOLD — Features de negócio e agregações

# COMMAND ----------

df_gold = spark.table("silver_telco_churn")

# Faixas de tempo de contrato (tenure)
df_gold = df_gold.withColumn(
    "tenure_segment",
    F.when(F.col("tenure_months") <= 12, "0-12 meses")
     .when(F.col("tenure_months") <= 24, "13-24 meses")
     .when(F.col("tenure_months") <= 48, "25-48 meses")
     .otherwise("48+ meses")
)

# Receita mensal recorrente por segmento já está em monthly_charges (MRR por cliente)

df_gold.write.format("delta").mode("overwrite").saveAsTable("gold_telco_churn")

# Tabela agregada para o dashboard: churn por segmento
df_gold_summary = (
    df_gold.groupBy("tenure_segment", "contract_type", "internet_service")
    .agg(
        F.count("*").alias("total_clientes"),
        F.sum("churn").alias("total_churn"),
        F.round(F.avg("churn") * 100, 2).alias("taxa_churn_pct"),
        F.round(F.avg("monthly_charges"), 2).alias("mrr_medio"),
    )
    .orderBy(F.desc("taxa_churn_pct"))
)

df_gold_summary.write.format("delta").mode("overwrite").saveAsTable("gold_churn_summary")

display(df_gold_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Análises com Spark SQL
# MAGIC Aqui mostramos domínio de SQL em ambiente distribuído — importante para o portfólio.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   contract_type,
# MAGIC   COUNT(*) AS total_clientes,
# MAGIC   ROUND(AVG(churn) * 100, 2) AS taxa_churn_pct,
# MAGIC   ROUND(AVG(monthly_charges), 2) AS mrr_medio
# MAGIC FROM gold_telco_churn
# MAGIC GROUP BY contract_type
# MAGIC ORDER BY taxa_churn_pct DESC

# COMMAND ----------

# MAGIC %sql
# MAGIC -- Cohort simplificado: taxa de churn por faixa de tenure
# MAGIC SELECT
# MAGIC   tenure_segment,
# MAGIC   COUNT(*) AS total_clientes,
# MAGIC   ROUND(AVG(churn) * 100, 2) AS taxa_churn_pct
# MAGIC FROM gold_telco_churn
# MAGIC GROUP BY tenure_segment
# MAGIC ORDER BY taxa_churn_pct DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bônus: este dataset traz `churn_category` e `churn_reason` (motivo do cancelamento)
# MAGIC Isso é uma vantagem em relação à versão clássica do dataset — dá pra fazer uma análise de causa-raiz do churn,
# MAGIC que é um diferencial forte pra mostrar em entrevista.

# COMMAND ----------

# MAGIC %sql
# MAGIC SELECT
# MAGIC   churn_category,
# MAGIC   COUNT(*) AS total_clientes
# MAGIC FROM gold_telco_churn
# MAGIC WHERE churn = 1 AND churn_category IS NOT NULL
# MAGIC GROUP BY churn_category
# MAGIC ORDER BY total_clientes DESC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Modelo preditivo simples (regressão logística)
# MAGIC Usamos MLlib (nativo do Spark) para prever probabilidade de churn.

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, VectorAssembler
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

df_model = spark.table("gold_telco_churn")

categorical_cols = ["contract_type", "internet_service", "payment_method"]
numeric_cols = ["tenure_months", "monthly_charges", "total_charges", "senior_citizen"]

indexers = [StringIndexer(inputCol=c, outputCol=f"{c}_idx", handleInvalid="keep") for c in categorical_cols]
assembler = VectorAssembler(
    inputCols=[f"{c}_idx" for c in categorical_cols] + numeric_cols,
    outputCol="features"
)
lr = LogisticRegression(featuresCol="features", labelCol="churn")

pipeline = Pipeline(stages=indexers + [assembler, lr])

train, test = df_model.randomSplit([0.8, 0.2], seed=42)
model = pipeline.fit(train)
predictions = model.transform(test)

evaluator = BinaryClassificationEvaluator(labelCol="churn")
auc = evaluator.evaluate(predictions)
print(f"AUC do modelo: {auc:.4f}")

# COMMAND ----------

df_export = spark.table("gold_telco_churn")
df_summary_export = spark.table("gold_churn_summary")

# Cria um Volume no Unity Catalog para guardar os arquivos exportados
spark.sql("CREATE VOLUME IF NOT EXISTS workspace.default.churn_exports")

EXPORT_PATH = "/Volumes/workspace/default/churn_exports"

df_export.toPandas().to_csv(f"{EXPORT_PATH}/gold_telco_churn_export.csv", index=False)
df_summary_export.toPandas().to_csv(f"{EXPORT_PATH}/gold_churn_summary_export.csv", index=False)

print(f"Arquivos exportados para {EXPORT_PATH}")
print("Para baixar: vá em Catalog > workspace > default > Volumes > churn_exports, clique no arquivo e use o botão de download.")
