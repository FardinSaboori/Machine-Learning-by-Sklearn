# Databricks notebook source
# MAGIC %md
# MAGIC # K-means Clustering Lab
# MAGIC 
# MAGIC **Objective**: *Apply K-means clustering to a dataset to learn more about how the records are related to one another.*

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC we will create a user-level table with the following columns:
# MAGIC 
# MAGIC 1. `avg_resting_heartrate` – the average resting heartrate
# MAGIC 1. `avg_active_heartrate` - the average active heartrate
# MAGIC 1. `avg_bmi` – the average BMI
# MAGIC 1. `avg_vo2` - the average oxygen volume
# MAGIC 1. `sum_workout_minutes` - the sum of total workout minutes
# MAGIC 1. `sum_steps` - the sum of total steps

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from adsda.ht_daily_metrics

# COMMAND ----------

# MAGIC %sql
# MAGIC drop database FARDIN

# COMMAND ----------

# MAGIC %sql
# MAGIC -- TODO
# MAGIC CREATE OR REPLACE TABLE fardin.ht_user_metrics_lab
# MAGIC USING DELTA LOCATION "/fardin/ht-user-metrics-lab" AS (
# MAGIC   SELECT avg(resting_heartrate) AS avg_resting_heartrate,
# MAGIC          avg(active_heartrate) AS avg_active_heartrate,
# MAGIC          avg(bmi) AS avg_bmi,
# MAGIC          avg(vo2) AS avg_vo2,
# MAGIC          sum(workout_minutes) AS sum_workout_minutes,
# MAGIC          sum(steps) AS sum_steps
# MAGIC   FROM adsda.ht_daily_metrics
# MAGIC   GROUP BY device_id
# MAGIC )

# COMMAND ----------

df = spark.table("adsda.ht_user_metrics_lab").toPandas()
df.shape

# COMMAND ----------

# MAGIC %md-sandbox
# MAGIC 
# MAGIC we will split the data into an training set and an inference set.

# COMMAND ----------

# TODO
from sklearn.model_selection import train_test_split

train_df, inference_df = train_test_split(df, train_size=0.85, test_size=0.15, random_state=42)

# COMMAND ----------

print(f"{train_df.shape[0]} and {inference_df.shape[0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC we will identify the optimal number of clusters for K-means using the training set.

# COMMAND ----------

# TODO
from sklearn.cluster import KMeans
from sklearn.preprocessing import scale

distortions = []
values_of_k = range(2, 16)

for k in values_of_k:
  k_means = KMeans(n_clusters=k, max_iter=500)
  k_means.fit(scale(train_df))
  distortion = k_means.score(scale(train_df))
  distortions.append(-distortion)

# COMMAND ----------

list(zip(distortions, values_of_k))

# COMMAND ----------

import matplotlib.pyplot as plt

plt.plot(values_of_k, distortions, 'bx-') 
plt.xlabel('Values of K') 
plt.ylabel('Distortion') 
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC 
# MAGIC retrain the model with the optimal number of clusters.

# COMMAND ----------

# TODO
k_means = KMeans(n_clusters=6, max_iter=500, random_state=1234)
k_means.fit(scale(train_df))

# COMMAND ----------

# TODO
k_means.cluster_centers_

# COMMAND ----------

# TODO
inference_df_clusters = k_means.predict(scale(inference_df))
clusters_df = inference_df.copy()
clusters_df["cluster"] = inference_df_clusters

# COMMAND ----------

clusters_df

# COMMAND ----------

clusters_df["cluster"].value_counts()

# COMMAND ----------

clusters_df['cluster'].value_counts()

# COMMAND ----------

clusters_df.groupby(["cluster"])[["sum_steps"]].mean()

# COMMAND ----------

clusters_df.groupby(["cluster"])[['sum_steps']].mean()

# COMMAND ----------

cluster_sparkdf = spark.createDataFrame(clusters_df)

# COMMAND ----------

cluster_sparkdf.write.format("delta").mode("overwrite").save("/adsda/cluster_sparkdf")

# COMMAND ----------

# MAGIC %sql
# MAGIC drop table if exists adsda.cluster_sparkdf;
# MAGIC create table if not exists adsda.cluster_sparkdf
# MAGIC using delta location "/adsda/cluster_sparkdf"

# COMMAND ----------

# MAGIC %sql
# MAGIC select * from adsda.cluster_sparkdf

# COMMAND ----------

# MAGIC %sql
# MAGIC select cluster,int(avg(sum_steps)) from adsda.cluster_sparkdf group by cluster
