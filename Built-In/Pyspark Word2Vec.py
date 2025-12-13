# -*- coding: utf-8 -*-
"""
Created on Sat Dec 22 00:35:58 2025

@author: PRINCELY OSEJI
"""

from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.ml.feature import Word2Vec
from pyspark.ml.linalg import Vectors, VectorUDT
import numpy as np

spark = SparkSession.builder.appName("Goodreads_Word2Vec_Recs").getOrCreate()

# Load datasets
# Interactions: user_id, book_id, is_read, rating, is_reviewed
interactions = spark.read.parquet("user_book_interactions.parquet")
# Metadata: book_id, title, authors, genres, description, average_rating
metadata = spark.read.parquet("book_metadata.parquet")  

interactions = (
    interactions
    .select(
        F.col("user_id").cast("string").alias("user_id"),
        F.col("book_id").cast("string").alias("book_id"),
        F.col("rating").cast("double").alias("rating")
    )
    .dropna(subset=["user_id", "book_id", "rating"])
)

metadata = (
    metadata
    .select(
        F.col("book_id").cast("string").alias("book_id"),
        F.col("title").alias("title"),
        F.col("authors").alias("authors"),
        F.col("genres").alias("genres"),
        F.col("average_rating").alias("average_rating")
    )
)

# Keep "positive" interactions (so sequences represent preferences)
# TO DO: Tune threshold 
POS_RATING = 3.0
pos = interactions.filter(F.col("rating") >= F.lit(POS_RATING))

#  downsample extreme power users or filter very short histories
pos = pos.dropDuplicates(["user_id", "book_id"])

# Build user "sentences" (book sequences)
# No timestamp in data so can't model order.
# Generate stable pseudo-order via random shuffle per user for training.
seed = 42
w_rand = Window.partitionBy("user_id").orderBy(F.rand(seed))

sentences = (
    pos
    .withColumn("rn", F.row_number().over(w_rand))
    .groupBy("user_id")
    .agg(F.collect_list("book_id").alias("sentence"))
    .filter(F.size("sentence") >= 2)
)

# Model training
w2v = Word2Vec(
    vectorSize=100,
    windowSize=5,
    minCount=5,     # for huge corpora, minCount>1 helps
    maxIter=10,
    inputCol="sentence",
    outputCol="sentence_vec"
)

w2v_model = w2v.fit(sentences)

item_vecs = w2v_model.getVectors().withColumnRenamed("word", "book_id")
# item_vecs: book_id, vector


# 20% holdout evaluation setup (per user)
w_rand2 = Window.partitionBy("user_id").orderBy(F.rand(seed + 1))
w_user = Window.partitionBy("user_id")

split_df = (
    pos
    .withColumn("rn", F.row_number().over(w_rand2))
    .withColumn("cnt", F.count("*").over(w_user))
    .withColumn("holdout_n", F.greatest(F.lit(1), F.ceil(F.col("cnt") * F.lit(0.20)).cast("int")))
    .withColumn("is_holdout", F.col("rn") <= F.col("holdout_n"))
)

train_pos = split_df.filter("NOT is_holdout").select("user_id", "book_id")
holdout = split_df.filter("is_holdout").select("user_id", "book_id")

ground_truth = holdout.groupBy("user_id").agg(F.collect_set("book_id").alias("gt_books"))
seen_train = train_pos.groupBy("user_id").agg(F.collect_set("book_id").alias("seen_books"))

#
# User profile vectors = mean of training book vectors
user_book_vecs = (
    train_pos
    .join(item_vecs, on="book_id", how="inner")
)

@F.udf(returnType=VectorUDT())
def avg_vectors(vs):
    arr = np.array([v.toArray() for v in vs], dtype=np.float32)
    if arr.size == 0:
        return Vectors.dense([])
    return Vectors.dense(arr.mean(axis=0))

user_vecs = (
    user_book_vecs
    .groupBy("user_id")
    .agg(F.collect_list("vector").alias("vec_list"))
    .withColumn("user_vector", avg_vectors("vec_list"))
    .select("user_id", "user_vector")
)


# Recommend Top-K by cosine similarity (baseline cross-join)
@F.udf("double")
def cosine_sim(v1, v2):
    a = v1.toArray()
    b = v2.toArray()
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)

K = 10

# For 2.3M books, broadcasting vectors is unrealistic
# This baseline is for smaller subsets
scores = (
    user_vecs.crossJoin(item_vecs.select("book_id", F.col("vector").alias("book_vector")))
    .withColumn("score", cosine_sim("user_vector", "book_vector"))
    .select("user_id", "book_id", "score")
)

# Remove seen training books
seen_pairs = train_pos.dropDuplicates(["user_id", "book_id"])
scores_unseen = scores.join(seen_pairs, ["user_id", "book_id"], "left_anti")

w_rank = Window.partitionBy("user_id").orderBy(F.col("score").desc())
recs = (
    scores_unseen
    .withColumn("rank", F.row_number().over(w_rank))
    .filter(F.col("rank") <= K)
    .groupBy("user_id")
    .agg(F.collect_list("book_id").alias("recommended_books"))
)

# attach metadata for display (top-10 list exploded)
recs_pretty = (
    recs
    .select("user_id", F.posexplode("recommended_books").alias("pos", "book_id"))
    .join(metadata, on="book_id", how="left")
    .orderBy("user_id", "pos")
)

recs_pretty.show(50, truncate=False)

# Evaluate Precision@10 and Recall@10

eval_df = (
    ground_truth.join(recs, on="user_id", how="inner")
    .withColumn("hits", F.size(F.array_intersect("gt_books", "recommended_books")))
    .withColumn("precision_at_10", F.col("hits") / F.lit(10.0))
    .withColumn("recall_at_10", F.col("hits") / F.greatest(F.lit(1.0), F.size("gt_books").cast("double")))
)

metrics = eval_df.agg(
    F.avg("precision_at_10").alias("Precision@10"),
    F.avg("recall_at_10").alias("Recall@10"),
    F.count("*").alias("NumEvaluatedUsers")
)

metrics.show(truncate=False)
