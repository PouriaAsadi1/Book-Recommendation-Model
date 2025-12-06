import pandas as pd
import numpy as np
import pyspark as ps
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, collect_list, pandas_udf, PandasUDFType
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql.column import Column
from pyspark.ml.linalg import Vectors, VectorUDT

# Set up Spark session
spark = SparkSession.builder \
    .appName("Word2VecBookRecommendation") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()


"Phase 1: Data Preparation"

# 1: Load pre-trained Word2Vec embeddings
def load_user_sequences(path):
    sequences_df = spark.read.parquet(path).select("user_id", "book_ids")
    return sequences_df

# 2: Build vocabulary by creating Spark-based mappings of book IDs to indices and vice versa
def build_vocabulary(sequences_df):
    exploded = sequences_df.select(F.explode("book_ids").alias("book_id")).distinct()
    window_spec = Window.orderBy("book_id")
    vocab_df = exploded.withColumn("book_index", F.row_number().over(window_spec) - 1)
    vocab_size = vocab_df.count()
    return vocab_df, vocab_size

# 3: Generate training pairs using Spark DataFrame operations
def generate_training_pairs(sequences_df, window_size, sequence_col="book_ids"):
    exploded = sequences_df.select(
        "user_id",
        F.posexplode(sequence_col).alias("position", "book_id")
    )
    joined = exploded.alias("left").join(
        exploded.alias("right"),
        on="user_id",
        how="inner"
    )
    pairs_df = joined.filter(
        (F.abs(F.col("left.position") - F.col("right.position")) <= window_size) &
        (F.col("left.position") != F.col("right.position"))
    ).select(
        F.col("left.book_id").alias("target_book_id"),
        F.col("right.book_id").alias("context_book_id")
    )
    return pairs_df


"Phase 2 & 3: Model Architecture and Training Components"

# Implementing forward pass by: 
# 1. looking up input embeddings for target words
# 2. computing dot products with output embedddings 
# 3. applying activation function (sigmoid) 

# 4: Initialize embeddings as Spark DataFrames with uniform random values
def initialize_embeddings(vocab_df, embedding_dim):
    limit = 0.5 / embedding_dim

    def random_vector():
        return F.array(
            *[
                (F.rand() * (2 * limit) - limit).cast("float")
                for _ in range(embedding_dim)
            ]
        )

    embeddings_df = vocab_df.select(
        "book_index",
        random_vector().alias("input_vector"),
        random_vector().alias("output_vector")
    )
    return embeddings_df

# 5: Defining the Skip-Gram with Negative Sampling (SGNS) architecture
def sigmoid(column):
    if not isinstance(column, Column):
        raise TypeError("sigmoid expects a Spark Column")
    return F.lit(1.0) / (F.lit(1.0) + F.exp(-column))

# 6 & 7: Forward pass and loss computation using Spark DataFrame columns
def skip_gram_negative_sampling(pairs_with_vectors_df):
    positive_dot = F.expr(
        "aggregate(zip_with(target_vector, context_vector, (x, y) -> x * y), 0D, (acc, value) -> acc + value)"
    )
    result_df = pairs_with_vectors_df.withColumn("positive_score", sigmoid(positive_dot))

    negative_dot = F.expr(
        "transform(negative_vectors, neg -> aggregate(zip_with(neg, target_vector, (x, y) -> x * y), 0D, (acc, value) -> acc + value))"
    )
    result_df = result_df.withColumn("negative_dot_products", negative_dot)
    result_df = result_df.withColumn(
        "negative_scores",
        F.expr("transform(negative_dot_products, value -> 1.0 / (1.0 + exp(value)))")
    )

    negative_log_sum = F.expr("aggregate(negative_scores, 0D, (acc, value) -> acc + log(value))")

    result_df = result_df.withColumn("negative_log_sum", negative_log_sum)
    result_df = result_df.withColumn("loss", -F.log(F.col("positive_score")) - F.col("negative_log_sum"))
    return result_df

# 8: Implementing backpropagation using Spark Pandas UDFs to compute gradients: 
# - computing gradients of loss with respect to embeddings
# - updating both input and output embeddings 
# - gradient for positive sample: (σ(score) - 1) * embedding
# - gradient for negative samples: σ(score) * embedding
@pandas_udf("struct<target_gradient:array<float>, context_gradient:array<float>, negative_gradients:array<array<float>>>")
def _gradient_udf(target_vectors, context_vectors, positive_scores, negative_vectors, negative_dot_products):
    import pandas as pd
    import numpy as np

    target_list = []
    context_list = []
    negative_list = []

    for target_vec, context_vec, pos_score, neg_vecs, neg_dots in zip(
        target_vectors, context_vectors, positive_scores, negative_vectors, negative_dot_products
    ):
        target_arr = np.array(target_vec, dtype=np.float32)
        context_arr = np.array(context_vec, dtype=np.float32)
        neg_arrays = [np.array(vec, dtype=np.float32) for vec in (neg_vecs or [])]

        grad_positive = pos_score - 1.0
        neg_scores = 1.0 / (1.0 + np.exp(-np.array(neg_dots, dtype=np.float32))) if neg_arrays else np.array([], dtype=np.float32)

        # Target/input gradient combines positive context and negative samples
        target_grad = grad_positive * context_arr
        if len(neg_arrays) > 0:
            stacked_neg = np.stack(neg_arrays)
            target_grad += np.sum(neg_scores[:, np.newaxis] * stacked_neg, axis=0)

        # Context/output gradient for the positive pair
        context_grad = grad_positive * target_arr

        # Negative output gradients share the target vector scaled by neg_scores
        negative_grads = [(score * target_arr).tolist() for score in neg_scores] if len(neg_arrays) > 0 else []

        target_list.append(target_grad.tolist())
        context_list.append(context_grad.tolist())
        negative_list.append(negative_grads)

    return pd.DataFrame(
        {
            "target_gradient": target_list,
            "context_gradient": context_list,
            "negative_gradients": negative_list,
        }
    )


@pandas_udf("array<float>", PandasUDFType.GROUPED_AGG)
def _sum_vectors_udf(vectors):
    import numpy as np
    arrays = [np.array(vec, dtype=np.float32) for vec in vectors if vec is not None]
    if not arrays:
        return []
    return np.sum(np.stack(arrays), axis=0).tolist()


def backpropagation(sgns_df, learning_rate):
    gradients = _gradient_udf(
        F.col("target_vector"),
        F.col("context_vector"),
        F.col("positive_score"),
        F.col("negative_vectors"),
        F.col("negative_dot_products"),
    )
    result_df = sgns_df.withColumn("gradients", gradients)
    result_df = result_df.withColumn("target_gradient", F.col("gradients.target_gradient"))
    result_df = result_df.withColumn("context_gradient", F.col("gradients.context_gradient"))
    result_df = result_df.withColumn("negative_gradients", F.col("gradients.negative_gradients"))
    result_df = result_df.drop("gradients")

    result_df = result_df.withColumn(
        "negative_update_entries",
        F.expr(
            "transform(arrays_zip(negative_book_indices, negative_gradients), "
            "pair -> struct(pair.negative_book_indices as book_index, pair.negative_gradients as gradient))"
        )
    )
    return result_df


def apply_embedding_updates(backprop_df, embeddings_df, learning_rate):
    target_updates = backprop_df.groupBy("target_book_index").agg(
        _sum_vectors_udf(F.col("target_gradient")).alias("input_gradient")
    ).withColumnRenamed("target_book_index", "book_index")

    context_updates = backprop_df.groupBy("context_book_index").agg(
        _sum_vectors_udf(F.col("context_gradient")).alias("context_gradient_sum")
    ).withColumnRenamed("context_book_index", "book_index")

    negative_updates = backprop_df.select(
        F.explode("negative_update_entries").alias("neg_entry")
    ).select(
        F.col("neg_entry.book_index").alias("book_index"),
        F.col("neg_entry.gradient").alias("gradient")
    )
    negative_updates = negative_updates.groupBy("book_index").agg(
        _sum_vectors_udf(F.col("gradient")).alias("negative_gradient_sum")
    )

    output_updates = context_updates.select(
        "book_index",
        F.col("context_gradient_sum").alias("gradient")
    ).unionByName(
        negative_updates.select(
            "book_index",
            F.col("negative_gradient_sum").alias("gradient")
        ),
        allowMissingColumns=True
    ).groupBy("book_index").agg(
        _sum_vectors_udf(F.col("gradient")).alias("output_gradient")
    )

    updated_embeddings = embeddings_df.alias("emb") \
        .join(target_updates, on="book_index", how="left") \
        .join(output_updates, on="book_index", how="left")

    updated_embeddings = updated_embeddings.withColumn(
        "input_vector",
        F.when(
            F.col("input_gradient").isNotNull(),
            F.expr(
                f"zip_with(input_vector, input_gradient, (v, g) -> cast(v - ({learning_rate}) * g as float))"
            )
        ).otherwise(F.col("input_vector"))
    )

    updated_embeddings = updated_embeddings.withColumn(
        "output_vector",
        F.when(
            F.col("output_gradient").isNotNull(),
            F.expr(
                f"zip_with(output_vector, output_gradient, (v, g) -> cast(v - ({learning_rate}) * g as float))"
            )
        ).otherwise(F.col("output_vector"))
    )

    updated_embeddings = updated_embeddings.drop("input_gradient", "output_gradient")
    return updated_embeddings


"Phase 4: Training Loop"

# Utility function to map book IDs in sequences to their corresponding indices (since training loop works with indices)
def map_sequences_to_indices(sequences_df, vocab_df):
    exploded = sequences_df.select(
        "user_id",
        F.posexplode("book_ids").alias("position", "book_id")
    )
    mapped = exploded.join(vocab_df, on="book_id", how="inner")
    indexed = mapped.orderBy("user_id", "position").groupBy("user_id").agg(
        F.collect_list("book_index").alias("book_indices")
    )
    return indexed

# Building the negative sampling distribution by counting how often each book index appears in the indexed sequences
# Raises counts to the 0.75 power and normalizes to create a probability distribution
def build_negative_sampling_distribution(indexed_sequences_df, vocab_size, power=0.75):
    counts = indexed_sequences_df.select(F.explode("book_indices").alias("book_index")).groupBy("book_index").count()
    all_books = spark.range(0, vocab_size).select(F.col("id").alias("book_index"))
    counts = all_books.join(counts, on="book_index", how="left").fillna(0, subset=["count"])
    weights = counts.select(
        "book_index",
        (F.col("count") ** power).alias("weight")
    )
    total_weight = weights.agg(F.sum("weight").alias("sum_weight"))
    probs_df = weights.crossJoin(total_weight).select(
        "book_index",
        (F.col("weight") / F.col("sum_weight")).alias("probability")
    )
    return probs_df

def prepare_training_pairs(sequences_df, window_size, vocab_df):
    indexed_sequences_df = map_sequences_to_indices(sequences_df, vocab_df)
    training_pairs_df = generate_training_pairs(indexed_sequences_df, window_size, sequence_col="book_indices").select(
        F.col("target_book_id").alias("target_book_index"),
        F.col("context_book_id").alias("context_book_index")
    )
    return indexed_sequences_df, training_pairs_df

# 9, 10, & 11: Optimization using Stochastic Gradient Descent (SGD), batch processing, and multiple epochs
def train_word2vec(
    sequences_df,
    window_size,
    embedding_dim,
    negative_samples,
    learning_rate,
    total_epochs,
):
    
    # Distributed training loop that keeps all heavy computations inside Spark.
    # Returns the final embeddings DataFrame along with the vocabulary DataFrame.
    vocab_df, vocab_size = build_vocabulary(sequences_df)
    embeddings_df = initialize_embeddings(vocab_df, embedding_dim)
    indexed_sequences_df, training_pairs_df = prepare_training_pairs(sequences_df, window_size, vocab_df)
    training_pairs_df = training_pairs_df.withColumn("pair_id", F.monotonically_increasing_id()).cache()
    neg_sampling_probs_df = build_negative_sampling_distribution(indexed_sequences_df, vocab_size).orderBy("book_index")

    probs_pd = neg_sampling_probs_df.toPandas()
    prob_array = probs_pd["probability"].to_numpy(dtype=np.float64)
    index_array = probs_pd["book_index"].to_numpy(dtype=np.int32)
    replace_flag = vocab_size <= negative_samples

    bc_prob = spark.sparkContext.broadcast(prob_array)
    bc_idx = spark.sparkContext.broadcast(index_array)

    @pandas_udf("array<int>")
    def sample_negatives_udf(_):
        rng = np.random.default_rng()
        probs = bc_prob.value
        idxs = bc_idx.value
        samples = [
            rng.choice(idxs, size=negative_samples, replace=replace_flag, p=probs).tolist()
            for _ in range(len(_))
        ]
        return pd.Series(samples)

    for epoch in range(total_epochs):
        epoch_lr = learning_rate * (1 - epoch / total_epochs)

        epoch_pairs = training_pairs_df.orderBy(F.rand()).withColumn(
            "negative_book_indices", sample_negatives_udf(F.col("pair_id"))
        )

        target_embedding_df = embeddings_df.select(
            F.col("book_index").alias("target_book_index"),
            F.col("input_vector").alias("target_vector")
        )
        context_embedding_df = embeddings_df.select(
            F.col("book_index").alias("context_book_index"),
            F.col("output_vector").alias("context_vector")
        )

        pairs_with_vectors = epoch_pairs \
            .join(target_embedding_df, on="target_book_index", how="left") \
            .join(context_embedding_df, on="context_book_index", how="left")

        negative_indices_exploded = pairs_with_vectors.select(
            "pair_id",
            F.posexplode("negative_book_indices").alias("neg_position", "negative_book_index")
        )

        negative_vectors = negative_indices_exploded.join(
            embeddings_df.select(
                F.col("book_index").alias("negative_book_index"),
                F.col("output_vector").alias("negative_vector")
            ),
            on="negative_book_index",
            how="left"
        ).groupBy("pair_id").agg(
            F.array_sort(
                F.collect_list(
                    F.struct("neg_position", "negative_book_index", "negative_vector")
                )
            ).alias("neg_structs")
        ).select(
            "pair_id",
            F.expr("transform(neg_structs, x -> x.negative_book_index)").alias("negative_book_indices"),
            F.expr("transform(neg_structs, x -> x.negative_vector)").alias("negative_vectors")
        )

        pairs_with_vectors = pairs_with_vectors.drop("negative_book_indices").join(
            negative_vectors,
            on="pair_id",
            how="left"
        )

        sgns_df = skip_gram_negative_sampling(pairs_with_vectors)
        backprop_df = backpropagation(sgns_df, epoch_lr)
        embeddings_df = apply_embedding_updates(backprop_df, embeddings_df, epoch_lr)

        avg_loss = sgns_df.agg(F.avg("loss").alias("avg_loss")).collect()[0]["avg_loss"]
        print(f"Epoch {epoch + 1}/{total_epochs} - Avg Loss: {avg_loss:.4f}")

    return embeddings_df.select("book_index", "input_vector", "output_vector"), vocab_df


"Phase 5: Evaluation & Usage"

# 12: Extract learned embeddings as a Spark DataFrame and optionally persist
def extract_learned_embeddings(embeddings_df, vocab_df, output_path=None):
    book_embeddings_df = embeddings_df.join(vocab_df, on="book_index", how="inner") \
        .select("book_index", "book_id", "input_vector")
    if output_path:
        book_embeddings_df.write.mode("overwrite").parquet(output_path)
    return book_embeddings_df

# Step 13: Similarity search using cosine similarity
def find_similar_books(embeddings_df, book_id, top_k=5):

    target_df = embeddings_df.filter(F.col("book_id") == book_id).select(
        F.col("input_vector").alias("target_vector")
    )
    if target_df.rdd.isEmpty():
        return []

    target_vector = target_df.collect()[0]["target_vector"]
    target_norm = np.linalg.norm(target_vector)

    similarity_df = embeddings_df.filter(F.col("book_id") != book_id).select(
        "book_id",
        F.expr(
            f"aggregate(zip_with(input_vector, array{target_vector}, (x, y) -> x * y), 0D, (acc, value) -> acc + value)"
        ).alias("dot_product"),
        F.expr(
            "aggregate(zip_with(input_vector, input_vector, (x, y) -> x * y), 0D, (acc, value) -> acc + value)"
        ).alias("self_dot")
    ).withColumn(
        "similarity",
        F.col("dot_product") / (F.sqrt(F.col("self_dot")) * target_norm)
    )

    top_similar = similarity_df.orderBy(F.col("similarity").desc()).limit(top_k)
    return [(row["book_id"], row["similarity"]) for row in top_similar.collect()]
