import pandas as pd
import numpy as np
import pyspark as ps
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, collect_list, pandas_udf, PandasUDFType
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql import types as T

# Set up Spark session
spark = SparkSession.builder \
    .appName("Word2VecBookRecommendation") \
    .config("spark.driver.memory", "8g") \
    .config("spark.sql.shuffle.partitions", "200") \
    .getOrCreate()
spark.sparkContext.setCheckpointDir("/tmp/spark-checkpoints")


def log_phase_step(phase, step, message):
    print(f"[{phase}][{step}] {message}", flush=True)


GRADIENT_SCHEMA = T.StructType([
    T.StructField("book_index", T.IntegerType(), nullable=False),
    T.StructField("input_gradient", T.ArrayType(T.FloatType()), nullable=True),
    T.StructField("output_gradient", T.ArrayType(T.FloatType()), nullable=True),
])


"Phase 1: Data Preparation"

# Step 1: Load pre-trained Word2Vec embeddings
def load_user_sequences(path):
    sequences_df = spark.read.parquet(path).select("user_id", "book_ids")
    log_phase_step("Phase 1", "Step 1", f"Loaded user sequences from {path}")
    return sequences_df

# Step 2: Build vocabulary by creating Spark-based mappings of book IDs to indices and vice versa
def build_vocabulary(sequences_df):
    exploded = sequences_df.select(F.explode("book_ids").alias("book_id")).distinct()
    window_spec = Window.orderBy("book_id")
    vocab_df = exploded.withColumn("book_index", F.row_number().over(window_spec) - 1)
    vocab_df = vocab_df.cache()
    vocab_size = vocab_df.count()
    log_phase_step("Phase 1", "Step 2", f"Built vocabulary with {vocab_size} unique book IDs")
    return vocab_df, vocab_size

# Step 3: Generate training pairs using Spark DataFrame operations
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
    log_phase_step("Phase 1", "Step 3", f"Generated training pairs using window size {window_size}")
    return pairs_df


"Phase 2 & 3: Model Architecture and Training Components"

# Implementing forward pass by: 
# 1. looking up input embeddings for target words
# 2. computing dot products with output embedddings 
# 3. applying activation function (sigmoid) 

# Step 4: Initialize embeddings as Spark DataFrames with uniform random values
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
    log_phase_step("Phase 2/3", "Step 4", f"Initialized embeddings with dimension {embedding_dim}")
    return embeddings_df

def save_trained_model(embeddings_df, vocab_df, output_dir):
    embeddings_path = f"{output_dir.rstrip('/')}/embeddings"
    vocab_path = f"{output_dir.rstrip('/')}/vocab"
    embeddings_df.write.mode("overwrite").parquet(embeddings_path)
    vocab_df.write.mode("overwrite").parquet(vocab_path)
    log_phase_step("Phase 2/3", "Model Persistence", f"Saved embeddings and vocabulary to {output_dir}")

@pandas_udf("array<float>", PandasUDFType.GROUPED_AGG)
def _sum_vectors_udf(vectors):
    import numpy as np
    arrays = [np.array(vec, dtype=np.float32) for vec in vectors if vec is not None]
    if not arrays:
        return []
    return np.sum(np.stack(arrays), axis=0).tolist()


def _collect_output_vectors(embeddings_df):
    vector_map = {}
    for row in embeddings_df.select("book_index", "output_vector").toLocalIterator():
        vector_map[row["book_index"]] = list(row["output_vector"])
    return vector_map


def _build_negative_lookup_udf(output_vector_broadcast):
    @pandas_udf("array<array<float>>")
    def _lookup(negative_indices_series):
        data = output_vector_broadcast.value
        results = []
        for indices in negative_indices_series:
            if indices is None:
                results.append([])
                continue
            vectors = []
            for idx in indices:
                vec = data.get(idx)
                vectors.append(vec if vec is None else list(vec))
            results.append(vectors)
        return pd.Series(results)

    return _lookup


def compute_partition_gradients(pairs_with_vectors_df):
    def _partition_iterator(pdf_iter):
        import numpy as np
        import pandas as pd

        for pdf in pdf_iter:
            if pdf.empty:
                continue

            input_acc = {}
            output_acc = {}

            for row in pdf.itertuples(index=False):
                if row.target_vector is None or row.context_vector is None:
                    continue

                target_vec = np.array(row.target_vector, dtype=np.float32)
                context_vec = np.array(row.context_vector, dtype=np.float32)

                pos_dot = float(np.dot(target_vec, context_vec))
                pos_score = 1.0 / (1.0 + np.exp(-pos_dot))
                grad_positive = pos_score - 1.0

                if row.negative_book_indices is None:
                    neg_indices = []
                else:
                    neg_indices = list(row.negative_book_indices)

                if row.negative_vectors is None:
                    neg_vectors = []
                else:
                    neg_vectors = list(row.negative_vectors)

                neg_pairs = [
                    (idx, vec)
                    for idx, vec in zip(neg_indices, neg_vectors)
                    if vec is not None
                ]

                if neg_pairs:
                    neg_vec_arrays = [np.array(vec, dtype=np.float32) for _, vec in neg_pairs]
                    neg_dots = np.array(
                        [float(np.dot(target_vec, neg_vec)) for neg_vec in neg_vec_arrays],
                        dtype=np.float32,
                    )
                    neg_scores = 1.0 / (1.0 + np.exp(-neg_dots))
                else:
                    neg_vec_arrays = []
                    neg_scores = np.array([], dtype=np.float32)

                target_grad = grad_positive * context_vec
                if neg_vec_arrays:
                    stacked_neg = np.stack(neg_vec_arrays)
                    target_grad += np.sum(neg_scores[:, np.newaxis] * stacked_neg, axis=0)

                context_grad = grad_positive * target_vec

                input_acc.setdefault(row.target_book_index, np.zeros_like(target_vec))
                input_acc[row.target_book_index] += target_grad

                output_acc.setdefault(row.context_book_index, np.zeros_like(target_vec))
                output_acc[row.context_book_index] += context_grad

                for (neg_idx, _), neg_score in zip(neg_pairs, neg_scores):
                    output_acc.setdefault(neg_idx, np.zeros_like(target_vec))
                    output_acc[neg_idx] += neg_score * target_vec

            if not input_acc and not output_acc:
                continue

            rows = []
            for idx, grad in input_acc.items():
                rows.append((int(idx), grad.tolist(), None))
            for idx, grad in output_acc.items():
                rows.append((int(idx), None, grad.tolist()))

            yield pd.DataFrame(rows, columns=["book_index", "input_gradient", "output_gradient"])

    return pairs_with_vectors_df.mapInPandas(_partition_iterator, schema=GRADIENT_SCHEMA)


def apply_aggregated_updates(aggregated_gradients_df, embeddings_df, learning_rate):
    updated_embeddings = embeddings_df.alias("emb").join(
        aggregated_gradients_df.alias("grad"), on="book_index", how="left"
    )

    updated_embeddings = updated_embeddings.withColumn(
        "input_vector",
        F.when(
            (F.col("input_gradient").isNotNull()) & (F.size("input_gradient") > 0),
            F.expr(
                f"zip_with(input_vector, input_gradient, (v, g) -> cast(v - ({learning_rate}) * g as float))"
            ),
        ).otherwise(F.col("input_vector")),
    )

    updated_embeddings = updated_embeddings.withColumn(
        "output_vector",
        F.when(
            (F.col("output_gradient").isNotNull()) & (F.size("output_gradient") > 0),
            F.expr(
                f"zip_with(output_vector, output_gradient, (v, g) -> cast(v - ({learning_rate}) * g as float))"
            ),
        ).otherwise(F.col("output_vector")),
    )

    updated_embeddings = updated_embeddings.drop("input_gradient", "output_gradient")
    log_phase_step("Phase 4", "Partition Updates", "Applied aggregated gradients to embeddings")
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
    log_phase_step("Phase 4", "Sequence Mapping", "Mapped user sequences to vocabulary indices")
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
    ).orderBy("book_index").cache()
    log_phase_step("Phase 4", "Negative Sampling", f"Built negative sampling distribution for vocab size {vocab_size}")
    return probs_df

def build_alias_table(probabilities):
    n = len(probabilities)
    prob = np.zeros(n, dtype=np.float64)
    alias = np.zeros(n, dtype=np.int32)
    scaled = probabilities * n
    small = []
    large = []

    for idx, val in enumerate(scaled):
        if val < 1.0:
            small.append(idx)
        else:
            large.append(idx)

    while small and large:
        s = small.pop()
        l = large.pop()
        prob[s] = scaled[s]
        alias[s] = l
        scaled[l] = scaled[l] - (1.0 - scaled[s])
        if scaled[l] < 1.0:
            small.append(l)
        else:
            large.append(l)

    for remaining in large + small:
        prob[remaining] = 1.0
        alias[remaining] = remaining

    log_phase_step("Phase 4", "Alias Table", f"Built alias table for {n} probabilities")
    return prob, alias

def prepare_training_pairs(sequences_df, window_size, vocab_df):
    indexed_sequences_df = map_sequences_to_indices(sequences_df, vocab_df)
    training_pairs_df = generate_training_pairs(indexed_sequences_df, window_size, sequence_col="book_indices").select(
        F.col("target_book_id").alias("target_book_index"),
        F.col("context_book_id").alias("context_book_index")
    )
    log_phase_step("Phase 4", "Training Pair Prep", "Prepared indexed sequences and training pairs")
    return indexed_sequences_df, training_pairs_df

# Step 9, 10, & 11: Optimization using Stochastic Gradient Descent (SGD), batch processing, and multiple epochs
def train_word2vec(
    sequences_df,
    window_size,
    embedding_dim,
    negative_samples,
    learning_rate,
    total_epochs,
    max_pairs_per_epoch=None,
    training_partitions=None,
):
    
    # Distributed training loop that keeps all heavy computations inside Spark.
    # Returns the final embeddings DataFrame along with the vocabulary DataFrame.
    vocab_df, vocab_size = build_vocabulary(sequences_df)
    embeddings_df = initialize_embeddings(vocab_df, embedding_dim)
    embeddings_df = embeddings_df.checkpoint(eager=True)
    embeddings_df.count()
    indexed_sequences_df, training_pairs_df = prepare_training_pairs(sequences_df, window_size, vocab_df)
    if training_partitions is None:
        training_partitions = spark.sparkContext.defaultParallelism
    training_pairs_df = training_pairs_df \
        .repartition(training_partitions, "target_book_index") \
        .withColumn("pair_id", F.monotonically_increasing_id()) \
        .cache()
    log_phase_step(
        "Phase 4",
        "Partitioning",
        f"Repartitioned training pairs into {training_partitions} partitions keyed by target_book_index",
    )
    neg_sampling_probs_df = build_negative_sampling_distribution(indexed_sequences_df, vocab_size)
    index_list = []
    prob_list = []
    for row in neg_sampling_probs_df.select("book_index", "probability").toLocalIterator():
        index_list.append(row["book_index"])
        prob_list.append(row["probability"])
    index_array = np.array(index_list, dtype=np.int32)
    probability_array = np.array(prob_list, dtype=np.float64)
    alias_prob, alias_idx = build_alias_table(probability_array)

    bc_indices = spark.sparkContext.broadcast(index_array)
    bc_alias_prob = spark.sparkContext.broadcast(alias_prob)
    bc_alias_idx = spark.sparkContext.broadcast(alias_idx)
    log_phase_step("Phase 4", "Training Loop", "Initialized negative sampling broadcasts")

    @pandas_udf("array<int>")
    def sample_negatives_udf(_):
        rng = np.random.default_rng()
        idxs = bc_indices.value
        prob = bc_alias_prob.value
        alias = bc_alias_idx.value
        vocab_len = len(idxs)
        samples = []
        for _ in range(len(_)):
            k = rng.integers(0, vocab_len, size=negative_samples)
            accept = rng.random(negative_samples)
            final = np.where(accept < prob[k], k, alias[k])
            samples.append(idxs[final].tolist())
        return pd.Series(samples)

    for epoch in range(total_epochs):
        log_phase_step("Phase 4", "Epoch", f"Starting epoch {epoch + 1}/{total_epochs}")
        epoch_lr = learning_rate * (1 - epoch / total_epochs)
        output_vector_map = _collect_output_vectors(embeddings_df)
        bc_output_vectors = spark.sparkContext.broadcast(output_vector_map)
        lookup_negative_vectors_udf = _build_negative_lookup_udf(bc_output_vectors)

        epoch_pairs = training_pairs_df.withColumn("rand_key", F.rand(seed=epoch + 1)) \
            .sortWithinPartitions("rand_key") \
            .drop("rand_key")
        if max_pairs_per_epoch is not None:
            epoch_pairs = epoch_pairs.limit(max_pairs_per_epoch)

        epoch_pairs = epoch_pairs.withColumn(
            "negative_book_indices", sample_negatives_udf(F.col("pair_id"))
        ).withColumn(
            "negative_vectors",
            lookup_negative_vectors_udf(F.col("negative_book_indices"))
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

        partition_gradients = compute_partition_gradients(pairs_with_vectors)
        aggregated_gradients = partition_gradients.groupBy("book_index").agg(
            _sum_vectors_udf(F.col("input_gradient")).alias("input_gradient"),
            _sum_vectors_udf(F.col("output_gradient")).alias("output_gradient"),
        )
        embeddings_df = apply_aggregated_updates(aggregated_gradients, embeddings_df, epoch_lr)
        embeddings_df = embeddings_df.checkpoint(eager=True)
        embeddings_df.count()
        bc_output_vectors.unpersist()
        log_phase_step("Phase 4", "Epoch", f"Finished epoch {epoch + 1}/{total_epochs}")

    log_phase_step("Phase 4", "Training Loop", "Completed training loop")
    return embeddings_df.select("book_index", "input_vector", "output_vector"), vocab_df


"Phase 5: Evaluation & Usage"

# Step 12: Extract learned embeddings as a Spark DataFrame and optionally persist
def extract_learned_embeddings(embeddings_df, vocab_df, output_path=None):
    book_embeddings_df = embeddings_df.join(vocab_df, on="book_index", how="inner") \
        .select("book_index", "book_id", "input_vector")
    if output_path:
        book_embeddings_df.write.mode("overwrite").parquet(output_path)
    message = "Extracted learned embeddings"
    if output_path:
        message += f" and saved to {output_path}"
    log_phase_step("Phase 5", "Step 12", message)
    return book_embeddings_df

# Step 13: Similarity search using cosine similarity
def find_similar_books(embeddings_df, book_id, top_k=5):

    target_df = embeddings_df.filter(F.col("book_id") == book_id).select(
        F.col("input_vector").alias("target_vector")
    )
    if target_df.rdd.isEmpty():
        log_phase_step("Phase 5", "Step 13", f"No embeddings found for book_id={book_id}")
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
    results = [(row["book_id"], row["similarity"]) for row in top_similar.collect()]
    log_phase_step("Phase 5", "Step 13", f"Found top {len(results)} similar books for book_id={book_id}")
    return results

# Example usage: 
if __name__ == "__main__":
    # data_path = #"gs://word2vec_brm/user_sequences"
    data_path = "/Users/pouriaasadi/BigDataAnalytics/Book-Recommendation-Model/user_sequences"
    sequences_df = load_user_sequences(data_path)

    trained_embeddings_df, vocab_df = train_word2vec(
        sequences_df,
        window_size=5,
        embedding_dim=100,
        negative_samples=5,
        learning_rate=0.025,
        total_epochs=10,
        max_pairs_per_epoch=50000,
    )
    save_trained_model(
        trained_embeddings_df,
        vocab_df,
        # output_dir="gs://word2vec_brm/word2Vec_model"
        output_dir="/Users/pouriaasadi/BigDataAnalytics/Book-Recommendation-Model/Word2Vec_model"
    )

    book_embeddings_df = extract_learned_embeddings(
        trained_embeddings_df,
        vocab_df,
        # output_path= "gs://word2vec_brm/results"
        output_path="/Users/pouriaasadi/BigDataAnalytics/Book-Recommendation-Model/results"
    )
