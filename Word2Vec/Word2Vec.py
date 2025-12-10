import pandas as pd
import numpy as np
import pyspark as ps
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, explode, collect_list, pandas_udf, PandasUDFType
import pyspark.sql.functions as F
from pyspark.sql.window import Window
from pyspark.sql import types as T

# Set up Spark session with optimized settings for Dataproc clusters
spark = SparkSession.builder \
    .appName("Word2VecBookRecommendation") \
    .config("spark.driver.memory", "8g") \
    .config("spark.sql.shuffle.partitions", "48") \
    .config("spark.driver.maxResultSize", "2g") \
    .config("spark.memory.fraction", "0.6") \
    .config("spark.memory.storageFraction", "0.3") \
    .config("spark.sql.adaptive.enabled", "true") \
    .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
    .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
    .getOrCreate()
spark.sparkContext.setCheckpointDir("/tmp/spark-checkpoints")

# Temporary path for breaking lineage - use GCS for Dataproc, local path for testing
# Change this to a local path when running locally
LINEAGE_BREAK_PATH = "gs://word2vec_brm/temp/lineage_break"
# LINEAGE_BREAK_PATH = "/tmp/spark-lineage-break"  # Use for local testing


def log_phase_step(phase, step, message):
    print(f"[{phase}][{step}] {message}", flush=True)


def break_lineage(df, name="temp"):
    """
    Break DataFrame lineage by writing to parquet and reading back.
    This prevents StackOverflowError from deep lineage chains.
    """
    import uuid
    temp_path = f"{LINEAGE_BREAK_PATH}/{name}_{uuid.uuid4().hex}"
    df.write.mode("overwrite").parquet(temp_path)
    result = spark.read.parquet(temp_path)
    log_phase_step("Lineage", "Break", f"Broke lineage for {name} via {temp_path}")
    return result


GRADIENT_SCHEMA = T.StructType([
    T.StructField("book_index", T.IntegerType(), nullable=False),
    T.StructField("input_gradient", T.ArrayType(T.FloatType()), nullable=True),
    T.StructField("output_gradient", T.ArrayType(T.FloatType()), nullable=True),
])

PAIR_SCHEMA = T.StructType([
    T.StructField("target_book_index", T.IntegerType(), nullable=False),
    T.StructField("context_book_index", T.IntegerType(), nullable=False),
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


def _build_sequence_pair_generator(window_size):
    win = int(window_size)

    def _generate(pdf_iter):
        for pdf in pdf_iter:
            for row in pdf.itertuples(index=False):
                indices = row.book_indices
                if indices is None:
                    continue
                length = len(indices)
                if length <= 1:
                    continue
                rows = []
                for pos, target_idx in enumerate(indices):
                    start = max(0, pos - win)
                    end = min(length, pos + win + 1)
                    for ctx_pos in range(start, end):
                        if ctx_pos == pos:
                            continue
                        rows.append((int(target_idx), int(indices[ctx_pos])))
                if rows:
                    yield pd.DataFrame(rows, columns=["target_book_index", "context_book_index"])

    return _generate


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
    """
    Apply gradient updates by collecting gradients to driver and broadcasting.
    This avoids deep lineage from joins that cause StackOverflowError.
    """
    # Collect gradients to driver (should be manageable - one row per book in vocab)
    gradient_map = {}
    for row in aggregated_gradients_df.toLocalIterator():
        idx = row["book_index"]
        if idx not in gradient_map:
            gradient_map[idx] = {"input": None, "output": None}
        if row["input_gradient"] is not None and len(row["input_gradient"]) > 0:
            gradient_map[idx]["input"] = list(row["input_gradient"])
        if row["output_gradient"] is not None and len(row["output_gradient"]) > 0:
            gradient_map[idx]["output"] = list(row["output_gradient"])
    
    log_phase_step("Phase 4", "Gradient Collection", f"Collected gradients for {len(gradient_map)} book indices")
    
    # Broadcast gradient map
    bc_gradients = spark.sparkContext.broadcast(gradient_map)
    lr = float(learning_rate)
    
    def apply_updates_partition(pdf_iter):
        import numpy as np
        grads = bc_gradients.value
        for pdf in pdf_iter:
            if pdf.empty:
                yield pdf
                continue
            
            new_input_vectors = []
            new_output_vectors = []
            
            for _, row in pdf.iterrows():
                idx = row["book_index"]
                input_vec = np.array(row["input_vector"], dtype=np.float32)
                output_vec = np.array(row["output_vector"], dtype=np.float32)
                
                if idx in grads:
                    if grads[idx]["input"] is not None:
                        input_grad = np.array(grads[idx]["input"], dtype=np.float32)
                        input_vec = input_vec - lr * input_grad
                    if grads[idx]["output"] is not None:
                        output_grad = np.array(grads[idx]["output"], dtype=np.float32)
                        output_vec = output_vec - lr * output_grad
                
                new_input_vectors.append(input_vec.tolist())
                new_output_vectors.append(output_vec.tolist())
            
            pdf = pdf.copy()
            pdf["input_vector"] = new_input_vectors
            pdf["output_vector"] = new_output_vectors
            yield pdf
    
    updated_schema = T.StructType([
        T.StructField("book_index", T.IntegerType(), nullable=False),
        T.StructField("input_vector", T.ArrayType(T.FloatType()), nullable=True),
        T.StructField("output_vector", T.ArrayType(T.FloatType()), nullable=True),
    ])
    
    updated_embeddings = embeddings_df.mapInPandas(apply_updates_partition, schema=updated_schema)
    log_phase_step("Phase 4", "Partition Updates", "Applied aggregated gradients to embeddings")
    return updated_embeddings, bc_gradients


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
    log_phase_step("Phase 4", "Training Pair Prep", "Prepared indexed sequences for training")
    return indexed_sequences_df


def generate_epoch_pairs(indexed_sequences_df, window_size, sample_fraction, pair_sample_fraction, max_pairs, epoch_seed):
    sequences_df = indexed_sequences_df
    if sample_fraction is not None and sample_fraction < 1.0:
        sequences_df = sequences_df.sample(withReplacement=False, fraction=sample_fraction, seed=epoch_seed)
    pair_generator = _build_sequence_pair_generator(window_size)
    epoch_pairs_df = sequences_df.mapInPandas(pair_generator, schema=PAIR_SCHEMA)
    if max_pairs is not None:
        if pair_sample_fraction is not None and pair_sample_fraction < 1.0:
            epoch_pairs_df = epoch_pairs_df.sample(
                withReplacement=False,
                fraction=pair_sample_fraction,
                seed=epoch_seed + 17,
            )
        epoch_pairs_df = epoch_pairs_df.limit(max_pairs)
    elif pair_sample_fraction is not None and pair_sample_fraction < 1.0:
        epoch_pairs_df = epoch_pairs_df.sample(
            withReplacement=False,
            fraction=pair_sample_fraction,
            seed=epoch_seed + 17,
        )
    return epoch_pairs_df

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
    # Break lineage early to prevent StackOverflowError
    embeddings_df = break_lineage(embeddings_df, "embeddings_init")
    embeddings_df = embeddings_df.cache()
    embeddings_df.count()
    indexed_sequences_df = prepare_training_pairs(sequences_df, window_size, vocab_df)
    indexed_sequences_df = indexed_sequences_df.cache()
    total_sequences = indexed_sequences_df.count()
    log_phase_step("Phase 4", "Sequence Cache", f"Cached {total_sequences} indexed user sequences")

    def _count_pairs(indices):
        if indices is None:
            return 0
        length = len(indices)
        if length <= 1:
            return 0
        max_offset = min(window_size, length - 1)
        return int(2 * (max_offset * length - (max_offset * (max_offset + 1) // 2)))

    pair_count_udf = F.udf(_count_pairs, T.LongType())
    total_training_pairs_row = indexed_sequences_df.select(
        pair_count_udf(F.col("book_indices")).alias("pair_count")
    ).agg(F.sum("pair_count").alias("total_pairs")).collect()[0]
    total_training_pairs = total_training_pairs_row["total_pairs"] if total_training_pairs_row else 0
    if total_training_pairs is None or total_training_pairs == 0:
        raise ValueError("No training pairs were generated from the input sequences.")
    log_phase_step(
        "Phase 4",
        "Training Pair Count",
        f"Computed {total_training_pairs} target-context pairs from indexed sequences"
    )

    if training_partitions is None:
        training_partitions = spark.sparkContext.defaultParallelism
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

    if max_pairs_per_epoch is not None:
        desired_fraction = (max_pairs_per_epoch * 1.5) / float(total_training_pairs)
        min_fraction = (1.0 / total_sequences) if total_sequences > 0 else 0.0
        sequence_sample_fraction = min(1.0, max(desired_fraction, min_fraction))
        if sequence_sample_fraction >= 1.0:
            log_phase_step(
                "Phase 4",
                "Sequence Sampling",
                "Max pairs per epoch is larger than total pair count; processing all sequences.",
            )
        expected_pairs = float(total_training_pairs) * sequence_sample_fraction
        if expected_pairs > 0:
            pair_sample_fraction = min(1.0, max_pairs_per_epoch / expected_pairs)
        else:
            pair_sample_fraction = 1.0
    else:
        sequence_sample_fraction = None
        pair_sample_fraction = None

    for epoch in range(total_epochs):
        log_phase_step("Phase 4", "Epoch", f"Starting epoch {epoch + 1}/{total_epochs}")
        epoch_lr = learning_rate * (1 - epoch / total_epochs)
        
        # Collect ALL embeddings to driver and broadcast (avoids join lineage)
        input_vector_map = {}
        output_vector_map = {}
        for row in embeddings_df.toLocalIterator():
            input_vector_map[row["book_index"]] = list(row["input_vector"])
            output_vector_map[row["book_index"]] = list(row["output_vector"])
        
        bc_input_vectors = spark.sparkContext.broadcast(input_vector_map)
        bc_output_vectors = spark.sparkContext.broadcast(output_vector_map)
        log_phase_step("Phase 4", "Epoch", f"Broadcasted {len(input_vector_map)} embeddings")
        
        lookup_negative_vectors_udf = _build_negative_lookup_udf(bc_output_vectors)

        epoch_pairs = generate_epoch_pairs(
            indexed_sequences_df,
            window_size,
            sequence_sample_fraction,
            pair_sample_fraction,
            max_pairs_per_epoch,
            epoch_seed=epoch + 1,
        )
        if sequence_sample_fraction is not None:
            log_phase_step(
                "Phase 4",
                "Epoch Sampling",
                f"Sampled sequences with fraction {sequence_sample_fraction:.6f} for epoch {epoch + 1}",
            )
        if pair_sample_fraction is not None and pair_sample_fraction < 1.0:
            log_phase_step(
                "Phase 4",
                "Pair Sampling",
                f"Applied pair sampling fraction {pair_sample_fraction:.6f} for epoch {epoch + 1}",
            )

        epoch_pairs = epoch_pairs \
            .repartition(training_partitions, "target_book_index") \
            .withColumn("pair_id", F.monotonically_increasing_id()) \
            .withColumn("rand_key", F.rand(seed=epoch + 1)) \
            .sortWithinPartitions("rand_key") \
            .drop("rand_key")

        epoch_pairs = epoch_pairs.withColumn(
            "negative_book_indices", sample_negatives_udf(F.col("pair_id"))
        ).withColumn(
            "negative_vectors",
            lookup_negative_vectors_udf(F.col("negative_book_indices"))
        )

        # Use broadcast lookups instead of joins to avoid deep lineage
        @pandas_udf("array<float>")
        def lookup_input_vector(indices):
            data = bc_input_vectors.value
            return pd.Series([data.get(idx) for idx in indices])
        
        @pandas_udf("array<float>")
        def lookup_output_vector(indices):
            data = bc_output_vectors.value
            return pd.Series([data.get(idx) for idx in indices])
        
        pairs_with_vectors = epoch_pairs \
            .withColumn("target_vector", lookup_input_vector(F.col("target_book_index"))) \
            .withColumn("context_vector", lookup_output_vector(F.col("context_book_index")))

        # CRITICAL: Break lineage BEFORE computing gradients
        # This prevents StackOverflowError when collecting gradient results
        pairs_with_vectors = break_lineage(pairs_with_vectors, f"pairs_epoch_{epoch}")

        partition_gradients = compute_partition_gradients(pairs_with_vectors)
        aggregated_gradients = partition_gradients.groupBy("book_index").agg(
            _sum_vectors_udf(F.col("input_gradient")).alias("input_gradient"),
            _sum_vectors_udf(F.col("output_gradient")).alias("output_gradient"),
        )
        
        # Break lineage again before collecting gradients
        aggregated_gradients = break_lineage(aggregated_gradients, f"gradients_epoch_{epoch}")
        
        embeddings_df, bc_gradients = apply_aggregated_updates(aggregated_gradients, embeddings_df, epoch_lr)
        # Break lineage by writing to parquet and reading back
        # This prevents StackOverflowError from deep transformation chains
        old_embeddings = embeddings_df
        embeddings_df = break_lineage(embeddings_df, f"embeddings_epoch_{epoch}")
        embeddings_df = embeddings_df.cache()
        embeddings_df.count()
        old_embeddings.unpersist()
        bc_gradients.unpersist()
        bc_input_vectors.unpersist()
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
    data_path = "gs://word2vec_brm/full_dataset/processed_artifacts/user_sequences"
    # data_path = "/Users/pouriaasadi/BigDataAnalytics/Book-Recommendation-Model/user_sequences"
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
        output_dir="gs://word2vec_brm/word2Vec_model"
        # output_dir="/Users/pouriaasadi/BigDataAnalytics/Book-Recommendation-Model/Word2Vec_model"
    )

    book_embeddings_df = extract_learned_embeddings(
        trained_embeddings_df,
        vocab_df,
        output_path= "gs://word2vec_brm/results"
        # output_path="/Users/pouriaasadi/BigDataAnalytics/Book-Recommendation-Model/results"
    )
