from __future__ import annotations

from typing import Iterable, List, Sequence, Tuple

import numpy as np
import pandas as pd
from pyspark.sql import DataFrame, SparkSession, Window
from pyspark.sql import functions as F, types as T


_SPARK_SESSION: SparkSession | None = None
_PAIR_SCHEMA = T.StructType(
    [
        T.StructField("target", T.IntegerType(), False),
        T.StructField("context", T.IntegerType(), False),
    ]
)
_SEQUENCE_SCHEMA = T.StructType(\n    [T.StructField("sequence", T.ArrayType(T.IntegerType(), containsNull=False), False)]
)
_STRING_SEQUENCE_SCHEMA = T.StructType(
    [T.StructField("sequence", T.ArrayType(T.StringType(), containsNull=False), False)]
)
_EMBEDDING_SCHEMA = T.StructType(
    [
        T.StructField("word", T.StringType(), False),
        T.StructField("vector", T.ArrayType(T.DoubleType(), containsNull=False), False),
    ]
)


def get_spark_session(app_name: str = "Word2VecSpark") -> SparkSession:
    """Create or return a cached Spark session."""
    global _SPARK_SESSION
    if _SPARK_SESSION is None:
        _SPARK_SESSION = SparkSession.builder.appName(app_name).getOrCreate()
    return _SPARK_SESSION


def _ensure_sequence_dataframe(
    spark: SparkSession, sequences, schema: T.StructType
) -> DataFrame:
    """Return a DataFrame with a single 'sequence' column built from Python iterables."""
    if isinstance(sequences, DataFrame):
        if "sequence" not in sequences.columns:
            raise ValueError("Sequence DataFrame must include a 'sequence' column.")
        return sequences.select("sequence")

    rows = [(list(sequence),) for sequence in sequences]
    return spark.createDataFrame(rows, schema=schema)


def _ensure_pairs_dataframe(spark: SparkSession, training_pairs) -> DataFrame:
    if isinstance(training_pairs, DataFrame):
        if {"target", "context"}.issubset(set(training_pairs.columns)):
            return training_pairs.select("target", "context")
        raise ValueError("Training pair DataFrame must contain 'target' and 'context'.")

    rows = [(int(target), int(context)) for target, context in training_pairs]
    return spark.createDataFrame(rows, schema=_PAIR_SCHEMA)


def _compute_negative_sampling_distribution(training_pairs_df: DataFrame, vocab_size: int) -> np.ndarray:
    spark = training_pairs_df.sparkSession
    union_df = training_pairs_df.select(F.col("target").alias("word")).unionAll(
        training_pairs_df.select(F.col("context").alias("word"))
    )
    counts_df = union_df.groupBy("word").agg(F.count(F.lit(1)).alias("count"))
    counts = np.ones(vocab_size, dtype=np.float64)
    for row in counts_df.collect():
        counts[row.word] = float(row["count"])
    powered = counts ** 0.75
    total = powered.sum()
    if total == 0:
        return np.full(vocab_size, 1.0 / vocab_size, dtype=np.float64)
    return powered / total


"Phase 1: Data Preparation"

# Step 1: Load pre-trained Word2Vec embeddings
def load_word2vec_embeddings(embedding_path: str, spark: SparkSession | None = None) -> dict:
    spark = spark or get_spark_session()
    lines_df = spark.read.text(embedding_path)

    def _parse_embeddings(iterator: Iterable[pd.DataFrame]):
        for pdf in iterator:
            words: List[str] = []
            vectors: List[List[float]] = []
            for value in pdf["value"]:
                tokens = value.strip().split()
                if not tokens:
                    continue
                words.append(tokens[0])
                vectors.append([float(x) for x in tokens[1:]])
            yield pd.DataFrame({"word": words, "vector": vectors})

    embeddings_df = lines_df.mapInPandas(_parse_embeddings, schema=_EMBEDDING_SCHEMA)
    return {row.word: np.asarray(row.vector, dtype="float32") for row in embeddings_df.collect()}


# Step 2: Build vocabulary by creating a mapping of book IDs to indices and vice versa, 
# and calculating vocabulary size for embedded matrix dimension
def build_vocabulary(book_ids: Sequence[str], spark: SparkSession | None = None):
    spark = spark or get_spark_session()
    book_df = spark.createDataFrame(
        [(book_id,) for book_id in book_ids],
        schema=T.StructType([T.StructField("book_id", T.StringType(), False)]),
    )
    window = Window.orderBy("book_id")
    indexed_df = (
        book_df.dropDuplicates(["book_id"])
        .orderBy("book_id")
        .withColumn("idx", F.row_number().over(window) - 1)
    )
    vocab_entries = indexed_df.collect()
    book_to_index = {row.book_id: row.idx for row in vocab_entries}
    index_to_book = {row.idx: row.book_id for row in vocab_entries}
    vocab_size = len(book_to_index)
    return book_to_index, index_to_book, vocab_size


# Step 3: Generating training pairs 
# using a sliding window approach to create (target, context) pairs
def generate_training_pairs(
    sequences, window_size: int, spark: SparkSession | None = None
) -> DataFrame:
    spark = spark or get_spark_session()
    sequences_df = _ensure_sequence_dataframe(spark, sequences, _SEQUENCE_SCHEMA)

    def _pairs_from_sequences(iterator: Iterable[pd.DataFrame]):
        for pdf in iterator:
            rows: list[tuple[int, int]] = []
            for sequence in pdf["sequence"]:
                if sequence is None:
                    continue
                seq = list(sequence)
                seq_length = len(seq)
                for i, target in enumerate(seq):
                    start = max(0, i - window_size)
                    end = min(seq_length, i + window_size + 1)
                    for j in range(start, end):
                        if j == i:
                            continue
                        rows.append((int(seq[i]), int(seq[j])))
            if rows:
                yield pd.DataFrame(rows, columns=["target", "context"])
            else:
                yield pd.DataFrame(columns=["target", "context"])

    return sequences_df.mapInPandas(_pairs_from_sequences, schema=_PAIR_SCHEMA)


"Phase 2 & 3: Model Architecture and Training Components"

# Implementing forward pass by: 
# 1. looking up input embeddings for target words
# 2. computing dot products with output embedddings 
# 3. applying activation function (sigmoid) 

# Step 4: Initalizing embeddings by creating an input embedding matrix and an output embedding matrix
def initialize_embeddings(vocab_size: int, embedding_dim: int) -> Tuple[np.ndarray, np.ndarray]:
    limit = 0.5 / embedding_dim
    input_embeddings = np.random.uniform(-limit, limit, (vocab_size, embedding_dim))
    output_embeddings = np.random.uniform(-limit, limit, (vocab_size, embedding_dim))
    return input_embeddings, output_embeddings


# Step 5: Defining the Skip-Gram with Negative Sampling (SGNS) architecture
def sigmoid(x):
    return 1 / (1 + np.exp(-x))


# Step 6 & 7: Forward pass and loss computation
# using Skip-Gram with Negative Sampling (SGNS) architecture
def skip_gram_negative_sampling(
    target_idx: int,
    context_idx: int,
    input_embeddings: np.ndarray,
    output_embeddings: np.ndarray,
    negative_indices: np.ndarray,
):
    target_vector = input_embeddings[target_idx]
    context_vector = output_embeddings[context_idx]
    
    positive_score = sigmoid(np.dot(target_vector, context_vector))
    
    negative_vectors = output_embeddings[negative_indices]
    
    negative_scores = sigmoid(-np.dot(negative_vectors, target_vector))
    
    loss = -np.log(positive_score) - np.sum(np.log(negative_scores))
    
    return loss


# Step 8: Implementing backpropagation by: 
# - computing gradients of loss with respect to embeddings
# - updating both input and output embeddings 
# - gradient for positive sample: (σ(score) - 1) * embedding
# - gradient for negative samples: σ(score) * embedding
def backpropagation(
    target_idx: int,
    context_idx: int,
    input_embeddings: np.ndarray,
    output_embeddings: np.ndarray,
    negative_indices: np.ndarray,
    learning_rate: float,
):
    target_vector = input_embeddings[target_idx]
    context_vector = output_embeddings[context_idx]
    
    positive_score = sigmoid(np.dot(target_vector, context_vector))
    grad_positive = (positive_score - 1)
    
    negative_vectors = output_embeddings[negative_indices]
    
    negative_scores = sigmoid(np.dot(negative_vectors, target_vector))
    
    # Update output embeddings
    output_embeddings[context_idx] -= learning_rate * grad_positive * target_vector
    for i, neg_idx in enumerate(negative_indices):
        output_embeddings[neg_idx] -= learning_rate * negative_scores[i] * target_vector
    
    # Update input embeddings
    input_embeddings[target_idx] -= learning_rate * (
        grad_positive * context_vector + np.sum(negative_scores[:, np.newaxis] * negative_vectors, axis=0)
    )
    
    return input_embeddings, output_embeddings


"Phase 4: Training Loop"

# Utility function to map book IDs in sequences to their corresponding indices (since training loop works with indices)
def map_sequences_to_indices(
    sequences, book_to_index: dict, spark: SparkSession | None = None
) -> DataFrame:
    spark = spark or get_spark_session()
    sequences_df = _ensure_sequence_dataframe(spark, sequences, _STRING_SEQUENCE_SCHEMA)
    broadcast_mapping = spark.sparkContext.broadcast(book_to_index)

    @F.udf(returnType=T.ArrayType(T.IntegerType()))
    def map_sequence(sequence: List[str]):
        if sequence is None:
            return []
        mapping = broadcast_mapping.value
        return [mapping[book_id] for book_id in sequence if book_id in mapping]

    indexed_sequences = sequences_df.select(map_sequence(F.col("sequence")).alias("sequence")).filter(
        F.size("sequence") > 1
    )
    return indexed_sequences


def prepare_training_pairs(
    sequences,
    window_size: int,
    book_to_index: dict,
    spark: SparkSession | None = None,
) -> DataFrame:
    spark = spark or get_spark_session()
    indexed_sequences_df = map_sequences_to_indices(sequences, book_to_index, spark)
    return generate_training_pairs(indexed_sequences_df, window_size, spark)


# Step 9 & 10: Optimization using Stochastic Gradient Descent (SGD), batch processing, and multiple epochs
def train_word2vec(
    training_pairs,
    vocab_size: int,
    embedding_dim: int,
    negative_samples: int,
    learning_rate: float,
    total_epochs: int,
    batch_size: int,
    spark: SparkSession | None = None,
    compute_loss: bool = False,
) -> Tuple[np.ndarray, np.ndarray]:
    spark = spark or get_spark_session()

    training_pairs_df = _ensure_pairs_dataframe(spark, training_pairs).select("target", "context").cache()
    total_pairs = training_pairs_df.count()
    if total_pairs == 0:
        raise ValueError("Training pairs DataFrame is empty.")

    negative_probabilities = _compute_negative_sampling_distribution(training_pairs_df, vocab_size)
    input_embeddings, output_embeddings = initialize_embeddings(vocab_size, embedding_dim)
    initial_learning_rate = learning_rate
    effective_batch_size = max(1, batch_size)

    for epoch in range(total_epochs):
        current_lr = initial_learning_rate * (1 - epoch / total_epochs)
        shuffled_pairs_df = training_pairs_df.withColumn("rand", F.rand()).orderBy("rand").drop("rand")
        epoch_loss = 0.0
        batch_rows: List = []

        def process_batch(rows: List):
            nonlocal epoch_loss, input_embeddings, output_embeddings
            for row in rows:
                target_idx = int(row.target)
                context_idx = int(row.context)
                negative_indices = np.random.choice(
                    vocab_size,
                    negative_samples,
                    replace=(vocab_size <= negative_samples),
                    p=negative_probabilities,
                )
                if compute_loss:
                    loss = skip_gram_negative_sampling(
                        target_idx, context_idx, input_embeddings, output_embeddings, negative_indices
                    )
                    epoch_loss += loss

                input_embeddings, output_embeddings = backpropagation(
                    target_idx, context_idx, input_embeddings, output_embeddings, negative_indices, current_lr
                )

        for row in shuffled_pairs_df.toLocalIterator():
            batch_rows.append(row)
            if len(batch_rows) == effective_batch_size:
                process_batch(batch_rows)
                batch_rows = []

        if batch_rows:
            process_batch(batch_rows)

        if compute_loss:
            avg_loss = epoch_loss / total_pairs
            print(f'Epoch {epoch+1}/{total_epochs}, Avg Loss: {avg_loss:.4f}')
        else:
            print(f'Epoch {epoch+1}/{total_epochs} completed.')

    return input_embeddings, output_embeddings
