"""
Utility script for inspecting a trained Word2Vec model. Loads the parquet outputs,
summarizes embeddings, optionally joins human-readable book metadata, and runs
cosine-similarity lookups for specific or random book IDs.
"""

import argparse
from pathlib import Path
from typing import Iterable, List

import numpy as np
from pyspark.sql import DataFrame
import pyspark.sql.functions as F

from Word2Vec import extract_learned_embeddings, log_phase_step, spark


def load_trained_model(model_dir: str) -> tuple[DataFrame, DataFrame, DataFrame]:
    # Read the stored embeddings/vocabulary parquet files and materialize the
    # book-level embeddings DataFrame for downstream inspection.
    embeddings_path = f"{model_dir.rstrip('/')}/embeddings"
    vocab_path = f"{model_dir.rstrip('/')}/vocab"
    embeddings_df = spark.read.parquet(embeddings_path)
    vocab_df = spark.read.parquet(vocab_path)
    book_embeddings_df = extract_learned_embeddings(embeddings_df, vocab_df).cache()
    book_embeddings_df.count()
    log_phase_step("Evaluation", "Load Model", f"Loaded embeddings from {model_dir}")
    return embeddings_df, vocab_df, book_embeddings_df


def describe_embeddings(book_embeddings_df: DataFrame) -> None:
    # Print high-level stats (count, dimension, vector norms) so users can sanity-check
    # the trained embeddings before drilling into recommendations.
    total = book_embeddings_df.count()
    sample_vec = book_embeddings_df.select("input_vector").limit(1).collect()
    dim = len(sample_vec[0]["input_vector"]) if sample_vec else 0
    norm_expr = F.sqrt(
        F.expr("aggregate(transform(input_vector, x -> x * x), 0D, (acc, value) -> acc + value)")
    ).alias("norm")
    norm_stats = (
        book_embeddings_df.select(norm_expr)
        .select(
            F.min("norm").alias("min_norm"),
            F.avg("norm").alias("avg_norm"),
            F.max("norm").alias("max_norm"),
        )
        .first()
    )
    print("Embedding Summary")
    print("-----------------")
    print(f"Total books: {total}")
    print(f"Embedding dimension: {dim}")
    if norm_stats:
        print(
            f"Vector norms -> min: {norm_stats['min_norm']:.4f}, "
            f"avg: {norm_stats['avg_norm']:.4f}, max: {norm_stats['max_norm']:.4f}"
        )
    print()


def collect_local_embeddings(book_embeddings_df: DataFrame):
    # Pull all embeddings to the driver once, normalize them, and keep both an ID→index
    # map and the normalized matrix for cosine-similarity queries.
    rows = book_embeddings_df.select("book_id", "input_vector").collect()
    if not rows:
        return {}, [], np.zeros((0, 0), dtype=np.float32)

    book_ids = []
    vectors = []
    for row in rows:
        book_ids.append(row["book_id"])
        vectors.append(np.array(row["input_vector"], dtype=np.float32))
    matrix = np.vstack(vectors)
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    normalized = matrix / norms
    id_to_index = {book_id: idx for idx, book_id in enumerate(book_ids)}
    return id_to_index, book_ids, normalized


def find_similar_local(local_embeddings, book_id: str, top_k: int):
    # Compute cosine similarity purely in NumPy to avoid Spark expressions during
    # evaluation. Returns the top-k neighbors along with their similarity scores.
    id_to_index, index_to_id, matrix = local_embeddings
    if book_id not in id_to_index or matrix.size == 0:
        return []

    query_idx = id_to_index[book_id]
    query_vec = matrix[query_idx]
    similarities = matrix @ query_vec
    similarities[query_idx] = -np.inf

    available = np.sum(np.isfinite(similarities))
    k = min(top_k, available)
    if k <= 0:
        return []

    top_indices = np.argpartition(-similarities, range(k))[:k]
    top_indices = top_indices[np.argsort(similarities[top_indices])[::-1]]
    return [(index_to_id[idx], float(similarities[idx])) for idx in top_indices]


def load_metadata_lookup(metadata_path: str | None):
    # Optionally load the metadata parquet, caching it so repeated lookups stay fast.
    if not metadata_path:
        return None
    path = Path(metadata_path)
    if not path.exists():
        print(f"[Evaluation][Metadata] Path not found: {metadata_path}. Skipping metadata lookup.")
        return None
    df = spark.read.parquet(str(path)).select("book_id", "title").cache()
    df.count()
    log_phase_step("Evaluation", "Metadata", f"Loaded metadata from {metadata_path}")
    return df


def fetch_metadata_titles(metadata_df: DataFrame | None, book_ids: Iterable[str]) -> dict:
    # Pull a small subset of titles for the requested IDs. Avoids shipping the entire
    # metadata table to the driver by filtering before collecting.
    if metadata_df is None:
        return {}
    unique_ids = list({str(book_id) for book_id in book_ids if book_id is not None})
    if not unique_ids:
        return {}
    rows = metadata_df.filter(F.col("book_id").isin(unique_ids)).select("book_id", "title").collect()
    return {row["book_id"]: row["title"] for row in rows}


def format_book_label(book_id: str, metadata_map: dict) -> str:
    # Helper to show both title and ID when metadata is available.
    title = metadata_map.get(book_id)
    return f"{title} ({book_id})" if title else str(book_id)


def display_similarities(local_embeddings, book_ids: Iterable[str], top_k: int, metadata_df: DataFrame | None) -> None:
    # Main reporting loop: for each query ID we fetch metadata, run cosine
    # similarity, and print a friendly table of neighbors.
    for book_id in book_ids:
        neighbors = find_similar_local(local_embeddings, book_id, top_k=top_k)
        metadata_ids = [book_id] + [neighbor_id for neighbor_id, _ in neighbors]
        metadata_lookup = fetch_metadata_titles(metadata_df, metadata_ids)
        if metadata_df is not None and book_id not in metadata_lookup:
            # Skip printing this book entirely if no metadata match exists.
            continue

        if metadata_df is not None:
            neighbors = [
                (neighbor_id, score)
                for neighbor_id, score in neighbors
                if neighbor_id in metadata_lookup
            ]
        label = format_book_label(book_id, metadata_lookup)
        print(f"Top {top_k} similar books for {label}:")
        if not neighbors:
            if metadata_df is not None:
                print("  No metadata-backed matches found.")
            else:
                print("  No matches found.")
        else:
            for idx, (neighbor_id, score) in enumerate(neighbors, start=1):
                neighbor_label = format_book_label(neighbor_id, metadata_lookup)
                print(f"  {idx}. {neighbor_label} — similarity={score:.4f}")
        print()


def sample_random_books(book_embeddings_df: DataFrame, sample_size: int) -> List[str]:
    if sample_size <= 0:
        return []
    sampled = (
        book_embeddings_df.orderBy(F.rand())
        .limit(sample_size)
        .select("book_id")
        .collect()
    )
    return [row["book_id"] for row in sampled]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained Word2Vec model.")
    parser.add_argument(
        "--model-dir",
        default="Word2Vec_model",
        help="Directory containing embeddings/ and vocab/ parquet outputs.",
    )
    parser.add_argument(
        "--book-ids",
        nargs="*",
        default=[],
        help="Specific book IDs to inspect.",
    )
    parser.add_argument(
        "--random-samples",
        type=int,
        default=0,
        help="Number of random book IDs to sample for similarity checks.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="Number of similar books to display for each query.",
    )
    parser.add_argument(
        "--metadata-path",
        default=None,
        help="Optional path to book metadata lookup parquet directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    _, _, book_embeddings_df = load_trained_model(args.model_dir)
    describe_embeddings(book_embeddings_df)
    local_embeddings = collect_local_embeddings(book_embeddings_df)
    metadata_df = load_metadata_lookup(args.metadata_path)

    book_ids = list(args.book_ids)
    if args.random_samples > 0:
        book_ids.extend(sample_random_books(book_embeddings_df, args.random_samples))

    if book_ids:
        display_similarities(local_embeddings, book_ids, args.top_k, metadata_df)
    else:
        print("No book IDs provided; skipping similarity checks.")


if __name__ == "__main__":
    main()
