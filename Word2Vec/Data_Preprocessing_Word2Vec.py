from pathlib import Path
from datetime import datetime
import json
import pyspark as ps
import shutil

DATASET_ROOT = "gs://word2vec_brm/full_dataset"

# import interaction data
interaction_data_path = f"{DATASET_ROOT}/goodreads_interactions.csv"
books_metadata_path = f"{DATASET_ROOT}/goodreads_books.json"
book_authors_metadata_path = f"{DATASET_ROOT}/goodreads_book_authors.json"
book_genres_metadata_path = f"{DATASET_ROOT}/goodreads_book_genres_initial.json"
book_works_metadata_path = f"{DATASET_ROOT}/goodreads_book_works.json"
artifacts_root = f"{DATASET_ROOT}/processed_artifacts"
sequences_output_path = f"{artifacts_root}/user_sequences"
metadata_output_path = f"{artifacts_root}/book_metadata_lookup"
manifest_path = f"{artifacts_root}/manifest.json"
validation_report_path = f"{artifacts_root}/validation_report.json"

# Initialize Spark session
spark = (
    ps.sql.SparkSession.builder
    .appName("Data Preprocessing for Word2Vec")
    .config("spark.driver.memory", "8g")
    .config("spark.executor.memory", "8g")
    .getOrCreate()
)

REMOTE_URI_PREFIXES = ("gs://", "s3://", "hdfs://")


def is_remote_path(path: str) -> bool:
    return path.startswith(REMOTE_URI_PREFIXES)


def _get_fs_and_path(path: str):
    hadoop_path = spark._jvm.org.apache.hadoop.fs.Path(path)
    filesystem = spark._jvm.org.apache.hadoop.fs.FileSystem.get(
        hadoop_path.toUri(),
        spark._jsc.hadoopConfiguration()
    )
    return filesystem, hadoop_path


def delete_path_if_exists(path: str) -> None:
    if is_remote_path(path):
        filesystem, hadoop_path = _get_fs_and_path(path)
        if filesystem.exists(hadoop_path):
            filesystem.delete(hadoop_path, True)
    else:
        local_path = Path(path)
        if local_path.is_dir():
            shutil.rmtree(local_path)
        elif local_path.exists():
            local_path.unlink()


def cleanup_output_path(path: str) -> None:
    """Remove an existing directory path so Spark can overwrite cleanly."""
    delete_path_if_exists(path)


def ensure_local_directory(path: str) -> None:
    if not is_remote_path(path):
        Path(path).mkdir(parents=True, exist_ok=True)


def write_json_artifact(path: str, payload: dict) -> None:
    serialized = json.dumps(payload, indent=2)
    if is_remote_path(path):
        filesystem, hadoop_path = _get_fs_and_path(path)
        output_stream = filesystem.create(hadoop_path, True)
        try:
            output_stream.write(bytearray(serialized, "utf-8"))
        finally:
            output_stream.close()
    else:
        path_obj = Path(path)
        path_obj.parent.mkdir(parents=True, exist_ok=True)
        path_obj.write_text(serialized, encoding="utf-8")

# 1. Load interactions
# Inspect columns of the dataset
interaction_df = spark.read.csv(interaction_data_path, header=True, inferSchema=True)
interaction_df.printSchema()

# Keep only rows with is_read=1 and a positive rating
filtered_interaction_df = interaction_df.filter((interaction_df.is_read == 1) & (interaction_df.rating > 0))   

# Cast user_id and book_id to string type
filtered_interaction_df = filtered_interaction_df.withColumn("user_id", filtered_interaction_df.user_id.cast("string"))
filtered_interaction_df = filtered_interaction_df.withColumn("book_id", filtered_interaction_df.book_id.cast("string"))


# 2. Sentence builder
from pyspark.sql import functions as F

# Add a deterministic row index before grouping
# Convert DataFrame to RDD and add index
indexed_rdd = filtered_interaction_df.rdd.zipWithIndex()

# Convert back to DataFrame with index
indexed_df = indexed_rdd.map(lambda row_index: (row_index[1],) + tuple(row_index[0])).toDF(["row_index"] + filtered_interaction_df.columns)

# Group by user_id and collect book_ids in order of row_index
grouped_df = indexed_df.groupBy("user_id").agg(F.collect_list(F.struct("row_index", "book_id")).alias("book_list"))

# Sort book_list by row_index and extract book_id
sorted_df = grouped_df.withColumn("ordered_books", F.expr("transform(array_sort(book_list), x -> x.book_id)")).select("user_id", "ordered_books")

# Rename columns for clarity
user_books_df = sorted_df.select(F.col("user_id"), F.col("ordered_books").alias("book_ids"))    

# Show a sample of the resulting DataFrame
user_books_df.show(5, truncate=False)


# 3. Sentence cleanup
# Remove consecutive duplicates to avoid overweighting repeated reads
dedup_expr = """
    transform(
        filter(
            zip_with(
                book_ids,
                concat(
                    array(null),
                    slice(book_ids, 1, greatest(size(book_ids) - 1, 0))
                ),
                (curr, prev) -> struct(curr as curr, prev as prev)
            ),
            x -> x.prev IS NULL OR x.curr <> x.prev
        ),
        x -> x.curr
    )
"""

deduped_user_books_df = user_books_df.withColumn("book_ids_dedup", F.expr(dedup_expr))

# Trim histories to the most recent 200 titles
trimmed_user_books_df = deduped_user_books_df.withColumn(
    "book_ids_trimmed",
    F.expr("slice(book_ids_dedup, 1, 200)")
)

# Keep only users with at least two remaining books
filtered_length_df = trimmed_user_books_df.filter(F.size("book_ids_trimmed") >= 2)

# Compute book frequencies and retain tokens that appear at least 10 times overall
book_freq_df = (
    filtered_length_df
    .select(F.explode("book_ids_trimmed").alias("book_id"))
    .groupBy("book_id")
    .count()
    .filter(F.col("count") >= 10)
)

# Explode positions, join with valid books, and rebuild ordered arrays
tokens_df = filtered_length_df.select(
    "user_id",
    F.posexplode("book_ids_trimmed").alias("pos", "book_id")
)

tokens_with_min_count_df = tokens_df.join(
    F.broadcast(book_freq_df.select("book_id")),
    on="book_id",
    how="inner"
)

final_user_books_df = tokens_with_min_count_df.groupBy("user_id").agg(
    F.expr("transform(array_sort(collect_list(struct(pos, book_id))), x -> x.book_id)").alias("book_ids")
)

# Drop any users that lost too many books after filtering
final_user_books_df = final_user_books_df.filter(F.size("book_ids") >= 2)


# 4. Metadata join cache
surviving_book_ids_df = final_user_books_df.select(F.explode("book_ids").alias("book_id")).distinct()

# Base book attributes with series/work identifiers
books_raw_df = spark.read.json(books_metadata_path)
books_base_df = (
    books_raw_df
    .select(
        F.col("book_id").cast("string").alias("book_id"),
        F.col("title"),
        F.col("title_without_series"),
        F.col("average_rating").cast("double").alias("average_rating"),
        F.col("ratings_count").cast("long").alias("ratings_count"),
        F.col("publication_year").cast("int").alias("publication_year"),
        F.col("work_id").cast("string").alias("work_id"),
        F.col("series").alias("series_ids"),
        F.col("authors")
    )
    .join(F.broadcast(surviving_book_ids_df), on="book_id", how="inner")
)

# Attach readable author names via author lookup file
author_lookup_df = (
    spark.read.json(book_authors_metadata_path)
    .select(
        F.col("author_id").cast("string").alias("author_id"),
        F.col("name").alias("author_name")
    )
)

book_author_link_df = (
    books_base_df
    .select("book_id", F.explode_outer("authors").alias("author_struct"))
    .withColumn("author_id", F.col("author_struct.author_id").cast("string"))
    .drop("author_struct")
)

author_names_df = (
    book_author_link_df
    .join(F.broadcast(author_lookup_df), on="author_id", how="left")
    .groupBy("book_id")
    .agg(F.expr("array_remove(collect_set(author_name), NULL)").alias("author_names"))
)

# Build ordered top genres using the initial genre distribution file
genres_raw_df = spark.read.json(book_genres_metadata_path)
genres_struct_df = genres_raw_df.select(
    F.col("book_id").cast("string").alias("book_id"),
    F.col("genres")
)
genres_map_df = genres_struct_df.withColumn(
    "genres_map",
    F.from_json(F.to_json("genres"), "map<string,int>")
)
genres_exploded_df = (
    genres_map_df
    .select("book_id", F.explode("genres_map").alias("genre", "genre_count"))
    .withColumn("genre_count", F.col("genre_count").cast("int"))
    .join(F.broadcast(surviving_book_ids_df), on="book_id", how="inner")
)

top_genres_df = genres_exploded_df.groupBy("book_id").agg(
    F.expr(
        "slice(transform(array_sort(collect_list(struct(-genre_count as neg_count, genre))), x -> x.genre), 1, 5)"
    ).alias("top_genres")
)

# Original work metadata for labeling consistency
works_raw_df = spark.read.json(book_works_metadata_path)
work_ids_df = books_base_df.select("work_id").where(F.col("work_id").isNotNull()).distinct()
works_filtered_df = (
    works_raw_df
    .select(
        F.col("work_id").cast("string").alias("work_id"),
        F.col("original_title").alias("work_original_title"),
        F.col("original_publication_year").cast("int").alias("work_original_publication_year")
    )
    .join(F.broadcast(work_ids_df), on="work_id", how="inner")
)

book_metadata_lookup_df = (
    books_base_df.drop("authors")
    .join(author_names_df, on="book_id", how="left")
    .join(top_genres_df, on="book_id", how="left")
    .join(F.broadcast(works_filtered_df), on="work_id", how="left")
    .select(
        "book_id",
        "title",
        "title_without_series",
        "author_names",
        "top_genres",
        "average_rating",
        "ratings_count",
        "publication_year",
        "series_ids",
        "work_id",
        "work_original_title",
        "work_original_publication_year"
    )
    .cache()
)

book_metadata_lookup_df.show(5, truncate=False)


# 5. Persist artifacts
ensure_local_directory(artifacts_root)

cleanup_output_path(sequences_output_path)
cleanup_output_path(metadata_output_path)

# Store the cleaned user sequences for reuse
final_user_books_df.write.mode("overwrite").parquet(sequences_output_path)

# Store the hydrated metadata lookup
book_metadata_lookup_df.write.mode("overwrite").parquet(metadata_output_path)

# Persist a manifest describing the run configuration and data locations
manifest_payload = {
    "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    "interaction_source": interaction_data_path,
    "sequence_output_path": sequences_output_path,
    "metadata_output_path": metadata_output_path,
    "trim_limit": 200,
    "min_count": 10,
    "user_count": final_user_books_df.count(),
    "vocabulary_size": surviving_book_ids_df.count(),
}

delete_path_if_exists(manifest_path)
write_json_artifact(manifest_path, manifest_payload)

print(f"User sequences saved to: {sequences_output_path}")
print(f"Book metadata saved to: {metadata_output_path}")
print(f"Manifest written to: {manifest_path}")


# 6. Train-ready validation
validation_sequences_df = spark.read.parquet(sequences_output_path).cache()
validation_metadata_df = spark.read.parquet(metadata_output_path).cache()

user_count = validation_sequences_df.count()
distinct_user_count = validation_sequences_df.select("user_id").distinct().count()
duplicate_user_count = user_count - distinct_user_count

length_stats_row = (
    validation_sequences_df
    .select(F.size("book_ids").alias("sentence_length"))
    .agg(
        F.sum("sentence_length").alias("total_tokens"),
        F.avg("sentence_length").alias("avg_sentence_length"),
        F.expr("percentile_approx(sentence_length, 0.5)").alias("median_sentence_length"),
        F.max("sentence_length").alias("max_sentence_length")
    )
    .first()
)

total_tokens = int(length_stats_row.total_tokens or 0)
avg_sentence_length = float(length_stats_row.avg_sentence_length or 0.0)
median_sentence_length = int(length_stats_row.median_sentence_length or 0)
max_sentence_length = int(length_stats_row.max_sentence_length or 0)

short_sentence_count = (
    validation_sequences_df
    .filter(F.size("book_ids") < 2)
    .count()
)

vocab_df = validation_sequences_df.select(F.explode("book_ids").alias("book_id")).distinct().cache()
vocab_size = vocab_df.count()

metadata_book_ids_df = validation_metadata_df.select("book_id").distinct()
missing_tokens_df = vocab_df.join(metadata_book_ids_df, on="book_id", how="left_anti")
missing_token_count = missing_tokens_df.count()
covered_token_count = vocab_size - missing_token_count
coverage_pct = (covered_token_count / vocab_size * 100.0) if vocab_size else 0.0

validation_payload = {
    "generated_at_utc": datetime.utcnow().isoformat(timespec="seconds") + "Z",
    "user_count": user_count,
    "distinct_user_count": distinct_user_count,
    "duplicate_user_count": duplicate_user_count,
    "total_tokens": total_tokens,
    "avg_sentence_length": round(avg_sentence_length, 4),
    "median_sentence_length": median_sentence_length,
    "max_sentence_length": max_sentence_length,
    "short_sentence_count": short_sentence_count,
    "vocabulary_size": vocab_size,
    "metadata_coverage": {
        "covered_tokens": covered_token_count,
        "missing_tokens": missing_token_count,
        "coverage_pct": round(coverage_pct, 4)
    }
}

delete_path_if_exists(validation_report_path)
write_json_artifact(validation_report_path, validation_payload)

print("Validation summary:")
print(json.dumps(validation_payload, indent=2))
