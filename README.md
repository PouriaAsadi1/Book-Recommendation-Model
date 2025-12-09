# Book Recommendation System

[![Project](https://img.shields.io/badge/project-book--recommender-blue.svg)](https://github.com/PouriaAsadi1/Book-Recommendation-Model)

A scalable book recommendation system built on the Goodreads dataset. This repository contains preprocessing utilities, two recommendation model approaches (SASRec and Word2Vec-style item embeddings), evaluation scripts, and examples for training and inference.

Authors: Princely Oseji, Pouria Asadi, Tianchi Wu

---

## Table of contents

- [About](#about)
- [Features](#features)
- [Environment setup](#environment-setup)
- [Dataset & preprocessing details](#dataset--preprocessing-details)
- [How to run the code](#how-to-run-the-code)
- [Results & discussion](#results--discussion)
- [Evaluation metrics](#evaluation-metrics)
- [Acknowledgements & References](#acknowledgements--references)

---

## About

The goal of this project is to learn latent representations that capture complex user preferences and book characteristics by leveraging large-scale Goodreads user–book interactions and book metadata. Two complementary approaches are implemented:

- SASRec — a Transformer-based sequential recommendation model for next-item prediction.
- Word2Vec-style item embeddings — learns item-to-item similarity from interaction sequences for similarity-based retrieval and discovery.

This system focuses on reproducible offline evaluation (Precision@K, Recall@K) and is designed to be extensible to hybrid models and online evaluation.

---

## Features

- Data preprocessing pipeline for the UCSD Goodreads dataset
- SASRec implementation (sequential/temporal modeling)
- Word2Vec-style item embedding training and ANN retrieval support
- Offline evaluation scripts (Precision@K, Recall@K)
- Example CLI-style scripts for preprocessing, training, evaluation, and inference
- Configurable hyperparameters and dataset filters

---

## Environment setup

These are the exact versions and dependencies used for the latest experiments.

| Dependency | Version | Notes |
| --- | --- | --- |
| Python | 3.11.8 | `python --version` |
| Java | 21.0.9 | Required for Spark (OpenJDK via Homebrew) |
| Apache Spark / PySpark | 3.5.6 | Installed through `pip install pyspark==3.5.6` |
| Pandas | 2.1.4 | Used in pandas UDFs |
| NumPy | 1.26.4 | Used for gradient math and evaluation |

### Local environment steps

```bash
python -m venv .venv
source .venv/bin/activate
pip install pyspark==3.5.6 pandas==2.1.4 numpy==1.26.4 pyarrow==15.0.2
```

Additional requirements:
- A JDK (11+) on `PATH` so that PySpark can spin up a JVM.
- Access to the [UCSD Goodreads dataset](https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home) stored locally or in a bucket (GCS/S3/HDFS). Update the `DATASET_ROOT` constant in `Data_Preprocessing_Word2Vec.py` to point at your copy.
- Adequate disk and memory. Processing the trimmed dataset still touches ~755k user histories and ~330k book tokens, so we recommend at least `spark.driver.memory=8g`.

---

## Dataset & preprocessing details

We use the UCSD Goodreads dataset (public, large-scale). It contains:
- User–book interactions (~4.1 GB): `user_id`, `book_id`, `is_read`, `rating`, `is_reviewed` (we use `user_id`, `book_id`, `rating`, `is_read`).
- Book metadata (~2.3 GB, ~2.3M books): `book_id`, `title`, `authors`, `genres`, `description`, `average_rating`, publication data.

Source / reference:
- UCSD Goodreads dataset: https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home

NOTE: The raw dataset is large; consider filtering or sampling for local experiments.

### Variables and preprocessing workflow

The `Data_Preprocessing_Word2Vec.py` Spark job (configurable via the `DATASET_ROOT` constant) performs the following:

1. **Interaction filtering** – loads `goodreads_interactions.csv`, keeps rows with `is_read == 1` and `rating > 0`, casts `user_id`/`book_id` to strings, and preserves the chronological order provided in the CSV.
2. **Sequence construction** – groups rows by user, sorts by the original row index, and collects ordered `book_id` sentences. Consecutive duplicates are removed so repeated re-reads do not overweight a title.
3. **Trimming & minimum counts** – each user history is truncated to the most recent 200 titles, and only titles with global frequency ≥10 are retained. Users with fewer than 2 valid titles are dropped.
4. **Metadata hydration** – surviving book IDs are joined with `goodreads_books.json`, `goodreads_book_authors.json`, `goodreads_book_genres_initial.json`, and `goodreads_book_works.json` to assemble titles, author arrays, top-5 genres (sorted by counts), ratings, publication year, and work/series identifiers.
5. **Artifact persistence & validation** – writes user sequences to `user_sequences/`, metadata to `book_metadata_lookup/`, and emits run manifests plus validation stats (user counts, token coverage) so training jobs can be reproduced.

### Processed dataset characteristics

Using the included parquet outputs (`user_sequences/` and `book_metadata_lookup/`), we observe:

- 755,311 user reading sequences (after filtering and trimming).
- Sequence lengths range from 2 to 200 titles with an average of 51.0 and a median of 29 titles.
- Approximate unique vocabulary size of ~344k book IDs after filtering; the trained model embeds 331,434 of them (see results below).
- Metadata lookup rows include: `title`, `title_without_series`, `author_names[]`, `top_genres[]`, `average_rating`, `ratings_count`, `publication_year`, `series_ids`, and work-level fields for labeling downstream recommendations.

All artifacts are saved as parquet so Spark-based training/evaluation scripts can load them efficiently.

---

## How to run the code

The repository ships CLI-style scripts for the Word2Vec workflow. Update the paths inside each script (e.g., `DATASET_ROOT` in `Data_Preprocessing_Word2Vec.py`, `data_path`/`output_dir` in `Word2Vec.py`) to point at your environment.

1. **Preprocess Goodreads interactions and metadata**

   ```bash
   python Data_Preprocessing_Word2Vec.py
   ```

   Outputs (configurable in the script):
   - `user_sequences/` parquet folder with cleaned user sentences.
   - `book_metadata_lookup/` parquet folder containing curated metadata.
   - `processed_artifacts/manifest.json` and `processed_artifacts/validation_report.json` describing the run and dataset stats.

2. **Train the distributed Word2Vec model**

   ```bash
   # Optional: edit Word2Vec.py to change data paths or hyperparameters.
   python Word2Vec.py
   ```

   Default hyperparameters: window size 5, embedding dimension 100, 5 negative samples, learning rate 0.025 with 10 epochs, and checkpointed Spark DataFrames. Artifacts:
   - `Word2Vec_model/embeddings/` – learned input/output vectors keyed by `book_index`.
   - `Word2Vec_model/vocab/` – `book_id` ↔ `book_index` mapping.
   - `results/` – optional book-level embeddings for downstream ANN retrieval.

3. **Evaluate or inspect the trained model**

   ```bash
   python evaluate_word2vec_model.py \
     --model-dir Word2Vec_model \
     --metadata-path book_metadata_lookup \
     --book-ids 132538 426023 \
     --random-samples 3 \
     --top-k 5
   ```

   The script prints embedding statistics, joins metadata when available, and reports cosine-similarity neighbors for provided or random book IDs. Use `--random-samples 0` to skip sampling and only inspect explicit IDs.

SASRec experiments follow the same philosophy: reuse the trimmed `user_sequences/`, configure hyperparameters under `sasrec-model/`, and log Precision@K/Recall@K with the included evaluation helpers.

---

## Results & discussion

We validate every run by executing the evaluation script against the saved parquet artifacts. Running:

```bash
python evaluate_word2vec_model.py --model-dir Word2Vec_model \
  --metadata-path book_metadata_lookup --random-samples 3 --top-k 5
```

produced the following key outputs on the provided data:

```
Embedding Summary
-----------------
Total books: 331434
Embedding dimension: 100
Vector norms -> min: 0.0222, avg: 0.0289, max: 2.3510

Top 5 similar books for 132538:
  1. L'Histoire de Pi (1230) — similarity=0.4424
  2. 22965 — similarity=0.4341
  3. 819303 — similarity=0.4173
  4. 273550 — similarity=0.4143
  5. 141728 — similarity=0.4123
```

Additional sampled IDs such as `426023` and `223384` return fantasy and classic-literature neighbors, respectively, with cosine scores ≈0.39–0.44. Interpretation:
- The embedding summary confirms the model captured 331,434 distinct titles with stable vector norms (no gradient explosion).
- Top neighbors cluster translations or installments of the same work (e.g., *Life of Pi* variants) as well as frequently co-read classics, showing that trimming + deduplication prevented noisy transitions.
- Metadata joins allow downstream services to replace integer book IDs with human-readable titles when presenting recommendations.

These metrics serve as offline sanity checks before deploying ANN indexes or SASRec-based next-item predictors. Precision@K/Recall@K evaluation can be layered on top using the saved embeddings and interaction holdouts.

---

## Evaluation metrics

This project focuses on standard offline accuracy metrics:
- Precision@K — fraction of recommended items in top-K that are relevant.
- Recall@K — fraction of relevant items captured in top-K recommendations.

Other useful metrics to consider (future work): NDCG@K, MAP, diversity, novelty, calibration, and online/business metrics (CTR, retention).

---

## Acknowledgements & References

- UCSD Goodreads dataset — https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home
- Kang, W.-C., & McAuley, J. (2018). Self-Attentive Sequential Recommendation.
- Chamberlain, B.P., Rossi, E., Shiebler, D., Sedhain, S., Bronstein, M.M. (2020). Tuning Word2vec for Large Scale Recommendation Systems.
- Additional references available in repository documentation and report.

---
