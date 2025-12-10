"""
Streamlit interface for inspecting the Word2Vec book embeddings.

The app lets users type book titles, resolves them to book IDs via the metadata
lookup parquet, and reuses the evaluate_word2vec_model utilities to compute the
top-5 cosine-similar titles.
"""

from typing import Dict, Tuple

import pandas as pd
import pyspark.sql.functions as F
import streamlit as st

from evaluate_word2vec_model import (
    collect_local_embeddings,
    fetch_metadata_titles,
    find_similar_local,
    load_metadata_lookup,
    load_trained_model,
)

MODEL_DIR = "word2Vec_model_large_dataset"
METADATA_PATH = "book_metadata_lookup_large_dataset"
TOP_K = 5


@st.cache_resource(show_spinner=False)
def load_app_resources(model_dir: str, metadata_path: str | None):
    """
    Cache the heavy Spark parquet loads so the model and metadata are only read once
    per model_dir/metadata_path combination.
    """
    _, _, embeddings_df = load_trained_model(model_dir)
    local_embeddings = collect_local_embeddings(embeddings_df)
    metadata_df = load_metadata_lookup(metadata_path)
    return local_embeddings, metadata_df


def resolve_title(metadata_df, query: str) -> Tuple[str, str] | None:
    """Resolve a user-provided title/partial title to a single metadata record."""
    if metadata_df is None or not query:
        return None
    normalized = query.strip()
    if not normalized:
        return None
    lowered = normalized.lower()
    lowered_title_col = F.lower(F.col("title"))

    def _fetch(filter_expr):
        rows = (
            metadata_df.filter(filter_expr)
            .select("book_id", "title")
            .orderBy(F.length("title"))
            .limit(1)
            .collect()
        )
        if rows:
            row = rows[0]
            return row["book_id"], row["title"]
        return None

    exact_match = _fetch(lowered_title_col == F.lit(lowered))
    if exact_match:
        return exact_match

    return _fetch(lowered_title_col.contains(lowered))


def render_similarity_results(local_embeddings, metadata_df, selections: Dict[str, Tuple[str, str]]):
    """Compute and render top-k similar books for each resolved title."""
    for original_query, (book_id, resolved_title) in selections.items():
        neighbors = find_similar_local(local_embeddings, str(book_id), top_k=TOP_K)
        if not neighbors:
            st.info(f"No neighbors found for '{resolved_title}' (query: '{original_query}').")
            continue
        metadata_ids = [book_id] + [neighbor_id for neighbor_id, _ in neighbors]
        metadata_lookup = fetch_metadata_titles(metadata_df, metadata_ids)
        resolved_label = metadata_lookup.get(book_id, resolved_title)
        st.subheader(f"Top {TOP_K} similar books for {resolved_label} ({book_id})")
        rows = []
        for rank, (neighbor_id, score) in enumerate(neighbors, start=1):
            neighbor_label = metadata_lookup.get(neighbor_id, neighbor_id)
            rows.append(
                {
                    "Rank": rank,
                    "Title": neighbor_label,
                    "Book ID": neighbor_id,
                    "Cosine Similarity": f"{score:.4f}",
                }
            )
        st.table(pd.DataFrame(rows))


def main():
    st.set_page_config(page_title="Word2Vec Book Similarity", layout="centered")
    st.title("Book Similarity Explorer")
    st.caption(
        "Resolve Goodreads book titles to the trained Word2Vec space and inspect the "
        "top-5 cosine-similar recommendations."
    )

    tabs = st.tabs(["Word2Vec", "SASRec"])
    for tab in tabs:
        with tab:
            pass

    with st.spinner("Loading embeddings and metadata. This can take a minute..."):
        try:
            local_embeddings, metadata_df = load_app_resources(MODEL_DIR, METADATA_PATH)
        except Exception as exc:  # pragma: no cover - surfaces Spark errors in UI
            st.error(f"Failed to load model or metadata: {exc}")
            st.stop()

    if metadata_df is None:
        st.error("Metadata failed to load; please double-check the parquet path.")
        st.stop()

    st.markdown("### 1. Enter book titles")
    raw_titles = st.text_area(
        "Type one title per line (exact or partial matches are accepted):",
    )
    queries = [title.strip() for title in raw_titles.splitlines() if title.strip()]

    title_selections: Dict[str, Tuple[str, str]] = {}
    if queries:
        for query in queries:
            match = resolve_title(metadata_df, query)
            if match is None:
                st.warning(f"No metadata matches found for '{query}'.")
                continue
            title_selections[query] = match
    st.markdown("### 2. Retrieve similar books")
    if st.button("Find similar books"):
        if not title_selections:
            st.warning("No valid book titles selected.")
        else:
            render_similarity_results(local_embeddings, metadata_df, title_selections)


if __name__ == "__main__":
    main()
