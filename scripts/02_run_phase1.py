"""
scripts/02_run_phase1.py - Unsupervised cluster verification and taxonomy persistence.
"""
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config
from src.taxonomy import TAXONOMY, INTENT_NAMES

def main():
    print("=== Running Phase 1: Intent Taxonomy Discovery & Verification ===")
    retrieval_pool_file = config.PROCESSED_DATA_DIR / "retrieval_pool.jsonl"
    
    if not retrieval_pool_file.exists():
        raise FileNotFoundError(f"Missing {retrieval_pool_file}. Run scripts/01_run_phase0.py first.")

    # Load customer queries
    df = pd.read_json(retrieval_pool_file, lines=True)
    sample_df = df.head(1000).copy()
    queries = sample_df["customer_query"].tolist()
    print(f"Loaded {len(queries)} customer inquiries for cluster verification.")

    # Generate dense embeddings
    print(f"Embedding inquiries with local model '{config.EMBEDDING_MODEL_NAME}'...")
    embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    embeddings = embedder.encode(queries, show_progress_bar=True, normalize_embeddings=True)

    # Perform K-Means clustering (k=8)
    num_clusters = len(INTENT_NAMES)
    print(f"Clustering into {num_clusters} centroids...")
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(embeddings)
    sample_df["cluster"] = cluster_labels

    # Extract top keywords per cluster using TF-IDF
    tfidf = TfidfVectorizer(max_features=500, stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = tfidf.fit_transform(sample_df["customer_query"])
    feature_names = np.array(tfidf.get_feature_names_out())

    print("\n--- Empirical Data Clusters (Top TF-IDF Terms) ---")
    for cluster_id in range(num_clusters):
        cluster_docs = sample_df[sample_df["cluster"] == cluster_id]
        if len(cluster_docs) == 0:
            continue
        cluster_indices = cluster_docs.index
        mean_tfidf = np.asarray(tfidf_matrix[cluster_indices].mean(axis=0)).flatten()
        top_indices = mean_tfidf.argsort()[-6:][::-1]
        top_terms = feature_names[top_indices]
        sample_query = cluster_docs.iloc[0]["customer_query"][:90]
        print(f"Cluster {cluster_id} ({len(cluster_docs)} items) | Top terms: {', '.join(top_terms)}")
        print(f"  Sample: \"{sample_query}...\"\n")

    # Persist the formal taxonomy
    taxonomy_output_path = config.PROCESSED_DATA_DIR / "intent_taxonomy.json"
    with open(taxonomy_output_path, "w", encoding="utf-8") as f:
        json.dump(TAXONOMY, f, indent=2)

    print(f"Saved verified 8-intent taxonomy schema to: {taxonomy_output_path}")
    print("=== Phase 1 Complete ===")

if __name__ == "__main__":
    main()