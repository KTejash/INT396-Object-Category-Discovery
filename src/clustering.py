"""
Clustering Module
Implements K-Means experiments across k=2..12, multi-metric evaluation (Inertia,
Silhouette, Davies-Bouldin), random seed stability analysis, and cluster centroid
representative exemplar extraction.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score
from scipy.spatial.distance import cdist
import joblib

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    EMBEDDINGS_DIR,
    CLUSTERS_DIR,
    METRICS_DIR,
    MODELS_DIR,
    MANIFEST_FILE,
    K_RANGE,
    RANDOM_SEED,
    STABILITY_SEEDS,
    DEFAULT_K,
    KMEANS_BEST_MODEL_PATH
)


def evaluate_kmeans_grid(features, feature_label="PCA-95%"):
    """
    Evaluates K-Means clustering across k=2..12.
    Computes WCSS (Inertia), Silhouette Score, and Davies-Bouldin Index.
    """
    print("=" * 65)
    print(f"Evaluating K-Means Clustering on {feature_label} (k=2..12)...")

    results = []

    for k in K_RANGE:
        t0 = time.time()
        kmeans = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=RANDOM_SEED
        )
        labels = kmeans.fit_predict(features)
        elapsed = time.time() - t0

        inertia = float(kmeans.inertia_)
        sil = float(silhouette_score(features, labels))
        db = float(davies_bouldin_score(features, labels))

        # Save assignment for each k
        np.save(CLUSTERS_DIR / f"labels_{feature_label}_k{k}.npy", labels)

        results.append({
            "k": k,
            "Representation": feature_label,
            "WCSS_Inertia": round(inertia, 2),
            "Silhouette_Score": round(sil, 4),
            "Davies_Bouldin_Index": round(db, 4),
            "Fit_Time_Seconds": round(elapsed, 2)
        })
        print(f"  k={k:2d} | WCSS: {inertia:10.1f} | Silhouette: {sil:.4f} | Davies-Bouldin: {db:.4f} | Time: {elapsed:.2f}s")

    df_results = pd.DataFrame(results)
    return df_results


def compare_pca_thresholds_clustering():
    """
    Compares clustering performance on PCA-90% vs PCA-95% features.
    """
    feats_90 = np.load(EMBEDDINGS_DIR / "pca_90_features.npy")
    feats_95 = np.load(EMBEDDINGS_DIR / "pca_95_features.npy")

    df_90 = evaluate_kmeans_grid(feats_90, feature_label="PCA-90")
    df_95 = evaluate_kmeans_grid(feats_95, feature_label="PCA-95")

    df_combined = pd.concat([df_90, df_95], ignore_index=True)
    out_path = METRICS_DIR / "kmeans_metrics_comparison.csv"
    df_combined.to_csv(out_path, index=False)
    print(f"\nSaved K-Means multi-metric comparison -> {out_path}")
    return df_combined


def test_seed_stability(features=None, k=DEFAULT_K):
    """
    Evaluates sensitivity of K-Means to random initialization seeds [21, 42, 100, 123].
    Computes pairwise Adjusted Rand Index (ARI) between independent runs.
    """
    print("=" * 65)
    print(f"Testing K-Means Random Seed Stability for k={k} across seeds {STABILITY_SEEDS}...")

    if features is None:
        features = np.load(EMBEDDINGS_DIR / "pca_95_features.npy")

    run_labels = {}
    run_metrics = []

    for seed in STABILITY_SEEDS:
        km = KMeans(
            n_clusters=k,
            init="k-means++",
            n_init=10,
            max_iter=300,
            random_state=seed
        )
        lbls = km.fit_predict(features)
        run_labels[seed] = lbls
        inertia = float(km.inertia_)
        sil = float(silhouette_score(features, lbls))

        # Size distribution
        sizes = [int(np.sum(lbls == c)) for c in range(k)]
        min_size, max_size = min(sizes), max(sizes)

        run_metrics.append({
            "Random_Seed": seed,
            "WCSS_Inertia": round(inertia, 2),
            "Silhouette_Score": round(sil, 4),
            "Min_Cluster_Size": min_size,
            "Max_Cluster_Size": max_size,
            "Cluster_Sizes": str(sizes)
        })

    df_seeds = pd.DataFrame(run_metrics)
    df_seeds.to_csv(METRICS_DIR / "seed_stability_metrics.csv", index=False)

    # Compute Pairwise Adjusted Rand Index (ARI) Matrix between runs
    n_seeds = len(STABILITY_SEEDS)
    ari_matrix = np.zeros((n_seeds, n_seeds))

    for i, s1 in enumerate(STABILITY_SEEDS):
        for j, s2 in enumerate(STABILITY_SEEDS):
            ari_matrix[i, j] = adjusted_rand_score(run_labels[s1], run_labels[s2])

    df_ari = pd.DataFrame(
        np.round(ari_matrix, 4),
        index=[f"Seed_{s}" for s in STABILITY_SEEDS],
        columns=[f"Seed_{s}" for s in STABILITY_SEEDS]
    )
    df_ari.to_csv(METRICS_DIR / "seed_pairwise_ari.csv")

    print("\nRandom Seed Stability Summary:")
    print(df_seeds.to_string(index=False))
    print("\nPairwise ARI Agreement Between Random Initializations:")
    print(df_ari.to_string())

    return df_seeds, df_ari


def fit_best_model_and_extract_exemplars(features=None, k=DEFAULT_K):
    """
    Fits the definitive K-Means model on PCA-95% features at optimal k.
    Extracts 8-12 representative images closest to each cluster centroid.
    """
    print("=" * 65)
    print(f"Fitting Definitive K-Means Model at k={k} and Extracting Exemplars...")

    if features is None:
        features = np.load(EMBEDDINGS_DIR / "pca_95_features.npy")

    df_manifest = pd.read_csv(MANIFEST_FILE)
    image_ids = df_manifest["image_id"].values
    file_paths = df_manifest["file_path"].values

    kmeans = KMeans(
        n_clusters=k,
        init="k-means++",
        n_init=20,
        max_iter=300,
        random_state=RANDOM_SEED
    )
    labels = kmeans.fit_predict(features)
    centroids = kmeans.cluster_centers_

    # Save definitive model & assignments
    joblib.dump(kmeans, KMEANS_BEST_MODEL_PATH)
    np.save(MODELS_DIR / "kmeans_centroids.npy", centroids)
    np.save(CLUSTERS_DIR / f"final_cluster_labels_k{k}.npy", labels)

    df_manifest_assigned = df_manifest.copy()
    df_manifest_assigned["Cluster_ID"] = labels
    df_manifest_assigned.to_csv(CLUSTERS_DIR / "final_manifest_with_clusters.csv", index=False)

    # For each cluster, compute distance of all assigned members to its centroid
    exemplars = []

    for c in range(k):
        member_indices = np.where(labels == c)[0]
        cluster_size = len(member_indices)
        cluster_feats = features[member_indices]
        centroid = centroids[c:c+1]

        # Euclidean distances
        distances = cdist(cluster_feats, centroid, metric="euclidean").flatten()
        sorted_order = np.argsort(distances)
        top_k_indices = member_indices[sorted_order[:12]]
        top_distances = distances[sorted_order[:12]]

        for rank, (idx, dist) in enumerate(zip(top_k_indices, top_distances), start=1):
            exemplars.append({
                "Cluster_ID": c,
                "Rank": rank,
                "Image_ID": image_ids[idx],
                "File_Path": file_paths[idx],
                "Distance_To_Centroid": round(float(dist), 4),
                "Cluster_Total_Size": cluster_size
            })

    df_exemplars = pd.DataFrame(exemplars)
    df_exemplars.to_csv(METRICS_DIR / "cluster_representative_images.csv", index=False)

    # Cluster summary
    summary = []
    for c in range(k):
        members = np.where(labels == c)[0]
        summary.append({
            "Cluster_ID": c,
            "Size": len(members),
            "Percentage (%)": round((len(members) / len(labels)) * 100, 2),
            "Mean_Centroid_Distance": round(float(np.mean(cdist(features[members], centroids[c:c+1]))), 3)
        })
    df_summary = pd.DataFrame(summary)
    df_summary.to_csv(METRICS_DIR / "cluster_profiles.csv", index=False)

    print("\nCluster Profile Summary:")
    print(df_summary.to_string(index=False))
    print(f"\nSaved Representative Exemplars -> {METRICS_DIR / 'cluster_representative_images.csv'}")

    return kmeans, labels, df_exemplars, df_summary


def run_clustering_pipeline():
    df_comparison = compare_pca_thresholds_clustering()
    test_seed_stability()
    kmeans, labels, df_exemplars, df_summary = fit_best_model_and_extract_exemplars(k=DEFAULT_K)
    return df_comparison, df_exemplars


if __name__ == "__main__":
    run_clustering_pipeline()
