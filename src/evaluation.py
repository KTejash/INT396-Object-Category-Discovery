"""
Post-Hoc Unsupervised Model Validation and Sensitivity Analysis Module
Strictly loads ground-truth labels ONLY after clustering is complete.
Computes Adjusted Rand Index (ARI), Normalized Mutual Information (NMI),
Contingency Matrices, and comparative sensitivity across feature representations.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score
)
from sklearn.metrics.cluster import contingency_matrix
from sklearn.cluster import KMeans

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    GROUND_TRUTH_FILE,
    CLUSTERS_DIR,
    METRICS_DIR,
    EMBEDDINGS_DIR,
    DEFAULT_K,
    RANDOM_SEED
)


def evaluate_post_hoc(cluster_labels=None, k=DEFAULT_K):
    """
    Performs external validation using hidden ground truth labels.
    Never exposed during model training or parameter selection.
    """
    print("=" * 65)
    print("Performing Post-Hoc External Validation (Ground Truth Unlocked)...")

    if not GROUND_TRUTH_FILE.exists():
        raise FileNotFoundError(f"Ground truth not found at {GROUND_TRUTH_FILE}")

    df_gt = pd.read_csv(GROUND_TRUTH_FILE)

    if cluster_labels is None:
        cluster_labels = np.load(CLUSTERS_DIR / f"final_cluster_labels_k{k}.npy")

    true_categories = df_gt["category"].values

    # 1. Adjusted Rand Index (ARI)
    ari = float(adjusted_rand_score(true_categories, cluster_labels))

    # 2. Normalized Mutual Information (NMI)
    nmi = float(normalized_mutual_info_score(true_categories, cluster_labels, average_method="arithmetic"))

    # 3. Contingency Matrix
    unique_cats = sorted(list(set(true_categories)))
    unique_clusters = sorted(list(set(cluster_labels)))

    cont_mat = contingency_matrix(true_categories, cluster_labels)
    df_cont = pd.DataFrame(
        cont_mat,
        index=unique_cats,
        columns=[f"Cluster_{c}" for c in unique_clusters]
    )
    df_cont.to_csv(METRICS_DIR / "contingency_matrix.csv")

    # Normalized Contingency (Row-normalized: percentage of each true category in each cluster)
    row_sums = cont_mat.sum(axis=1, keepdims=True)
    cont_norm = np.where(row_sums > 0, cont_mat / row_sums * 100, 0)
    df_cont_norm = pd.DataFrame(
        np.round(cont_norm, 2),
        index=unique_cats,
        columns=[f"Cluster_{c}" for c in unique_clusters]
    )
    df_cont_norm.to_csv(METRICS_DIR / "contingency_matrix_normalized.csv")

    # Post-hoc validation summary table
    validation_summary = {
        "Metric": [
            "Adjusted Rand Index (ARI)",
            "Normalized Mutual Information (NMI)",
            "Cluster Count (k)",
            "Ground Truth Categories",
            "Sample Count Evaluated"
        ],
        "Value": [
            round(ari, 4),
            round(nmi, 4),
            k,
            len(unique_cats),
            len(cluster_labels)
        ],
        "Theoretical Range": [
            "[-1.0, +1.0] (0 = random chance)",
            "[0.0, 1.0] (0 = no shared info)",
            "Integer (2..12)",
            "8 Object Classes",
            f"{len(cluster_labels)} images"
        ]
    }
    df_val = pd.DataFrame(validation_summary)
    df_val.to_csv(METRICS_DIR / "post_hoc_validation_metrics.csv", index=False)

    print("\nExternal Validation Results:")
    print(df_val.to_string(index=False))

    print("\nContingency Matrix (True Categories vs Discovered Clusters):")
    print(df_cont.to_string())

    return ari, nmi, df_cont, df_cont_norm


def compare_feature_representations():
    """
    Compares clustering performance using:
      1. Deep CNN Features (ResNet50 GAP)
      2. Color Histogram Features (RGB 512-d)
    Demonstrates the difference between semantic vs color-based feature clustering.
    """
    print("=" * 65)
    print("Running Comparative Sensitivity: CNN Features vs Color Histograms...")

    df_gt = pd.read_csv(GROUND_TRUTH_FILE)
    true_categories = df_gt["category"].values

    # Load scaled feature spaces
    cnn_feats = np.load(EMBEDDINGS_DIR / "pca_95_features.npy")
    hist_scaled = np.load(EMBEDDINGS_DIR / "hist_features_scaled.npy")

    # Fit K-Means on Color Histograms at k=8
    km_hist = KMeans(n_clusters=DEFAULT_K, init="k-means++", n_init=10, random_state=RANDOM_SEED)
    hist_labels = km_hist.fit_predict(hist_scaled)

    # Load CNN labels
    cnn_labels = np.load(CLUSTERS_DIR / f"final_cluster_labels_k{DEFAULT_K}.npy")

    from sklearn.metrics import silhouette_score, davies_bouldin_score
    cnn_sil = float(silhouette_score(cnn_feats, cnn_labels))
    cnn_db = float(davies_bouldin_score(cnn_feats, cnn_labels))
    cnn_ari = float(adjusted_rand_score(true_categories, cnn_labels))
    cnn_nmi = float(normalized_mutual_info_score(true_categories, cnn_labels))

    hist_sil = float(silhouette_score(hist_scaled, hist_labels))
    hist_db = float(davies_bouldin_score(hist_scaled, hist_labels))
    hist_ari = float(adjusted_rand_score(true_categories, hist_labels))
    hist_nmi = float(normalized_mutual_info_score(true_categories, hist_labels))

    comp_data = [
        {
            "Feature Representation": "Deep CNN (ResNet50 GAP + PCA-95%)",
            "Dimensions": cnn_feats.shape[1],
            "Silhouette Score": round(cnn_sil, 4),
            "Davies-Bouldin Index": round(cnn_db, 4),
            "Adjusted Rand Index (ARI)": round(cnn_ari, 4),
            "Normalized Mutual Info (NMI)": round(cnn_nmi, 4),
            "Clustering Basis": "Semantic / Structural Shape & Deep Hierarchical Semantics"
        },
        {
            "Feature Representation": "3D Color Histograms (RGB 512-d)",
            "Dimensions": hist_scaled.shape[1],
            "Silhouette Score": round(hist_sil, 4),
            "Davies-Bouldin Index": round(hist_db, 4),
            "Adjusted Rand Index (ARI)": round(hist_ari, 4),
            "Normalized Mutual Info (NMI)": round(hist_nmi, 4),
            "Clustering Basis": "Color Palette / Illumination / Background Tones"
        }
    ]

    df_comp = pd.DataFrame(comp_data)
    df_comp.to_csv(METRICS_DIR / "feature_representation_comparison.csv", index=False)

    print("\nFeature Representation Comparison Table:")
    print(df_comp.to_string(index=False))

    return df_comp


def run_evaluation_pipeline():
    ari, nmi, df_cont, df_cont_norm = evaluate_post_hoc()
    df_comp = compare_feature_representations()
    return ari, nmi, df_cont, df_comp


if __name__ == "__main__":
    run_evaluation_pipeline()
