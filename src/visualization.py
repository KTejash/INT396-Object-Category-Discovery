"""
Visualization Module
Generates high-resolution publication-quality figures for PCA, t-SNE,
K-Means elbow/silhouette curves, cluster galleries, and post-hoc contingency heatmaps.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    FIGURES_DIR,
    METRICS_DIR,
    EMBEDDINGS_DIR,
    CLUSTERS_DIR,
    GROUND_TRUTH_FILE,
    MANIFEST_FILE,
    TSNE_PERPLEXITIES,
    TSNE_LEARNING_RATES,
    DEFAULT_K
)

# Set global publication styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 15,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight"
})

PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]


def plot_data_quality():
    """Generates figure for dataset distribution and dimension checks."""
    dims_file = METRICS_DIR / "image_dimensions.csv"
    gt_file = METRICS_DIR / "class_distribution.csv"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    if gt_file.exists():
        df_dist = pd.read_csv(gt_file)
        sns.barplot(data=df_dist, x="Category", y="Image Count", palette="viridis", ax=axes[0])
        axes[0].set_title("Dataset Balance Across 8 Object Categories", fontweight="bold")
        axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=30, ha="right")
        for p in axes[0].patches:
            axes[0].annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width() / 2., p.get_height()),
                             ha='center', va='bottom', xytext=(0, 3), textcoords='offset points')
    else:
        axes[0].text(0.5, 0.5, "Class distribution data pending", ha="center")

    if dims_file.exists():
        df_dims = pd.read_csv(dims_file)
        sns.scatterplot(data=df_dims, x="width", y="height", alpha=0.4, color="#1f77b4", ax=axes[1])
        axes[1].set_title("Original Image Resolutions (px)", fontweight="bold")
        axes[1].set_xlabel("Width (px)")
        axes[1].set_ylabel("Height (px)")
    else:
        axes[1].text(0.5, 0.5, "Image dimension data pending", ha="center")

    plt.tight_layout()
    out = FIGURES_DIR / "fig01_data_quality_and_balance.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out.name}")


def plot_pca_variance():
    """Generates explained variance curves for PCA with 90% and 95% marked."""
    curve_file = METRICS_DIR / "pca_full_curve.csv"
    if not curve_file.exists():
        print("PCA curve data not found. Skipping plot.")
        return

    df = pd.read_csv(curve_file)
    n_90 = int(np.argmax(df["Cumulative_Explained_Variance"] >= 0.90) + 1)
    n_95 = int(np.argmax(df["Cumulative_Explained_Variance"] >= 0.95) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(15, 5))

    # Individual variance
    axes[0].plot(df["Component"][:100], df["Explained_Variance_Ratio"][:100] * 100, color="#1f77b4", lw=2)
    axes[0].set_title("Individual Explained Variance (First 100 PCs)", fontweight="bold")
    axes[0].set_xlabel("Principal Component Index")
    axes[0].set_ylabel("Explained Variance Ratio (%)")

    # Cumulative variance
    axes[1].plot(df["Component"], df["Cumulative_Explained_Variance"] * 100, color="#2ca02c", lw=2.5, label="Cumulative Variance")
    axes[1].axhline(90, color="#ff7f0e", linestyle="--", label=f"90% Threshold (n={n_90})")
    axes[1].axvline(n_90, color="#ff7f0e", linestyle="--")
    axes[1].axhline(95, color="#d62728", linestyle=":", label=f"95% Threshold (n={n_95})")
    axes[1].axvline(n_95, color="#d62728", linestyle=":")

    axes[1].set_title("Cumulative Explained Variance & Retention Thresholds", fontweight="bold")
    axes[1].set_xlabel("Principal Component Count")
    axes[1].set_ylabel("Cumulative Variance (%)")
    axes[1].legend(loc="lower right")

    plt.tight_layout()
    out = FIGURES_DIR / "fig03_pca_explained_variance.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out.name}")


def plot_tsne_grid():
    """Generates 6-panel grid for t-SNE parameter combinations."""
    labels_file = CLUSTERS_DIR / f"final_cluster_labels_k{DEFAULT_K}.npy"
    labels = np.load(labels_file) if labels_file.exists() else None

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    axes = axes.flatten()

    idx = 0
    for lr in TSNE_LEARNING_RATES:
        for perp in TSNE_PERPLEXITIES:
            ax = axes[idx]
            emb_file = EMBEDDINGS_DIR / f"tsne_perp{perp}_lr{lr}.npy"
            if emb_file.exists():
                coords = np.load(emb_file)
                scatter = ax.scatter(
                    coords[:, 0], coords[:, 1],
                    c=labels if labels is not None else "#1f77b4",
                    cmap="tab10" if labels is not None else None,
                    alpha=0.6, s=12
                )
                ax.set_title(f"Perplexity = {perp} | Learning Rate = {lr}", fontweight="bold", fontsize=11)
                ax.set_xticks([])
                ax.set_yticks([])
            else:
                ax.text(0.5, 0.5, "Embedding not generated", ha="center")
            idx += 1

    fig.suptitle("t-SNE Hyperparameter Grid: Perplexity {5, 30, 50} x Learning Rate {100, 200}", fontsize=15, fontweight="bold")
    plt.tight_layout()
    out = FIGURES_DIR / "fig05_tsne_parameter_grid.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out.name}")


def plot_pca_vs_tsne():
    """Generates comparative side-by-side visualization of PCA 2D vs t-SNE 2D."""
    pca_file = EMBEDDINGS_DIR / "pca_2d.npy"
    tsne_file = EMBEDDINGS_DIR / "tsne_perp30_lr200.npy"
    labels_file = CLUSTERS_DIR / f"final_cluster_labels_k{DEFAULT_K}.npy"

    if not (pca_file.exists() and tsne_file.exists() and labels_file.exists()):
        print("PCA/t-SNE embeddings missing. Skipping PCA vs t-SNE plot.")
        return

    pca_2d = np.load(pca_file)
    tsne_2d = np.load(tsne_file)
    labels = np.load(labels_file)

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))

    # PCA 2D
    scatter0 = axes[0].scatter(pca_2d[:, 0], pca_2d[:, 1], c=labels, cmap="tab10", alpha=0.6, s=15)
    axes[0].set_title("Global Linear Projection: PCA 2D Space", fontweight="bold")
    axes[0].set_xlabel("Principal Component 1")
    axes[0].set_ylabel("Principal Component 2")

    # t-SNE 2D
    scatter1 = axes[1].scatter(tsne_2d[:, 0], tsne_2d[:, 1], c=labels, cmap="tab10", alpha=0.6, s=15)
    axes[1].set_title("Non-Linear Manifold: t-SNE 2D Space (Perp=30, LR=200)", fontweight="bold")
    axes[1].set_xlabel("t-SNE Dimension 1")
    axes[1].set_ylabel("t-SNE Dimension 2")

    fig.suptitle("Dimensionality Reduction Comparison: Global PCA vs Local Manifold t-SNE", fontsize=15, fontweight="bold")
    plt.tight_layout()
    out = FIGURES_DIR / "fig06_pca_vs_tsne_comparison.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out.name}")


def plot_kmeans_evaluation():
    """Generates Elbow curve, Silhouette curve, and Davies-Bouldin curve across k."""
    metrics_file = METRICS_DIR / "kmeans_metrics_comparison.csv"
    if not metrics_file.exists():
        print("Clustering comparison metrics missing. Skipping.")
        return

    df = pd.read_csv(metrics_file)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # 1. Elbow (WCSS)
    sns.lineplot(data=df, x="k", y="WCSS_Inertia", hue="Representation", marker="o", lw=2, ax=axes[0])
    axes[0].set_title("Elbow Method: Within-Cluster Sum of Squares", fontweight="bold")
    axes[0].set_xlabel("Number of Clusters (k)")
    axes[0].set_ylabel("WCSS / Inertia")

    # 2. Silhouette Score
    sns.lineplot(data=df, x="k", y="Silhouette_Score", hue="Representation", marker="s", lw=2, ax=axes[1])
    axes[1].set_title("Silhouette Score Comparison (Higher is Better)", fontweight="bold")
    axes[1].set_xlabel("Number of Clusters (k)")
    axes[1].set_ylabel("Mean Silhouette Score")

    # 3. Davies-Bouldin Index
    sns.lineplot(data=df, x="k", y="Davies_Bouldin_Index", hue="Representation", marker="^", lw=2, ax=axes[2])
    axes[2].set_title("Davies-Bouldin Index (Lower is Better)", fontweight="bold")
    axes[2].set_xlabel("Number of Clusters (k)")
    axes[2].set_ylabel("Davies-Bouldin Score")

    plt.tight_layout()
    out = FIGURES_DIR / "fig07_kmeans_evaluation_metrics.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out.name}")


def plot_seed_stability_heatmap():
    """Generates heatmap of pairwise ARI across random seeds."""
    ari_file = METRICS_DIR / "seed_pairwise_ari.csv"
    if not ari_file.exists():
        return

    df_ari = pd.read_csv(ari_file, index_col=0)
    plt.figure(figsize=(7, 6))
    sns.heatmap(df_ari, annot=True, fmt=".4f", cmap="Blues", cbar=True, vmin=0.5, vmax=1.0)
    plt.title("K-Means Initialization Stability:\nPairwise Adjusted Rand Index Across Seeds", fontweight="bold")
    plt.tight_layout()
    out = FIGURES_DIR / "fig09_seed_stability_ari.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out.name}")


def plot_contingency_heatmap():
    """Generates row-normalized Contingency Heatmap (True Categories vs Discovered Clusters)."""
    cont_file = METRICS_DIR / "contingency_matrix_normalized.csv"
    if not cont_file.exists():
        return

    df_cont = pd.read_csv(cont_file, index_col=0)
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cont, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={'label': 'Percentage of True Category (%)'})
    plt.title("Post-Hoc Contingency Analysis:\nTrue Object Categories vs Discovered Clusters (%)", fontweight="bold")
    plt.xlabel("Discovered Unsupervised Clusters")
    plt.ylabel("Ground-Truth Categories (Revealed Post-Hoc)")
    plt.tight_layout()
    out = FIGURES_DIR / "fig11_contingency_heatmap.png"
    plt.savefig(out)
    plt.close()
    print(f"Saved: {out.name}")


def generate_all_figures():
    print("=" * 65)
    print("Generating Publication-Ready Figures in outputs/figures/...")
    plot_data_quality()
    plot_pca_variance()
    plot_tsne_grid()
    plot_pca_vs_tsne()
    plot_kmeans_evaluation()
    plot_seed_stability_heatmap()
    plot_contingency_heatmap()
    print("All figures successfully rendered and saved.")


if __name__ == "__main__":
    generate_all_figures()
