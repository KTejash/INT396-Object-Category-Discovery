"""
Static Figure Generation Script for VisionCluster AI
Generates and exports publication-grade plots (Scree, Elbow, t-SNE,
Cluster Distribution, Contingency Heatmap) to outputs/figures/.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    FIGURES_DIR,
    CNN_FEATURES_RAW,
    CNN_FEATURES_SCALED,
    IMAGE_IDS_FILE,
    MANIFEST_FILE,
    GROUND_TRUTH_FILE
)
from src.preprocessing import PreprocessingPipeline
from src.dimensionality_reduction import PCAManager, TSNEManager
from src.clustering import ClusteringEngine
from src.evaluation import ExternalValidator


def generate_all_figures():
    print("=" * 60)
    print("   GENERATING PUBLICATION FIGURES FOR ACADEMIC REPORT")
    print("=" * 60)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    if not CNN_FEATURES_RAW.exists():
        print(f"[SKIP] Raw features not found at {CNN_FEATURES_RAW}. Run extraction first.")
        return

    X_raw = np.load(CNN_FEATURES_RAW)
    print(f"[INFO] Loaded raw features: shape {X_raw.shape}")

    # 1. Scale
    scaler = PreprocessingPipeline(scaler_type="standard")
    X_scaled, _ = scaler.fit_transform(X_raw)

    # 2. PCA
    pca = PCAManager()
    X_pca, pca_dict = pca.fit_transform(X_scaled, mode="variance", target_variance=0.95)
    df_scree = pca_dict["scree_df"]

    # Figure 1: Scree Plot
    plt.figure(figsize=(9, 4.5))
    plt.bar(df_scree["Component"][:40], df_scree["Individual_Variance_Ratio"][:40] * 100, color="#00b4d8", alpha=0.8, label="Individual Variance (%)")
    plt.plot(df_scree["Component"][:40], df_scree["Cumulative_Variance_Ratio"][:40] * 100, color="#d90429", marker="o", markersize=3, label="Cumulative Variance (%)")
    plt.axhline(95, color="#ffb703", linestyle="--", label="95% Target Variance")
    plt.xlabel("Principal Component Index")
    plt.ylabel("Explained Variance (%)")
    plt.title("PCA Scree Curve: Feature Dimensionality Reduction")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    fig1_path = FIGURES_DIR / "pca_scree_curve.png"
    plt.savefig(fig1_path, dpi=200)
    plt.close()
    print(f"[SAVED] {fig1_path.name}")

    # 3. K-Means
    cluster_engine = ClusteringEngine()
    labels, metrics = cluster_engine.fit_predict(X_pca, k=8)

    # Figure 2: Cluster Distribution
    unique, counts = np.unique(labels, return_counts=True)
    plt.figure(figsize=(7, 4))
    plt.bar([f"Cluster {c}" for c in unique], counts, color="#3a86ff", edgecolor="#023e8a")
    plt.xlabel("Discovered Cluster")
    plt.ylabel("Number of Assigned Images")
    plt.title(f"K-Means Cluster Size Distribution (k=8, Total: {len(labels)})")
    plt.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    fig2_path = FIGURES_DIR / "cluster_size_distribution.png"
    plt.savefig(fig2_path, dpi=200)
    plt.close()
    print(f"[SAVED] {fig2_path.name}")

    # Figure 3: Contingency Heatmap
    if GROUND_TRUTH_FILE.exists():
        df_gt = pd.read_csv(GROUND_TRUTH_FILE)
        val_res = ExternalValidator.evaluate(df_gt["category"].values, labels)
        df_norm = val_res["contingency_matrix_normalized"]

        plt.figure(figsize=(9, 6.5))
        sns.heatmap(df_norm, annot=True, fmt=".1f", cmap="YlGnBu", cbar_kws={"label": "Class Allocation (%)"})
        plt.xlabel("Discovered Cluster")
        plt.ylabel("Ground-Truth Object Category")
        plt.title(f"Contingency Matrix Heatmap (ARI: {val_res['ari']:.4f}, NMI: {val_res['nmi']:.4f})")
        plt.tight_layout()
        fig3_path = FIGURES_DIR / "contingency_matrix_heatmap.png"
        plt.savefig(fig3_path, dpi=200)
        plt.close()
        print(f"[SAVED] {fig3_path.name}")

    print("=" * 60)
    print("Static publication figures successfully created in outputs/figures/")
    print("=" * 60)


if __name__ == "__main__":
    generate_all_figures()
