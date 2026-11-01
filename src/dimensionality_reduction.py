"""
Dimensionality Reduction Module
Implements Principal Component Analysis (PCA) with 90% & 95% variance thresholds
and systematic t-SNE experiments across perplexity and learning rate sweeps.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import joblib

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    CNN_FEATURES_SCALED,
    HIST_FEATURES_SCALED,
    EMBEDDINGS_DIR,
    MODELS_DIR,
    METRICS_DIR,
    RANDOM_SEED,
    PCA_VARIANCE_THRESHOLDS,
    TSNE_PERPLEXITIES,
    TSNE_LEARNING_RATES,
    TSNE_ITERATIONS,
    PCA_MODEL_90_PATH,
    PCA_MODEL_95_PATH
)


def run_pca_experiments(features=None):
    """
    Executes PCA on standardized features.
    Computes individual and cumulative explained variance.
    Identifies components for 90% and 95% variance thresholds.
    """
    print("=" * 65)
    print("Executing Principal Component Analysis (PCA)...")

    if features is None:
        features = np.load(CNN_FEATURES_SCALED)

    n_samples, n_features = features.shape
    max_components = min(n_samples, n_features)

    print(f"Fitting Full PCA on matrix of shape {features.shape}...")
    pca_full = PCA(n_components=max_components, random_state=RANDOM_SEED)
    pca_full.fit(features)

    exp_var = pca_full.explained_variance_ratio_
    cum_var = np.cumsum(exp_var)

    # Find components for 90% and 95% variance
    n_90 = int(np.argmax(cum_var >= 0.90) + 1)
    n_95 = int(np.argmax(cum_var >= 0.95) + 1)

    print(f"  Components for 90% Retained Variance: {n_90} / {n_features} (Compression: {100*(1 - n_90/n_features):.1f}%)")
    print(f"  Components for 95% Retained Variance: {n_95} / {n_features} (Compression: {100*(1 - n_95/n_features):.1f}%)")

    # Fit and transform for 90% and 95%
    pca_90 = PCA(n_components=n_90, random_state=RANDOM_SEED)
    feats_pca_90 = pca_90.fit_transform(features).astype(np.float32)
    joblib.dump(pca_90, PCA_MODEL_90_PATH)
    np.save(EMBEDDINGS_DIR / "pca_90_features.npy", feats_pca_90)

    pca_95 = PCA(n_components=n_95, random_state=RANDOM_SEED)
    feats_pca_95 = pca_95.fit_transform(features).astype(np.float32)
    joblib.dump(pca_95, PCA_MODEL_95_PATH)
    np.save(EMBEDDINGS_DIR / "pca_95_features.npy", feats_pca_95)

    # 2D PCA for visual mapping
    pca_2d = PCA(n_components=2, random_state=RANDOM_SEED)
    feats_pca_2d = pca_2d.fit_transform(features).astype(np.float32)
    np.save(EMBEDDINGS_DIR / "pca_2d.npy", feats_pca_2d)

    # Save Variance Metrics Table
    var_records = []
    milestones = [1, 2, 5, 10, 20, 50, n_90, n_95, min(200, max_components), max_components]
    milestones = sorted(list(set(milestones)))

    for k in milestones:
        var_records.append({
            "Principal Component Count": k,
            "Individual Variance Ratio (%)": round(float(exp_var[k - 1]) * 100, 4),
            "Cumulative Variance Ratio (%)": round(float(cum_var[k - 1]) * 100, 3)
        })

    df_var = pd.DataFrame(var_records)
    df_var.to_csv(METRICS_DIR / "pca_variance_milestones.csv", index=False)

    # Save full curve data for plotting
    df_curve = pd.DataFrame({
        "Component": np.arange(1, len(exp_var) + 1),
        "Explained_Variance_Ratio": exp_var,
        "Cumulative_Explained_Variance": cum_var
    })
    df_curve.to_csv(METRICS_DIR / "pca_full_curve.csv", index=False)

    print("\nPCA Key Milestones:")
    print(df_var.to_string(index=False))

    return {
        "n_90": n_90,
        "n_95": n_95,
        "feats_pca_90": feats_pca_90,
        "feats_pca_95": feats_pca_95,
        "feats_pca_2d": feats_pca_2d,
        "pca_90": pca_90,
        "pca_95": pca_95,
        "pca_2d": pca_2d
    }


def run_tsne_experiments(pca_reduced_features=None):
    """
    Executes systematic t-SNE grid experiments across perplexity [5, 30, 50]
    and learning rates [100, 200] on the PCA-95% reduced space.
    """
    print("=" * 65)
    print("Executing Systematic t-SNE Hyperparameter Sweeps...")

    if pca_reduced_features is None:
        pca_reduced_features = np.load(EMBEDDINGS_DIR / "pca_95_features.npy")

    results = []
    total_runs = len(TSNE_PERPLEXITIES) * len(TSNE_LEARNING_RATES)
    run_idx = 0

    for perp in TSNE_PERPLEXITIES:
        for lr in TSNE_LEARNING_RATES:
            run_idx += 1
            print(f"[{run_idx}/{total_runs}] Running t-SNE (Perplexity={perp}, Learning Rate={lr})...")
            t0 = time.time()

            tsne = TSNE(
                n_components=2,
                perplexity=perp,
                learning_rate=lr,
                max_iter=TSNE_ITERATIONS,
                random_state=RANDOM_SEED,
                init="pca",
                n_jobs=-1
            )
            embedding_2d = tsne.fit_transform(pca_reduced_features).astype(np.float32)
            elapsed = time.time() - t0
            kl_div = getattr(tsne, "kl_divergence_", None)

            emb_filename = f"tsne_perp{perp}_lr{lr}.npy"
            np.save(EMBEDDINGS_DIR / emb_filename, embedding_2d)

            results.append({
                "Perplexity": perp,
                "Learning_Rate": lr,
                "KL_Divergence": round(float(kl_div), 4) if kl_div is not None else np.nan,
                "Runtime_Seconds": round(elapsed, 2),
                "Embedding_File": emb_filename
            })
            print(f"   -> Completed in {elapsed:.2f}s | KL Divergence: {kl_div:.4f}")

    df_tsne = pd.DataFrame(results)
    df_tsne.to_csv(METRICS_DIR / "tsne_parameter_comparison.csv", index=False)
    print("\nt-SNE Parameter Comparison Summary:")
    print(df_tsne.to_string(index=False))

    return df_tsne


def run_dimensionality_reduction_pipeline():
    pca_results = run_pca_experiments()
    tsne_results = run_tsne_experiments(pca_results["feats_pca_95"])
    return pca_results, tsne_results


if __name__ == "__main__":
    run_dimensionality_reduction_pipeline()
