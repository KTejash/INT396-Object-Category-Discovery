"""
Dimensionality Reduction Module for Object Discovery Lab
Implements interactive Principal Component Analysis (PCA) and t-Distributed
Stochastic Neighbor Embedding (t-SNE) with strict parameter validation and dynamic recomputation.
"""

import sys
import time
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import joblib

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RANDOM_SEED, MODELS_DIR


class PCAManager:
    """
    Interactive PCA engine supporting target variance thresholds or fixed component counts.
    """

    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.pca = None
        self.explained_variance_ratio_ = None
        self.cumulative_variance_ = None
        self.n_components_ = None

    def fit_transform(
        self,
        X: np.ndarray,
        mode: str = "variance",
        target_variance: float = 0.95,
        n_components: int = 50
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Fits PCA and transforms feature matrix X.
        Args:
            X: Standardized input matrix (N, D)
            mode: 'variance' (e.g. 0.90, 0.95, 0.99) or 'fixed' (explicit integer)
            target_variance: Float in (0.0, 1.0]
            n_components: Integer <= min(N, D)
        """
        n_samples, n_features = X.shape
        max_possible = min(n_samples, n_features)

        if max_possible < 2:
            raise ValueError(f"Insufficient samples or features for PCA: shape={X.shape}")

        if mode == "variance":
            target_var = min(max(float(target_variance), 0.05), 0.999)
            # Fit preliminary or full PCA to determine exact component cutoff
            pca_pre = PCA(n_components=max_possible, random_state=self.random_state)
            pca_pre.fit(X)
            cum_var = np.cumsum(pca_pre.explained_variance_ratio_)
            k_components = int(np.argmax(cum_var >= target_var) + 1)
            k_components = min(max(k_components, 2), max_possible)
        else:
            k_components = min(max(int(n_components), 2), max_possible)

        self.pca = PCA(n_components=k_components, random_state=self.random_state)
        X_pca = self.pca.fit_transform(X).astype(np.float32)

        self.explained_variance_ratio_ = self.pca.explained_variance_ratio_
        self.cumulative_variance_ = np.cumsum(self.explained_variance_ratio_)
        self.n_components_ = k_components

        compression_pct = round((1.0 - (k_components / n_features)) * 100, 2)
        total_explained_var = round(float(self.cumulative_variance_[-1]) * 100, 2)

        metrics = {
            "original_dimensions": n_features,
            "reduced_dimensions": k_components,
            "compression_ratio": f"{compression_pct}% reduction",
            "explained_variance_total": f"{total_explained_var}%",
            "first_pc_variance": round(float(self.explained_variance_ratio_[0]) * 100, 2),
            "second_pc_variance": round(float(self.explained_variance_ratio_[1]) * 100, 2) if k_components > 1 else 0.0
        }

        # Build scree curve DataFrame
        df_scree = pd.DataFrame({
            "Component": np.arange(1, k_components + 1),
            "Individual_Variance_Ratio": self.explained_variance_ratio_,
            "Cumulative_Variance_Ratio": self.cumulative_variance_
        })

        return X_pca, {"metrics": metrics, "scree_df": df_scree}

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Projects new sample vectors onto fitted principal component axes."""
        if self.pca is None:
            raise RuntimeError("PCA model has not been fitted yet.")
        return self.pca.transform(X).astype(np.float32)

    def get_loadings(self, top_n_pcs: int = 5) -> pd.DataFrame:
        """Returns component loadings (eigenvectors) for the top N principal components."""
        if self.pca is None:
            return pd.DataFrame()
        n_pcs = min(top_n_pcs, self.pca.n_components_)
        loadings = self.pca.components_[:n_pcs]
        df_loadings = pd.DataFrame(
            loadings,
            index=[f"PC_{i+1}" for i in range(n_pcs)],
            columns=[f"Feat_{j}" for j in range(loadings.shape[1])]
        )
        return df_loadings

    def save(self, filepath: Path):
        """Saves fitted PCA model."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.pca, filepath)

    @classmethod
    def load(cls, filepath: Path, random_state: int = RANDOM_SEED):
        inst = cls(random_state=random_state)
        if filepath.exists():
            inst.pca = joblib.load(filepath)
            inst.explained_variance_ratio_ = inst.pca.explained_variance_ratio_
            inst.cumulative_variance_ = np.cumsum(inst.explained_variance_ratio_)
            inst.n_components_ = inst.pca.n_components_
        return inst


class TSNEManager:
    """
    Interactive t-SNE engine with dynamic parameter validation and execution tracking.
    """

    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state

    def validate_perplexity(self, n_samples: int, perplexity: float) -> float:
        """Ensures perplexity adheres to mathematical constraints (p < N/3 and p >= 2)."""
        if n_samples < 5:
            raise ValueError(f"Too few samples for t-SNE ({n_samples} samples). Need at least 5.")
        max_valid = max(float(n_samples - 1) / 3.0, 2.0)
        safe_perp = min(max(float(perplexity), 2.0), max_valid)
        return round(safe_perp, 1)

    def fit_transform(
        self,
        X: np.ndarray,
        perplexity: float = 30.0,
        learning_rate: float = 200.0,
        max_iter: int = 1000,
        metric: str = "euclidean"
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Executes real t-SNE projection onto 2D space.
        """
        n_samples = X.shape[0]
        safe_perp = self.validate_perplexity(n_samples, perplexity)

        # If dimensions > 50, apply preliminary PCA compression for speed and noise filtering
        if X.shape[1] > 50:
            pca_pre = PCA(n_components=min(50, n_samples - 1), random_state=self.random_state)
            X_input = pca_pre.fit_transform(X)
            init_strategy = "pca"
        else:
            X_input = X
            init_strategy = "random"

        t0 = time.time()
        tsne = TSNE(
            n_components=2,
            perplexity=safe_perp,
            learning_rate=max(float(learning_rate), 10.0),
            max_iter=int(max_iter),
            metric=metric,
            random_state=self.random_state,
            init=init_strategy
        )
        embedding_2d = tsne.fit_transform(X_input).astype(np.float32)
        elapsed = time.time() - t0

        kl_div = getattr(tsne, "kl_divergence_", None)

        metadata = {
            "perplexity_requested": perplexity,
            "perplexity_used": safe_perp,
            "learning_rate": learning_rate,
            "max_iter": max_iter,
            "metric": metric,
            "runtime_seconds": round(elapsed, 2),
            "kl_divergence": round(float(kl_div), 4) if kl_div is not None else "N/A"
        }

        return embedding_2d, metadata
