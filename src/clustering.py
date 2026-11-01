"""
Clustering Module for Object Discovery Lab
Implements interactive K-Means clustering, multi-metric evaluation (Inertia,
Silhouette, Davies-Bouldin, Calinski-Harabasz), Elbow/Silhouette sweeps,
random seed stability analysis, and centroid exemplar extraction.
"""

import sys
import time
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    davies_bouldin_score,
    calinski_harabasz_score,
    adjusted_rand_score
)
from scipy.spatial.distance import cdist
import joblib

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import RANDOM_SEED, STABILITY_SEEDS, DEFAULT_K


class ClusteringEngine:
    """
    Interactive Clustering Engine supporting dynamic K-Means configurations and exemplar extraction.
    """

    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.model = None
        self.labels_ = None
        self.centroids_ = None

    def fit_predict(
        self,
        X: np.ndarray,
        k: int = DEFAULT_K,
        init_method: str = "k-means++",
        n_init: int = 10,
        max_iter: int = 300
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Fits K-Means and returns cluster labels and full evaluation metrics.
        """
        n_samples = X.shape[0]
        safe_k = min(max(int(k), 2), max(n_samples - 1, 2))

        t0 = time.time()
        kmeans = KMeans(
            n_clusters=safe_k,
            init=init_method,
            n_init=max(int(n_init), 1),
            max_iter=max(int(max_iter), 50),
            random_state=self.random_state
        )
        labels = kmeans.fit_predict(X)
        elapsed = time.time() - t0

        self.model = kmeans
        self.labels_ = labels
        self.centroids_ = kmeans.cluster_centers_

        inertia = float(kmeans.inertia_)
        
        # Calculate cluster metrics safely
        unique_labels = np.unique(labels)
        if len(unique_labels) > 1 and len(unique_labels) < n_samples:
            try:
                sil = float(silhouette_score(X, labels))
            except Exception:
                sil = np.nan

            try:
                db = float(davies_bouldin_score(X, labels))
            except Exception:
                db = np.nan

            try:
                ch = float(calinski_harabasz_score(X, labels))
            except Exception:
                ch = np.nan
        else:
            sil, db, ch = np.nan, np.nan, np.nan

        # Cluster size breakdown
        sizes = [int(np.sum(labels == c)) for c in range(safe_k)]
        size_dist = {f"Cluster_{c}": s for c, s in enumerate(sizes)}

        metrics = {
            "k": safe_k,
            "inertia_wcss": round(inertia, 2),
            "silhouette_score": round(sil, 4) if not np.isnan(sil) else "N/A",
            "davies_bouldin_index": round(db, 4) if not np.isnan(db) else "N/A",
            "calinski_harabasz_score": round(ch, 2) if not np.isnan(ch) else "N/A",
            "fit_time_seconds": round(elapsed, 3),
            "cluster_sizes": size_dist,
            "min_cluster_size": min(sizes),
            "max_cluster_size": max(sizes)
        }

        return labels, metrics

    def run_k_sweep(
        self,
        X: np.ndarray,
        k_min: int = 2,
        k_max: int = 12,
        init_method: str = "k-means++",
        n_init: int = 10
    ) -> pd.DataFrame:
        """
        Runs multi-K evaluation sweep from k_min to k_max.
        Calculates WCSS (Elbow), Silhouette, Davies-Bouldin, and Calinski-Harabasz.
        """
        n_samples = X.shape[0]
        max_k_allowed = min(k_max, n_samples - 1)
        results = []

        for k in range(k_min, max_k_allowed + 1):
            km = KMeans(
                n_clusters=k,
                init=init_method,
                n_init=n_init,
                random_state=self.random_state
            )
            lbls = km.fit_predict(X)
            inertia = float(km.inertia_)

            try:
                sil = float(silhouette_score(X, lbls))
            except Exception:
                sil = np.nan

            try:
                db = float(davies_bouldin_score(X, lbls))
            except Exception:
                db = np.nan

            try:
                ch = float(calinski_harabasz_score(X, lbls))
            except Exception:
                ch = np.nan

            results.append({
                "k": k,
                "WCSS_Inertia": round(inertia, 2),
                "Silhouette_Score": round(sil, 4),
                "Davies_Bouldin_Index": round(db, 4),
                "Calinski_Harabasz_Score": round(ch, 2)
            })

        return pd.DataFrame(results)

    def test_seed_stability(
        self,
        X: np.ndarray,
        k: int = DEFAULT_K,
        seeds: Optional[List[int]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Runs K-Means across distinct initialization seeds.
        Computes pairwise Adjusted Rand Index (ARI) matrix to quantify cluster stability.
        """
        if seeds is None:
            seeds = STABILITY_SEEDS

        run_labels = {}
        run_stats = []

        for s in seeds:
            km = KMeans(n_clusters=k, init="k-means++", n_init=10, random_state=s)
            lbls = km.fit_predict(X)
            run_labels[s] = lbls

            sil = float(silhouette_score(X, lbls)) if X.shape[0] > k else np.nan
            sizes = [int(np.sum(lbls == c)) for c in range(k)]

            run_stats.append({
                "Seed": s,
                "Inertia_WCSS": round(float(km.inertia_), 2),
                "Silhouette": round(sil, 4) if not np.isnan(sil) else "N/A",
                "Min_Cluster_Size": min(sizes),
                "Max_Cluster_Size": max(sizes)
            })

        df_stats = pd.DataFrame(run_stats)

        # Pairwise ARI matrix
        n_seeds = len(seeds)
        ari_mat = np.zeros((n_seeds, n_seeds))
        for i, s1 in enumerate(seeds):
            for j, s2 in enumerate(seeds):
                ari_mat[i, j] = adjusted_rand_score(run_labels[s1], run_labels[s2])

        df_ari = pd.DataFrame(
            np.round(ari_mat, 4),
            index=[f"Seed_{s}" for s in seeds],
            columns=[f"Seed_{s}" for s in seeds]
        )

        return df_stats, df_ari

    def extract_centroid_exemplars(
        self,
        X: np.ndarray,
        df_manifest: pd.DataFrame,
        top_n: int = 12
    ) -> pd.DataFrame:
        """
        Extracts top N images closest to each cluster centroid in Euclidean feature space.
        """
        if self.centroids_ is None or self.labels_ is None:
            raise RuntimeError("Must fit K-Means model before extracting exemplars.")

        exemplar_records = []
        k = self.centroids_.shape[0]
        image_ids = df_manifest["image_id"].values
        file_paths = df_manifest["file_path"].values

        for c in range(k):
            members = np.where(self.labels_ == c)[0]
            if len(members) == 0:
                continue

            c_feats = X[members]
            centroid = self.centroids_[c:c+1]
            distances = cdist(c_feats, centroid, metric="euclidean").flatten()
            sorted_idx = np.argsort(distances)

            n_extract = min(top_n, len(members))
            top_members = members[sorted_idx[:n_extract]]
            top_dists = distances[sorted_idx[:n_extract]]

            for rank, (m_idx, d_val) in enumerate(zip(top_members, top_dists), start=1):
                exemplar_records.append({
                    "Cluster_ID": c,
                    "Rank": rank,
                    "Image_ID": str(image_ids[m_idx]),
                    "File_Path": str(file_paths[m_idx]),
                    "Distance_To_Centroid": round(float(d_val), 4),
                    "Cluster_Total_Size": len(members)
                })

        return pd.DataFrame(exemplar_records)

    def save(self, model_path: Path):
        """Saves fitted KMeans model."""
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, model_path)
