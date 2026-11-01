"""
Post-Hoc External Validation and Evaluation Module
Strictly evaluates clustering partitions against reference labels post-hoc.
Computes Adjusted Rand Index (ARI), Normalized Mutual Information (NMI),
Contingency Matrices, and normalized category purity heatmaps.
"""

import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    completeness_score,
    homogeneity_score,
    v_measure_score
)
from sklearn.metrics.cluster import contingency_matrix

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class ExternalValidator:
    """
    Evaluator for comparing discovered unsupervised cluster assignments
    with ground-truth reference labels post-hoc.
    """

    @staticmethod
    def evaluate(
        true_labels: np.ndarray,
        cluster_labels: np.ndarray
    ) -> Dict[str, Any]:
        """
        Computes external agreement metrics between reference labels and discovered clusters.
        """
        if true_labels is None or len(true_labels) == 0:
            return {"available": False, "message": "No ground truth labels provided."}

        if len(true_labels) != len(cluster_labels):
            raise ValueError(f"Length mismatch: {len(true_labels)} true labels vs {len(cluster_labels)} cluster assignments.")

        ari = float(adjusted_rand_score(true_labels, cluster_labels))
        nmi = float(normalized_mutual_info_score(true_labels, cluster_labels, average_method="arithmetic"))
        v_meas = float(v_measure_score(true_labels, cluster_labels))
        homo = float(homogeneity_score(true_labels, cluster_labels))
        comp = float(completeness_score(true_labels, cluster_labels))

        unique_cats = sorted(list(set(true_labels)))
        unique_clusters = sorted(list(set(cluster_labels)))

        # Absolute contingency matrix
        cont_mat = contingency_matrix(true_labels, cluster_labels)
        df_cont = pd.DataFrame(
            cont_mat,
            index=unique_cats,
            columns=[f"Cluster_{c}" for c in unique_clusters]
        )

        # Row-normalized contingency matrix (Percentage of each true class assigned to each cluster)
        row_sums = cont_mat.sum(axis=1, keepdims=True)
        cont_norm = np.where(row_sums > 0, cont_mat / row_sums * 100.0, 0.0)
        df_cont_norm = pd.DataFrame(
            np.round(cont_norm, 2),
            index=unique_cats,
            columns=[f"Cluster_{c}" for c in unique_clusters]
        )

        metrics_summary = pd.DataFrame({
            "Metric": [
                "Adjusted Rand Index (ARI)",
                "Normalized Mutual Information (NMI)",
                "V-Measure",
                "Homogeneity Score",
                "Completeness Score",
                "Reference Category Count",
                "Discovered Cluster Count"
            ],
            "Score": [
                round(ari, 4),
                round(nmi, 4),
                round(v_meas, 4),
                round(homo, 4),
                round(comp, 4),
                len(unique_cats),
                len(unique_clusters)
            ],
            "Interpretation": [
                "Chance-adjusted pairwise agreement [-1.0, 1.0]. 0 = random chance.",
                "Information-theoretic overlap [0.0, 1.0]. Higher = stronger mutual info.",
                "Harmonic mean of homogeneity and completeness.",
                "Measures if each cluster contains only members of a single class.",
                "Measures if all members of a given class are assigned to the same cluster.",
                f"{len(unique_cats)} human categories",
                f"{len(unique_clusters)} algorithmic clusters"
            ]
        })

        return {
            "available": True,
            "ari": round(ari, 4),
            "nmi": round(nmi, 4),
            "v_measure": round(v_meas, 4),
            "homogeneity": round(homo, 4),
            "completeness": round(comp, 4),
            "summary_df": metrics_summary,
            "contingency_matrix": df_cont,
            "contingency_matrix_normalized": df_cont_norm
        }
