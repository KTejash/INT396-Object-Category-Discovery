"""
Experiment Manager Module for Object Discovery Lab
Logs, serializes, retrieves, compares, and exports unsupervised ML experiment runs.
Maintains persistent JSON history across application restarts.
"""

import sys
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import pandas as pd

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
EXPERIMENTS_FILE = EXPERIMENTS_DIR / "experiments_log.json"


class ExperimentManager:
    """
    Manages experiment metadata, metric persistence, and comparative analysis.
    """

    def __init__(self, storage_file: Path = EXPERIMENTS_FILE):
        self.storage_file = storage_file
        self._ensure_storage()

    def _ensure_storage(self):
        if not self.storage_file.exists():
            with open(self.storage_file, "w", encoding="utf-8") as f:
                json.dump([], f, indent=2)

    def _load_raw(self) -> List[Dict[str, Any]]:
        try:
            with open(self.storage_file, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []

    def _save_raw(self, experiments: List[Dict[str, Any]]):
        with open(self.storage_file, "w", encoding="utf-8") as f:
            json.dump(experiments, f, indent=2)

    def log_experiment(
        self,
        name: str,
        dataset_name: str,
        sample_count: int,
        feature_type: str,
        scaler_type: str,
        pca_config: Dict[str, Any],
        kmeans_config: Dict[str, Any],
        metrics: Dict[str, Any],
        tsne_config: Optional[Dict[str, Any]] = None,
        notes: str = ""
    ) -> str:
        """
        Logs a new experiment run and returns its unique ID.
        """
        exp_id = f"EXP_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        record = {
            "experiment_id": exp_id,
            "experiment_name": name or f"Run_{exp_id}",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "dataset_name": dataset_name,
            "sample_count": sample_count,
            "feature_type": feature_type,
            "scaler_type": scaler_type,
            "pca_config": pca_config,
            "kmeans_config": kmeans_config,
            "tsne_config": tsne_config or {},
            "metrics": metrics,
            "notes": notes
        }

        experiments = self._load_raw()
        experiments.append(record)
        self._save_raw(experiments)
        return exp_id

    def list_experiments(self) -> pd.DataFrame:
        """
        Returns a tabular summary DataFrame of all saved experiments.
        """
        experiments = self._load_raw()
        if not experiments:
            return pd.DataFrame()

        rows = []
        for exp in experiments:
            m = exp.get("metrics", {})
            km = exp.get("kmeans_config", {})
            rows.append({
                "Experiment ID": str(exp.get("experiment_id", "")),
                "Name": str(exp.get("experiment_name", "")),
                "Timestamp": str(exp.get("timestamp", "")),
                "Dataset": str(exp.get("dataset_name", "")),
                "Samples": str(exp.get("sample_count", "")),
                "Features": str(exp.get("feature_type", "")),
                "Scaler": str(exp.get("scaler_type", "")),
                "k (Clusters)": str(km.get("k", "N/A")),
                "Silhouette": f"{float(m.get('silhouette_score', 0)):.4f}" if m.get("silhouette_score") not in [None, "N/A"] else "N/A",
                "Davies-Bouldin": f"{float(m.get('davies_bouldin_index', 0)):.4f}" if m.get("davies_bouldin_index") not in [None, "N/A"] else "N/A",
                "Calinski-Harabasz": f"{float(m.get('calinski_harabasz_score', 0)):.2f}" if m.get("calinski_harabasz_score") not in [None, "N/A"] else "N/A",
                "ARI": f"{float(m.get('ari', 0)):.4f}" if m.get("ari") not in [None, "N/A"] else "N/A",
                "NMI": f"{float(m.get('nmi', 0)):.4f}" if m.get("nmi") not in [None, "N/A"] else "N/A"
            })

        df = pd.DataFrame(rows)
        return df.astype(str)

    def get_experiment(self, exp_id: str) -> Optional[Dict[str, Any]]:
        """Retrieves complete metadata for an experiment."""
        experiments = self._load_raw()
        for exp in experiments:
            if exp.get("experiment_id") == exp_id:
                return exp
        return None

    def delete_experiment(self, exp_id: str) -> bool:
        """Deletes an experiment by ID."""
        experiments = self._load_raw()
        filtered = [exp for exp in experiments if exp.get("experiment_id") != exp_id]
        if len(filtered) != len(experiments):
            self._save_raw(filtered)
            return True
        return False

    def compare_experiments(self, exp_ids: List[str]) -> pd.DataFrame:
        """
        Generates a comparative side-by-side metric matrix for selected experiments.
        """
        experiments = self._load_raw()
        selected = [e for e in experiments if e.get("experiment_id") in exp_ids]
        if not selected:
            return pd.DataFrame()

        records = {}
        for exp in selected:
            name = f"{exp.get('experiment_name')} ({exp.get('experiment_id')})"
            m = exp.get("metrics", {})
            km = exp.get("kmeans_config", {})
            pca = exp.get("pca_config", {})
            
            records[name] = {
                "Dataset": str(exp.get("dataset_name", "")),
                "Sample Count": str(exp.get("sample_count", "")),
                "Feature Extraction": str(exp.get("feature_type", "")),
                "Scaling Method": str(exp.get("scaler_type", "")),
                "PCA Mode": str(pca.get("mode", "None")),
                "PCA Components": str(pca.get("components", "None")),
                "Cluster Count (k)": str(km.get("k", "N/A")),
                "Inertia (WCSS)": f"{float(m.get('inertia_wcss', 0)):.2f}" if m.get("inertia_wcss") not in [None, "N/A"] else "N/A",
                "Silhouette Score": f"{float(m.get('silhouette_score', 0)):.4f}" if m.get("silhouette_score") not in [None, "N/A"] else "N/A",
                "Davies-Bouldin Index": f"{float(m.get('davies_bouldin_index', 0)):.4f}" if m.get("davies_bouldin_index") not in [None, "N/A"] else "N/A",
                "Calinski-Harabasz": f"{float(m.get('calinski_harabasz_score', 0)):.2f}" if m.get("calinski_harabasz_score") not in [None, "N/A"] else "N/A",
                "Adjusted Rand Index (ARI)": f"{float(m.get('ari', 0)):.4f}" if m.get("ari") not in [None, "N/A"] else "N/A",
                "Normalized Mutual Info (NMI)": f"{float(m.get('nmi', 0)):.4f}" if m.get("nmi") not in [None, "N/A"] else "N/A"
            }

        df = pd.DataFrame(records)
        return df.astype(str)

    def export_all_json(self) -> str:
        """Exports entire experiment catalog as JSON string."""
        return json.dumps(self._load_raw(), indent=2)

    def export_all_csv(self) -> str:
        """Exports experiment summary as CSV string."""
        df = self.list_experiments()
        return df.to_csv(index=False)

    def delete_experiment(self, experiment_id: str) -> bool:
        """Removes a specific experiment record from the registry."""
        experiments = self._load_raw()
        filtered = [e for e in experiments if e.get("experiment_id") != experiment_id]
        if len(filtered) < len(experiments):
            self._save_raw(filtered)
            return True
        return False

    def generate_markdown_report(self, experiment_id: str) -> str:
        """Generates an academic report formatted in Markdown for a single experiment."""
        experiments = self._load_raw()
        match = next((e for e in experiments if e.get("experiment_id") == experiment_id), None)
        if not match:
            return f"Experiment '{experiment_id}' not found."

        m = match.get("metrics", {})
        km = match.get("kmeans_config", {})
        pca = match.get("pca_config", {})

        report = f"""# Experiment Report: {match.get('experiment_name')}
**Experiment ID:** `{match.get('experiment_id')}`  
**Timestamp:** {match.get('timestamp')}  
**Dataset:** `{match.get('dataset_name')}` ({match.get('sample_count')} images)

---

## 1. Feature Representation & Pipeline
- **Feature Extraction Method:** `{match.get('feature_type')}`
- **Feature Preprocessing:** `{match.get('scaler_type')}`
- **PCA Dimensionality Reduction:** {pca}
- **Clustering Configuration:** K-Means with k={km.get('k', 'N/A')}

---

## 2. Quantitative Evaluation Metrics
| Metric | Value |
|---|---|
| Inertia (WCSS) | `{m.get('inertia_wcss', 'N/A')}` |
| Silhouette Score | `{m.get('silhouette_score', 'N/A')}` |
| Davies-Bouldin Index | `{m.get('davies_bouldin_index', 'N/A')}` |
| Calinski-Harabasz Score | `{m.get('calinski_harabasz_score', 'N/A')}` |
| Adjusted Rand Index (ARI) | `{m.get('ari', 'N/A')}` |
| Normalized Mutual Information (NMI) | `{m.get('nmi', 'N/A')}` |

---

## 3. Notes & Observations
> {match.get('notes', 'No notes provided.')}
"""
        return report

