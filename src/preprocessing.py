"""
Preprocessing Module for Object Discovery Lab
Handles missing value verification, feature matrix validation,
and applies scalable transformations (StandardScaler, MinMaxScaler, L2 Normalizer).
Computes before-and-after distribution diagnostics.
"""

import sys
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer, RobustScaler
import joblib

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODELS_DIR, RANDOM_SEED


class PreprocessingPipeline:
    """
    Standardization, MinMax scaling, or L2 normalization for high-dimensional feature spaces.
    """

    def __init__(self, scaler_type: str = "standard"):
        self.scaler_type = scaler_type.lower()
        self.scaler = None
        self._init_scaler()

    def _init_scaler(self):
        if self.scaler_type == "standard":
            self.scaler = StandardScaler()
        elif self.scaler_type == "minmax":
            self.scaler = MinMaxScaler(feature_range=(0.0, 1.0))
        elif self.scaler_type == "robust":
            self.scaler = RobustScaler()
        elif self.scaler_type == "l2_norm":
            self.scaler = Normalizer(norm="l2")
        elif self.scaler_type == "none":
            self.scaler = None
        else:
            raise ValueError(f"Unknown scaler_type: '{self.scaler_type}'. Choose 'standard', 'minmax', 'robust', 'l2_norm', or 'none'.")

    def validate_features(self, X: np.ndarray) -> Dict[str, Any]:
        """
        Audits input feature matrix for numerical stability:
        - Non-empty check
        - Finite values (NaN / Inf check)
        - Variance check (detect zero-variance dimensions)
        """
        if X is None or X.size == 0:
            raise ValueError("Feature matrix is empty or None.")

        if not isinstance(X, np.ndarray):
            X = np.asarray(X)

        nan_count = int(np.isnan(X).sum())
        inf_count = int(np.isinf(X).sum())
        zero_var_cols = int((np.var(X, axis=0) == 0).sum()) if X.shape[0] > 1 else 0

        return {
            "n_samples": X.shape[0],
            "n_features": X.shape[1],
            "nan_count": nan_count,
            "inf_count": inf_count,
            "zero_variance_features": zero_var_cols,
            "is_valid": (nan_count == 0 and inf_count == 0)
        }

    def compute_distribution_stats(self, X: np.ndarray, label: str) -> Dict[str, Any]:
        """Computes summary statistics for a feature matrix."""
        return {
            "Stage": label,
            "Samples": X.shape[0],
            "Dimensions": X.shape[1],
            "Mean (Overall)": round(float(np.mean(X)), 4),
            "Std (Overall)": round(float(np.std(X)), 4),
            "Min": round(float(np.min(X)), 4),
            "Max": round(float(np.max(X)), 4)
        }

    def fit_transform(self, X: np.ndarray) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Fits the scaler on X and returns transformed X and before/after stats DataFrame.
        """
        audit = self.validate_features(X)
        if not audit["is_valid"]:
            # Clean NaNs if present by imputing column mean
            if audit["nan_count"] > 0:
                col_means = np.nanmean(X, axis=0)
                inds = np.where(np.isnan(X))
                X[inds] = np.take(col_means, inds[1])
            if audit["inf_count"] > 0:
                X = np.nan_to_num(X)

        before_stats = self.compute_distribution_stats(X, "Before Scaling (Raw)")

        if self.scaler is not None:
            X_scaled = self.scaler.fit_transform(X).astype(np.float32)
        else:
            X_scaled = X.copy().astype(np.float32)

        after_stats = self.compute_distribution_stats(X_scaled, f"After Scaling ({self.scaler_type.capitalize()})")
        df_stats = pd.DataFrame([before_stats, after_stats])

        return X_scaled, df_stats

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transforms new samples using already fitted scaler."""
        if self.scaler is not None:
            return self.scaler.transform(X).astype(np.float32)
        return X.astype(np.float32)

    def save(self, filepath: Path):
        """Saves fitted scaler to disk."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.scaler, filepath)

    @classmethod
    def load(cls, filepath: Path, scaler_type: str = "standard"):
        """Loads fitted scaler from disk."""
        inst = cls(scaler_type=scaler_type)
        if filepath.exists():
            inst.scaler = joblib.load(filepath)
        return inst
