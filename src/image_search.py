"""
Image Discovery Studio & Nearest-Neighbor Search Engine
Infers the natural cluster placement of any novel uploaded image,
computes distances to all cluster centroids, and retrieves the most
visually similar exemplar images from the active feature space.
"""

import sys
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
import pandas as pd
from PIL import Image
from scipy.spatial.distance import cdist

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.feature_extractor import FeatureExtractor
from src.preprocessing import PreprocessingPipeline
from src.dimensionality_reduction import PCAManager


class ImageDiscoveryStudio:
    """
    Inference and visual similarity retrieval engine for arbitrary user-provided images.
    """

    def __init__(
        self,
        feature_extractor: Optional[FeatureExtractor] = None,
        scaler: Optional[PreprocessingPipeline] = None,
        pca_manager: Optional[PCAManager] = None,
        centroids: Optional[np.ndarray] = None,
        dataset_features: Optional[np.ndarray] = None,
        df_manifest: Optional[pd.DataFrame] = None
    ):
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.scaler = scaler
        self.pca_manager = pca_manager
        self.centroids = centroids
        self.dataset_features = dataset_features
        self.df_manifest = df_manifest

    def process_and_assign(
        self,
        image_input,  # PIL.Image or Path or bytes
        feature_type: str = "resnet50",
        bins_per_channel: int = 8,
        top_k_neighbors: int = 6
    ) -> Dict[str, Any]:
        """
        Executes full inference pipeline for a single novel image:
        1. Extract feature vector using identical architecture.
        2. Apply fitted scaler (if trained).
        3. Apply fitted PCA reduction (if trained).
        4. Calculate Euclidean distance to all cluster centroids.
        5. Assign to nearest cluster.
        6. Retrieve nearest neighboring images from the training dataset.
        """
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            import io
            img = Image.open(io.BytesIO(image_input)).convert("RGB")

        # 1. Feature Extraction
        if feature_type == "resnet50":
            # Extract 2048-d GAP
            import torch
            t_img = self.feature_extractor.resnet_transform(img).unsqueeze(0).to(self.feature_extractor.device)
            model = self.feature_extractor.get_resnet50_model()
            with torch.no_grad():
                feat = model(t_img).cpu().numpy().astype(np.float32)
        elif feature_type == "color_hist":
            bins = (bins_per_channel, bins_per_channel, bins_per_channel)
            arr = np.array(img)
            hist, _ = np.histogramdd(arr.reshape(-1, 3), bins=bins, range=[(0, 256), (0, 256), (0, 256)])
            h_flat = hist.flatten().astype(np.float32)
            norm = np.sum(h_flat)
            if norm > 0:
                h_flat /= norm
            feat = h_flat.reshape(1, -1)
        elif feature_type == "combined":
            # ResNet + Hist
            import torch
            t_img = self.feature_extractor.resnet_transform(img).unsqueeze(0).to(self.feature_extractor.device)
            model = self.feature_extractor.get_resnet50_model()
            with torch.no_grad():
                f_cnn = model(t_img).cpu().numpy().astype(np.float32)
            bins = (bins_per_channel, bins_per_channel, bins_per_channel)
            arr = np.array(img)
            hist, _ = np.histogramdd(arr.reshape(-1, 3), bins=bins, range=[(0, 256), (0, 256), (0, 256)])
            h_flat = hist.flatten().astype(np.float32)
            norm = np.sum(h_flat)
            if norm > 0:
                h_flat /= norm
            f_hist = h_flat.reshape(1, -1)
            feat = np.hstack([f_cnn, f_hist]).astype(np.float32)
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}")

        # 2. Scaling
        transformed_feat = feat
        if self.scaler is not None and self.scaler.scaler is not None:
            try:
                transformed_feat = self.scaler.transform(transformed_feat)
            except Exception as e:
                print(f"Scaler warning: {e}")

        # 3. PCA Transformation
        final_embedding = transformed_feat
        if self.pca_manager is not None and self.pca_manager.pca is not None:
            try:
                final_embedding = self.pca_manager.transform(transformed_feat)
            except Exception as e:
                print(f"PCA transform warning: {e}")

        # 4. Nearest Cluster Centroid Calculation
        centroid_distances = []
        nearest_cluster_id = 0
        min_distance = 0.0

        if self.centroids is not None and len(self.centroids) > 0:
            dists = cdist(final_embedding, self.centroids, metric="euclidean").flatten()
            sorted_clusters = np.argsort(dists)
            nearest_cluster_id = int(sorted_clusters[0])
            min_distance = float(dists[nearest_cluster_id])

            for c_id in sorted_clusters:
                centroid_distances.append({
                    "Cluster_ID": int(c_id),
                    "Distance": round(float(dists[c_id]), 4),
                    "Relative_Affinity": round(1.0 / (1.0 + float(dists[c_id])), 4)
                })

        # 5. Visual Neighbor Retrieval
        similar_images = []
        if self.dataset_features is not None and self.df_manifest is not None and not self.df_manifest.empty:
            # Pairwise distance to all images in dataset
            sample_dists = cdist(final_embedding, self.dataset_features, metric="euclidean").flatten()
            top_indices = np.argsort(sample_dists)[:top_k_neighbors]

            for rank, idx in enumerate(top_indices, start=1):
                row = self.df_manifest.iloc[idx]
                similar_images.append({
                    "Rank": rank,
                    "Image_ID": str(row["image_id"]),
                    "File_Path": str(row["file_path"]),
                    "Distance": round(float(sample_dists[idx]), 4)
                })

        return {
            "nearest_cluster_id": nearest_cluster_id,
            "min_centroid_distance": round(min_distance, 4),
            "cluster_distance_ranking": pd.DataFrame(centroid_distances),
            "similar_images": pd.DataFrame(similar_images),
            "embedding_shape": final_embedding.shape,
            "feature_vector": final_embedding
        }
