"""
Image Discovery Studio & Real-Time Prediction Engine
Predicts the object category / cluster placement of any user-uploaded image,
computes confidence scores against dataset classes via distance-weighted k-NN voting,
evaluates cluster centroid proximity, and retrieves visual nearest neighbors.
"""

import io
import sys
from pathlib import Path
from typing import Tuple, Dict, Any, List, Optional, Union
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
    Inference, prediction, and visual similarity retrieval engine for arbitrary user-provided images.
    """

    def __init__(
        self,
        feature_extractor: Optional[FeatureExtractor] = None,
        scaler: Optional[PreprocessingPipeline] = None,
        pca_manager: Optional[PCAManager] = None,
        centroids: Optional[np.ndarray] = None,
        dataset_features: Optional[np.ndarray] = None,
        df_manifest: Optional[pd.DataFrame] = None,
        df_ground_truth: Optional[pd.DataFrame] = None,
        cluster_labels: Optional[np.ndarray] = None
    ):
        self.feature_extractor = feature_extractor or FeatureExtractor()
        self.scaler = scaler
        self.pca_manager = pca_manager
        self.centroids = centroids
        self.dataset_features = dataset_features
        self.df_manifest = df_manifest
        self.df_ground_truth = df_ground_truth
        self.cluster_labels = cluster_labels

        # Build label lookup map if ground truth is available
        self.id_to_label: Dict[str, str] = {}
        if self.df_ground_truth is not None and not self.df_ground_truth.empty:
            for _, row in self.df_ground_truth.iterrows():
                self.id_to_label[str(row["image_id"])] = str(row["category"])

        # Precompute dominant category and purity per cluster if cluster_labels exist
        self.cluster_dominant_info: Dict[int, Dict[str, Any]] = {}
        if self.cluster_labels is not None and self.df_manifest is not None and self.id_to_label:
            self._compute_cluster_purity_mapping()

    def _compute_cluster_purity_mapping(self):
        """Calculates dominant category and purity score for each discovered cluster."""
        if self.cluster_labels is None or self.df_manifest is None:
            return

        manifest_ids = [str(x) for x in self.df_manifest["image_id"].values]
        unique_clusters = np.unique(self.cluster_labels)

        for c_id in unique_clusters:
            mask = (self.cluster_labels == c_id)
            c_ids = [manifest_ids[i] for i, m in enumerate(mask) if m and i < len(manifest_ids)]
            labels_in_cluster = [self.id_to_label[img_id] for img_id in c_ids if img_id in self.id_to_label]

            if labels_in_cluster:
                s = pd.Series(labels_in_cluster)
                dominant = s.value_counts().index[0]
                count = s.value_counts().iloc[0]
                total = len(labels_in_cluster)
                purity = count / total
                self.cluster_dominant_info[int(c_id)] = {
                    "dominant_category": dominant,
                    "purity": round(float(purity), 4),
                    "dominant_count": int(count),
                    "total_labeled": total
                }
            else:
                self.cluster_dominant_info[int(c_id)] = {
                    "dominant_category": f"Cluster {c_id}",
                    "purity": 1.0,
                    "dominant_count": int(np.sum(mask)),
                    "total_labeled": int(np.sum(mask))
                }

    def predict_image(
        self,
        image_input: Union[str, Path, Image.Image, bytes],
        feature_type: str = "resnet50",
        bins_per_channel: int = 8,
        top_k_neighbors: int = 7
    ) -> Dict[str, Any]:
        """
        Executes end-to-end inference and predicts object category according to the dataset:
        1. Preprocesses image and extracts visual features.
        2. Applies fitted scaler and PCA transformations.
        3. Identifies nearest cluster centroid and centroid distance.
        4. Predicts semantic object category via distance-weighted k-NN voting over dataset embeddings.
        5. Computes probability breakdown across all dataset classes.
        6. Retrieves visual nearest neighbors with thumbnails and similarity scores.
        """
        if isinstance(image_input, (str, Path)):
            img = Image.open(image_input).convert("RGB")
        elif isinstance(image_input, Image.Image):
            img = image_input.convert("RGB")
        else:
            img = Image.open(io.BytesIO(image_input)).convert("RGB")

        # 1. Feature Extraction
        if feature_type == "resnet50":
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
                print(f"Scaler notice: {e}")

        # 3. PCA Transformation
        final_embedding = transformed_feat
        if self.pca_manager is not None and self.pca_manager.pca is not None:
            try:
                final_embedding = self.pca_manager.transform(transformed_feat)
            except Exception as e:
                print(f"PCA notice: {e}")

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
                c_int = int(c_id)
                dom_cat = self.cluster_dominant_info.get(c_int, {}).get("dominant_category", f"Cluster {c_int}")
                purity = self.cluster_dominant_info.get(c_int, {}).get("purity", 1.0)
                centroid_distances.append({
                    "Cluster_ID": c_int,
                    "Dominant_Category": dom_cat,
                    "Cluster_Purity": f"{purity * 100:.1f}%",
                    "Centroid_Distance": round(float(dists[c_int]), 4),
                    "Affinity": round(1.0 / (1.0 + float(dists[c_int])), 4)
                })

        # 5. Visual Neighbor Retrieval & Category Prediction via k-NN
        similar_images = []
        category_votes: Dict[str, float] = {}
        predicted_category = "Unassigned"
        prediction_confidence = 0.0
        probabilities_df = pd.DataFrame(columns=["Category", "Confidence_Pct"])

        if self.dataset_features is not None and self.df_manifest is not None and not self.df_manifest.empty:
            sample_dists = cdist(final_embedding, self.dataset_features, metric="euclidean").flatten()
            safe_k = min(int(top_k_neighbors), len(sample_dists))
            top_indices = np.argsort(sample_dists)[:safe_k]

            for rank, idx in enumerate(top_indices, start=1):
                row = self.df_manifest.iloc[idx]
                img_id = str(row["image_id"])
                file_p = str(row["file_path"])
                dist_val = float(sample_dists[idx])
                sim_pct = round((1.0 / (1.0 + dist_val)) * 100, 1)

                cat_label = self.id_to_label.get(img_id, f"Cluster {self.cluster_labels[idx] if self.cluster_labels is not None else 'Unknown'}")

                similar_images.append({
                    "Rank": rank,
                    "Image_ID": img_id,
                    "File_Path": file_p,
                    "Category": cat_label,
                    "Distance": round(dist_val, 4),
                    "Similarity_Pct": f"{sim_pct}%"
                })

                # Distance-weighted voting
                weight = 1.0 / (dist_val + 1e-5)
                category_votes[cat_label] = category_votes.get(cat_label, 0.0) + weight

            # Calculate prediction confidence percentages
            total_vote = sum(category_votes.values())
            if total_vote > 0:
                prob_records = []
                for cat, v in sorted(category_votes.items(), key=lambda x: x[1], reverse=True):
                    pct = round((v / total_vote) * 100, 2)
                    prob_records.append({"Category": cat, "Confidence_Pct": pct})

                probabilities_df = pd.DataFrame(prob_records)
                predicted_category = str(probabilities_df.iloc[0]["Category"])
                prediction_confidence = float(probabilities_df.iloc[0]["Confidence_Pct"])
            elif nearest_cluster_id in self.cluster_dominant_info:
                predicted_category = self.cluster_dominant_info[nearest_cluster_id]["dominant_category"]
                prediction_confidence = round(self.cluster_dominant_info[nearest_cluster_id]["purity"] * 100, 1)
        elif nearest_cluster_id in self.cluster_dominant_info:
            predicted_category = self.cluster_dominant_info[nearest_cluster_id]["dominant_category"]
            prediction_confidence = round(self.cluster_dominant_info[nearest_cluster_id]["purity"] * 100, 1)

        return {
            "predicted_category": predicted_category,
            "prediction_confidence": prediction_confidence,
            "probabilities_df": probabilities_df,
            "nearest_cluster_id": nearest_cluster_id,
            "min_centroid_distance": round(min_distance, 4),
            "cluster_dominant_info": self.cluster_dominant_info.get(nearest_cluster_id, {}),
            "cluster_distance_ranking": pd.DataFrame(centroid_distances),
            "similar_images": pd.DataFrame(similar_images),
            "embedding_shape": final_embedding.shape,
            "feature_vector": final_embedding
        }

    def process_and_assign(
        self,
        image_input: Union[str, Path, Image.Image, bytes],
        feature_type: str = "resnet50",
        bins_per_channel: int = 8,
        top_k_neighbors: int = 6
    ) -> Dict[str, Any]:
        """
        Backward-compatible wrapper for existing pipeline calls.
        """
        return self.predict_image(
            image_input=image_input,
            feature_type=feature_type,
            bins_per_channel=bins_per_channel,
            top_k_neighbors=top_k_neighbors
        )
