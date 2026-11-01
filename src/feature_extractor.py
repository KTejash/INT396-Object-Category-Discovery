"""
Feature Extractor Module for Object Discovery Lab
Extracts:
1. Deep CNN Features (Pretrained ResNet50 GAP 2048-d)
2. 3D Color Histograms (Configurable Bins, L1/L2 Normalized)
3. Combined Representations (Weighted ResNet50 + Histogram Concatenation)
Supports batch inference, progress callbacks, and on-disk/memory caching.
"""

import sys
import time
from pathlib import Path
from typing import List, Tuple, Optional, Callable, Dict, Any
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BASE_DIR,
    EMBEDDINGS_DIR,
    CNN_FEATURES_RAW,
    HIST_FEATURES_RAW,
    IMAGE_IDS_FILE,
    BATCH_SIZE
)

_CACHED_RESNET_MODEL = None


def get_resnet50_model(device: Optional[torch.device] = None) -> nn.Module:
    """Loads and caches pretrained ResNet50 with classification head replaced by Identity."""
    global _CACHED_RESNET_MODEL
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if _CACHED_RESNET_MODEL is None:
        weights = models.ResNet50_Weights.DEFAULT
        model = models.resnet50(weights=weights)
        model.fc = nn.Identity()  # 2048-d GAP output
        model = model.to(device)
        model.eval()
        _CACHED_RESNET_MODEL = model
    return _CACHED_RESNET_MODEL


class PathDataset(torch.utils.data.Dataset):
    """PyTorch dataset that loads and transforms images directly from absolute file paths."""
    def __init__(self, file_paths: List[Path], transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int):
        p = self.file_paths[idx]
        with Image.open(p) as img:
            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, idx


class FeatureExtractor:
    """
    Unified Feature Extraction engine supporting ResNet50, Color Histograms, and Combined spaces.
    """

    def __init__(self, device: Optional[str] = None):
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)
            
        self.weights = models.ResNet50_Weights.DEFAULT
        self.resnet_transform = self.weights.transforms()

    def extract_resnet50(
        self,
        image_paths: List[Path],
        batch_size: int = BATCH_SIZE,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> np.ndarray:
        """
        Extracts 2048-dimensional features via ResNet50 GAP layer.
        """
        model = get_resnet50_model(self.device)
        dataset = PathDataset(image_paths, transform=self.resnet_transform)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=False, num_workers=0
        )

        features = []
        total = len(dataset)
        processed = 0

        with torch.no_grad():
            for i, (imgs, _) in enumerate(dataloader):
                imgs = imgs.to(self.device)
                feats = model(imgs)  # shape (B, 2048)
                features.append(feats.cpu().numpy())
                
                processed += len(imgs)
                if progress_callback:
                    pct = processed / total
                    progress_callback(pct, f"ResNet50: {processed}/{total} images extracted")

        feats_matrix = np.vstack(features).astype(np.float32)
        return feats_matrix

    def extract_color_histograms(
        self,
        image_paths: List[Path],
        bins_per_channel: int = 8,
        normalization: str = "l1",
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> np.ndarray:
        """
        Extracts 3D RGB color histograms.
        bins_per_channel^3 dimensions (e.g. 4^3=64, 8^3=512, 16^3=4096).
        """
        histograms = []
        total = len(image_paths)
        bins = (bins_per_channel, bins_per_channel, bins_per_channel)

        for i, p in enumerate(image_paths):
            with Image.open(p) as img:
                arr = np.array(img.convert("RGB"))

            hist, _ = np.histogramdd(
                arr.reshape(-1, 3),
                bins=bins,
                range=[(0, 256), (0, 256), (0, 256)]
            )
            h_flat = hist.flatten().astype(np.float32)

            if normalization == "l1":
                norm = np.sum(h_flat)
                if norm > 0:
                    h_flat /= norm
            elif normalization == "l2":
                norm = np.linalg.norm(h_flat)
                if norm > 0:
                    h_flat /= norm

            histograms.append(h_flat)

            if progress_callback and ((i + 1) % 50 == 0 or (i + 1) == total):
                progress_callback((i + 1) / total, f"Color Histogram: {i + 1}/{total} images extracted")

        hist_matrix = np.vstack(histograms).astype(np.float32)
        return hist_matrix

    def combine_features(
        self,
        cnn_features: np.ndarray,
        hist_features: np.ndarray,
        cnn_weight: float = 1.0,
        hist_weight: float = 1.0
    ) -> np.ndarray:
        """
        Combines semantic CNN features with Color Histograms with configurable weighting.
        """
        w_cnn = cnn_features * float(cnn_weight)
        w_hist = hist_features * float(hist_weight)
        combined = np.hstack([w_cnn, w_hist]).astype(np.float32)
        return combined

    def extract_features_for_manifest(
        self,
        df_manifest: pd.DataFrame,
        feature_type: str = "resnet50",
        bins_per_channel: int = 8,
        batch_size: int = BATCH_SIZE,
        cnn_weight: float = 1.0,
        hist_weight: float = 1.0,
        progress_callback: Optional[Callable[[float, str], None]] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extracts features for all images listed in a manifest DataFrame.
        Returns:
            (features_matrix, image_ids)
        """
        if df_manifest.empty:
            raise ValueError("Manifest is empty. No images to extract features from.")

        image_ids = df_manifest["image_id"].values
        abs_paths = [PROJECT_ROOT / str(p) for p in df_manifest["file_path"]]

        if feature_type == "resnet50":
            feats = self.extract_resnet50(abs_paths, batch_size=batch_size, progress_callback=progress_callback)
        elif feature_type == "color_hist":
            feats = self.extract_color_histograms(abs_paths, bins_per_channel=bins_per_channel, progress_callback=progress_callback)
        elif feature_type == "combined":
            if progress_callback:
                progress_callback(0.0, "Starting dual feature extraction...")
            f_cnn = self.extract_resnet50(abs_paths, batch_size=batch_size, progress_callback=lambda p, m: progress_callback(p * 0.5, m))
            f_hist = self.extract_color_histograms(abs_paths, bins_per_channel=bins_per_channel, progress_callback=lambda p, m: progress_callback(0.5 + p * 0.5, m))
            feats = self.combine_features(f_cnn, f_hist, cnn_weight=cnn_weight, hist_weight=hist_weight)
        else:
            raise ValueError(f"Unknown feature_type: {feature_type}. Choose 'resnet50', 'color_hist', or 'combined'.")

        return feats, image_ids
