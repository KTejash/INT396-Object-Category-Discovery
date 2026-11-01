"""
Feature Extraction and Scaling Module
Extracts 2048-d Deep CNN features using Pretrained ResNet50 (GAP) and
512-d normalized 3D Color Histogram features. Performs StandardScaler normalization.
"""

import sys
import time
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.preprocessing import StandardScaler
import joblib

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    MANIFEST_FILE,
    EMBEDDINGS_DIR,
    MODELS_DIR,
    METRICS_DIR,
    CNN_FEATURES_RAW,
    CNN_FEATURES_SCALED,
    HIST_FEATURES_RAW,
    HIST_FEATURES_SCALED,
    IMAGE_IDS_FILE,
    SCALER_MODEL_PATH,
    BATCH_SIZE
)


class ImageListDataset(torch.utils.data.Dataset):
    """PyTorch Dataset for batch image feature extraction."""
    def __init__(self, file_paths, transform=None):
        self.file_paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        with Image.open(path) as img:
            img = img.convert("RGB")
            if self.transform:
                img = self.transform(img)
            return img, idx


def extract_cnn_features(manifest_path=MANIFEST_FILE, batch_size=BATCH_SIZE):
    """
    Extracts 2048-dimensional Global Average Pooling features using pretrained ResNet50.
    """
    print("=" * 65)
    print("Extracting Deep CNN Features (ResNet50 GAP 2048-d)...")

    df_manifest = pd.read_csv(manifest_path)
    file_paths = [PROJECT_ROOT / rel_path for rel_path in df_manifest["file_path"]]
    image_ids = df_manifest["image_id"].values

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Official ResNet50 ImageNet preprocessing pipeline
    weights = models.ResNet50_Weights.DEFAULT
    transform = weights.transforms()

    model = models.resnet50(weights=weights)
    model.fc = nn.Identity()  # Remove classification head, keep 2048-d GAP
    model = model.to(device)
    model.eval()

    dataset = ImageListDataset(file_paths, transform=transform)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False, num_workers=0
    )

    features = []
    t0 = time.time()
    total_samples = len(dataset)

    with torch.no_grad():
        for i, (imgs, indices) in enumerate(dataloader):
            imgs = imgs.to(device)
            feats = model(imgs)  # Shape: (B, 2048)
            features.append(feats.cpu().numpy())

            if (i + 1) % 10 == 0 or (i + 1) == len(dataloader):
                processed = min((i + 1) * batch_size, total_samples)
                pct = (processed / total_samples) * 100
                print(f"  Processed {processed}/{total_samples} ({pct:.1f}%) images...")

    features_matrix = np.vstack(features).astype(np.float32)
    elapsed = time.time() - t0
    print(f"CNN extraction complete in {elapsed:.1f}s. Matrix shape: {features_matrix.shape}")

    # Save to disk
    np.save(CNN_FEATURES_RAW, features_matrix)
    np.save(IMAGE_IDS_FILE, image_ids)
    print(f"Saved raw CNN features -> {CNN_FEATURES_RAW}")
    print(f"Saved image IDs         -> {IMAGE_IDS_FILE}")

    return features_matrix, image_ids


def extract_color_histogram_features(manifest_path=MANIFEST_FILE, bins=(8, 8, 8)):
    """
    Extracts 3D normalized color histogram (8x8x8 = 512 dimensions) in RGB space.
    """
    print("=" * 65)
    print(f"Extracting 3D Color Histogram Features ({bins[0]}x{bins[1]}x{bins[2]} = {np.prod(bins)}-d)...")

    df_manifest = pd.read_csv(manifest_path)
    file_paths = [PROJECT_ROOT / rel_path for rel_path in df_manifest["file_path"]]

    hist_features = []
    t0 = time.time()
    total = len(file_paths)

    for i, path in enumerate(file_paths):
        with Image.open(path) as img:
            img = img.convert("RGB")
            arr = np.array(img)

        # 3D RGB Histogram with 8 bins per channel [0-256)
        hist, _ = np.histogramdd(
            arr.reshape(-1, 3),
            bins=bins,
            range=[(0, 256), (0, 256), (0, 256)]
        )
        hist_flat = hist.flatten().astype(np.float32)
        # L1 Normalization (so total sum equals 1.0)
        norm = np.sum(hist_flat)
        if norm > 0:
            hist_flat /= norm
        hist_features.append(hist_flat)

        if (i + 1) % 500 == 0 or (i + 1) == total:
            print(f"  Extracted histogram for {i + 1}/{total} images...")

    hist_matrix = np.vstack(hist_features).astype(np.float32)
    elapsed = time.time() - t0
    print(f"Histogram extraction complete in {elapsed:.1f}s. Matrix shape: {hist_matrix.shape}")

    np.save(HIST_FEATURES_RAW, hist_matrix)
    print(f"Saved raw Histogram features -> {HIST_FEATURES_RAW}")

    return hist_matrix


def scale_and_standardize_features():
    """
    Applies StandardScaler to both CNN and Histogram features.
    Computes distribution diagnostics before and after scaling.
    """
    print("=" * 65)
    print("Applying StandardScaler Normalization...")

    cnn_raw = np.load(CNN_FEATURES_RAW)
    hist_raw = np.load(HIST_FEATURES_RAW)

    # 1. Standardize CNN features
    scaler_cnn = StandardScaler()
    cnn_scaled = scaler_cnn.fit_transform(cnn_raw).astype(np.float32)
    np.save(CNN_FEATURES_SCALED, cnn_scaled)
    joblib.dump(scaler_cnn, SCALER_MODEL_PATH)

    # 2. Standardize Histogram features
    scaler_hist = StandardScaler()
    hist_scaled = scaler_hist.fit_transform(hist_raw).astype(np.float32)
    np.save(HIST_FEATURES_SCALED, hist_scaled)

    # Compute Feature Distribution Diagnostics Table
    stats = [
        {
            "Representation": "CNN Features (Raw)",
            "Dimensions": cnn_raw.shape[1],
            "Mean (Overall)": round(float(np.mean(cnn_raw)), 4),
            "Std (Overall)": round(float(np.std(cnn_raw)), 4),
            "Min": round(float(np.min(cnn_raw)), 4),
            "Max": round(float(np.max(cnn_raw)), 4),
        },
        {
            "Representation": "CNN Features (Standardized)",
            "Dimensions": cnn_scaled.shape[1],
            "Mean (Overall)": round(float(np.mean(cnn_scaled)), 4),
            "Std (Overall)": round(float(np.std(cnn_scaled)), 4),
            "Min": round(float(np.min(cnn_scaled)), 4),
            "Max": round(float(np.max(cnn_scaled)), 4),
        },
        {
            "Representation": "Color Histograms (Raw)",
            "Dimensions": hist_raw.shape[1],
            "Mean (Overall)": round(float(np.mean(hist_raw)), 6),
            "Std (Overall)": round(float(np.std(hist_raw)), 6),
            "Min": round(float(np.min(hist_raw)), 4),
            "Max": round(float(np.max(hist_raw)), 4),
        },
        {
            "Representation": "Color Histograms (Standardized)",
            "Dimensions": hist_scaled.shape[1],
            "Mean (Overall)": round(float(np.mean(hist_scaled)), 4),
            "Std (Overall)": round(float(np.std(hist_scaled)), 4),
            "Min": round(float(np.min(hist_scaled)), 4),
            "Max": round(float(np.max(hist_scaled)), 4),
        }
    ]

    df_stats = pd.DataFrame(stats)
    summary_csv = METRICS_DIR / "feature_scaling_summary.csv"
    df_stats.to_csv(summary_csv, index=False)

    print("\nFeature Scaling Summary Table:")
    print(df_stats.to_string(index=False))

    return cnn_scaled, hist_scaled, df_stats


def run_feature_pipeline():
    extract_cnn_features()
    extract_color_histogram_features()
    scale_and_standardize_features()


if __name__ == "__main__":
    run_feature_pipeline()
