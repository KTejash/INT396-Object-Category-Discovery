"""
Project Configuration Module
Centralized configuration parameters for reproducibility across data acquisition,
feature extraction, dimensionality reduction, clustering, evaluation, and dashboard.
"""

from pathlib import Path
import os

# Root Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_IMG_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
METADATA_DIR = DATA_DIR / "metadata"

OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
MODELS_DIR = OUTPUTS_DIR / "models"
EMBEDDINGS_DIR = OUTPUTS_DIR / "embeddings"
CLUSTERS_DIR = OUTPUTS_DIR / "cluster_assignments"
REPORTS_DIR = BASE_DIR / "reports"

# Ensure essential output directories exist
for p in [RAW_IMG_DIR, PROCESSED_DIR, METADATA_DIR, FIGURES_DIR, METRICS_DIR,
          MODELS_DIR, EMBEDDINGS_DIR, CLUSTERS_DIR, REPORTS_DIR]:
    p.mkdir(parents=True, exist_ok=True)

# Random Seeds
RANDOM_SEED = 42
STABILITY_SEEDS = [21, 42, 100, 123]

# Open Images Target Categories and Class Identifiers
# Open Images V5/V6/V7 MID mappings
TARGET_CLASSES = {
    "/m/01yrx": "Cat",
    "/m/0bt9lr": "Dog",
    "/m/0k4j": "Car",
    "/m/01bjv": "Bus",
    "/m/0c9ph5": "Flower",
    "/m/0cgh4": "Building",
    "/m/019jd": "Boat",
    "/m/015p6": "Bird",
}

CATEGORY_NAMES = sorted(list(TARGET_CLASSES.values()))

# Sampling Configuration
# Bus has ~355 images across validation and test splits.
# Setting TARGET_PER_CLASS to 300 gives 2,400 balanced images (or 320 for 2,560 images).
TARGET_PER_CLASS = 300
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 32

# Feature Extraction Settings
CNN_FEATURE_DIM = 2048  # ResNet50 Global Average Pooling
COLOR_HIST_BINS = (8, 8, 8)  # 512-dimensional normalized 3D RGB/HSV histogram

# Dimensionality Reduction Parameters
PCA_VARIANCE_THRESHOLDS = [0.90, 0.95]
TSNE_PERPLEXITIES = [5, 30, 50]
TSNE_LEARNING_RATES = [100, 200]
TSNE_ITERATIONS = 1000

# Clustering Parameters
K_RANGE = list(range(2, 13))
DEFAULT_K = 8

# Metadata & File Names
MANIFEST_FILE = METADATA_DIR / "image_manifest.csv"
GROUND_TRUTH_FILE = METADATA_DIR / "ground_truth.csv"
DATA_QUALITY_FILE = METRICS_DIR / "data_quality_report.csv"

# Precomputed Feature Matrix Paths
CNN_FEATURES_RAW = EMBEDDINGS_DIR / "cnn_features_raw.npy"
CNN_FEATURES_SCALED = EMBEDDINGS_DIR / "cnn_features_scaled.npy"
HIST_FEATURES_RAW = EMBEDDINGS_DIR / "hist_features_raw.npy"
HIST_FEATURES_SCALED = EMBEDDINGS_DIR / "hist_features_scaled.npy"
IMAGE_IDS_FILE = EMBEDDINGS_DIR / "image_ids.npy"
SCALER_MODEL_PATH = MODELS_DIR / "standard_scaler.joblib"
PCA_MODEL_90_PATH = MODELS_DIR / "pca_90.joblib"
PCA_MODEL_95_PATH = MODELS_DIR / "pca_95.joblib"
KMEANS_BEST_MODEL_PATH = MODELS_DIR / "kmeans_best.joblib"
