"""
VisionCluster AI Configuration Module
Defines project directory hierarchy, supported image formats,
cache paths, and experiment defaults for dual dataset management.
"""

from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Main directories
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
METADATA_DIR = DATA_DIR / "metadata"

# Dataset sources
OPEN_IMAGES_DIR = RAW_DATA_DIR / "open_images"
CUSTOM_UPLOAD_DIR = DATA_DIR / "custom_uploads"

# Processing directories
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Output directories
OUTPUT_DIR = PROJECT_ROOT / "outputs"
CACHE_DIR = OUTPUT_DIR / "cache"
FEATURE_CACHE_DIR = CACHE_DIR / "features"

EXPERIMENTS_DIR = OUTPUT_DIR / "experiments"
FIGURES_DIR = OUTPUT_DIR / "figures"
METRICS_DIR = OUTPUT_DIR / "metrics"
CLUSTER_DIR = OUTPUT_DIR / "cluster_assignments"

REPORTS_DIR = PROJECT_ROOT / "reports"

# Supported image formats
SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".webp",
    ".bmp"
}

# Initial experiment defaults
DEFAULT_CONFIG = {
    "model": "resnet50",
    "feature_dimensions": 2048,
    "pca_components": 100,
    "n_clusters": 8,
    "random_state": 42,
    "batch_size": 32,
}

# Create required directories safely
REQUIRED_DIRECTORIES = [
    DATA_DIR,
    RAW_DATA_DIR,
    METADATA_DIR,
    OPEN_IMAGES_DIR,
    CUSTOM_UPLOAD_DIR,
    PROCESSED_DATA_DIR,
    OUTPUT_DIR,
    CACHE_DIR,
    FEATURE_CACHE_DIR,
    EXPERIMENTS_DIR,
    FIGURES_DIR,
    METRICS_DIR,
    CLUSTER_DIR,
    REPORTS_DIR,
]


def create_project_directories():
    """Ensures all necessary project directories exist safely on disk."""
    for directory in REQUIRED_DIRECTORIES:
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    create_project_directories()
    print("VisionCluster AI directories initialized.")
    print(f"Project root: {PROJECT_ROOT}")
