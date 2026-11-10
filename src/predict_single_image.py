"""
Standalone CLI Prediction Tool for VisionCluster AI
Allows users to classify any image from the command line using
the fitted unsupervised feature pipeline, centroid proximity, and k-NN dataset voting.

Usage:
    python src/predict_single_image.py --image path/to/image.jpg
"""

import sys
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_manager import DatasetManager
from src.feature_extractor import FeatureExtractor
from src.preprocessing import PreprocessingPipeline
from src.dimensionality_reduction import PCAManager
from src.clustering import ClusteringEngine
from src.image_search import ImageDiscoveryStudio
from src.config import get_category_display


def parse_args():
    parser = argparse.ArgumentParser(description="Predict object category of an image using VisionCluster AI.")
    parser.add_argument("--image", type=str, required=True, help="Path to the image file to analyze.")
    parser.add_argument("--dataset", type=str, default="open_images_benchmark", help="Dataset name to query against.")
    parser.add_argument("--feature_type", type=str, default="resnet50", choices=["resnet50", "color_hist", "combined"])
    parser.add_argument("--top_k", type=int, default=5, help="Number of visual nearest neighbors to retrieve.")
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = Path(args.image)

    if not image_path.exists():
        print(f"[ERROR] Specified image path does not exist: {image_path}")
        sys.exit(1)

    print("=" * 65)
    print("      VISIONCLUSTER AI — COMMAND-LINE IMAGE PREDICTOR")
    print("=" * 65)
    print(f"Target Image:  {image_path.name}")
    print(f"Feature Space: {args.feature_type.upper()}")
    print("-" * 65)

    # 1. Load dataset manager and manifest
    dm = DatasetManager(dataset_name=args.dataset)
    df_manifest = dm.load_manifest()
    df_gt = dm.load_ground_truth()

    if df_manifest.empty:
        print(f"[ERROR] No images found in manifest for dataset '{args.dataset}'.")
        sys.exit(1)

    print(f"[INFO] Loaded manifest with {len(df_manifest)} reference images.")

    # 2. Extract features for dataset or load cached
    extractor = FeatureExtractor()
    abs_paths = [PROJECT_ROOT / str(p) for p in df_manifest["file_path"]]

    print(f"[INFO] Extracting features for reference dataset...")
    feats, _ = extractor.extract_features_for_manifest(
        df_manifest,
        feature_type=args.feature_type,
        batch_size=32
    )

    # 3. Fit scaler and PCA
    scaler = PreprocessingPipeline(scaler_type="standard")
    feats_scaled, _ = scaler.fit_transform(feats)

    pca = PCAManager()
    feats_pca, _ = pca.fit_transform(feats_scaled, mode="variance", target_variance=0.95)

    # 4. Fit K-Means
    k_clusters = len(df_gt["category"].unique()) if df_gt is not None else 8
    clustering = ClusteringEngine()
    labels, metrics = clustering.fit_predict(feats_pca, k=k_clusters)

    # 5. Initialize studio and predict
    studio = ImageDiscoveryStudio(
        feature_extractor=extractor,
        scaler=scaler,
        pca_manager=pca,
        centroids=clustering.centroids_,
        dataset_features=feats_pca,
        df_manifest=df_manifest,
        df_ground_truth=df_gt,
        cluster_labels=labels
    )

    print(f"[INFO] Running inference on query image...")
    results = studio.predict_image(image_path, feature_type=args.feature_type, top_k_neighbors=args.top_k)

    pred_cat = results["predicted_category"]
    pred_conf = results["prediction_confidence"]
    cluster_id = results["nearest_cluster_id"]
    centroid_dist = results["min_centroid_distance"]

    print("\n" + "=" * 65)
    print("                     PREDICTION RESULTS")
    print("=" * 65)
    print(f"Predicted Category:   {get_category_display(pred_cat)}")
    print(f"Confidence Score:     {pred_conf:.1f}%")
    print(f"Assigned Cluster:     Cluster #{cluster_id} (Centroid distance: {centroid_dist:.4f})")
    print("-" * 65)

    df_probs = results["probabilities_df"]
    if not df_probs.empty:
        print("\nTop Category Probabilities:")
        for _, row in df_probs.head(4).iterrows():
            bar = "█" * int(row["Confidence_Pct"] // 5)
            print(f"  {row['Category']:15} : {row['Confidence_Pct']:5.1f}% | {bar}")

    df_sim = results["similar_images"]
    if not df_sim.empty:
        print("\nTop Visually Similar Reference Matches:")
        for _, row in df_sim.head(args.top_k).iterrows():
            print(f"  Rank #{row['Rank']}: {row['Image_ID']} | {row.get('Category', 'N/A'):10} | Sim: {row.get('Similarity_Pct', 'N/A')}")

    print("=" * 65)


if __name__ == "__main__":
    main()
