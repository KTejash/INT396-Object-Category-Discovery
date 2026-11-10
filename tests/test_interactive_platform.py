"""
Comprehensive Automated Test Suite for Object Discovery Lab
Verifies custom dataset ingestion, corrupt image handling, feature extraction,
preprocessing, PCA, t-SNE, K-Means clustering, visual search, and experiment logging.
"""

import sys
import io
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_manager import DatasetManager
from src.feature_extractor import FeatureExtractor
from src.preprocessing import PreprocessingPipeline
from src.dimensionality_reduction import PCAManager, TSNEManager
from src.clustering import ClusteringEngine
from src.image_search import ImageDiscoveryStudio
from src.experiment_manager import ExperimentManager
from src.evaluation import ExternalValidator


class MockUpload:
    def __init__(self, name: str, data: bytes):
        self.name = name
        self.data = data

    def read(self):
        return self.data


def test_end_to_end_platform():
    print("=" * 70)
    print("RUNNING AUTOMATED TEST SUITE: Object Discovery Lab")
    print("=" * 70)

    # 1. Test Dataset Workspace with synthetic & corrupted images
    print("\n[TEST 1] Testing Dataset Ingestion & Corruption Filtering...")
    mock_files = []
    
    # Generate 6 valid synthetic test images
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
    for i, c in enumerate(colors):
        im = Image.new("RGB", (64, 64), color=c)
        buf = io.BytesIO()
        im.save(buf, format="JPEG")
        mock_files.append(MockUpload(f"test_{i}.jpg", buf.getvalue()))

    # Add 1 duplicate image
    mock_files.append(MockUpload("test_dup.jpg", mock_files[0].data))

    # Add 1 corrupt image
    mock_files.append(MockUpload("corrupt.jpg", b"This is not a real image header"))

    dm = DatasetManager("test_suite_dataset")
    valid_c, corrupt_c, msg = dm.create_custom_dataset_from_files("test_suite_dataset", mock_files)
    print(f"  Ingestion Result: {valid_c} valid, {corrupt_c} corrupt. Message: {msg}")
    assert valid_c == 6, f"Expected 6 valid images, got {valid_c}"
    assert corrupt_c == 1, f"Expected 1 corrupt image, got {corrupt_c}"

    df_m = dm.load_manifest()
    assert len(df_m) == 6, f"Manifest length mismatch: {len(df_m)}"
    assert "label" not in [c.lower() for c in df_m.columns], "Label leakage in manifest!"
    print("  [PASS] Dataset creation, duplicate detection, and label isolation verified.")

    # 2. Test Feature Extraction
    print("\n[TEST 2] Testing Feature Extraction (ResNet50, Color Hist, Combined)...")
    extractor = FeatureExtractor()
    
    # Test Color Histogram
    f_hist, ids = extractor.extract_features_for_manifest(df_m, feature_type="color_hist", bins_per_channel=4)
    assert f_hist.shape == (6, 64), f"Histogram shape mismatch: {f_hist.shape}"
    print(f"  [PASS] Color Histogram (4 bins) extracted: {f_hist.shape}")

    # Test ResNet50
    f_cnn, ids = extractor.extract_features_for_manifest(df_m, feature_type="resnet50", batch_size=4)
    assert f_cnn.shape == (6, 2048), f"ResNet50 shape mismatch: {f_cnn.shape}"
    print(f"  [PASS] ResNet50 GAP extracted: {f_cnn.shape}")

    # Test Combined
    f_comb, ids = extractor.extract_features_for_manifest(df_m, feature_type="combined", bins_per_channel=4)
    assert f_comb.shape == (6, 2048 + 64), f"Combined shape mismatch: {f_comb.shape}"
    print(f"  [PASS] Combined feature matrix extracted: {f_comb.shape}")

    # 3. Test Preprocessing Pipeline
    print("\n[TEST 3] Testing Preprocessing Transformations...")
    scaler = PreprocessingPipeline(scaler_type="standard")
    X_scaled, stats_df = scaler.fit_transform(f_cnn)
    assert X_scaled.shape == (6, 2048)
    assert np.allclose(np.mean(X_scaled, axis=0), 0, atol=1e-5), "StandardScaler mean not zero"
    print("  [PASS] StandardScaler zero-mean centering verified.")

    # 4. Test PCA Exploration
    print("\n[TEST 4] Testing PCA Decomposition...")
    pca_mgr = PCAManager(random_state=42)
    X_pca, pca_meta = pca_mgr.fit_transform(X_scaled, mode="fixed", n_components=3)
    assert X_pca.shape == (6, 3), f"PCA shape mismatch: {X_pca.shape}"
    assert "scree_df" in pca_meta
    print(f"  [PASS] PCA 3-component projection verified: {X_pca.shape} (Explained Var: {pca_meta['metrics']['explained_variance_total']})")

    # 5. Test t-SNE Embedding
    print("\n[TEST 5] Testing t-SNE Manifold Projection...")
    tsne_mgr = TSNEManager(random_state=42)
    coords_2d, tsne_meta = tsne_mgr.fit_transform(X_pca, perplexity=2.0, learning_rate=100.0, max_iter=250)
    assert coords_2d.shape == (6, 2), f"t-SNE shape mismatch: {coords_2d.shape}"
    print(f"  [PASS] t-SNE 2D mapping verified: {coords_2d.shape} in {tsne_meta['runtime_seconds']}s")

    # 6. Test K-Means Clustering
    print("\n[TEST 6] Testing K-Means Clustering & Parameter Sensitivity...")
    cluster_engine = ClusteringEngine(random_state=42)
    labels_k2, m_k2 = cluster_engine.fit_predict(X_pca, k=2)
    labels_k3, m_k3 = cluster_engine.fit_predict(X_pca, k=3)
    assert len(np.unique(labels_k2)) == 2
    assert len(np.unique(labels_k3)) == 3
    assert m_k2["k"] == 2 and m_k3["k"] == 3
    assert m_k2["inertia_wcss"] > m_k3["inertia_wcss"], "Inertia should decrease as k increases"
    print(f"  [PASS] K-Means k=2 (Inertia: {m_k2['inertia_wcss']}) vs k=3 (Inertia: {m_k3['inertia_wcss']}) verified.")

    # Test Exemplar Extraction
    exemplars_df = cluster_engine.extract_centroid_exemplars(X_pca, df_m, top_n=2)
    assert not exemplars_df.empty
    print(f"  [PASS] Extracted {len(exemplars_df)} centroid exemplars.")

    # 7. Test Image Discovery Studio & Real-Time Category Prediction
    print("\n[TEST 7] Testing Image Discovery Studio (Query Inference & Prediction)...")
    synthetic_gt = pd.DataFrame({
        "image_id": df_m["image_id"],
        "category": ["Vehicle", "Vehicle", "Animal", "Animal", "Plant", "Plant"]
    })
    studio = ImageDiscoveryStudio(
        feature_extractor=extractor,
        scaler=scaler,
        pca_manager=pca_mgr,
        centroids=cluster_engine.centroids_,
        dataset_features=X_pca,
        df_manifest=df_m,
        df_ground_truth=synthetic_gt,
        cluster_labels=labels_k3
    )
    query_im = Image.new("RGB", (64, 64), color=(250, 10, 10))
    query_results = studio.predict_image(query_im, feature_type="resnet50", top_k_neighbors=3)
    assert "nearest_cluster_id" in query_results
    assert "predicted_category" in query_results
    assert "prediction_confidence" in query_results
    assert len(query_results["similar_images"]) == 3
    print(f"  [PASS] Query assigned to Cluster {query_results['nearest_cluster_id']} (Distance: {query_results['min_centroid_distance']})")
    print(f"  [PASS] Predicted Category: {query_results['predicted_category']} (Confidence: {query_results['prediction_confidence']}%)")

    # 8. Test Experiment Manager
    print("\n[TEST 8] Testing Experiment Manager Logging & Comparison...")
    em = ExperimentManager()
    exp_id = em.log_experiment(
        name="Automated_Test_Run",
        dataset_name="test_suite_dataset",
        sample_count=6,
        feature_type="resnet50",
        scaler_type="standard",
        pca_config=pca_meta["metrics"],
        kmeans_config={"k": 3},
        metrics=m_k3,
        notes="Automated test validation"
    )
    assert exp_id.startswith("EXP_")
    exps_df = em.list_experiments()
    assert not exps_df.empty
    print(f"  [PASS] Experiment logged and verified in persistent storage: {exp_id}")

    # 9. Test External Validation
    print("\n[TEST 9] Testing External Validation...")
    synthetic_labels = np.array(["A", "A", "B", "B", "C", "C"])
    val_out = ExternalValidator.evaluate(synthetic_labels, labels_k3)
    assert val_out["available"] is True
    assert "ari" in val_out and "nmi" in val_out
    print(f"  [PASS] External validation computed: ARI={val_out['ari']}, NMI={val_out['nmi']}")

    # Clean up test dataset
    shutil.rmtree(dm.dataset_dir, ignore_errors=True)
    shutil.rmtree(dm.metadata_dir, ignore_errors=True)

    print("\n" + "=" * 70)
    print("ALL 9 TEST SUITES PASSED CLEANLY! ZERO FAILURES.")
    print("=" * 70)


if __name__ == "__main__":
    test_end_to_end_platform()
