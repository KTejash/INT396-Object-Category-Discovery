# VisionCluster AI — Architecture & Extensibility Guide

This document describes the design principles, modular pipeline architecture, and instructions for extending the VisionCluster AI platform with new feature extractors, clustering algorithms, and distance metrics.

---

## 1. Modular System Architecture

VisionCluster AI is built using decoupled, state-isolated modules adhering to the Single Responsibility Principle:

```text
Input (Upload / Benchmark)
        │
        ▼
[src/data_manager.py]
  - Multi-image & ZIP extraction
  - MD5 bitwise deduplication
  - Label isolation (ground truth strictly separated)
        │
        ▼
[src/feature_extractor.py]
  - ResNet-50 GAP (2048-d)
  - 3D RGB Color Histogram (512-d)
  - Combined Representation (2560-d)
        │
        ▼
[src/preprocessing.py]
  - StandardScaler / MinMaxScaler / RobustScaler / L2 Norm
        │
        ▼
[src/dimensionality_reduction.py]
  - PCA: Dynamic variance cutoff (90%, 95%, 99%) & scree analysis
  - t-SNE: Perplexity-constrained manifold embedding (2D)
        │
        ▼
[src/clustering.py]
  - K-Means++ clustering & k-sweep
  - Silhouette, Davies-Bouldin, Calinski-Harabasz metrics
  - Seed stability analysis (pairwise ARI)
        │
        ▼
[src/image_search.py]  <── Novel Image Query
  - Real-time feature extraction & projection
  - Nearest centroid assignment
  - Distance-weighted k-NN category prediction & confidence scoring
  - Visual nearest neighbors retrieval
        │
        ▼
[src/evaluation.py]
  - Post-hoc ground-truth unlocking
  - Adjusted Rand Index (ARI), NMI, Hungarian Accuracy (ACC)
  - Contingency matrix & allocation heatmap
```

---

## 2. Extending Feature Extractors

To introduce a new visual backbone (e.g., Vision Transformer `ViT-B/16` or `CLIP`):
1. Add the model definition in `src/feature_extractor.py`.
2. Implement a extraction method:
```python
def extract_vit_features(self, image_paths: List[Path], batch_size: int = 32) -> np.ndarray:
    # Load ViT, discard classification head, extract CLS token (768-d)
    ...
```
3. Register the new option in `extract_features_for_manifest` and in the Streamlit UI selector in `app/streamlit_app.py`.

---

## 3. Extending Clustering Algorithms

To add alternative clustering paradigms (such as Gaussian Mixture Models or Agglomerative Hierarchical Clustering):
1. In `src/clustering.py`, add a new method:
```python
def fit_gmm(self, X: np.ndarray, k: int = 8) -> Tuple[np.ndarray, Dict[str, Any]]:
    from sklearn.mixture import GaussianMixture
    gmm = GaussianMixture(n_components=k, random_state=self.random_state)
    labels = gmm.fit_predict(X)
    ...
```
2. Compute the standard intrinsic metrics (Silhouette, Davies-Bouldin, BIC/AIC).

---

## 4. Label Isolation Guarantee

In unsupervised machine learning, accidental data leakage invalidates research conclusions.
VisionCluster AI guarantees:
- Manifest files (`image_manifest.csv`) never contain category or class columns.
- Preprocessing, PCA, and Clustering pipelines take exclusively raw feature matrices `X`.
- Category annotations are loaded solely in `src/evaluation.py` for post-hoc validation and in `src/image_search.py` for reference label mapping.
