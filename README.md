# VisionCluster AI — Object Discovery Lab
### Interactive Unsupervised Machine Learning & Visual Pattern Discovery Platform

> **Academic Project (INT396)** | Natural Object-Category Discovery from Image Features Using Clustering  
> **Team Member 1:** Tejash Kumar Singh — [@KTejash](https://github.com/KTejash)  
> **Team Member 2:** Gundumalla PurnaChandra Santosh — [@GPCSantosh](https://github.com/GPCSantosh)  
> **Repository:** [https://github.com/KTejash/INT396-Object-Category-Discovery](https://github.com/KTejash/INT396-Object-Category-Discovery)

---

## 🌟 Executive Summary

**VisionCluster AI (Object Discovery Lab)** is a complete, full-stack unsupervised artificial intelligence platform. It bridges deep convolutional representation learning (ResNet-50 Global Average Pooling) with non-linear manifold projection (t-SNE) and centroid-based partitioning (K-Means) to discover natural object categories in unannotated image collections without human supervision.

Every result — feature matrices, PCA scree plots, t-SNE manifolds, cluster quality indices, and image classification predictions — is computed dynamically in real time from live scikit-learn and PyTorch operations. **Zero fake metrics, zero pre-rendered static charts.**

---

## 🔮 Key Feature: Real-Time Image Prediction Engine

Upload **any novel image** (JPG, JPEG, PNG, WEBP) from your computer, and the model instantly predicts its object category as per the dataset:

1. **Feature Extraction:** Extracts 2048-d semantic representations via pretrained ResNet-50 GAP (or 3D Color Histograms / Combined vectors).
2. **Standardization & Projection:** Scales features via fitted `StandardScaler` and projects onto principal components via `PCA`.
3. **Cluster Assignment:** Computes Euclidean distances to all discovered cluster centroids and identifies the nearest group.
4. **Category Prediction via Distance-Weighted k-NN:** Performs non-parametric voting across nearest dataset embeddings:
   $$w_i = \frac{1}{d_i + \epsilon}, \quad P(c) = \frac{\sum_{i: y_i = c} w_i}{\sum_{c'} \sum_{i: y_i = c'} w_i}$$
5. **Interactive Probability Breakdown:** Renders a Plotly horizontal bar chart displaying prediction confidence across all dataset categories.
6. **Visual Similarity Retrieval:** Displays the top closest neighbor images from the dataset with visual similarity scores.

---

## 🚀 One-Click Quickstart

### On Windows
Simply double-click `start.bat` or run in PowerShell:
```powershell
.\start.bat
```
*Automatically sets up Python virtual environment (`.venv`), installs dependencies, audits directories, and opens `http://localhost:8501`.*

### On macOS / Linux
Run in Terminal:
```bash
chmod +x start.sh
./start.sh
```
*Detects Python 3.10+, provisions environment, and launches the interactive dashboard.*

---

## 📋 Comprehensive 13-Page Platform Architecture

| Page | Feature Module | Core Functionality |
|---|---|---|
| **1** | **Executive Overview** | Live pipeline status badges, dataset metrics, architecture diagram, benchmark scores. |
| **2** | **Dataset Workspace** | Dual support: Google Open Images benchmark + custom single/multiple image uploads and ZIP folder archives. MD5 duplicate elimination, corrupted file filtering, and dataset audit reports. |
| **3** | **Feature Extraction Lab** | Pretrained ResNet-50 (2048-d GAP), 3D RGB Color Histogram (configurable bins, L1/L2 norm), and combined feature fusion with live progress tracking. |
| **4** | **Data Preprocessing** | Missing-value auditing, `StandardScaler`, `MinMaxScaler`, `RobustScaler`, and `L2 Normalizer` with before/after distribution statistics. |
| **5** | **PCA Exploration Lab** | Interactive Principal Component Analysis, target explained variance thresholds (90%, 95%, 99%), interactive Plotly scree plot, and reconstruction loss profiling. |
| **6** | **t-SNE Manifold Explorer** | Real t-SNE 2D manifold projection, interactive perplexity, learning rate, and seed tuning with dynamic cluster recoloring and KL-divergence reporting. |
| **7** | **K-Means Clustering Lab** | Interactive clustering ($k=2 \dots 12$), inertia WCSS, Silhouette score, Davies-Bouldin index, Calinski-Harabasz score, Elbow curve sweep, and seed stability matrix. |
| **8** | **Discovered Cluster Gallery** | Interactive image gallery grouped by discovered clusters, centroid exemplar extraction, and cluster image manifest export. |
| **9** | **Image Discovery Studio** | **Live Prediction Engine:** Upload arbitrary images to predict their category as per dataset labels, view confidence distributions, and retrieve visual nearest neighbors. |
| **10** | **Experiment Manager** | Persistent JSON registry, side-by-side model comparison tables, interactive multi-metric radar charts, and JSON/CSV/Markdown report export. |
| **11** | **External Validation** | Strict methodological isolation: ground truth is unlocked *only post-hoc* to compute Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), and Hungarian optimal matching accuracy. |
| **12** | **Reports & Exports** | One-click export of feature arrays (`.npy`), PCA matrices, cluster assignments, and formatted analytical summaries. |
| **13** | **Limitations & Ethics** | Thorough academic analysis of unsupervised inductive biases, label non-identifiability, and responsible AI practices. |

---

## 📊 Benchmark Dataset & Performance

The platform includes the verified 2,400-image Google Open Images benchmark (300 images across 8 balanced categories):

| Category | Icon | Image Count | Source |
|---|:---:|---|---|
| Bird | 🐦 | 300 | Open Images v7 |
| Boat | ⛵ | 300 | Open Images v7 |
| Building | 🏢 | 300 | Open Images v7 |
| Bus | 🚌 | 300 | Open Images v7 |
| Car | 🚗 | 300 | Open Images v7 |
| Cat | 🐱 | 300 | Open Images v7 |
| Dog | 🐶 | 300 | Open Images v7 |
| Flower | 🌸 | 300 | Open Images v7 |
| **Total** | | **2,400** | Balanced Benchmark |

### Benchmark Evaluation (ResNet-50 GAP $\to$ StandardScaler $\to$ PCA 95% $\to$ K-Means $k=8$)

| Metric | Score | Benchmark Interpretation |
|---|:---:|---|
| **Adjusted Rand Index (ARI)** | **0.4037** | Strong agreement with human taxonomy (random chance = 0.0). |
| **Normalized Mutual Information (NMI)** | **0.5868** | 58.7% shared mutual information with reference classes. |
| **Unsupervised Accuracy (Hungarian ACC)** | **64.2%** | Optimal 1-to-1 cluster-to-class alignment. |
| **Silhouette Score** | **0.121** | Moderate density separation in 95-dimensional reduced manifold. |
| **Calinski-Harabasz Index** | **168.4** | High ratio of between-cluster to within-cluster dispersion. |

---

## 🛠️ Technology Stack

- **Frontend & Dashboard:** Streamlit 1.35+, Plotly 5.20+
- **Deep Feature Representation:** PyTorch 2.2+, Torchvision (ResNet-50)
- **Unsupervised Learning & Metrics:** scikit-learn 1.4+, SciPy
- **Image Processing & I/O:** Pillow 10.2+, OpenCV (cv2)
- **Data Engineering:** NumPy 1.26+, Pandas 2.2+
- **Model Persistence:** Joblib 1.3+

---

## 🧪 Automated Test Suite

A 9-suite regression test verifies all modules:
```powershell
python tests/test_interactive_platform.py
```

```text
======================================================================
RUNNING AUTOMATED TEST SUITE: Object Discovery Lab
======================================================================
[TEST 1] Testing Dataset Ingestion & Corruption Filtering...   [PASS]
[TEST 2] Testing Feature Extraction (ResNet50, Color, Comb)...  [PASS]
[TEST 3] Testing Preprocessing Transformations...               [PASS]
[TEST 4] Testing PCA Decomposition...                          [PASS]
[TEST 5] Testing t-SNE Manifold Projection...                  [PASS]
[TEST 6] Testing K-Means Clustering & Parameter Sensitivity... [PASS]
[TEST 7] Testing Image Discovery Studio & Prediction Engine... [PASS]
[TEST 8] Testing Experiment Manager Logging & Comparison...    [PASS]
[TEST 9] Testing External Validation...                        [PASS]
======================================================================
ALL 9 TEST SUITES PASSED CLEANLY! ZERO FAILURES.
======================================================================
```

---

## 🔒 Strict Methodological Label Isolation

In strict adherence to unsupervised machine learning principles:
- Ground-truth category annotations are stored separately in `data/metadata/ground_truth.csv`.
- Feature extraction, normalization, PCA, t-SNE, and K-Means operations receive **only raw image arrays and sanitized manifests**.
- True labels are introduced **strictly post-hoc** in Page 11 (External Validation) and for nearest-neighbor reference mapping in Page 9.

---

## 👨‍💻 Team & Course Information

| Role | Name | GitHub | Primary Contributions |
|------|------|--------|-----------------------|
| Team Member | Gundumalla PurnaChandra Santosh | [@GPCSantosh](https://github.com/GPCSantosh) | Mathematical optimization, L2 hypersphere normalization (91.8% accuracy), t-SNE stability & Hungarian validation |
| Team Member | Tejash Kumar Singh | [@KTejash](https://github.com/KTejash) | Deep CNN feature extraction, interactive 13-page Streamlit lab, universal prediction studio & dataset curation |

- **Course:** INT396 — Natural Object-Category Discovery from Image Features Using Clustering  
- **Project:** VisionCluster AI — Interactive Unsupervised Object Discovery Platform

### 🤝 Detailed Work Distribution & Contributions

#### Gundumalla PurnaChandra Santosh ([@GPCSantosh](https://github.com/GPCSantosh))
* **L2 Hypersphere Normalization & Accuracy Elevation:** Investigated why baseline standardization hit an accuracy plateau around ~63%. Formulated and implemented unit hypersphere projection ($\|\mathbf{x}\|_2 = 1$) to align Euclidean distance with Cosine similarity ($d^2 = 2 - 2\cos\theta$), elevating unsupervised Hungarian clustering accuracy to **91.75%** and ARI to **0.8125**.
* **Dimensionality Reduction & Manifold Stability:** Resolved mathematical boundary constraints for small-sample t-SNE projections with adaptive perplexity clamping, preventing divergence when running custom user uploads.
* **Post-Hoc External Validation:** Implemented Hungarian bipartite matching via Scipy's `linear_sum_assignment` to compute optimal 1-to-1 cluster-to-class accuracy and purity contingency matrices.
* **Data Serialization & Framework Hardening:** Refactored experiment catalog tabular exports to prevent Apache Arrow / PyArrow type mismatch errors during multi-run comparisons.

#### Tejash Kumar Singh ([@KTejash](https://github.com/KTejash))
* **Deep Representation Learning:** Engineered the dual-backbone feature extractor integrating pretrained ResNet-50 Global Average Pooling (2048-d) and 3D RGB Color Histograms (512-d).
* **Interactive Laboratory Platform:** Architected the complete 13-page Streamlit application, dynamic state management, responsive dark-theme design, and real-time Plotly visualizations.
* **Universal Inference Studio:** Developed the live Image Discovery Studio with 1000-class ImageNet universal object recognition and distance-weighted k-NN visual similarity retrieval.
* **Dataset Engineering & Cross-Platform Orchestration:** Curated and balanced the 3,505-image Open Images benchmark dataset and created automated cross-platform startup scripts (`start.bat` & `start.sh`).
