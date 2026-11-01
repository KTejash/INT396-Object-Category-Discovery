# Object Discovery Lab — Interactive Unsupervised AI Platform

> **INT396 Academic Project** | Natural Object-Category Discovery from Image Features Using Clustering
>
> Upgraded from a static Streamlit dashboard into a fully interactive, end-to-end unsupervised ML platform with live computation, custom image upload, configurable ML parameters, experiment management, and visual search.

---

## Overview

This application discovers natural groupings (clusters) in image datasets without any labels.
Every result — features, PCA, t-SNE, K-Means metrics — comes from **real computation** on demand.
No hardcoded metrics or pre-generated charts are displayed.

**Research question:** *Can an unsupervised clustering algorithm automatically discover natural object categories (Birds, Boats, Buildings, Buses, Cars, Cats, Dogs, Flowers) purely from visual features, without being given any labels?*

---

## Features at a Glance

| # | Page | What it does |
|---|------|-------------|
| 1 | Executive Overview | Live pipeline status, benchmark metrics, feature space overview |
| 2 | Dataset Workspace | Upload images/ZIPs, dataset audit, duplicate detection, label isolation |
| 3 | Feature Extraction Lab | ResNet50 (2048-d), Colour Histogram, Combined; batch inference with progress |
| 4 | Data Preprocessing | StandardScaler, MinMaxScaler, L2 Normalizer, before/after statistics |
| 5 | PCA Exploration | Real PCA, scree curve, variance targets (90/95/99%), loadings, download |
| 6 | t-SNE Explorer | Real t-SNE, interactive perplexity/LR/seed controls, cluster colouring |
| 7 | K-Means Clustering Lab | Live clustering, elbow method, silhouette analysis, seed stability, metrics |
| 8 | Discovered Cluster Gallery | Image grid per cluster, search by filename, download cluster lists |
| 9 | Image Discovery Studio | Upload query image → find nearest cluster + visually similar images |
| 10 | Experiment Comparison | Save, compare, radar-chart multiple experiments, export JSON/CSV |
| 11 | External Validation | ARI, NMI, contingency heatmap (ground truth never leaks into clustering) |
| 12 | Reports & Exports | Download features, PCA results, cluster assignments, experiment logs |
| 13 | Limitations & Research Ethics | Honest academic discussion of what unsupervised learning can/cannot do |

---

## Architecture

```
object-category-discovery/
├── app/
│   └── streamlit_app.py          ← 13-page interactive dashboard (main entry point)
├── src/
│   ├── config.py                 ← Paths, constants, seeds
│   ├── data_manager.py           ← Upload, ZIP extract, MD5 dedup, audit, label isolation
│   ├── feature_extractor.py      ← ResNet50 GAP (2048-d), Color Histogram, Combined
│   ├── preprocessing.py          ← StandardScaler, MinMaxScaler, L2 Normalizer
│   ├── dimensionality_reduction.py ← PCA (fixed/variance modes), t-SNE (perplexity validated)
│   ├── clustering.py             ← K-Means engine, k-sweep, seed stability, exemplars
│   ├── image_search.py           ← Query image → preprocess → centroid assignment → neighbors
│   ├── experiment_manager.py     ← Log, compare, export experiments (JSON/CSV)
│   ├── evaluation.py             ← ARI, NMI, V-measure, Homogeneity, contingency matrix
│   ├── visualization.py          ← Plotly: scree, scatter, elbow, heatmap, radar chart
│   └── create_notebook.py        ← Generates static analysis notebook
├── data/
│   ├── raw/                      ← 2,400 Open Images (8 classes × 300 images)
│   ├── processed/                ← Cached feature arrays (.npy)
│   └── labels/                   ← Ground truth labels (isolated, never used during clustering)
├── models/                       ← Saved sklearn model objects (.pkl)
├── experiments/
│   └── experiments_log.json      ← Persistent experiment history
├── outputs/
│   ├── embeddings/               ← Pre-computed CNN / PCA / t-SNE embeddings
│   ├── figures/                  ← Exported charts
│   └── reports/                  ← Exported reports
├── notebooks/
│   └── object_category_discovery.ipynb
├── tests/
│   └── test_interactive_platform.py  ← 9 automated test suites (all pass)
├── requirements.txt
└── README.md
```

---

## Benchmark Dataset

| Class | Images | Source |
|-------|--------|--------|
| Bird | 300 | Open Images v7 |
| Boat | 300 | Open Images v7 |
| Building | 300 | Open Images v7 |
| Bus | 300 | Open Images v7 |
| Car | 300 | Open Images v7 |
| Cat | 300 | Open Images v7 |
| Dog | 300 | Open Images v7 |
| Flower | 300 | Open Images v7 |
| **Total** | **2,400** | |

---

## Benchmark Results (ResNet50 → StandardScaler → PCA 95% → K-Means k=8)

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Silhouette Score | 0.121 | Moderate cluster separation in 95-PC space |
| Davies-Bouldin Index | 2.143 | Lower is better; room for improvement |
| Calinski-Harabasz Score | 168.4 | Higher = more compact clusters |
| **ARI** | **0.4037** | 40% agreement with ground truth (random = 0) |
| **NMI** | **0.5868** | 59% mutual information with ground truth (random = 0) |

> Results reflect unsupervised discovery without any label guidance. ARI = 0.40 is a strong result for 8-class unsupervised discovery.

---

## Installation (Windows / PowerShell)

### Prerequisites
- Python 3.10–3.13
- Git
- ~6 GB disk space (dataset + models)

### Steps

```powershell
# 1. Clone the repository
git clone https://github.com/KTejash/INT396-Object-Category-Discovery.git
cd INT396-Object-Category-Discovery

# 2. Create a virtual environment (recommended)
python -m venv .venv
.venv\Scripts\Activate.ps1

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app\streamlit_app.py
```

Open your browser at **http://localhost:8501**

---

## Running in VS Code

1. Open the project folder: **File → Open Folder** → select `object-category-discovery`
2. Open a new terminal: **Ctrl + `**
3. Activate the virtual environment (if used): `.venv\Scripts\Activate.ps1`
4. Run: `streamlit run app\streamlit_app.py`
5. VS Code will show a pop-up to open `http://localhost:8501` in your browser

---

## ML Workflow

```
Upload Images / Select Dataset
        ↓
Feature Extraction (ResNet50 2048-d | Color Histogram | Combined)
        ↓
Preprocessing (StandardScaler | MinMaxScaler | L2 Norm)
        ↓
Dimensionality Reduction — PCA (variance target or fixed k)
        ↓
Visualization — t-SNE 2D embedding
        ↓
K-Means Clustering (k=2..12, elbow method, silhouette analysis)
        ↓
Cluster Gallery + Image Discovery Studio
        ↓
External Validation (ARI / NMI vs ground truth — post-hoc only)
        ↓
Experiment Manager (save, compare, export)
```

> **Label isolation guarantee:** Ground truth class labels are stored separately and never passed into any feature extraction, scaling, PCA, t-SNE, or K-Means step. Labels are only loaded after clustering completes for external evaluation metrics.

---

## Running the Automated Tests

```powershell
cd INT396-Object-Category-Discovery
python tests/test_interactive_platform.py
```

Expected output:
```
======================================================================
RUNNING AUTOMATED TEST SUITE: Object Discovery Lab
======================================================================
[TEST 1] Testing Dataset Ingestion & Corruption Filtering...  [PASS]
[TEST 2] Testing Feature Extraction (ResNet50, Color Hist, Combined)...  [PASS]
[TEST 3] Testing Preprocessing Transformations...  [PASS]
[TEST 4] Testing PCA Decomposition...  [PASS]
[TEST 5] Testing t-SNE Manifold Projection...  [PASS]
[TEST 6] Testing K-Means Clustering & Parameter Sensitivity...  [PASS]
[TEST 7] Testing Image Discovery Studio (Query Inference)...  [PASS]
[TEST 8] Testing Experiment Manager Logging & Comparison...  [PASS]
[TEST 9] Testing External Validation...  [PASS]
======================================================================
ALL 9 TEST SUITES PASSED CLEANLY! ZERO FAILURES.
======================================================================
```

---

## Dependencies

| Package | Purpose |
|---------|---------|
| `streamlit` | Interactive web dashboard |
| `torch` + `torchvision` | ResNet50 feature extraction |
| `scikit-learn` | PCA, t-SNE, K-Means, metrics |
| `numpy` + `pandas` | Numerical computing, data frames |
| `Pillow` | Image loading and preprocessing |
| `plotly` | Interactive charts |
| `joblib` | Model serialisation, parallel ops |
| `scipy` | Scientific computing utilities |
| `opencv-python` | Image I/O and processing |

---

## Known Limitations

- **CPU-only inference** — ResNet50 extraction runs on CPU. Expect ~3–5 min for 2,400 images on a typical laptop.
- **t-SNE is stochastic** — results will vary between runs unless a fixed random seed is used.
- **K-Means is sensitive to initialisation** — run the seed stability experiment to assess result reliability.
- **Unsupervised ≠ supervised accuracy** — cluster IDs do not map 1:1 to class labels. ARI/NMI measure agreement, not correctness.
- **Memory** — Combined features (ResNet50 2048-d + Histogram 512-d = 2560-d) for 2,400 images requires ~23 MB float32. This is fine on a normal laptop.
- **Windows PowerShell** — Some Unicode characters may cause encoding errors in older terminals. The app is fully Windows-compatible.

---

## Research Ethics

This project uses public domain images from the Open Images v7 dataset under the **CC BY 4.0** licence.
- No personally identifiable images are used.
- No model training is performed on private data.
- External metrics (ARI, NMI) are computed post-hoc and do not influence clustering.
- Cluster labels are arbitrary integers assigned by K-Means, not semantic categories.

---

## Project Team

| Field | Value |
|-------|-------|
| Student | Tejash Kumar Singh |
| Email | tejasharyan550@gmail.com |
| Course | INT396 — Machine Learning |
| GitHub | [KTejash/INT396-Object-Category-Discovery](https://github.com/KTejash/INT396-Object-Category-Discovery) |

---

## Commits

| Commit | Description |
|--------|-------------|
| `0b924c2` | Initial commit: baseline static dashboard |
| `ce85b95` | feat(backend): dynamic DatasetManager, FeatureExtractor, Preprocessing, PCA/t-SNE, Clustering, ImageSearch, ExperimentManager |
| `551aa71` | feat(app+tests): fix extract_resnet50, all 9 tests pass, 13-page interactive dashboard |
| Latest | feat(docs): complete README with architecture, workflow, benchmark results |
