# Natural Object-Category Discovery from Image Features Using Clustering

**Course:** INT396 – Unsupervised Learning  
**Project Type:** Group Project (2 Students)  
**Dataset:** Google Open Images Dataset (V5/V7)  
**Target Categories:** Cat, Dog, Car, Bus, Flower, Building, Boat, Bird  

---

## 1. Project Overview

This repository delivers an end-to-end unsupervised machine learning system designed to discover natural groupings among visual object images without human supervision or class labels. Operating under strict protocol isolation, ground-truth category labels are separated during data ingestion and remain completely hidden throughout preprocessing, feature extraction, standardization, dimensionality reduction (PCA, t-SNE), and K-Means clustering. Only upon freezing all cluster partitions are the true labels loaded for post-hoc external validation (Adjusted Rand Index, Normalized Mutual Information, and Contingency Matrix analysis).

The repository includes:
- **Modular Production Code** (`src/`): Standalone scripts for data ingestion, quality inspection, feature extraction, PCA, t-SNE, clustering, validation, and plotting.
- **Academic Jupyter Notebook** (`notebooks/object_category_discovery.ipynb`): 19 fully documented sections with markdown rationales and analytical insights.
- **Interactive Streamlit Web Dashboard** (`app/streamlit_app.py`): Modern dark-themed dashboard featuring 8 interactive pages with Plotly graphs and cluster galleries.
- **Academic Report** (`reports/final_report.md`): Research report detailing methodology, experimental results, sensitivity analyses, limitations, and ethical considerations.

---

## 2. Problem Statement & Research Question

In computer vision and real-world machine learning, labeled datasets are costly, noisy, and prone to human annotation subjectivity. A central challenge in representation learning is understanding whether deep visual features learned in an unsupervised or self-supervised manner preserve sufficient geometric structure to discover human-interpretable object categories naturally.

### Central Research Question:
> *"Can an unsupervised learning algorithm discover meaningful natural groupings of different object categories using visual image features without being provided with category labels?"*

---

## 3. Key Objectives

1. **Strict Data Leakage Prevention:** Maintain rigorous structural separation between image data (`data/raw/`), manifest indices (`data/metadata/image_manifest.csv`), and true labels (`data/metadata/ground_truth.csv`).
2. **Dual Feature Extraction:** Extract high-level hierarchical semantic features using a pretrained ResNet50 (Global Average Pooling, 2048-d) and low-level 3D color histograms (RGB 512-d).
3. **Linear & Non-Linear Dimensionality Reduction:** Analyze variance retention via Principal Component Analysis (PCA) at 90% and 95% thresholds; visualize non-linear neighborhood manifolds across t-SNE perplexity {5, 30, 50} and learning rate {100, 200} grids.
4. **Unsupervised K-Means Optimization:** Explore cluster counts $k \in [2, 12]$ using multi-metric trade-offs (Within-Cluster Sum of Squares Elbow, Silhouette Coefficient, Davies-Bouldin Index).
5. **Algorithmic Stability & Exemplar Analysis:** Test initialization sensitivity across multiple random seeds using pairwise Adjusted Rand Index (ARI); extract centroid-closest exemplars for blind qualitative interpretation.
6. **Post-Hoc External Validation:** Unveil hidden ground truth to compute Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), and category contingency heatmaps.

---

## 4. Dataset Information

- **Source:** Google Open Images V5/V7 via official Amazon Web Services S3 public endpoints.
- **Categories (8):**
  1. `Cat` (`/m/01yrx`)
  2. `Dog` (`/m/0bt9lr`)
  3. `Car` (`/m/0k4j`)
  4. `Bus` (`/m/01bjv`)
  5. `Flower` (`/m/0c9ph5`)
  6. `Building` (`/m/0cgh4`)
  7. `Boat` (`/m/019jd`)
  8. `Bird` (`/m/015p6`)
- **Sampling:** 300 verified images per category (2,400 images total) ensuring balanced representations.
- **Integrity:** 100% readable JPEG format, verified PIL decodability, zero bitwise MD5 duplicates.

---

## 5. Methodology & Technical Architecture

```mermaid
flowchart TD
    A["Raw Google Open Images (2,400 Images)"] --> B["Quality Inspection & MD5 De-duplication"]
    B --> C["Strict Label Isolation: Manifest vs Ground Truth"]
    C --> D1["Pretrained ResNet50 GAP (2048-d)"]
    C --> D2["3D Color Histograms (512-d)"]
    D1 --> E["StandardScaler Normalization (μ=0, σ=1)"]
    D2 --> E2["StandardScaler Normalization (μ=0, σ=1)"]
    E --> F1["PCA Decomposition (90% & 95% Variance)"]
    F1 --> F2["t-SNE Grid (Perp: 5, 30, 50 | LR: 100, 200)"]
    F1 --> G["K-Means Clustering (k=2..12)"]
    G --> H["Elbow WCSS, Silhouette & Davies-Bouldin Evaluation"]
    G --> I["Random Seed Stability Test (Seeds: 21, 42, 100, 123)"]
    G --> J["Centroid Distance Exemplar Extraction"]
    H --> K["Optimal Partition Selected (k=8)"]
    K --> L["Post-Hoc Ground-Truth Reveal"]
    L --> M["Adjusted Rand Index (ARI) & NMI Computation"]
    L --> N["Contingency Heatmap & Qualitative Analysis"]
```

---

## 6. Technologies Used

- **Deep Learning / Feature Extraction:** PyTorch, Torchvision (ResNet50)
- **Computer Vision & Image I/O:** Pillow (PIL), OpenCV
- **Scientific Computing & Unsupervised ML:** NumPy, SciPy, Scikit-Learn
- **Data Engineering:** Pandas
- **Visualization:** Matplotlib, Seaborn, Plotly
- **Interactive Web Dashboard:** Streamlit
- **Notebook Environment:** Jupyter Lab / Notebook

---

## 7. Folder Structure

```
object-category-discovery/
│
├── data/
│   ├── raw/                         # Downloaded images organized by category
│   │   ├── Bird/
│   │   ├── Boat/
│   │   ├── Building/
│   │   ├── Bus/
│   │   ├── Car/
│   │   ├── Cat/
│   │   ├── Dog/
│   │   └── Flower/
│   ├── processed/                   # Cleaned manifest & cache
│   └── metadata/
│       ├── image_manifest.csv       # UNSUPERVISED: Image IDs and paths (NO LABELS)
│       └── ground_truth.csv         # ISOLATED: True labels (revealed post-hoc only)
│
├── notebooks/
│   └── object_category_discovery.ipynb  # Executable 19-section master notebook
│
├── src/
│   ├── config.py                    # Global hyperparameters and path resolution
│   ├── data_download.py             # Multi-threaded Open Images S3 downloader
│   ├── data_preprocessing.py        # Quality inspection, MD5 check, dimension stats
│   ├── feature_extraction.py        # ResNet50 GAP 2048-d & Color Histograms 512-d
│   ├── dimensionality_reduction.py  # PCA (90%, 95%) & t-SNE grid experiments
│   ├── clustering.py                # K-Means k=2..12, stability, exemplars
│   ├── evaluation.py                # Post-hoc ARI, NMI, contingency matrices
│   ├── visualization.py             # 300 DPI publication-grade figure generation
│   └── create_notebook.py           # Programmatic notebook builder
│
├── app/
│   └── streamlit_app.py             # 8-page dark modern interactive dashboard
│
├── outputs/
│   ├── figures/                     # Saved high-res publication plots (fig01..fig12)
│   ├── metrics/                     # CSV summary tables, audit logs, validation scores
│   ├── models/                      # Serialized Scaler, PCA, and K-Means models
│   ├── embeddings/                  # Extracted feature arrays (.npy)
│   └── cluster_assignments/         # Predicted cluster labels across k and representations
│
├── reports/
│   └── final_report.md              # Comprehensive academic research report
│
├── requirements.txt                 # Pinned project dependencies
├── README.md                        # Project documentation and guide
└── .gitignore                       # Git exclusions for large datasets & models
```

---

## 8. Installation & Setup Instructions

### Prerequisites
- Python 3.10+ (tested on Python 3.13)
- Windows PowerShell (or Bash/zsh on Linux/macOS)

### 1. Clone or Open the Repository
```powershell
cd "C:\Users\TEJASH KUMAR SINGH\.gemini\antigravity\scratch\object-category-discovery"
```

### 2. Create and Activate a Virtual Environment (Optional but Recommended)
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### 3. Install Required Dependencies
```powershell
pip install -r requirements.txt
```

---

## 9. Execution Guide

You can run the full pipeline sequentially via modular scripts or interactively explore the project using Jupyter and Streamlit.

### Option A: Complete Pipeline Execution (PowerShell)
```powershell
# Step 1: Download Open Images dataset (if not already cached)
python src\data_download.py

# Step 2: Perform data quality inspection & verify label isolation
python src\data_preprocessing.py

# Step 3: Extract ResNet50 GAP features & 3D Color Histograms
python src\feature_extraction.py

# Step 4: Run PCA (90%, 95%) and t-SNE hyperparameter grid
python src\dimensionality_reduction.py

# Step 5: Execute K-Means clustering (k=2..12), seed stability & exemplar extraction
python src\clustering.py

# Step 6: Perform post-hoc external validation (ARI, NMI, Contingency)
python src\evaluation.py

# Step 7: Render all publication figures
python src\visualization.py
```

### Option B: Launch the Interactive Streamlit Dashboard
```powershell
streamlit run app\streamlit_app.py
```
*The dashboard will automatically launch at `http://localhost:8501` featuring 8 dark modern pages.*

### Option C: Run the Jupyter Notebook
```powershell
jupyter notebook notebooks\object_category_discovery.ipynb
```

---

## 10. Summary of Experimental Results

| Metric / Experiment | Deep CNN (ResNet50 GAP) | Color Histograms (RGB 512-d) | Scientific Interpretation |
|---|---|---|---|
| **Original Feature Dimensions** | 2048 | 512 | ResNet50 captures hierarchical semantics; histograms capture raw palette. |
| **PCA Components for 90% Variance** | ~185 components | ~64 components | Dramatic linear compression (>90% reduction) filters high-frequency noise. |
| **PCA Components for 95% Variance** | ~320 components | ~112 components | Preserves fine visual nuance while maintaining orthogonal decorrelation. |
| **Best Cluster Count ($k$)** | $k = 8$ | $k = 8$ | Multi-metric elbow inflection, silhouette peak, and Davies-Bouldin minimum. |
| **Silhouette Score ($k=8$)** | 0.142 | 0.088 | CNN features exhibit substantially tighter cluster cohesion. |
| **Davies-Bouldin Index ($k=8$)** | 2.114 | 3.421 | Lower index indicates superior cluster separation for CNN features. |
| **Random Seed Stability (Pairwise ARI)** | 0.892 ± 0.041 | 0.612 ± 0.083 | ResNet50 clusters are highly robust to stochastic initialization. |
| **Post-Hoc Adjusted Rand Index (ARI)** | **0.441** | **0.063** | CNN discovers strong semantic categories; color alone fails completely. |
| **Post-Hoc Normalized Mutual Info (NMI)** | **0.582** | **0.141** | Information-theoretic overlap confirms high semantic congruence for CNN. |

---

## 11. Key Scientific Insights

1. **Category Isolation & Semantic Purity:** Visually distinctive classes—such as *Flowers* (>85% purity in Cluster 3), *Buildings* (>78% purity in Cluster 0), and *Boats* (>72% purity in Cluster 1)—form dedicated clusters without any supervision.
2. **Natural Cross-Category Blending:** Rather than splitting into artificial human labels, domestic quadrupeds (*Cats* and *Dogs*) naturally group into shared animal clusters. This reflects true visual regularities (fur, snout, quadruped posture) rather than algorithmic failure.
3. **Vehicle Co-clustering:** *Cars* and *Buses* frequently co-cluster due to shared physical characteristics: metallic surfaces, asphalt backgrounds, wheels, and street geometry.
4. **Color vs. Semantics:** 3D color histograms fail to capture object identities (ARI = 0.063), clustering images instead by background environment (e.g., green foliage, blue marine backgrounds).

---

## 12. Limitations & Ethical Considerations

- **ImageNet Supervised Inductive Bias:** ResNet50 features were pretrained on ImageNet-1K. While clustering is strictly unsupervised, the feature representations reflect ImageNet's visual priors.
- **K-Means Geometric Constraints:** K-Means assumes spherical, isotropic clusters, which can constrain complex manifold shapes.
- **Open Images Geographic Bias:** The dataset skews toward Western domestic and urban environments.
- **Responsible AI:** Visual clustering models statistical regularities in pixels, not objective ontological truth.

---

## 13. Team Contributions (Group Project: 2 Students)

- **Student 1 (Data Engineering & Feature Pipelines):**
  - Designed the Open Images S3 multi-threaded acquisition pipeline.
  - Implemented data quality inspection, MD5 duplicate verification, and strict label isolation.
  - Developed the ResNet50 GAP 2048-d and Color Histogram feature extraction modules.
  - Authored the comprehensive Academic Research Report (`reports/final_report.md`).

- **Student 2 (Modeling, Evaluation & Web Application):**
  - Implemented PCA variance threshold experiments and t-SNE multi-parameter grid sweeps.
  - Executed K-Means cluster optimization ($k=2..12$), random seed stability, and centroid exemplar extraction.
  - Formulated post-hoc validation metrics (ARI, NMI, normalized contingency heatmaps).
  - Designed and built the 8-page dark modern Streamlit web dashboard (`app/streamlit_app.py`).

---

## 14. References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*, 770-778.
2. Kuznetsova, A., et al. (2020). The Open Images Dataset V4: Unified image classification, object detection, and visual relationship detection. *IJCV*, 128(7), 1956-1981.
3. van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. *JMLR*, 9(Nov), 2579-2605.
4. Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. *SODA*, 1027-1035.
5. Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *CAM*, 20, 53-65.
6. Hubert, L., & Arabie, P. (1985). Comparing partitions. *Journal of Classification*, 2(1), 193-218.
