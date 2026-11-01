# Natural Object-Category Discovery from Image Features Using Clustering

**Course:** INT396 – Unsupervised Learning  
**Project Title:** Natural Object-Category Discovery from Image Features Using Clustering  
**Authors:** Student 1 & Student 2 (Group Project)  
**Affiliation:** School of Computer Science and Engineering  
**Date:** November 2026  

---

## Abstract

Supervised image classification models achieve high accuracy but remain dependent on vast quantities of human-annotated category labels, which are costly, prone to subjective bias, and infeasible in open-world exploratory settings. This study investigates the fundamental unsupervised learning question: *Can an unsupervised learning algorithm discover meaningful natural groupings of different object categories using visual image features without being provided with category labels?* 

We design, implement, and evaluate an end-to-end unsupervised pipeline using a balanced corpus of 2,400 images across eight diverse categories from Google Open Images (*Cat, Dog, Car, Bus, Flower, Building, Boat, Bird*). To guarantee scientific integrity, category labels were strictly isolated and barred from preprocessing, scaling, dimensionality reduction, parameter tuning, and clustering. We extract two distinct feature representations: a 2048-dimensional deep hierarchical representation via pretrained ResNet50 Global Average Pooling (GAP) and a 512-dimensional normalized 3D color histogram. Features were standardized and compressed using Principal Component Analysis (PCA) retaining 90% and 95% total variance, followed by non-linear manifold mapping using t-Distributed Stochastic Neighbor Embedding (t-SNE) across six hyperparameter combinations. K-Means clustering was executed across $k \in [2, 12]$, with optimal partitions determined via Within-Cluster Sum of Squares (WCSS) elbow analysis, Silhouette scores, and Davies-Bouldin indices. Initialization stability was established across multiple random seeds, and centroid exemplars were extracted for qualitative cluster naming.

Post-hoc external validation using the unlocked ground-truth labels yielded an Adjusted Rand Index (ARI) of **0.441** and a Normalized Mutual Information (NMI) of **0.582** for deep CNN features, compared to an ARI of **0.063** for color histograms. The findings demonstrate that deep visual features contain sufficient geometric regularity to naturally discover semantic object boundaries without supervision, while highlighting that visual clustering naturally blurs arbitrary human taxonomic divisions where objects share visual geometry (e.g., domestic quadrupeds or motorized vehicles).

---

## 1. Introduction

The predominant paradigm in computer vision has historically relied on supervised learning, where large-scale networks are trained to map input pixel matrices to fixed categorical targets. However, natural intelligence discovers regularities and object concepts largely without explicit supervision, organizing sensory input through perceptual clustering, topological proximity, and structural invariance.

Unsupervised object category discovery seeks to model this capability by discovering coherent visual clusters directly from unannotated image collections. By eliminating label supervision, unsupervised discovery offers major advantages for automated cataloging, robotic perception, medical anomaly discovery, and exploratory data analysis. 

In this investigation, conducted as part of INT396 (Unsupervised Learning), we evaluate the capacity of classical clustering algorithms operating on deep convolutional representations to discover natural object groupings. We adhere strictly to a blind protocol: the machine learning system receives only raw pixels, generates feature vectors, normalizes feature distributions, compresses dimensionality, and discovers clusters without ever inspecting category labels. Only upon final model freezing are true labels loaded to evaluate the semantic fidelity of the discovered partitions.

---

## 2. Problem Statement

Given an unlabeled dataset of natural images $\mathcal{X} = \{\mathbf{x}_1, \mathbf{x}_2, \dots, \mathbf{x}_N\}$ drawn from unknown discrete object categories $\mathcal{Y} = \{y_1, y_2, \dots, y_C\}$, our goal is to partition $\mathcal{X}$ into $k$ disjoint clusters $\mathcal{C} = \{C_1, C_2, \dots, C_k\}$ such that:
1. Images within the same cluster share high visual and structural similarity.
2. Images in different clusters exhibit distinct feature characteristics.
3. The discovered clusters $\mathcal{C}$ correlate strongly with the underlying semantic categories $\mathcal{Y}$, despite the algorithm having zero access to $\mathcal{Y}$ during training.

### Core Research Question:
> *"Can an unsupervised learning algorithm discover meaningful natural groupings of different object categories using visual image features without being provided with category labels?"*

---

## 3. Project Objectives

1. **Protocol Rigor & Label Isolation:** Implement programmatic separation between image files, manifest identifiers, and true labels to guarantee zero data leakage.
2. **Dual Feature Extraction & Comparison:** Contrast high-level deep semantic features (ResNet50 GAP, 2048-d) against low-level color distributions (RGB histograms, 512-d).
3. **Linear & Non-Linear Dimensionality Reduction:** Determine the intrinsic dimensionality of image features using PCA at 90% and 95% variance thresholds; project high-dimensional manifolds into 2D using t-SNE across systematic perplexity and learning rate sweeps.
4. **Clustering Optimization:** Apply K-Means across $k \in [2, 12]$, diagnosing cluster quality using internal unsupervised metrics: Within-Cluster Sum of Squares (WCSS / Inertia), Silhouette Coefficient, and Davies-Bouldin Index.
5. **Algorithmic Stability:** Validate the stability of discovered clusters across multiple random seeds using pairwise Adjusted Rand Index (ARI).
6. **Centroid Exemplar Interpretation:** Extract images closest to cluster centroids in Euclidean space to formulate descriptive cluster profiles without inspecting labels.
7. **Post-Hoc External Validation:** Evaluate the discovered clusters against hidden ground-truth labels using Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), and normalized contingency heatmaps.

---

## 4. Dataset Description & Acquisition

### 4.1 Dataset Source
Images were acquired from the **Google Open Images Dataset (V5/V7)** using its official Amazon Web Services (AWS) S3 public endpoints (`https://s3.amazonaws.com/open-images-dataset/`). Open Images is a widely recognized benchmark featuring millions of real-world photographs with complex natural backgrounds, varied lighting, and natural object viewpoints.

### 4.2 Target Categories
We selected eight visually diverse object categories spanning living organisms, engineered vehicles, and static environmental structures:
1. `Cat` (`/m/01yrx`): Domestic felines; organic shape, facial features, fur textures.
2. `Dog` (`/m/0bt9lr`): Domestic canines; variable morphology, distinct postures.
3. `Car` (`/m/0k4j`): Compact passenger vehicles; metallic surfaces, wheels, road context.
4. `Bus` (`/m/01bjv`): Large transit vehicles; rectangular silhouettes, commercial livery.
5. `Flower` (`/m/0c9ph5`): Botanical specimens; radial petal symmetry, vibrant pigmentation.
6. `Building` (`/m/0cgh4`): Architectural structures; vertical/horizontal edges, facades.
7. `Boat` (`/m/019jd`): Aquatic vessels; hulls, water reflections, open horizon.
8. `Bird` (`/m/015p6`): Avian species; plumage, beaks, perched or flight postures.

### 4.3 Sampling Strategy
To ensure class balance, exactly 300 verified images were sampled per category, yielding a total dataset of **$N = 2,400$ images**. Sampling was filtered to prioritize single-label instances to minimize multi-object visual ambiguity while preserving natural backgrounds.

---

## 5. Data Preprocessing & Quality Assurance

### 5.1 Strict Label Separation
To prevent data leakage, files were segregated into:
- `data/raw/<Category>/<ImageID>.jpg`: Image asset storage.
- `data/metadata/image_manifest.csv`: Image IDs, relative paths, and file sizes. **Strictly devoid of label columns.**
- `data/metadata/ground_truth.csv`: Image IDs, ground-truth category names, and confidence scores. **Stored in isolation.**

A programmatic validation step was executed prior to modeling to verify that no column in the manifest matched any forbidden target terms (`label`, `category`, `class`, `target`).

### 5.2 Quality Inspection Audit
All images underwent strict automated verification:
- **Missing File Audit:** Verified that all paths in the manifest resolve to valid disk locations.
- **Corrupt File Detection:** Every image was validated using PIL's `verify()` routine to detect truncated bitstreams.
- **Bitwise Duplicate Detection:** Computed MD5 hashes for all files; zero duplicate hashes were detected.
- **Resolution Analysis:** Image resolutions averaged $1024 \times 768$ pixels with median aspect ratio of 1.33.

---

## 6. Feature Extraction & Normalization

### 6.1 Deep Convolutional Features (ResNet50 GAP)
We utilized a deep ResNet50 architecture pretrained on ImageNet-1K. The final 1000-way linear classification layer was detached, exposing the 2048-dimensional Global Average Pooling (GAP) tensor:
$$\mathbf{f}_{\text{CNN}}(\mathbf{x}) = \frac{1}{H \times W} \sum_{h=1}^H \sum_{w=1}^W \mathbf{A}_{h,w} \in \mathbb{R}^{2048}$$
Images were resized to $256 \times 256$, center-cropped to $224 \times 224$, and normalized using standard ImageNet parameters ($\mu = [0.485, 0.456, 0.406]$, $\sigma = [0.229, 0.224, 0.225]$).

### 6.2 3D Color Histogram Features
To establish a baseline representing low-level color distributions without spatial or structural awareness, we extracted 3D normalized RGB histograms with 8 bins per color channel ($8 \times 8 \times 8 = 512$ dimensions). Histograms were normalized via $L_1$ norm:
$$\mathbf{h}_i = \frac{h_i}{\sum_{j=1}^{512} h_j}, \quad \sum_{i=1}^{512} \mathbf{h}_i = 1.0$$

### 6.3 Feature Standardization (StandardScaler)
Because distance-based algorithms like PCA and K-Means are sensitive to feature variances, we applied `StandardScaler` ($\mu = 0, \sigma = 1$) fitted exclusively on the feature matrices:
$$z_{ij} = \frac{x_{ij} - \mu_j}{\sigma_j}$$
Standardization ensures that high-magnitude feature dimensions do not dominate distance calculations.

---

## 7. Dimensionality Reduction

### 7.1 Principal Component Analysis (PCA)
PCA computes orthogonal eigenvectors of the sample covariance matrix $\mathbf{\Sigma} = \frac{1}{N-1}\mathbf{Z}^T\mathbf{Z}$. We analyzed the cumulative explained variance to identify optimal compression thresholds:

- **90% Variance Retained:** Reached at **$n_{90} \approx 185$ components** (a 91.0% compression from 2048 dimensions).
- **95% Variance Retained:** Reached at **$n_{95} \approx 320$ components** (an 84.4% compression from 2048 dimensions).

PCA compression removes high-frequency noise and decorrelates features, mitigating the curse of dimensionality for subsequent distance-based clustering.

### 7.2 t-Distributed Stochastic Neighbor Embedding (t-SNE)
t-SNE models local neighborhood similarities using Gaussian distributions in the high-dimensional space and Student-t distributions in the low-dimensional embedding:
$$p_{j|i} = \frac{\exp(-\|\mathbf{z}_i - \mathbf{z}_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq i} \exp(-\|\mathbf{z}_i - \mathbf{z}_k\|^2 / 2\sigma_i^2)}, \quad q_{ij} = \frac{(1 + \|\mathbf{y}_i - \mathbf{y}_j\|^2)^{-1}}{\sum_{k} \sum_{l \neq k} (1 + \|\mathbf{y}_k - \mathbf{y}_l\|^2)^{-1}}$$

We evaluated a $3 \times 2$ parameter grid:
- **Perplexity:** 5, 30, 50
- **Learning Rate:** 100, 200

**Findings:** Perplexity 5 created fragmented local clumps. Perplexity 30 with learning rate 200 provided optimal separation, revealing distinct global clusters with minimal KL divergence (0.914). Perplexity 50 excessively contracted distinct visual groups.

---

## 8. K-Means Clustering & Model Selection

### 8.1 Algorithm
K-Means minimizes Within-Cluster Sum of Squares (Inertia):
$$J = \sum_{c=1}^k \sum_{\mathbf{z}_i \in C_c} \|\mathbf{z}_i - \boldsymbol{\mu}_c\|^2$$
We used `k-means++` initialization to ensure well-separated initial centroids and ran 10 random restarts per experiment.

### 8.2 Determining Optimal $k$
We tested $k \in [2, 12]$ across both PCA-90% and PCA-95% spaces, tracking three complementary metrics:
1. **Elbow Method (WCSS):** Observed a pronounced inflection point between $k=7$ and $k=9$.
2. **Silhouette Score:** Evaluates cohesion versus separation (bounded $[-1, 1]$). Peaked at $k=8$ ($\text{Score} = 0.142$).
3. **Davies-Bouldin Index:** Measures average similarity between each cluster and its most similar counterpart (lower is better). Reached its minimum at $k=8$ ($\text{DB} = 2.114$).

**Conclusion:** The multi-metric consensus identifies **$k=8$** as the definitive natural partition.

### 8.3 Initialization Stability Analysis
To ensure solutions were not artifacts of lucky initializations, we executed K-Means with random seeds `[21, 42, 100, 123]`. The pairwise Adjusted Rand Index across independent runs averaged **$0.892 \pm 0.041$**, indicating high algorithmic stability and reproducible cluster boundaries.

---

## 9. Centroid Exemplar Extraction & Cluster Interpretation

Without inspecting ground-truth labels, we extracted the 10 images with minimum Euclidean distance to each cluster centroid:
$$i^* = \arg\min_i \|\mathbf{z}_i - \boldsymbol{\mu}_c\|_2$$

Visual qualitative inspection of these exemplars enabled descriptive naming of all eight discovered clusters:
- **Cluster 0:** *Architectural Structures & Facades* (Buildings, urban masonry, geometric lines).
- **Cluster 1:** *Aquatic Vessels & Watercraft* (Boats, sea reflections, open horizons).
- **Cluster 2:** *Domestic Quadrupeds* (Cats and dogs in domestic, carpet, and lawn settings).
- **Cluster 3:** *Botanical Flora & Blossoms* (Flowers, colorful petals, outdoor greenery).
- **Cluster 4:** *Heavy Transit & Commercial Vehicles* (Buses, motorcoaches, large street silhouettes).
- **Cluster 5:** *Avian Wildlife* (Birds perched on branches, flight postures, plumage).
- **Cluster 6:** *Passenger Automobiles* (Sedans, sports cars, road wheels).
- **Cluster 7:** *Animal Facial Close-ups & Textured Fur* (Macro animal portraits, facial details).

---

## 10. Post-Hoc External Validation

Only after freezing the model, hyperparameter selections, and cluster assignments were the ground-truth labels loaded.

### 10.1 Quantitative Scorecard
- **Adjusted Rand Index (ARI):** **0.441**  
  (Significantly exceeds 0.0 chance baseline; reflects strong clustering agreement).
- **Normalized Mutual Information (NMI):** **0.582**  
  (Demonstrates strong shared mutual information between discovered clusters and true classes).

### 10.2 Contingency Heatmap Analysis
The normalized contingency matrix reveals key insights:
1. **High-Purity Clusters:**
   - *Flowers* isolated into Cluster 3 with **88.3% purity**.
   - *Buildings* isolated into Cluster 0 with **79.7% purity**.
   - *Boats* isolated into Cluster 1 with **74.0% purity**.
   - *Birds* isolated into Cluster 5 with **71.3% purity**.
2. **Semantic Quadruped Merging:**
   - *Cats* and *Dogs* split across Cluster 2 (body postures) and Cluster 7 (facial close-ups), but did not separate cleanly from one another. Unsupervised features grouped them by shared biological morphology (four legs, fur, domestic contexts) rather than human linguistic categories.
3. **Vehicle Co-clustering:**
   - *Cars* and *Buses* shared assignments in Cluster 4 and Cluster 6, reflecting shared vehicular geometry (wheels, windshields, asphalt roads).

---

## 11. Sensitivity Analysis: Deep CNN vs Color Histograms

To evaluate the contribution of deep semantic abstractions versus raw color palettes, we replicated the pipeline using 3D color histograms:

| Metric | ResNet50 GAP + PCA-95% | 3D Color Histograms (RGB 512-d) | Difference |
|---|---|---|---|
| **Silhouette Score** | **0.142** | 0.088 | +61.4% |
| **Davies-Bouldin Index** | **2.114** | 3.421 | -38.2% (Lower is Better) |
| **Adjusted Rand Index (ARI)** | **0.441** | **0.063** | **+600.0%** |
| **Normalized Mutual Info (NMI)** | **0.582** | **0.141** | **+312.8%** |

**Conclusion:** Color histograms fail to capture object categories ($\text{ARI} = 0.063$, near random chance), clustering images instead by dominant background illumination (e.g., sunny sky vs green grass). In contrast, deep CNN representations successfully discover semantic categories by leveraging hierarchical spatial abstractions.

---

## 12. Discussion & Research Findings

Returning to our central research question:
> *"Can an unsupervised learning algorithm discover meaningful natural groupings of different object categories using visual image features without being provided with category labels?"*

The empirical answer is an unequivocal **yes**. The unsupervised pipeline achieved an ARI of 0.441 and an NMI of 0.582 without ever seeing a label. 

Critically, the discrepancies between discovered clusters and ground truth are themselves scientifically informative:
1. **Unsupervised systems group by physical appearance, not semantic labels.** A dog and a cat look substantially more similar to each other than either looks to a building or a boat. Co-clustering them is a demonstration of sound perceptual geometry, not algorithmic failure.
2. **Background Context Influence:** Natural images carry contextual co-occurrences (e.g., boats on water, cars on asphalt). The network clusters the complete visual scene rather than isolated object bounding boxes.

---

## 13. Limitations

1. **Pretrained Supervised Representation Bias:** Although clustering was completely unsupervised, ResNet50 was pretrained on ImageNet-1K with supervised cross-entropy loss. Future work should evaluate self-supervised representations (e.g., DINOv2 or SimCLR).
2. **K-Means Geometric Assumption:** K-Means assumes spherical, isotropic clusters with equal variance. Visual manifolds often have complex topological curvature that non-parametric methods (e.g., HDBSCAN) may model more faithfully.
3. **Fixed Cluster Count ($k$):** Partitioning into a fixed $k$ forces ambiguous or transitional images into existing centroids.

---

## 14. Ethical Considerations & Responsible AI

1. **Geographic and Cultural Bias:** Google Open Images draws primarily from Flickr, which historically reflects Western domestic environments, temperate climates, and daylight photography.
2. **Privacy & Incidental Subjects:** Real-world street scenes (*Bus*, *Building*, *Car*) contain incidental bystanders and vehicle identifiers, requiring compliance with Creative Commons licensing and privacy regulations.
3. **Epistemic Clarity:** Discovered clusters should never be treated as authoritative ground-truth classifications; they represent statistical similarities within a specific feature space.

---

## 15. Conclusion & Future Work

This project established that unsupervised machine learning operating on deep CNN features can discover natural object categories from raw images without human labels. Dimensionality reduction via PCA compressed 2048 dimensions to 320 while retaining 95% variance, and t-SNE effectively mapped non-linear manifold topologies. K-Means identified $k=8$ as the optimal partition with high initialization stability. Post-hoc validation proved strong alignment with ground-truth categories while demonstrating natural cross-category visual blending.

### Future Scope:
- Evaluating fully self-supervised foundational representations (e.g., DINOv2, CLIP without text).
- Exploring density-based clustering algorithms (HDBSCAN) to discover arbitrary non-spherical visual manifolds.
- Scaling to 100+ categories using hierarchical agglomerative clustering.

---

## 16. References

1. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep residual learning for image recognition. *CVPR*, 770-778.
2. Kuznetsova, A., et al. (2020). The Open Images Dataset V4: Unified image classification, object detection, and visual relationship detection. *IJCV*, 128(7), 1956-1981.
3. van der Maaten, L., & Hinton, G. (2008). Visualizing data using t-SNE. *JMLR*, 9(Nov), 2579-2605.
4. Arthur, D., & Vassilvitskii, S. (2007). k-means++: The advantages of careful seeding. *SODA*, 1027-1035.
5. Rousseeuw, P. J. (1987). Silhouettes: a graphical aid to the interpretation and validation of cluster analysis. *CAM*, 20, 53-65.
6. Hubert, L., & Arabie, P. (1985). Comparing partitions. *Journal of Classification*, 2(1), 193-218.
