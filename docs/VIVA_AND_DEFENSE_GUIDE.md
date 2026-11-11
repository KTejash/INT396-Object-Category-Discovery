# INT396 Academic Defense & Viva Voce Guide
## Natural Object-Category Discovery from Image Features Using Clustering

> **Course:** INT396 — Natural Object-Category Discovery  
> **Platform:** VisionCluster AI (Object Discovery Lab)  
> **Authors:** Tejash Kumar Singh ([@KTejash](https://github.com/KTejash)) & Gundumalla PurnaChandra Santosh ([@GPCSantosh](https://github.com/GPCSantosh))

---

## 🎯 Executive Project Summary

This project investigates whether an unsupervised learning system can discover natural visual object categories (Bird, Boat, Building, Bus, Car, Cat, Dog, Flower) without human supervision or category labels.

* **Dataset:** Google Open Images Benchmark (2,400 images across 8 balanced categories).
* **Feature Representation:** Pretrained ResNet-50 Global Average Pooling (2048-d) vs 3D RGB Color Histograms (512-d).
* **Normalization Breakthrough:** L2 Hypersphere Normalization ($\|\mathbf{x}\|_2 = 1$).
* **Dimensionality Reduction:** Orthogonal PCA (95% variance retention) & t-SNE non-linear manifold projection.
* **Clustering Engine:** K-Means ($k=8$, $n\_init=25$, K-Means++ initialization).
* **Peak Unsupervised Accuracy:** **91.75%** (Hungarian Optimal Bipartite Matching), **ARI = 0.8125**, **NMI = 0.8503**.

---

## ❓ High-Probability Viva Questions & Model Answers

### Q1: Why did baseline accuracy plateau at ~63% and how did you elevate it to 91.8%?
**Answer:**
* **The Root Cause:** The original baseline used `StandardScaler` (zero mean, unit variance). In high-dimensional deep neural network representations (ResNet-50 GAP), semantic concepts are encoded as **angular directions** (cosine similarity), not raw Euclidean magnitudes. `StandardScaler` scales each dimension independently, blowing up low-variance background dimensions (sky, grass, asphalt lighting) and blurring fine boundaries between similar quadrupeds (Cats vs Dogs).
* **The Mathematical Fix:** We implemented **L2 Hypersphere Normalization**:
  $$\mathbf{x}_{\text{norm}} = \frac{\mathbf{x}}{\|\mathbf{x}\|_2}$$
  In this spherical space, Euclidean distance between two vectors is mathematically equivalent to **Cosine Distance**:
  $$\|\mathbf{u} - \mathbf{v}\|^2 = 2 - 2 \cos(\mathbf{u}, \mathbf{v})$$
  This eliminates scale and lighting magnitude distortions, allowing K-Means to cluster strictly by semantic orientation. This single optimization lifted clustering accuracy from **63.5% → 91.75%** and ARI from **0.32 → 0.8125**.

---

### Q2: Why did you use K-Means instead of DBSCAN or Agglomerative Clustering?
**Answer:**
1. **DBSCAN Limitations:** DBSCAN relies on a global density threshold (`eps`, `min_samples`). In 2048-dimensional spaces (or 95-d PCA space), high-dimensional sparsity causes distance concentration, making global density estimation ineffective. DBSCAN also labels sparse peripheral samples as noise rather than partitioning all data.
2. **Agglomerative Clustering:** Has $O(N^2)$ memory and $O(N^3)$ computational complexity, scaling poorly to large collections.
3. **K-Means Strengths:** Combined with K-Means++ initialization and L2 normalization (Spherical K-Means), it converges rapidly, scales to thousands of images, and provides explicit centroid exemplars for cluster interpretability.

---

### Q3: How do you guarantee the principle of unsupervised learning?
**Answer:**
* **Strict Label Isolation:** Ground-truth labels are stored in a physically separate file (`data/metadata/ground_truth.csv`).
* During image ingestion, feature extraction, normalization, PCA, t-SNE, and K-Means clustering, the algorithms receive **only raw image pixels and anonymous image IDs**.
* Ground-truth labels are **unlocked strictly post-hoc** in Page 11 (External Validation) purely to evaluate how well unsupervised clusters correspond to human taxonomic classifications.

---

### Q4: What is the Hungarian Matching Algorithm and why is it needed?
**Answer:**
Unsupervised clustering outputs arbitrary cluster indices (e.g. Cluster 0, Cluster 1 ... Cluster 7). It does not know that Cluster 0 corresponds to "Building" or Cluster 3 to "Flower".
The **Hungarian Algorithm (Kuhn-Munkres Algorithm)** solves the minimum weight bipartite matching problem on the contingency cost matrix in $O(k^3)$ time:
$$\text{ACC} = \max_{m \in \Pi} \frac{1}{N} \sum_{i=1}^N \mathbf{1}\left(y_i = m(c_i)\right)$$
Where $\Pi$ represents all permutations of 1-to-1 mappings between cluster IDs $c_i$ and human labels $y_i$. It finds the optimal global alignment without human bias.

---

### Q5: Why is ResNet-50 GAP superior to Color Histograms?
**Answer:**
* **Color Histograms (512-d):** Only capture low-level chromatic distribution. They group green plants together with green cars, or blue boats with blue sky, achieving an ARI of only **0.0812** and ACC of **26.8%**.
* **ResNet-50 GAP (2048-d):** Hierarchical convolution filters capture invariant visual features (edges $\to$ textures $\to$ object parts $\to$ semantic entities). Global Average Pooling (GAP) enforces spatial translation invariance, capturing *what* the object is rather than *where* or *what color* it is.

---

### Q6: What role does PCA play in the pipeline?
**Answer:**
* ResNet-50 features have 2048 dimensions. Many dimensions share co-linear covariance.
* Using PCA with a **95% retained variance threshold**, the dimensionality is compressed from 2048 down to **~95 principal components** (a 95.4% reduction).
* This removes orthogonal noise, eliminates the curse of dimensionality, accelerates K-Means convergence by ~8x, and preserves the essential semantic structure.

---

### Q7: What are the internal vs external evaluation metrics used?
**Answer:**
* **Internal Metrics (No labels required):**
  * **Silhouette Score:** Ratio of inter-cluster separation to intra-cluster tightness $[-1.0, 1.0]$.
  * **Davies-Bouldin Index:** Similarity between each cluster and its most similar counterpart (lower is better).
  * **Calinski-Harabasz Score:** Ratio of between-cluster variance to within-cluster variance.
  * **WCSS (Elbow Method):** Within-cluster sum of squared distances.
* **External Metrics (Post-hoc ground-truth unlocked):**
  * **Adjusted Rand Index (ARI):** Measures pairwise agreement corrected for chance $[-1.0, 1.0]$.
  * **Normalized Mutual Information (NMI):** Information-theoretic dependency $[0.0, 1.0]$.
  * **Hungarian Clustering Accuracy (ACC):** 1-to-1 optimal mapped accuracy ($91.75\%$).

---

## 📋 Key Numerical Highlights for Presentation Slides

```text
• Benchmark Sample Size: 2,400 balanced images (8 classes x 300 images)
• Raw Feature Vector: 2048-d (ResNet-50 Global Average Pooling)
• PCA Compression: 95.4% reduction (~95 components retained at 95% variance)
• Peak Unsupervised Accuracy: 91.75%
• Peak Adjusted Rand Index (ARI): 0.8125
• Peak Normalized Mutual Info (NMI): 0.8503
• Universal Inference: 1000-class ImageNet Real-Time Prediction Studio
```
