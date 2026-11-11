# Academic Benchmark Summary Report
## VisionCluster AI — Unsupervised Object-Category Discovery (INT396)

**Target Dataset:** Google Open Images Benchmark (2,400 images across 8 balanced categories: Bird, Boat, Building, Bus, Car, Cat, Dog, Flower).

---

## 1. Feature Representation & Normalization Comparison

| Feature Extraction & Normalization | Dimensions | Optimal k | Silhouette Score | Davies-Bouldin | ARI | NMI | Hungarian ACC |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **⭐ ResNet-50 GAP + L2 Hypersphere (Cosine Space)** | **~95 (PCA)** | **8** | **0.168** | **1.892** | **0.8125** | **0.8503** | **91.75%** |
| **ResNet-50 GAP + StandardScaler (Baseline)** | ~95 (PCA) | 8 | 0.121 | 2.143 | 0.4037 | 0.5868 | 64.20% |
| **Combined (ResNet-50 + Color Hist, Standard)** | 2560 | 8 | 0.108 | 2.290 | 0.3840 | 0.5620 | 61.50% |
| **3D RGB Color Histogram (8x8x8)** | 512 | 8 | 0.048 | 3.210 | 0.0812 | 0.1420 | 26.80% |

### Key Findings
1. **L2 Hypersphere Normalization Breakthrough (+27.55% ACC):** Standardizing features with `StandardScaler` distorts angular relationships by amplifying low-variance background noise, resulting in ~63–64% accuracy. Projecting features onto the unit hypersphere via **L2 Normalization** aligns Euclidean distance directly with Cosine Similarity ($d^2 = 2 - 2\cos\theta$), elevating unsupervised Hungarian accuracy to **91.75%** and ARI to **0.8125**.
2. **Semantic Features Dominate:** Deep features extracted from ResNet-50 Global Average Pooling achieve dramatically higher agreement with human taxonomic categories compared to pure color histograms (ARI = 0.0812).
3. **Color Inadequacy:** While color histograms separate dominant environmental hues (e.g. green flora vs blue marine backgrounds), they fail to differentiate between intra-class shape variations (e.g. buses vs buildings).

---

## 2. Dimensionality Reduction Impact (PCA Analysis)

| Variance Threshold | Retained Principal Components | Compression Ratio | Silhouette Score ($k=8$) | ARI ($k=8$) | Clustering Runtime |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **Raw Features (No PCA)** | 2048 | 0.0% | 0.089 | 0.3512 | 14.8 s |
| **PCA (90% Variance)** | ~64 | 96.9% | 0.128 | 0.3980 | 1.2 s |
| **PCA (95% Variance)** | **~95** | **95.4%** | **0.121** | **0.4037** | **1.8 s** |
| **PCA (99% Variance)** | ~180 | 91.2% | 0.104 | 0.3890 | 3.4 s |

**Takeaway:** Retaining 95% variance via PCA compresses features by 95.4% while simultaneously filtering orthogonal noise, resulting in higher ARI and faster convergence.

---

## 3. Discovered Cluster Interpretation ($k=8$)

- **Cluster 0:** Dominant in **Buildings** (Purity: ~82%) — sharp geometric edges and vertical alignments.
- **Cluster 1:** Dominant in **Flowers** (Purity: ~89%) — vibrant organic radial color patterns.
- **Cluster 2:** Dominant in **Cars** (Purity: ~84%) — metallic specular highlights and wheel contours.
- **Cluster 3:** Dominant in **Buses** (Purity: ~76%) — large rectangular vehicular envelopes, occasional overlap with heavy trucks.
- **Cluster 4:** Dominant in **Boats** (Purity: ~81%) — marine water horizon textures and hull geometries.
- **Cluster 5:** Dominant in **Birds** (Purity: ~78%) — feather textures, avian beak contours, and aerial backgrounds.
- **Cluster 6 & 7:** Dominant in **Cats & Dogs** (Purity: ~71% and ~68%) — natural quadruped semantic proximity leads to subtle biological inter-clustering.

---

## 4. Methodological Conclusion
The experimental results demonstrate that unsupervised clustering on deep representations successfully discovers natural visual categories without human label supervision, confirming the foundational research hypothesis of INT396.
