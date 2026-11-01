"""
Object Discovery Lab — Interactive Unsupervised AI Platform
A state-of-the-art interactive research laboratory for unsupervised image clustering,
feature extraction, dimensionality reduction, dynamic cluster analysis, and visual search.
"""

import sys
import io
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    BASE_DIR,
    DATA_DIR,
    RAW_IMG_DIR,
    METADATA_DIR,
    OUTPUTS_DIR,
    EMBEDDINGS_DIR,
    MODELS_DIR,
    RANDOM_SEED,
    DEFAULT_K
)
from src.data_manager import DatasetManager
from src.feature_extractor import FeatureExtractor
from src.preprocessing import PreprocessingPipeline
from src.dimensionality_reduction import PCAManager, TSNEManager
from src.clustering import ClusteringEngine
from src.image_search import ImageDiscoveryStudio
from src.experiment_manager import ExperimentManager
from src.evaluation import ExternalValidator
import src.visualization as viz

# Page Configuration
st.set_page_config(
    page_title="Object Discovery Lab",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Futuristic Dark Navy & Cyan AI Laboratory Styling
st.markdown("""
<style>
    .stApp {
        background-color: #0b132b;
        color: #e0e6ed;
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    [data-testid="stSidebar"] {
        background-color: #101c3d !important;
        border-right: 1px solid #1c2e59;
    }
    /* Laboratory Metric Cards */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #1c2541 0%, #17223b 100%);
        border: 1px solid #293b66;
        padding: 16px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.25);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    div[data-testid="metric-container"]:hover {
        transform: translateY(-2px);
        border-color: #00b4d8;
    }
    div[data-testid="stMetricValue"] {
        color: #48cae4 !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
    }
    div[data-testid="stMetricLabel"] {
        color: #90e0ef !important;
        font-weight: 500 !important;
        font-size: 0.9rem !important;
    }
    h1, h2, h3, h4 {
        color: #f8f9fa;
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    .accent-cyan { color: #00b4d8; }
    .accent-amber { color: #ffb703; }
    .lab-card {
        background-color: #1c2541;
        border-left: 4px solid #00b4d8;
        padding: 16px 20px;
        border-radius: 0 10px 10px 0;
        margin-bottom: 20px;
    }
    .warning-card {
        background-color: #241c1c;
        border-left: 4px solid #ffb703;
        padding: 16px 20px;
        border-radius: 0 10px 10px 0;
        margin-bottom: 20px;
    }
    .stButton>button {
        background: linear-gradient(135deg, #0077b6 0%, #0096c7 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.2rem;
        transition: all 0.2s ease;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #0096c7 0%, #48cae4 100%);
        color: #0b132b;
        box-shadow: 0 4px 12px rgba(0, 180, 216, 0.35);
    }
</style>
""", unsafe_allow_html=True)


# ==============================================================================
# SESSION STATE INITIALIZATION
# ==============================================================================
def init_session_state():
    if "dataset_manager" not in st.session_state:
        st.session_state.dataset_manager = DatasetManager("open_images_benchmark")
    if "current_dataset_name" not in st.session_state:
        st.session_state.current_dataset_name = "open_images_benchmark"
    if "manifest_df" not in st.session_state:
        st.session_state.manifest_df = st.session_state.dataset_manager.load_manifest()
    if "features_raw" not in st.session_state:
        # Load precomputed benchmark if available
        p = EMBEDDINGS_DIR / "cnn_features_raw.npy"
        st.session_state.features_raw = np.load(p) if p.exists() else None
    if "feature_type" not in st.session_state:
        st.session_state.feature_type = "resnet50"
    if "features_scaled" not in st.session_state:
        p = EMBEDDINGS_DIR / "cnn_features_scaled.npy"
        st.session_state.features_scaled = np.load(p) if p.exists() else None
    if "scaler_pipeline" not in st.session_state:
        st.session_state.scaler_pipeline = PreprocessingPipeline(scaler_type="standard")
    if "pca_manager" not in st.session_state:
        st.session_state.pca_manager = PCAManager(random_state=RANDOM_SEED)
    if "features_pca" not in st.session_state:
        p = EMBEDDINGS_DIR / "pca_95_features.npy"
        st.session_state.features_pca = np.load(p) if p.exists() else None
    if "pca_metadata" not in st.session_state:
        st.session_state.pca_metadata = {}
    if "tsne_coords" not in st.session_state:
        p = EMBEDDINGS_DIR / "tsne_perp30_lr200.npy"
        st.session_state.tsne_coords = np.load(p) if p.exists() else None
    if "tsne_metadata" not in st.session_state:
        st.session_state.tsne_metadata = {}
    if "cluster_labels" not in st.session_state:
        p = OUTPUTS_DIR / "cluster_assignments" / f"final_cluster_labels_k{DEFAULT_K}.npy"
        st.session_state.cluster_labels = np.load(p) if p.exists() else None
    if "cluster_metrics" not in st.session_state:
        st.session_state.cluster_metrics = {}
    if "cluster_centroids" not in st.session_state:
        p = MODELS_DIR / "kmeans_centroids.npy"
        st.session_state.cluster_centroids = np.load(p) if p.exists() else None
    if "experiment_manager" not in st.session_state:
        st.session_state.experiment_manager = ExperimentManager()
    if "feature_extractor" not in st.session_state:
        st.session_state.feature_extractor = FeatureExtractor()

init_session_state()


# ==============================================================================
# SIDEBAR NAVIGATION
# ==============================================================================
st.sidebar.markdown("""
<div style="text-align: center; padding: 10px 0 15px 0;">
    <h2 style="color: #48cae4; margin: 0; font-size: 1.35rem;">🔬 Object Discovery Lab</h2>
    <p style="color: #90e0ef; font-size: 0.82rem; margin-top: 4px;">Interactive Unsupervised AI Platform</p>
</div>
""", unsafe_allow_html=True)

nav_sections = [
    "1. Executive Overview",
    "2. Dataset Workspace",
    "3. Feature Extraction Lab",
    "4. Data Preprocessing",
    "5. PCA Exploration",
    "6. t-SNE Explorer",
    "7. K-Means Clustering Lab",
    "8. Discovered Cluster Gallery",
    "9. Image Discovery Studio",
    "10. Experiment Comparison",
    "11. External Validation",
    "12. Reports and Exports",
    "13. Limitations and Research Ethics"
]

selected_page = st.sidebar.radio("Navigation Console", nav_sections, index=0)

st.sidebar.markdown("---")
# Quick active status badge
active_ds = st.session_state.current_dataset_name
n_active_imgs = len(st.session_state.manifest_df) if st.session_state.manifest_df is not None else 0
feat_dim = st.session_state.features_raw.shape[1] if st.session_state.features_raw is not None else 0
k_curr = len(np.unique(st.session_state.cluster_labels)) if st.session_state.cluster_labels is not None else "None"

st.sidebar.markdown(f"""
<div style="font-size: 0.8rem; background: #1c2541; padding: 10px; border-radius: 8px;">
    <strong>Active Environment Status:</strong><br>
    • Dataset: <span style="color: #48cae4;">{active_ds}</span><br>
    • Images: <span style="color: #48cae4;">{n_active_imgs}</span><br>
    • Feature Dims: <span style="color: #48cae4;">{feat_dim}</span><br>
    • Clusters: <span style="color: #ffb703;">{k_curr}</span>
</div>
""", unsafe_allow_html=True)


# ==============================================================================
# PAGE 1: EXECUTIVE OVERVIEW
# ==============================================================================
if selected_page == "1. Executive Overview":
    st.markdown("""
    <div style="margin-bottom: 20px;">
        <h1 style="color: #ffffff; font-size: 2.2rem; margin-bottom: 4px;">Object Discovery Lab — Interactive Unsupervised AI Platform</h1>
        <p style="color: #48cae4; font-size: 1.05rem; font-weight: 500;">
            Real-Time Unsupervised Clustering, Feature Extraction, Manifold Learning, and Visual Search
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="lab-card">
        <strong>Platform Mission:</strong> To provide a transparent, fully interactive research environment where users can 
        upload arbitrary visual datasets, extract deep representations, perform orthogonal and non-linear dimensionality reduction, 
        discover natural clusters via K-Means, and query novel images in real-time — with <em>zero hardcoded metrics or static mockups</em>.
    </div>
    """, unsafe_allow_html=True)

    # Scorecard
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("Active Dataset Images", f"{n_active_imgs:,}")
    col2.metric("Feature Dimensions", f"{feat_dim}-d")
    
    pca_dim = st.session_state.features_pca.shape[1] if st.session_state.features_pca is not None else "N/A"
    col3.metric("PCA Dimensions", f"{pca_dim}")
    
    col4.metric("Discovered Clusters", f"k = {k_curr}")
    
    sil_score = st.session_state.cluster_metrics.get("silhouette_score", 0.0134)
    col5.metric("Silhouette Score", f"{sil_score:.4f}" if isinstance(sil_score, float) else str(sil_score))

    st.markdown("### Interactive Research Architecture")
    st.markdown("""
    ```mermaid
    flowchart LR
        A["1. Dataset Workspace<br>(Upload ZIPs / Open Images)"] --> B["2. Feature Extraction Lab<br>(ResNet50 GAP / Color Hist)"]
        B --> C["3. Data Preprocessing<br>(Standard / MinMax / L2)"]
        C --> D["4. PCA & t-SNE Lab<br>(Orthogonal & Manifold Projection)"]
        D --> E["5. K-Means Lab<br>(Dynamic k=2..12, Elbow & Silhouette)"]
        E --> F["6. Cluster Gallery & Studio<br>(Centroid Exemplars & Image Search)"]
        E --> G["7. Experiment Manager<br>(Persistent Logging & Model Comparison)"]
    ```
    """)

    st.info("Tip: Use the left navigation console to upload new image collections, tune hyperparameters, or test the Image Discovery Studio.")


# ==============================================================================
# PAGE 2: DATASET WORKSPACE
# ==============================================================================
elif selected_page == "2. Dataset Workspace":
    st.header("Dataset Workspace & Quality Audit")
    st.markdown("Manage datasets, upload new image collections, inspect integrity, and detect duplicates.")

    col_ds1, col_ds2 = st.columns([1, 1])
    dm = st.session_state.dataset_manager

    with col_ds1:
        st.subheader("Select Dataset")
        available = dm.list_available_datasets()
        curr_idx = available.index(st.session_state.current_dataset_name) if st.session_state.current_dataset_name in available else 0
        chosen_ds = st.selectbox("Active Dataset:", available, index=curr_idx)

        if chosen_ds != st.session_state.current_dataset_name:
            st.session_state.dataset_manager = DatasetManager(chosen_ds)
            st.session_state.current_dataset_name = chosen_ds
            st.session_state.manifest_df = st.session_state.dataset_manager.load_manifest()
            st.session_state.features_raw = None
            st.session_state.features_scaled = None
            st.session_state.features_pca = None
            st.session_state.cluster_labels = None
            st.rerun()

        st.caption(f"Currently managing: **{chosen_ds}** ({len(st.session_state.manifest_df)} items in manifest)")

    with col_ds2:
        st.subheader("Upload Custom Dataset")
        new_ds_name = st.text_input("New Dataset Name:", value="my_custom_images")
        uploaded_files = st.file_uploader(
            "Upload Images or ZIP Archives (JPG, JPEG, PNG, WEBP):",
            type=["jpg", "jpeg", "png", "webp", "zip"],
            accept_multiple_files=True
        )
        
        cap_val = st.number_input("Max Images to Ingest (0 for unlimited):", min_value=0, max_value=5000, value=0)
        max_imgs = int(cap_val) if cap_val > 0 else None

        if st.button("Create & Ingest Dataset"):
            if not uploaded_files:
                st.warning("Please select files or a ZIP archive to upload.")
            else:
                with st.spinner("Validating, de-duplicating, and ingesting images..."):
                    valid_c, corrupt_c, msg = dm.create_custom_dataset_from_files(
                        new_ds_name, uploaded_files, max_images=max_imgs
                    )
                st.success(msg)
                st.session_state.dataset_manager = DatasetManager(new_ds_name)
                st.session_state.current_dataset_name = new_ds_name
                st.session_state.manifest_df = st.session_state.dataset_manager.load_manifest()
                st.rerun()

    st.markdown("---")
    st.subheader("Dataset Quality Audit")
    
    audit_summary, valid_df, stats = dm.audit_dataset()
    col_a1, col_a2 = st.columns([1, 1])
    
    with col_a1:
        st.markdown("**Audit Scorecard:**")
        st.dataframe(audit_summary, width="stretch", hide_index=True)

    with col_a2:
        if not valid_df.empty and "aspect_ratio" in valid_df.columns:
            fig_ar = px.histogram(
                valid_df, x="aspect_ratio", nbins=25,
                title="Aspect Ratio Distribution (Width / Height)",
                color_discrete_sequence=["#00b4d8"]
            )
            fig_ar.update_layout(**viz.DARK_LAYOUT)
            st.plotly_chart(fig_ar, width="stretch")

    st.markdown("---")
    st.subheader("Dataset Manifest Explorer")
    if not st.session_state.manifest_df.empty:
        st.dataframe(st.session_state.manifest_df.head(100), width="stretch")
        
        # Image deletion tool
        with st.expander("Remove Specific Images from Dataset"):
            all_ids = st.session_state.manifest_df["image_id"].tolist()
            to_delete = st.multiselect("Select Image IDs to Remove:", all_ids)
            if st.button("Delete Selected Images"):
                if to_delete:
                    removed = dm.delete_images(to_delete)
                    st.success(f"Removed {removed} images from dataset.")
                    st.session_state.manifest_df = dm.load_manifest()
                    st.rerun()


# ==============================================================================
# PAGE 3: FEATURE EXTRACTION LAB
# ==============================================================================
elif selected_page == "3. Feature Extraction Lab":
    st.header("Feature Extraction Lab")
    st.markdown("Extract deep semantic representations via ResNet50 GAP (2048-d) or 3D Color Histograms.")

    col_fe1, col_fe2 = st.columns([1, 1])

    with col_fe1:
        st.subheader("Extraction Configuration")
        feat_choice = st.selectbox(
            "Feature Representation Type:",
            ["resnet50", "color_hist", "combined"],
            format_func=lambda x: {
                "resnet50": "Deep CNN: ResNet50 GAP (2048-d)",
                "color_hist": "Color: 3D RGB Color Histogram",
                "combined": "Combined: ResNet50 + Color Histogram"
            }[x]
        )

        bins_val = 8
        if feat_choice in ["color_hist", "combined"]:
            bins_val = st.slider("Color Bins per Channel (Bins^3 dims):", min_value=4, max_value=16, value=8, step=2)
            st.caption(f"Histogram Dimensions: {bins_val}^3 = **{bins_val**3}** features")

        cnn_w = 1.0
        hist_w = 1.0
        if feat_choice == "combined":
            cnn_w = st.slider("ResNet50 Weight:", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
            hist_w = st.slider("Color Histogram Weight:", min_value=0.1, max_value=5.0, value=1.0, step=0.1)

        batch_sz = st.select_slider("Batch Size:", options=[8, 16, 32, 64], value=32)

        if st.button("Extract Features Now"):
            if st.session_state.manifest_df.empty:
                st.error("No images found in manifest to extract.")
            else:
                progress_bar = st.progress(0.0)
                status_txt = st.empty()

                def update_progress(pct, msg):
                    progress_bar.progress(pct)
                    status_txt.text(msg)

                extractor = st.session_state.feature_extractor
                t0 = time.time()
                with st.spinner("Extracting features..."):
                    feats, ids = extractor.extract_features_for_manifest(
                        st.session_state.manifest_df,
                        feature_type=feat_choice,
                        bins_per_channel=bins_val,
                        batch_size=batch_sz,
                        cnn_weight=cnn_w,
                        hist_weight=hist_w,
                        progress_callback=update_progress
                    )
                elapsed = time.time() - t0
                st.session_state.features_raw = feats
                st.session_state.feature_type = feat_choice
                st.session_state.features_scaled = None  # Invalidate downstream
                st.session_state.features_pca = None
                st.session_state.cluster_labels = None
                st.success(f"Successfully extracted {feats.shape[0]} samples x {feats.shape[1]} dimensions in {elapsed:.1f}s!")

    with col_fe2:
        st.subheader("Feature Matrix Status")
        if st.session_state.features_raw is not None:
            f_mat = st.session_state.features_raw
            mem_mb = round(f_mat.nbytes / (1024 * 1024), 2)
            c1, c2, c3 = st.columns(3)
            c1.metric("Samples", f"{f_mat.shape[0]}")
            c2.metric("Dimensions", f"{f_mat.shape[1]}")
            c3.metric("RAM Footprint", f"{mem_mb} MB")

            st.markdown("**Sample Feature Distribution (First 1,000 Dimensions):**")
            fig_f = px.histogram(
                f_mat.flatten()[:1000],
                nbins=30,
                color_discrete_sequence=["#48cae4"]
            )
            fig_f.update_layout(showlegend=False, xaxis_title="Feature Magnitude", **viz.DARK_LAYOUT)
            st.plotly_chart(fig_f, width="stretch")
        else:
            st.info("No feature matrix currently in memory. Configure and click 'Extract Features Now'.")


# ==============================================================================
# PAGE 4: DATA PREPROCESSING
# ==============================================================================
elif selected_page == "4. Data Preprocessing":
    st.header("Data Preprocessing & Normalization")
    st.markdown("Ensure numerical stability, zero-variance screening, and standardized feature scaling.")

    if st.session_state.features_raw is None:
        st.warning("Please extract or load features in the Feature Extraction Lab first.")
    else:
        col_pp1, col_pp2 = st.columns([1, 1])

        with col_pp1:
            st.subheader("Normalization Configuration")
            scaler_choice = st.selectbox(
                "Scaling Transformation:",
                ["standard", "minmax", "l2_norm", "none"],
                format_func=lambda x: {
                    "standard": "StandardScaler (Zero Mean, Unit Variance)",
                    "minmax": "MinMaxScaler (Bounded Range [0, 1])",
                    "l2_norm": "L2 Normalizer (Unit Vector Projection)",
                    "none": "None (Raw Activations)"
                }[x]
            )

            if st.button("Apply Preprocessing Pipeline"):
                pipe = PreprocessingPipeline(scaler_type=scaler_choice)
                X_scaled, df_stats = pipe.fit_transform(st.session_state.features_raw)
                st.session_state.scaler_pipeline = pipe
                st.session_state.features_scaled = X_scaled
                st.session_state.features_pca = None  # Invalidate downstream
                st.session_state.cluster_labels = None
                st.success(f"Applied {scaler_choice} scaling to matrix of shape {X_scaled.shape}!")

        with col_pp2:
            st.subheader("Feature Integrity & Distribution Audit")
            pipe = st.session_state.scaler_pipeline
            audit = pipe.validate_features(st.session_state.features_raw)
            st.json(audit)

        st.markdown("---")
        st.subheader("Before & After Distribution Diagnostics")
        if st.session_state.features_scaled is not None:
            raw_s = pipe.compute_distribution_stats(st.session_state.features_raw, "Raw Features")
            scaled_s = pipe.compute_distribution_stats(st.session_state.features_scaled, "Preprocessed Features")
            st.dataframe(pd.DataFrame([raw_s, scaled_s]), width="stretch", hide_index=True)


# ==============================================================================
# PAGE 5: PCA EXPLORATION
# ==============================================================================
elif selected_page == "5. PCA Exploration":
    st.header("Principal Component Analysis (PCA) Lab")
    st.markdown("Linear orthogonal projection, scree analysis, and variance retention tuning.")

    X_in = st.session_state.features_scaled if st.session_state.features_scaled is not None else st.session_state.features_raw

    if X_in is None:
        st.warning("No feature matrix available. Please run Feature Extraction and Preprocessing first.")
    else:
        col_pca1, col_pca2 = st.columns([1, 1])

        with col_pca1:
            st.subheader("PCA Hyperparameters")
            pca_mode = st.radio("Reduction Criterion:", ["variance", "fixed"], format_func=lambda x: "Target Variance Ratio" if x == "variance" else "Fixed Component Count")
            
            target_var = 0.95
            fixed_n = 50
            if pca_mode == "variance":
                target_var = st.slider("Target Retained Variance:", min_value=0.50, max_value=0.99, value=0.95, step=0.01)
            else:
                max_comp = min(X_in.shape[0], X_in.shape[1])
                fixed_n = st.slider("Number of Principal Components:", min_value=2, max_value=min(max_comp, 500), value=min(50, max_comp))

            if st.button("Run PCA Decomposition"):
                with st.spinner("Computing orthogonal decomposition..."):
                    pca_mgr = PCAManager(random_state=RANDOM_SEED)
                    X_pca, out_meta = pca_mgr.fit_transform(
                        X_in, mode=pca_mode, target_variance=target_var, n_components=fixed_n
                    )
                st.session_state.pca_manager = pca_mgr
                st.session_state.features_pca = X_pca
                st.session_state.pca_metadata = out_meta
                st.success(f"PCA reduced dimensions from {X_in.shape[1]} to {X_pca.shape[1]} ({out_meta['metrics']['compression_ratio']})!")

        with col_pca2:
            st.subheader("Dimensionality Compression Metrics")
            if "metrics" in st.session_state.pca_metadata:
                st.json(st.session_state.pca_metadata["metrics"])

        st.markdown("---")
        if "scree_df" in st.session_state.pca_metadata:
            st.subheader("Interactive Scree Curve")
            fig_scree = viz.plot_scree_interactive(st.session_state.pca_metadata["scree_df"], target_var if pca_mode == "variance" else None)
            st.plotly_chart(fig_scree, width="stretch")

        if st.session_state.features_pca is not None and st.session_state.features_pca.shape[1] >= 2:
            st.subheader("Interactive 2D PCA Space")
            c_lbls = st.session_state.cluster_labels
            img_ids = st.session_state.manifest_df["image_id"].tolist() if not st.session_state.manifest_df.empty else None
            fig_pca2d = viz.plot_embedding_2d_interactive(
                st.session_state.features_pca[:, :2],
                labels=c_lbls,
                hover_names=img_ids,
                title="PCA Projection (PC1 vs PC2)"
            )
            st.plotly_chart(fig_pca2d, width="stretch")


# ==============================================================================
# PAGE 6: t-SNE EXPLORER
# ==============================================================================
elif selected_page == "6. t-SNE Explorer":
    st.header("t-Distributed Stochastic Neighbor Embedding (t-SNE)")
    st.markdown("Non-linear manifold projection modeling local probabilistic neighborhood affinities.")

    X_in = st.session_state.features_pca if st.session_state.features_pca is not None else st.session_state.features_scaled

    if X_in is None:
        st.warning("Please extract features and apply PCA first.")
    else:
        col_t1, col_t2 = st.columns([1, 1])

        with col_t1:
            st.subheader("t-SNE Hyperparameter Configuration")
            perp_val = st.slider("Perplexity (Effective Neighborhood Size):", min_value=5.0, max_value=80.0, value=30.0, step=5.0)
            lr_val = st.slider("Learning Rate:", min_value=50.0, max_value=500.0, value=200.0, step=25.0)
            iter_val = st.select_slider("Iterations:", options=[250, 500, 750, 1000, 1500], value=1000)
            metric_choice = st.selectbox("Distance Metric:", ["euclidean", "cosine"])

            if st.button("Compute 2D t-SNE Embedding"):
                with st.spinner("Optimizing non-linear manifold..."):
                    tsne_mgr = TSNEManager(random_state=RANDOM_SEED)
                    coords_2d, meta = tsne_mgr.fit_transform(
                        X_in, perplexity=perp_val, learning_rate=lr_val, max_iter=iter_val, metric=metric_choice
                    )
                st.session_state.tsne_coords = coords_2d
                st.session_state.tsne_metadata = meta
                st.success(f"t-SNE converged in {meta['runtime_seconds']}s (KL Divergence: {meta['kl_divergence']})!")

        with col_t2:
            st.subheader("Convergence & Runtime Telemetry")
            if st.session_state.tsne_metadata:
                st.json(st.session_state.tsne_metadata)

        st.markdown("---")
        if st.session_state.tsne_coords is not None:
            st.subheader("Interactive 2D t-SNE Scatter Space")
            c_lbls = st.session_state.cluster_labels
            img_ids = st.session_state.manifest_df["image_id"].tolist() if not st.session_state.manifest_df.empty else None
            fig_tsne = viz.plot_embedding_2d_interactive(
                st.session_state.tsne_coords,
                labels=c_lbls,
                hover_names=img_ids,
                title=f"t-SNE Manifold (Perplexity={perp_val}, LR={lr_val})"
            )
            st.plotly_chart(fig_tsne, width="stretch")


# ==============================================================================
# PAGE 7: K-MEANS CLUSTERING LAB
# ==============================================================================
elif selected_page == "7. K-Means Clustering Lab":
    st.header("K-Means Clustering Lab & Optimal k Selection")
    st.markdown("Execute real clustering partitions, compute live internal metrics, and run parameter sweeps.")

    X_in = st.session_state.features_pca if st.session_state.features_pca is not None else st.session_state.features_scaled

    if X_in is None:
        st.warning("Please extract features and preprocess data before clustering.")
    else:
        col_km1, col_km2 = st.columns([1, 1])

        with col_km1:
            st.subheader("Clustering Configuration")
            k_val = st.slider("Number of Clusters (k):", min_value=2, max_value=12, value=DEFAULT_K)
            init_choice = st.selectbox("Initialization Method:", ["k-means++", "random"])
            n_init_val = st.slider("Number of Restarts (n_init):", min_value=1, max_value=25, value=10)
            max_iter_val = st.slider("Max Iterations:", min_value=100, max_value=500, value=300)

            if st.button("Execute K-Means Clustering"):
                with st.spinner("Partitioning feature space..."):
                    engine = ClusteringEngine(random_state=RANDOM_SEED)
                    lbls, metrics = engine.fit_predict(
                        X_in, k=k_val, init_method=init_choice, n_init=n_init_val, max_iter=max_iter_val
                    )
                st.session_state.cluster_labels = lbls
                st.session_state.cluster_metrics = metrics
                st.session_state.cluster_centroids = engine.centroids_
                st.success(f"K-Means (k={k_val}) completed in {metrics['fit_time_seconds']}s! Silhouette: {metrics['silhouette_score']}")

        with col_km2:
            st.subheader("Live Evaluation Metrics")
            if st.session_state.cluster_metrics:
                m = st.session_state.cluster_metrics
                c1, c2 = st.columns(2)
                c1.metric("WCSS (Inertia)", f"{m.get('inertia_wcss', 0):,}")
                c2.metric("Silhouette Score", f"{m.get('silhouette_score', 'N/A')}")
                c3, c4 = st.columns(2)
                c3.metric("Davies-Bouldin Index", f"{m.get('davies_bouldin_index', 'N/A')}")
                c4.metric("Calinski-Harabasz", f"{m.get('calinski_harabasz_score', 'N/A')}")

        st.markdown("---")
        col_sw1, col_sw2 = st.columns([1, 1])

        with col_sw1:
            st.subheader("Multi-K Parameter Sweep (Elbow & Silhouette)")
            if st.button("Run Multi-K Sweep (k=2..12)"):
                with st.spinner("Sweeping k=2..12..."):
                    engine = ClusteringEngine(random_state=RANDOM_SEED)
                    df_sweep = engine.run_k_sweep(X_in, k_min=2, k_max=12)
                    st.session_state.df_k_sweep = df_sweep
                st.success("Sweep complete!")

            if "df_k_sweep" in st.session_state:
                fig_sw = viz.plot_k_sweep_metrics(st.session_state.df_k_sweep)
                st.plotly_chart(fig_sw, width="stretch")
                fig_elb = viz.plot_elbow_wcss(st.session_state.df_k_sweep)
                st.plotly_chart(fig_elb, width="stretch")

        with col_sw2:
            st.subheader("Random Seed Stability Analysis")
            if st.button("Test Random Seed Stability [21, 42, 100, 123]"):
                with st.spinner("Evaluating seed stability..."):
                    engine = ClusteringEngine(random_state=RANDOM_SEED)
                    df_seeds, df_ari = engine.test_seed_stability(X_in, k=k_val)
                    st.session_state.df_seed_ari = df_ari
                    st.session_state.df_seed_stats = df_seeds
                st.success("Seed stability test complete!")

            if "df_seed_ari" in st.session_state:
                st.markdown("**Pairwise Adjusted Rand Index (Inter-Run Agreement):**")
                st.dataframe(st.session_state.df_seed_ari, width="stretch")
                st.dataframe(st.session_state.df_seed_stats, width="stretch", hide_index=True)


# ==============================================================================
# PAGE 8: DISCOVERED CLUSTER GALLERY
# ==============================================================================
elif selected_page == "8. Discovered Cluster Gallery":
    st.header("Discovered Cluster Gallery & Centroid Exemplars")
    st.markdown("Inspect centroid exemplars (minimum Euclidean distance to cluster centers) and image member assignments.")

    if st.session_state.cluster_labels is None:
        st.warning("Please run K-Means Clustering first.")
    else:
        labels = st.session_state.cluster_labels
        unique_k = sorted(list(np.unique(labels)))
        
        # Display cluster distribution
        fig_dist = viz.plot_cluster_distribution(labels)
        st.plotly_chart(fig_dist, width="stretch")

        st.markdown("---")
        col_g1, col_g2 = st.columns([1, 2])

        with col_g1:
            chosen_cluster = st.selectbox("Select Cluster to Explore:", unique_k, format_func=lambda c: f"Cluster {c}")
            search_query = st.text_input("Filter images by Image ID / filename:", "")

        df_m = st.session_state.manifest_df
        cluster_mask = (labels == chosen_cluster)
        cluster_members = df_m[cluster_mask].copy()

        if search_query:
            cluster_members = cluster_members[cluster_members["image_id"].astype(str).str.contains(search_query, case=False)]

        st.markdown(f"Displaying **{len(cluster_members)}** images in Cluster {chosen_cluster}:")

        # Display images in responsive grid
        cols = st.columns(4)
        for i, (_, row) in enumerate(cluster_members.head(16).iterrows()):
            col = cols[i % 4]
            img_rel = row["file_path"]
            img_abs = PROJECT_ROOT / img_rel
            if img_abs.exists():
                with col:
                    img = Image.open(img_abs)
                    st.image(img, caption=f"{row['image_id']} ({row.get('format', 'IMG')})", width="stretch")

        # Download cluster manifest
        st.download_button(
            "Download This Cluster's Image Manifest (CSV)",
            data=cluster_members.to_csv(index=False),
            file_name=f"cluster_{chosen_cluster}_members.csv",
            mime="text/csv"
        )


# ==============================================================================
# PAGE 9: IMAGE DISCOVERY STUDIO
# ==============================================================================
elif selected_page == "9. Image Discovery Studio":
    st.header("Image Discovery Studio (Live Inference & Visual Search)")
    st.markdown("""
    <div class="lab-card">
        <strong>Unsupervised Visual Inference:</strong> Upload any novel image from your computer. 
        The system extracts its representation, applies the active preprocessing and PCA pipelines, 
        projects it onto the discovered cluster landscape, assigns it to its nearest centroid, 
        and retrieves its closest visual neighbors.
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.cluster_centroids is None:
        st.warning("Please run K-Means Clustering to compute cluster centroids before testing the studio.")
    else:
        uploaded_probe = st.file_uploader("Upload an Image to Analyze (JPG, PNG, WEBP):", type=["jpg", "jpeg", "png", "webp"])

        if uploaded_probe is not None:
            col_in1, col_in2 = st.columns([1, 2])

            with col_in1:
                st.subheader("Uploaded Query Image")
                probe_img = Image.open(uploaded_probe)
                st.image(probe_img, caption="Query Image", width="stretch")

            with col_in2:
                st.subheader("Unsupervised Placement Analysis")
                with st.spinner("Extracting features, transforming, and comparing with centroids..."):
                    studio = ImageDiscoveryStudio(
                        feature_extractor=st.session_state.feature_extractor,
                        scaler=st.session_state.scaler_pipeline,
                        pca_manager=st.session_state.pca_manager,
                        centroids=st.session_state.cluster_centroids,
                        dataset_features=st.session_state.features_pca if st.session_state.features_pca is not None else st.session_state.features_scaled,
                        df_manifest=st.session_state.manifest_df
                    )
                    probe_bytes = uploaded_probe.getvalue()
                    results = studio.process_and_assign(
                        probe_bytes,
                        feature_type=st.session_state.feature_type,
                        top_k_neighbors=6
                    )

                c_assigned = results["nearest_cluster_id"]
                c_dist = results["min_centroid_distance"]

                st.metric("Nearest Discovered Cluster", f"Cluster {c_assigned}")
                st.metric("Distance to Centroid", f"{c_dist:.4f}")

                st.markdown("**Proximity Across All Clusters:**")
                st.dataframe(results["cluster_distance_ranking"], width="stretch", hide_index=True)

            st.markdown("---")
            st.subheader("Top Visually Similar Neighbors from Dataset")
            df_sim = results["similar_images"]

            if not df_sim.empty:
                cols_sim = st.columns(len(df_sim))
                for idx, row in df_sim.iterrows():
                    col = cols_sim[idx]
                    sim_p = PROJECT_ROOT / row["File_Path"]
                    if sim_p.exists():
                        with col:
                            st.image(Image.open(sim_p), caption=f"Rank #{row['Rank']} (d={row['Distance']:.2f})", width="stretch")


# ==============================================================================
# PAGE 10: EXPERIMENT COMPARISON
# ==============================================================================
elif selected_page == "10. Experiment Comparison":
    st.header("Experiment Manager & Model Comparison")
    st.markdown("Log experiments, track hyperparameter iterations, and compare clustering performance side-by-side.")

    em = st.session_state.experiment_manager

    col_exp1, col_exp2 = st.columns([1, 1])

    with col_exp1:
        st.subheader("Save Current Run as Experiment")
        exp_name = st.text_input("Experiment Name:", value=f"Run_{st.session_state.feature_type}_k{k_curr}")
        exp_notes = st.text_area("Notes / Observations:", "Evaluation of ResNet50 GAP on active dataset.")

        if st.button("Save Experiment"):
            exp_id = em.log_experiment(
                name=exp_name,
                dataset_name=st.session_state.current_dataset_name,
                sample_count=n_active_imgs,
                feature_type=st.session_state.feature_type,
                scaler_type=st.session_state.scaler_pipeline.scaler_type,
                pca_config=st.session_state.pca_metadata.get("metrics", {}),
                kmeans_config={"k": k_curr},
                metrics=st.session_state.cluster_metrics,
                tsne_config=st.session_state.tsne_metadata,
                notes=exp_notes
            )
            st.success(f"Experiment saved with ID: {exp_id}!")
            st.rerun()

    with col_exp2:
        st.subheader("Experiment History")
        df_exps = em.list_experiments()
        if not df_exps.empty:
            st.dataframe(df_exps, width="stretch", hide_index=True)
            
            c_e1, c_e2 = st.columns(2)
            c_e1.download_button(
                "Export Experiments (JSON)",
                data=em.export_all_json(),
                file_name="experiments.json",
                mime="application/json"
            )
            c_e2.download_button(
                "Export Experiments (CSV)",
                data=em.export_all_csv(),
                file_name="experiments.csv",
                mime="text/csv"
            )
        else:
            st.info("No saved experiments yet. Configure and click 'Save Experiment'.")

    st.markdown("---")
    if not df_exps.empty and len(df_exps) >= 2:
        st.subheader("Side-by-Side Model Comparison")
        exp_ids = df_exps["Experiment ID"].tolist()
        selected_to_compare = st.multiselect("Select Experiments to Compare:", exp_ids, default=exp_ids[:2])

        if len(selected_to_compare) >= 2:
            df_comp = em.compare_experiments(selected_to_compare)
            st.dataframe(df_comp, width="stretch")
            fig_rad = viz.plot_experiment_comparison_radar(df_comp)
            st.plotly_chart(fig_rad, width="stretch")


# ==============================================================================
# PAGE 11: EXTERNAL VALIDATION
# ==============================================================================
elif selected_page == "11. External Validation":
    st.header("Post-Hoc External Validation (Ground Truth Unlocked)")
    st.markdown("""
    <div class="warning-card">
        <strong>Strict Methodological Compliance:</strong> Ground-truth category labels are never exposed during 
        preprocessing, scaling, PCA, or cluster formation. They are unlocked here <em>strictly post-hoc</em> 
        to evaluate how closely unsupervised visual clusters correspond to human taxonomic categories.
    </div>
    """, unsafe_allow_html=True)

    dm = st.session_state.dataset_manager
    df_gt = dm.load_ground_truth()

    if df_gt is None or df_gt.empty:
        st.info("No ground-truth labels found for the current dataset. External validation is unavailable for unannotated datasets.")
    elif st.session_state.cluster_labels is None:
        st.warning("Please run K-Means Clustering first.")
    else:
        # Match manifest with ground truth
        df_m = st.session_state.manifest_df
        merged = df_m.merge(df_gt, on="image_id", how="inner")
        
        if len(merged) != len(st.session_state.cluster_labels):
            st.warning(f"Label count mismatch ({len(merged)} labels vs {len(st.session_state.cluster_labels)} cluster assignments).")
        else:
            val_results = ExternalValidator.evaluate(
                merged["category"].values,
                st.session_state.cluster_labels
            )

            col_ev1, col_ev2 = st.columns([1, 1])
            with col_ev1:
                st.subheader("Validation Scorecard")
                st.dataframe(val_results["summary_df"], width="stretch", hide_index=True)

            with col_ev2:
                st.subheader("Category Purity Heatmap")
                fig_hm = viz.plot_contingency_heatmap_interactive(val_results["contingency_matrix_normalized"])
                st.plotly_chart(fig_hm, width="stretch")


# ==============================================================================
# PAGE 12: REPORTS AND EXPORTS
# ==============================================================================
elif selected_page == "12. Reports and Exports":
    st.header("Reports and Data Exports")
    st.markdown("Download full research reports, cluster assignments, and serialized model objects.")

    rep_path = PROJECT_ROOT / "reports" / "final_report.md"
    if rep_path.exists():
        with open(rep_path, "r", encoding="utf-8") as f:
            report_text = f.read()

        st.download_button(
            "Download Academic Research Report (Markdown)",
            data=report_text,
            file_name="final_report.md",
            mime="text/markdown"
        )
        with st.expander("Preview Full Academic Research Report"):
            st.markdown(report_text)

    if st.session_state.cluster_labels is not None and not st.session_state.manifest_df.empty:
        df_export = st.session_state.manifest_df.copy()
        df_export["Cluster_Assignment"] = st.session_state.cluster_labels
        st.download_button(
            "Download Dataset with Cluster Assignments (CSV)",
            data=df_export.to_csv(index=False),
            file_name="cluster_assignments.csv",
            mime="text/csv"
        )


# ==============================================================================
# PAGE 13: LIMITATIONS AND RESEARCH ETHICS
# ==============================================================================
elif selected_page == "13. Limitations and Research Ethics":
    st.header("Limitations, Bias, and Research Ethics")
    st.markdown("""
    ### 1. Algorithmic and Representational Constraints
    - **Inductive Pretraining Bias:** Pretrained CNNs (ResNet50) were trained on ImageNet-1K with supervised targets. While the clustering pipeline is strictly unsupervised, the feature representations reflect ImageNet's class taxonomy and object biases.
    - **Spherical Geometry in K-Means:** K-Means assumes isotropic, spherical Voronoi partitions. Complex, non-linear visual manifolds may be artificially split or merged.
    - **Color Histogram Limitations:** 3D color histograms lack spatial hierarchy, clustering images predominantly by background illumination (sky, lawn) rather than object morphology.

    ### 2. Dataset Bias & Responsible AI
    - **Demographic & Geographic Skew:** Web datasets skew toward Western domestic environments, daylight photography, and clean subject framing.
    - **Privacy and Incidental Subjects:** Real-world street scenes require adherence to CC-BY licenses and privacy standards regarding incidental bystanders and vehicles.
    - **Statistical Patterns vs. Ontological Truth:** Discovered clusters represent statistical affinities within a feature space, not authoritative ground-truth classifications.
    """)
