"""
Visualization Module for Object Discovery Lab
Builds high-performance interactive Plotly charts for the Streamlit dashboard
and static publication-grade Matplotlib plots.
"""

import sys
from pathlib import Path
from typing import Optional, List, Dict, Any
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DARK_LAYOUT = dict(
    template="plotly_dark",
    plot_bgcolor="#101c3d",
    paper_bgcolor="#101c3d",
    font=dict(color="#e0e6ed", family="Inter, sans-serif"),
    margin=dict(l=40, r=40, t=50, b=40)
)


def plot_scree_interactive(df_scree: pd.DataFrame, target_var: Optional[float] = None) -> go.Figure:
    """Builds interactive scree curve showing individual and cumulative explained variance."""
    fig = go.Figure()

    fig.add_trace(go.Bar(
        x=df_scree["Component"][:50],
        y=df_scree["Individual_Variance_Ratio"][:50] * 100,
        name="Individual Variance (%)",
        marker_color="#00b4d8",
        opacity=0.7
    ))

    fig.add_trace(go.Scatter(
        x=df_scree["Component"],
        y=df_scree["Cumulative_Variance_Ratio"] * 100,
        name="Cumulative Variance (%)",
        mode="lines+markers",
        line=dict(color="#48cae4", width=3),
        yaxis="y2"
    ))

    if target_var is not None:
        fig.add_hline(
            y=target_var * 100,
            line_dash="dash",
            line_color="#ffb703",
            annotation_text=f"Target: {int(target_var*100)}%",
            annotation_position="bottom right",
            yref="y2"
        )

    fig.update_layout(
        title="PCA Scree Curve: Individual & Cumulative Explained Variance",
        xaxis=dict(title="Principal Component"),
        yaxis=dict(title="Individual Variance (%)", side="left"),
        yaxis2=dict(title="Cumulative Variance (%)", side="right", overlaying="y", range=[0, 105]),
        legend=dict(x=0.02, y=0.98),
        **DARK_LAYOUT
    )
    return fig


def plot_embedding_2d_interactive(
    coords_2d: np.ndarray,
    labels: Optional[np.ndarray] = None,
    hover_names: Optional[List[str]] = None,
    title: str = "2D Feature Embedding"
) -> go.Figure:
    """Builds interactive 2D scatter plot colored by cluster or label."""
    df_plot = pd.DataFrame({
        "Dim_1": coords_2d[:, 0],
        "Dim_2": coords_2d[:, 1],
        "Group": [f"Cluster {c}" if isinstance(c, (int, np.integer)) else str(c) for c in (labels if labels is not None else np.zeros(len(coords_2d), dtype=int))],
        "Image_ID": hover_names if hover_names is not None else [f"Img_{i}" for i in range(len(coords_2d))]
    })

    fig = px.scatter(
        df_plot,
        x="Dim_1",
        y="Dim_2",
        color="Group",
        hover_data=["Image_ID"],
        opacity=0.75,
        title=title,
        color_discrete_sequence=px.colors.qualitative.Dark24
    )
    fig.update_traces(marker=dict(size=7))
    fig.update_layout(**DARK_LAYOUT)
    return fig


def plot_k_sweep_metrics(df_sweep: pd.DataFrame) -> go.Figure:
    """Builds interactive multi-line chart for K-Means sweep across k=2..12."""
    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=df_sweep["k"],
        y=df_sweep["Silhouette_Score"],
        mode="lines+markers",
        name="Silhouette Score (Higher is Better)",
        line=dict(color="#48cae4", width=3)
    ))

    fig.add_trace(go.Scatter(
        x=df_sweep["k"],
        y=df_sweep["Davies_Bouldin_Index"],
        mode="lines+markers",
        name="Davies-Bouldin Index (Lower is Better)",
        line=dict(color="#ffb703", width=3)
    ))

    fig.update_layout(
        title="Clustering Metric Comparison Across Cluster Counts (k=2..12)",
        xaxis=dict(title="Number of Clusters (k)", tickmode="linear", dtick=1),
        yaxis=dict(title="Metric Score"),
        legend=dict(x=0.02, y=0.98),
        **DARK_LAYOUT
    )
    return fig


def plot_elbow_wcss(df_sweep: pd.DataFrame) -> go.Figure:
    """Builds interactive Elbow (WCSS) plot."""
    fig = px.line(
        df_sweep,
        x="k",
        y="WCSS_Inertia",
        markers=True,
        title="Elbow Method: Within-Cluster Sum of Squares (WCSS)",
        labels={"WCSS_Inertia": "Inertia / WCSS", "k": "Cluster Count (k)"}
    )
    fig.update_traces(line=dict(color="#00b4d8", width=3), marker=dict(size=8, color="#ffb703"))
    fig.update_layout(**DARK_LAYOUT)
    return fig


def plot_cluster_distribution(labels: np.ndarray) -> go.Figure:
    """Builds interactive bar chart of cluster sizes."""
    unique, counts = np.unique(labels, return_counts=True)
    df_sizes = pd.DataFrame({
        "Cluster": [f"Cluster {c}" for c in unique],
        "Image_Count": counts,
        "Percentage": np.round((counts / len(labels)) * 100, 2)
    })

    fig = px.bar(
        df_sizes,
        x="Cluster",
        y="Image_Count",
        text="Percentage",
        color="Cluster",
        color_discrete_sequence=px.colors.qualitative.Prism,
        title="Cluster Size & Member Distribution"
    )
    fig.update_traces(texttemplate="%{text}%", textposition="outside")
    fig.update_layout(showlegend=False, **DARK_LAYOUT)
    return fig


def plot_contingency_heatmap_interactive(df_cont_norm: pd.DataFrame) -> go.Figure:
    """Builds interactive heatmap of cluster vs ground truth category."""
    fig = px.imshow(
        df_cont_norm,
        text_auto=True,
        color_continuous_scale="YlGnBu",
        title="Post-Hoc Contingency Analysis: Ground Truth vs Discovered Clusters (%)",
        labels=dict(x="Discovered Cluster", y="Ground Truth Category", color="Percentage (%)")
    )
    fig.update_layout(**DARK_LAYOUT)
    return fig


def plot_experiment_comparison_radar(df_compare: pd.DataFrame) -> go.Figure:
    """Builds comparative radar or bar chart across experiments."""
    metrics_to_plot = ["Silhouette Score", "Davies-Bouldin Index", "Adjusted Rand Index (ARI)", "Normalized Mutual Info (NMI)"]
    
    fig = go.Figure()
    for col in df_compare.columns:
        vals = []
        for m in metrics_to_plot:
            v = df_compare.loc[m, col] if m in df_compare.index else 0
            try:
                vals.append(float(v))
            except Exception:
                vals.append(0.0)

        fig.add_trace(go.Bar(
            name=str(col),
            x=metrics_to_plot,
            y=vals
        ))

    fig.update_layout(
        barmode="group",
        title="Direct Metric Comparison Across Experiments",
        yaxis=dict(title="Score"),
        **DARK_LAYOUT
    )
    return fig
