#!/usr/bin/env python3
"""
Streamlit Visualization App for Prediction Results
Interactive dashboard for comparing RECONSTRUCTION MODELS by a selectable prediction error metric.
Prediction model serves as a filter, not the main comparison axis.
Part of uniTS-MissRecoPred framework.
"""

from __future__ import annotations

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from typing import NamedTuple

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from prediction_metrics import (
    PredictionMetricSpec,
    get_metric_spec,
    list_metric_specs_ordered,
)
from utils.config_loader import load_config


def _heatmap_texttemplate(value_is_percent: bool) -> str:
    return "%{text}%" if value_is_percent else "%{text}"


class PredictionMetricView(NamedTuple):
    """Selected prediction error metric (CSV column = key)."""

    key: str
    label: str
    lower_is_better: bool
    value_is_percent: bool

    @staticmethod
    def from_spec(spec: PredictionMetricSpec) -> PredictionMetricView:
        return PredictionMetricView(
            spec.key, spec.label, spec.lower_is_better, spec.value_is_percent
        )


def _metric_cmap(lower_is_better: bool) -> str:
    return "RdYlGn_r" if lower_is_better else "RdYlGn"


def _heatmap_text_decimals(value_is_percent: bool) -> int:
    return 2 if value_is_percent else 4


def _style_row_best_worst(row, lower_is_better: bool):
    min_val = row.min()
    max_val = row.max()
    tol = 1e-6

    def is_best(v):
        if pd.isna(v):
            return False
        if lower_is_better:
            return abs(v - min_val) <= tol or min_val == max_val
        return abs(v - max_val) <= tol or min_val == max_val

    def is_worst(v):
        if pd.isna(v) or min_val == max_val:
            return False
        if lower_is_better:
            return abs(v - max_val) <= tol
        return abs(v - min_val) <= tol

    return [
        "background-color: #90EE90" if is_best(v)
        else "background-color: #FFB6C1" if is_worst(v)
        else ""
        for v in row
    ]


def load_results(file_path: str) -> pd.DataFrame:
    """Load results from CSV file"""
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        st.error(f"Error loading results: {e}")
        return pd.DataFrame()


def get_available_results() -> list:
    """Get list of available prediction result files"""
    results_dir = Path("prediction_experiment_results")
    if not results_dir.exists():
        return []
    
    # Look for prediction_results_*.csv files
    return sorted(results_dir.glob("prediction_results_*.csv"), reverse=True)


def get_available_training_metrics() -> list:
    """Get list of available training metrics files"""
    results_dir = Path("prediction_experiment_results")
    if not results_dir.exists():
        return []
    
    return sorted(results_dir.glob("training_metrics_*.csv"), reverse=True)


def load_training_metrics() -> pd.DataFrame:
    """Load the most recent training metrics file"""
    files = get_available_training_metrics()
    if not files:
        return pd.DataFrame()
    
    try:
        return pd.read_csv(files[0])
    except Exception:
        return pd.DataFrame()


def plot_mape_by_reconstruction_model_main(
    df: pd.DataFrame,
    m: PredictionMetricView,
    technique: str = None,
    rate: int = None,
):
    """Main plot: selected metric by reconstruction model (primary axis)."""
    df_filtered = df[df["source_type"] == "reconstructed"].copy()

    if technique:
        df_filtered = df_filtered[df_filtered["technique"] == technique]
    if rate:
        df_filtered = df_filtered[df_filtered["rate_percent"] == rate]

    if df_filtered.empty:
        st.warning("No reconstructed data available for selected filters")
        return

    df_stats = (
        df_filtered.groupby("reconstruction_model")[m.key]
        .agg(["mean", "std", "min", "max", "count"])
        .reset_index()
    )
    df_stats = df_stats.sort_values("mean", ascending=m.lower_is_better)

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=df_stats["reconstruction_model"],
            y=df_stats["mean"],
            error_y=dict(type="data", array=df_stats["std"]),
            marker_color="teal",
            name=f"Mean {m.label}",
        )
    )
    fig.update_layout(
        title=f"Prediction {m.label} by Reconstruction Model",
        xaxis_title="Reconstruction Model",
        yaxis_title=m.label,
        xaxis_tickangle=-45,
        height=500,
    )
    st.plotly_chart(fig, width="stretch")

    st.subheader("Statistics")
    mean_col = "Mean"
    df_stats_extended = df_filtered.groupby("reconstruction_model").agg(
        {
            m.key: ["mean", "std", "median", "min", "max", "count"],
            "dataset_name": "nunique",
            "prediction_model": "nunique",
        }
    ).reset_index()
    df_stats_extended.columns = [
        "Reconstruction Model",
        mean_col,
        "Std",
        "Median",
        "Min",
        "Max",
        "N Predictions",
        "N Datasets",
        "N Pred Models",
    ]
    df_stats_extended["CV (%)"] = (
        df_stats_extended["Std"] / df_stats_extended[mean_col].replace(0, np.nan) * 100
    ).round(2)
    df_stats_extended = df_stats_extended.sort_values(
        mean_col, ascending=m.lower_is_better
    )
    df_stats_extended["Rank"] = range(1, len(df_stats_extended) + 1)
    if m.lower_is_better:
        best_v = df_stats_extended[mean_col].min()
        df_stats_extended["Δ Best"] = (df_stats_extended[mean_col] - best_v).round(4)
    else:
        best_v = df_stats_extended[mean_col].max()
        df_stats_extended["Δ Best"] = (best_v - df_stats_extended[mean_col]).round(4)

    df_stats_extended = df_stats_extended[
        [
            "Rank",
            "Reconstruction Model",
            mean_col,
            "Median",
            "Std",
            "CV (%)",
            "Min",
            "Max",
            "Δ Best",
            "N Predictions",
            "N Datasets",
            "N Pred Models",
        ]
    ]

    mfmt = "{:.2f}%" if m.value_is_percent else "{:.4f}"
    st.dataframe(
        df_stats_extended.style.format(
            {
                mean_col: mfmt,
                "Median": mfmt,
                "Std": mfmt,
                "CV (%)": "{:.1f}%",
                "Min": mfmt,
                "Max": mfmt,
                "Δ Best": mfmt,
                "N Predictions": "{:.0f}",
                "N Datasets": "{:.0f}",
                "N Pred Models": "{:.0f}",
                "Rank": "{:.0f}",
            }
        ).background_gradient(subset=[mean_col], cmap=_metric_cmap(m.lower_is_better)),
        width="stretch",
    )

    with st.expander("📖 Column explanations"):
        st.markdown(
            f"""
        - **Rank**: Sorted by mean {m.label} (1 = best for current direction)
        - **Mean**: Average {m.label} across predictions
        - **Median / Std / Min / Max**: Distribution of {m.label}
        - **CV (%)**: Std/Mean × 100 (watch for near-zero mean)
        - **Δ Best**: Gap vs best model (lower is better for this column when error metrics)
        """
        )


def plot_recon_by_technique(df: pd.DataFrame, m: PredictionMetricView):
    """Plot reconstruction model metric grouped by missingness technique."""
    df_filtered = df[df["source_type"] == "reconstructed"].copy()

    if df_filtered.empty:
        st.warning("No reconstructed data available")
        return

    df_stats = (
        df_filtered.groupby(["reconstruction_model", "technique"])[m.key].mean().reset_index()
    )

    fig = px.bar(
        df_stats,
        x="reconstruction_model",
        y=m.key,
        color="technique",
        barmode="group",
        title=f"Reconstruction Model {m.label} by Missingness Technique",
        labels={
            m.key: f"Mean {m.label}",
            "reconstruction_model": "Reconstruction Model",
            "technique": "Technique",
        },
    )

    fig.update_layout(xaxis_tickangle=-45, height=500)

    st.plotly_chart(fig, width="stretch")


def plot_recon_by_rate(df: pd.DataFrame, m: PredictionMetricView, technique: str = None):
    """Plot reconstruction model metric by missing rate."""
    df_filtered = df[df["source_type"] == "reconstructed"].copy()

    if technique:
        df_filtered = df_filtered[df_filtered["technique"] == technique]

    if df_filtered.empty:
        st.warning("No reconstructed data available")
        return

    df_stats = (
        df_filtered.groupby(["reconstruction_model", "rate_percent"])[m.key].mean().reset_index()
    )

    fig = px.line(
        df_stats,
        x="rate_percent",
        y=m.key,
        color="reconstruction_model",
        markers=True,
        title=f"Reconstruction Model {m.label} by Missing Rate",
        labels={
            m.key: f"Mean {m.label}",
            "rate_percent": "Missing Rate (%)",
            "reconstruction_model": "Reconstruction Model",
        },
    )

    fig.update_layout(height=500)
    st.plotly_chart(fig, width="stretch")


def plot_mape_by_technique(
    df: pd.DataFrame, m: PredictionMetricView, model: str = None, rate: int = None
):
    """Plot metric by missingness technique (reconstructed data only)."""
    df_filtered = df[df["source_type"] == "reconstructed"]

    if model:
        df_filtered = df_filtered[df_filtered["prediction_model"] == model]
    if rate:
        df_filtered = df_filtered[df_filtered["rate_percent"] == rate]

    if df_filtered.empty:
        st.warning(
            "No data available for selected filters (only reconstructed data has techniques)"
        )
        return

    df_stats = df_filtered.groupby("technique")[m.key].agg(["mean", "std", "min", "max"]).reset_index()
    df_stats = df_stats.sort_values("mean", ascending=m.lower_is_better)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df_stats["technique"],
            y=df_stats["mean"],
            error_y=dict(type="data", array=df_stats["std"]),
            marker_color="coral",
            name=f"Mean {m.label}",
        )
    )

    fig.update_layout(
        title=f"{m.label} by Missingness Technique (Reconstructed Data)",
        xaxis_title="Missingness Technique",
        yaxis_title=m.label,
        height=500,
    )

    st.plotly_chart(fig, width="stretch")

    st.subheader("Statistics")
    mfmt = "{:.2f}%" if m.value_is_percent else "{:.4f}"
    st.dataframe(
        df_stats.style.format(
            {"mean": mfmt, "std": mfmt, "min": mfmt, "max": mfmt}
        ),
        width="stretch",
    )


def plot_mape_by_rate(
    df: pd.DataFrame, m: PredictionMetricView, model: str = None, technique: str = None
):
    """Plot metric by missing rate (reconstructed data only)."""
    df_filtered = df[df["source_type"] == "reconstructed"]

    if model:
        df_filtered = df_filtered[df_filtered["prediction_model"] == model]
    if technique:
        df_filtered = df_filtered[df_filtered["technique"] == technique]

    if df_filtered.empty:
        st.warning("No data available for selected filters")
        return

    df_stats = (
        df_filtered.groupby("rate_percent")[m.key].agg(["mean", "std", "min", "max"]).reset_index()
    )
    df_stats = df_stats.sort_values("rate_percent")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df_stats["rate_percent"],
            y=df_stats["mean"],
            mode="lines+markers",
            error_y=dict(type="data", array=df_stats["std"]),
            marker=dict(size=10, color="mediumpurple"),
            line=dict(width=2),
            name=f"Mean {m.label}",
        )
    )

    fig.update_layout(
        title=f"{m.label} by Missing Rate (Reconstructed Data)",
        xaxis_title="Missing Rate (%)",
        yaxis_title=m.label,
        height=500,
    )

    st.plotly_chart(fig, width="stretch")

    st.subheader("Statistics")
    df_stats_display = df_stats.copy()
    df_stats_display["rate_percent"] = df_stats_display["rate_percent"].astype(str) + "%"
    mfmt = "{:.2f}%" if m.value_is_percent else "{:.4f}"
    st.dataframe(
        df_stats_display.style.format(
            {"mean": mfmt, "std": mfmt, "min": mfmt, "max": mfmt}
        ),
        width="stretch",
    )


def plot_mape_by_reconstruction_model(
    df: pd.DataFrame, m: PredictionMetricView, pred_model: str = None, technique: str = None
):
    """Plot metric by reconstruction model (reconstructed data only)."""
    df_filtered = df[df["source_type"] == "reconstructed"]

    if pred_model:
        df_filtered = df_filtered[df_filtered["prediction_model"] == pred_model]
    if technique:
        df_filtered = df_filtered[df_filtered["technique"] == technique]

    if df_filtered.empty:
        st.warning("No data available for selected filters")
        return

    df_stats = (
        df_filtered.groupby("reconstruction_model")[m.key]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    df_stats = df_stats.sort_values("mean", ascending=m.lower_is_better)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df_stats["reconstruction_model"],
            y=df_stats["mean"],
            error_y=dict(type="data", array=df_stats["std"]),
            marker_color="teal",
            name=f"Mean {m.label}",
        )
    )

    fig.update_layout(
        title=f"Prediction {m.label} by Reconstruction Model Used",
        xaxis_title="Reconstruction Model",
        yaxis_title=m.label,
        xaxis_tickangle=-45,
        height=500,
    )

    st.plotly_chart(fig, width="stretch")

    st.subheader("Statistics")
    mfmt = "{:.2f}%" if m.value_is_percent else "{:.4f}"
    st.dataframe(
        df_stats.style.format({"mean": mfmt, "std": mfmt, "min": mfmt, "max": mfmt}),
        width="stretch",
    )


def plot_heatmap_recon_vs_technique(df: pd.DataFrame, m: PredictionMetricView):
    """Heatmap: reconstruction model vs technique."""
    df_filtered = df[df["source_type"] == "reconstructed"].copy()

    if df_filtered.empty:
        st.warning("No reconstructed data available")
        return

    pivot_data = df_filtered.pivot_table(
        values=m.key, index="reconstruction_model", columns="technique", aggfunc="mean"
    )

    pivot_data["_mean"] = pivot_data.mean(axis=1)
    pivot_data = pivot_data.sort_values("_mean", ascending=m.lower_is_better)
    pivot_data = pivot_data.drop("_mean", axis=1)

    dec = _heatmap_text_decimals(m.value_is_percent)
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale=_metric_cmap(m.lower_is_better),
            text=np.round(pivot_data.values, dec),
            texttemplate=_heatmap_texttemplate(m.value_is_percent),
            textfont={"size": 10},
            colorbar=dict(title=m.label),
        )
    )

    fig.update_layout(
        title="Heatmap: Reconstruction Model vs Missingness Technique",
        xaxis_title="Missingness Technique",
        yaxis_title="Reconstruction Model",
        height=max(500, len(pivot_data.index) * 35),
    )

    st.plotly_chart(fig, width="stretch")


def plot_heatmap_recon_vs_rate(df: pd.DataFrame, m: PredictionMetricView, technique: str = None):
    """Heatmap: reconstruction model vs missing rate."""
    df_filtered = df[df["source_type"] == "reconstructed"].copy()

    if technique:
        df_filtered = df_filtered[df_filtered["technique"] == technique]

    if df_filtered.empty:
        st.warning("No reconstructed data available")
        return

    pivot_data = df_filtered.pivot_table(
        values=m.key, index="reconstruction_model", columns="rate_percent", aggfunc="mean"
    )

    pivot_data["_mean"] = pivot_data.mean(axis=1)
    pivot_data = pivot_data.sort_values("_mean", ascending=m.lower_is_better)
    pivot_data = pivot_data.drop("_mean", axis=1)

    dec = _heatmap_text_decimals(m.value_is_percent)
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_data.values,
            x=[f"{int(r)}%" for r in pivot_data.columns],
            y=pivot_data.index,
            colorscale=_metric_cmap(m.lower_is_better),
            text=np.round(pivot_data.values, dec),
            texttemplate=_heatmap_texttemplate(m.value_is_percent),
            textfont={"size": 10},
            colorbar=dict(title=m.label),
        )
    )

    title = "Heatmap: Reconstruction Model vs Missing Rate"
    if technique:
        title += f" ({technique})"

    fig.update_layout(
        title=title,
        xaxis_title="Missing Rate",
        yaxis_title="Reconstruction Model",
        height=max(500, len(pivot_data.index) * 35),
    )

    st.plotly_chart(fig, width="stretch")


def plot_heatmap_pred_vs_recon(df: pd.DataFrame, m: PredictionMetricView, sort_by_model: str = None):
    """Heatmap: prediction model vs reconstruction model."""
    df_filtered = df[df["source_type"] == "reconstructed"].copy()

    if df_filtered.empty:
        st.warning("No reconstructed data available")
        return

    pivot_data = df_filtered.pivot_table(
        values=m.key, index="prediction_model", columns="reconstruction_model", aggfunc="mean"
    )

    if sort_by_model and sort_by_model in pivot_data.columns:
        pivot_data = pivot_data.sort_values(
            by=sort_by_model, ascending=m.lower_is_better
        )
        sort_info = f" (sorted by {sort_by_model})"
    else:
        pivot_data = pivot_data.sort_index()
        sort_info = " (alphabetical)"

    dec = _heatmap_text_decimals(m.value_is_percent)
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale=_metric_cmap(m.lower_is_better),
            text=np.round(pivot_data.values, dec),
            texttemplate=_heatmap_texttemplate(m.value_is_percent),
            textfont={"size": 10},
            colorbar=dict(title=m.label),
        )
    )

    fig.update_layout(
        title=f"Heatmap: {m.label} by Prediction Model vs Reconstruction Model{sort_info}",
        xaxis_title="Reconstruction Model",
        yaxis_title="Prediction Model",
        height=max(500, len(pivot_data.index) * 35),
    )

    st.plotly_chart(fig, width="stretch")


def plot_heatmap_pred_vs_technique(
    df: pd.DataFrame, m: PredictionMetricView, sort_by_technique: str = None
):
    """Heatmap: prediction model vs missingness technique."""
    df_filtered = df[df["source_type"] == "reconstructed"].copy()

    if df_filtered.empty:
        st.warning("No reconstructed data available")
        return

    pivot_data = df_filtered.pivot_table(
        values=m.key, index="prediction_model", columns="technique", aggfunc="mean"
    )

    if sort_by_technique and sort_by_technique in pivot_data.columns:
        pivot_data = pivot_data.sort_values(
            by=sort_by_technique, ascending=m.lower_is_better
        )
        sort_info = f" (sorted by {sort_by_technique})"
    else:
        pivot_data = pivot_data.sort_index()
        sort_info = " (alphabetical)"

    dec = _heatmap_text_decimals(m.value_is_percent)
    fig = go.Figure(
        data=go.Heatmap(
            z=pivot_data.values,
            x=pivot_data.columns,
            y=pivot_data.index,
            colorscale=_metric_cmap(m.lower_is_better),
            text=np.round(pivot_data.values, dec),
            texttemplate=_heatmap_texttemplate(m.value_is_percent),
            textfont={"size": 10},
            colorbar=dict(title=m.label),
        )
    )

    fig.update_layout(
        title=f"Heatmap: {m.label} by Prediction Model vs Technique{sort_info}",
        xaxis_title="Missingness Technique",
        yaxis_title="Prediction Model",
        height=max(500, len(pivot_data.index) * 35),
    )

    st.plotly_chart(fig, width="stretch")


def plot_dataset_comparison(df: pd.DataFrame, m: PredictionMetricView):
    """Compare metric across datasets."""
    df_stats = df.groupby("dataset_name")[m.key].agg(["mean", "std"]).reset_index()
    df_stats = df_stats.sort_values("mean", ascending=m.lower_is_better)

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            x=df_stats["dataset_name"],
            y=df_stats["mean"],
            error_y=dict(type="data", array=df_stats["std"]),
            marker_color="mediumpurple",
            name=f"Mean {m.label}",
        )
    )

    fig.update_layout(
        title=f"{m.label} by Dataset",
        xaxis_title="Dataset",
        yaxis_title=m.label,
        xaxis_tickangle=-45,
        height=500,
    )

    st.plotly_chart(fig, width="stretch")


def plot_best_worst_reconstruction_models(df: pd.DataFrame, m: PredictionMetricView, top_n: int = 10):
    """Best and worst reconstruction models by mean metric."""
    df_filtered = df[df["source_type"] == "reconstructed"].copy()

    if df_filtered.empty:
        st.warning("No reconstructed data available")
        return

    df_stats = df_filtered.groupby("reconstruction_model")[m.key].mean().reset_index()
    df_stats = df_stats.sort_values(m.key, ascending=m.lower_is_better)

    global_min = df_stats[m.key].min()
    global_max = df_stats[m.key].max()
    axis_range = [global_min * 0.95, global_max * 1.05]

    top_n = min(top_n, len(df_stats) // 2) if len(df_stats) > 2 else len(df_stats)

    best_models = df_stats.head(top_n).iloc[::-1]
    worst_models = df_stats.tail(top_n)

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=(
            f"Top {top_n} Best Reconstruction Models",
            f"Top {top_n} Worst Reconstruction Models",
        ),
    )

    fig.add_trace(
        go.Bar(
            x=best_models[m.key],
            y=best_models["reconstruction_model"],
            orientation="h",
            marker_color="green",
            name="Best",
        ),
        row=1,
        col=1,
    )

    fig.add_trace(
        go.Bar(
            x=worst_models[m.key],
            y=worst_models["reconstruction_model"],
            orientation="h",
            marker_color="red",
            name="Worst",
        ),
        row=1,
        col=2,
    )

    fig.update_layout(height=max(500, top_n * 40), showlegend=False)
    fig.update_xaxes(title_text=m.label, range=axis_range, row=1, col=1)
    fig.update_xaxes(title_text=m.label, range=axis_range, row=1, col=2)

    st.plotly_chart(fig, width="stretch")


def plot_iteration_analysis(df: pd.DataFrame, m: PredictionMetricView):
    """Metric variance across prediction iterations."""
    df_stats = (
        df.groupby(["prediction_model", "prediction_iteration"])[m.key].mean().reset_index()
    )

    if df_stats["prediction_iteration"].nunique() <= 1:
        st.info("Only one iteration available - no variance to analyze")
        return

    fig = px.line(
        df_stats,
        x="prediction_iteration",
        y=m.key,
        color="prediction_model",
        markers=True,
        title=f"{m.label} by Prediction Iteration per Model",
        labels={
            m.key: m.label,
            "prediction_iteration": "Iteration",
            "prediction_model": "Model",
        },
    )

    fig.update_layout(height=500)
    st.plotly_chart(fig, width="stretch")

    st.subheader("Iteration Variance by Model")
    variance_stats = (
        df.groupby("prediction_model")[m.key].agg(["mean", "std", "min", "max"]).reset_index()
    )
    variance_stats["cv"] = (
        variance_stats["std"] / variance_stats["mean"].replace(0, np.nan) * 100
    ).round(2)
    variance_stats = variance_stats.sort_values("cv", ascending=False)
    mean_hdr = f"Mean {m.label}"
    variance_stats.columns = ["Model", mean_hdr, "Std", "Min", "Max", "CV (%)"]

    mfmt = "{:.2f}%" if m.value_is_percent else "{:.4f}"
    st.dataframe(
        variance_stats.style.format(
            {
                mean_hdr: mfmt,
                "Std": mfmt,
                "Min": mfmt,
                "Max": mfmt,
                "CV (%)": "{:.2f}%",
            }
        ),
        width="stretch",
    )

    st.caption(
        "CV (Coefficient of Variation) = Std/Mean × 100% — higher values indicate more variability"
    )


def main():
    st.set_page_config(
        page_title="Reconstruction Model Comparison",
        page_icon="🔧",
        layout="wide"
    )
    
    st.title("🔧 Reconstruction Model Comparison by Prediction Error")
    st.caption("Compare how different reconstruction methods affect prediction accuracy")
    st.markdown("---")
    
    # Sidebar for file selection
    st.sidebar.header("Settings")
    
    # Get available result files
    available_files = get_available_results()
    
    if not available_files:
        st.error("No result files found in `prediction_experiment_results/` directory.")
        st.info("Run `python 9_calculate_prediction_error.py` first to generate results.")
        return
    
    # File selection
    file_names = [f.name for f in available_files]
    selected_file_name = st.sidebar.selectbox(
        "Select Results File",
        file_names,
        help="Choose a prediction results file to visualize"
    )
    
    selected_file = next(f for f in available_files if f.name == selected_file_name)
    
    # Load data
    df = load_results(selected_file)
    
    if df.empty:
        st.error("Failed to load data or file is empty")
        return

    config = load_config()
    available_specs = [s for s in list_metric_specs_ordered() if s.key in df.columns]
    if not available_specs:
        st.error(
            "This CSV has no registered prediction metric columns. "
            "Re-run `python 9_calculate_prediction_error.py` with an up-to-date config."
        )
        return

    default_key = config.get_visualization_default_prediction_metric()
    metric_keys = [s.key for s in available_specs]
    if default_key not in metric_keys:
        default_key = metric_keys[0]
    default_idx = metric_keys.index(default_key)

    st.sidebar.subheader("📏 Prediction error metric")
    selected_metric_key = st.sidebar.selectbox(
        "Metric",
        metric_keys,
        index=default_idx,
        format_func=lambda k: get_metric_spec(k).label,
        help="Columns come from script 9; switch metric to refresh all charts and tables.",
    )
    m = PredictionMetricView.from_spec(get_metric_spec(selected_metric_key))
    
    # Display file info
    st.sidebar.success(f"✓ Loaded {len(df)} records")
    st.sidebar.info(f"File: {selected_file_name}")
    
    # Main filters
    st.sidebar.header("🔍 Filters")
    st.sidebar.info("Prediction model = filter, Reconstruction model = main comparison axis")
    
    # Get unique values
    all_datasets = sorted(df['dataset_name'].unique().tolist())
    all_pred_models = sorted(df['prediction_model'].unique().tolist())
    
    # Get techniques and rates (may have NaN for original data)
    all_techniques = sorted([t for t in df['technique'].dropna().unique().tolist()])
    all_rates = sorted([r for r in df['rate_percent'].dropna().unique().tolist()])
    all_recon_models = sorted(
        [rm for rm in df["reconstruction_model"].dropna().unique().tolist()]
    )
    
    # Global filters in sidebar
    st.sidebar.subheader("🌍 Global Filters")
    selected_pred_models = st.sidebar.multiselect(
        "Prediction Models",
        all_pred_models,
        default=all_pred_models,
        help="Select prediction models to include in analysis"
    )
    selected_datasets = st.sidebar.multiselect("Dataset", all_datasets, default=all_datasets)
    selected_techniques = st.sidebar.multiselect("Technique", all_techniques, default=all_techniques)
    selected_rates = st.sidebar.multiselect("Missing Rate (%)", [int(r) for r in all_rates], default=[int(r) for r in all_rates])
    
    # Apply filters to dataframe
    df_filtered = df.copy()
    if selected_datasets:
        df_filtered = df_filtered[df_filtered['dataset_name'].isin(selected_datasets)]
    if selected_pred_models:
        df_filtered = df_filtered[df_filtered['prediction_model'].isin(selected_pred_models)]
    
    # Filter reconstructed data by technique and rate
    df_recon = df_filtered[df_filtered['source_type'] == 'reconstructed'].copy()
    if selected_techniques:
        df_recon = df_recon[df_recon['technique'].isin(selected_techniques)]
    if selected_rates:
        df_recon = df_recon[df_recon['rate_percent'].isin(selected_rates)]
    
    # Filter out NaN values for selected metric
    df_filtered = df_filtered[df_filtered[m.key].notna()]
    df_recon = df_recon[df_recon[m.key].notna()]
    
    # Display overview metrics for RECONSTRUCTION MODELS
    st.header("📈 Overview: Reconstruction Models")
    
    if len(df_recon) > 0:
        col1, col2, col3, col4, col5 = st.columns(5)
        mean_v = df_recon[m.key].mean()
        by_recon = df_recon.groupby("reconstruction_model")[m.key].mean()
        if m.lower_is_better:
            best_model = by_recon.idxmin()
            best_v = by_recon.min()
        else:
            best_model = by_recon.idxmax()
            best_v = by_recon.max()
        mean_s = f"{mean_v:.2f}%" if m.value_is_percent else f"{mean_v:.4f}"
        best_s = f"{best_v:.2f}%" if m.value_is_percent else f"{best_v:.4f}"
        with col1:
            st.metric("Recon Models", df_recon["reconstruction_model"].nunique())
        with col2:
            st.metric(f"Mean {m.label}", mean_s)
        with col3:
            st.metric("Best Model", best_model)
        with col4:
            st.metric(f"Best {m.label}", best_s)
        with col5:
            st.metric("Total Records", len(df_recon))
    else:
        st.warning("No reconstructed data with current filters")
    
    st.markdown("---")
    
    # Load training metrics for new tabs
    df_training = load_training_metrics()
    
    # Visualization tabs - RECONSTRUCTION MODEL focused
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs([
        "🔧 By Recon Model",
        "📉 By Technique", 
        "📈 By Missing Rate",
        "🔥 Heatmaps",
        "🏆 Best/Worst",
        "📊 Statistical Tests",
        "📁 By Dataset",
        "⏱️ Performance",
        "💻 Resources",
        "📋 Raw Data"
    ])
    
    # Tab 1: Main comparison - Reconstruction Models
    with tab1:
        st.header("🔧 Comparison by Reconstruction Model")
        st.caption("Main axis: Which reconstruction method produces best prediction results?")
        
        # Local filters - 3 columns
        col1, col2, col3 = st.columns(3)
        with col1:
            filter_pred_model = st.selectbox(
                "Filter by Prediction Model",
                ['All'] + all_pred_models,
                key='tab1_pred_model',
                index=1,
                help="Select ONE prediction model to compare reconstruction methods"
            )
        with col2:
            filter_technique = st.selectbox(
                "Filter by Technique",
                ['All'] + all_techniques,
                key='tab1_technique'
            )
        with col3:
            filter_rate = st.selectbox(
                "Filter by Missing Rate (%)",
                ['All'] + [int(r) for r in all_rates],
                key='tab1_rate'
            )
        
        # Apply local prediction model filter
        df_tab1 = df_recon.copy()
        if filter_pred_model != 'All':
            df_tab1 = df_tab1[df_tab1['prediction_model'] == filter_pred_model]
            st.info(f"📊 Showing results for prediction model: **{filter_pred_model}**")
        
        plot_mape_by_reconstruction_model_main(
            df_tab1,
            m,
            technique=None if filter_technique == "All" else filter_technique,
            rate=None if filter_rate == "All" else filter_rate,
        )
        
        # Data for detailed stats (technique + rate only, so all prediction models)
        df_tab1_detail = df_recon.copy()
        if filter_technique != 'All':
            df_tab1_detail = df_tab1_detail[df_tab1_detail['technique'] == filter_technique]
        if filter_rate != 'All':
            df_tab1_detail = df_tab1_detail[df_tab1_detail['rate_percent'] == filter_rate]
        
        st.markdown("---")
        st.subheader(
            f"📊 Detailed Statistics: {m.label} per Reconstruction × Prediction Model"
        )
        st.caption(
            f"Mean {m.label} for each combination of reconstruction model and prediction model"
        )
        
        if len(df_tab1_detail) > 0:
            pivot_dec = 2 if m.value_is_percent else 4
            pivot_mean = df_tab1_detail.pivot_table(
                values=m.key,
                index='prediction_model',
                columns='reconstruction_model',
                aggfunc='mean'
            ).round(pivot_dec)
            pivot_count = df_tab1_detail.pivot_table(
                values=m.key,
                index='prediction_model',
                columns='reconstruction_model',
                aggfunc='count'
            )
            
            st.write(
                f"**Mean {m.label}** — per row: green = best reconstruction model, red = worst"
            )
            pivot_fmt = "{:.2f}%" if m.value_is_percent else "{:.4f}"
            st.dataframe(
                pivot_mean.style.apply(
                    lambda row: _style_row_best_worst(row, m.lower_is_better), axis=1
                ).format(pivot_fmt, na_rep='—'),
                width='stretch'
            )
            
            with st.expander("📋 Show count of predictions per cell"):
                st.dataframe(pivot_count.astype(int), width='stretch')
            
            st.write("**Summary per Reconstruction Model (across all prediction models)**")
            mean_hdr = f"Mean {m.label}"
            recon_summary = df_tab1_detail.groupby('reconstruction_model')[m.key].agg([
                'mean', 'std', 'median', 'min', 'max', 'count'
            ]).round(pivot_dec)
            recon_summary.columns = [mean_hdr, 'Std', 'Median', 'Min', 'Max', 'N']
            recon_summary = recon_summary.sort_values(
                mean_hdr, ascending=m.lower_is_better
            )
            mfmt = "{:.2f}%" if m.value_is_percent else "{:.4f}"
            st.dataframe(
                recon_summary.style.format({
                    mean_hdr: mfmt,
                    'Std': mfmt,
                    'Median': mfmt,
                    'Min': mfmt,
                    'Max': mfmt,
                    'N': '{:.0f}'
                }),
                width='stretch'
            )
        else:
            st.warning("No data for detailed statistics with current filters.")
        
        st.markdown("---")
        st.subheader(f"📋 {m.label} per Prediction (each row = one prediction)")
        st.caption(
            "Every prediction record: dataset, technique, rate, reconstruction model, prediction model, "
            f"and {m.label}"
        )
        
        if len(df_tab1) > 0:
            row_cols = [
                'dataset_name', 'technique', 'rate_percent',
                'reconstruction_model', 'prediction_model', 'prediction_iteration',
                m.key,
            ]
            if 'n_samples' in df_tab1.columns:
                row_cols.append('n_samples')
            
            df_rows = df_tab1[[c for c in row_cols if c in df_tab1.columns]].copy()
            df_rows = df_rows.sort_values(['reconstruction_model', 'prediction_model', 'dataset_name', 'technique', 'rate_percent'])
            metric_col_disp = m.label
            df_rows = df_rows.rename(columns={
                'dataset_name': 'Dataset',
                'technique': 'Technique',
                'rate_percent': 'Rate %',
                'reconstruction_model': 'Recon Model',
                'prediction_model': 'Pred Model',
                'prediction_iteration': 'Pred Iter',
                m.key: metric_col_disp,
                'n_samples': 'N Samples'
            })
            
            n_show = st.slider("Show first N rows", 10, 500, min(100, len(df_rows)), key='tab1_rows_slider')
            fmt_dict = {
                metric_col_disp: '{:.2f}%' if m.value_is_percent else '{:.4f}'
            }
            if 'Rate %' in df_rows.columns:
                fmt_dict['Rate %'] = '{:.0f}'
            if 'N Samples' in df_rows.columns:
                fmt_dict['N Samples'] = '{:.0f}'
            st.dataframe(
                df_rows.head(n_show).style.format(fmt_dict, na_rep='—'),
                width='stretch',
                height=400
            )
            st.caption(f"Showing {min(n_show, len(df_rows))} of {len(df_rows)} prediction rows.")
            
            csv_rows = df_rows.to_csv(index=False)
            st.download_button(
                label="📥 Download full per-prediction data (CSV)",
                data=csv_rows,
                file_name=f"prediction_{m.key}_per_row_tab1.csv",
                mime="text/csv",
                key='tab1_download_rows'
            )
        else:
            st.warning("No prediction rows with current filters.")
    
    # Tab 2: By Technique
    with tab2:
        st.header("📉 Reconstruction Models by Missingness Technique")
        st.caption("How do reconstruction models perform across different missingness types?")
        
        if len(df_recon) > 0:
            plot_recon_by_technique(df_recon, m)
        else:
            st.warning("No reconstructed data available")
    
    # Tab 3: By Missing Rate
    with tab3:
        st.header("📈 Reconstruction Models by Missing Rate")
        st.caption(f"How does {m.label} change with increasing missing rate?")
        
        filter_technique = st.selectbox(
            "Filter by Technique",
            ['All'] + all_techniques,
            key='tab3_technique'
        )
        
        if len(df_recon) > 0:
            plot_recon_by_rate(
                df_recon,
                m,
                technique=None if filter_technique == 'All' else filter_technique
            )
        else:
            st.warning("No reconstructed data available")
    
    # Tab 4: Heatmaps
    with tab4:
        st.header("🔥 Heatmaps")
        st.caption("Visual comparison matrices for reconstruction models")
        
        heatmap_type = st.radio(
            "Select heatmap type",
            ["Recon Model vs Technique", "Recon Model vs Missing Rate", "Pred Model vs Recon Model"],
            horizontal=True
        )
        
        if len(df_recon) == 0:
            st.warning("No reconstructed data available")
        elif heatmap_type == "Recon Model vs Technique":
            plot_heatmap_recon_vs_technique(df_recon, m)
        elif heatmap_type == "Recon Model vs Missing Rate":
            filter_technique = st.selectbox(
                "Filter by Technique",
                ['All'] + all_techniques,
                key='tab4_technique_filter'
            )
            plot_heatmap_recon_vs_rate(
                df_recon,
                m,
                technique=None if filter_technique == 'All' else filter_technique
            )
        else:
            plot_heatmap_pred_vs_recon(df_recon, m)
    
    # Tab 5: Best/Worst
    with tab5:
        st.header("🏆 Best and Worst Reconstruction Models")
        
        top_n = st.slider("Number of models to show", 3, 10, min(5, len(all_recon_models)))
        
        if len(df_recon) > 0:
            plot_best_worst_reconstruction_models(df_recon, m, top_n=top_n)
        else:
            st.warning("No reconstructed data available")
    
    # Tab 6: Statistical Tests
    with tab6:
        st.header("📊 Statistical Significance Tests")
        st.caption("Pairwise t-tests between RECONSTRUCTION models")
        
        if len(df_recon) == 0:
            st.warning("No reconstructed data available")
        else:
            from utils.statistical_tests import (
                perform_pairwise_ttests, 
                get_model_statistics,
                get_significance_summary
            )
            
            with st.expander("ℹ️ How to interpret this analysis", expanded=False):
                st.markdown(f"""
                **Statistical Significance Testing**:
                
                Pairwise t-tests between reconstruction models to determine if **{m.label}** differences are statistically significant.
                
                **Legend**:
                - **🟩 +2 (p<0.01)**: Row model is **significantly better** than column model
                - **🟢 +1 (p<0.05)**: Row model is **significantly better** than column model
                - **⬜ 0**: No significant difference
                - **🔴 -1 (p<0.05)**: Row model is **significantly worse** than column model
                - **🟥 -2 (p<0.01)**: Row model is **significantly worse** than column model
                """)
            
            st.divider()
            
            # Rename column for statistical functions
            df_for_stats = df_recon.rename(columns={'reconstruction_model': 'model'})
            
            st.subheader("Reconstruction Model Statistics")
            model_stats = get_model_statistics(
                df_for_stats, metric=m.key, lower_is_better=m.lower_is_better
            )
            st.dataframe(
                model_stats.style.background_gradient(
                    subset=['mean'], cmap=_metric_cmap(m.lower_is_better)
                ),
                width='stretch'
            )
            
            st.divider()
            
            st.subheader("Pairwise Statistical Significance Matrix")
            significance_matrix = perform_pairwise_ttests(
                df_for_stats,
                metric=m.key,
                alpha_01=0.01,
                alpha_05=0.05,
                lower_is_better=m.lower_is_better,
            )
            
            def color_significance(val):
                if val == 2:
                    return 'background-color: #006400; color: white'
                elif val == 1:
                    return 'background-color: #90EE90; color: black'
                elif val == 0:
                    return 'background-color: #FFFFFF; color: black'
                elif val == -1:
                    return 'background-color: #FF6B6B; color: black'
                elif val == -2:
                    return 'background-color: #8B0000; color: white'
                else:
                    return 'background-color: #CCCCCC; color: black'
            
            styled_matrix = significance_matrix.style.map(color_significance)
            st.dataframe(styled_matrix, width='stretch', height=500)
            
            st.caption("**Legend**: 🟩 +2 (p<0.01 better) | 🟢 +1 (p<0.05 better) | ⬜ 0 (no diff) | 🔴 -1 (p<0.05 worse) | 🟥 -2 (p<0.01 worse)")
            
            st.divider()
            
            st.subheader("Significance Summary")
            significance_summary = get_significance_summary(significance_matrix)
            summary_df = pd.DataFrame(significance_summary).T.reset_index()
            summary_df.columns = ['Recon Model', 'Better (p<0.01)', 'Better (p<0.05)', 'No Diff', 'Worse (p<0.05)', 'Worse (p<0.01)']
            summary_df = summary_df.sort_values('Better (p<0.01)', ascending=False)
            st.dataframe(summary_df, width='stretch')
            
            # ====================================================================
            # PREDICTION MODELS T-TESTS
            # ====================================================================
            st.markdown("---")
            st.header("📊 Statistical Tests: Prediction Models")
            st.caption("Pairwise t-tests between PREDICTION models (which prediction model gives better results?)")
            
            # Rename column for statistical functions
            df_pred_stats = df_recon.rename(columns={'prediction_model': 'model'})
            
            st.subheader("Prediction Model Statistics")
            pred_model_stats = get_model_statistics(
                df_pred_stats, metric=m.key, lower_is_better=m.lower_is_better
            )
            st.dataframe(
                pred_model_stats.style.background_gradient(
                    subset=['mean'], cmap=_metric_cmap(m.lower_is_better)
                ),
                width='stretch'
            )
            
            st.divider()
            
            st.subheader("Pairwise Statistical Significance Matrix (Prediction Models)")
            pred_significance_matrix = perform_pairwise_ttests(
                df_pred_stats,
                metric=m.key,
                alpha_01=0.01,
                alpha_05=0.05,
                lower_is_better=m.lower_is_better,
            )
            
            styled_pred_matrix = pred_significance_matrix.style.map(color_significance)
            st.dataframe(styled_pred_matrix, width='stretch', height=500)
            
            st.caption("**Legend**: 🟩 +2 (p<0.01 better) | 🟢 +1 (p<0.05 better) | ⬜ 0 (no diff) | 🔴 -1 (p<0.05 worse) | 🟥 -2 (p<0.01 worse)")
            
            st.divider()
            
            st.subheader("Significance Summary (Prediction Models)")
            pred_significance_summary = get_significance_summary(pred_significance_matrix)
            pred_summary_df = pd.DataFrame(pred_significance_summary).T.reset_index()
            pred_summary_df.columns = ['Pred Model', 'Better (p<0.01)', 'Better (p<0.05)', 'No Diff', 'Worse (p<0.05)', 'Worse (p<0.01)']
            pred_summary_df = pred_summary_df.sort_values('Better (p<0.01)', ascending=False)
            st.dataframe(pred_summary_df, width='stretch')
    
    # Tab 7: By Dataset
    with tab7:
        st.header("📁 Comparison by Dataset")
        
        if len(df_recon) > 0:
            df_stats = (
                df_recon.groupby(["reconstruction_model", "dataset_name"])[m.key]
                .mean()
                .reset_index()
            )
            
            fig = px.bar(
                df_stats,
                x='dataset_name',
                y=m.key,
                color='reconstruction_model',
                barmode='group',
                title=f'Reconstruction Model {m.label} by Dataset',
                labels={
                    m.key: f'Mean {m.label}',
                    'dataset_name': 'Dataset',
                    'reconstruction_model': 'Recon Model'
                }
            )
            fig.update_layout(xaxis_tickangle=-45, height=500)
            st.plotly_chart(fig, width='stretch')
        else:
            st.warning("No reconstructed data available")
    
    # Tab 8: Performance (Time) - Training + Prediction
    with tab8:
        st.header("⏱️ Time Analysis (Training + Prediction)")
        
        # Load training metrics
        df_training = load_training_metrics()
        has_training = not df_training.empty and 'time_seconds' in df_training.columns
        has_prediction = 'time_seconds' in df_filtered.columns and not df_filtered['time_seconds'].isna().all()
        
        if not has_training and not has_prediction:
            st.warning("⚠️ No time metrics available.")
        else:
            # === TRAINING TIME SECTION ===
            st.subheader("🎓 Training Time")
            
            if has_training:
                df_train_valid = df_training[df_training['time_seconds'].notna()].copy()
                
                if len(df_train_valid) > 0:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Training", f"{df_train_valid['time_seconds'].sum():.1f}s")
                    with col2:
                        st.metric("Avg per Model", f"{df_train_valid['time_seconds'].mean():.1f}s")
                    with col3:
                        st.metric("Fastest", f"{df_train_valid['time_seconds'].min():.1f}s")
                    with col4:
                        st.metric("Slowest", f"{df_train_valid['time_seconds'].max():.1f}s")
                    
                    # Training time by model
                    train_time = df_train_valid.groupby('model')['time_seconds'].agg(['mean', 'std', 'sum']).reset_index()
                    train_time = train_time.sort_values('mean', ascending=False)
                    
                    fig = px.bar(
                        train_time,
                        x='mean',
                        y='model',
                        orientation='h',
                        error_x='std',
                        title="Training Time by Model",
                        labels={'mean': 'Avg Time (s)', 'model': 'Model'},
                        color='mean',
                        color_continuous_scale='Blues'
                    )
                    fig.update_layout(height=max(350, len(train_time) * 35), showlegend=False)
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.info("No training data available")
            else:
                st.info("No training metrics file found. Run `make train-models` first.")
            
            st.markdown("---")
            
            # === PREDICTION TIME SECTION ===
            st.subheader("🔮 Prediction Time")
            
            if has_prediction:
                df_perf = df_filtered[df_filtered['time_seconds'].notna()].copy()
                
                if len(df_perf) > 0:
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Prediction", f"{df_perf['time_seconds'].sum():.1f}s")
                    with col2:
                        st.metric("Avg per File", f"{df_perf['time_seconds'].mean():.2f}s")
                    with col3:
                        st.metric("Fastest", f"{df_perf['time_seconds'].min():.2f}s")
                    with col4:
                        st.metric("Slowest", f"{df_perf['time_seconds'].max():.2f}s")
                    
                    # Prediction time by model
                    pred_time = df_perf.groupby('prediction_model')['time_seconds'].agg(['mean', 'std', 'sum']).reset_index()
                    pred_time = pred_time.sort_values('mean', ascending=False)
                    
                    fig = px.bar(
                        pred_time,
                        x='mean',
                        y='prediction_model',
                        orientation='h',
                        error_x='std',
                        title="Prediction Time by Model",
                        labels={'mean': 'Avg Time (s)', 'prediction_model': 'Model'},
                        color='mean',
                        color_continuous_scale='Reds'
                    )
                    fig.update_layout(height=max(350, len(pred_time) * 35), showlegend=False)
                    st.plotly_chart(fig, width='stretch')
                else:
                    st.info("No prediction time data")
            else:
                st.info("No prediction metrics available.")
            
            st.markdown("---")
            
            # === TOTAL TIME (TRAINING + PREDICTION) ===
            st.subheader("📊 Total Time (Training + Prediction)")
            
            if has_training and has_prediction:
                df_train_valid = df_training[df_training['time_seconds'].notna()].copy()
                df_perf = df_filtered[df_filtered['time_seconds'].notna()].copy()
                
                # Aggregate
                train_totals = df_train_valid.groupby('model')['time_seconds'].sum().reset_index()
                train_totals.columns = ['model', 'training_time']
                
                pred_totals = df_perf.groupby('prediction_model')['time_seconds'].sum().reset_index()
                pred_totals.columns = ['model', 'prediction_time']
                
                total_times = train_totals.merge(pred_totals, on='model', how='outer').fillna(0)
                total_times['total_time'] = total_times['training_time'] + total_times['prediction_time']
                total_times = total_times.sort_values('total_time', ascending=True)
                
                # Stacked bar chart
                fig = go.Figure()
                fig.add_trace(go.Bar(
                    y=total_times['model'],
                    x=total_times['training_time'],
                    name='Training',
                    orientation='h',
                    marker_color='#3498db'
                ))
                fig.add_trace(go.Bar(
                    y=total_times['model'],
                    x=total_times['prediction_time'],
                    name='Prediction',
                    orientation='h',
                    marker_color='#e74c3c'
                ))
                
                fig.update_layout(
                    barmode='stack',
                    title='Total Time by Model (Training + Prediction)',
                    xaxis_title='Time (seconds)',
                    yaxis_title='Model',
                    height=max(400, len(total_times) * 40),
                    legend=dict(orientation='h', yanchor='bottom', y=1.02)
                )
                st.plotly_chart(fig, width='stretch')
                
                # Summary metrics
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Training", f"{total_times['training_time'].sum():.1f}s")
                with col2:
                    st.metric("Total Prediction", f"{total_times['prediction_time'].sum():.1f}s")
                with col3:
                    st.metric("Grand Total", f"{total_times['total_time'].sum():.1f}s")
    
    # Tab 9: Resources (CPU, RAM, GPU) - Training + Prediction
    with tab9:
        st.header("💻 Resource Usage (Training + Prediction)")
        
        # Load training metrics
        df_training = load_training_metrics()
        has_training = not df_training.empty and 'cpu_cores_used' in df_training.columns
        has_prediction = 'cpu_cores_used' in df_filtered.columns and not df_filtered['cpu_cores_used'].isna().all()
        
        if not has_training and not has_prediction:
            st.warning("⚠️ No resource metrics available.")
        else:
            # === TRAINING RESOURCES ===
            st.subheader("🎓 Training Resources")
            
            if has_training:
                df_train = df_training[df_training['cpu_cores_used'].notna()].copy()
                
                if len(df_train) > 0:
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Avg CPU Cores", f"{df_train['cpu_cores_used'].mean():.2f}")
                    with col2:
                        st.metric("Avg RAM", f"{df_train['memory_mb'].mean():.1f} MB")
                    with col3:
                        if 'gpu_percent' in df_train.columns and df_train['gpu_percent'].notna().any():
                            st.metric("Avg GPU %", f"{df_train['gpu_percent'].mean():.1f}%")
                        else:
                            st.metric("GPU", "N/A")
                    with col4:
                        if 'gpu_memory_mb' in df_train.columns and df_train['gpu_memory_mb'].notna().any():
                            st.metric("Avg GPU Mem", f"{df_train['gpu_memory_mb'].mean():.0f} MB")
                        else:
                            st.metric("GPU Mem", "N/A")
                    
                    # Training resources by model
                    agg_dict = {'cpu_cores_used': 'mean', 'memory_mb': 'mean'}
                    if 'gpu_percent' in df_train.columns:
                        agg_dict['gpu_percent'] = 'mean'
                    if 'gpu_memory_mb' in df_train.columns:
                        agg_dict['gpu_memory_mb'] = 'mean'
                    
                    train_resources = df_train.groupby('model').agg(agg_dict).reset_index()
                    
                    # CPU + RAM chart
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = px.bar(
                            train_resources.sort_values('cpu_cores_used', ascending=False),
                            x='cpu_cores_used',
                            y='model',
                            orientation='h',
                            title="Training: CPU Cores by Model",
                            labels={'cpu_cores_used': 'CPU Cores', 'model': 'Model'},
                            color='cpu_cores_used',
                            color_continuous_scale='Oranges'
                        )
                        fig.update_layout(height=300, showlegend=False)
                        st.plotly_chart(fig, width='stretch')
                    
                    with col2:
                        fig = px.bar(
                            train_resources.sort_values('memory_mb', ascending=False),
                            x='memory_mb',
                            y='model',
                            orientation='h',
                            title="Training: RAM by Model",
                            labels={'memory_mb': 'RAM (MB)', 'model': 'Model'},
                            color='memory_mb',
                            color_continuous_scale='Purples'
                        )
                        fig.update_layout(height=300, showlegend=False)
                        st.plotly_chart(fig, width='stretch')
                    
                    # GPU chart if available
                    if 'gpu_memory_mb' in train_resources.columns and train_resources['gpu_memory_mb'].notna().any():
                        fig = px.bar(
                            train_resources.sort_values('gpu_memory_mb', ascending=False),
                            x='gpu_memory_mb',
                            y='model',
                            orientation='h',
                            title="Training: GPU Memory by Model",
                            labels={'gpu_memory_mb': 'GPU Memory (MB)', 'model': 'Model'},
                            color='gpu_memory_mb',
                            color_continuous_scale='Greens'
                        )
                        fig.update_layout(height=300, showlegend=False)
                        st.plotly_chart(fig, width='stretch')
                else:
                    st.info("No training resource data")
            else:
                st.info("No training metrics file found.")
            
            st.markdown("---")
            
            # === PREDICTION RESOURCES ===
            st.subheader("🔮 Prediction Resources")
            
            if has_prediction:
                df_perf = df_filtered[df_filtered['cpu_cores_used'].notna()].copy()
                
                if len(df_perf) > 0:
                    # Summary metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Avg CPU Cores", f"{df_perf['cpu_cores_used'].mean():.2f}")
                    with col2:
                        st.metric("Avg RAM", f"{df_perf['memory_mb'].mean():.1f} MB")
                    with col3:
                        if 'gpu_percent' in df_perf.columns and df_perf['gpu_percent'].notna().any():
                            st.metric("Avg GPU %", f"{df_perf['gpu_percent'].mean():.1f}%")
                        else:
                            st.metric("GPU", "N/A")
                    with col4:
                        if 'gpu_memory_mb' in df_perf.columns and df_perf['gpu_memory_mb'].notna().any():
                            st.metric("Avg GPU Mem", f"{df_perf['gpu_memory_mb'].mean():.0f} MB")
                        else:
                            st.metric("GPU Mem", "N/A")
                    
                    # Prediction resources by model
                    agg_dict = {'cpu_cores_used': 'mean', 'memory_mb': 'mean'}
                    if 'gpu_percent' in df_perf.columns:
                        agg_dict['gpu_percent'] = 'mean'
                    if 'gpu_memory_mb' in df_perf.columns:
                        agg_dict['gpu_memory_mb'] = 'mean'
                    
                    pred_resources = df_perf.groupby('prediction_model').agg(agg_dict).reset_index()
                    
                    # CPU + RAM chart
                    col1, col2 = st.columns(2)
                    with col1:
                        fig = px.bar(
                            pred_resources.sort_values('cpu_cores_used', ascending=False),
                            x='cpu_cores_used',
                            y='prediction_model',
                            orientation='h',
                            title="Prediction: CPU Cores by Model",
                            labels={'cpu_cores_used': 'CPU Cores', 'prediction_model': 'Model'},
                            color='cpu_cores_used',
                            color_continuous_scale='Oranges'
                        )
                        fig.update_layout(height=300, showlegend=False)
                        st.plotly_chart(fig, width='stretch')
                    
                    with col2:
                        fig = px.bar(
                            pred_resources.sort_values('memory_mb', ascending=False),
                            x='memory_mb',
                            y='prediction_model',
                            orientation='h',
                            title="Prediction: RAM by Model",
                            labels={'memory_mb': 'RAM (MB)', 'prediction_model': 'Model'},
                            color='memory_mb',
                            color_continuous_scale='Purples'
                        )
                        fig.update_layout(height=300, showlegend=False)
                        st.plotly_chart(fig, width='stretch')
                    
                    # GPU chart if available
                    if 'gpu_memory_mb' in pred_resources.columns and pred_resources['gpu_memory_mb'].notna().any():
                        fig = px.bar(
                            pred_resources.sort_values('gpu_memory_mb', ascending=False),
                            x='gpu_memory_mb',
                            y='prediction_model',
                            orientation='h',
                            title="Prediction: GPU Memory by Model",
                            labels={'gpu_memory_mb': 'GPU Memory (MB)', 'prediction_model': 'Model'},
                            color='gpu_memory_mb',
                            color_continuous_scale='Greens'
                        )
                        fig.update_layout(height=300, showlegend=False)
                        st.plotly_chart(fig, width='stretch')
                else:
                    st.info("No prediction resource data")
            else:
                st.info("No prediction metrics available.")
            
            st.markdown("---")
            
            # === COMBINED CPU+GPU USAGE ===
            st.subheader("🔥 Combined CPU + GPU Usage")
            
            if has_training or has_prediction:
                combined_data = []
                
                if has_training:
                    df_train = df_training[df_training['cpu_cores_used'].notna()].copy()
                    for _, row in df_train.iterrows():
                        gpu_usage = row.get('gpu_percent', 0)
                        gpu_usage = 0 if pd.isna(gpu_usage) else gpu_usage
                        cpu_usage = row['cpu_cores_used']
                        cpu_usage = 0 if pd.isna(cpu_usage) else cpu_usage
                        
                        combined_data.append({
                            'model': row['model'],
                            'phase': 'Training',
                            'cpu_cores': cpu_usage,
                            'gpu_percent': gpu_usage,
                            'combined_score': cpu_usage + (gpu_usage / 10)  # Weighted combination
                        })
                
                if has_prediction:
                    df_perf = df_filtered[df_filtered['cpu_cores_used'].notna()].copy()
                    
                    # Build aggregation dict carefully
                    agg_dict = {'cpu_cores_used': 'mean'}
                    if 'gpu_percent' in df_perf.columns and df_perf['gpu_percent'].notna().any():
                        agg_dict['gpu_percent'] = 'mean'
                    
                    pred_agg = df_perf.groupby('prediction_model').agg(agg_dict).reset_index()
                    
                    for _, row in pred_agg.iterrows():
                        gpu_usage = row.get('gpu_percent', 0)
                        gpu_usage = 0 if pd.isna(gpu_usage) else gpu_usage
                        cpu_usage = row['cpu_cores_used']
                        cpu_usage = 0 if pd.isna(cpu_usage) else cpu_usage
                        
                        combined_data.append({
                            'model': row['prediction_model'],
                            'phase': 'Prediction',
                            'cpu_cores': cpu_usage,
                            'gpu_percent': gpu_usage,
                            'combined_score': cpu_usage + (gpu_usage / 10)
                        })
                
                if combined_data:
                    df_combined = pd.DataFrame(combined_data)
                    
                    # Fill any remaining NaN values
                    df_combined = df_combined.fillna(0)
                    
                    # Ensure combined_score has minimum value for visibility
                    df_combined['combined_score'] = df_combined['combined_score'].clip(lower=0.1)
                    
                    fig = px.scatter(
                        df_combined,
                        x='cpu_cores',
                        y='gpu_percent',
                        color='phase',
                        text='model',
                        size='combined_score',
                        title="CPU vs GPU Usage (Training & Prediction)",
                        labels={'cpu_cores': 'CPU Cores', 'gpu_percent': 'GPU %', 'phase': 'Phase'},
                        color_discrete_map={'Training': '#3498db', 'Prediction': '#e74c3c'}
                    )
                    fig.update_traces(textposition='top center')
                    fig.update_layout(height=500)
                    st.plotly_chart(fig, width='stretch')
    
    # Tab 10: Raw Data
    with tab10:
        st.header("📋 Raw Data")
        
        data_source = st.radio("Data source", ["Reconstructed only", "All data"], horizontal=True)
        df_display = df_recon if data_source == "Reconstructed only" else df_filtered
        
        search_term = st.text_input("Search", "")
        if search_term:
            mask = df_display.astype(str).apply(lambda x: x.str.contains(search_term, case=False)).any(axis=1)
            df_display = df_display[mask]
        
        col1, col2 = st.columns(2)
        with col1:
            cols_list = df_display.columns.tolist()
            sort_default = cols_list.index(m.key) if m.key in cols_list else 0
            sort_column = st.selectbox("Sort by", cols_list, index=sort_default)
        with col2:
            sort_order = st.radio("Order", ['Ascending', 'Descending'])
        
        df_display = df_display.sort_values(sort_column, ascending=(sort_order == 'Ascending'))
        
        st.dataframe(df_display, width='stretch')
        
        csv = df_display.to_csv(index=False)
        st.download_button(
            label="📥 Download CSV",
            data=csv,
            file_name=f"recon_comparison_{selected_file_name}",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
