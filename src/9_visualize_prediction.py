#!/usr/bin/env python3
"""
Streamlit Visualization App for Prediction Results
Interactive dashboard for comparing prediction models and MAPE metrics.
Part of uniTS-MissRecoPred framework.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Add src directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))


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


def plot_mape_by_model(df: pd.DataFrame, source_type: str = None, technique: str = None):
    """Plot MAPE comparison by prediction model"""
    df_filtered = df.copy()
    
    # Apply filters
    if source_type:
        df_filtered = df_filtered[df_filtered['source_type'] == source_type]
    if technique:
        df_filtered = df_filtered[df_filtered['technique'] == technique]
    
    if df_filtered.empty:
        st.warning("No data available for selected filters")
        return
    
    # Group by prediction_model and calculate statistics
    df_stats = df_filtered.groupby('prediction_model')['mape'].agg(['mean', 'std', 'min', 'max']).reset_index()
    df_stats = df_stats.sort_values('mean')
    
    # Create bar plot
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_stats['prediction_model'],
        y=df_stats['mean'],
        error_y=dict(type='data', array=df_stats['std']),
        marker_color='lightblue',
        name='Mean MAPE'
    ))
    
    fig.update_layout(
        title='Mean Absolute Percentage Error by Prediction Model',
        xaxis_title='Prediction Model',
        yaxis_title='MAPE (%)',
        xaxis_tickangle=-45,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show statistics table
    st.subheader("Statistics")
    st.dataframe(df_stats.style.format({
        'mean': '{:.2f}%',
        'std': '{:.2f}%',
        'min': '{:.2f}%',
        'max': '{:.2f}%'
    }), use_container_width=True)


def plot_mape_by_source_type(df: pd.DataFrame, model: str = None):
    """Plot MAPE comparison by source type (original vs reconstructed)"""
    df_filtered = df.copy()
    
    # Apply filters
    if model:
        df_filtered = df_filtered[df_filtered['prediction_model'] == model]
    
    if df_filtered.empty:
        st.warning("No data available for selected filters")
        return
    
    # Group by source_type and calculate statistics
    df_stats = df_filtered.groupby('source_type')['mape'].agg(['mean', 'std', 'min', 'max']).reset_index()
    df_stats = df_stats.sort_values('mean')
    
    # Create bar plot
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_stats['source_type'],
        y=df_stats['mean'],
        error_y=dict(type='data', array=df_stats['std']),
        marker_color='lightgreen',
        name='Mean MAPE'
    ))
    
    fig.update_layout(
        title='Mean Absolute Percentage Error by Source Type',
        xaxis_title='Source Type',
        yaxis_title='MAPE (%)',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show statistics table
    st.subheader("Statistics")
    st.dataframe(df_stats.style.format({
        'mean': '{:.2f}%',
        'std': '{:.2f}%',
        'min': '{:.2f}%',
        'max': '{:.2f}%'
    }), use_container_width=True)


def plot_mape_by_technique(df: pd.DataFrame, model: str = None, rate: int = None):
    """Plot MAPE comparison by missingness technique (for reconstructed data only)"""
    df_filtered = df.copy()
    
    # Filter to only reconstructed data (technique is null for original)
    df_filtered = df_filtered[df_filtered['source_type'] == 'reconstructed']
    
    # Apply filters
    if model:
        df_filtered = df_filtered[df_filtered['prediction_model'] == model]
    if rate:
        df_filtered = df_filtered[df_filtered['rate_percent'] == rate]
    
    if df_filtered.empty:
        st.warning("No data available for selected filters (only reconstructed data has techniques)")
        return
    
    # Group by technique and calculate statistics
    df_stats = df_filtered.groupby('technique')['mape'].agg(['mean', 'std', 'min', 'max']).reset_index()
    df_stats = df_stats.sort_values('mean')
    
    # Create bar plot
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_stats['technique'],
        y=df_stats['mean'],
        error_y=dict(type='data', array=df_stats['std']),
        marker_color='coral',
        name='Mean MAPE'
    ))
    
    fig.update_layout(
        title='Mean Absolute Percentage Error by Missingness Technique (Reconstructed Data)',
        xaxis_title='Missingness Technique',
        yaxis_title='MAPE (%)',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show statistics table
    st.subheader("Statistics")
    st.dataframe(df_stats.style.format({
        'mean': '{:.2f}%',
        'std': '{:.2f}%',
        'min': '{:.2f}%',
        'max': '{:.2f}%'
    }), use_container_width=True)


def plot_mape_by_rate(df: pd.DataFrame, model: str = None, technique: str = None):
    """Plot MAPE comparison by missing rate (for reconstructed data only)"""
    df_filtered = df.copy()
    
    # Filter to only reconstructed data
    df_filtered = df_filtered[df_filtered['source_type'] == 'reconstructed']
    
    # Apply filters
    if model:
        df_filtered = df_filtered[df_filtered['prediction_model'] == model]
    if technique:
        df_filtered = df_filtered[df_filtered['technique'] == technique]
    
    if df_filtered.empty:
        st.warning("No data available for selected filters")
        return
    
    # Group by rate and calculate statistics
    df_stats = df_filtered.groupby('rate_percent')['mape'].agg(['mean', 'std', 'min', 'max']).reset_index()
    df_stats = df_stats.sort_values('rate_percent')
    
    # Create line plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_stats['rate_percent'],
        y=df_stats['mean'],
        mode='lines+markers',
        error_y=dict(type='data', array=df_stats['std']),
        marker=dict(size=10, color='mediumpurple'),
        line=dict(width=2),
        name='Mean MAPE'
    ))
    
    fig.update_layout(
        title='Mean Absolute Percentage Error by Missing Rate (Reconstructed Data)',
        xaxis_title='Missing Rate (%)',
        yaxis_title='MAPE (%)',
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show statistics table
    st.subheader("Statistics")
    df_stats_display = df_stats.copy()
    df_stats_display['rate_percent'] = df_stats_display['rate_percent'].astype(str) + '%'
    st.dataframe(df_stats_display.style.format({
        'mean': '{:.2f}%',
        'std': '{:.2f}%',
        'min': '{:.2f}%',
        'max': '{:.2f}%'
    }), use_container_width=True)


def plot_mape_by_reconstruction_model(df: pd.DataFrame, pred_model: str = None, technique: str = None):
    """Plot MAPE comparison by reconstruction model (for reconstructed data only)"""
    df_filtered = df.copy()
    
    # Filter to only reconstructed data
    df_filtered = df_filtered[df_filtered['source_type'] == 'reconstructed']
    
    # Apply filters
    if pred_model:
        df_filtered = df_filtered[df_filtered['prediction_model'] == pred_model]
    if technique:
        df_filtered = df_filtered[df_filtered['technique'] == technique]
    
    if df_filtered.empty:
        st.warning("No data available for selected filters")
        return
    
    # Group by reconstruction_model and calculate statistics
    df_stats = df_filtered.groupby('reconstruction_model')['mape'].agg(['mean', 'std', 'min', 'max']).reset_index()
    df_stats = df_stats.sort_values('mean')
    
    # Create bar plot
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_stats['reconstruction_model'],
        y=df_stats['mean'],
        error_y=dict(type='data', array=df_stats['std']),
        marker_color='teal',
        name='Mean MAPE'
    ))
    
    fig.update_layout(
        title='Prediction MAPE by Reconstruction Model Used',
        xaxis_title='Reconstruction Model',
        yaxis_title='MAPE (%)',
        xaxis_tickangle=-45,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Show statistics table
    st.subheader("Statistics")
    st.dataframe(df_stats.style.format({
        'mean': '{:.2f}%',
        'std': '{:.2f}%',
        'min': '{:.2f}%',
        'max': '{:.2f}%'
    }), use_container_width=True)


def plot_heatmap_pred_vs_recon(df: pd.DataFrame, sort_by_model: str = None):
    """Plot heatmap of MAPE for prediction model vs reconstruction model"""
    # Filter to only reconstructed data
    df_filtered = df[df['source_type'] == 'reconstructed'].copy()
    
    if df_filtered.empty:
        st.warning("No reconstructed data available")
        return
    
    # Calculate mean MAPE for each prediction-reconstruction model combination
    pivot_data = df_filtered.pivot_table(
        values='mape',
        index='prediction_model',
        columns='reconstruction_model',
        aggfunc='mean'
    )
    
    # Sort by selected model or alphabetically
    if sort_by_model and sort_by_model in pivot_data.columns:
        pivot_data = pivot_data.sort_values(by=sort_by_model, ascending=True)
        sort_info = f" (sorted by {sort_by_model})"
    else:
        pivot_data = pivot_data.sort_index()
        sort_info = " (alphabetical)"
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale='RdYlGn_r',
        text=np.round(pivot_data.values, 2),
        texttemplate='%{text}%',
        textfont={"size": 10},
        colorbar=dict(title='MAPE (%)')
    ))
    
    fig.update_layout(
        title=f'Heatmap: MAPE by Prediction Model vs Reconstruction Model{sort_info}',
        xaxis_title='Reconstruction Model',
        yaxis_title='Prediction Model',
        height=max(500, len(pivot_data.index) * 35)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_heatmap_pred_vs_technique(df: pd.DataFrame, sort_by_technique: str = None):
    """Plot heatmap of MAPE for prediction model vs missingness technique"""
    # Filter to only reconstructed data
    df_filtered = df[df['source_type'] == 'reconstructed'].copy()
    
    if df_filtered.empty:
        st.warning("No reconstructed data available")
        return
    
    # Calculate mean MAPE for each prediction model - technique combination
    pivot_data = df_filtered.pivot_table(
        values='mape',
        index='prediction_model',
        columns='technique',
        aggfunc='mean'
    )
    
    # Sort
    if sort_by_technique and sort_by_technique in pivot_data.columns:
        pivot_data = pivot_data.sort_values(by=sort_by_technique, ascending=True)
        sort_info = f" (sorted by {sort_by_technique})"
    else:
        pivot_data = pivot_data.sort_index()
        sort_info = " (alphabetical)"
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale='RdYlGn_r',
        text=np.round(pivot_data.values, 2),
        texttemplate='%{text}%',
        textfont={"size": 10},
        colorbar=dict(title='MAPE (%)')
    ))
    
    fig.update_layout(
        title=f'Heatmap: MAPE by Prediction Model vs Technique{sort_info}',
        xaxis_title='Missingness Technique',
        yaxis_title='Prediction Model',
        height=max(500, len(pivot_data.index) * 35)
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_dataset_comparison(df: pd.DataFrame):
    """Compare MAPE across different datasets"""
    df_stats = df.groupby('dataset_name')['mape'].agg(['mean', 'std']).reset_index()
    df_stats = df_stats.sort_values('mean')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_stats['dataset_name'],
        y=df_stats['mean'],
        error_y=dict(type='data', array=df_stats['std']),
        marker_color='mediumpurple',
        name='Mean MAPE'
    ))
    
    fig.update_layout(
        title='Mean Absolute Percentage Error by Dataset',
        xaxis_title='Dataset',
        yaxis_title='MAPE (%)',
        xaxis_tickangle=-45,
        height=500
    )
    
    st.plotly_chart(fig, use_container_width=True)


def plot_best_worst_models(df: pd.DataFrame, top_n: int = 10):
    """Show best and worst performing prediction models"""
    df_stats = df.groupby('prediction_model')['mape'].mean().reset_index()
    df_stats = df_stats.sort_values('mape')
    
    # Calculate global MAPE range for consistent axis scaling
    global_min = df_stats['mape'].min()
    global_max = df_stats['mape'].max()
    axis_range = [global_min * 0.95, global_max * 1.05]
    
    # Best models (lowest MAPE) - reverse order for display (best on top)
    best_models = df_stats.head(top_n).iloc[::-1]
    
    # Worst models (highest MAPE) - reverse order for display (worst on top)
    worst_models = df_stats.tail(top_n)
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'Top {top_n} Best Models (Lowest MAPE)', 
                       f'Top {top_n} Worst Models (Highest MAPE)')
    )
    
    # Best models
    fig.add_trace(
        go.Bar(x=best_models['mape'], y=best_models['prediction_model'], 
               orientation='h', marker_color='green', name='Best'),
        row=1, col=1
    )
    
    # Worst models
    fig.add_trace(
        go.Bar(x=worst_models['mape'], y=worst_models['prediction_model'], 
               orientation='h', marker_color='red', name='Worst'),
        row=1, col=2
    )
    
    fig.update_layout(height=max(500, top_n * 40), showlegend=False)
    fig.update_xaxes(title_text="MAPE (%)", range=axis_range, row=1, col=1)
    fig.update_xaxes(title_text="MAPE (%)", range=axis_range, row=1, col=2)
    
    st.plotly_chart(fig, use_container_width=True)


def plot_iteration_analysis(df: pd.DataFrame):
    """Analyze MAPE variance across prediction iterations"""
    # Group by prediction_model and prediction_iteration
    df_stats = df.groupby(['prediction_model', 'prediction_iteration'])['mape'].mean().reset_index()
    
    if df_stats['prediction_iteration'].nunique() <= 1:
        st.info("Only one iteration available - no variance to analyze")
        return
    
    # Create line plot
    fig = px.line(
        df_stats,
        x='prediction_iteration',
        y='mape',
        color='prediction_model',
        markers=True,
        title='MAPE by Prediction Iteration per Model',
        labels={'mape': 'MAPE (%)', 'prediction_iteration': 'Iteration', 'prediction_model': 'Model'}
    )
    
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)
    
    # Show variance statistics
    st.subheader("Iteration Variance by Model")
    variance_stats = df.groupby('prediction_model')['mape'].agg(['mean', 'std', 'min', 'max']).reset_index()
    variance_stats['cv'] = (variance_stats['std'] / variance_stats['mean'] * 100).round(2)  # Coefficient of variation
    variance_stats = variance_stats.sort_values('cv', ascending=False)
    variance_stats.columns = ['Model', 'Mean MAPE', 'Std', 'Min', 'Max', 'CV (%)']
    
    st.dataframe(variance_stats.style.format({
        'Mean MAPE': '{:.2f}%',
        'Std': '{:.2f}%',
        'Min': '{:.2f}%',
        'Max': '{:.2f}%',
        'CV (%)': '{:.2f}%'
    }), use_container_width=True)
    
    st.caption("CV (Coefficient of Variation) = Std/Mean × 100% - higher values indicate more variability")


def main():
    st.set_page_config(
        page_title="Time Series Prediction Visualization",
        page_icon="🔮",
        layout="wide"
    )
    
    st.title("🔮 Time Series Prediction Results Visualization")
    st.markdown("---")
    
    # Sidebar for file selection
    st.sidebar.header("Settings")
    
    # Get available result files
    available_files = get_available_results()
    
    if not available_files:
        st.error("No result files found in `prediction_experiment_results/` directory.")
        st.info("Run `python 8_calculate_prediction_error.py` first to generate results.")
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
    
    # Display file info
    st.sidebar.success(f"✓ Loaded {len(df)} records")
    st.sidebar.info(f"File: {selected_file_name}")
    
    # Main filters
    st.sidebar.header("Filters")
    st.sidebar.info("🌍 **Global filters** - apply to all tabs")
    
    # Get unique values
    all_datasets = sorted(df['dataset_name'].unique().tolist())
    all_pred_models = sorted(df['prediction_model'].unique().tolist())
    all_source_types = sorted(df['source_type'].unique().tolist())
    
    # Get techniques and rates (may have NaN for original data)
    all_techniques = sorted([t for t in df['technique'].dropna().unique().tolist()])
    all_rates = sorted([r for r in df['rate_percent'].dropna().unique().tolist()])
    all_recon_models = sorted([m for m in df['reconstruction_model'].dropna().unique().tolist()])
    
    selected_datasets = st.sidebar.multiselect("Dataset", all_datasets, default=all_datasets)
    selected_pred_models = st.sidebar.multiselect("Prediction Model", all_pred_models, default=all_pred_models)
    selected_source_types = st.sidebar.multiselect("Source Type", all_source_types, default=all_source_types)
    
    # Apply filters to dataframe
    df_filtered = df.copy()
    if selected_datasets:
        df_filtered = df_filtered[df_filtered['dataset_name'].isin(selected_datasets)]
    if selected_pred_models:
        df_filtered = df_filtered[df_filtered['prediction_model'].isin(selected_pred_models)]
    if selected_source_types:
        df_filtered = df_filtered[df_filtered['source_type'].isin(selected_source_types)]
    
    # Filter out NaN MAPE values
    df_filtered = df_filtered[df_filtered['mape'].notna()]
    
    # Display overview metrics
    st.header("📈 Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Records", len(df_filtered))
    with col2:
        st.metric("Mean MAPE", f"{df_filtered['mape'].mean():.2f}%")
    with col3:
        st.metric("Median MAPE", f"{df_filtered['mape'].median():.2f}%")
    with col4:
        st.metric("Best MAPE", f"{df_filtered['mape'].min():.2f}%")
    with col5:
        st.metric("Worst MAPE", f"{df_filtered['mape'].max():.2f}%")
    
    st.markdown("---")
    
    # Visualization tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11, tab12, tab13, tab14 = st.tabs([
        "📊 By Pred Model", 
        "🎯 By Source Type",
        "🔧 By Recon Model",
        "📉 By Technique", 
        "📈 By Missing Rate",
        "📁 By Dataset",
        "⚡ By Efficiency",
        "🔥 Heatmaps",
        "📊 Statistical Tests",
        "🏆 Best/Worst",
        "🔄 Iteration Analysis",
        "⏱️ Computation Time",
        "💻 Resource Usage",
        "📋 Raw Data"
    ])
    
    with tab1:
        st.header("Comparison by Prediction Model")
        st.caption("📍 Local filters - apply only to this tab")
        
        # Sub-filters
        col1, col2 = st.columns(2)
        with col1:
            filter_source = st.selectbox(
                "Filter by Source Type",
                ['All'] + all_source_types,
                key='tab1_source'
            )
        with col2:
            filter_technique = st.selectbox(
                "Filter by Technique",
                ['All'] + all_techniques,
                key='tab1_technique'
            )
        
        plot_mape_by_model(
            df_filtered,
            source_type=None if filter_source == 'All' else filter_source,
            technique=None if filter_technique == 'All' else filter_technique
        )
    
    with tab2:
        st.header("Comparison by Source Type")
        st.caption("Original (clean training data) vs Reconstructed (fixed missing data)")
        
        filter_model = st.selectbox(
            "Filter by Prediction Model",
            ['All'] + all_pred_models,
            key='tab2_model'
        )
        
        plot_mape_by_source_type(
            df_filtered,
            model=None if filter_model == 'All' else filter_model
        )
    
    with tab3:
        st.header("Comparison by Reconstruction Model")
        st.caption("Which reconstruction methods produce best data for prediction?")
        
        col1, col2 = st.columns(2)
        with col1:
            filter_pred = st.selectbox(
                "Filter by Prediction Model",
                ['All'] + all_pred_models,
                key='tab3_pred'
            )
        with col2:
            filter_technique = st.selectbox(
                "Filter by Technique",
                ['All'] + all_techniques,
                key='tab3_technique'
            )
        
        plot_mape_by_reconstruction_model(
            df_filtered,
            pred_model=None if filter_pred == 'All' else filter_pred,
            technique=None if filter_technique == 'All' else filter_technique
        )
    
    with tab4:
        st.header("Comparison by Missingness Technique")
        st.caption("📍 Only for reconstructed data")
        
        col1, col2 = st.columns(2)
        with col1:
            filter_model = st.selectbox(
                "Filter by Prediction Model",
                ['All'] + all_pred_models,
                key='tab4_model'
            )
        with col2:
            filter_rate = st.selectbox(
                "Filter by Missing Rate (%)",
                ['All'] + [int(r) for r in all_rates],
                key='tab4_rate'
            )
        
        plot_mape_by_technique(
            df_filtered,
            model=None if filter_model == 'All' else filter_model,
            rate=None if filter_rate == 'All' else filter_rate
        )
    
    with tab5:
        st.header("Comparison by Missing Rate")
        st.caption("📍 Only for reconstructed data")
        
        col1, col2 = st.columns(2)
        with col1:
            filter_model = st.selectbox(
                "Filter by Prediction Model",
                ['All'] + all_pred_models,
                key='tab5_model'
            )
        with col2:
            filter_technique = st.selectbox(
                "Filter by Technique",
                ['All'] + all_techniques,
                key='tab5_technique'
            )
        
        plot_mape_by_rate(
            df_filtered,
            model=None if filter_model == 'All' else filter_model,
            technique=None if filter_technique == 'All' else filter_technique
        )
    
    with tab6:
        st.header("Comparison by Dataset")
        
        if len(df_filtered) > 0:
            plot_dataset_comparison(df_filtered)
        else:
            st.warning("No data available with current filters")
    
    with tab7:
        st.header("⚡ Resource Efficiency Analysis")
        st.caption("Lower values = more efficient (less time and resources)")
        
        # Check if performance metrics are available
        if 'cpu_cores_used' not in df_filtered.columns or df_filtered['cpu_cores_used'].isna().all():
            st.warning("⚠️ No performance metrics available. Run predictions first to collect performance data.")
        else:
            df_perf = df_filtered[df_filtered['cpu_cores_used'].notna()].copy()
            
            if len(df_perf) == 0:
                st.warning("No performance data available with current filters")
            else:
                # Explanation of efficiency score calculation
                with st.expander("ℹ️ How is the Efficiency Score calculated?", expanded=False):
                    st.markdown("""
                    The **Efficiency Score** combines four normalized metrics to provide an overall computational efficiency rating:
                    
                    **Formula:**
                    ```
                    Efficiency Score = Time_norm + CPU_norm + Memory_norm + GPU_norm
                    ```
                    
                    **Components:**
                    - **Time_norm**: Normalized execution time (0 to 1 scale)
                        - `(time - min_time) / (max_time - min_time)`
                    - **CPU_norm**: CPU cores utilized relative to total available cores
                        - `cpu_cores_used / cpu_cores_total`
                    - **Memory_norm**: Normalized RAM usage (0 to 1 scale)
                        - `(memory - min_memory) / (max_memory - min_memory)`
                    - **GPU_norm**: GPU memory utilized relative to total GPU memory (0 to 1 scale)
                        - `gpu_memory_used / gpu_memory_total` (0 for CPU-only models)
                    
                    **Interpretation:**
                    - **Lower score** = more efficient (faster execution, less CPU/RAM/GPU usage)
                    - **Higher score** = less efficient (slower, more resource-intensive)
                    - Score typically ranges from ~0 (most efficient) to ~4 (least efficient)
                    - GPU-based models typically have higher scores due to GPU memory usage
                    
                    **Use Cases:**
                    - Select models for resource-constrained environments
                    - Balance prediction quality (MAPE) vs. computational cost
                    - Compare CPU-based vs. GPU-based models fairly
                    """)
                
                st.divider()
                
                # Create efficiency score: normalize time, CPU, RAM, and GPU
                df_perf_copy = df_perf.copy()
                df_perf_copy['time_norm'] = (df_perf_copy['time_seconds'] - df_perf_copy['time_seconds'].min()) / (df_perf_copy['time_seconds'].max() - df_perf_copy['time_seconds'].min() + 0.001)
                # Normalize CPU cores (divide by total available cores)
                total_cores = df_perf_copy['cpu_cores_total'].mode()[0] if 'cpu_cores_total' in df_perf_copy.columns and df_perf_copy['cpu_cores_total'].notna().any() else 1
                df_perf_copy['cpu_norm'] = df_perf_copy['cpu_cores_used'] / total_cores
                df_perf_copy['mem_norm'] = (df_perf_copy['memory_mb'] - df_perf_copy['memory_mb'].min()) / (df_perf_copy['memory_mb'].max() - df_perf_copy['memory_mb'].min() + 0.001)
                
                # Normalize GPU memory (if available)
                if 'gpu_memory_mb' in df_perf_copy.columns and 'gpu_memory_total_mb' in df_perf_copy.columns:
                    df_perf_copy['gpu_norm'] = df_perf_copy.apply(
                        lambda row: (row['gpu_memory_mb'] / row['gpu_memory_total_mb']) 
                        if pd.notna(row['gpu_memory_mb']) and pd.notna(row['gpu_memory_total_mb']) and row['gpu_memory_total_mb'] > 0
                        else 0.0,
                        axis=1
                    )
                else:
                    df_perf_copy['gpu_norm'] = 0.0
                
                # Combined efficiency score: lower = better
                df_perf_copy['efficiency_score'] = df_perf_copy['time_norm'] + df_perf_copy['cpu_norm'] + df_perf_copy['mem_norm'] + df_perf_copy['gpu_norm']
                
                efficiency = df_perf_copy.groupby('prediction_model')['efficiency_score'].mean().reset_index()
                efficiency = efficiency.sort_values('efficiency_score', ascending=False)
                
                # Overall efficiency score by model
                st.subheader("Overall Efficiency Score by Prediction Model")
                st.caption("Models sorted by efficiency (best to worst)")
                
                fig = px.bar(
                    efficiency,
                    x='efficiency_score',
                    y='prediction_model',
                    orientation='h',
                    title="Overall Efficiency Score by Model (lower = better)",
                    labels={'efficiency_score': 'Efficiency Score', 'prediction_model': 'Model'},
                    color='efficiency_score',
                    color_continuous_scale='RdYlGn_r'
                )
                fig.update_layout(height=max(400, len(efficiency) * 30), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Combined scatter plot
                st.subheader("Time vs Memory Usage")
                st.caption("Bubble size represents combined CPU + GPU utilization")
                
                # Aggregate metrics including GPU
                agg_cols = {
                    'time_seconds': 'mean',
                    'memory_mb': 'mean',
                    'cpu_cores_used': 'mean',
                }
                if 'gpu_memory_mb' in df_perf.columns:
                    agg_cols['gpu_memory_mb'] = 'mean'
                if 'gpu_memory_total_mb' in df_perf.columns:
                    agg_cols['gpu_memory_total_mb'] = 'mean'
                
                model_summary = df_perf.groupby('prediction_model').agg(agg_cols).reset_index()
                
                # Calculate combined CPU + GPU metric for bubble size
                if 'gpu_memory_mb' in model_summary.columns and 'gpu_memory_total_mb' in model_summary.columns:
                    model_summary['gpu_normalized'] = model_summary.apply(
                        lambda row: (row['gpu_memory_mb'] / row['gpu_memory_total_mb']) * 4.0
                        if pd.notna(row['gpu_memory_mb']) and pd.notna(row['gpu_memory_total_mb']) and row['gpu_memory_total_mb'] > 0
                        else 0.0,
                        axis=1
                    )
                else:
                    model_summary['gpu_normalized'] = 0.0
                
                model_summary['combined_compute'] = model_summary['cpu_cores_used'] + model_summary['gpu_normalized']
                
                # Determine if model uses GPU for coloring
                model_summary['compute_type'] = model_summary['gpu_normalized'].apply(
                    lambda x: 'CPU+GPU' if x > 0 else 'CPU only'
                )
                
                fig = px.scatter(
                    model_summary,
                    x='time_seconds',
                    y='memory_mb',
                    size='combined_compute',
                    text='prediction_model',
                    title="Time vs Memory (bubble size = CPU + GPU utilization)",
                    labels={'time_seconds': 'Avg Time (seconds)', 'memory_mb': 'Avg Memory (MB)'},
                    color='compute_type',
                    color_discrete_map={'CPU only': '#3498db', 'CPU+GPU': '#e74c3c'}
                )
                fig.update_traces(textposition='top center')
                st.plotly_chart(fig, use_container_width=True)
    
    with tab8:
        st.header("Heatmaps")
        
        heatmap_type = st.radio(
            "Select heatmap type",
            ["Prediction Model vs Reconstruction Model", "Prediction Model vs Technique"],
            horizontal=True
        )
        
        if heatmap_type == "Prediction Model vs Reconstruction Model":
            # Get available reconstruction models for sorting
            if len(df_filtered) > 0:
                recon_models = ['Alphabetical'] + all_recon_models
            else:
                recon_models = ['Alphabetical']
            
            sort_choice = st.selectbox(
                "Sort prediction models by",
                recon_models,
                help="Sort prediction models by MAPE on selected reconstruction model"
            )
            
            if len(df_filtered) > 0:
                sort_by = None if sort_choice == 'Alphabetical' else sort_choice
                plot_heatmap_pred_vs_recon(df_filtered, sort_by_model=sort_by)
            else:
                st.warning("No data available")
        else:
            # Get available techniques for sorting
            if len(df_filtered) > 0:
                techniques = ['Alphabetical'] + all_techniques
            else:
                techniques = ['Alphabetical']
            
            sort_choice = st.selectbox(
                "Sort prediction models by",
                techniques,
                help="Sort prediction models by MAPE on selected technique"
            )
            
            if len(df_filtered) > 0:
                sort_by = None if sort_choice == 'Alphabetical' else sort_choice
                plot_heatmap_pred_vs_technique(df_filtered, sort_by_technique=sort_by)
            else:
                st.warning("No data available")
    
    with tab9:
        st.header("📊 Statistical Significance Tests")
        st.caption("Pairwise t-tests between prediction models - which differences are statistically significant?")
        
        if len(df_filtered) == 0:
            st.warning("No data available with current filters")
        else:
            # Import statistical test functions
            from utils.statistical_tests import (
                perform_pairwise_ttests, 
                get_pairwise_pvalues,
                get_model_statistics,
                get_significance_summary
            )
            
            # Info box
            with st.expander("ℹ️ How to interpret this analysis", expanded=False):
                st.markdown("""
                **Statistical Significance Testing**:
                
                This tab performs **pairwise t-tests** between all prediction models to determine if performance differences are statistically significant.
                
                **Legend**:
                - **🟩 +2 (p<0.01)**: Row model is **significantly better** than column model (highly significant)
                - **🟢 +1 (p<0.05)**: Row model is **significantly better** than column model
                - **⬜ 0**: No significant difference
                - **🔴 -1 (p<0.05)**: Row model is **significantly worse** than column model
                - **🟥 -2 (p<0.01)**: Row model is **significantly worse** than column model (highly significant)
                
                **Note**: Lower MAPE = better performance.
                """)
            
            st.divider()
            
            # Rename column for statistical functions
            df_for_stats = df_filtered.rename(columns={'prediction_model': 'model'})
            
            # Calculate statistics
            st.subheader("Model Performance Statistics")
            model_stats = get_model_statistics(df_for_stats, metric='mape')
            st.dataframe(
                model_stats.style.background_gradient(subset=['mean'], cmap='RdYlGn_r'),
                use_container_width=True
            )
            
            st.divider()
            
            # Perform pairwise t-tests
            st.subheader("Pairwise Statistical Significance Matrix")
            significance_matrix = perform_pairwise_ttests(df_for_stats, metric='mape', alpha_01=0.01, alpha_05=0.05)
            
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
            st.dataframe(styled_matrix, use_container_width=True, height=600)
            
            st.caption("""
            **Legend**: 🟩 +2 (p<0.01 better) | 🟢 +1 (p<0.05 better) | ⬜ 0 (no diff) | 🔴 -1 (p<0.05 worse) | 🟥 -2 (p<0.01 worse)
            """)
            
            st.divider()
            
            # Summary
            st.subheader("Significance Summary by Model")
            significance_summary = get_significance_summary(significance_matrix)
            summary_df = pd.DataFrame(significance_summary).T
            summary_df = summary_df.reset_index()
            summary_df.columns = ['Model', 'Better (p<0.01)', 'Better (p<0.05)', 'No Difference', 'Worse (p<0.05)', 'Worse (p<0.01)']
            summary_df = summary_df.sort_values('Better (p<0.01)', ascending=False)
            st.dataframe(summary_df, use_container_width=True)
    
    with tab10:
        st.header("Best and Worst Performing Prediction Models")
        
        top_n = st.slider("Number of models to show", 3, 15, min(10, len(all_pred_models)))
        
        if len(df_filtered) > 0:
            plot_best_worst_models(df_filtered, top_n=top_n)
        else:
            st.warning("No data available with current filters")
    
    with tab11:
        st.header("🔄 Iteration Analysis")
        st.caption("Analyze variance across multiple training iterations (for non-deterministic models)")
        
        if len(df_filtered) > 0:
            plot_iteration_analysis(df_filtered)
        else:
            st.warning("No data available with current filters")
    
    with tab12:
        st.header("⏱️ Computation Time Analysis")
        st.caption("📍 Computational complexity metrics - execution time")
        
        if 'time_seconds' not in df_filtered.columns or df_filtered['time_seconds'].isna().all():
            st.warning("⚠️ No performance metrics available in this results file.")
            st.info("Run `7_predict_datasets.py` again to collect performance metrics, then `8_calculate_prediction_error.py` to merge them.")
        else:
            df_perf = df_filtered[df_filtered['time_seconds'].notna()].copy()
            
            if df_perf.empty:
                st.warning("❌ No performance data after filtering")
            else:
                st.success(f"✅ Showing {len(df_perf)} records with performance metrics")
                
                # Summary statistics
                st.subheader("⏱️ Execution Time Summary")
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Total Time", f"{df_perf['time_seconds'].sum():.1f}s")
                with col2:
                    st.metric("Average Time", f"{df_perf['time_seconds'].mean():.2f}s")
                with col3:
                    st.metric("Fastest", f"{df_perf['time_seconds'].min():.2f}s")
                with col4:
                    st.metric("Slowest", f"{df_perf['time_seconds'].max():.2f}s")
                
                st.markdown("---")
                
                # Time by prediction model
                st.subheader("Execution Time by Prediction Model")
                model_time = df_perf.groupby('prediction_model')['time_seconds'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
                model_time = model_time.sort_values('mean', ascending=False)
                
                fig = px.bar(
                    model_time,
                    x='mean',
                    y='prediction_model',
                    orientation='h',
                    error_x='std',
                    title="Average Execution Time by Prediction Model (with std dev)",
                    labels={'mean': 'Average Time (seconds)', 'prediction_model': 'Model'},
                    color='mean',
                    color_continuous_scale='Reds'
                )
                fig.update_layout(height=max(400, len(model_time) * 30), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Time comparison: boxplot
                st.subheader("Time Distribution by Model")
                fig = px.box(
                    df_perf,
                    x='prediction_model',
                    y='time_seconds',
                    title="Execution Time Distribution",
                    labels={'time_seconds': 'Time (seconds)', 'prediction_model': 'Model'},
                    color='prediction_model'
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
                
                # Time by source type and reconstruction model
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Average Time by Source Type")
                    source_time = df_perf.groupby('source_type')['time_seconds'].mean().reset_index()
                    fig = px.bar(
                        source_time,
                        x='source_type',
                        y='time_seconds',
                        title="Average Time by Source Type",
                        labels={'time_seconds': 'Avg Time (seconds)', 'source_type': 'Source'},
                        color='time_seconds',
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    # Time by reconstruction model (only for reconstructed data)
                    df_recon = df_perf[df_perf['source_type'] == 'reconstructed']
                    if len(df_recon) > 0:
                        st.subheader("Average Time by Reconstruction Model")
                        recon_time = df_recon.groupby('reconstruction_model')['time_seconds'].mean().reset_index()
                        recon_time = recon_time.sort_values('time_seconds', ascending=False)
                        fig = px.bar(
                            recon_time,
                            x='reconstruction_model',
                            y='time_seconds',
                            title="Average Time by Reconstruction Model",
                            labels={'time_seconds': 'Avg Time (seconds)', 'reconstruction_model': 'Model'},
                            color='time_seconds',
                            color_continuous_scale='Greens'
                        )
                        fig.update_xaxes(tickangle=45)
                        st.plotly_chart(fig, use_container_width=True)
                
                # Detailed table
                st.subheader("Detailed Statistics")
                st.dataframe(model_time, use_container_width=True)
    
    with tab13:
        st.header("💻 Resource Usage Analysis")
        st.caption("📍 Computational complexity metrics - CPU, RAM, GPU usage")
        
        # Check if performance metrics are available in the data
        if 'cpu_cores_used' not in df_filtered.columns or df_filtered['cpu_cores_used'].isna().all():
            st.warning("⚠️ No performance metrics available in this results file.")
            st.info("Run `7_predict_datasets.py` again to collect performance metrics, then `8_calculate_prediction_error.py` to merge them.")
        else:
            df_perf = df_filtered[df_filtered['cpu_cores_used'].notna()].copy()
            
            if df_perf.empty:
                st.warning("❌ No performance data after filtering")
            else:
                st.success(f"✅ Showing {len(df_perf)} records with performance metrics")
                
                # Summary statistics
                st.subheader("💻 Resource Usage Summary")
                col1, col2, col3 = st.columns(3)
                with col1:
                    avg_cores = df_perf['cpu_cores_used'].mean()
                    total_cores = df_perf['cpu_cores_total'].mode()[0] if 'cpu_cores_total' in df_perf.columns and df_perf['cpu_cores_total'].notna().any() else 0
                    st.metric("Avg CPU Cores", f"{avg_cores:.2f} / {total_cores:.0f}")
                with col2:
                    if 'memory_total_mb' in df_perf.columns and df_perf['memory_total_mb'].notna().any():
                        avg_memory = df_perf['memory_mb'].mean()
                        total_memory = df_perf['memory_total_mb'].iloc[0]
                        st.metric("Avg RAM Usage", f"{avg_memory:.1f} / {total_memory:.0f} MB")
                    else:
                        st.metric("Avg RAM Usage", f"{df_perf['memory_mb'].mean():.1f} MB")
                with col3:
                    if 'gpu_percent' in df_perf.columns and df_perf['gpu_percent'].notna().any():
                        st.metric("Avg GPU Usage", f"{df_perf['gpu_percent'].mean():.1f}%")
                    else:
                        st.metric("GPU Usage", "N/A")
                
                st.markdown("---")
                
                # CPU cores usage by model
                st.subheader("CPU+GPU Cores Utilized by Model")
                model_cpu = df_perf.groupby('prediction_model')['cpu_cores_used'].agg(['mean', 'std', 'max']).reset_index()
                model_cpu = model_cpu.sort_values('mean', ascending=False)
                
                fig = px.bar(
                    model_cpu,
                    x='mean',
                    y='prediction_model',
                    orientation='h',
                    title="Average CPU+GPU Cores Utilized by Prediction Model",
                    labels={'mean': 'Avg CPU+GPU Cores', 'prediction_model': 'Model'},
                    color='mean',
                    color_continuous_scale='Oranges'
                )
                fig.update_layout(height=max(400, len(model_cpu) * 30), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
                
                # Combined CPU + GPU utilization for comparison
                if 'gpu_percent' in df_perf.columns and df_perf['gpu_percent'].notna().any():
                    # Prepare data: for each model show CPU and GPU side by side
                    model_compute = df_perf.groupby('prediction_model').agg({
                        'cpu_cores_used': 'mean',
                        'gpu_percent': 'mean'
                    }).reset_index()
                    
                    # Fill NaN GPU values with 0 for CPU-only models
                    model_compute['gpu_percent'] = model_compute['gpu_percent'].fillna(0)
                    
                    # Create grouped bar chart
                    model_compute_melted = model_compute.melt(
                        id_vars=['prediction_model'],
                        value_vars=['cpu_cores_used', 'gpu_percent'],
                        var_name='Resource Type',
                        value_name='Usage'
                    )
                    
                    # Rename for better display
                    model_compute_melted['Resource Type'] = model_compute_melted['Resource Type'].map({
                        'cpu_cores_used': 'CPU Cores',
                        'gpu_percent': 'GPU Utilization (%)'
                    })
                    
                    # Sort by combined usage (CPU + normalized GPU)
                    model_compute['combined'] = model_compute['cpu_cores_used'] + (model_compute['gpu_percent'] / 100.0) * 4
                    model_compute = model_compute.sort_values('combined', ascending=True)
                    model_order = model_compute['prediction_model'].tolist()
                    
                    fig = px.bar(
                        model_compute_melted,
                        x='Usage',
                        y='prediction_model',
                        color='Resource Type',
                        orientation='h',
                        title="CPU vs GPU Utilization by Model",
                        labels={'Usage': 'Utilization', 'prediction_model': 'Model'},
                        barmode='group',
                        category_orders={'prediction_model': model_order},
                        color_discrete_map={'CPU Cores': '#FFA500', 'GPU Utilization (%)': '#4CAF50'}
                    )
                    fig.update_layout(height=max(400, len(model_compute) * 40))
                    st.plotly_chart(fig, use_container_width=True)
                
                # RAM usage by model
                st.subheader("Memory (RAM) Usage by Model")
                model_mem = df_perf.groupby('prediction_model')['memory_mb'].agg(['mean', 'std', 'max']).reset_index()
                model_mem = model_mem.sort_values('mean', ascending=False)
                
                fig = px.bar(
                    model_mem,
                    x='mean',
                    y='prediction_model',
                    orientation='h',
                    title="Average Memory Usage by Prediction Model",
                    labels={'mean': 'Avg Memory (MB)', 'prediction_model': 'Model'},
                    color='mean',
                    color_continuous_scale='Purples'
                )
                fig.update_layout(height=max(400, len(model_mem) * 30), showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
    
    with tab14:
        st.header("Raw Data")
        
        # Search functionality
        search_term = st.text_input("Search in data", "")
        
        if search_term:
            mask = df_filtered.astype(str).apply(lambda x: x.str.contains(search_term, case=False)).any(axis=1)
            df_display = df_filtered[mask]
        else:
            df_display = df_filtered
        
        # Display options
        col1, col2 = st.columns(2)
        with col1:
            sort_column = st.selectbox("Sort by", df_display.columns.tolist())
        with col2:
            sort_order = st.radio("Order", ['Ascending', 'Descending'])
        
        df_display = df_display.sort_values(
            sort_column,
            ascending=(sort_order == 'Ascending')
        )
        
        st.dataframe(df_display, use_container_width=True)
        
        # Download button
        csv = df_display.to_csv(index=False)
        st.download_button(
            label="📥 Download filtered data as CSV",
            data=csv,
            file_name=f"filtered_{selected_file_name}",
            mime="text/csv"
        )


if __name__ == "__main__":
    main()
