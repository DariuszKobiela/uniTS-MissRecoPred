#!/usr/bin/env python3
"""
Streamlit Visualization App for Reconstruction & Prediction Results
Interactive dashboard for comparing reconstruction models, techniques, and missing rates.
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
    """Get list of available result files"""
    results_dir = Path("reconstruction_experiments_results")
    if not results_dir.exists():
        return []
    
    return sorted(results_dir.glob("*.csv"), reverse=True)


# Note: Performance metrics are now included in reconstruction_results_*.csv files
# No need for separate performance_metrics_*.csv files


def plot_mad_by_model(df: pd.DataFrame, technique: str = None, rate: int = None):
    """Plot MAD comparison by reconstruction model"""
    df_filtered = df.copy()
    
    # Apply filters
    if technique:
        df_filtered = df_filtered[df_filtered['technique'] == technique]
    if rate:
        df_filtered = df_filtered[df_filtered['rate_percent'] == rate]
    
    if df_filtered.empty:
        st.warning("No data available for selected filters")
        return
    
    # Group by model and calculate statistics
    df_stats = df_filtered.groupby('model')['mad'].agg(['mean', 'std', 'min', 'max']).reset_index()
    df_stats = df_stats.sort_values('mean')
    
    # Create bar plot
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_stats['model'],
        y=df_stats['mean'],
        error_y=dict(type='data', array=df_stats['std']),
        marker_color='lightblue',
        name='Mean MAD'
    ))
    
    fig.update_layout(
        title='Mean Absolute Difference by Reconstruction Model',
        xaxis_title='Reconstruction Model',
        yaxis_title='MAD',
        xaxis_tickangle=-45,
        height=500
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # Show statistics table
    st.subheader("Statistics")
    st.dataframe(df_stats.style.format({
        'mean': '{:.4f}',
        'std': '{:.4f}',
        'min': '{:.4f}',
        'max': '{:.4f}'
    }), width='stretch')


def plot_mad_by_technique(df: pd.DataFrame, model: str = None, rate: int = None):
    """Plot MAD comparison by missingness technique"""
    df_filtered = df.copy()
    
    # Apply filters
    if model:
        df_filtered = df_filtered[df_filtered['model'] == model]
    if rate:
        df_filtered = df_filtered[df_filtered['rate_percent'] == rate]
    
    if df_filtered.empty:
        st.warning("No data available for selected filters")
        return
    
    # Group by technique and calculate statistics
    df_stats = df_filtered.groupby('technique')['mad'].agg(['mean', 'std', 'min', 'max']).reset_index()
    df_stats = df_stats.sort_values('mean')
    
    # Create bar plot
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_stats['technique'],
        y=df_stats['mean'],
        error_y=dict(type='data', array=df_stats['std']),
        marker_color='lightgreen',
        name='Mean MAD'
    ))
    
    fig.update_layout(
        title='Mean Absolute Difference by Missingness Technique',
        xaxis_title='Missingness Technique',
        yaxis_title='MAD',
        height=500
    )
    
    st.plotly_chart(fig, width='stretch')
    
    # Show statistics table
    st.subheader("Statistics")
    st.dataframe(df_stats.style.format({
        'mean': '{:.4f}',
        'std': '{:.4f}',
        'min': '{:.4f}',
        'max': '{:.4f}'
    }), width='stretch')


def plot_mad_by_rate(df: pd.DataFrame, model: str = None, technique: str = None):
    """Plot MAD comparison by missing rate"""
    df_filtered = df.copy()
    
    # Apply filters
    if model:
        df_filtered = df_filtered[df_filtered['model'] == model]
    if technique:
        df_filtered = df_filtered[df_filtered['technique'] == technique]
    
    if df_filtered.empty:
        st.warning("No data available for selected filters")
        return
    
    # Group by rate and calculate statistics
    df_stats = df_filtered.groupby('rate_percent')['mad'].agg(['mean', 'std', 'min', 'max']).reset_index()
    df_stats = df_stats.sort_values('rate_percent')
    
    # Create line plot
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=df_stats['rate_percent'],
        y=df_stats['mean'],
        mode='lines+markers',
        error_y=dict(type='data', array=df_stats['std']),
        marker=dict(size=10, color='coral'),
        line=dict(width=2),
        name='Mean MAD'
    ))
    
    fig.update_layout(
        title='Mean Absolute Difference by Missing Rate',
        xaxis_title='Missing Rate (%)',
        yaxis_title='MAD',
        height=500
    )
    
    st.plotly_chart(fig, width='stretch')
        
    # Show statistics table
    st.subheader("Statistics")
    df_stats_display = df_stats.copy()
    df_stats_display['rate_percent'] = df_stats_display['rate_percent'].astype(str) + '%'
    st.dataframe(df_stats_display.style.format({
        'mean': '{:.4f}',
        'std': '{:.4f}',
        'min': '{:.4f}',
        'max': '{:.4f}'
    }), width='stretch')


def plot_heatmap(df: pd.DataFrame, metric: str = 'mad', sort_by_technique: str = None):
    """Plot heatmap of MAD for model vs technique
    
    Args:
        df: DataFrame with results
        metric: Metric to display (always 'mad')
        sort_by_technique: Technique name to sort models by, or None for alphabetical
    """
    # Calculate mean MAD for each model-technique combination
    pivot_data = df.pivot_table(
        values='mad',
        index='model',
        columns='technique',
        aggfunc='mean'
    )
    
    # Sort models by selected technique or alphabetically
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
        text=np.round(pivot_data.values, 4),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title='MAD')
    ))
    
    fig.update_layout(
        title=f'Heatmap: MAD by Model and Technique{sort_info}',
        xaxis_title='Missingness Technique',
        yaxis_title='Reconstruction Model',
        height=max(500, len(pivot_data.index) * 30)
    )
    
    st.plotly_chart(fig, width='stretch')


def plot_dataset_comparison(df: pd.DataFrame):
    """Compare MAD across different datasets"""
    df_stats = df.groupby('dataset_name')['mad'].agg(['mean', 'std']).reset_index()
    df_stats = df_stats.sort_values('mean')
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        x=df_stats['dataset_name'],
        y=df_stats['mean'],
        error_y=dict(type='data', array=df_stats['std']),
        marker_color='mediumpurple',
        name='Mean MAD'
    ))
    
    fig.update_layout(
        title='Mean Absolute Difference by Dataset',
        xaxis_title='Dataset',
        yaxis_title='MAD',
        height=500
    )
    
    st.plotly_chart(fig, width='stretch')


def plot_best_worst_models(df: pd.DataFrame, top_n: int = 10):
    """Show best and worst performing models"""
    df_stats = df.groupby('model')['mad'].mean().reset_index()
    df_stats = df_stats.sort_values('mad')
    
    # Calculate global MAD range for consistent axis scaling
    global_min = df_stats['mad'].min()
    global_max = df_stats['mad'].max()
    # Add 5% padding for better visualization
    axis_range = [global_min * 0.95, global_max * 1.05]
    
    # Best models (lowest MAD) - reverse order for display (best on top)
    best_models = df_stats.head(top_n).iloc[::-1]
    
    # Worst models (highest MAD) - reverse order for display (worst on top)
    worst_models = df_stats.tail(top_n)
    
    # Create subplots
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(f'Top {top_n} Best Models (Lowest MAD)', 
                       f'Top {top_n} Worst Models (Highest MAD)')
    )
    
    # Best models
    fig.add_trace(
        go.Bar(x=best_models['mad'], y=best_models['model'], 
               orientation='h', marker_color='green', name='Best'),
        row=1, col=1
    )
    
    # Worst models
    fig.add_trace(
        go.Bar(x=worst_models['mad'], y=worst_models['model'], 
               orientation='h', marker_color='red', name='Worst'),
        row=1, col=2
    )
    
    fig.update_layout(height=max(500, top_n * 40), showlegend=False)
    # Set the same x-axis range for both subplots
    fig.update_xaxes(title_text="MAD", range=axis_range, row=1, col=1)
    fig.update_xaxes(title_text="MAD", range=axis_range, row=1, col=2)
    
    st.plotly_chart(fig, width='stretch')


def main():
    st.set_page_config(
        page_title="Time Series Reconstruction Visualization",
        page_icon="📊",
        layout="wide"
    )
    
    st.title("📊 Time Series Reconstruction Results Visualization")
    st.markdown("---")
    
    # Sidebar for file selection
    st.sidebar.header("Settings")
    
    # Get available result files
    available_files = get_available_results()
    
    if not available_files:
        st.error("No result files found in `reconstruction_experiments_results/` directory.")
        st.info("Run `python 5_calculate_mad.py` first to generate results.")
        return
    
    # File selection
    file_names = [f.name for f in available_files]
    selected_file_name = st.sidebar.selectbox(
        "Select Results File",
        file_names,
        help="Choose a results file to visualize"
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
    all_models = sorted(df['model'].unique().tolist())
    all_techniques = sorted(df['technique'].unique().tolist())
    all_rates = sorted(df['rate_percent'].unique().tolist())
    
    selected_datasets = st.sidebar.multiselect("Dataset", all_datasets, default=all_datasets)
    selected_models = st.sidebar.multiselect("Model", all_models, default=all_models)
    selected_techniques = st.sidebar.multiselect("Technique", all_techniques, default=all_techniques)
    selected_rates = st.sidebar.multiselect("Missing Rate (%)", all_rates, default=all_rates)
    
    # Apply filters to dataframe
    df_filtered = df.copy()
    if selected_datasets:
        df_filtered = df_filtered[df_filtered['dataset_name'].isin(selected_datasets)]
    if selected_models:
        df_filtered = df_filtered[df_filtered['model'].isin(selected_models)]
    if selected_techniques:
        df_filtered = df_filtered[df_filtered['technique'].isin(selected_techniques)]
    if selected_rates:
        df_filtered = df_filtered[df_filtered['rate_percent'].isin(selected_rates)]
    
    # Display overview metrics
    st.header("📈 Overview")
    
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Records", len(df_filtered))
    with col2:
        st.metric("Mean MAD", f"{df_filtered['mad'].mean():.4f}")
    with col3:
        st.metric("Median MAD", f"{df_filtered['mad'].median():.4f}")
    with col4:
        st.metric("Best MAD", f"{df_filtered['mad'].min():.4f}")
    with col5:
        st.metric("Worst MAD", f"{df_filtered['mad'].max():.4f}")
    
    st.markdown("---")
    
    # Visualization tabs
    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs([
        "📊 By Model", 
        "🎯 By Technique", 
        "📉 By Missing Rate",
        "📁 By Dataset",
        "⚡ By Efficiency",
        "🔥 Heatmap",
        "📊 Statistical Tests",
        "🏆 Best/Worst",
        "⏱️ Computation Time",
        "💻 Resource Usage",
        "📋 Raw Data"
    ])
    
    with tab1:
        st.header("Comparison by Reconstruction Model")
        st.caption("📍 Local filters - apply only to this tab")
        
        # Sub-filters
        col1, col2 = st.columns(2)
        with col1:
            filter_technique = st.selectbox(
                "Filter by Technique",
                ['All'] + sorted(df['technique'].unique().tolist()),
                key='tab1_technique'
            )
        with col2:
            filter_rate = st.selectbox(
                "Filter by Missing Rate (%)",
                ['All'] + sorted(df['rate_percent'].unique().tolist()),
                key='tab1_rate'
            )
        
        plot_mad_by_model(
            df_filtered,
            technique=None if filter_technique == 'All' else filter_technique,
            rate=None if filter_rate == 'All' else filter_rate
        )
    
    with tab2:
        st.header("Comparison by Missingness Technique")
        st.caption("📍 Local filters - apply only to this tab")
        
        # Sub-filters
        col1, col2 = st.columns(2)
        with col1:
            filter_model = st.selectbox(
                "Filter by Model",
                ['All'] + sorted(df['model'].unique().tolist()),
                key='tab2_model'
            )
        with col2:
            filter_rate = st.selectbox(
                "Filter by Missing Rate (%)",
                ['All'] + sorted(df['rate_percent'].unique().tolist()),
                key='tab2_rate'
            )
        
        plot_mad_by_technique(
            df_filtered,
            model=None if filter_model == 'All' else filter_model,
            rate=None if filter_rate == 'All' else filter_rate
        )
    
    with tab3:
        st.header("Comparison by Missing Rate")
        st.caption("📍 Local filters - apply only to this tab")
        
        # Sub-filters
        col1, col2 = st.columns(2)
        with col1:
            filter_model = st.selectbox(
                "Filter by Model",
                ['All'] + sorted(df['model'].unique().tolist()),
                key='tab3_model'
            )
        with col2:
            filter_technique = st.selectbox(
                "Filter by Technique",
                ['All'] + sorted(df['technique'].unique().tolist()),
                key='tab3_technique'
            )
        
        plot_mad_by_rate(
            df_filtered,
            model=None if filter_model == 'All' else filter_model,
            technique=None if filter_technique == 'All' else filter_technique
        )
    
    with tab4:
        st.header("Comparison by Dataset")
        
        if len(df_filtered) > 0:
            plot_dataset_comparison(df_filtered)
        else:
            st.warning("No data available with current filters")
    
    with tab6:
        st.header("Heatmap: MAD by Model vs Technique")
        
        # Get available techniques for sorting
        if len(df_filtered) > 0:
            techniques = ['Alphabetical'] + sorted(df_filtered['technique'].unique().tolist())
        else:
            techniques = ['Alphabetical']
        
        sort_choice = st.selectbox(
            "Sort models by",
            techniques,
            help="Sort models by MAD performance on selected technique"
        )
        
        if len(df_filtered) > 0:
            sort_by = None if sort_choice == 'Alphabetical' else sort_choice
            plot_heatmap(df_filtered, metric='mad', sort_by_technique=sort_by)
        else:
            st.warning("No data available for heatmap with current filters")
    
    with tab7:
        st.header("📊 Statistical Significance Tests")
        st.caption("Pairwise t-tests between models - which differences are statistically significant?")
        
        if len(df_filtered) == 0:
            st.warning("No data available with current filters")
        else:
            # Import statistical test functions
            import sys
            from pathlib import Path
            sys.path.insert(0, str(Path(__file__).parent.parent))
            from utils.statistical_tests import (
                perform_pairwise_ttests, 
                get_pairwise_pvalues,
                get_model_statistics,
                get_significance_summary
            )
            
            # Info box explaining the analysis
            with st.expander("ℹ️ How to interpret this analysis", expanded=False):
                st.markdown("""
                **Statistical Significance Testing**:
                
                This tab performs **pairwise t-tests** between all models to determine if performance differences are statistically significant or just due to random chance.
                
                **Legend**:
                - **🟩 +2 (p<0.01)**: Row model is **significantly better** than column model (highly significant)
                - **🟢 +1 (p<0.05)**: Row model is **significantly better** than column model (significant)
                - **⬜ 0**: No significant difference
                - **🔴 -1 (p<0.05)**: Row model is **significantly worse** than column model (significant)
                - **🟥 -2 (p<0.01)**: Row model is **significantly worse** than column model (highly significant)
                
                **How to use**:
                1. Find your model in the row
                2. Look across the columns to see how it compares to other models
                3. Positive values (green) = your model is better
                4. Negative values (red) = your model is worse
                5. Absolute value shows strength: |2| = very strong (p<0.01), |1| = strong (p<0.05)
                
                **Example**: If "interpolate_linear" row shows "+2" in "knn" column, it means interpolate_linear is significantly better than knn (p<0.01).
                
                **Note**: Lower MAD = better performance. Tests use independent samples t-test on multiple iterations.
                """)
            
            st.divider()
            
            # Calculate statistics
            st.subheader("Model Performance Statistics")
            model_stats = get_model_statistics(df_filtered, metric='mad')
            
            # Display statistics table
            st.dataframe(
                model_stats.style.background_gradient(subset=['mean'], cmap='RdYlGn_r'),
                width='stretch'
            )
            
            st.divider()
            
            # Perform pairwise t-tests
            st.subheader("Pairwise Statistical Significance Matrix")
            st.caption("Each cell shows if row model is significantly different from column model")
            
            # Calculate significance matrix
            significance_matrix = perform_pairwise_ttests(df_filtered, metric='mad', alpha_01=0.01, alpha_05=0.05)
            
            # Create color mapping for heatmap
            # +2: dark green, +1: light green, 0: white, -1: red, -2: dark red
            def color_significance(val):
                if val == 2:
                    return 'background-color: #006400; color: white'  # Dark green (+2)
                elif val == 1:
                    return 'background-color: #90EE90; color: black'  # Light green (+1)
                elif val == 0:
                    return 'background-color: #FFFFFF; color: black'  # White (0)
                elif val == -1:
                    return 'background-color: #FF6B6B; color: black'  # Red (-1)
                elif val == -2:
                    return 'background-color: #8B0000; color: white'  # Dark red (-2)
                else:
                    return 'background-color: #CCCCCC; color: black'  # Gray
            
            # Apply styling
            styled_matrix = significance_matrix.style.map(color_significance)
            
            # Display matrix
            st.dataframe(styled_matrix, width='stretch', height=600)
            
            st.caption("""
            **Legend**: 🟩 +2 (p<0.01 better) | 🟢 +1 (p<0.05 better) | ⬜ 0 (no diff) | 🔴 -1 (p<0.05 worse) | 🟥 -2 (p<0.01 worse)
            """)
            
            st.divider()
            
            # Summary statistics per model
            st.subheader("Significance Summary by Model")
            st.caption("How many models is each model significantly better/worse than?")
            
            significance_summary = get_significance_summary(significance_matrix)
            summary_df = pd.DataFrame(significance_summary).T
            summary_df = summary_df.reset_index()
            summary_df.columns = ['Model', 'Better (p<0.01)', 'Better (p<0.05)', 'No Difference', 'Worse (p<0.05)', 'Worse (p<0.01)']
            
            # Sort by number of models it's significantly better than
            summary_df = summary_df.sort_values('Better (p<0.01)', ascending=False)
            
            st.dataframe(summary_df, width='stretch')
            
            st.divider()
            
            # Optional: Show p-values matrix
            with st.expander("🔬 Show detailed p-values matrix", expanded=False):
                st.caption("Exact p-values for all pairwise comparisons")
                pvalue_matrix = get_pairwise_pvalues(df_filtered, metric='mad')
                
                # Style p-values: highlight significant ones
                def color_pvalue(val):
                    if val < 0.01:
                        return 'background-color: #90EE90; font-weight: bold'
                    elif val < 0.05:
                        return 'background-color: #FFFFCC'
                    else:
                        return ''
                
                styled_pvalues = pvalue_matrix.style.map(color_pvalue).format("{:.4f}")
                st.dataframe(styled_pvalues, width='stretch', height=600)
    
    with tab8:
        st.header("Best and Worst Performing Models")
        
        top_n = st.slider("Number of models to show", 5, 20, 10)
        
        if len(df_filtered) > 0:
            plot_best_worst_models(df_filtered, top_n=top_n)
        else:
            st.warning("No data available with current filters")
    
    with tab9:
        st.header("⏱️ Computation Time Analysis")
        st.caption("📍 Computational complexity metrics - execution time")
        
        # Check if performance metrics are available in the data
        if 'time_seconds' not in df_filtered.columns or df_filtered['time_seconds'].isna().all():
            st.warning("⚠️ No performance metrics available in this results file.")
            st.info("Run `4_reconstruct_datasets.py` again to collect performance metrics, then `5_calculate_mad.py` to merge them.")
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
                
                # Time by model
                st.subheader("Execution Time by Model")
                model_time = df_perf.groupby('model')['time_seconds'].agg(['mean', 'std', 'min', 'max', 'count']).reset_index()
                model_time = model_time.sort_values('mean', ascending=False)
                
                fig = px.bar(
                    model_time,
                    x='mean',
                    y='model',
                    orientation='h',
                    error_x='std',
                    title="Average Execution Time by Model (with std dev)",
                    labels={'mean': 'Average Time (seconds)', 'model': 'Model'},
                    color='mean',
                    color_continuous_scale='Reds'
                )
                fig.update_layout(height=max(400, len(model_time) * 30), showlegend=False)
                st.plotly_chart(fig, width='stretch')
                
                # Time comparison: boxplot
                st.subheader("Time Distribution by Model")
                fig = px.box(
                    df_perf,
                    x='model',
                    y='time_seconds',
                    title="Execution Time Distribution",
                    labels={'time_seconds': 'Time (seconds)', 'model': 'Model'},
                    color='model'
                )
                fig.update_xaxes(tickangle=45)
                st.plotly_chart(fig, width='stretch')
                
                # Time by technique and dataset
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Average Time by Technique")
                    tech_time = df_perf.groupby('technique')['time_seconds'].mean().reset_index()
                    fig = px.bar(
                        tech_time,
                        x='technique',
                        y='time_seconds',
                        title="Average Time by Missingness Technique",
                        labels={'time_seconds': 'Avg Time (seconds)', 'technique': 'Technique'},
                        color='time_seconds',
                        color_continuous_scale='Blues'
                    )
                    st.plotly_chart(fig, width='stretch')
                
                with col2:
                    st.subheader("Average Time by Missing Rate")
                    rate_time = df_perf.groupby('rate_percent')['time_seconds'].mean().reset_index()
                    fig = px.bar(
                        rate_time,
                        x='rate_percent',
                        y='time_seconds',
                        title="Average Time by Missing Rate",
                        labels={'time_seconds': 'Avg Time (seconds)', 'rate_percent': 'Missing Rate (%)'},
                        color='time_seconds',
                        color_continuous_scale='Greens'
                    )
                    st.plotly_chart(fig, width='stretch')
                
                # Detailed table
                st.subheader("Detailed Statistics")
                st.dataframe(model_time, width='stretch')
    
    with tab10:
        st.header("💻 Resource Usage Analysis")
        st.caption("📍 Computational complexity metrics - CPU, RAM, GPU usage")
        
        # Check if performance metrics are available in the data
        if 'cpu_cores_used' not in df_filtered.columns or df_filtered['cpu_cores_used'].isna().all():
            st.warning("⚠️ No performance metrics available in this results file.")
            st.info("Run `4_reconstruct_datasets.py` again to collect performance metrics, then `5_calculate_mad.py` to merge them.")
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
                    total_cores = df_perf['cpu_cores_total'].mode()[0] if 'cpu_cores_total' in df_perf.columns else 0
                    st.metric("Avg CPU Cores", f"{avg_cores:.2f} / {total_cores:.0f}")
                with col2:
                    if 'memory_total_mb' in df_perf.columns and df_perf['memory_total_mb'].notna().any():
                        avg_memory = df_perf['memory_mb'].mean()
                        total_memory = df_perf['memory_total_mb'].iloc[0]
                        st.metric("Avg RAM Usage", f"{avg_memory:.1f} / {total_memory:.0f} MB")
                    else:
                        st.metric("Avg RAM Usage", f"{df_perf['memory_mb'].mean():.1f} MB")
                with col3:
                    if df_perf['gpu_percent'].notna().any():
                        st.metric("Avg GPU Usage", f"{df_perf['gpu_percent'].mean():.1f}%")
                    else:
                        st.metric("GPU Usage", "N/A")
                
                st.markdown("---")
                
                # CPU cores usage by model
                st.subheader("CPU+GPU Cores Utilized by Model")
                model_cpu = df_perf.groupby('model')['cpu_cores_used'].agg(['mean', 'std', 'max']).reset_index()
                model_cpu = model_cpu.sort_values('mean', ascending=False)
                
                fig = px.bar(
                    model_cpu,
                    x='mean',
                    y='model',
                    orientation='h',
                    title="Average CPU+GPU Cores Utilized by Model",
                    labels={'mean': 'Avg CPU+GPU Cores', 'model': 'Model'},
                    color='mean',
                    color_continuous_scale='Oranges'
                )
                fig.update_layout(height=max(400, len(model_cpu) * 30), showlegend=False)
                st.plotly_chart(fig, width='stretch')
                
                # Combined CPU + GPU utilization for comparison
                if df_perf['gpu_percent'].notna().any():

                    # Prepare data: for each model show CPU and GPU side by side
                    model_compute = df_perf.groupby('model').agg({
                        'cpu_cores_used': 'mean',
                        'gpu_percent': 'mean'
                    }).reset_index()
                    
                    # Fill NaN GPU values with 0 for CPU-only models
                    model_compute['gpu_percent'] = model_compute['gpu_percent'].fillna(0)
                    
                    # Create grouped bar chart
                    model_compute_melted = model_compute.melt(
                        id_vars=['model'],
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
                    model_order = model_compute['model'].tolist()
                    
                    fig = px.bar(
                        model_compute_melted,
                        x='Usage',
                        y='model',
                        color='Resource Type',
                        orientation='h',
                        title="CPU vs GPU Utilization by Model",
                        labels={'Usage': 'Utilization', 'model': 'Model'},
                        barmode='group',
                        category_orders={'model': model_order},
                        color_discrete_map={'CPU Cores': '#FFA500', 'GPU Utilization (%)': '#4CAF50'}
                    )
                    fig.update_layout(height=max(400, len(model_compute) * 40))
                    st.plotly_chart(fig, width='stretch')
                
                # RAM usage by model
                st.subheader("Memory (RAM) Usage by Model")
                model_mem = df_perf.groupby('model')['memory_mb'].agg(['mean', 'std', 'max']).reset_index()
                model_mem = model_mem.sort_values('mean', ascending=False)
                
                fig = px.bar(
                    model_mem,
                    x='mean',
                    y='model',
                    orientation='h',
                    title="Average Memory Usage by Model",
                    labels={'mean': 'Avg Memory (MB)', 'model': 'Model'},
                    color='mean',
                    color_continuous_scale='Purples'
                )
                fig.update_layout(height=max(400, len(model_mem) * 30), showlegend=False)
                st.plotly_chart(fig, width='stretch')
    
    with tab5:
        st.header("⚡ Resource Efficiency Analysis")
        st.caption("Lower values = more efficient (less time and resources)")
        
        # Check if performance metrics are available in the data
        if 'cpu_cores_used' not in df_filtered.columns or df_filtered['cpu_cores_used'].isna().all():
            st.warning("⚠️ No performance metrics available. Run reconstructions first to collect performance data.")
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
                    - GPU-based models (e.g., Stable Diffusion) typically have higher scores due to GPU memory usage
                    
                    **Use Cases:**
                    - Select models for resource-constrained environments (edge devices, embedded systems)
                    - Balance reconstruction quality (MAD) vs. computational cost
                    - Compare CPU-based vs. GPU-based models fairly
                    - Optimize for deployment scenarios (cloud costs, energy efficiency)
                    """)
                
                st.divider()
                
                # Create efficiency score: normalize time, CPU, RAM, and GPU
                df_perf_copy = df_perf.copy()
                df_perf_copy['time_norm'] = (df_perf_copy['time_seconds'] - df_perf_copy['time_seconds'].min()) / (df_perf_copy['time_seconds'].max() - df_perf_copy['time_seconds'].min() + 0.001)
                # Normalize CPU cores (divide by total available cores)
                total_cores = df_perf_copy['cpu_cores_total'].mode()[0] if 'cpu_cores_total' in df_perf_copy.columns else 1
                df_perf_copy['cpu_norm'] = df_perf_copy['cpu_cores_used'] / total_cores
                df_perf_copy['mem_norm'] = (df_perf_copy['memory_mb'] - df_perf_copy['memory_mb'].min()) / (df_perf_copy['memory_mb'].max() - df_perf_copy['memory_mb'].min() + 0.001)
                
                # Normalize GPU memory (if available)
                if 'gpu_memory_mb' in df_perf_copy.columns and 'gpu_memory_total_mb' in df_perf_copy.columns:
                    # For models with GPU usage, normalize by total GPU memory
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
                
                efficiency = df_perf_copy.groupby('model')['efficiency_score'].mean().reset_index()
                efficiency = efficiency.sort_values('efficiency_score', ascending=False)  # Ascending: lower = better
                
                # Overall efficiency score by model
                st.subheader("Overall Efficiency Score by Model")
                st.caption("Models sorted by efficiency (best to worst)")
                
                fig = px.bar(
                    efficiency,
                    x='efficiency_score',
                    y='model',
                    orientation='h',
                    title="Overall Efficiency Score by Model (lower = better)",
                    labels={'efficiency_score': 'Efficiency Score', 'model': 'Model'},
                    color='efficiency_score',
                    color_continuous_scale='RdYlGn_r'
                )
                fig.update_layout(height=max(400, len(efficiency) * 30), showlegend=False)
                st.plotly_chart(fig, width='stretch')
                
                # Combined scatter plot
                st.subheader("Time vs Memory Usage")
                st.caption("Bubble size represents combined CPU + GPU utilization")
                
                # Aggregate metrics including GPU
                model_summary = df_perf.groupby('model').agg({
                    'time_seconds': 'mean',
                    'memory_mb': 'mean',
                    'cpu_cores_used': 'mean',
                    'gpu_memory_mb': 'mean',
                    'gpu_memory_total_mb': 'mean'
                }).reset_index()
                
                # Calculate combined CPU + GPU metric for bubble size
                # GPU normalized to equivalent "cores" (0-4 scale to match CPU range)
                model_summary['gpu_normalized'] = model_summary.apply(
                    lambda row: (row['gpu_memory_mb'] / row['gpu_memory_total_mb']) * 4.0
                    if pd.notna(row['gpu_memory_mb']) and pd.notna(row['gpu_memory_total_mb']) and row['gpu_memory_total_mb'] > 0
                    else 0.0,
                    axis=1
                )
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
                    text='model',
                    title="Time vs Memory (bubble size = CPU + GPU utilization)",
                    labels={'time_seconds': 'Avg Time (seconds)', 'memory_mb': 'Avg Memory (MB)'},
                    color='compute_type',
                    color_discrete_map={'CPU only': '#3498db', 'CPU+GPU': '#e74c3c'}
                )
                fig.update_traces(textposition='top center')
                st.plotly_chart(fig, width='stretch')
    
    with tab11:
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
        
        st.dataframe(df_display, width='stretch')
        
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
