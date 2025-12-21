#!/usr/bin/env python3
"""
Streamlit Visualization App for Reconstruction Results
Interactive dashboard for comparing reconstruction models, techniques, and missing rates.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from pathlib import Path
import sys


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
    results_dir = Path("experiments_results")
    if not results_dir.exists():
        return []
    
    return sorted(results_dir.glob("*.csv"), reverse=True)


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


def plot_heatmap(df: pd.DataFrame, metric: str = 'mad'):
    """Plot heatmap of MAD for model vs technique"""
    # Calculate mean MAD for each model-technique combination
    pivot_data = df.pivot_table(
        values=metric,
        index='model',
        columns='technique',
        aggfunc='mean'
    )
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_data.values,
        x=pivot_data.columns,
        y=pivot_data.index,
        colorscale='RdYlGn_r',
        text=np.round(pivot_data.values, 4),
        texttemplate='%{text}',
        textfont={"size": 10},
        colorbar=dict(title=metric.upper())
    ))
    
    fig.update_layout(
        title=f'Heatmap: {metric.upper()} by Model and Technique',
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
    
    # Best models
    best_models = df_stats.head(top_n)
    
    # Worst models
    worst_models = df_stats.tail(top_n).sort_values('mad', ascending=False)
    
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
    fig.update_xaxes(title_text="MAD", row=1, col=1)
    fig.update_xaxes(title_text="MAD", row=1, col=2)
    
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
        st.error("No result files found in `experiments_results/` directory.")
        st.info("Run `python calculate_differences.py` first to generate results.")
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
    
    # Get unique values
    all_datasets = ['All'] + sorted(df['dataset_name'].unique().tolist())
    all_models = ['All'] + sorted(df['model'].unique().tolist())
    all_techniques = ['All'] + sorted(df['technique'].unique().tolist())
    all_rates = ['All'] + sorted(df['rate_percent'].unique().tolist())
    
    selected_dataset = st.sidebar.selectbox("Dataset", all_datasets)
    selected_model = st.sidebar.selectbox("Model", all_models)
    selected_technique = st.sidebar.selectbox("Technique", all_techniques)
    selected_rate = st.sidebar.selectbox("Missing Rate (%)", all_rates)
    
    # Apply filters to dataframe
    df_filtered = df.copy()
    if selected_dataset != 'All':
        df_filtered = df_filtered[df_filtered['dataset_name'] == selected_dataset]
    if selected_model != 'All':
        df_filtered = df_filtered[df_filtered['model'] == selected_model]
    if selected_technique != 'All':
        df_filtered = df_filtered[df_filtered['technique'] == selected_technique]
    if selected_rate != 'All':
        df_filtered = df_filtered[df_filtered['rate_percent'] == selected_rate]
    
    # Display overview metrics
    st.header("📈 Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Records", len(df_filtered))
    with col2:
        st.metric("Mean MAD", f"{df_filtered['mad'].mean():.4f}")
    with col3:
        st.metric("Median MAD", f"{df_filtered['mad'].median():.4f}")
    with col4:
        st.metric("Best MAD", f"{df_filtered['mad'].min():.4f}")
    
    st.markdown("---")
    
    # Visualization tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 By Model", 
        "🎯 By Technique", 
        "📉 By Missing Rate",
        "🔥 Heatmap",
        "🏆 Best/Worst",
        "📁 Raw Data"
    ])
    
    with tab1:
        st.header("Comparison by Reconstruction Model")
        
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
        st.header("Heatmap: Model vs Technique")
        
        metric_choice = st.selectbox(
            "Metric",
            ['mad', 'max_diff', 'std_diff'],
            format_func=lambda x: x.upper().replace('_', ' ')
        )
        
        if len(df_filtered) > 0:
            plot_heatmap(df_filtered, metric=metric_choice)
        else:
            st.warning("No data available for heatmap with current filters")
    
    with tab5:
        st.header("Best and Worst Performing Models")
        
        top_n = st.slider("Number of models to show", 5, 20, 10)
        
        if len(df_filtered) > 0:
            plot_best_worst_models(df_filtered, top_n=top_n)
            
            # Also show dataset comparison
            st.subheader("Dataset Comparison")
            plot_dataset_comparison(df_filtered)
        else:
            st.warning("No data available with current filters")
    
    with tab6:
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
