#!/usr/bin/env python3
"""
Statistical significance testing for model comparison
"""

import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Tuple, List


def perform_pairwise_ttests(
    df: pd.DataFrame,
    metric: str = 'mad',
    alpha_01: float = 0.01,
    alpha_05: float = 0.05,
    lower_is_better: bool = True,
) -> pd.DataFrame:
    """
    Perform pairwise t-tests between all models.
    
    Args:
        df: DataFrame with columns: model, mad (or other metric), and multiple observations per model
        metric: Column name to compare (default: 'mad')
        alpha_01: Significance level for highly significant differences (default: 0.01)
        alpha_05: Significance level for significant differences (default: 0.05)
        lower_is_better: If False (e.g. R²), higher mean counts as better
    
    Returns:
        DataFrame with pairwise comparison results:
        - Rows and columns are model names
        - Values indicate significance and direction:
            2: Row model significantly better than column model (p < 0.01)
            1: Row model significantly better than column model (p < 0.05)
            0: No significant difference
           -1: Row model significantly worse than column model (p < 0.05)
           -2: Row model significantly worse than column model (p < 0.01)
    """
    # Get unique models
    models = sorted(df['model'].unique())
    n_models = len(models)
    
    # Initialize result matrix
    result_matrix = pd.DataFrame(0, index=models, columns=models)
    
    # Perform pairwise t-tests
    for i, model_a in enumerate(models):
        for j, model_b in enumerate(models):
            if i == j:
                # Same model - no comparison needed
                result_matrix.loc[model_a, model_b] = 0
                continue
            
            # Get metric values for both models
            values_a = df[df['model'] == model_a][metric].dropna()
            values_b = df[df['model'] == model_b][metric].dropna()
            
            # Need at least 2 samples for t-test
            if len(values_a) < 2 or len(values_b) < 2:
                result_matrix.loc[model_a, model_b] = 0
                continue
            
            # Independent samples t-test; direction uses lower_is_better
            t_stat, p_value = stats.ttest_ind(values_a, values_b)
            
            # Determine significance and direction
            mean_a = values_a.mean()
            mean_b = values_b.mean()
            a_better = mean_a < mean_b if lower_is_better else mean_a > mean_b
            
            if p_value < alpha_01:
                # Highly significant difference (p < 0.01)
                if a_better:
                    result_matrix.loc[model_a, model_b] = 2  # model_a is significantly better
                else:
                    result_matrix.loc[model_a, model_b] = -2  # model_a is significantly worse
            elif p_value < alpha_05:
                # Significant difference (p < 0.05)
                if a_better:
                    result_matrix.loc[model_a, model_b] = 1  # model_a is significantly better
                else:
                    result_matrix.loc[model_a, model_b] = -1  # model_a is significantly worse
            else:
                # No significant difference
                result_matrix.loc[model_a, model_b] = 0
    
    return result_matrix


def get_pairwise_pvalues(df: pd.DataFrame, metric: str = 'mad') -> pd.DataFrame:
    """
    Get matrix of p-values for all pairwise comparisons.
    
    Args:
        df: DataFrame with columns: model, mad (or other metric)
        metric: Column name to compare (default: 'mad')
    
    Returns:
        DataFrame with p-values for each pair of models
    """
    models = sorted(df['model'].unique())
    pvalue_matrix = pd.DataFrame(1.0, index=models, columns=models)
    
    for i, model_a in enumerate(models):
        for j, model_b in enumerate(models):
            if i == j:
                pvalue_matrix.loc[model_a, model_b] = 1.0
                continue
            
            values_a = df[df['model'] == model_a][metric].dropna()
            values_b = df[df['model'] == model_b][metric].dropna()
            
            if len(values_a) < 2 or len(values_b) < 2:
                pvalue_matrix.loc[model_a, model_b] = 1.0
                continue
            
            _, p_value = stats.ttest_ind(values_a, values_b)
            pvalue_matrix.loc[model_a, model_b] = p_value
    
    return pvalue_matrix


def get_model_statistics(df: pd.DataFrame, metric: str = 'mad', lower_is_better: bool = True) -> pd.DataFrame:
    """
    Calculate summary statistics for each model.
    
    Args:
        df: DataFrame with columns: model, mad (or other metric)
        metric: Column name to analyze (default: 'mad')
        lower_is_better: Sort order for mean column (False for R²)
    
    Returns:
        DataFrame with statistics per model (mean, std, count, etc.)
    """
    stats_df = df.groupby('model')[metric].agg([
        ('mean', 'mean'),
        ('std', 'std'),
        ('median', 'median'),
        ('min', 'min'),
        ('max', 'max'),
        ('count', 'count')
    ]).reset_index()
    
    # Calculate standard error
    stats_df['se'] = stats_df['std'] / np.sqrt(stats_df['count'])
    
    stats_df = stats_df.sort_values('mean', ascending=lower_is_better)
    
    return stats_df


def get_significance_summary(significance_matrix: pd.DataFrame) -> Dict[str, int]:
    """
    Summarize significance results for each model.
    
    Args:
        significance_matrix: Output from perform_pairwise_ttests()
    
    Returns:
        Dictionary with model names as keys and summary statistics:
        - significantly_better_001: Count of models this model is better than (p<0.01)
        - significantly_better_005: Count of models this model is better than (p<0.05)
        - significantly_worse_001: Count of models this model is worse than (p<0.01)
        - significantly_worse_005: Count of models this model is worse than (p<0.05)
        - no_difference: Count of models with no significant difference
    """
    summary = {}
    
    for model in significance_matrix.index:
        row = significance_matrix.loc[model]
        
        summary[model] = {
            'significantly_better_p001': (row == 2).sum(),
            'significantly_better_p005': (row == 1).sum(),
            'no_difference': (row == 0).sum() - 1,  # -1 to exclude self-comparison
            'significantly_worse_p005': (row == -1).sum(),
            'significantly_worse_p001': (row == -2).sum(),
        }
    
    return summary


def format_significance_text(value: int) -> str:
    """
    Format significance value as human-readable text.
    
    Args:
        value: Significance code from perform_pairwise_ttests()
    
    Returns:
        Text description
    """
    if value == 2:
        return "+2 (p<0.01)"
    elif value == 1:
        return "+1 (p<0.05)"
    elif value == 0:
        return "0"
    elif value == -1:
        return "-1 (p<0.05)"
    elif value == -2:
        return "-2 (p<0.01)"
    else:
        return "?"

