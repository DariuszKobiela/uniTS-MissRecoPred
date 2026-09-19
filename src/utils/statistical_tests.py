#!/usr/bin/env python3
"""Paired statistical tests for model comparisons without pseudoreplication."""

from itertools import combinations
import pandas as pd
import numpy as np
from scipy import stats
from typing import Dict, Sequence


DEFAULT_PAIR_CANDIDATES = (
    ("dataset_name", "dataset"),
    ("technique", "mechanism"),
    ("rate_percent", "rate"),
    ("structure", "gap_pattern"),
    ("iteration", "reconstruction_iteration", "seed"),
)


def _resolve_pair_columns(df: pd.DataFrame, pair_columns: Sequence[str] | None) -> list[str]:
    if pair_columns is not None:
        missing = [column for column in pair_columns if column not in df.columns]
        if missing:
            raise ValueError(f"Missing pairing columns: {missing}")
        columns = list(pair_columns)
    else:
        columns = [
            next((name for name in candidates if name in df.columns), "")
            for candidates in DEFAULT_PAIR_CANDIDATES
        ]
        columns = [column for column in columns if column]

    dataset_column = next(
        (column for column in ("dataset_name", "dataset") if column in columns), None
    )
    if dataset_column is None:
        raise ValueError(
            "Paired inference requires `dataset_name` or `dataset` in the pairing columns"
        )
    return columns


def _holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values, preserving the input order."""
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan)
    finite = np.flatnonzero(np.isfinite(values))
    if not len(finite):
        return adjusted
    order = finite[np.argsort(values[finite])]
    running_max = 0.0
    m = len(order)
    for rank, index in enumerate(order):
        running_max = max(running_max, (m - rank) * values[index])
        adjusted[index] = min(1.0, running_max)
    return adjusted


def _bootstrap_mean_ci(
    differences: np.ndarray,
    confidence: float = 0.95,
    n_resamples: int = 10_000,
    random_state: int = 42,
) -> tuple[float, float]:
    if len(differences) < 2:
        return np.nan, np.nan
    if np.allclose(differences, differences[0]):
        value = float(differences[0])
        return value, value
    result = stats.bootstrap(
        (differences,),
        np.mean,
        confidence_level=confidence,
        n_resamples=n_resamples,
        random_state=random_state,
        method="percentile",
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


def _rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0]
    if not len(nonzero):
        return 0.0
    ranks = stats.rankdata(np.abs(nonzero))
    positive = ranks[nonzero > 0].sum()
    negative = ranks[nonzero < 0].sum()
    return float((positive - negative) / (positive + negative))


def _paired_values(
    df: pd.DataFrame,
    model_a: str,
    model_b: str,
    metric: str,
    pair_columns: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, int]:
    """Match exact conditions, then collapse repeated conditions within each dataset."""
    subset = df.loc[df["model"].isin((model_a, model_b)), [*pair_columns, "model", metric]]
    duplicate_keys = [*pair_columns, "model"]
    subset = subset.groupby(duplicate_keys, dropna=False, as_index=False)[metric].mean()
    wide = subset.pivot(index=list(pair_columns), columns="model", values=metric).dropna(
        subset=[model_a, model_b]
    )
    exact_pairs = len(wide)
    dataset_column = next(c for c in ("dataset_name", "dataset") if c in pair_columns)
    by_dataset = wide[[model_a, model_b]].groupby(level=dataset_column).mean()
    return (
        by_dataset[model_a].to_numpy(dtype=float),
        by_dataset[model_b].to_numpy(dtype=float),
        exact_pairs,
    )


def pairwise_comparisons(
    df: pd.DataFrame,
    metric: str = "mad",
    pair_columns: Sequence[str] | None = None,
    alpha: float = 0.05,
    normality_alpha: float = 0.05,
    lower_is_better: bool = True,
) -> pd.DataFrame:
    """Compare every model pair using dataset-level paired observations.

    Exact records are matched by dataset × mechanism × rate × gap pattern × seed
    (using the corresponding column names available in ``df``). Differences from
    repeated conditions are averaged within each dataset before inference.
    """
    required = {"model", metric}
    if missing := required.difference(df.columns):
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    pairs = _resolve_pair_columns(df, pair_columns)
    records = []
    for model_a, model_b in combinations(sorted(df["model"].dropna().unique()), 2):
        values_a, values_b, exact_pairs = _paired_values(
            df, model_a, model_b, metric, pairs
        )
        differences = values_a - values_b
        n = len(differences)
        shapiro_p = (
            float(stats.shapiro(differences).pvalue) if 3 <= n <= 5000 else np.nan
        )
        use_t = n >= 3 and np.isfinite(shapiro_p) and shapiro_p >= normality_alpha
        if n < 2:
            test, statistic, p_value = "insufficient_pairs", np.nan, np.nan
            effect_name, effect_size = "cohen_dz", np.nan
        elif use_t:
            result = stats.ttest_rel(values_a, values_b)
            test, statistic, p_value = "paired_t", float(result.statistic), float(result.pvalue)
            sd = differences.std(ddof=1)
            effect_name = "cohen_dz"
            effect_size = float(differences.mean() / sd) if sd > 0 else 0.0
        else:
            test = "wilcoxon"
            if np.allclose(differences, 0):
                statistic, p_value = 0.0, 1.0
            else:
                result = stats.wilcoxon(differences, zero_method="wilcox")
                statistic, p_value = float(result.statistic), float(result.pvalue)
            effect_name, effect_size = "rank_biserial", _rank_biserial(differences)
        ci_low, ci_high = _bootstrap_mean_ci(differences)
        mean_difference = float(differences.mean()) if n else np.nan
        records.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "test": test,
                "n_datasets": n,
                "n_exact_pairs": exact_pairs,
                "shapiro_p": shapiro_p,
                "statistic": statistic,
                "p_value": p_value,
                "mean_difference": mean_difference,
                "ci_95_low": ci_low,
                "ci_95_high": ci_high,
                "effect_name": effect_name,
                "effect_size": effect_size,
                "a_better": (
                    mean_difference < 0 if lower_is_better else mean_difference > 0
                ) if np.isfinite(mean_difference) else False,
            }
        )
    result = pd.DataFrame.from_records(records)
    if result.empty:
        return result
    result["p_holm"] = _holm_adjust(result["p_value"])
    result["significant"] = result["p_holm"] < alpha
    return result


def friedman_test(
    df: pd.DataFrame,
    metric: str = "mad",
    pair_columns: Sequence[str] | None = None,
) -> Dict[str, float]:
    """Friedman omnibus test on complete, dataset-level model blocks."""
    pairs = _resolve_pair_columns(df, pair_columns)
    models = sorted(df["model"].dropna().unique())
    grouped = df.groupby([*pairs, "model"], dropna=False, as_index=False)[metric].mean()
    wide = grouped.pivot(index=pairs, columns="model", values=metric)
    dataset_column = next(c for c in ("dataset_name", "dataset") if c in pairs)
    wide = wide.groupby(level=dataset_column).mean().dropna(subset=models)
    if len(models) < 3 or len(wide) < 2:
        return {"statistic": np.nan, "p_value": np.nan, "n_datasets": len(wide), "n_models": len(models)}
    result = stats.friedmanchisquare(*(wide[model].to_numpy() for model in models))
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        "n_datasets": len(wide),
        "n_models": len(models),
    }


def perform_pairwise_ttests(
    df: pd.DataFrame,
    metric: str = 'mad',
    alpha_01: float = 0.01,
    alpha_05: float = 0.05,
    lower_is_better: bool = True,
    pair_columns: Sequence[str] | None = None,
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
    comparisons = pairwise_comparisons(
        df,
        metric=metric,
        lower_is_better=lower_is_better,
        alpha=alpha_05,
        pair_columns=pair_columns,
    )
    models = sorted(df['model'].dropna().unique())
    n_models = len(models)
    
    # Initialize result matrix
    result_matrix = pd.DataFrame(0, index=models, columns=models)
    
    for row in comparisons.itertuples():
        value = 0
        if row.p_holm < alpha_01:
            value = 2 if row.a_better else -2
        elif row.p_holm < alpha_05:
            value = 1 if row.a_better else -1
        result_matrix.loc[row.model_a, row.model_b] = value
        result_matrix.loc[row.model_b, row.model_a] = -value
    
    return result_matrix


def get_pairwise_pvalues(
    df: pd.DataFrame,
    metric: str = 'mad',
    pair_columns: Sequence[str] | None = None,
) -> pd.DataFrame:
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
    
    for row in pairwise_comparisons(
        df, metric=metric, pair_columns=pair_columns
    ).itertuples():
        pvalue_matrix.loc[row.model_a, row.model_b] = row.p_holm
        pvalue_matrix.loc[row.model_b, row.model_a] = row.p_holm
    
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

