#!/usr/bin/env python3
"""Paired statistical tests for model comparisons without pseudoreplication."""

import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from itertools import combinations
from typing import Dict, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from utils.progress import tqdm

DEFAULT_PAIR_CANDIDATES = (
    ("dataset_name", "dataset"),
    ("technique", "mechanism"),
    ("rate_percent", "rate"),
    ("structure", "gap_pattern"),
    ("iteration", "reconstruction_iteration", "seed"),
)

REPLICATE_COLUMNS = {"iteration", "reconstruction_iteration", "seed"}


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


SHAPIRO_MAX_N = 5_000
SHAPIRO_RANDOM_STATE = 42


def _shapiro_wilk(differences: np.ndarray) -> tuple[float, int, bool]:
    """Run Shapiro-Wilk, using a reproducible random sample above 5,000."""
    values = np.asarray(differences, dtype=float)
    n = len(values)
    if n < 3 or np.allclose(values, values[0]):
        return np.nan, 0, False
    sampled = n > SHAPIRO_MAX_N
    if sampled:
        rng = np.random.default_rng(SHAPIRO_RANDOM_STATE)
        values = values[rng.choice(n, size=SHAPIRO_MAX_N, replace=False)]
    return float(stats.shapiro(values).pvalue), len(values), sampled


def _evaluate_pair(task: tuple[str, str, np.ndarray, np.ndarray, int, int, str, float, bool]) -> dict:
    """Run one paired comparison on dataset- or condition-level summaries."""
    (model_a, model_b, values_a, values_b, exact_pairs, n_source_datasets,
     inference_unit, normality_alpha, lower_is_better) = task
    differences = values_a - values_b
    n = len(differences)
    # SciPy warns that p-values are inaccurate above 5,000 observations.
    # Use a reproducible random subsample rather than leaving shapiro_p empty.
    shapiro_p, shapiro_n, shapiro_sampled = _shapiro_wilk(differences)
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
    return {
        "model_a": model_a, "model_b": model_b, "test": test,
        "n_units": n, "n_datasets": n_source_datasets,
        "inference_unit": inference_unit, "n_exact_pairs": exact_pairs,
        "shapiro_p": shapiro_p, "shapiro_n": shapiro_n,
        "shapiro_sampled": shapiro_sampled,
        "statistic": statistic, "p_value": p_value, "mean_difference": mean_difference,
        "ci_95_low": ci_low, "ci_95_high": ci_high, "effect_name": effect_name,
        "effect_size": effect_size,
        "a_better": (mean_difference < 0 if lower_is_better else mean_difference > 0)
        if np.isfinite(mean_difference) else False,
    }


def pairwise_comparisons(
    df: pd.DataFrame,
    metric: str = "mad",
    pair_columns: Sequence[str] | None = None,
    alpha: float = 0.05,
    normality_alpha: float = 0.05,
    lower_is_better: bool = True,
    show_progress: bool = False,
    progress_desc: str = "Pairwise comparisons",
    n_jobs: int = 1,
    inference_unit: str = "dataset",
) -> pd.DataFrame:
    """Compare models after exact pairing and aggregation to inference units.

    Condition inference uses dataset x mechanism x structure x rate cells and
    averages stochastic iterations within each cell. Dataset inference also
    averages all experimental conditions within each source dataset.
    """
    required = {"model", metric}
    if missing := required.difference(df.columns):
        raise ValueError(f"Missing required columns: {sorted(missing)}")
    pairs = _resolve_pair_columns(df, pair_columns)
    if inference_unit not in {"dataset", "condition"}:
        raise ValueError("inference_unit must be dataset or condition")
    models = sorted(df["model"].dropna().unique())
    grouped = df.groupby([*pairs, "model"], dropna=False, as_index=False)[metric].mean()
    wide = grouped.pivot(index=pairs, columns="model", values=metric)
    dataset_column = next(c for c in ("dataset_name", "dataset") if c in pairs)
    condition_columns = [column for column in pairs if column not in REPLICATE_COLUMNS]
    tasks = []
    for model_a, model_b in combinations(models, 2):
        paired = wide[[model_a, model_b]].dropna(subset=[model_a, model_b])
        n_source_datasets = paired.index.get_level_values(dataset_column).nunique()
        if inference_unit == "dataset":
            inference_frame = paired.groupby(level=dataset_column).mean()
        else:
            inference_frame = paired.groupby(level=condition_columns).mean()
        tasks.append((model_a, model_b,
                      inference_frame[model_a].to_numpy(dtype=float),
                      inference_frame[model_b].to_numpy(dtype=float), len(paired),
                      n_source_datasets, inference_unit, normality_alpha,
                      lower_is_better))
    workers = max(1, int(n_jobs))
    executor = None
    if workers == 1 or len(tasks) < 2:
        results = map(_evaluate_pair, tasks)
    else:
        executor = ProcessPoolExecutor(
            max_workers=min(workers, len(tasks)),
            mp_context=mp.get_context("spawn"),
        )
        results = executor.map(_evaluate_pair, tasks)
    iterator = tqdm(results, total=len(tasks), desc=progress_desc, unit="pair",
                    dynamic_ncols=True, disable=not show_progress)
    try:
        records = list(iterator)
    finally:
        if executor is not None:
            executor.shutdown()
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
    inference_unit: str = "dataset",
) -> Dict[str, float]:
    """Friedman omnibus test on dataset or iteration-averaged condition blocks."""
    pairs = _resolve_pair_columns(df, pair_columns)
    if inference_unit not in {"dataset", "condition"}:
        raise ValueError("inference_unit must be dataset or condition")
    models = sorted(df["model"].dropna().unique())
    grouped = df.groupby([*pairs, "model"], dropna=False, as_index=False)[metric].mean()
    wide = grouped.pivot(index=pairs, columns="model", values=metric)
    dataset_column = next(c for c in ("dataset_name", "dataset") if c in pairs)
    n_datasets = wide.index.get_level_values(dataset_column).nunique()
    if inference_unit == "dataset":
        wide = wide.groupby(level=dataset_column).mean()
    else:
        condition_columns = [column for column in pairs if column not in REPLICATE_COLUMNS]
        wide = wide.groupby(level=condition_columns).mean()
    wide = wide.dropna(subset=models)
    base = {"n_datasets": n_datasets, "n_units": len(wide),
            "n_models": len(models), "inference_unit": inference_unit}
    if len(models) < 3 or len(wide) < 2:
        return {"statistic": np.nan, "p_value": np.nan, **base}
    result = stats.friedmanchisquare(*(wide[model].to_numpy() for model in models))
    return {
        "statistic": float(result.statistic),
        "p_value": float(result.pvalue),
        **base,
    }


def perform_pairwise_ttests(
    df: pd.DataFrame,
    metric: str = 'mad',
    alpha_01: float = 0.01,
    alpha_05: float = 0.05,
    lower_is_better: bool = True,
    pair_columns: Sequence[str] | None = None,
    inference_unit: str = "dataset",
    n_jobs: int = 1,
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
        inference_unit=inference_unit,
        n_jobs=n_jobs,
    )
    models = sorted(df['model'].dropna().unique())

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


def significance_matrix_from_comparisons(
    comparisons: pd.DataFrame,
    models: Sequence[str],
    alpha_01: float = 0.01,
    alpha_05: float = 0.05,
) -> pd.DataFrame:
    """Build the display matrix from already-computed pairwise results."""
    result = pd.DataFrame(0, index=sorted(models), columns=sorted(models))
    for row in comparisons.itertuples():
        value = 0
        if row.p_holm < alpha_01:
            value = 2 if row.a_better else -2
        elif row.p_holm < alpha_05:
            value = 1 if row.a_better else -1
        result.loc[row.model_a, row.model_b] = value
        result.loc[row.model_b, row.model_a] = -value
    return result


def get_pairwise_pvalues(
    df: pd.DataFrame,
    metric: str = 'mad',
    pair_columns: Sequence[str] | None = None,
    inference_unit: str = "dataset",
    n_jobs: int = 1,
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
        df, metric=metric, pair_columns=pair_columns,
        inference_unit=inference_unit, n_jobs=n_jobs
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
