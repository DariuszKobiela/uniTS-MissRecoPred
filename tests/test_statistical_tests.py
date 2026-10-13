import numpy as np
import pandas as pd

from src.utils.statistical_tests import (
    _holm_adjust,
    _shapiro_wilk,
    friedman_test,
    pairwise_comparisons,
    perform_pairwise_ttests,
)


def _comparison_frame() -> pd.DataFrame:
    rows = []
    scores = {
        "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        "b": [2.2, 3.1, 4.3, 5.0, 6.4, 7.2],
        "c": [3.0, 4.0, 5.0, 6.0, 7.0, 8.0],
    }
    for dataset_index in range(6):
        for rate in (10, 20):
            for model, values in scores.items():
                rows.append(
                    {
                        "dataset_name": f"d{dataset_index}",
                        "technique": "MCAR",
                        "rate_percent": rate,
                        "structure": "scattered",
                        "iteration": 1,
                        "model": model,
                        "mad": values[dataset_index] + rate / 1000,
                    }
                )
    return pd.DataFrame(rows)


def test_holm_adjustment_is_monotone_in_ordered_pvalues():
    adjusted = _holm_adjust([0.01, 0.04, 0.03])
    np.testing.assert_allclose(adjusted, [0.03, 0.06, 0.06])


def test_pairwise_comparisons_match_conditions_but_infer_by_dataset():
    results = pairwise_comparisons(_comparison_frame())
    ab = results.query("model_a == 'a' and model_b == 'b'").iloc[0]

    assert ab["n_exact_pairs"] == 12
    assert ab["n_datasets"] == 6
    assert ab["n_units"] == 6
    assert ab["test"] in {"paired_t", "wilcoxon"}
    assert ab["mean_difference"] < 0
    assert ab["ci_95_high"] < 0
    assert ab["effect_name"] in {"cohen_dz", "rank_biserial"}


def test_significance_matrix_uses_holm_adjusted_pvalues_and_direction():
    matrix = perform_pairwise_ttests(_comparison_frame())
    assert matrix.loc["a", "c"] > 0
    assert matrix.loc["c", "a"] < 0
    assert matrix.loc["a", "a"] == 0


def test_friedman_uses_one_complete_block_per_dataset():
    result = friedman_test(_comparison_frame())
    assert result["n_datasets"] == 6
    assert result["n_units"] == 6
    assert result["n_models"] == 3
    assert result["p_value"] < 0.05


def test_unmatched_conditions_are_not_compared():
    frame = _comparison_frame()
    frame = frame[
        ~(
            (frame["model"] == "b")
            & (frame["dataset_name"] == "d0")
            & (frame["rate_percent"] == 10)
        )
    ]
    result = pairwise_comparisons(frame)
    ab = result.query("model_a == 'a' and model_b == 'b'").iloc[0]
    assert ab["n_exact_pairs"] == 11
    assert ab["n_datasets"] == 6


def test_parallel_pairwise_matches_sequential():
    sequential = pairwise_comparisons(_comparison_frame(), n_jobs=1)
    parallel = pairwise_comparisons(_comparison_frame(), n_jobs=2)
    pd.testing.assert_frame_equal(sequential, parallel)


def test_condition_level_inference_uses_all_matched_blocks():
    frame = _comparison_frame()
    comparisons = pairwise_comparisons(frame, inference_unit="condition")
    ab = comparisons[(comparisons.model_a == "a") & (comparisons.model_b == "b")].iloc[0]
    assert ab["n_datasets"] == 6
    assert ab["n_units"] == 12
    assert ab["n_exact_pairs"] == 12
    assert ab["inference_unit"] == "condition"

    omnibus = friedman_test(frame, inference_unit="condition")
    assert omnibus["n_datasets"] == 6
    assert omnibus["n_units"] == 12
    assert omnibus["inference_unit"] == "condition"


def test_condition_level_averages_iterations_within_experimental_cell():
    frame = _comparison_frame()
    repeated = pd.concat(
        [
            frame.assign(iteration=iteration, mad=frame["mad"] + iteration / 1000)
            for iteration in range(1, 11)
        ],
        ignore_index=True,
    )

    comparisons = pairwise_comparisons(repeated, inference_unit="condition")
    ab = comparisons[(comparisons.model_a == "a") & (comparisons.model_b == "b")].iloc[0]

    assert ab["n_exact_pairs"] == 120
    assert ab["n_units"] == 12

    omnibus = friedman_test(repeated, inference_unit="condition")
    assert omnibus["n_units"] == 12


def test_invalid_inference_unit_is_rejected():
    with np.testing.assert_raises(ValueError):
        pairwise_comparisons(_comparison_frame(), inference_unit="row")


def test_shapiro_uses_reproducible_sample_above_5000():
    values = np.random.default_rng(7).normal(size=6000)
    p_first, n_used, sampled = _shapiro_wilk(values)
    p_second, _, _ = _shapiro_wilk(values)

    assert np.isfinite(p_first)
    assert p_first == p_second
    assert n_used == 5000
    assert sampled is True
