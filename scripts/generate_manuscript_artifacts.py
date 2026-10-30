#!/usr/bin/env python3
"""Generate publication tables, figures, statistics, and a provenance manifest.

The script intentionally reads only the frozen 2026-10-10 merged result files
named below.  It is the single source for quantitative claims in the DMKD
manuscript; no result is copied manually from a dashboard.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from utils.statistical_tests import friedman_test, pairwise_comparisons  # noqa: E402

ARTICLE = ROOT / "LG" / "DataMiningAndKnowledgeDiscovery_v4"
DATA = ARTICLE / "data"
TABLES = ARTICLE / "tables"
METADATA = ARTICLE / "metadata"
FIG = ARTICLE / "figures" / "results"
FIG_SOURCE = ARTICLE / "figures" / "source"

RUN_SUMMARY = ROOT / "experiment_run_summary.json"
MERGE_MANIFEST = ROOT / "merged_results_manifest_20261010_202224.json"
RECONSTRUCTION = ROOT / "reconstruction_experiments_results" / "reconstruction_results_merged_20261010_202224.csv"
PREDICTION = ROOT / "prediction_experiment_results" / "prediction_results_rolling_origins_merged_20261010_202224.csv"
RECONSTRUCTION_SOURCE = ROOT / "data" / "1_cleaned_data"
DEGRADED_TRAIN = ROOT / "data" / "3_missing_data"
SPLIT = ROOT / "data" / "2_splitted_data" / "split_manifest.json"
HORIZONS = ROOT / "data" / "1_5_horizon_recommendation" / "dataset_metadata.json"
MISSINGNESS = ROOT / "data" / "3_missing_data" / "reports" / "missingness_realizations.csv"
ABLATION = ROOT / "data" / "1_6_sd2_optimization" / "sd2_representation_ablations.csv"
SYNTHETIC_PCA = ROOT / "data" / "1_6_sd2_optimization" / "synthetic_real_gap" / "synthetic_real_pca.csv"
SYNTHETIC_CLASSIFIER = ROOT / "data" / "1_6_sd2_optimization" / "synthetic_real_gap" / "synthetic_real_classifier.json"
TRAINING_REPORT = ROOT / "models" / "sd2_windowed" / "training_report.json"
TRAINING_SUMMARY = ROOT / "data" / "sd2_windowed_training" / "dataset_summary.json"
HPO_TRIALS = ROOT / "data" / "1_6_sd2_optimization" / "sd2_inference_trials.csv"
HPO_RECOMMENDATIONS = ROOT / "data" / "1_6_sd2_optimization" / "sd2_recommendations.json"
HPO_OVERRIDES = ROOT / "data" / "1_6_sd2_optimization" / "sd2_runtime_overrides.json"
VIBRATION_TRUTH = ROOT / "data" / "2_splitted_data" / "train" / "vibration_sensor_S1.csv"
VIBRATION_MISSING = ROOT / "data" / "3_missing_data" / "vibration_sensor_S1_MNAR-scattered_5p_1.csv"
VIBRATION_FIXED = ROOT / "data" / "4_fixed_data" / "vibration_sensor_S1" / "MNAR-scattered" / "5p"

EXPECTED_DATASETS = {"boiler_outlet_temp_univ", "pump_sensor_28_univ", "vibration_sensor_S1"}
EXPECTED_MECHANISMS = {"MCAR", "MAR", "MNAR"}
EXPECTED_STRUCTURES = {"scattered", "contiguous", "mixed"}
EXPECTED_RATES = {2, 5, 20, 50}
EXPECTED_RECONSTRUCTORS = {
    "impute_mean", "impute_median", "impute_ffill", "impute_bfill",
    "interpolate_linear", "interpolate_cubic", "interpolate_pchip", "knn", "sarimax",
    "stable_diffusion_2_gaf", "stable_diffusion_2_mtf", "stable_diffusion_2_rp",
    "stable_diffusion_2_spec", "stable_diffusion_2_gaf_finetuned",
    "stable_diffusion_2_mtf_finetuned", "stable_diffusion_2_rp_finetuned",
    "stable_diffusion_2_spec_finetuned",
}
EXPECTED_FORECASTERS = {"holt_winters", "prophet", "sarimax", "xgboost"}

DISPLAY = {
    "original": "Original (no reconstruction)",
    "impute_mean": "Mean", "impute_median": "Median", "impute_ffill": "Forward fill",
    "impute_bfill": "Backward fill", "interpolate_linear": "Linear",
    "interpolate_cubic": "Cubic", "interpolate_pchip": "PCHIP", "knn": "KNN",
    "sarimax": "ARIMA", "stable_diffusion_2_gaf": "GAF",
    "stable_diffusion_2_mtf": "MTF", "stable_diffusion_2_rp": "RP",
    "stable_diffusion_2_spec": "SPEC", "stable_diffusion_2_gaf_finetuned": "GAF-FT",
    "stable_diffusion_2_mtf_finetuned": "MTF-FT", "stable_diffusion_2_rp_finetuned": "RP-FT",
    "stable_diffusion_2_spec_finetuned": "SPEC-FT", "holt_winters": "Holt--Winters",
    "prophet": "Prophet", "xgboost": "XGBoost",
}


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def tex(value: object) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%")


def write_table(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def assert_set(actual: pd.Series, expected: set, label: str) -> None:
    values = set(actual.dropna().unique())
    if values != expected:
        raise ValueError(f"{label}: expected {sorted(expected)}, got {sorted(values)}")


def validate_reconstruction(frame: pd.DataFrame) -> dict:
    keys = ["dataset_name", "technique", "structure", "rate_percent", "iteration", "model"]
    assert len(frame) == 18_360
    assert_set(frame.dataset_name, EXPECTED_DATASETS, "reconstruction datasets")
    assert_set(frame.technique, EXPECTED_MECHANISMS, "reconstruction mechanisms")
    assert_set(frame.structure, EXPECTED_STRUCTURES, "reconstruction structures")
    assert_set(frame.rate_percent, EXPECTED_RATES, "reconstruction rates")
    assert_set(frame.model, EXPECTED_RECONSTRUCTORS, "reconstructors")
    duplicates = int(frame.duplicated(keys).sum())
    cells = frame.groupby(["dataset_name", "technique", "structure", "rate_percent", "model"]).size()
    if duplicates or not cells.eq(10).all():
        raise ValueError(f"Invalid reconstruction grid: duplicates={duplicates}, counts={cells.value_counts().to_dict()}")
    return {"rows": len(frame), "duplicates": duplicates, "replicates_per_cell": 10, "cells_per_model": 108}


def validate_prediction(frame: pd.DataFrame) -> dict:
    assert len(frame) == 2_497_368
    assert_set(frame.prediction_model, EXPECTED_FORECASTERS, "forecasters")
    reconstructed = frame[frame.source_type.eq("reconstructed")]
    assert_set(reconstructed.reconstruction_model, EXPECTED_RECONSTRUCTORS, "prediction reconstructors")
    assert_set(reconstructed.dataset_name, EXPECTED_DATASETS, "prediction datasets")
    fallback = reconstructed.forecast_fallback_used.fillna(False).astype(bool)
    return {
        "rows": len(frame),
        "reconstructed_rows": len(reconstructed),
        "original_rows": int(frame.source_type.eq("original").sum()),
        "cumulative_rows": int(frame.metric_scope.eq("cumulative").sum()),
        "lead_time_rows": int(frame.metric_scope.eq("lead_time_bin").sum()),
        "fallback_rows": int(fallback.sum()),
    }


def hpo_audit() -> dict:
    result = {"override_exists": HPO_OVERRIDES.is_file(), "valid_for_publication": False, "reason": "missing"}
    if not HPO_TRIALS.is_file() or not HPO_RECOMMENDATIONS.is_file() or not HPO_OVERRIDES.is_file():
        return result
    trials = pd.read_csv(HPO_TRIALS)
    recs = pd.DataFrame(read_json(HPO_RECOMMENDATIONS).get("recommendations", []))
    finite = bool(len(recs) == 12 and np.isfinite(pd.to_numeric(recs.get("objective"), errors="coerce")).all())
    successful = trials[trials.status.eq("success")].groupby(["dataset", "model"]).size()
    complete = len(successful) == 12 and successful.gt(0).all()
    result.update({
        "recommendations": len(recs), "all_objectives_finite": finite,
        "dataset_model_groups_with_success": len(successful), "valid_for_publication": finite and complete,
        "reason": "complete" if finite and complete else "incomplete empirical validation",
    })
    return result


def add_missing_reconstruction_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    """Recover RMSE and R2 omitted from the frozen reconstruction CSV.

    RMSE follows exactly from the stored mean and population standard deviation
    of absolute errors. R2 additionally needs the total sum of squares of the
    ground-truth values at each realization's missing positions.
    """
    enriched = frame.copy()
    if "rmse" not in enriched:
        enriched["rmse"] = np.sqrt(enriched["mae"] ** 2 + enriched["std_diff"] ** 2)
    if "r2" in enriched:
        return enriched

    realization_keys = [
        "dataset_name", "technique", "structure", "rate_percent", "iteration"
    ]
    realizations = enriched[realization_keys].drop_duplicates().sort_values(realization_keys)
    source_cache: dict[str, pd.Series] = {}
    sst_rows: list[dict] = []
    for row in realizations.itertuples(index=False):
        if row.dataset_name not in source_cache:
            source = pd.read_csv(
                RECONSTRUCTION_SOURCE / f"{row.dataset_name}.csv", index_col=0
            )
            source_values = pd.to_numeric(source.iloc[:, 0], errors="raise")
            source_values.index = source_values.index.astype(str)
            source_cache[row.dataset_name] = source_values
        truth = source_cache[row.dataset_name]
        degraded_path = DEGRADED_TRAIN / (
            f"{row.dataset_name}_{row.technique}-{row.structure}_"
            f"{int(row.rate_percent)}p_{int(row.iteration)}.csv"
        )
        degraded = pd.read_csv(degraded_path, index_col=0)
        degraded_values = pd.to_numeric(degraded.iloc[:, 0], errors="coerce")
        degraded_values.index = degraded_values.index.astype(str)
        matching_truth = truth.iloc[: len(degraded_values)]
        if not matching_truth.index.equals(degraded_values.index):
            raise ValueError(f"Index mismatch while deriving R2: {degraded_path}")
        mask = degraded_values.isna().to_numpy()
        target = matching_truth.to_numpy(dtype=np.float64)[mask]
        if len(target) < 2:
            target_sst = np.nan
        else:
            target_sst = float(np.sum((target - target.mean()) ** 2))
        sst_rows.append(
            {
                **{key: getattr(row, key) for key in realization_keys},
                "target_sst": target_sst,
                "derived_n_missing": len(target),
            }
        )

    sst = pd.DataFrame(sst_rows)
    enriched = enriched.merge(sst, on=realization_keys, how="left", validate="many_to_one")
    if not np.allclose(
        enriched["n_missing"], enriched["derived_n_missing"], equal_nan=False
    ):
        raise ValueError("Stored and derived missing-value counts differ")
    sse = enriched["n_missing"] * enriched["rmse"] ** 2
    enriched["r2"] = np.where(
        enriched["target_sst"] > 1e-12,
        1.0 - sse / enriched["target_sst"],
        np.nan,
    )
    return enriched.drop(columns=["target_sst", "derived_n_missing"])


def balanced_reconstruction(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = add_missing_reconstruction_metrics(frame)
    cell_keys = ["dataset_name", "technique", "structure", "rate_percent", "model"]
    cells = frame.groupby(cell_keys, as_index=False).agg(
        smape=("smape", "mean"), mae=("mae", "mean"), mad=("mad", "mean"),
        rmse=("rmse", "mean"), r2=("r2", "mean")
    )
    ranking = cells.groupby("model", as_index=False).agg(
        smape=("smape", "mean"), mae=("mae", "mean"), mad=("mad", "mean"),
        rmse=("rmse", "mean"), r2=("r2", "mean")
    ).sort_values("smape")
    ranking["rank"] = np.arange(1, len(ranking) + 1)
    ranking["method"] = ranking.model.map(DISPLAY)
    return cells, ranking


def balanced_prediction(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    cumulative = frame[frame.source_type.eq("reconstructed") & frame.metric_scope.eq("cumulative")].copy()
    # First average correlated repeated evaluations within a reconstruction realization.
    realization_keys = ["prediction_model", "reconstruction_model", "dataset_name", "technique",
                        "structure", "rate_percent", "reconstruction_iteration"]
    realization = cumulative.groupby(realization_keys, as_index=False).agg(
        smape=("smape", "mean"), mae=("mae", "mean"), rmse=("rmse", "mean"),
        mase=("mase", "mean"), mape=("mape", "mean")
    )
    # Then average ten missingness realizations: one inferential cell remains.
    cell_keys = ["prediction_model", "reconstruction_model", "dataset_name", "technique", "structure", "rate_percent"]
    cells = realization.groupby(cell_keys, as_index=False).agg(
        smape=("smape", "mean"), mae=("mae", "mean"), rmse=("rmse", "mean"),
        mase=("mase", "mean"), mape=("mape", "mean")
    )
    ranking = cells.groupby(["prediction_model", "reconstruction_model"], as_index=False).agg(
        smape=("smape", "mean"), mae=("mae", "mean"), rmse=("rmse", "mean"),
        mase=("mase", "mean"), mape=("mape", "mean")
    )
    ranking["rank"] = ranking.groupby("prediction_model").smape.rank(method="min").astype(int)

    # A complementary within-evaluation rank is scale-free across heterogeneous horizons.
    exact = ["prediction_model", "dataset_name", "technique", "structure", "rate_percent",
             "reconstruction_iteration", "origin", "forecast_horizon"]
    ranked = cumulative.copy()
    ranked["paired_rank"] = ranked.groupby(exact).smape.rank(method="average")
    paired_rank = ranked.groupby(["prediction_model", "reconstruction_model"], as_index=False).paired_rank.mean()
    paired_rank["paired_rank_position"] = paired_rank.groupby("prediction_model").paired_rank.rank(method="min").astype(int)
    ranking = ranking.merge(paired_rank, on=["prediction_model", "reconstruction_model"])
    return cumulative, cells, ranking


def original_prediction_benchmark(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Balance the complete-history reference over datasets without fake cells."""
    original = frame[
        frame.source_type.eq("original") & frame.metric_scope.eq("cumulative")
    ].copy()
    by_dataset = original.groupby(
        ["prediction_model", "dataset_name"], as_index=False
    ).agg(
        smape=("smape", "mean"), mae=("mae", "mean"), rmse=("rmse", "mean"),
        mase=("mase", "mean"), mape=("mape", "mean")
    )
    global_reference = by_dataset.groupby("prediction_model", as_index=False).agg(
        smape=("smape", "mean"), mae=("mae", "mean"), rmse=("rmse", "mean"),
        mase=("mase", "mean"), mape=("mape", "mean")
    )
    global_reference["reconstruction_model"] = "original"
    return by_dataset, global_reference


def statistical_exports(recon_cells: pd.DataFrame, pred_cells: pd.DataFrame) -> dict:
    recon_for_stats = recon_cells.rename(columns={"model": "model"})
    recon_pairs = pairwise_comparisons(
        recon_for_stats, metric="smape",
        pair_columns=["dataset_name", "technique", "structure", "rate_percent"],
        inference_unit="condition", n_jobs=1,
    )
    recon_omnibus = friedman_test(
        recon_for_stats, metric="smape",
        pair_columns=["dataset_name", "technique", "structure", "rate_percent"],
        inference_unit="condition",
    )

    pred_for_stats = pred_cells.rename(columns={"prediction_model": "model"})
    pred_pairs = pairwise_comparisons(
        pred_for_stats, metric="smape",
        pair_columns=["dataset_name", "technique", "structure", "rate_percent", "reconstruction_model"],
        inference_unit="condition", n_jobs=1,
    )
    pred_omnibus = friedman_test(
        pred_for_stats, metric="smape",
        pair_columns=["dataset_name", "technique", "structure", "rate_percent", "reconstruction_model"],
        inference_unit="condition",
    )
    recon_pairs.to_csv(DATA / "reconstruction_pairwise_smape.csv", index=False)
    pred_pairs.to_csv(DATA / "forecast_algorithms_pairwise_smape.csv", index=False)
    forecast_lines = [
        r"\begin{table}[t]", r"\caption{Post-hoc forecasting-algorithm comparisons on 1,836 paired cells.}",
        r"\label{tab:forecast-posthoc}", r"\centering\small",
        r"\begin{tabular}{llrrrr}", r"\toprule",
        r"Algorithm A & Algorithm B & Test & Holm $p$ & Effect & 95\% CI \\",
        r"\midrule",
    ]
    for row in pred_pairs.itertuples():
        test = "T" if row.test == "paired_t" else "W"
        forecast_lines.append(
            f"{tex(DISPLAY[row.model_a])} & {tex(DISPLAY[row.model_b])} & {test} & "
            f"{row.p_holm:.2e} & {row.effect_size:.3f} & [{row.ci_95_low:.3f}, {row.ci_95_high:.3f}] " + r"\\"
        )
    forecast_lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\footnotesize T: paired $t$-test; W: Wilcoxon signed-rank. Effect is Cohen $d_z$ for T and rank-biserial correlation for W. The original-history benchmark is excluded because it has no missingness or reconstruction cells.",
        r"\end{table}",
    ]
    write_table(TABLES / "forecast_pairwise_smape.tex", forecast_lines)

    reconstruction_lines = [
        r"\begin{longtable}{llrrrr}", r"\caption{All reconstruction post-hoc comparisons.}\label{tab:reconstruction-posthoc}\\",
        r"\toprule", r"Method A & Method B & Test & Holm $p$ & Effect & 95\% CI \\",
        r"\midrule\endfirsthead", r"\toprule Method A & Method B & Test & Holm $p$ & Effect & 95\% CI \\",
        r"\midrule\endhead",
    ]
    for row in recon_pairs.itertuples():
        test = "T" if row.test == "paired_t" else "W"
        reconstruction_lines.append(
            f"{tex(DISPLAY[row.model_a])} & {tex(DISPLAY[row.model_b])} & {test} & "
            f"{row.p_holm:.2e} & {row.effect_size:.3f} & [{row.ci_95_low:.3f}, {row.ci_95_high:.3f}] " + r"\\"
        )
    reconstruction_lines += [
        r"\bottomrule", r"\end{longtable}",
        r"{\footnotesize T: paired $t$-test; W: Wilcoxon signed-rank. Effect is Cohen $d_z$ for T and rank-biserial correlation for W.\par}",
    ]
    write_table(TABLES / "reconstruction_pairwise_smape.tex", reconstruction_lines)
    (METADATA / "reconstruction_friedman_smape.json").write_text(json.dumps(recon_omnibus, indent=2), encoding="utf-8")
    (METADATA / "forecast_algorithms_friedman_smape.json").write_text(json.dumps(pred_omnibus, indent=2), encoding="utf-8")
    return {
        "reconstruction": {"omnibus": recon_omnibus, "pairwise_rows": len(recon_pairs),
                           "normality_rejected": int((recon_pairs.shapiro_p < 0.05).sum()),
                           "normality_not_rejected": int((recon_pairs.shapiro_p >= 0.05).sum())},
        "forecast_algorithms": {"omnibus": pred_omnibus, "pairwise_rows": len(pred_pairs),
                                "normality_rejected": int((pred_pairs.shapiro_p < 0.05).sum()),
                                "normality_not_rejected": int((pred_pairs.shapiro_p >= 0.05).sum())},
    }


def design_table() -> None:
    split = read_json(SPLIT)["datasets"]
    meta = read_json(HORIZONS)["series"]
    lines = [
        r"\begin{table}[t]", r"\caption{Temporal splits and rolling-origin design.}", r"\label{tab:split-design}",
        r"\centering\small", r"\begin{tabularx}{\textwidth}{lrrrrX}", r"\toprule",
        r"Dataset & Reconstruction & SD2 val. & Test & Origins & Sampling and horizons \\", r"\midrule",
    ]
    for filename in ["boiler_outlet_temp_univ.csv", "pump_sensor_28_univ.csv", "vibration_sensor_S1.csv"]:
        s, m = split[filename], meta[filename]
        name = filename.split("_")[0].title() if not filename.startswith("vibration") else "Vibration"
        horizons = ", ".join(map(str, s["horizons"]))
        lines.append(f"{name} & {s['reconstruction_train']['length']:,} & {s['sd2_validation']['length']:,} & "
                     f"{s['rolling_test']['length']:,} & {s['n_origins']} & {tex(m['sampling_label'])}; $h={horizons}$ " + r"\\")
    lines += [r"\bottomrule", r"\end{tabularx}",
              r"\footnotesize Vibration is irregular; its horizons denote events rather than a fixed elapsed duration.",
              r"\end{table}"]
    write_table(TABLES / "split_design.tex", lines)


def _format_r2_for_tex(value: float) -> str:
    if not np.isfinite(value):
        return "--"
    if abs(value) < 1_000:
        return f"{value:.2f}"
    mantissa, exponent = f"{value:.2e}".split("e")
    return rf"${mantissa}\times 10^{{{int(exponent)}}}$"


def write_reconstruction_ranking_table(recon_ranking: pd.DataFrame) -> None:
    lines = [r"\begin{table}[t]", r"\caption{Global reconstruction ranking with all five reconstruction metrics after balancing the 108 experimental cells.}",
             r"\label{tab:reconstruction-ranking}", r"\centering\scriptsize",
             r"\begin{tabular}{rlrrrrr}", r"\toprule", r"Rank & Method & sMAPE (\%) & MAE & MAD & RMSE & $R^2$ \\", r"\midrule"]
    for row in recon_ranking.itertuples():
        lines.append(
            f"{row.rank} & {tex(row.method)} & {row.smape:.2f} & {row.mae:.2f} & "
            f"{row.mad:.2f} & {row.rmse:.2f} & {_format_r2_for_tex(row.r2)} " + r"\\"
        )
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\footnotesize Each dataset $\times$ mechanism $\times$ structure $\times$ rate cell has equal weight; ten realizations are averaged first. Lower is better for sMAPE, MAE, MAD, and RMSE; higher is better for $R^2$. Negative $R^2$ is worse than predicting the missing-position mean. Very large negative values arise in realizations with low target variance and large reconstruction errors.",
              r"\end{table}"]
    write_table(TABLES / "reconstruction_ranking.tex", lines)


def ranking_tables(
    recon_ranking: pd.DataFrame,
    pred_ranking: pd.DataFrame,
    original_global: pd.DataFrame,
) -> None:
    write_reconstruction_ranking_table(recon_ranking)

    pivot = pred_ranking.pivot(index="reconstruction_model", columns="prediction_model", values="smape")
    ranks = pred_ranking.pivot(index="reconstruction_model", columns="prediction_model", values="rank")
    order = pred_ranking.groupby("reconstruction_model").smape.mean().sort_values().index
    original = original_global.set_index("prediction_model").smape
    forecast_order = ["holt_winters", "prophet", "sarimax", "xgboost"]
    lines = [r"\begin{table*}[t]", r"\caption{Balanced forecasting sMAPE (\%) by forecaster and training-history variant. Parentheses give ranks among the 17 reconstructed histories; the complete original history is an unranked reference.}",
             r"\label{tab:forecast-matrix}", r"\centering\scriptsize", r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Training history & Holt--Winters & Prophet & ARIMA & XGBoost \\", r"\midrule",
             "Original (no reconstruction) & " + " & ".join(
                 f"{original[model]:.3f} (ref.)" for model in forecast_order
             ) + r" \\", r"\midrule"]
    for model in order:
        vals = [f"{pivot.loc[model, p]:.3f} ({int(ranks.loc[model, p])})" for p in forecast_order]
        lines.append(f"{tex(DISPLAY[model])} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\footnotesize Reconstructed histories are balanced over 108 missingness cells after averaging origins, cumulative horizons, and ten realizations. The original-history reference is first averaged over origins and horizons within each dataset and then equally over the three datasets; it is not replicated across missingness cells.",
              r"\end{table*}"]
    write_table(TABLES / "forecast_matrix.tex", lines)


def condition_appendix(cells: pd.DataFrame) -> None:
    pivot = cells.pivot(index=["dataset_name", "technique", "structure", "rate_percent"], columns="model", values="smape")
    models = sorted(EXPECTED_RECONSTRUCTORS, key=lambda x: ("stable_diffusion" in x, DISPLAY[x]))
    pivot = pivot[models]
    pivot.to_csv(DATA / "reconstruction_108_conditions_smape.csv")
    groups = [
        ("classical", [m for m in models if not m.startswith("stable_diffusion")]),
        ("SD2", [m for m in models if m.startswith("stable_diffusion")]),
    ]
    lines = [r"\begin{landscape}", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}"]
    for part_number, (label, selected) in enumerate(groups):
        table_label = r"\label{tab:appendix-108}" if part_number == 0 else ""
        lines += [
            r"\begin{longtable}{llll" + "r" * len(selected) + "}",
            rf"\caption{{Complete reconstruction sMAPE matrix: {label} methods.}}{table_label}\\",
            r"\toprule", "Dataset & Mechanism & Structure & Rate & " + " & ".join(tex(DISPLAY[m]) for m in selected) + r" \\",
            r"\midrule\endfirsthead", r"\toprule",
            "Dataset & Mechanism & Structure & Rate & " + " & ".join(tex(DISPLAY[m]) for m in selected) + r" \\",
            r"\midrule\endhead",
        ]
        for idx, row in pivot[selected].iterrows():
            dataset, mechanism, structure, rate = idx
            short = {"boiler_outlet_temp_univ": "Boiler", "pump_sensor_28_univ": "Pump", "vibration_sensor_S1": "Vibration"}[dataset]
            vals = " & ".join(f"{v:.2f}" for v in row)
            lines.append(f"{short} & {mechanism} & {structure} & {rate}\\% & {vals} " + r"\\")
        lines += [r"\bottomrule", r"\end{longtable}", r"\clearpage"]
    lines += [r"\end{landscape}"]
    write_table(TABLES / "appendix_reconstruction_108_conditions.tex", lines)


def interaction_table(cells: pd.DataFrame) -> None:
    lines = [r"\begin{table*}[t]", r"\caption{Best balanced reconstruction result within each experimental axis.}",
             r"\label{tab:reconstruction-interactions}", r"\centering\small", r"\begin{tabular}{lllrr}", r"\toprule",
             r"Axis & Level & Best method & sMAPE (\%) & Runner-up \\", r"\midrule"]
    axes = [("Dataset", "dataset_name"), ("Mechanism", "technique"), ("Structure", "structure"), ("Rate", "rate_percent")]
    for axis, col in axes:
        for value, part in cells.groupby(col, sort=True):
            summary = part.groupby("model").smape.mean().sort_values()
            first, second = summary.index[:2]
            level = f"{value}\\%" if col == "rate_percent" else tex(value)
            lines.append(f"{axis} & {level} & {tex(DISPLAY[first])} & {summary.iloc[0]:.3f} & {tex(DISPLAY[second])} ({summary.iloc[1]:.3f}) " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    write_table(TABLES / "reconstruction_interactions.tex", lines)


def ablation_table() -> dict:
    frame = pd.read_csv(ABLATION)
    summary = frame.groupby(["ablation", "encoding"], as_index=False).agg(smape=("smape", "mean"), rmse=("rmse", "mean"))
    lines = [r"\begin{table}[t]", r"\caption{Transform-only controls averaged across datasets, windows, resolutions, and masks.}",
             r"\label{tab:transform-controls}", r"\centering\small", r"\begin{tabular}{llrr}", r"\toprule",
             r"Control & Encoding & sMAPE (\%) & RMSE \\", r"\midrule"]
    for row in summary.sort_values(["ablation", "smape"]).itertuples():
        lines.append(f"{tex(row.ablation)} & {row.encoding.upper()} & {row.smape:.3f} & {row.rmse:.3f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_table(TABLES / "transform_controls.tex", lines)
    return summary.to_dict(orient="records")


def forecast_diagnostics(frame: pd.DataFrame) -> dict:
    cumulative = frame[frame.metric_scope.eq("cumulative")].copy()
    reconstructed = cumulative[cumulative.source_type.eq("reconstructed")].copy()
    original = cumulative[cumulative.source_type.eq("original")].copy()

    original_summary = original.groupby(["prediction_model", "dataset_name"], as_index=False).agg(
        smape=("smape", "mean"), mae=("mae", "mean"), rmse=("rmse", "mean"),
        mase=("mase", "mean"), mape=("mape", "mean"),
    )
    original_summary.to_csv(DATA / "forecast_original_reference.csv", index=False)

    horizon = reconstructed.groupby(
        ["prediction_model", "reconstruction_model", "dataset_name", "forecast_horizon"], as_index=False
    ).agg(smape=("smape", "mean"), mae=("mae", "mean"), rmse=("rmse", "mean"),
          mase=("mase", "mean"), mape=("mape", "mean"))
    horizon["family"] = np.where(horizon.reconstruction_model.str.startswith("stable_diffusion"), "SD2", "classical")
    horizon.to_csv(DATA / "forecast_horizon_balanced.csv", index=False)

    horizon_best = horizon.groupby(
        ["dataset_name", "forecast_horizon", "prediction_model", "family"], as_index=False
    ).smape.min()
    original_horizon = original.groupby(
        ["dataset_name", "forecast_horizon", "prediction_model"], as_index=False
    ).agg(smape=("smape", "mean"))
    original_horizon["family"] = "original history"
    horizon_best = pd.concat(
        [horizon_best, original_horizon], ignore_index=True, sort=False
    )
    horizon_best.to_csv(DATA / "forecast_horizon_best_family.csv", index=False)

    lead = frame[frame.source_type.eq("reconstructed") & frame.metric_scope.eq("lead_time_bin")].groupby(
        ["prediction_model", "reconstruction_model", "dataset_name", "lead_time_start", "lead_time_end"],
        as_index=False,
    ).agg(smape=("smape", "mean"), mae=("mae", "mean"), rmse=("rmse", "mean"),
          mase=("mase", "mean"), mape=("mape", "mean"))
    lead.to_csv(DATA / "forecast_lead_time_balanced.csv", index=False)

    origin = reconstructed.groupby(
        ["prediction_model", "reconstruction_model", "dataset_name", "origin"], as_index=False
    ).smape.mean()
    origin["rank"] = origin.groupby(["prediction_model", "dataset_name", "origin"]).smape.rank(method="average")
    stability = origin.groupby(["prediction_model", "reconstruction_model"], as_index=False).agg(
        mean_rank=("rank", "mean"), rank_sd=("rank", "std"), origin_evaluations=("rank", "size")
    )
    stability.to_csv(DATA / "forecast_origin_rank_stability.csv", index=False)

    axes = []
    for column in ["dataset_name", "technique", "structure", "rate_percent"]:
        summary = reconstructed.groupby([column, "prediction_model", "reconstruction_model"], as_index=False).smape.mean()
        winners = summary.loc[summary.groupby([column, "prediction_model"]).smape.idxmin()].copy()
        winners.insert(0, "axis", column)
        winners.rename(columns={column: "level"}, inplace=True)
        axes.append(winners[["axis", "level", "prediction_model", "reconstruction_model", "smape"]])
    pd.concat(axes, ignore_index=True).to_csv(DATA / "forecast_axis_winners.csv", index=False)

    fig, axes_plot = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    names = {"boiler_outlet_temp_univ": "Boiler", "pump_sensor_28_univ": "Pump", "vibration_sensor_S1": "Vibration"}
    for ax, dataset in zip(axes_plot, names):
        part = horizon_best[horizon_best.dataset_name.eq(dataset)].groupby(
            ["forecast_horizon", "family"], as_index=False
        ).smape.mean()
        for family, values in part.groupby("family"):
            linestyle = "--" if family == "original history" else "-"
            marker = "s" if family == "original history" else "o"
            ax.plot(
                values.forecast_horizon.astype(str), values.smape,
                marker=marker, linestyle=linestyle, label=family,
            )
        ax.set_title(names[dataset])
        ax.set_xlabel("Horizon (samples)")
        ax.tick_params(axis="x", rotation=40)
    axes_plot[0].set_ylabel("Best-family sMAPE (%)")
    axes_plot[-1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "forecast_horizon_families.pdf", bbox_inches="tight")
    fig.savefig(FIG / "09_horizon_classical_vs_sd2.png", bbox_inches="tight", dpi=200)
    plt.close(fig)

    return {
        "original_reference_rows": len(original_summary),
        "horizon_summary_rows": len(horizon),
        "lead_time_summary_rows": len(lead),
        "origin_stability_rows": len(stability),
    }


def reconstruction_ranking_figure(recon_cells: pd.DataFrame) -> None:
    """Plot reconstructors with the complete-history zero-error reference."""
    plot_display = {
        "interpolate_linear": "Linear interpolation",
        "sarimax": "ARIMA",
        "stable_diffusion_2_mtf_finetuned": "MTF (fine-tuned)",
        "stable_diffusion_2_mtf": "MTF (original)",
        "interpolate_pchip": "PCHIP",
        "impute_bfill": "Backward fill",
        "impute_ffill": "Forward fill",
        "stable_diffusion_2_rp_finetuned": "RP (fine-tuned)",
        "stable_diffusion_2_rp": "RP (original)",
        "stable_diffusion_2_gaf_finetuned": "GAF (fine-tuned)",
        "stable_diffusion_2_gaf": "GAF (original)",
        "impute_median": "Median",
        "interpolate_cubic": "Cubic interpolation",
        "impute_mean": "Mean",
        "knn": "KNN",
        "stable_diffusion_2_spec": "Spectrogram (original)",
        "stable_diffusion_2_spec_finetuned": "Spectrogram (fine-tuned)",
    }
    ranking = recon_cells.groupby("model")["smape"].mean().sort_values()
    labels = ["Original (complete history)"] + [plot_display[model] for model in ranking.index]
    values = np.r_[0.0, ranking.to_numpy()]
    colors = ["#2ca02c"] + [
        "#f58518" if model.startswith("stable_diffusion") else "#4c78a8"
        for model in ranking.index
    ]

    fig, ax = plt.subplots(figsize=(8, 7.2), dpi=150)
    positions = np.arange(len(values))
    ax.barh(positions, values, color=colors, height=0.78)
    ax.scatter([0], [0], color="#2ca02c", marker="D", s=58, zorder=4)
    ax.text(1.2, 0, "0.000 (reference)", color="#1b7f1b", va="center", fontsize=8)
    ax.axhline(0.5, color="#777777", linewidth=1.0, alpha=0.45)
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlim(-2, max(values) * 1.05)
    ax.set_xticks([0, 20, 40, 60, 80])
    ax.set_xlabel("balanced sMAPE [%]")
    ax.set_title(
        "Reconstruction quality with complete-history reference — full range\n"
        "(lower is better)",
        fontweight="bold",
    )
    ax.grid(axis="x", alpha=0.22)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(FIG / "01_reconstruction_ranking.png", bbox_inches="tight")
    plt.close(fig)


def figures(
    recon_cells: pd.DataFrame,
    pred_ranking: pd.DataFrame,
    original_global: pd.DataFrame,
) -> None:
    plt.rcParams.update({"font.size": 8, "figure.dpi": 150})

    reconstruction_ranking_figure(recon_cells)

    # Reconstruction heat map by principal experimental axes.
    facets = []
    for col in ["dataset_name", "technique", "structure", "rate_percent"]:
        part = recon_cells.groupby(["model", col]).smape.mean().unstack(col)
        part.columns = [str(c) for c in part.columns]
        facets.append(part)
    matrix = pd.concat(facets, axis=1)
    order = recon_cells.groupby("model").smape.mean().sort_values().index
    matrix = matrix.loc[order]
    fig, ax = plt.subplots(figsize=(12, 6))
    image = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="viridis_r")
    ax.set_yticks(range(len(matrix)), [DISPLAY[m] for m in matrix.index])
    ax.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=45, ha="right")
    ax.set_xlabel("Dataset, mechanism, structure, and missingness-rate facets")
    ax.set_title("Balanced reconstruction sMAPE (lower is better)")
    fig.colorbar(image, ax=ax, label="sMAPE (%)")
    fig.tight_layout()
    fig.savefig(FIG / "reconstruction_rank_heatmap.pdf", bbox_inches="tight")
    plt.close(fig)

    # Forecaster x reconstructor matrix.
    pivot = pred_ranking.pivot(index="reconstruction_model", columns="prediction_model", values="smape")
    order = pred_ranking.groupby("reconstruction_model").smape.mean().sort_values().index
    pivot = pivot.loc[order, ["holt_winters", "prophet", "sarimax", "xgboost"]]
    original = original_global.set_index("prediction_model").smape
    pivot = pd.concat(
        [
            pd.DataFrame(
                [[original[column] for column in pivot.columns]],
                index=["original"], columns=pivot.columns,
            ),
            pivot,
        ]
    )
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="magma_r")
    ax.set_yticks(range(len(pivot)), [DISPLAY[m] for m in pivot.index])
    ax.set_xticks(range(4), ["Holt--Winters", "Prophet", "SARIMAX", "XGBoost"], rotation=20, ha="right")
    ax.axhline(0.5, color="white", linewidth=2)
    for i in range(len(pivot)):
        for j in range(4):
            ax.text(j, i, f"{pivot.iloc[i, j]:.1f}", ha="center", va="center", fontsize=6,
                    color="white" if pivot.iloc[i, j] > pivot.to_numpy().mean() else "black")
    ax.set_title("Forecasting sMAPE by training-history variant")
    fig.colorbar(image, ax=ax, label="sMAPE (%)")
    fig.tight_layout()
    fig.savefig(FIG / "forecast_reconstruction_heatmap.pdf", bbox_inches="tight")
    fig.savefig(FIG / "06_reconstruction_by_forecasting_method.png", bbox_inches="tight", dpi=200)
    plt.close(fig)

    # Missingness topology.
    missing = pd.read_csv(MISSINGNESS)
    summary = missing.groupby("structure").agg(mean_gap=("gap_length_samples_mean", "mean"),
                                                singleton=("singleton_gap_percent", "mean"),
                                                gaps=("n_gaps", "mean")).reindex(["scattered", "mixed", "contiguous"])
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.8))
    for ax, column, title in zip(axes, ["mean_gap", "singleton", "gaps"],
                                 ["Mean gap length", "Singleton gaps (%)", "Number of gaps"]):
        ax.bar(summary.index, summary[column], color=["#4c78a8", "#f2cf5b", "#e45756"])
        ax.set_title(title)
        ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(FIG / "missingness_topology.pdf", bbox_inches="tight")
    plt.close(fig)

    # Synthetic-to-real PCA.
    pca = pd.read_csv(SYNTHETIC_PCA)
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for kind, part in pca.groupby("source_kind"):
        ax.scatter(part.pca_1, part.pca_2, s=9, alpha=0.45, label=kind.title())
    ax.set(xlabel="PC1", ylabel="PC2", title="Synthetic-to-real feature space")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "synthetic_real_pca.pdf", bbox_inches="tight")
    plt.close(fig)

    # Training curve.
    history = pd.DataFrame(read_json(TRAINING_REPORT)["history"])
    fig, ax = plt.subplots(figsize=(5.5, 3.5))
    ax.plot(history.epoch, history.train_loss, marker="o", label="Train")
    ax.plot(history.epoch, history.validation_loss, marker="o", label="Validation")
    ax.axvline(3, color="black", linestyle="--", linewidth=0.8, label="Best epoch")
    ax.set(xlabel="Epoch", ylabel="Loss", title="SD2 fine-tuning history")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "finetuning_loss.pdf", bbox_inches="tight")
    plt.close(fig)

    # Original, masked, and absolute image-space difference for all encodings.
    examples = {
        "GAF": ("000017_gaf_original.png", "000017_gaf_missing.png"),
        "MTF": ("000015_mtf_original.png", "000015_mtf_missing.png"),
        "RP": ("000007_rp_original.png", "000007_rp_missing.png"),
        "SPEC": ("000046_spec_original.png", "000046_spec_missing.png"),
    }
    fig, axes = plt.subplots(4, 3, figsize=(7.2, 9))
    for row, (encoding, (original_name, missing_name)) in enumerate(examples.items()):
        original_image = plt.imread(FIG_SOURCE / original_name)[..., :3]
        missing_image = plt.imread(FIG_SOURCE / missing_name)[..., :3]
        difference = np.abs(original_image.astype(float) - missing_image.astype(float)).mean(axis=2)
        for col, (image_data, title) in enumerate([
            (original_image, "Original"), (missing_image, "Missing"), (difference, "Absolute difference")
        ]):
            axes[row, col].imshow(image_data, cmap="inferno" if col == 2 else None)
            axes[row, col].set_title(f"{encoding}: {title}")
            axes[row, col].axis("off")
    fig.tight_layout()
    fig.savefig(FIG / "representation_missing_difference.pdf", bbox_inches="tight")
    plt.close(fig)

    # Schematic induced image masks: a short 1-D gap becomes nonlocal for pairwise encodings.
    n = 64
    one_dimensional = np.zeros(n)
    one_dimensional[27:32] = 1
    pairwise = np.logical_or(one_dimensional[:, None], one_dimensional[None, :])
    local_columns = np.repeat(one_dimensional[None, :], 32, axis=0)
    fig, axes = plt.subplots(1, 3, figsize=(8.5, 2.7))
    axes[0].step(np.arange(n), one_dimensional, where="mid")
    axes[0].set(xlabel="Time index", ylabel="Missing", title="1-D gap")
    axes[1].imshow(pairwise, cmap="Greys", origin="lower", aspect="auto")
    axes[1].set(title="GAF/MTF/RP row--column mask", xlabel="Time", ylabel="Time")
    axes[2].imshow(local_columns, cmap="Greys", origin="lower", aspect="auto")
    axes[2].set(title="SPEC time-column mask", xlabel="Time", ylabel="Frequency")
    fig.tight_layout()
    fig.savefig(FIG / "representation_mask_locality.pdf", bbox_inches="tight")
    plt.close(fig)

    # Ground truth is overlaid on every representation-specific reconstruction.
    truth = pd.read_csv(VIBRATION_TRUTH).iloc[:, 1].to_numpy(dtype=float)
    missing_values = pd.read_csv(VIBRATION_MISSING).iloc[:, 1]
    missing_positions = missing_values.isna().to_numpy()
    model_groups = [
        ("zero_shot", ["stable_diffusion_2_gaf", "stable_diffusion_2_mtf",
                       "stable_diffusion_2_rp", "stable_diffusion_2_spec"]),
        ("finetuned", ["stable_diffusion_2_gaf_finetuned", "stable_diffusion_2_mtf_finetuned",
                       "stable_diffusion_2_rp_finetuned", "stable_diffusion_2_spec_finetuned"]),
    ]
    for suffix, models in model_groups:
        fig, axes = plt.subplots(2, 2, figsize=(10, 5), sharex=True, sharey=True)
        for ax, model in zip(axes.flat, models):
            reconstructed = pd.read_csv(VIBRATION_FIXED / model / "1.csv").iloc[:, 1].to_numpy(dtype=float)
            label = DISPLAY[model]
            ax.plot(truth, color="black", linewidth=1.0, label="Ground truth")
            ax.plot(reconstructed, color="#4c78a8", linewidth=0.9, alpha=0.9, label=label)
            ax.scatter(np.flatnonzero(missing_positions), reconstructed[missing_positions],
                       color="#e45756", s=18, zorder=3, label="Reconstructed missing")
            ax.set_title(label)
            ax.grid(alpha=0.15)
        handles, labels = axes.flat[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
        fig.supxlabel("Sample")
        fig.supylabel("Vibration (Hz)")
        fig.tight_layout(rect=(0, 0, 1, .92))
        fig.savefig(FIG / f"vibration_reconstruction_{suffix}.pdf", bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    (DATA).mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)
    METADATA.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    run_summary, merge_manifest = read_json(RUN_SUMMARY), read_json(MERGE_MANIFEST)
    reconstruction = pd.read_csv(RECONSTRUCTION)
    prediction = pd.read_csv(PREDICTION, low_memory=False)

    validation = {
        "reconstruction": validate_reconstruction(reconstruction),
        "prediction": validate_prediction(prediction),
    }
    recon_cells, recon_ranking = balanced_reconstruction(reconstruction)
    cumulative, pred_cells, pred_ranking = balanced_prediction(prediction)
    original_by_dataset, original_global = original_prediction_benchmark(prediction)
    recon_cells.to_csv(DATA / "reconstruction_balanced_cells.csv", index=False)
    recon_ranking.to_csv(DATA / "reconstruction_global_ranking.csv", index=False)
    pred_cells.to_csv(DATA / "prediction_balanced_cells.csv", index=False)
    pred_ranking.to_csv(DATA / "prediction_reconstructor_rankings.csv", index=False)
    original_by_dataset.to_csv(DATA / "forecast_original_reference.csv", index=False)

    stats = statistical_exports(recon_cells, pred_cells)
    design_table()
    ranking_tables(recon_ranking, pred_ranking, original_global)
    condition_appendix(recon_cells)
    interaction_table(recon_cells)
    ablations = ablation_table()
    forecast_checks = forecast_diagnostics(prediction)
    figures(recon_cells, pred_ranking, original_global)

    classifier = read_json(SYNTHETIC_CLASSIFIER)
    training = read_json(TRAINING_REPORT)
    training_summary = read_json(TRAINING_SUMMARY)
    manifest = {
        "schema_version": "1.0",
        "provenance": {
            "run_summary": str(RUN_SUMMARY.relative_to(ROOT)),
            "merge_manifest": str(MERGE_MANIFEST.relative_to(ROOT)),
            "reconstruction": str(RECONSTRUCTION.relative_to(ROOT)),
            "prediction": str(PREDICTION.relative_to(ROOT)),
            "frozen_generated_at": merge_manifest["generated_at"],
        },
        "validation": validation,
        "design": {
            "datasets": 3, "mechanisms": 3, "structures": 3, "rates": 4, "iterations": 10,
            "missingness_realizations": 1080, "reconstructors": 17, "forecasters": 4,
            "balanced_reconstruction_cells_per_model": 108,
            "forecast_inference_cells": len(pred_cells),
        },
        "cleaning_report": {"path": run_summary.get("cleaning_report"), "available": bool(run_summary.get("cleaning_report"))},
        "hpo": hpo_audit(),
        "training": {
            "base_series": training_summary["base_series"], "triplets": training_summary["training_triplets"],
            "train_triplets": training["train_triplets"], "validation_triplets": training["validation_triplets"],
            "best_epoch": int(pd.DataFrame(training["history"]).set_index("epoch").validation_loss.idxmin()),
            "stopped_after_epoch": int(pd.DataFrame(training["history"]).epoch.max()),
            "best_validation_loss": training["best_validation_loss"],
        },
        "synthetic_to_real": classifier,
        "statistics": stats,
        "forecast_diagnostics": forecast_checks,
        "ablation_summary": ablations,
        "primary_metric": "smape",
        "secondary_metrics": ["mae", "mad", "rmse", "mase", "mape"],
    }
    (METADATA / "manuscript_numbers_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manuscript artifacts to data, tables, metadata, and {FIG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
