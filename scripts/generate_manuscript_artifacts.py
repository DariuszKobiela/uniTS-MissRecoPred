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

ARTICLE = ROOT / "LG" / "DataMiningAndKnowledgeDiscovery"
OUT = ARTICLE / "generated"
FIG = ARTICLE / "images" / "generated"

RUN_SUMMARY = ROOT / "experiment_run_summary.json"
MERGE_MANIFEST = ROOT / "merged_results_manifest_20261010_202224.json"
RECONSTRUCTION = ROOT / "reconstruction_experiments_results" / "reconstruction_results_merged_20261010_202224.csv"
PREDICTION = ROOT / "prediction_experiment_results" / "prediction_results_rolling_origins_merged_20261010_202224.csv"
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
    "impute_mean": "Mean", "impute_median": "Median", "impute_ffill": "Forward fill",
    "impute_bfill": "Backward fill", "interpolate_linear": "Linear",
    "interpolate_cubic": "Cubic", "interpolate_pchip": "PCHIP", "knn": "KNN",
    "sarimax": "SARIMAX", "stable_diffusion_2_gaf": "GAF",
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


def balanced_reconstruction(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cell_keys = ["dataset_name", "technique", "structure", "rate_percent", "model"]
    cells = frame.groupby(cell_keys, as_index=False).agg(
        smape=("smape", "mean"), mae=("mae", "mean"), mad=("mad", "mean")
    )
    ranking = cells.groupby("model", as_index=False).agg(
        smape=("smape", "mean"), mae=("mae", "mean"), mad=("mad", "mean")
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
    recon_pairs.to_csv(OUT / "reconstruction_pairwise_smape.csv", index=False)
    pred_pairs.to_csv(OUT / "forecast_algorithms_pairwise_smape.csv", index=False)
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
        r"\footnotesize T: paired $t$-test; W: Wilcoxon signed-rank. Effect is Cohen $d_z$ for T and rank-biserial correlation for W.",
        r"\end{table}",
    ]
    write_table(OUT / "forecast_pairwise_smape.tex", forecast_lines)

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
    write_table(OUT / "reconstruction_pairwise_smape.tex", reconstruction_lines)
    (OUT / "reconstruction_friedman_smape.json").write_text(json.dumps(recon_omnibus, indent=2), encoding="utf-8")
    (OUT / "forecast_algorithms_friedman_smape.json").write_text(json.dumps(pred_omnibus, indent=2), encoding="utf-8")
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
    write_table(OUT / "split_design.tex", lines)


def ranking_tables(recon_ranking: pd.DataFrame, pred_ranking: pd.DataFrame) -> None:
    lines = [r"\begin{table}[t]", r"\caption{Global reconstruction ranking after balancing all 108 experimental cells.}",
             r"\label{tab:reconstruction-ranking}", r"\centering\small",
             r"\begin{tabular}{rlrrr}", r"\toprule", r"Rank & Method & sMAPE (\%) & MAE & MAD \\", r"\midrule"]
    for row in recon_ranking.itertuples():
        lines.append(f"{row.rank} & {tex(row.method)} & {row.smape:.3f} & {row.mae:.3f} & {row.mad:.3f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\footnotesize Each dataset $\times$ mechanism $\times$ structure $\times$ rate cell has equal weight; ten realizations are averaged first.",
              r"\end{table}"]
    write_table(OUT / "reconstruction_ranking.tex", lines)

    pivot = pred_ranking.pivot(index="reconstruction_model", columns="prediction_model", values="smape")
    ranks = pred_ranking.pivot(index="reconstruction_model", columns="prediction_model", values="rank")
    order = pred_ranking.groupby("reconstruction_model").smape.mean().sort_values().index
    lines = [r"\begin{table*}[t]", r"\caption{Balanced forecasting sMAPE (\%) by forecaster and reconstruction method. Parentheses give the within-forecaster rank.}",
             r"\label{tab:forecast-matrix}", r"\centering\scriptsize", r"\begin{tabular}{lrrrr}", r"\toprule",
             r"Reconstruction & Holt--Winters & Prophet & SARIMAX & XGBoost \\", r"\midrule"]
    for model in order:
        vals = [f"{pivot.loc[model, p]:.3f} ({int(ranks.loc[model, p])})" for p in ["holt_winters", "prophet", "sarimax", "xgboost"]]
        lines.append(f"{tex(DISPLAY[model])} & " + " & ".join(vals) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}",
              r"\footnotesize Origins and cumulative horizons are averaged within each realization, then ten realizations are averaged within each of 108 balanced cells.",
              r"\end{table*}"]
    write_table(OUT / "forecast_matrix.tex", lines)


def condition_appendix(cells: pd.DataFrame) -> None:
    pivot = cells.pivot(index=["dataset_name", "technique", "structure", "rate_percent"], columns="model", values="smape")
    models = sorted(EXPECTED_RECONSTRUCTORS, key=lambda x: ("stable_diffusion" in x, DISPLAY[x]))
    pivot = pivot[models]
    pivot.to_csv(OUT / "reconstruction_108_conditions_smape.csv")
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
    write_table(OUT / "appendix_reconstruction_108_conditions.tex", lines)


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
    write_table(OUT / "reconstruction_interactions.tex", lines)


def ablation_table() -> dict:
    frame = pd.read_csv(ABLATION)
    summary = frame.groupby(["ablation", "encoding"], as_index=False).agg(smape=("smape", "mean"), rmse=("rmse", "mean"))
    lines = [r"\begin{table}[t]", r"\caption{Transform-only controls averaged across datasets, windows, resolutions, and masks.}",
             r"\label{tab:transform-controls}", r"\centering\small", r"\begin{tabular}{llrr}", r"\toprule",
             r"Control & Encoding & sMAPE (\%) & RMSE \\", r"\midrule"]
    for row in summary.sort_values(["ablation", "smape"]).itertuples():
        lines.append(f"{tex(row.ablation)} & {row.encoding.upper()} & {row.smape:.3f} & {row.rmse:.3f} " + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    write_table(OUT / "transform_controls.tex", lines)
    return summary.to_dict(orient="records")


def forecast_diagnostics(frame: pd.DataFrame) -> dict:
    cumulative = frame[frame.metric_scope.eq("cumulative")].copy()
    reconstructed = cumulative[cumulative.source_type.eq("reconstructed")].copy()
    original = cumulative[cumulative.source_type.eq("original")].copy()

    original_summary = original.groupby(["prediction_model", "dataset_name"], as_index=False).agg(
        smape=("smape", "mean"), mae=("mae", "mean"), rmse=("rmse", "mean"),
        mase=("mase", "mean"), mape=("mape", "mean"),
    )
    original_summary.to_csv(OUT / "forecast_original_reference.csv", index=False)

    horizon = reconstructed.groupby(
        ["prediction_model", "reconstruction_model", "dataset_name", "forecast_horizon"], as_index=False
    ).agg(smape=("smape", "mean"), mae=("mae", "mean"), rmse=("rmse", "mean"),
          mase=("mase", "mean"), mape=("mape", "mean"))
    horizon["family"] = np.where(horizon.reconstruction_model.str.startswith("stable_diffusion"), "SD2", "classical")
    horizon.to_csv(OUT / "forecast_horizon_balanced.csv", index=False)

    horizon_best = horizon.groupby(
        ["dataset_name", "forecast_horizon", "prediction_model", "family"], as_index=False
    ).smape.min()
    horizon_best.to_csv(OUT / "forecast_horizon_best_family.csv", index=False)

    lead = frame[frame.source_type.eq("reconstructed") & frame.metric_scope.eq("lead_time_bin")].groupby(
        ["prediction_model", "reconstruction_model", "dataset_name", "lead_time_start", "lead_time_end"],
        as_index=False,
    ).agg(smape=("smape", "mean"), mae=("mae", "mean"), rmse=("rmse", "mean"),
          mase=("mase", "mean"), mape=("mape", "mean"))
    lead.to_csv(OUT / "forecast_lead_time_balanced.csv", index=False)

    origin = reconstructed.groupby(
        ["prediction_model", "reconstruction_model", "dataset_name", "origin"], as_index=False
    ).smape.mean()
    origin["rank"] = origin.groupby(["prediction_model", "dataset_name", "origin"]).smape.rank(method="average")
    stability = origin.groupby(["prediction_model", "reconstruction_model"], as_index=False).agg(
        mean_rank=("rank", "mean"), rank_sd=("rank", "std"), origin_evaluations=("rank", "size")
    )
    stability.to_csv(OUT / "forecast_origin_rank_stability.csv", index=False)

    axes = []
    for column in ["dataset_name", "technique", "structure", "rate_percent"]:
        summary = reconstructed.groupby([column, "prediction_model", "reconstruction_model"], as_index=False).smape.mean()
        winners = summary.loc[summary.groupby([column, "prediction_model"]).smape.idxmin()].copy()
        winners.insert(0, "axis", column)
        winners.rename(columns={column: "level"}, inplace=True)
        axes.append(winners[["axis", "level", "prediction_model", "reconstruction_model", "smape"]])
    pd.concat(axes, ignore_index=True).to_csv(OUT / "forecast_axis_winners.csv", index=False)

    fig, axes_plot = plt.subplots(1, 3, figsize=(10, 3.2), sharey=True)
    names = {"boiler_outlet_temp_univ": "Boiler", "pump_sensor_28_univ": "Pump", "vibration_sensor_S1": "Vibration"}
    for ax, dataset in zip(axes_plot, names):
        part = horizon_best[horizon_best.dataset_name.eq(dataset)].groupby(
            ["forecast_horizon", "family"], as_index=False
        ).smape.mean()
        for family, values in part.groupby("family"):
            ax.plot(values.forecast_horizon.astype(str), values.smape, marker="o", label=family)
        ax.set_title(names[dataset])
        ax.set_xlabel("Horizon (samples)")
        ax.tick_params(axis="x", rotation=40)
    axes_plot[0].set_ylabel("Best-family sMAPE (%)")
    axes_plot[-1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "forecast_horizon_families.pdf", bbox_inches="tight")
    plt.close(fig)

    return {
        "original_reference_rows": len(original_summary),
        "horizon_summary_rows": len(horizon),
        "lead_time_summary_rows": len(lead),
        "origin_stability_rows": len(stability),
    }


def figures(recon_cells: pd.DataFrame, pred_ranking: pd.DataFrame) -> None:
    plt.rcParams.update({"font.size": 8, "figure.dpi": 150})

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
    fig, ax = plt.subplots(figsize=(7, 6))
    image = ax.imshow(pivot.to_numpy(), aspect="auto", cmap="magma_r")
    ax.set_yticks(range(len(pivot)), [DISPLAY[m] for m in pivot.index])
    ax.set_xticks(range(4), ["Holt--Winters", "Prophet", "SARIMAX", "XGBoost"], rotation=20, ha="right")
    for i in range(len(pivot)):
        for j in range(4):
            ax.text(j, i, f"{pivot.iloc[i, j]:.1f}", ha="center", va="center", fontsize=6,
                    color="white" if pivot.iloc[i, j] > pivot.to_numpy().mean() else "black")
    ax.set_title("Balanced forecasting sMAPE")
    fig.colorbar(image, ax=ax, label="sMAPE (%)")
    fig.tight_layout()
    fig.savefig(FIG / "forecast_reconstruction_heatmap.pdf", bbox_inches="tight")
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
        original_image = plt.imread(ARTICLE / "images" / original_name)[..., :3]
        missing_image = plt.imread(ARTICLE / "images" / missing_name)[..., :3]
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
    OUT.mkdir(parents=True, exist_ok=True)
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
    recon_cells.to_csv(OUT / "reconstruction_balanced_cells.csv", index=False)
    recon_ranking.to_csv(OUT / "reconstruction_global_ranking.csv", index=False)
    pred_cells.to_csv(OUT / "prediction_balanced_cells.csv", index=False)
    pred_ranking.to_csv(OUT / "prediction_reconstructor_rankings.csv", index=False)

    stats = statistical_exports(recon_cells, pred_cells)
    design_table()
    ranking_tables(recon_ranking, pred_ranking)
    condition_appendix(recon_cells)
    interaction_table(recon_cells)
    ablations = ablation_table()
    forecast_checks = forecast_diagnostics(prediction)
    figures(recon_cells, pred_ranking)

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
    (OUT / "manuscript_numbers_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Generated manuscript artifacts in {OUT.relative_to(ROOT)} and {FIG.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
