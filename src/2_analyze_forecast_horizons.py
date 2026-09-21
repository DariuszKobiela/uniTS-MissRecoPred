"""
Forecast horizon analysis script (pipeline step 1.5).

Reads cleaned univariate time series from data/1_cleaned_data, infers the
actual sampling interval, recommends per-series H_short / H_long, and writes
artifacts to data/1_5_horizon_recommendation/:
  - dataset_metadata.json
  - horizon_recommendations.csv
  - horizon_recommendations.md

Usage:
    python 2_analyze_forecast_horizons.py [--input-dir DIR] [--dataset FILENAME]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logger import setup_logging

setup_logging("2_analyze_forecast_horizons")

from utils.config_loader import load_config
from utils.progress import tqdm
from utils.horizon_recommender import (
    HorizonConstraints,
    SeriesProfile,
    analyze_profiles_batch,
    format_horizon_span,
    infer_series_profile,
    metadata_document,
)
from utils.split_plan import plan_three_way_split, resolve_n_origins, split_manifest_entry


def _load_cleaned_series(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0)
    try:
        df.index = pd.to_datetime(df.index)
    except (ValueError, TypeError):
        pass
    return df


def _constraints_from_config(config) -> HorizonConstraints:
    hz = config.get_horizon_settings()
    return HorizonConstraints(
        max_holdout_share=float(hz.get("max_holdout_share", 0.20)),
        min_train_length=int(hz.get("min_train_length", 200)),
        ideal_holdout_share=float(hz.get("ideal_holdout_share", 0.15)),
        unsafe_train_threshold=int(hz.get("unsafe_train_threshold", 190)),
        h_short=int(hz.get("h_short", 12)),
        h_long=int(hz.get("h_long", 96)),
        short_series_h_short=int(hz.get("short_series_h_short", 12)),
        short_series_h_long=int(hz.get("short_series_h_long", 24)),
    )


def _build_split_plans(config, series_recs, constraints: HorizonConstraints) -> dict[str, dict]:
    """Calculate the same concrete three-way split later used by step 3."""
    origin_counts = config.get_rolling_origin_counts()
    fallback_origins = int(config.get_rolling_origin_settings().get("n_origins", 5))
    plans: dict[str, dict] = {}
    for rec in series_recs:
        settings = config.get_dataset_split_settings(rec.series_id)
        series_constraints = HorizonConstraints(
            max_holdout_share=settings["max_holdout_share"],
            min_train_length=settings["min_reconstruction_length"],
            ideal_holdout_share=constraints.ideal_holdout_share,
            unsafe_train_threshold=constraints.unsafe_train_threshold,
            h_short=constraints.h_short,
            h_long=constraints.h_long,
            short_series_h_short=constraints.short_series_h_short,
            short_series_h_long=constraints.short_series_h_long,
        )
        n_origins = resolve_n_origins(rec.series_id, origin_counts, fallback_origins)
        boundaries = plan_three_way_split(
            rec.n,
            rec.h_long,
            n_origins,
            constraints=series_constraints,
            validation_share=settings["validation_share"],
            validation_min_samples=settings["validation_min_samples"],
            validation_max_samples=settings["validation_max_samples"],
            min_reconstruction_length=settings["min_reconstruction_length"],
        )
        plan = split_manifest_entry(rec.series_id, boundaries, rec.horizons)
        plan["dataset_override_applied"] = settings["override_applied"]
        plans[rec.series_id] = plan
    return plans


def _series_table_rows(series_recs, split_plans: dict[str, dict]) -> list[dict]:
    rows = []
    for rec in series_recs:
        plan = split_plans[rec.series_id]
        rows.append(
            {
                "series_id": rec.series_id,
                "sampling_label": rec.sampling_label,
                "sampling_interval_seconds": rec.sampling_interval_seconds,
                "inferred_freq": rec.inferred_freq or "",
                "mean_delta_seconds": rec.mean_delta_seconds,
                "median_delta_seconds": rec.median_delta_seconds,
                "freq_confidence": round(rec.freq_confidence, 4),
                "is_irregular": rec.is_irregular,
                "n": rec.n,
                "trimmed_length": rec.trimmed_length,
                "h_short": rec.h_short,
                "h_short_span": format_horizon_span(
                    rec.h_short, rec.sampling_interval_seconds
                ),
                "h_long": rec.h_long,
                "h_long_span": format_horizon_span(
                    rec.h_long, rec.sampling_interval_seconds
                ),
                "horizons": rec.horizons,
                "horizons_span": ", ".join(
                    format_horizon_span(h, rec.sampling_interval_seconds)
                    for h in rec.horizons
                ),
                "train_length": rec.train_length,
                "holdout_share": round(rec.holdout_share, 6),
                "status": rec.status,
                "short_cycle_label": rec.short_cycle_label,
                "long_cycle_label": rec.long_cycle_label,
                "notes": "; ".join(rec.notes),
                "reconstruction_length": plan["reconstruction_train"]["length"],
                "validation_length": plan["sd2_validation"]["length"],
                "rolling_test_length": plan["rolling_test"]["length"],
                "rolling_test_share": plan["holdout_share"],
                "n_origins": plan["n_origins"],
                "dataset_override_applied": plan["dataset_override_applied"],
                "split_status": (
                    "Feasible (dataset override)"
                    if plan["dataset_override_applied"]
                    else "Feasible"
                ),
            }
        )
    return rows


def _write_markdown_report(
    path: str,
    *,
    input_dir: str,
    output_dir: str,
    constraints: HorizonConstraints,
    series_rows: list[dict],
) -> None:
    lines = [
        "# Forecast horizon recommendations",
        "",
        f"Input directory: `{input_dir}`",
        f"Output directory: `{output_dir}`",
        "",
        "Three-way split: reconstruction-train, sd2_validation (HPO only), rolling_test.",
        "Rolling test = largest safe holdout up to 20% accommodating H_max and configured origins.",
        "Shorter horizons are prefixes of one H_max forecast at each origin.",
        "",
        "## Per-series recommendations",
        "",
        "| Series | Sampling | n | Horizons | Horizons (time) | H_max | Reconstruction | SD2 validation | Rolling test | Origins | Split status |",
        "| --- | --- | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]

    for row in series_rows:
        test_share_pct = f"{100 * row['rolling_test_share']:.1f}%"
        horizons = ", ".join(str(h) for h in row.get("horizons", [row["h_short"], row["h_long"]]))
        lines.append(
            f"| {row['series_id']} | {row['sampling_label']} | {row['n']} | "
            f"{horizons} | {row.get('horizons_span', '')} | {row['h_long']} | "
            f"{row['reconstruction_length']} | {row['validation_length']} | "
            f"{row['rolling_test_length']} ({test_share_pct}) | {row['n_origins']} | "
            f"{row['split_status']} |"
        )

    lines.append("")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def run_analyze_forecast_horizons(
    config,
    input_dir: str | None = None,
    dataset: str | None = None,
    metadata_path: str | None = None,
    report_csv_path: str | None = None,
    report_md_path: str | None = None,
) -> bool:
    """Step 1.5: infer sampling interval and recommend forecast horizons."""
    input_dir = input_dir or config.get_cleaned_dir()
    output_dir = config.get_horizon_dir()
    metadata_path = metadata_path or config.get_dataset_metadata_path()
    report_csv_path = report_csv_path or config.get_horizon_report_csv_path()
    report_md_path = report_md_path or config.get_horizon_report_md_path()

    constraints = _constraints_from_config(config)
    experiment_horizons = config.get_experiment_horizons()

    print(f"\n{'=' * 60}")
    print("FORECAST HORIZON ANALYSIS")
    print(f"{'=' * 60}")
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Metadata output:  {metadata_path}")

    if dataset:
        datasets = [dataset]
    else:
        if not os.path.exists(input_dir):
            print(f"\n❌ Error: Input directory does not exist: {input_dir}")
            return False
        datasets = sorted(
            f
            for f in os.listdir(input_dir)
            if f.endswith(".csv") and not f.startswith("horizon_recommendations")
        )

    if not datasets:
        print(f"\n⚠️  No CSV files found in {input_dir}")
        return False

    print(f"\n📋 Found {len(datasets)} dataset(s) to analyze")

    profiles: list[SeriesProfile] = []
    for ds in tqdm(datasets, desc="Horizon analysis", unit="file"):
        path = os.path.join(input_dir, ds)
        print(f"\n📂 Analyzing: {ds}")
        try:
            df = _load_cleaned_series(path)
            profile = infer_series_profile(ds, df.index)
            profiles.append(profile)
            print(
                f"  n={profile.n}, interval={profile.sampling_label}, "
                f"confidence={profile.freq_confidence:.2f}"
            )
            if profile.notes:
                for note in profile.notes:
                    print(f"  ℹ️  {note}")
        except Exception as exc:
            print(f"  ❌ Error analyzing {ds}: {exc}")
            import traceback

            traceback.print_exc()

    if not profiles:
        print("\n❌ No series profiles generated.")
        return False

    series_recs = analyze_profiles_batch(
        profiles,
        constraints=constraints,
        experiment_horizons=experiment_horizons,
    )

    split_plans = _build_split_plans(config, series_recs, constraints)
    generated_at = datetime.now(timezone.utc).isoformat()
    doc = metadata_document(
        series_recs,
        input_dir=input_dir,
        output_dir=output_dir,
        constraints=constraints,
        generated_at=generated_at,
        split_entries={
            series_id: {"three_way_split": plan}
            for series_id, plan in split_plans.items()
        },
    )

    os.makedirs(os.path.dirname(os.path.abspath(metadata_path)) or ".", exist_ok=True)
    with open(metadata_path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, indent=2, ensure_ascii=False)
    print(f"\n✅ Wrote metadata: {metadata_path}")

    series_rows = _series_table_rows(series_recs, split_plans)
    os.makedirs(os.path.dirname(os.path.abspath(report_csv_path)) or ".", exist_ok=True)
    pd.DataFrame(series_rows).to_csv(report_csv_path, index=False)
    print(f"✅ Wrote CSV report: {report_csv_path}")

    _write_markdown_report(
        report_md_path,
        input_dir=input_dir,
        output_dir=output_dir,
        constraints=constraints,
        series_rows=series_rows,
    )
    print(f"✅ Wrote Markdown report: {report_md_path}")

    for rec in series_recs:
        plan = split_plans[rec.series_id]
        print(
            f"  → {rec.series_id}: horizons={rec.horizons}, H_max={rec.h_long}, "
            f"split={plan['reconstruction_train']['length']}/"
            f"{plan['sd2_validation']['length']}/{plan['rolling_test']['length']}, "
            f"origins={plan['n_origins']}, "
            f"override={plan['dataset_override_applied']}"
        )

    print(f"\n{'=' * 60}")
    print("HORIZON ANALYSIS COMPLETE")
    print(f"{'=' * 60}\n")
    return True


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze cleaned time series and recommend forecast horizons"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to configuration file (default: config/config.yaml)",
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        help="Input directory with cleaned datasets (default: from config)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Specific dataset filename to analyze (default: all datasets)",
    )
    parser.add_argument(
        "--metadata-path",
        type=str,
        help="Output path for dataset_metadata.json (default: from config)",
    )
    parser.add_argument(
        "--report-csv",
        type=str,
        help="Output path for horizon_recommendations.csv (default: from config)",
    )
    parser.add_argument(
        "--report-md",
        type=str,
        help="Output path for horizon_recommendations.md (default: from config)",
    )

    args = parser.parse_args()

    try:
        config = load_config(args.config)
        print(f"✓ Loaded configuration from: {args.config}\n")
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {args.config}")
        return

    run_analyze_forecast_horizons(
        config,
        input_dir=args.input_dir,
        dataset=args.dataset,
        metadata_path=args.metadata_path,
        report_csv_path=args.report_csv,
        report_md_path=args.report_md,
    )


if __name__ == "__main__":
    main()
