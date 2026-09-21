#!/usr/bin/env python3
"""
Data Splitting Script

Splits cleaned univariate time series into three disjoint temporal partitions:
  1. reconstruction-train — degradation and reconstruction experiments
  2. sd2_validation — SD2 hyperparameter search only (never final evaluation)
  3. rolling_test — rolling-origin forecast evaluation

The rolling test holdout is the largest safe tail up to 20% of the series that
still accommodates H_max and the configured number of rolling origins.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from utils.logger import setup_logging

setup_logging("3_create_split")

import argparse
import json
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone

sys.path.insert(0, str(Path(__file__).parent))

from utils.config_loader import load_config
from utils.progress import tqdm
from utils.horizon_recommender import (
    HorizonConstraints,
    load_h_long_lookup,
    resolve_experiment_horizons,
)
from utils.split_plan import (
    plan_three_way_split,
    resolve_n_origins,
    split_manifest_entry,
)


def split_time_series_three_way(
    df: pd.DataFrame,
    boundaries,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into reconstruction train, SD2 validation, and rolling test."""
    reconstruction = df.iloc[: boundaries.reconstruction_end].copy()
    validation = df.iloc[boundaries.validation_start : boundaries.validation_end].copy()
    test = df.iloc[boundaries.test_start :].copy()
    return reconstruction, validation, test


def load_horizon_metadata(metadata_path: str) -> dict:
    if not os.path.exists(metadata_path):
        return {}
    with open(metadata_path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    return payload


def resolve_h_max(
    dataset_name: str,
    horizon_lookup: dict[str, int],
    experiment_horizons: dict,
    default_test_samples: int,
    total_samples: int,
) -> tuple[int, list[int], str]:
    forced = resolve_experiment_horizons(dataset_name, experiment_horizons)
    if forced:
        h_max = max(forced)
        return h_max, forced, f"experiment horizons {forced}; H_max={h_max}"
    if dataset_name in horizon_lookup:
        h_max = horizon_lookup[dataset_name]
        return h_max, [h_max], f"metadata h_long={h_max}"
    return default_test_samples, [default_test_samples], f"fallback test_samples={default_test_samples}"


def split_dataset(
    input_file: str,
    train_output_file: str,
    validation_output_file: str,
    test_output_file: str,
    config,
    *,
    horizon_lookup: dict[str, int] | None = None,
    origin_counts: dict[str, int] | None = None,
    constraints: HorizonConstraints | None = None,
) -> dict:
    print(f"\n📂 Splitting: {os.path.basename(input_file)}")

    train_exists = os.path.exists(train_output_file)
    validation_exists = os.path.exists(validation_output_file)
    test_exists = os.path.exists(test_output_file)

    try:
        df = pd.read_csv(input_file, index_col=0)
    except Exception as exc:
        print(f"  ❌ Error reading file: {exc}")
        return {"status": "error", "message": str(exc)}

    total_samples = len(df)
    print(f"  📊 Total samples: {total_samples}")

    dataset_name = os.path.basename(input_file)
    lookup = horizon_lookup or {}
    origin_counts = origin_counts or {}
    experiment_horizons = config.get_experiment_horizons()
    split_settings = config.get_dataset_split_settings(dataset_name)

    h_max, horizons, source = resolve_h_max(
        dataset_name,
        lookup,
        experiment_horizons,
        config.get_test_samples(),
        total_samples,
    )
    n_origins = resolve_n_origins(
        dataset_name,
        origin_counts,
        fallback=int(config.get_rolling_origin_settings().get("n_origins", 5)),
    )
    print(f"  🎯 H_max={h_max} ({source}), rolling origins={n_origins}")

    base_cons = constraints or HorizonConstraints(
        max_holdout_share=float(config.get_horizon_settings().get("max_holdout_share", 0.20)),
        min_train_length=int(config.get_horizon_settings().get("min_train_length", 200)),
    )
    cons = HorizonConstraints(
        max_holdout_share=split_settings["max_holdout_share"],
        min_train_length=split_settings["min_reconstruction_length"],
        ideal_holdout_share=base_cons.ideal_holdout_share,
        unsafe_train_threshold=base_cons.unsafe_train_threshold,
        h_short=base_cons.h_short,
        h_long=base_cons.h_long,
        short_series_h_short=base_cons.short_series_h_short,
        short_series_h_long=base_cons.short_series_h_long,
    )
    if split_settings["override_applied"]:
        print(
            "  ℹ️  Dataset split override: "
            f"min_reconstruction={split_settings['min_reconstruction_length']}, "
            f"validation={split_settings['validation_min_samples']}.."
            f"{split_settings['validation_max_samples']}"
        )

    try:
        boundaries = plan_three_way_split(
            total_samples,
            h_max,
            n_origins,
            constraints=cons,
            validation_share=split_settings["validation_share"],
            validation_min_samples=split_settings["validation_min_samples"],
            validation_max_samples=split_settings["validation_max_samples"],
            min_reconstruction_length=split_settings["min_reconstruction_length"],
        )
    except ValueError as exc:
        print(f"  ❌ Split planning failed: {exc}")
        return {"status": "error", "message": str(exc)}

    for note in boundaries.notes:
        print(f"  ℹ️  {note}")

    if (
        train_exists
        and validation_exists
        and test_exists
        and not config.get_overwrite_existing()
    ):
        try:
            existing_test_n = len(pd.read_csv(test_output_file, index_col=0))
        except Exception:
            existing_test_n = None
        if existing_test_n == boundaries.test_length:
            print(f"  ⏭️  Skipping (files exist with test holdout={boundaries.test_length})")
            return {"status": "skipped", "boundaries": boundaries, "horizons": horizons}

    train_df, validation_df, test_df = split_time_series_three_way(df, boundaries)

    print(
        f"  📈 Reconstruction train: {len(train_df)} "
        f"({len(train_df) / total_samples * 100:.1f}%)"
    )
    print(
        f"  🧪 SD2 validation:      {len(validation_df)} "
        f"({len(validation_df) / total_samples * 100:.1f}%)"
    )
    print(
        f"  📉 Rolling test:        {len(test_df)} "
        f"({len(test_df) / total_samples * 100:.1f}%)"
    )

    for path, frame in (
        (train_output_file, train_df),
        (validation_output_file, validation_df),
        (test_output_file, test_df),
    ):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        frame.to_csv(path)

    print(f"  ✅ Saved reconstruction train: {train_output_file}")
    print(f"  ✅ Saved SD2 validation:      {validation_output_file}")
    print(f"  ✅ Saved rolling test:        {test_output_file}")

    return {
        "status": "success",
        "total_samples": total_samples,
        "train_samples": len(train_df),
        "validation_samples": len(validation_df),
        "test_samples": len(test_df),
        "boundaries": boundaries,
        "horizons": horizons,
    }


def run_create_split(
    config,
    input_dir: str | None = None,
    output_dir: str | None = None,
    dataset: str | None = None,
    use_horizon_metadata: bool = True,
) -> bool:
    input_dir = input_dir or config.get_cleaned_dir()
    output_base_dir = output_dir or config.get_splitted_dir()
    train_output_dir = config.get_splitted_train_dir()
    validation_output_dir = config.get_splitted_sd2_validation_dir()
    test_output_dir = config.get_splitted_test_dir()
    manifest_path = config.get_split_manifest_path()

    metadata_path = config.get_dataset_metadata_path()
    metadata = load_horizon_metadata(metadata_path) if use_horizon_metadata else {}
    horizon_lookup = load_h_long_lookup(metadata) if metadata else {}
    origin_counts = config.get_rolling_origin_counts()
    constraints = HorizonConstraints(
        max_holdout_share=float(config.get_horizon_settings().get("max_holdout_share", 0.20)),
        min_train_length=int(config.get_horizon_settings().get("min_train_length", 200)),
        ideal_holdout_share=float(config.get_horizon_settings().get("ideal_holdout_share", 0.15)),
        unsafe_train_threshold=int(config.get_horizon_settings().get("unsafe_train_threshold", 190)),
    )

    print(f"\n{'=' * 60}")
    print("THREE-WAY DATA SPLITTING")
    print(f"{'=' * 60}")
    print(f"Input directory:            {input_dir}")
    print(f"Reconstruction train output: {train_output_dir}")
    print(f"SD2 validation output:       {validation_output_dir}")
    print(f"Rolling test output:         {test_output_dir}")
    print(f"Split manifest:              {manifest_path}")

    if dataset:
        datasets = [dataset]
    else:
        if not os.path.exists(input_dir):
            print(f"\n❌ Error: Input directory does not exist: {input_dir}")
            return False
        datasets = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

    if not datasets:
        print(f"\n⚠️  No CSV files found in {input_dir}")
        return False

    print(f"\n📋 Found {len(datasets)} dataset(s) to split")

    success_count = 0
    skip_count = 0
    error_count = 0
    manifest: dict = {
        "schema_version": "1.0",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_dir": input_dir,
        "datasets": {},
    }

    for ds in tqdm(datasets, desc="Splitting datasets", unit="file"):
        input_file = os.path.join(input_dir, ds)
        try:
            result = split_dataset(
                input_file,
                os.path.join(train_output_dir, ds),
                os.path.join(validation_output_dir, ds),
                os.path.join(test_output_dir, ds),
                config,
                horizon_lookup=horizon_lookup,
                origin_counts=origin_counts,
                constraints=constraints,
            )
            if result["status"] in {"success", "skipped"}:
                if result["status"] == "success":
                    success_count += 1
                else:
                    skip_count += 1
                manifest["datasets"][ds] = split_manifest_entry(
                    ds,
                    result["boundaries"],
                    result["horizons"],
                )
            else:
                error_count += 1
        except Exception as exc:
            print(f"\n❌ Error splitting {ds}: {exc}")
            import traceback

            traceback.print_exc()
            error_count += 1

    if manifest["datasets"]:
        os.makedirs(os.path.dirname(os.path.abspath(manifest_path)) or ".", exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as fh:
            json.dump(manifest, fh, indent=2, ensure_ascii=False)
        print(f"\n✅ Wrote split manifest: {manifest_path}")

    print(f"\n{'=' * 60}")
    print("SPLITTING COMPLETE")
    print(f"{'=' * 60}")
    print(f"Successfully split: {success_count}/{len(datasets)} datasets")
    print(f"Skipped (existing): {skip_count}")
    print(f"Errors:             {error_count}")
    return error_count == 0


def main():
    parser = argparse.ArgumentParser(
        description="Split cleaned time series into reconstruction, SD2 validation, and test"
    )
    parser.add_argument("--input-dir", type=str)
    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--dataset", type=str)
    parser.add_argument("--config", type=str, default="config/config.yaml")
    args = parser.parse_args()

    try:
        config = load_config(args.config)
        print(f"✓ Loaded configuration from: {args.config}\n")
    except FileNotFoundError:
        print(f"❌ Configuration file not found: {args.config}")
        return

    run_create_split(
        config,
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dataset=args.dataset,
    )


if __name__ == "__main__":
    main()
