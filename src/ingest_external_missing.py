#!/usr/bin/env python3
"""
Ingest external time series with missing values + test horizon for pipeline.entry external_missing.

Writes canonical degraded filenames into missing_dir, test CSVs into splitted test dir,
optional clean train CSVs, and external_missing_ingest_state.json for predict (original branch).

Run after setting pipeline.entry: external_missing and pipeline.external_missing.manifest in config.
Then: 4_reconstruct → 7_train → 8_predict → 9_calculate_prediction_error (skip 3, 5, 6).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).parent))

from utils.config_loader import load_config
from utils.logger import setup_logging

setup_logging("ingest_external_missing")

_SAFE_ID_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def normalize_id(raw: str) -> str:
    s = _SAFE_ID_RE.sub("_", str(raw).strip()).strip("_")
    if not s:
        raise ValueError("Dataset id is empty after normalization")
    return s


def resolve_path(manifest_dir: Path, p: str) -> Path:
    path = Path(p)
    if path.is_absolute():
        return path.resolve()
    return (manifest_dir / path).resolve()


def read_series_csv(path: Path, config) -> pd.DataFrame:
    fmt = config.get_csv_format(path.name)
    df = pd.read_csv(
        path,
        sep=fmt["sep"],
        decimal=fmt["decimal"],
        index_col=fmt["index_col"],
        na_values=["", " "],
    )
    if df.shape[1] < 1:
        raise ValueError(f"No value columns in {path}")
    df.iloc[:, 0] = pd.to_numeric(df.iloc[:, 0], errors="coerce")
    return df


def value_col(df: pd.DataFrame) -> pd.Series:
    return df.iloc[:, 0]


def validate_test_no_nan(s: pd.Series, label: str) -> None:
    if len(s) == 0:
        raise ValueError(f"{label}: segment is empty")
    if s.isna().any():
        raise ValueError(f"{label}: values must not contain NaN (complete test horizon)")


def degraded_filename(safe_id: str, technique: str, rate_percent: int, iteration: int) -> str:
    return f"{safe_id}_{technique}_{rate_percent}p_{iteration}.csv"


def process_dataset(
    entry: Dict[str, Any],
    manifest_dir: Path,
    config,
    require_train_missing: bool,
) -> Dict[str, Any]:
    safe_id = normalize_id(entry["id"])
    syn = entry.get("synthetic") or {}
    technique = str(syn.get("technique", "EXTERNAL")).strip()
    if "_" in technique or not technique:
        raise ValueError(
            f"dataset {safe_id}: synthetic.technique must be a single token (no underscores), got {technique!r}"
        )
    rate_percent = int(syn.get("rate_percent", 0))
    iteration = int(syn.get("iteration", 1))

    has_clean_train = False

    if "series_csv" in entry:
        spath = resolve_path(manifest_dir, str(entry["series_csv"]))
        if not spath.is_file():
            raise FileNotFoundError(f"series_csv not found: {spath}")
        df = read_series_csv(spath, config)
        test_cfg = entry.get("test") or {}
        if str(test_cfg.get("mode", "last_n")).lower() != "last_n":
            raise ValueError(f"dataset {safe_id}: only test.mode: last_n is supported")
        n = int(test_cfg["n"])
        if n <= 0:
            raise ValueError(f"dataset {safe_id}: test.n must be positive")
        if len(df) <= n:
            raise ValueError(
                f"dataset {safe_id}: series shorter than test.n ({len(df)} <= {n})"
            )
        train_df = df.iloc[:-n].copy()
        test_df = df.iloc[-n:].copy()
    elif "train_csv" in entry and "test_csv" in entry:
        ttrain = resolve_path(manifest_dir, str(entry["train_csv"]))
        ttest = resolve_path(manifest_dir, str(entry["test_csv"]))
        if not ttrain.is_file():
            raise FileNotFoundError(f"train_csv not found: {ttrain}")
        if not ttest.is_file():
            raise FileNotFoundError(f"test_csv not found: {ttest}")
        train_df = read_series_csv(ttrain, config)
        test_df = read_series_csv(ttest, config)
    else:
        raise ValueError(
            f"dataset {safe_id}: provide either series_csv+test or train_csv+test_csv"
        )

    validate_test_no_nan(value_col(test_df), f"dataset {safe_id} test")

    tr = value_col(train_df)
    if require_train_missing and not tr.isna().any():
        raise ValueError(
            f"dataset {safe_id}: train has no NaN but require_train_missing is true"
        )
    if not tr.isna().any():
        print(
            f"  ⚠️  {safe_id}: train has no NaN — reconstruction step may only copy values"
        )

    missing_dir = Path(config.get_external_missing_output_missing_dir())
    test_dir = Path(config.get_external_missing_output_test_dir())
    train_dir = Path(config.get_external_missing_output_train_dir())

    deg_name = degraded_filename(safe_id, technique, rate_percent, iteration)
    missing_dir.mkdir(parents=True, exist_ok=True)
    test_dir.mkdir(parents=True, exist_ok=True)
    train_dir.mkdir(parents=True, exist_ok=True)

    train_df.to_csv(missing_dir / deg_name)
    test_df.to_csv(test_dir / f"{safe_id}.csv")
    print(f"  ✓ {safe_id}: degraded → {missing_dir / deg_name}")
    print(f"  ✓ {safe_id}: test    → {test_dir / f'{safe_id}.csv'}")

    ckey = entry.get("train_clean_csv")
    if ckey:
        cpath = resolve_path(manifest_dir, str(ckey))
        if not cpath.is_file():
            raise FileNotFoundError(f"train_clean_csv not found: {cpath}")
        clean_df = read_series_csv(cpath, config)
        validate_test_no_nan(value_col(clean_df), f"dataset {safe_id} train_clean")
        if len(clean_df) != len(train_df):
            raise ValueError(
                f"dataset {safe_id}: train_clean_csv rows ({len(clean_df)}) must match "
                f"train segment ({len(train_df)})"
            )
        clean_df.to_csv(train_dir / f"{safe_id}.csv")
        has_clean_train = True
        print(f"  ✓ {safe_id}: clean train → {train_dir / f'{safe_id}.csv'}")

    return {"id": safe_id, "has_clean_train": has_clean_train}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest external missingness manifest (external_missing pipeline)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Main configuration file",
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default=None,
        help="Override manifest path (default: pipeline.external_missing.manifest)",
    )
    args = parser.parse_args()

    try:
        config = load_config(args.config)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        sys.exit(1)

    manifest_rel = args.manifest or config.get_external_missing_manifest_path()
    manifest_path = Path(manifest_rel)
    if not manifest_path.is_file():
        print(f"❌ Manifest not found: {manifest_path.resolve()}")
        sys.exit(1)

    manifest_dir = manifest_path.parent.resolve()

    with open(manifest_path, "r", encoding="utf-8") as f:
        doc = yaml.safe_load(f)

    if not isinstance(doc, dict) or "datasets" not in doc:
        print("❌ Manifest must be a YAML mapping with key 'datasets' (list)")
        sys.exit(1)

    require_train_missing = bool(doc.get("require_train_missing", False))
    datasets: List[Dict[str, Any]] = doc["datasets"]
    if not isinstance(datasets, list) or not datasets:
        print("❌ Manifest 'datasets' must be a non-empty list")
        sys.exit(1)

    print("=" * 70)
    print("INGEST EXTERNAL MISSING (manifest → missing_dir + test + optional train)")
    print("=" * 70)
    print(f"Manifest: {manifest_path.resolve()}")
    print(f"Missing dir: {config.get_external_missing_output_missing_dir()}")
    print(f"Test dir:    {config.get_external_missing_output_test_dir()}")
    print(f"Train dir:   {config.get_external_missing_output_train_dir()}")
    print(f"require_train_missing: {require_train_missing}")
    print("=" * 70)

    entries_out: List[Dict[str, Any]] = []
    for entry in datasets:
        if not isinstance(entry, dict):
            print("❌ Each manifest dataset entry must be a mapping")
            sys.exit(1)
        try:
            meta = process_dataset(entry, manifest_dir, config, require_train_missing)
            entries_out.append(meta)
        except Exception as e:
            print(f"❌ Error processing entry {entry.get('id')!r}: {e}")
            sys.exit(1)

    state = {
        "version": 1,
        "ingested_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path.resolve()),
        "entries": entries_out,
    }
    state_path = Path(config.get_external_missing_ingest_state_path())
    state_path.parent.mkdir(parents=True, exist_ok=True)
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    print(f"\n✓ Wrote ingest state: {state_path}")

    print("\nNext (with pipeline.entry: external_missing):")
    print("  - reconstruct (4) → train prediction models (7) → predict (8) → calculate prediction error (9)")
    print("  - Skip: degrade (3), reconstruction error (5), reconstruction dashboard (6)")
    if not any(e["has_clean_train"] for e in entries_out):
        print(
            "\nℹ️  No train_clean_csv entries: use prediction.predict_on_original_train: false "
            "or only reconstructed predictions will apply."
        )


if __name__ == "__main__":
    main()
