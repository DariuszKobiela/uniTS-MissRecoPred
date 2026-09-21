#!/usr/bin/env python3
"""Create reviewer-facing gap diagnostics for degraded time-series files."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config_loader import load_config
from utils.empirical_mask_coverage import report_empirical_mask_coverage
from utils.experiment_naming import decode_missingness_label
from utils.missingness_analysis import summarize_missingness
from utils.progress import tqdm


def parse_degraded_filename(filename: str) -> dict:
    parts = Path(filename).stem.split("_")
    rate_index = next(
        (
            position
            for position, part in enumerate(parts)
            if part.endswith("p") and part[:-1].isdigit()
        ),
        None,
    )
    if rate_index is None or rate_index < 1 or rate_index + 1 >= len(parts):
        raise ValueError(f"Invalid degraded filename: {filename}")
    mechanism, structure = decode_missingness_label(parts[rate_index - 1])
    return {
        "dataset_name": "_".join(parts[: rate_index - 1]),
        "mechanism": mechanism,
        "structure": structure,
        "rate_percent": int(parts[rate_index][:-1]),
        "requested_missing_rate": int(parts[rate_index][:-1]) / 100.0,
        "iteration": int(parts[rate_index + 1]),
    }


def run_analyze_missingness(
    config,
    *,
    input_dir: str | None = None,
    output_dir: str | None = None,
) -> bool:
    input_path = Path(input_dir or config.get_missing_dir())
    output_path = Path(output_dir) if output_dir else input_path / "reports"
    files = sorted(input_path.glob("*.csv"))
    if not files:
        print(f"❌ No degraded CSV files found in {input_path}")
        return False

    summaries: list[dict] = []
    gaps: list[dict] = []
    errors: list[str] = []
    for path in tqdm(
        files,
        desc="Missingness diagnostics",
        unit="file",
        dynamic_ncols=True,
    ):
        try:
            metadata = parse_degraded_filename(path.name)
            metadata["output_file"] = str(path)
            frame = pd.read_csv(path, index_col=0)
            series = pd.to_numeric(frame.iloc[:, 0], errors="coerce")
            summary, file_gaps = summarize_missingness(series, metadata=metadata)
            summaries.append(summary)
            gaps.extend(file_gaps)
        except Exception as exc:
            errors.append(f"{path.name}: {exc}")

    output_path.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(summaries).to_csv(
        output_path / "missingness_realizations.csv", index=False
    )
    pd.DataFrame(gaps).to_csv(output_path / "missingness_gaps.csv", index=False)

    sd_cfg = config.config.get("computation", {}).get("stable_diffusion", {})
    windowing = sd_cfg.get("windowing", {}) or {}
    mask_frame = report_empirical_mask_coverage(
        input_path,
        image_size=int(sd_cfg.get("image_size", 512)),
        window_samples=int(windowing.get("default_window_samples", 512)),
        context_samples=int(windowing.get("context_samples", 64)),
    )
    if not mask_frame.empty:
        mask_path = output_path / "empirical_mask_coverage.csv"
        mask_frame.to_csv(mask_path, index=False)
        print(f"✓ Empirical mask coverage: {mask_path}")

    print(f"✓ Realizations: {len(summaries)}")
    print(f"✓ Individual gaps: {len(gaps)}")
    print(f"✓ Reports: {output_path}")
    if errors:
        print(f"⚠️  Files not analyzed: {len(errors)}")
        for message in errors[:10]:
            print(f"  - {message}")
    return bool(summaries) and not errors


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Report gap counts and sample/time gap-length distributions"
    )
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--input-dir")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = load_config(args.config)
    success = run_analyze_missingness(
        config, input_dir=args.input_dir, output_dir=args.output_dir
    )
    raise SystemExit(0 if success else 1)


if __name__ == "__main__":
    main()
