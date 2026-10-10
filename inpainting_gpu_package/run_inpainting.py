#!/usr/bin/env python3
"""Portable launcher for fine-tuned SD2 time-series inpainting."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ENCODINGS = ("gaf", "mtf", "rp", "spec")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Reconstruct missing time-series data with the bundled fine-tuned model.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python run_inpainting.py --encodings gaf
  python run_inpainting.py --encodings mtf spec
  python run_inpainting.py --encodings all
  python run_inpainting.py --encodings gaf rp --filter-dataset vibration_sensor_S1
""",
    )
    parser.add_argument(
        "--encodings",
        nargs="+",
        required=True,
        choices=(*ENCODINGS, "all"),
        help="One or more image encodings, or 'all'.",
    )
    parser.add_argument("--filter-dataset", nargs="+", help="Only selected dataset stems.")
    parser.add_argument("--filter-technique", nargs="+", choices=("MCAR", "MAR", "MNAR"))
    parser.add_argument("--filter-structure", nargs="+", choices=("scattered", "contiguous", "mixed"))
    parser.add_argument("--filter-rate", nargs="+", type=int, choices=(2, 5, 20, 50))
    parser.add_argument("--filter-iteration", nargs="+", type=int)
    parser.add_argument("--force", action="store_true", help="Overwrite existing outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Print the delegated command and exit.")
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    encodings = list(ENCODINGS) if "all" in args.encodings else list(dict.fromkeys(args.encodings))
    models = [f"stable_diffusion_2_{name}_finetuned" for name in encodings]
    command = [
        sys.executable,
        str(root / "src" / "10_reconstruct_datasets.py"),
        "--config",
        str(root / "config" / "config.yaml"),
        "--models",
        *models,
        "--n-jobs",
        "1",
    ]
    for name in ("filter_dataset", "filter_technique", "filter_structure", "filter_rate", "filter_iteration"):
        values = getattr(args, name)
        if values:
            command.extend([f"--{name.replace('_', '-')}", *map(str, values)])
    if args.force:
        command.append("--force")

    print("Encodings:", ", ".join(encodings))
    print("Command:", " ".join(command))
    if args.dry_run:
        return 0

    env = os.environ.copy()
    env["SD2_FINETUNED_MODEL"] = str(root / "models" / "sd2_windowed" / "best_model")
    env["SD2_FAIL_CLOSED"] = "1"
    return subprocess.call(command, cwd=root, env=env)


if __name__ == "__main__":
    raise SystemExit(main())
