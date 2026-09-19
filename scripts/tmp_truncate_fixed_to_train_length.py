#!/usr/bin/env python3
"""
Temporary one-off: truncate CSVs in data/4_fixed_data to current train lengths.

Reconstructed files were produced on the old training window (horizon = 10).
The new split uses H_max from step 1.5 as the holdout tail. Extra samples
are dropped from the end so each reconstructed series matches:

    boiler_*  ->  data/2_splitted_data/train/boiler_outlet_temp_univ.csv
    pump_*    ->  data/2_splitted_data/train/pump_sensor_28_univ.csv
    vibr_*    ->  data/2_splitted_data/train/vibration_sensor_S1.csv

Usage:
    uv run python scripts/tmp_truncate_fixed_to_train_length.py --dry-run
    uv run python scripts/tmp_truncate_fixed_to_train_length.py
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

PREFIX_TO_TRAIN = {
    "boiler": REPO_ROOT / "data" / "2_splitted_data" / "train" / "boiler_outlet_temp_univ.csv",
    "pump": REPO_ROOT / "data" / "2_splitted_data" / "train" / "pump_sensor_28_univ.csv",
    "vibr": REPO_ROOT / "data" / "2_splitted_data" / "train" / "vibration_sensor_S1.csv",
}

FIXED_DIR = REPO_ROOT / "data" / "4_fixed_data"


def count_data_rows(path: Path) -> int:
    """Number of data rows (excludes header)."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        lines = sum(1 for _ in handle)
    if lines == 0:
        raise ValueError(f"Empty file: {path}")
    return lines - 1


def truncate_csv(path: Path, n_data_rows: int) -> int:
    """Keep header + first n_data_rows; drop the rest from the end. Returns kept."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    kept = 0
    with path.open("r", encoding="utf-8", newline="") as src, tmp_path.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        header = src.readline()
        if not header:
            tmp_path.unlink(missing_ok=True)
            raise ValueError(f"Empty file: {path}")
        dst.write(header)
        last = ""
        for _ in range(n_data_rows):
            line = src.readline()
            if not line:
                break
            dst.write(line)
            last = line
            kept += 1
        if kept != n_data_rows:
            tmp_path.unlink(missing_ok=True)
            raise ValueError(
                f"{path.name}: expected at least {n_data_rows} data rows, found {kept}"
            )
        if last and not last.endswith("\n"):
            dst.write("\n")
    tmp_path.replace(path)
    return kept


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Truncate data/4_fixed_data CSVs to current training-series lengths."
    )
    parser.add_argument(
        "--fixed-dir",
        type=Path,
        default=FIXED_DIR,
        help="Directory with reconstructed CSVs (default: data/4_fixed_data)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned truncations without writing files",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print one line per file",
    )
    parser.add_argument(
        "--progress-every",
        type=int,
        default=200,
        help="Print a progress line every N files (default: 200)",
    )
    parser.add_argument(
        "--check-all-lengths",
        action="store_true",
        help="In --dry-run, count rows in every file (slow). Default: first file per group.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    fixed_dir = args.fixed_dir if args.fixed_dir.is_absolute() else REPO_ROOT / args.fixed_dir

    targets = {prefix: count_data_rows(train_path) for prefix, train_path in PREFIX_TO_TRAIN.items()}

    print("Target lengths (training data rows, header excluded):")
    for prefix, n_rows in targets.items():
        print(f"  {prefix:7s}  {n_rows}  <- {PREFIX_TO_TRAIN[prefix].name}")
    print()

    csv_files = sorted(p for p in fixed_dir.glob("*.csv") if p.is_file())
    if not csv_files:
        print(f"No CSV files in {fixed_dir}", file=sys.stderr)
        return 1

    grouped: dict[str, list[Path]] = defaultdict(list)
    unknown: list[Path] = []
    for path in csv_files:
        prefix = path.name.split("_", 1)[0]
        if prefix in targets:
            grouped[prefix].append(path)
        else:
            unknown.append(path)

    if unknown:
        print("Unmapped files (skipped):")
        for path in unknown:
            print(f"  {path.name}")
        print()

    n_truncated = 0
    n_already_ok = 0
    n_errors = 0
    done = 0
    total = sum(len(v) for v in grouped.values())

    for prefix, files in grouped.items():
        target = targets[prefix]
        lengths: set[int] = set()
        print(f"{prefix}: {len(files)} files -> {target} data rows")
        sampled_len: int | None = None
        if args.dry_run and not args.check_all_lengths:
            sampled_len = count_data_rows(files[0])
            lengths.add(sampled_len)
            extra = sampled_len - target
            print(f"  sampled {files[0].name}: {sampled_len} rows (drop {extra})")
            if extra > 0:
                n_truncated += len(files)
            elif extra == 0:
                n_already_ok += len(files)
            else:
                print(f"  ERROR {files[0].name}: shorter than train ({sampled_len} < {target})")
                n_errors += 1
            print(f"  drop-from-end samples per file: {extra}")
            print()
            continue
        for path in files:
            current = count_data_rows(path)
            lengths.add(current)
            extra = current - target
            if extra == 0:
                n_already_ok += 1
            elif extra < 0:
                print(f"  ERROR {path.name}: shorter than train ({current} < {target})")
                n_errors += 1
            elif args.dry_run:
                if args.verbose:
                    print(f"  would truncate {path.name}: {current} -> {target} (drop {extra})")
                n_truncated += 1
            else:
                try:
                    kept = truncate_csv(path, target)
                    if args.verbose:
                        print(f"  {path.name}: {current} -> {kept} (dropped {extra})")
                    n_truncated += 1
                except ValueError as exc:
                    print(f"  ERROR {exc}")
                    n_errors += 1
            done += 1
            if args.progress_every > 0 and done % args.progress_every == 0:
                print(f"  progress {done}/{total}")
        extra_desc = ", ".join(str(x - target) for x in sorted(lengths))
        print(f"  drop-from-end samples per file: {extra_desc}")
        print()

    action = "Would change" if args.dry_run else "Changed"
    print(
        f"{action}: {n_truncated}  already matching: {n_already_ok}  errors: {n_errors}"
    )
    return 1 if n_errors else 0


if __name__ == "__main__":
    sys.exit(main())
