#!/usr/bin/env python3
"""
Temporary one-off: fill 16 pre-existing empty pump timestamps from train.

Those slots were empty in the original source, stayed NaN in missing files,
and were reconstructed as if they were experimental holes. They are not
MCAR/MAR/MNAR draws and have no measured ground truth for MAD.

This script copies the 16 values from the current training series into every
pump_*.csv under data/3_missing_data and data/4_fixed_data. After that they
are observed points: excluded from the missing mask (MAD) and consistent
for prediction.

Timestamps (16):

    2018-04-27 17:48:00 .. 17:55:00   (8)
    2018-05-04 16:42:00, 17:24:00     (2)
    2018-06-18 13:41:00               (1)
    2018-07-11 09:12:00 .. 09:14:00   (3)
    2018-07-11 09:55:00               (1)
    2018-07-12 13:19:00               (1)

Usage:
    uv run python scripts/tmp_fill_pump_preexisting_gaps.py --dry-run
    uv run python scripts/tmp_fill_pump_preexisting_gaps.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = REPO_ROOT / "data" / "2_splitted_data" / "train" / "pump_sensor_28_univ.csv"
DEFAULT_DIRS = (
    REPO_ROOT / "data" / "3_missing_data",
    REPO_ROOT / "data" / "4_fixed_data",
)

GAP_TIMESTAMPS = (
    "2018-04-27 17:48:00",
    "2018-04-27 17:49:00",
    "2018-04-27 17:50:00",
    "2018-04-27 17:51:00",
    "2018-04-27 17:52:00",
    "2018-04-27 17:53:00",
    "2018-04-27 17:54:00",
    "2018-04-27 17:55:00",
    "2018-05-04 16:42:00",
    "2018-05-04 17:24:00",
    "2018-06-18 13:41:00",
    "2018-07-11 09:12:00",
    "2018-07-11 09:13:00",
    "2018-07-11 09:14:00",
    "2018-07-11 09:55:00",
    "2018-07-12 13:19:00",
)
GAP_SET = set(GAP_TIMESTAMPS)


def load_train_gap_values(train_path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    with train_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            ts = row[0]
            if ts in GAP_SET:
                if len(row) < 2 or row[1].strip() == "":
                    raise ValueError(f"Train has empty value at {ts}")
                values[ts] = row[1]
                if len(values) == len(GAP_SET):
                    break
    missing = [ts for ts in GAP_TIMESTAMPS if ts not in values]
    if missing:
        raise ValueError(f"Train is missing timestamps: {missing}")
    return values


def inspect_gaps(path: Path) -> dict[str, str | None]:
    found: dict[str, str | None] = {ts: None for ts in GAP_TIMESTAMPS}
    n_found = 0
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.reader(handle)
        next(reader, None)
        for row in reader:
            if not row:
                continue
            ts = row[0]
            if ts in GAP_SET:
                found[ts] = row[1] if len(row) > 1 else ""
                n_found += 1
                if n_found == len(GAP_SET):
                    break
    return found


def fill_gaps(path: Path, replacements: dict[str, str]) -> tuple[int, int]:
    """Overwrite gap timestamps. Returns (n_replaced, n_already_matching)."""
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    n_replaced = 0
    n_already = 0
    n_seen = 0
    with path.open("r", encoding="utf-8", newline="") as src, tmp_path.open(
        "w", encoding="utf-8", newline=""
    ) as dst:
        reader = csv.reader(src)
        writer = csv.writer(dst, lineterminator="\n")
        header = next(reader, None)
        if header is None:
            tmp_path.unlink(missing_ok=True)
            raise ValueError(f"Empty file: {path}")
        writer.writerow(header)
        for row in reader:
            if row and row[0] in replacements:
                n_seen += 1
                new_val = replacements[row[0]]
                old_val = row[1] if len(row) > 1 else ""
                if old_val == new_val:
                    n_already += 1
                else:
                    if len(row) < 2:
                        row.append(new_val)
                    else:
                        row[1] = new_val
                    n_replaced += 1
            writer.writerow(row)
    if n_seen != len(replacements):
        tmp_path.unlink(missing_ok=True)
        raise ValueError(
            f"{path.name}: found {n_seen}/{len(replacements)} gap timestamps"
        )
    tmp_path.replace(path)
    return n_replaced, n_already


def collect_pump_csvs(directories: list[Path]) -> list[Path]:
    files: list[Path] = []
    for directory in directories:
        files.extend(sorted(p for p in directory.glob("pump_*.csv") if p.is_file()))
    return files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill 16 pre-existing pump gaps from the training series."
    )
    parser.add_argument(
        "--train",
        type=Path,
        default=TRAIN_PATH,
        help="Reference train CSV (default: data/2_splitted_data/train/pump_sensor_28_univ.csv)",
    )
    parser.add_argument(
        "--dirs",
        type=Path,
        nargs="+",
        default=list(DEFAULT_DIRS),
        help="Directories to patch (default: 3_missing_data and 4_fixed_data)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Inspect sample files and list targets without writing",
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
    return parser.parse_args()


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else REPO_ROOT / path


def main() -> int:
    args = parse_args()
    train_path = resolve(args.train)
    directories = [resolve(p) for p in args.dirs]
    replacements = load_train_gap_values(train_path)

    print("Reference values from train:")
    for ts in GAP_TIMESTAMPS:
        print(f"  {ts}  {replacements[ts]}")
    print()

    files = collect_pump_csvs(directories)
    if not files:
        print("No pump_*.csv files found", file=sys.stderr)
        return 1

    by_dir: dict[Path, list[Path]] = {}
    for path in files:
        by_dir.setdefault(path.parent, []).append(path)
    for directory, group in by_dir.items():
        print(f"{directory}: {len(group)} pump files")
    print()

    if args.dry_run:
        samples = [group[0] for group in by_dir.values()]
        for path in samples:
            found = inspect_gaps(path)
            n_absent = sum(1 for v in found.values() if v is None)
            n_empty = sum(
                1
                for v in found.values()
                if v is not None and str(v).strip() in ("", "nan", "NaN")
            )
            n_match = sum(1 for ts, v in found.items() if v == replacements[ts])
            print(f"sample {path}:")
            print(f"  timestamps present: {16 - n_absent}/16  empty/NaN: {n_empty}  already train: {n_match}")
            for ts in GAP_TIMESTAMPS:
                print(f"    {ts}  now={found[ts]!r}  train={replacements[ts]}")
            print()
        print(f"Would patch {len(files)} files")
        return 0

    n_ok = 0
    n_errors = 0
    n_replaced_files = 0
    n_already_files = 0
    for i, path in enumerate(files, 1):
        try:
            n_replaced, n_already = fill_gaps(path, replacements)
        except ValueError as exc:
            print(f"  ERROR {exc}")
            n_errors += 1
            continue
        n_ok += 1
        if n_replaced:
            n_replaced_files += 1
        if n_already == len(replacements):
            n_already_files += 1
        if args.verbose:
            print(f"  {path.name}: replaced {n_replaced}, already {n_already}")
        if args.progress_every > 0 and i % args.progress_every == 0:
            print(f"  progress {i}/{len(files)}")

    print(
        f"Patched: {n_ok}  files with changes: {n_replaced_files}  "
        f"already matching: {n_already_files}  errors: {n_errors}"
    )
    return 1 if n_errors else 0


if __name__ == "__main__":
    sys.exit(main())
