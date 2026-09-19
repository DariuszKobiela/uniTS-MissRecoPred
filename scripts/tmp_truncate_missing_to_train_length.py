#!/usr/bin/env python3
"""
Temporary one-off: drop the remaining holdout tail from data/3_missing_data.

The unzipped missingness files were built from the full series, then the last
ALREADY_REMOVED samples were cut. The current train/test split uses H_max as
the holdout tail (see data/1_5_horizon_recommendation). Remaining samples to
drop from the end:

    drop = H_max - ALREADY_REMOVED

Three groups (prefix -> H_max):

    boiler_*  H_max=720   already 10 gone  -> drop 710
    pump_*    H_max=1440  already 10 gone  -> drop 1430
    vibr_*    H_max=24    already 10 gone  -> drop 14

Usage:
    uv run python scripts/tmp_truncate_missing_to_train_length.py --dry-run
    uv run python scripts/tmp_truncate_missing_to_train_length.py
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]

# H_max from step 1.5 (longest experiment horizon = train/test cut).
PREFIX_TO_H_MAX = {
    "boiler": 720,
    "pump": 1440,
    "vibr": 24,
}
ALREADY_REMOVED = 10

MISSING_DIR = REPO_ROOT / "data" / "3_missing_data"


def count_data_rows(path: Path) -> int:
    """Number of data rows (excludes header)."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        lines = sum(1 for _ in handle)
    if lines == 0:
        raise ValueError(f"Empty file: {path}")
    return lines - 1


def truncate_csv(path: Path, n_data_rows: int) -> tuple[int, int]:
    """Keep header + first n_data_rows; drop the rest from the end. Returns (kept, dropped)."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        header = handle.readline()
        kept_rows: list[str] = []
        for _ in range(n_data_rows):
            line = handle.readline()
            if not line:
                break
            kept_rows.append(line)
        leftover = sum(1 for _ in handle)

    if len(kept_rows) != n_data_rows:
        raise ValueError(
            f"{path.name}: expected at least {n_data_rows} data rows, found {len(kept_rows)}"
        )

    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(header)
        handle.writelines(kept_rows)
        if kept_rows and not kept_rows[-1].endswith("\n"):
            handle.write("\n")
    tmp_path.replace(path)
    return len(kept_rows), leftover


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Drop remaining H_max-10 holdout samples from data/3_missing_data CSVs."
    )
    parser.add_argument(
        "--missing-dir",
        type=Path,
        default=MISSING_DIR,
        help="Directory with missingness CSVs (default: data/3_missing_data)",
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    missing_dir = args.missing_dir if args.missing_dir.is_absolute() else REPO_ROOT / args.missing_dir

    drop_n = {
        prefix: h_max - ALREADY_REMOVED
        for prefix, h_max in PREFIX_TO_H_MAX.items()
    }

    print("Drop remaining holdout tail from the end:")
    print(f"  already removed from ZIP series: {ALREADY_REMOVED}")
    for prefix, h_max in PREFIX_TO_H_MAX.items():
        print(f"  {prefix:7s}  H_max={h_max}  drop {drop_n[prefix]} more")
    print()

    csv_files = sorted(p for p in missing_dir.glob("*.csv") if p.is_file())
    if not csv_files:
        print(f"No CSV files in {missing_dir}", file=sys.stderr)
        return 1

    grouped: dict[str, list[Path]] = defaultdict(list)
    unknown: list[Path] = []
    for path in csv_files:
        prefix = path.name.split("_", 1)[0]
        if prefix in drop_n:
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

    for prefix, files in grouped.items():
        n_drop = drop_n[prefix]
        lengths: set[int] = set()
        print(f"{prefix}: {len(files)} files, drop {n_drop} from the end")
        for path in files:
            current = count_data_rows(path)
            lengths.add(current)
            target = current - n_drop
            if n_drop == 0:
                n_already_ok += 1
                continue
            if target <= 0:
                print(f"  ERROR {path.name}: {current} rows, cannot drop {n_drop}")
                n_errors += 1
                continue
            if args.dry_run:
                if args.verbose:
                    print(f"  would truncate {path.name}: {current} -> {target} (drop {n_drop})")
                n_truncated += 1
                continue
            kept, dropped = truncate_csv(path, target)
            if args.verbose:
                print(f"  {path.name}: {current} -> {kept} (dropped {dropped})")
            n_truncated += 1
        length_desc = ", ".join(str(x) for x in sorted(lengths))
        print(f"  current data-row lengths: {length_desc} -> {sorted(x - n_drop for x in lengths)}")
        print()

    action = "Would change" if args.dry_run else "Changed"
    print(
        f"{action}: {n_truncated}  already matching: {n_already_ok}  errors: {n_errors}"
    )
    return 1 if n_errors else 0


if __name__ == "__main__":
    sys.exit(main())
