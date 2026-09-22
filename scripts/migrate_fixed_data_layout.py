#!/usr/bin/env python3
"""Move flat reconstructed CSVs into the hierarchical fixed-data tree."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from utils.experiment_naming import (  # noqa: E402
    build_reconstructed_relative_path,
    parse_flat_reconstructed_stem,
)


def migrate(fixed_dir: Path, *, dry_run: bool) -> tuple[int, int, int]:
    moved = skipped = errors = 0
    for path in sorted(fixed_dir.glob("*.csv")):
        try:
            meta = parse_flat_reconstructed_stem(path.stem)
            target = fixed_dir / build_reconstructed_relative_path(
                meta["dataset_name"],
                meta["technique"],
                meta["structure"],
                meta["rate_percent"],
                meta["iteration"],
                meta["model"],
            )
        except ValueError as exc:
            print(f"SKIP (unrecognized): {path.name} — {exc}")
            errors += 1
            continue
        if target.exists():
            print(f"SKIP (target exists): {path.name} -> {target.relative_to(fixed_dir)}")
            skipped += 1
            continue
        print(f"MOVE: {path.name} -> {target.relative_to(fixed_dir)}")
        if not dry_run:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(path), str(target))
        moved += 1
    return moved, skipped, errors


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--fixed-dir",
        type=Path,
        default=REPO_ROOT / "data" / "4_fixed_data",
        help="Root of reconstructed datasets (default: data/4_fixed_data)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print moves without changing files",
    )
    args = parser.parse_args()
    fixed_dir = args.fixed_dir.resolve()
    if not fixed_dir.is_dir():
        raise SystemExit(f"Fixed directory not found: {fixed_dir}")

    moved, skipped, errors = migrate(fixed_dir, dry_run=args.dry_run)
    print(f"Done: moved={moved}, skipped={skipped}, errors={errors}")


if __name__ == "__main__":
    main()
