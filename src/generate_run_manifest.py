#!/usr/bin/env python3
"""Generate a reproducibility run manifest for the rebuttal experiment."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from utils.config_loader import load_config
from utils.run_manifest import build_run_manifest, write_run_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    config = load_config(args.config)
    run_id = args.run_id or os.environ.get("RUN_ID") or f"rebuttal_{Path.cwd().name}"
    output = args.output or f"runs/{run_id}/manifests/run_manifest.json"

    manifest = build_run_manifest(
        run_id=run_id,
        config_paths=[
            "config/config.yaml",
            "config/prediction_models_config.yaml",
            config.get_split_manifest_path(),
        ],
        seed=int(config.get_rolling_origin_settings().get("seed", 42)),
    )
    path = write_run_manifest(output, manifest)
    print(json.dumps({"run_id": run_id, "manifest": str(path)}, indent=2))


if __name__ == "__main__":
    main()
