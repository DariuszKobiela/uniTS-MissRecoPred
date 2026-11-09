#!/usr/bin/env python3
"""Check the final experiment release; --deep verifies every archived file."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import pandas as pd
import pyarrow.parquet as pq
import yaml


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--deep", action="store_true")
    args = parser.parse_args()
    package = Path(__file__).resolve().parent
    repo = package / "repo"
    release = repo / "releases/2026-10-10"
    expected_files = set()
    for line in (repo / "SHA256SUMS").read_text().splitlines():
        expected, name = line.split("  ", 1)
        relative = PurePosixPath(name)
        require(not relative.is_absolute() and ".." not in relative.parts, f"Unsafe checksum path: {name}")
        require(name not in expected_files, f"Duplicate checksum: {name}")
        expected_files.add(name)
        with (repo / name).open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        require(actual == expected, f"Checksum mismatch: {name}")
    actual_files = {path.relative_to(repo).as_posix() for path in repo.rglob("*") if path.is_file() and path.name != "SHA256SUMS"}
    require(actual_files == expected_files, "Missing or unchecked payload files")
    manifest = json.loads((release / "release_manifest.json").read_text())
    require(manifest["repository"] == "dariusz-kobiela/univariate-time-series-with-missingness-and-reconstructed", "Wrong repository")
    require(manifest["checks"] == {"missing_csvs": 720, "reconstructed_csvs": 12240, "recomputed_metric_checks": 24480}, "Incomplete build audit")
    allowed = {"boiler_outlet_temp_univ", "vibration_sensor_S1"}
    require(set(manifest["published_datasets"]) == allowed, "Wrong redistribution scope")
    require(manifest["licenses"] == {"boiler_outlet_temp_univ": "cc-by-4.0", "vibration_sensor_S1": "cc0-1.0"}, "Wrong component licenses")
    require((repo / "LICENSE.md").is_file(), "Missing license notice")
    require(not any("pump" in name.lower() for name in actual_files), "Pump payload path")
    require(not any(name.startswith(("data_", "releases/2026-10-10/provenance/", "releases/2026-10-10/reproducibility/", "releases/2026-10-10/results/batch_statistics/")) for name in actual_files), "Withdrawn mixed material")
    card = (repo / "README.md").read_text()
    frontmatter = yaml.safe_load(card.split("---", 2)[1])
    require(set(frontmatter["license"]) == {"cc-by-4.0", "cc0-1.0"}, "Wrong card licenses")
    require({config["config_name"] for config in frontmatter["configs"]} == {"forecasting_metrics", "reconstruction_metrics"}, "Wrong viewer configs")
    for config in frontmatter["configs"]:
        for entry in config["data_files"]:
            require(bool(list(repo.glob(entry["path"]))), f"Empty viewer path: {entry}")
    forecast = release / "results/prediction_by_model"
    models = {"sarimax", "xgboost", "prophet", "holt_winters"}
    require({path.stem for path in forecast.glob("*.parquet")} == models, "Unexpected forecast files")
    forecast_keys = ["dataset_name", "source_type", "technique", "structure", "rate_percent",
                     "reconstruction_iteration", "reconstruction_model", "prediction_model", "evaluation_scheme",
                     "origin", "forecast_horizon", "metric_scope", "lead_time_start", "lead_time_end"]
    require(manifest["forecast_rows"] == 4 * manifest["forecast_rows_per_model"], "Wrong forecast total")
    for model in sorted(models):
        path = forecast / f"{model}.parquet"
        require(pq.read_metadata(path).num_rows == manifest["forecast_rows_per_model"], f"Wrong forecast count: {model}")
        frame = pq.read_table(path, columns=forecast_keys + ["base_train_samples"]).to_pandas()
        require(set(frame.dataset_name) == allowed, f"Excluded forecast dataset: {model}")
        require(set(frame.prediction_model) == {model}, f"Wrong forecast model: {model}")
        require(not frame.duplicated(forecast_keys).any(), f"Duplicate frozen forecast keys: {model}")
        expected_train = manifest["historical_train_lengths"]
        for stem, group in frame.groupby("dataset_name"):
            require(set(group.base_train_samples) == {expected_train[stem]}, f"Historical train mismatch: {model}/{stem}")
    reconstruction = pd.read_parquet(release / "results/reconstruction_metrics.parquet")
    require(set(reconstruction.dataset_name) == allowed, "Excluded reconstruction dataset")
    require(len(reconstruction) == 12240, "Wrong reconstruction row count")
    require(not reconstruction.duplicated(["dataset_name", "technique", "structure", "rate_percent", "iteration", "model"]).any(),
            "Duplicate frozen reconstruction keys")
    split = json.loads((release / "metadata/split_manifest.json").read_text())
    lengths = {"boiler_outlet_temp_univ.csv": (68620, 500, 17280), "vibration_sensor_S1.csv": (144, 24, 42)}
    require(set(split["datasets"]) == set(lengths), "Excluded split dataset")
    for name, expected in lengths.items():
        entry = split["datasets"][name]
        ranges = [entry[key] for key in ["train", "sd2_validation", "test"]]
        require(tuple(block["length"] for block in ranges) == expected, f"Historical split mismatch: {name}")
        require(ranges[0]["start"] == 0 and ranges[0]["end_exclusive"] == ranges[1]["start"]
                and ranges[1]["end_exclusive"] == ranges[2]["start"]
                and ranges[2]["end_exclusive"] == entry["series_length"], f"Overlapping split: {name}")
    members = pd.read_parquet(release / "metadata/archive_members.parquet")
    require(len(members) == 11 + 720 + 12240, "Wrong total archive members")
    require(not members.path.str.contains("pump", case=False).any(), "Pump archive member")
    require(not members.duplicated(["archive", "path"]).any(), "Duplicate archive members")
    for name, meta in manifest["archives"].items():
        require(int((members.archive == f"archives/{name}").sum()) == meta["files"], f"Archive count mismatch: {name}")
        require((release / "archives" / name).stat().st_size == meta["bytes"], f"Archive size mismatch: {name}")
    for filename in ["missingness_realizations.csv", "empirical_mask_coverage.csv"]:
        frame = pd.read_csv(release / "metadata" / filename)
        require(set(frame.dataset_name) == allowed, f"Excluded report dataset: {filename}")
    require(len(pd.read_csv(release / "metadata/missingness_realizations.csv")) == 720, "Wrong missing report count")
    csv = pd.read_csv(release / "results/reconstruction_results_merged_20261010_202224.csv")
    require(set(csv.dataset_name) == allowed and len(csv) == 12240, "Excluded CSV metric records")
    deep_files = 0
    if args.deep:
        process = subprocess.Popen(["zstd", "-dcq", str(release / "reports/missingness_gaps.csv.zst")], stdout=subprocess.PIPE)
        try:
            for chunk in pd.read_csv(process.stdout, chunksize=100_000):
                require(set(chunk.dataset_name) <= allowed, "Excluded detailed gap records")
            process.stdout.close()
            require(process.wait() == 0, "Gap report decompression failed")
        except BaseException:
            process.terminate()
            process.wait()
            raise
        for archive, group in members.groupby("archive"):
            print(f"Checking archived content: {archive}", flush=True)
            expected_members = {row.path: row for row in group.itertuples(index=False)}
            seen = set()
            process = subprocess.Popen(["zstd", "-d", "-q", "-c", str(release / archive)], stdout=subprocess.PIPE)
            try:
                with tarfile.open(fileobj=process.stdout, mode="r|") as tar:
                    for member in tar:
                        path = PurePosixPath(member.name)
                        require(not path.is_absolute() and ".." not in path.parts and member.isfile(), f"Unsafe member: {member.name}")
                        require(member.name not in seen and member.name in expected_members, f"Unexpected archive member: {member.name}")
                        seen.add(member.name)
                        expected = expected_members[member.name]
                        require(member.size == expected.bytes, f"Member size mismatch: {member.name}")
                        with tar.extractfile(member) as handle:
                            actual = hashlib.file_digest(handle, "sha256").hexdigest()
                        require(actual == expected.sha256, f"Archived hash mismatch: {member.name}")
                        deep_files += 1
                # Drain tar padding so zstd can verify its frame checksum and exit normally.
                while process.stdout.read(8 * 1024 * 1024):
                    pass
                process.stdout.close()
                require(process.wait() == 0, f"Archive decompression failed: {archive}")
                require(seen == set(expected_members), f"Missing archive members: {archive}")
            except BaseException:
                process.stdout.close()
                process.terminate()
                process.wait()
                raise
    report = {"status": "ok", "validated_at": datetime.now(timezone.utc).isoformat(),
              "checked_public_files": len(expected_files), "deep": args.deep, "checked_archive_members": deep_files,
              "missing_series": 720, "reconstructed_series": 12240, "forecast_metric_rows": manifest["forecast_rows"],
              "reconstruction_metric_rows": 12240}
    report_path = package / ("validation_report.json" if args.deep else "validation_report_quick.json")
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
