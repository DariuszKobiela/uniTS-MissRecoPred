#!/usr/bin/env python3
"""Package the frozen 2026-10-10 experiment for its Hugging Face dataset."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import sys
import tarfile
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import pyarrow.parquet as pq
import yaml

from export_prediction_parquet_by_model import export_by_model
from correct_experiment_dataset_licenses import build as build_licensed_package

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = "dariusz-kobiela/univariate-time-series-with-missingness-and-reconstructed"
RELEASE_DATE = "2026-10-10"
STAMP = "20261010_202224"
RELATIVE = Path("releases") / RELEASE_DATE
TEMPLATES = ROOT / "docs" / "experiment_dataset_release"
STEMS = ("boiler_outlet_temp_univ", "pump_sensor_28_univ", "vibration_sensor_S1")
RECONSTRUCTORS = {
    "impute_mean", "impute_median", "impute_ffill", "impute_bfill",
    "interpolate_linear", "interpolate_cubic", "interpolate_pchip", "knn", "sarimax",
    "stable_diffusion_2_gaf", "stable_diffusion_2_mtf", "stable_diffusion_2_rp", "stable_diffusion_2_spec",
    "stable_diffusion_2_gaf_finetuned", "stable_diffusion_2_mtf_finetuned",
    "stable_diffusion_2_rp_finetuned", "stable_diffusion_2_spec_finetuned",
}
FORECASTERS = {"sarimax", "xgboost", "prophet", "holt_winters"}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def read_series(data: bytes, columns: list[str]) -> pa.Table:
    return pacsv.read_csv(
        pa.BufferReader(data),
        read_options=pacsv.ReadOptions(use_threads=False),
        convert_options=pacsv.ConvertOptions(column_types={columns[0]: pa.string(), columns[1]: pa.float64()}),
    ).combine_chunks()


def values(table: pa.Table) -> np.ndarray:
    return table.column(1).to_numpy()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "release/huggingface/experiment-data-2026-10-10-licensed")
    args = parser.parse_args()
    public_output = args.output.resolve()
    require(not public_output.exists(), f"Output exists: {public_output}")
    output = public_output.with_name(public_output.name + "-private-staging")
    require(not output.exists(), f"Output exists; preserve it or choose --output: {output}")
    require(shutil.which("zstd") is not None, "Install the zstd command before packaging")
    output.mkdir(parents=True)
    repository = output / "repo"
    release = repository / RELATIVE
    release.mkdir(parents=True)
    reconstruction_path = ROOT / "reconstruction_experiments_results" / f"reconstruction_results_merged_{STAMP}.csv"
    prediction_path = ROOT / "prediction_experiment_results" / f"prediction_results_rolling_origins_merged_{STAMP}.csv"
    reconstruction = pd.read_csv(reconstruction_path)
    key = ["dataset_name", "technique", "structure", "rate_percent", "iteration", "model"]
    require(len(reconstruction) == 18_360 and not reconstruction.duplicated(key).any(), "Invalid reconstruction keys")
    require(set(reconstruction.model) == RECONSTRUCTORS, "Unexpected reconstructors")
    require(set(reconstruction.dataset_name) == set(STEMS), "Unexpected datasets")
    require(set(reconstruction.technique) == {"MCAR", "MAR", "MNAR"}, "Unexpected mechanisms")
    require(set(reconstruction.structure) == {"scattered", "contiguous", "mixed"}, "Unexpected structures")
    require(set(reconstruction.rate_percent) == {2, 5, 20, 50}, "Unexpected rates")
    require(set(reconstruction.iteration) == set(range(1, 11)), "Unexpected repetitions")
    metrics = reconstruction.set_index(key)
    report = pd.read_csv(ROOT / "data/3_missing_data/reports/missingness_realizations.csv")
    realization_key = ["dataset_name", "mechanism", "structure", "rate_percent", "iteration"]
    require(len(report) == 1080 and not report.duplicated(realization_key).any(), "Invalid missingness report")
    report_lookup = report.set_index(realization_key)

    print("Exporting and validating frozen forecasting results...", flush=True)
    forecast_dir = release / "results/prediction_by_model"
    exported = export_by_model(prediction_path, forecast_dir)
    export_manifest = json.loads(exported.read_text())
    require(set(export_manifest["models"]) == FORECASTERS, "Unexpected forecasting models")
    lengths: dict[str, int] = {}
    forecast_layout: dict[str, dict] = {}
    for model, meta in export_manifest["models"].items():
        parquet = forecast_dir / f"{model}.parquet"
        require(pq.read_metadata(parquet).num_rows == meta["rows"] == 624_342, f"Wrong row count: {model}")
        frame = pq.read_table(parquet, columns=[
            "dataset_name", "prediction_model", "base_train_samples", "test_start", "test_stop_exclusive",
            "origin", "forecast_horizon_max", "forecast_horizon", "evaluation_scheme",
        ]).to_pandas()
        require(set(frame.prediction_model) == {model}, f"Mixed forecast models in {model}")
        for stem, group in frame.groupby("dataset_name"):
            sizes = group.base_train_samples.unique()
            require(len(sizes) == 1, f"Ambiguous historical train length: {stem}")
            n_train = int(sizes[0])
            require(stem not in lengths or lengths[stem] == n_train, f"Historical length disagreement: {stem}")
            lengths[stem] = n_train
            layout = {
                "n_origins": int(group.origin.nunique()),
                "horizons": sorted(map(int, group.forecast_horizon.unique())),
                "origins": group[["origin", "test_start", "test_stop_exclusive", "forecast_horizon_max"]]
                    .drop_duplicates().sort_values("origin").astype(int).to_dict("records"),
            }
            require(stem not in forecast_layout or forecast_layout[stem] == layout, f"Origin disagreement: {stem}")
            forecast_layout[stem] = layout
        meta["path"] = f"{model}.parquet"
    write_json(exported, export_manifest)
    require(lengths == {STEMS[0]: 68620, STEMS[1]: 175756, STEMS[2]: 144}, "Unexpected historical lengths")

    # Rebuild only inside this release. The working split was changed after the experiment.
    work = output / "split_work"
    split_dir = work / "data/2_splitted_data"
    clean_tables: dict[str, pa.Table] = {}
    truth: dict[str, np.ndarray] = {}
    split = {
        "schema_version": "1.0", "experiment_date": RELEASE_DATE,
        "reconstructed_for_release": True,
        "provenance": "Train lengths and origin offsets verified against all four frozen forecast tables; "
        "train/validation are slices of cleaned series; published test CSVs verified against cleaned tails.",
        "later_working_manifest_generated_at": json.loads((ROOT / "data/2_splitted_data/split_manifest.json").read_text())["generated_at"],
        "datasets": {},
    }
    for stem in STEMS:
        source = ROOT / "data/1_cleaned_data" / f"{stem}.csv"
        frame = pd.read_csv(source)
        columns = frame.columns.tolist()
        clean_tables[stem] = read_series(source.read_bytes(), columns)
        truth[stem] = values(clean_tables[stem])[:lengths[stem]]
        test = pd.read_csv(ROOT / "data/2_splitted_data/test" / f"{stem}.csv")
        test_start = len(frame) - len(test)
        pd.testing.assert_frame_equal(test.reset_index(drop=True), frame.iloc[test_start:].reset_index(drop=True))
        require(max(x["test_stop_exclusive"] for x in forecast_layout[stem]["origins"]) == len(test), f"Test mismatch: {stem}")
        validation_length = test_start - lengths[stem]
        require(validation_length == (24 if stem == STEMS[2] else 500), f"Unexpected historical validation: {stem}")
        ranges = {"train": (0, lengths[stem]), "sd2_validation": (lengths[stem], test_start), "test": (test_start, len(frame))}
        for name, (start, end) in ranges.items():
            path = split_dir / name / f"{stem}.csv"
            path.parent.mkdir(parents=True, exist_ok=True)
            if name == "test":
                shutil.copy2(ROOT / "data/2_splitted_data/test" / f"{stem}.csv", path)
            else:
                frame.iloc[start:end].to_csv(path, index=False)
        split["datasets"][f"{stem}.csv"] = {
            "series_length": len(frame), **forecast_layout[stem],
            **{name: {"start": start, "end_exclusive": end, "length": end - start} for name, (start, end) in ranges.items()},
        }
    write_json(split_dir / "split_manifest.json", split)
    write_json(release / "metadata/split_manifest.json", split)
    print("Historical split recovered: boiler=68620/500/17280, pump=175756/500/44064, vibration=144/24/42.", flush=True)

    @lru_cache(maxsize=24)
    def missing_series(stem: str, mechanism: str, structure: str, rate: int, iteration: int) -> tuple[np.ndarray, np.ndarray]:
        path = ROOT / "data/3_missing_data" / f"{stem}_{mechanism}-{structure}_{rate}p_{iteration}.csv"
        table = read_series(path.read_bytes(), clean_tables[stem].column_names)
        require(table.column(0).equals(clean_tables[stem].column(0).slice(0, lengths[stem])), f"Missing index mismatch: {path}")
        data = values(table)
        mask = np.isnan(data)
        row = report_lookup.loc[(stem, mechanism, structure, rate, iteration)]
        require(len(data) == int(row.n_total) == lengths[stem], f"Missing length mismatch: {path}")
        require(int(mask.sum()) == int(row.n_missing), f"Missing mask mismatch: {path}")
        require(np.allclose(data[~mask], truth[stem][~mask], rtol=1e-12, atol=1e-10), f"Corrupted observed values: {path}")
        return data, mask

    inventory: list[dict] = []
    stats = {"missing_csvs": 0, "reconstructed_csvs": 0, "recomputed_metric_checks": 0}

    def validate_csv(path: Path, data: bytes) -> int | None:
        relative = path.relative_to(ROOT) if path.is_relative_to(ROOT) else None
        if relative is None or relative.parts[:2] not in [("data", "3_missing_data"), ("data", "4_fixed_data")]:
            return None
        if relative.parts[1] == "3_missing_data":
            if len(relative.parts) != 3 or path.suffix != ".csv":
                return None
            stem, label, rate, iteration = path.stem.rsplit("_", 3)
            mechanism, structure = label.split("-", 1)
            missing_series(stem, mechanism, structure, int(rate[:-1]), int(iteration))
            stats["missing_csvs"] += 1
            return lengths[stem]
        stem, label, rate, model, iteration_file = relative.parts[2:]
        mechanism, structure = label.split("-", 1)
        iteration = int(Path(iteration_file).stem)
        rate_number = int(rate[:-1])
        degraded, mask = missing_series(stem, mechanism, structure, rate_number, iteration)
        table = read_series(data, clean_tables[stem].column_names)
        require(table.column(0).equals(clean_tables[stem].column(0).slice(0, lengths[stem])), f"Reconstruction index mismatch: {path}")
        reconstructed = values(table)
        require(np.isfinite(reconstructed).all(), f"Nonfinite reconstruction: {path}")
        require(np.allclose(reconstructed[~mask], degraded[~mask], rtol=1e-10, atol=1e-9), f"Observed values changed: {path}")
        expected = metrics.loc[(stem, mechanism, structure, rate_number, iteration, model)]
        require(int(expected.n_total) == len(reconstructed) and int(expected.n_missing) == int(mask.sum()), f"Metric mask mismatch: {path}")
        absolute_errors = np.abs(truth[stem][mask] - reconstructed[mask])
        for name, result in [("mae", absolute_errors.mean()), ("std_diff", absolute_errors.std())]:
            require(np.isclose(result, float(expected[name]), rtol=1e-7, atol=1e-8), f"Frozen {name} mismatch: {path}: {result} != {expected[name]}")
            stats["recomputed_metric_checks"] += 1
        stats["reconstructed_csvs"] += 1
        return len(reconstructed)

    def archive(name: str, entries: list[tuple[Path, str]]) -> None:
        destination = release / "archives" / name
        destination.parent.mkdir(exist_ok=True)
        print(f"Building {name}: {len(entries)} files...", flush=True)
        with destination.open("wb") as compressed:
            process = subprocess.Popen(["zstd", "-T4", "-3", "-q", "-c"], stdin=subprocess.PIPE, stdout=compressed)
            try:
                with tarfile.open(fileobj=process.stdin, mode="w|") as tar:
                    for number, (path, member) in enumerate(entries, 1):
                        # Keep one CSV in memory; validate and archive exactly the same bytes.
                        data = path.read_bytes()
                        rows = validate_csv(path, data) if path.suffix == ".csv" else None
                        info = tarfile.TarInfo(member)
                        info.size = len(data)
                        info.mode = 0o644
                        info.mtime = 0
                        tar.addfile(info, io.BytesIO(data))
                        inventory.append({"archive": f"archives/{name}", "path": member, "bytes": len(data),
                                          "sha256": hashlib.sha256(data).hexdigest(), "series_rows": rows})
                        if number % 1000 == 0:
                            print(f"  {name}: {number}/{len(entries)} files validated and packed", flush=True)
                process.stdin.close()
                require(process.wait() == 0, f"zstd failed: {name}")
            except BaseException:
                process.stdin.close()
                process.terminate()
                process.wait()
                raise
        print(f"Completed {name}: {destination.stat().st_size / 1e9:.3f} GB", flush=True)

    source_entries = []
    for folder in ["0_source_data", "1_cleaned_data"]:
        for stem in STEMS:
            path = ROOT / "data" / folder / f"{stem}.csv"
            source_entries.append((path, path.relative_to(ROOT).as_posix()))
    source_entries.extend((path, path.relative_to(work).as_posix()) for path in sorted(split_dir.rglob("*")) if path.is_file())
    archive("source_cleaned_split.tar.zst", source_entries)
    # Detailed gaps report is large; stream it separately through tar's file reader.
    missing_entries = [(path, path.relative_to(ROOT).as_posix()) for path in sorted((ROOT / "data/3_missing_data").glob("*.csv"))]
    require(len(missing_entries) == 1080, "Unexpected missing-series count")
    archive("missing.tar.zst", missing_entries)
    fixed_entries = [(path, path.relative_to(ROOT).as_posix()) for path in sorted((ROOT / "data/4_fixed_data").rglob("*.csv"))]
    require(len(fixed_entries) == 18_360, "Unexpected reconstructed file count")
    expected_fixed = {
        f"data/4_fixed_data/{row.dataset_name}/{row.technique}-{row.structure}/{int(row.rate_percent)}p/{row.model}/{int(row.iteration)}.csv"
        for row in reconstruction.itertuples(index=False)
    }
    require({member for _, member in fixed_entries} == expected_fixed, "Reconstructed paths differ from frozen metric keys")
    archive("reconstructed.tar.zst", fixed_entries)
    inventory_frame = pd.DataFrame(inventory)
    inventory_frame["series_rows"] = inventory_frame.series_rows.astype("Int64")
    inventory_frame.to_parquet(release / "metadata/archive_members.parquet", index=False, compression="zstd")

    # Reports and provenance stay directly accessible, outside compressed series archives.
    for filename in ["missingness_realizations.csv", "empirical_mask_coverage.csv"]:
        shutil.copy2(ROOT / "data/3_missing_data/reports" / filename, release / "metadata" / filename)
    gaps_source = ROOT / "data/3_missing_data/reports/missingness_gaps.csv"
    gap_dir = release / "reports"
    gap_dir.mkdir()
    with (gap_dir / "missingness_gaps.csv.zst").open("wb") as target:
        subprocess.run(["zstd", "-T4", "-3", "-q", "-c", str(gaps_source)], stdout=target, check=True)
    (release / "results").mkdir(exist_ok=True)
    reconstruction.to_parquet(release / "results/reconstruction_metrics.parquet", index=False, compression="zstd")
    shutil.copy2(reconstruction_path, release / "results" / reconstruction_path.name)
    provenance = release / "provenance"
    provenance.mkdir()
    for path in [ROOT / "experiment_run_summary.json", ROOT / f"merged_results_manifest_{STAMP}.json"]:
        shutil.copy2(path, provenance / path.name)
    later_split = ROOT / "data/2_splitted_data/split_manifest.json"
    shutil.copy2(later_split, provenance / "working_split_manifest_2026-10-13.json")
    config = yaml.safe_load((ROOT / "config/config.yaml").read_text())
    config["split"]["sd2_validation"]["max_samples"] = 500
    config["release_provenance"] = {
        "status": "reconstructed configuration, not an archived 2026-10-10 config",
        "change_from_working_config": "split.sd2_validation.max_samples restored to 500; confirmed by frozen results",
        "authoritative_split": (RELATIVE / "metadata/split_manifest.json").as_posix(),
        "source_working_config_sha256": digest(ROOT / "config/config.yaml"),
    }
    (provenance / "experiment_config_reconstructed.yaml").write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    statistics = release / "results/batch_statistics"
    statistics.mkdir()
    for stamp in ["20261010_202407", "20261010_202408"]:
        for path in (ROOT / "prediction_experiment_results/batch_statistics").glob(f"*{stamp}.*"):
            shutil.copy2(path, statistics / path.name)

    reproducibility = release / "reproducibility"
    reproducibility.mkdir()
    code_roots = [ROOT / "src", ROOT / "scripts", ROOT / "config"]
    with tarfile.open(reproducibility / "pipeline_source.tar.gz", "w:gz") as tar:
        for folder in code_roots:
            for path in sorted(folder.rglob("*")):
                if path.is_file() and path.suffix in {".py", ".yaml"} and "__pycache__" not in path.parts:
                    tar.add(path, arcname=path.relative_to(ROOT).as_posix())
        for name in ["pyproject.toml", "uv.lock", "requirements.txt", "EXPERIMENT_STEPS.md", "REBUTTAL_README.md"]:
            tar.add(ROOT / name, arcname=name)
    (reproducibility / "README.md").write_text(
        "# Reproduction source\n\nThis is a source snapshot taken when the release was packaged, not a recovered "
        "2026-10-10 checkout. The authoritative frozen result tables are unchanged. "
        "For the historical split, use metadata/split_manifest.json and "
        "provenance/experiment_config_reconstructed.yaml instead of the working configuration "
        "inside pipeline_source.tar.gz. That configuration restores the demonstrated split only; "
        "other settings are the current working settings, not independently recovered historical settings.\n",
        encoding="utf-8",
    )
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    manifest = {
        "schema_version": "1.0", "repository": REPO_ID, "experiment_date": RELEASE_DATE,
        "packaged_at": datetime.now(timezone.utc).isoformat(), "packaging_source_commit": commit,
        "frozen_result_stamp": STAMP, "reconstruction_rows": len(reconstruction), "forecast_rows": 2_497_368,
        "forecast_rows_per_model": 624_342, "missing_realizations": 1080, "reconstructed_series": 18_360,
        "checks": stats, "historical_train_lengths": lengths,
        "split_resolution": "restored from frozen forecast base_train_samples and verified test tails; "
        "500 validation samples for Boiler/Pump, 24 for Vibration; working split from 2026-10-13 retained as provenance only",
        "frozen_sources": {str(path.relative_to(ROOT)): {"sha256": digest(path), "bytes": path.stat().st_size}
                           for path in [reconstruction_path, prediction_path]},
        "archives": {name: {"files": int((inventory_frame.archive == f"archives/{name}").sum()),
                            "bytes": (release / "archives" / name).stat().st_size}
                     for name in ["source_cleaned_split.tar.zst", "missing.tar.zst", "reconstructed.tar.zst"]},
        "legacy_policy": "Earlier root-level archives remain available as legacy data; this release lives in releases/2026-10-10.",
    }
    require(stats == {"missing_csvs": 1080, "reconstructed_csvs": 18_360, "recomputed_metric_checks": 36_720}, "Incomplete series validation")
    write_json(release / "release_manifest.json", manifest)
    for source, destination in [
        (TEMPLATES / "DATASET_CARD.md", repository / "README.md"),
        (TEMPLATES / "validate_package.py", output / "validate_package.py"),
        (TEMPLATES / "upload.sh", output / "upload.sh"),
        (TEMPLATES / "UPLOAD_INSTRUCTIONS.md", output / "UPLOAD_INSTRUCTIONS.md"),
    ]:
        shutil.copy2(source, destination)
    (output / "upload.sh").chmod(0o755)
    with (repository / "SHA256SUMS").open("w", encoding="utf-8") as handle:
        for path in sorted(repository.rglob("*")):
            if path.is_file() and path.name != "SHA256SUMS":
                handle.write(f"{digest(path)}  {path.relative_to(repository).as_posix()}\n")
    shutil.rmtree(work)
    build_licensed_package(output, public_output)
    print(f"Prepared {public_output}\nRun: {sys.executable} {public_output / 'validate_package.py'} --deep", flush=True)


if __name__ == "__main__":
    main()
