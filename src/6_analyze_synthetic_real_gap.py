#!/usr/bin/env python3
"""Quantify the statistical gap between synthetic and real training windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from joblib import Parallel, delayed

FEATURE_COLUMNS = (
    "mean",
    "std",
    "skewness",
    "kurtosis",
    "acf_1",
    "acf_5",
    "acf_10",
    "acf_24",
    "trend_slope",
    "trend_strength",
    "seasonality_strength",
    "spectral_entropy",
    "dominant_frequency",
    "length",
    "diff_mean",
    "diff_std",
    "diff_abs_mean",
    "diff_abs_p95",
    "turning_point_rate",
)


def autocorrelation(values: np.ndarray, lag: int) -> float:
    if lag <= 0 or len(values) <= lag:
        return np.nan
    left, right = values[:-lag], values[lag:]
    if np.std(left) == 0 or np.std(right) == 0:
        return 0.0
    return float(np.corrcoef(left, right)[0, 1])


def extract_features(values: np.ndarray) -> dict[str, float]:
    """Extract scale, dependence, trend, spectral, and change features."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < 4:
        raise ValueError("At least four finite samples are required")

    time = np.arange(len(values), dtype=float)
    slope, intercept = np.polyfit(time, values, 1)
    trend = slope * time + intercept
    residual = values - trend
    variance = float(np.var(values))
    trend_strength = 0.0 if variance == 0 else max(0.0, 1.0 - float(np.var(residual)) / variance)

    spectrum = np.abs(np.fft.rfft(residual)) ** 2
    frequencies = np.fft.rfftfreq(len(residual))
    spectrum = spectrum[1:]
    frequencies = frequencies[1:]
    total_power = float(spectrum.sum())
    if total_power > 0 and len(spectrum):
        probabilities = spectrum / total_power
        spectral_entropy = (
            float(stats.entropy(probabilities) / np.log(len(probabilities)))
            if len(probabilities) > 1
            else 0.0
        )
        peak = int(np.argmax(spectrum))
        dominant_frequency = float(frequencies[peak])
        seasonality_strength = float(spectrum[peak] / total_power)
    else:
        spectral_entropy = dominant_frequency = seasonality_strength = 0.0

    differences = np.diff(values)
    signs = np.sign(differences)
    turning_points = np.sum(signs[1:] * signs[:-1] < 0) if len(signs) > 1 else 0
    return {
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=1)),
        "skewness": float(stats.skew(values, bias=False)),
        "kurtosis": float(stats.kurtosis(values, fisher=True, bias=False)),
        "acf_1": autocorrelation(values, 1),
        "acf_5": autocorrelation(values, 5),
        "acf_10": autocorrelation(values, 10),
        "acf_24": autocorrelation(values, 24),
        "trend_slope": float(slope),
        "trend_strength": trend_strength,
        "seasonality_strength": seasonality_strength,
        "spectral_entropy": spectral_entropy,
        "dominant_frequency": dominant_frequency,
        "length": float(len(values)),
        "diff_mean": float(np.mean(differences)),
        "diff_std": float(np.std(differences, ddof=1)),
        "diff_abs_mean": float(np.mean(np.abs(differences))),
        "diff_abs_p95": float(np.percentile(np.abs(differences), 95)),
        "turning_point_rate": float(turning_points / max(1, len(values) - 2)),
    }


def load_manifest_windows(dataset_dir: Path) -> list[dict]:
    manifest = dataset_dir / "manifest.jsonl"
    if not manifest.is_file():
        raise FileNotFoundError(f"Missing manifest: {manifest}")
    # Four encoding records refer to the same numeric series; keep one per ID.
    records: dict[int, dict] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        record = json.loads(line)
        records.setdefault(int(record["series_id"]), record)

    windows = []
    for record in records.values():
        relative = record.get("series_path")
        if not relative:
            raise ValueError(
                "Manifest has no `series_path`. Regenerate the dataset with the updated generator."
            )
        path = dataset_dir / relative
        windows.append(
            {
                "values": np.load(path),
                "source_kind": record["source_kind"],
                "source_name": record["source_name"],
                "series_id": str(record["series_id"]),
            }
        )
    return windows


def load_real_windows(
    cleaned_dir: Path,
    window_lengths: list[int],
    windows_per_series: int,
) -> list[dict]:
    windows = []
    for path in sorted(cleaned_dir.glob("*.csv")):
        frame = pd.read_csv(path, index_col=0)
        values = pd.to_numeric(frame.iloc[:, 0], errors="coerce").dropna().to_numpy()
        for length in window_lengths:
            if len(values) < length:
                continue
            starts = np.linspace(0, len(values) - length, windows_per_series, dtype=int)
            for number, start in enumerate(np.unique(starts)):
                windows.append(
                    {
                        "values": values[start : start + length],
                        "source_kind": "real",
                        "source_name": path.stem,
                        "series_id": f"{path.stem}_{length}_{number}",
                    }
                )
    return windows


def feature_frame(windows: list[dict], workers: int = 20) -> pd.DataFrame:
    def one(window: dict) -> dict:
        return {**{k: window[k] for k in ("source_kind", "source_name", "series_id")},
                **extract_features(window["values"])}
    rows = Parallel(n_jobs=workers, backend="loky")(delayed(one)(window) for window in windows)
    return pd.DataFrame(rows)


def classifier_diagnostics(features: pd.DataFrame, seed: int, workers: int = 20) -> dict:
    data = features.dropna(subset=list(FEATURE_COLUMNS)).copy()
    y = (data["source_kind"] == "synthetic").astype(int).to_numpy()
    groups = data["source_name"].astype(str).to_numpy()
    class_group_counts = data.groupby("source_kind")["source_name"].nunique()
    n_splits = min(5, int(class_group_counts.min()))
    if len(np.unique(y)) < 2 or n_splits < 2:
        return {"status": "insufficient_grouped_data"}
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(class_weight="balanced", max_iter=2000, random_state=seed),
    )
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    probabilities = cross_val_predict(
        model, data[list(FEATURE_COLUMNS)], y, groups=groups, cv=cv, method="predict_proba", n_jobs=workers
    )[:, 1]
    predictions = probabilities >= 0.5
    return {
        "status": "ok",
        "n_splits": n_splits,
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "balanced_accuracy": float(
            0.5
            * (
                accuracy_score(y[y == 0], predictions[y == 0])
                + accuracy_score(y[y == 1], predictions[y == 1])
            )
        ),
        "interpretation": "0.5 indicates overlap; values near 1.0 indicate a large synthetic-to-real gap",
    }


def write_report(output: Path, features: pd.DataFrame, diagnostics: dict) -> None:
    summary = features.groupby("source_kind")[list(FEATURE_COLUMNS)].agg(["mean", "std"])
    lines = [
        "# Synthetic-to-real gap analysis",
        "",
        f"- Synthetic windows: {(features.source_kind == 'synthetic').sum()}",
        f"- Real windows: {(features.source_kind == 'real').sum()}",
        f"- Grouped classifier ROC AUC: {diagnostics.get('roc_auc', 'not available')}",
        f"- Grouped classifier balanced accuracy: {diagnostics.get('balanced_accuracy', 'not available')}",
        "- Cross-validation groups windows by source series to limit leakage.",
        "- A classifier score near 0.5 indicates overlap; a score near 1.0 indicates easy separation.",
        "",
        "## Feature summary",
        "",
        "| feature | synthetic mean | synthetic std | real mean | real std |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for feature in FEATURE_COLUMNS:
        lines.append(
            f"| {feature} | {summary.loc['synthetic', (feature, 'mean')]:.6g} | "
            f"{summary.loc['synthetic', (feature, 'std')]:.6g} | "
            f"{summary.loc['real', (feature, 'mean')]:.6g} | "
            f"{summary.loc['real', (feature, 'std')]:.6g} |"
        )
    (output / "synthetic_real_gap_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="data/sd2_windowed_training")
    parser.add_argument("--cleaned-dir", default="data/1_cleaned_data")
    parser.add_argument("--output-dir", default="data/1_6_sd2_optimization/synthetic_real_gap")
    parser.add_argument("--real-windows-per-series", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=20)
    args = parser.parse_args()

    synthetic_windows = [
        item for item in load_manifest_windows(Path(args.dataset_dir))
        if item["source_kind"] == "synthetic"
    ]
    if not synthetic_windows:
        raise ValueError("No synthetic numeric windows found")
    lengths = sorted({len(item["values"]) for item in synthetic_windows})
    real_windows = load_real_windows(
        Path(args.cleaned_dir), lengths, args.real_windows_per_series
    )
    if not real_windows:
        raise ValueError("No real windows could be extracted")

    features = feature_frame([*synthetic_windows, *real_windows], workers=args.workers)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    features.to_csv(output / "synthetic_real_features.csv", index=False)

    complete = features.dropna(subset=list(FEATURE_COLUMNS)).copy()
    transformed = StandardScaler().fit_transform(complete[list(FEATURE_COLUMNS)])
    coordinates = PCA(n_components=2, random_state=args.seed).fit_transform(transformed)
    projection = complete[["source_kind", "source_name", "series_id"]].copy()
    projection[["pca_1", "pca_2"]] = coordinates
    projection.to_csv(output / "synthetic_real_pca.csv", index=False)

    diagnostics = classifier_diagnostics(features, args.seed, workers=args.workers)
    (output / "synthetic_real_classifier.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    write_report(output, features, diagnostics)
    print(f"Synthetic-to-real analysis written to {output}")


if __name__ == "__main__":
    main()
