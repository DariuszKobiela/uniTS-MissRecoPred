---
pretty_name: Univariate Time Series with Missingness and Reconstructions
license:
- cc-by-4.0
- cc0-1.0
language:
- en
tags:
- time-series
- missing-data
- imputation
- reconstruction
- forecasting
- industrial-sensors
- stable-diffusion
configs:
- config_name: forecasting_metrics
  default: true
  data_files:
  - split: test
    path: releases/2026-10-10/results/prediction_by_model/*.parquet
- config_name: reconstruction_metrics
  data_files:
  - split: test
    path: releases/2026-10-10/results/reconstruction_metrics.parquet
---

# Univariate time series with missingness and reconstructions

This repository publishes source series, controlled missingness realizations,
reconstructed series, and evaluation results for the
[uniTS-MissRecoPred framework](https://github.com/DariuszKobiela/uniTS-MissRecoPred).
The current release contains a licensed subset of the final experiment results frozen on
**2026-10-10** (`20261010_202224`). Packaging and recovery of the historical split
were performed later and are recorded separately in the release manifest.

**Start with [`releases/2026-10-10/`](releases/2026-10-10/).**
The 2026-11-09 license correction publishes only Boiler and Vibration.
Mixed root-level legacy archives have been withdrawn from the current revision.
Pump data and Pump metric records are excluded because the source has no stated
redistribution license. The original experiment also used Pump; this publication
is an explicitly filtered subset of that experiment, with retained metric values
unchanged. See [LICENSE.md](LICENSE.md) for component licenses and attribution.

## Scope and experiment grid

Two univariate series are included: Boiler, an industrial sensor signal, and
Vibration, a selected signal from the synthetic IIoT Edge Computing Dataset.

| Dimension | Values |
|---|---|
| Series | `boiler_outlet_temp_univ`, `vibration_sensor_S1` |
| Missingness mechanisms | MCAR, MAR, MNAR |
| Temporal structures | scattered, contiguous, mixed |
| Requested missing rates | 2%, 5%, 20%, 50% |
| Repetitions | 1–10 |
| Degraded series | 720 = 2 x 3 x 3 x 4 x 10 |
| Reconstructors | 17 |
| Reconstructed series / metric rows | 12,240 = 720 x 17 |
| Forecasting algorithms | SARIMAX, XGBoost, Prophet, Holt–Winters |
| Forecast metric rows | See `release_manifest.json` for filtered counts |

The reconstructors are `impute_mean`, `impute_median`, `impute_ffill`,
`impute_bfill`, `interpolate_linear`, `interpolate_cubic`, `interpolate_pchip`,
`knn`, `sarimax`, and zero-shot and fine-tuned Stable Diffusion 2 for each of
GAF, MTF, RP, and spectrogram (`stable_diffusion_2_{gaf,mtf,rp,spec}` and the
corresponding `_finetuned` variants).

Forecast table rows are **metric records**, not independent time series or
individual predicted values. They include cumulative horizons and lead-time
bins, repetitions, origins, reconstructed histories, and original-history
baselines. Individual forecast trajectory files are not included.

## Files

All paths below are relative to `releases/2026-10-10/`.

| Path | Contents |
|---|---|
| `archives/source_cleaned_split.tar.zst` | 2 source CSVs, 2 cleaned CSVs, 6 historical split CSVs, recovered split manifest |
| `archives/missing.tar.zst` | 720 degraded CSVs |
| `archives/reconstructed.tar.zst` | 12,240 reconstructed CSVs |
| `metadata/split_manifest.json` | Historical ranges and rolling-origin offsets, with recovery provenance |
| `metadata/archive_members.parquet` | Path, size, SHA-256 and series row count for each archive member |
| `metadata/missingness_realizations.csv` | Requested/realized rates, counts and gap summaries for 720 realizations |
| `metadata/empirical_mask_coverage.csv` | Mask coverage diagnostics |
| `reports/missingness_gaps.csv.zst` | Detailed gap report, compressed losslessly |
| `results/reconstruction_metrics.parquet` | Frozen reconstruction metric table |
| `results/reconstruction_results_merged_20261010_202224.csv` | Filtered frozen reconstruction CSV |
| `results/prediction_by_model/*.parquet` | Frozen forecast metrics exported from the final merged CSV |
| `results/prediction_by_model/manifest.json` | Forecast source, counts, sizes and relative paths |
| `release_manifest.json` | Counts, source hashes, provenance and performed checks |

`SHA256SUMS` at the repository root checks the public files. The archive-member
manifest additionally checks the extracted CSVs. Paths in metric columns describe the original workspace; they are provenance, not
promises that every referenced intermediate artifact is published here.

## Historical temporal split

| Series | Full cleaned series | Reconstruction train | SD2 validation | Final rolling test | Origins |
|---|---:|---:|---:|---:|---:|
| Boiler | 86,400 | 68,620 | 500 | 17,280 | 5 |
| Vibration | 210 | 144 | 24 | 42 | 3 |

These are consecutive, disjoint temporal blocks. Missingness and reconstruction
apply to the reconstruction-train block. At later forecast origins the observed
prefix of the test block is revealed as expanding history. Horizons are
`[12, 96, 180, 720, 1440]` for Boiler and
`[12, 24]` for Vibration. `test_start` and `test_stop_exclusive` in the forecast
tables are offsets inside the test block, not absolute positions in the full series.

The working split manifest was regenerated on 2026-10-13 with 4,096 validation
samples for Boiler and Pump in the original experiment. That later split does
not match the frozen results.
For this release the historical train lengths were recovered from all four
frozen forecast tables, checked against the degraded and reconstructed CSVs,
and used to slice the cleaned source series. The test CSVs were checked against
the cleaned tails. Mixed provenance bundles, batch statistics and the old source
snapshot are omitted from this filtered release. The recovered train/validation
CSVs represent the same numerical
samples; they are newly serialized files, not recovered byte-for-byte originals.

## CSV schema and naming

CSV files preserve their original two-column schema:

| Series | Time column | Value column |
|---|---|---|
| Boiler | `date` | `TE_8332A.AV_0#` |
| Vibration | `Timestamp` | `Vibration (Hz)` |

Degraded series contain empty values at hidden positions. Cleaned data is the
ground truth used for evaluation; upstream gaps may have been filled before
controlled missingness was injected. Original source CSVs are included separately.

```text
data/3_missing_data/{dataset}_{MECHANISM}-{structure}_{rate}p_{iteration}.csv
data/4_fixed_data/{dataset}/{MECHANISM}-{structure}/{rate}p/{model}/{iteration}.csv
```

Reconstruction metrics are evaluated only at hidden positions. The frozen
reconstruction table includes MAD, MAE, sMAPE, absolute-error summaries and
resource measurements. MAD and MAE in this experiment are numerically identical.
Missing reconstruction RMSE/R2 columns have not been silently derived or added.
Forecast tables include MAE, RMSE, MASE, MAPE and sMAPE plus horizon/origin,
fallback and runtime metadata. `sarimax` is the code identifier; SARIMAX fallback
details remain visible in their original columns.

Every degraded series was checked against the cleaned prefix and its realization
report. Every reconstruction was checked for the correct time index, finite
values and preservation of observed samples. MAE and the population standard
deviation of absolute errors were recomputed for all 12,240 reconstructions and
compared with the frozen results. Floating-point tolerances are recorded in the
local packaging script (`rtol=1e-7`, `atol=1e-8` for these metric comparisons).

## Load the metric tables

```python
from datasets import load_dataset

repo = "dariusz-kobiela/univariate-time-series-with-missingness-and-reconstructed"
forecast = load_dataset(repo, "forecasting_metrics", split="test")
reconstruction = load_dataset(repo, "reconstruction_metrics", split="test")
```

Both configurations use the split name `test` to identify evaluation records;
these tables are not a train/test split for fitting a model.

## Download and extract the series

```bash
hf download dariusz-kobiela/univariate-time-series-with-missingness-and-reconstructed \
  --repo-type dataset --include 'releases/2026-10-10/*' --local-dir ./experiment-data
cd experiment-data
tar --zstd -xf releases/2026-10-10/archives/source_cleaned_split.tar.zst
tar --zstd -xf releases/2026-10-10/archives/missing.tar.zst
tar --zstd -xf releases/2026-10-10/archives/reconstructed.tar.zst
```

Extraction needs GNU tar with Zstandard support and the `zstd` executable.
The archives preserve the `data/...` directory structure. For a reconstruction example:

```python
import pandas as pd

stem = "vibration_sensor_S1"
truth = pd.read_csv(f"data/2_splitted_data/train/{stem}.csv", index_col=0)
degraded = pd.read_csv(f"data/3_missing_data/{stem}_MNAR-scattered_5p_1.csv", index_col=0)
fixed = pd.read_csv(
    f"data/4_fixed_data/{stem}/MNAR-scattered/5p/interpolate_linear/1.csv", index_col=0
)
mask = degraded.iloc[:, 0].isna()
mae = (truth.iloc[:, 0][mask] - fixed.iloc[:, 0][mask]).abs().mean()
print(mae)
```

## Provenance, attribution and limitations

| Published component | Source | License |
|---|---|---|
| Boiler | Nikita Manaenkov, [Time-Series of Industrial Boiler Operations](https://www.kaggle.com/datasets/nikitamanaenkov/time-series-of-industrial-boiler-operations) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) |
| Vibration | Ziya / Ziya07, [IIoT Edge Computing Dataset](https://www.kaggle.com/datasets/ziya07/iiot-edge-computing-dataset) | [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) |

These licenses cover each corresponding source signal and its transformed
versions. Changes include signal selection, cleaning and gap filling, temporal
splitting, controlled missingness and reconstruction. Retain Boiler attribution,
source/license links and an indication of changes when redistributing it.
Original contributed evaluation metrics and release documentation are CC BY 4.0.
See [LICENSE.md](LICENSE.md) for full attribution and license scope.

**Pump is not redistributed.** Its upstream metadata reports an unknown license.
Obtain it directly from [Pump Sensor Data by nphantawee](https://www.kaggle.com/datasets/nphantawee/pump-sensor-data)
and check permissions with the provider. No license is granted here for Pump.
Earlier Git revisions may contain withdrawn material; use the corrected revision.

The release covers two selected signals, including one synthetic signal.
Controlled MAR/MNAR and temporal masks do not represent every real sensor failure.
Repeated masks, horizons and origins are dependent measurements. Model weights,
synthetic SD2 training images and intermediate image representations are separate
artifacts and are not part of this experiment-data release.

When citing the data, include this dataset repository, release date
`2026-10-10`, and the Hugging Face commit hash used for download.
