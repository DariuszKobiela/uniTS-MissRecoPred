# Rebuttal experiment: audit status and complete runbook

This document is the authoritative runbook for the rebuttal experiment. Run
commands from the repository root:

```bash
cd /home/darek/uniTS-MissRecoPred
```

## 1. Current decision: code blockers resolved; full GPU rerun still required

The rebuttal protocol is now implemented in code. Steps 7--9 remain available
as **legacy** diagnostics only. The authoritative forecasting path is:

```bash
make evaluate-rolling-origins   # or make rebuttal-forecast
make batch-statistics
make run-manifest
```

Resolved protocol elements:

1. **Three-way split:** `reconstruction-train`, `sd2_validation`, and
  `rolling_test` with `data/2_splitted_data/split_manifest.json`.
2. **Rolling-origin forecasting:** one local fit per origin to `H_max`, then
  cumulative and lead-time-bin metrics for every configured horizon.
3. **Baselines:** XGBoost, SARIMAX, persistence, seasonal-naive, and direct
  XGBoost on degraded data with native NaNs and missingness indicators.
4. **Horizons/origins:** boiler `[12,96,180,720,1440]` with five origins;
  pump `[12,96,360,1440]` with five origins; vibration `[12,24]` event steps
   with three origins.
5. **SD2 HPO partition:** `analyze_sd2_design.py --run-inference` reads only
  `sd2_validation`, uses production `context_samples`, and writes
   `sd2_hpo_disjointness_proof.json`. Legacy `optimize_sd_hyperparams.py` is
   blocked when the validation partition exists.
6. **Fine-tuned SD2 fail-closed:** `SD2_FAIL_CLOSED=1` (default) prevents
  silent fallback to legacy Hub weights; train
   `models/sd2_windowed/best_model/` before fine-tuned reconstruction.
7. **Reproducibility:** `uv.lock` is tracked; `make run-manifest` records run
  ID, Git state, config checksums, and seed.
8. **Batch statistics:** `src/11_batch_statistics.py` exports Friedman and
  Holm-adjusted paired comparisons with the full rolling-origin pairing key.

Before reporting paper results you must still execute the full rerun on GPU
data (Phase 1 onward below) and verify `make rebuttal-preflight` passes on
the frozen run directory.

The manuscript-specific requests about related work, tables, and figures are
not verifiable here because `softwarex_article.tex` is not present in this
repository.

## 2. Audit result



### Corrected

- MCAR samples observed positions uniformly.
- MAR depends on the fully observed time position.
- MNAR depends on the value that is subsequently hidden.
- MCAR/MAR/MNAR are crossed with `scattered`, `contiguous`, and `mixed`
structures.
- Degradation reports gap count, mean/median/p90/maximum gap length,
singleton percentage, sample length, and time coverage.
- SD2 reconstruction uses local overlapping windows and disjoint output cores.
- The GASF diagonal inverse is corrected.
- MTF is a quantile-binned transition field with an explicit approximate
decoder.
- RP uses a continuous pairwise-distance representation and a one-dimensional
geometric reconstruction.
- Spectrogram decoding uses inverse STFT with retained conditioning phase.
- CPU round-trip and clean-image oracle controls are implemented.
- Explicit SD2 generators/seeds and a five-seed sensitivity procedure exist.
- SARIMAX uses configured seasonality, `simple_differencing=False`, forecasts
in the original scale, and reports fallbacks in rolling-origin outputs.
- Prediction evaluation rejects unequal lengths, shifted indices, duplicate
indices, NaNs, and infinities instead of silently truncating.
- Paired t/Wilcoxon tests, Shapiro testing of paired differences, Holm
correction, effect sizes, bootstrap 95% confidence intervals, and the
Friedman omnibus test are implemented.



### Partly corrected

- Zero-shot and fine-tuned SD2 variants exist for every representation, but
corrected fine-tuned weights and final comparative results do not.
- Mask coverage is estimated analytically for local windows, not measured from
the actual masks generated in the final degradation run.
- Synthetic-to-real feature analysis exists, but generator parameter draws are
not recorded per generated series and no immutable generator version is
stored in the dataset manifest.
- Rolling-origin local XGBoost/SARIMAX exists for boiler and pump, but is
disconnected from the standard steps 7--9 and omits vibration.
- Forecast metrics include MAE, RMSE, MASE, MAPE, and sMAPE, but MAPE remains
the configured primary metric.
- Fine-tuning uses a grouped train/validation split by `series_id`, but
inference hyperparameter search lacks a separate final-validation dataset.
- Iteration identifies a degradation realization in reconstruction results,
while the exact generated seed is stored only in the missingness report.



### Not corrected

- A single consistent local forecasting protocol in the main pipeline.
- Boiler horizon 1440 and three rolling origins for vibration.
- Forecast-once-to-`H_max`, cumulative `metric@H`, and lead-time-bin metrics.
- Persistence and seasonal-naive forecasting baselines.
- Direct forecasting under missingness with NaNs plus missingness indicators.
- Independent validation for SD2 inference hyperparameter selection.
- Empirical final-run mask coverage.
- Batch statistical result files with the complete rolling-origin pairing key.
- Automated run manifests and committed dependency lock.



## 3. Verified repository state

At the time this runbook was created:

- `uv run pytest -q` passes: **119 tests passed**;
- `uv lock --check` passes locally;
- Python compilation of `src/` and `tests/` passes;
- Makefile dry-runs for the documented targets pass;
- Ruff does not pass: 1394 diagnostics, mostly legacy whitespace/import issues;
- raw, cleaned, split, degraded, reconstructed, and prediction data needed for
a rerun are not currently present;
- `data/sd2_windowed_training/manifest.jsonl` is a stale partial artifact;
- `sd2_runtime_overrides.json`, SD2 seed-sensitivity outputs, and corrected
local fine-tuned weights are absent.



## 4. Required decisions before changing code

Use these choices consistently:

1. Treat the forecasting question as local-model evaluation: fit both XGBoost
  and SARIMAX separately for each reconstructed series and origin.
2. Use horizons:
  - boiler: `12, 96, 180, 720, 1440`;
  - pump: `12, 96, 360, 1440`;
  - vibration: `12, 24` event steps, without converting them to fixed hours.
3. Use five rolling origins for boiler/pump and three for vibration.
4. Forecast once to the largest valid horizon at an origin, then compute
  cumulative metrics and disjoint lead-time intervals from that vector.
5. Make MAE, RMSE, and MASE primary; retain MAPE and sMAPE as secondary.
6. Compare against persistence and seasonal-naive forecasts.
7. Add a direct XGBoost baseline that receives the degraded series, native
  NaNs, and missingness indicators without reconstruction.
8. Either:
  - preregister fixed SD2 settings and do no inference HPO; or
  - reserve an independent validation segment/dataset for HPO and never use it
  in the reported reconstruction test.
9. Decide how the 16 pre-existing pump gaps are handled. Do not run the
  temporary repair script silently; document the chosen rule.



## 5. Phase 0 — implement and test the remaining blockers

There is no valid command-only workaround for the blockers in Section 1.
After implementing them, add tests for:

- local XGBoost and SARIMAX using identical source/origin sets;
- dataset-specific origin counts;
- one `H_max` forecast sliced into cumulative and lead-time metrics;
- persistence, seasonal-naive, and direct-missing XGBoost;
- irregular vibration labels expressed only as event steps;
- an SD2 HPO partition disjoint from final evaluation;
- empirical mask coverage from generated masks;
- batch statistics paired by dataset, mechanism, rate, structure, degradation
realization/seed, horizon, and origin.

Then run:

```bash
uv run pytest -q
uv run python -m compileall -q src tests
uv lock --check
```

Ruff is currently a legacy quality failure, not a valid pass/fail gate for the
experiment:

```bash
uv run ruff check src tests --statistics
```

Do not apply `ruff --fix` as part of the experimental rerun; keep formatting
cleanup in a separate reviewable change.

## 6. Phase 1 — freeze code, environment, and configuration



### 6.1 Commit all methodological changes

```bash
git status --short
git diff --check
git add src tests config Makefile README.md docs REBUTTAL_README.md pyproject.toml
git add -f uv.lock
git commit -m "Freeze rebuttal experiment protocol"
git status --short
```

The last command must print nothing. Because `uv.lock` is currently ignored,
remove its ignore rule or retain the explicit force-add command above.

### 6.2 Create an immutable run directory

```bash
export RUN_ID="rebuttal_$(date -u +%Y%m%dT%H%M%SZ)"
grep -qxF '/runs/' .git/info/exclude || printf '/runs/\n' >> .git/info/exclude
mkdir -p "runs/${RUN_ID}/"{config,environment,manifests,logs}
git rev-parse HEAD | tee "runs/${RUN_ID}/environment/git_commit.txt"
git status --porcelain=v1 | tee "runs/${RUN_ID}/environment/git_status.txt"
cp config/config.yaml "runs/${RUN_ID}/config/config.yaml"
cp config/prediction_models_config.yaml \
  "runs/${RUN_ID}/config/prediction_models_config.yaml"
```

`git_status.txt` must be empty.

### 6.3 Install the frozen environment

```bash
uv sync --frozen
uv run python --version | tee "runs/${RUN_ID}/environment/python_version.txt"
uv --version | tee "runs/${RUN_ID}/environment/uv_version.txt"
uv pip freeze | tee "runs/${RUN_ID}/environment/python_packages.txt"
uv run python - <<'PY' | tee "runs/${RUN_ID}/environment/torch_cuda.txt"
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("cuda_version", torch.version.cuda)
print("device_count", torch.cuda.device_count())
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
PY
```

CUDA must be available for SD2 inference and fine-tuning.

## 7. Phase 2 — prepare configuration and source data

Place the three source CSV files in `data/0_source_data/`. Verify:

```bash
uv run python - <<'PY'
from pathlib import Path
files = sorted(Path("data/0_source_data").glob("*.csv"))
print("\n".join(map(str, files)))
assert len(files) == 3, f"Expected 3 source CSV files, found {len(files)}"
PY
```

Before the run, verify `config/config.yaml` contains the final approved values:

```yaml
split:
  horizons:
    experiment:
      boiler_outlet_temp_univ.csv: [12, 96, 180, 720, 1440]
      pump_sensor_28_univ.csv: [12, 96, 360, 1440]
      vibration_sensor_S1.csv: [12, 24]

missingness_structures:
  selected: [scattered, contiguous, mixed]

missingness_rates:
  rates: [0.03, 0.08, 0.20]
  iterations: 10
  seed: 42

prediction:
  error_metrics:
    compute: [mae, rmse, mase, mape, smape]
    primary_metric: mae

overwrite:
  reconstruction: true
  prediction: true
```

If the rebuttal promises 2%, 5%, 20%, and 50%, use exactly
`[0.02, 0.05, 0.20, 0.50]` instead. Do not mix rates from different versions
of the paper.

After every config edit, refresh the frozen copy and commit:

```bash
cp config/config.yaml "runs/${RUN_ID}/config/config.yaml"
cp config/prediction_models_config.yaml \
  "runs/${RUN_ID}/config/prediction_models_config.yaml"
git add config REBUTTAL_README.md
git commit -m "Finalize rebuttal run configuration"
git tag -a rebuttal-experiment-v1 -m "Rebuttal experiment protocol v1"
git rev-parse HEAD | tee "runs/${RUN_ID}/environment/git_commit.txt"
```



## 8. Phase 3 — archive and remove stale generated artifacts

`make clean-all` is destructive. First archive any old results that must be
retained:

```bash
tar -czf "runs/${RUN_ID}/pre_rerun_generated_artifacts.tar.gz" \
  data/1_cleaned_data \
  data/1_5_horizon_recommendation \
  data/2_splitted_data \
  data/3_missing_data \
  data/4_fixed_data \
  reconstruction_experiments_results \
  prediction_experiment_results \
  trained_prediction_models \
  hyperparameter_optimization \
  2>/dev/null || true
make clean-all
```

The current Makefile does not clear all SD2-specific artifacts. Remove only the
stale corrected-training and local-model directories after the archive:

```bash
rm -rf data/sd2_windowed_training
rm -rf models/sd2_windowed
rm -f data/1_6_sd2_optimization/sd2_runtime_overrides.json
rm -f data/1_6_sd2_optimization/sd2_inference_trials.csv
rm -f data/1_6_sd2_optimization/sd2_seed_sensitivity.csv
rm -f data/1_6_sd2_optimization/sd2_seed_sensitivity_summary.csv
```

Do not delete `data/0_source_data/`.

## 9. Phase 4 — clean data, select horizons, and split

```bash
make clean-datasets 2>&1 | tee "runs/${RUN_ID}/logs/01_clean.log"
make recommend-horizons 2>&1 | tee "runs/${RUN_ID}/logs/02_horizons.log"
make create-split 2>&1 | tee "runs/${RUN_ID}/logs/03_split.log"
```

Validate the split and irregular-series description:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

meta = json.loads(Path(
    "data/1_5_horizon_recommendation/dataset_metadata.json"
).read_text())
for name, item in meta["series"].items():
    print(name, item["horizons"], item["train_length"], item["is_irregular"])
assert max(meta["series"]["boiler_outlet_temp_univ.csv"]["horizons"]) == 1440
assert meta["series"]["pump_sensor_28_univ.csv"]["horizons"] == [12, 96, 360, 1440]
assert meta["series"]["vibration_sensor_S1.csv"]["horizons"] == [12, 24]
assert meta["series"]["vibration_sensor_S1.csv"]["is_irregular"] is True
PY
```

Inspect `horizon_recommendations.md`. Vibration must be described as 12 and 24
event steps, not as fixed hours.

## 10. Phase 5 — representation ceilings and empirical masks

Run the CPU-only design and representation controls:

```bash
make analyze-sd2-design 2>&1 | tee "runs/${RUN_ID}/logs/04_sd2_design.log"
make analyze-sd2-ablation 2>&1 | tee "runs/${RUN_ID}/logs/05_sd2_ablation.log"
```

Required outputs:

```bash
test -s data/1_6_sd2_optimization/sd2_design_analysis.csv
test -s data/1_6_sd2_optimization/sd2_representation_ablations.csv
test -s data/1_6_sd2_optimization/sd2_design_report.md
```

The final implementation must additionally write empirical mask coverage for
each dataset × mechanism × structure × rate × iteration × representation.
Do not substitute the analytical expectation in
`sd2_design_analysis.csv` for this final-run measurement.

## 11. Phase 6 — generate the corrected SD2 training set

Generate 2000 base windows and four encodings per window. `--overwrite` is
mandatory because the repository previously contained a stale partial
manifest:

```bash
uv run python src/training/generate_sd2_windowed_dataset.py \
  --samples 2000 \
  --output data/sd2_windowed_training \
  --image-size 512 \
  --window-sizes 512,1024,2048 \
  --rates 0.03,0.08,0.20 \
  --mechanisms MCAR,MAR,MNAR \
  --structures scattered,contiguous,mixed \
  --source synthetic \
  --seed 42 \
  --overwrite \
  2>&1 | tee "runs/${RUN_ID}/logs/06_generate_sd2_training.log"
```

Use the same rates as the final degradation run. Validate:

```bash
uv run python - <<'PY'
import json
from pathlib import Path

root = Path("data/sd2_windowed_training")
rows = [json.loads(line) for line in (root / "manifest.jsonl").open()]
assert len(rows) == 8000, len(rows)
assert len({row["series_id"] for row in rows}) == 2000
required = {
    "series_path", "mechanism", "structure", "n_gaps",
    "gap_length_samples_mean", "gap_length_samples_median",
    "gap_length_samples_p90", "gap_length_samples_max",
    "singleton_gap_percent", "seed",
}
assert required <= rows[0].keys()
assert {row["encoding"] for row in rows} == {"gaf", "mtf", "rp", "spec"}
assert (root / "dataset_summary.json").is_file()
assert (root / "series").is_dir()
print("validated", len(rows), "triplets")
PY
```



## 12. Phase 7 — synthetic-to-real gap

```bash
make analyze-synthetic-real-gap \
  2>&1 | tee "runs/${RUN_ID}/logs/07_synthetic_real_gap.log"
```

Required outputs:

```bash
test -s data/1_6_sd2_optimization/synthetic_real_gap/synthetic_real_features.csv
test -s data/1_6_sd2_optimization/synthetic_real_gap/synthetic_real_pca.csv
test -s data/1_6_sd2_optimization/synthetic_real_gap/synthetic_real_classifier.json
test -s data/1_6_sd2_optimization/synthetic_real_gap/synthetic_real_gap_report.md
```

For publication, also retain the generator code commit and dataset manifest.
The current manifest does not store every randomly sampled generator parameter;
add those fields before claiming exact per-sample generator provenance.

## 13. Phase 8 — fine-tune the corrected SD2 model

```bash
uv run python src/training/finetune_sd2_windowed.py \
  --data-dir data/sd2_windowed_training \
  --output-dir models/sd2_windowed \
  --encodings gaf,mtf,rp,spec \
  --validation-share 0.15 \
  --epochs 30 \
  --early-stop-patience 5 \
  --learning-rate 1e-5 \
  --batch-size 1 \
  --gradient-accumulation-steps 8 \
  --mixed-precision fp16 \
  --seed 42 \
  2>&1 | tee "runs/${RUN_ID}/logs/08_finetune_sd2.log"
```

Validate the local corrected model:

```bash
test -s models/sd2_windowed/best_model/model_index.json
test -s models/sd2_windowed/training_report.json
```

Do not run fine-tuned reconstruction if these checks fail.

## 14. Phase 9 — choose SD2 inference settings without leakage



### Recommended until independent HPO validation is implemented

Preregister fixed `window_samples`, `image_size`, `num_inference_steps`,
`guidance_scale`, context, and prompt in the frozen config. Ensure no runtime
override exists:

```bash
test ! -e data/1_6_sd2_optimization/sd2_runtime_overrides.json
```

Use identical fixed inference settings for zero-shot and fine-tuned variants.

### Allowed only after independent validation is implemented

Run the search separately for base and corrected fine-tuned variants:

```bash
uv run python src/optimization/analyze_sd2_design.py \
  --run-inference \
  --image-sizes 512 \
  --n-trials 12 \
  --cases-per-dataset 3 \
  --sd2-seeds 42,43,44,45,46 \
  2>&1 | tee "runs/${RUN_ID}/logs/09a_sd2_hpo_base.log"

SD2_FINETUNED_MODEL="$PWD/models/sd2_windowed/best_model" \
uv run python src/optimization/analyze_sd2_design.py \
  --run-inference \
  --image-sizes 512 \
  --models \
    stable_diffusion_2_gaf_finetuned \
    stable_diffusion_2_mtf_finetuned \
    stable_diffusion_2_rp_finetuned \
    stable_diffusion_2_spec_finetuned \
  --n-trials 12 \
  --cases-per-dataset 3 \
  --sd2-seeds 42,43,44,45,46 \
  2>&1 | tee "runs/${RUN_ID}/logs/09b_sd2_hpo_finetuned.log"
```

The implementation must prove that HPO data are disjoint from final test data
and must use the same context contract as production before these commands are
valid for the final paper.

Never use this legacy command for final model selection:

```bash
# DO NOT RUN FOR FINAL RESULTS:
# make optimize
```



## 15. Phase 10 — create missingness realizations and reports

```bash
make degrade-datasets 2>&1 | tee "runs/${RUN_ID}/logs/10_degrade.log"
make analyze-missingness 2>&1 | tee "runs/${RUN_ID}/logs/11_missingness_report.log"
```

Validate the expected number of realizations:

```bash
uv run python - <<'PY'
import pandas as pd
from pathlib import Path

report = Path("data/3_missing_data/reports/missingness_realizations.csv")
df = pd.read_csv(report)
expected = (
    df["dataset_name"].nunique()
    * df["mechanism"].nunique()
    * df["structure"].nunique()
    * df["requested_missing_rate"].nunique()
    * df["iteration"].nunique()
)
assert len(df) == expected, (len(df), expected)
required = {
    "seed", "n_gaps", "gap_length_samples_mean",
    "gap_length_samples_median", "gap_length_samples_p90",
    "gap_length_samples_max", "singleton_gap_percent",
    "gap_coverage_seconds_mean", "gap_coverage_seconds_median",
    "gap_coverage_seconds_p90", "gap_coverage_seconds_max",
}
assert required <= df.columns
assert set(df["mechanism"]) == {"MCAR", "MAR", "MNAR"}
assert set(df["structure"]) == {"scattered", "contiguous", "mixed"}
print("validated", len(df), "missingness realizations")
PY
```



## 16. Phase 11 — reconstruct all final methods

Select the exact classical and SD2 methods in `config/config.yaml`; do not leave
`reconstruction_models.selected: []` unless the paper truly reports every
registered method.

Run classical methods first:

```bash
uv run python src/4_reconstruct_datasets.py \
  --models \
    impute_mean impute_median impute_ffill impute_bfill \
    interpolate_linear interpolate_cubic interpolate_pchip \
    knn sarimax \
  2>&1 | tee "runs/${RUN_ID}/logs/12a_reconstruct_classical.log"
```

Run all four zero-shot SD2 representations:

```bash
uv run python src/4_reconstruct_datasets.py \
  --models \
    stable_diffusion_2_gaf \
    stable_diffusion_2_mtf \
    stable_diffusion_2_rp \
    stable_diffusion_2_spec \
  2>&1 | tee "runs/${RUN_ID}/logs/12b_reconstruct_sd2_base.log"
```

Run all four corrected fine-tuned representations and force the local model:

```bash
SD2_FINETUNED_MODEL="$PWD/models/sd2_windowed/best_model" \
uv run python src/4_reconstruct_datasets.py \
  --models \
    stable_diffusion_2_gaf_finetuned \
    stable_diffusion_2_mtf_finetuned \
    stable_diffusion_2_rp_finetuned \
    stable_diffusion_2_spec_finetuned \
  2>&1 | tee "runs/${RUN_ID}/logs/12c_reconstruct_sd2_finetuned.log"
```

Calculate reconstruction metrics:

```bash
make calculate-reconstruction-error \
  2>&1 | tee "runs/${RUN_ID}/logs/13_reconstruction_metrics.log"
```

Check that the newest reconstruction result includes `structure`, all three
mechanisms, all three structures, all requested rates/iterations, and all eight
SD2 variants.

## 17. Phase 12 — forecasting



### Final protocol after the blockers are implemented

Run the unified local rolling-origin evaluator. It must include:

- local XGBoost and SARIMAX;
- persistence and seasonal-naive;
- direct-missing XGBoost;
- five origins for boiler/pump and three for vibration;
- one forecast to `H_max` per source/origin;
- cumulative and lead-time-bin MAE/RMSE/MASE plus secondary MAPE/sMAPE;
- explicit fallback records.

The expected entry point is:

```bash
make evaluate-rolling-origins \
  2>&1 | tee "runs/${RUN_ID}/logs/14_rolling_origins.log"
```

Do not use `make pipeline-full` for the final rebuttal while it still invokes
the global-XGBoost steps 7--9 and omits rolling origins.

### Current code: diagnostic only

The current command below gives a partial local XGBoost/SARIMAX diagnostic for
boiler and pump:

```bash
uv run python src/8_evaluate_rolling_origins.py \
  --models sarimax xgboost \
  --n-origins 5
```

Its outputs are not yet the complete rebuttal forecasting result.

## 18. Phase 13 — statistical inference

The final batch statistics command must consume reconstruction and
rolling-origin result files and export:

- Friedman omnibus statistics;
- paired t-test or Wilcoxon post-hoc comparisons;
- Shapiro p-values for paired differences;
- raw and Holm-adjusted p-values;
- effect size;
- bootstrap 95% confidence interval;
- exact-pair and independent-dataset counts.

The pairing key must include:

```text
dataset × mechanism × rate × structure × degradation seed/iteration
× forecast horizon × origin
```

No batch CLI currently exists. Until it is implemented, Streamlit can only be
used for inspection:

```bash
make visualize-reconstruction-error
make visualize-prediction
```

Do not treat screenshots from Streamlit as the only statistical artifact.

## 19. Phase 14 — final validation and immutable manifest

Run the test suite again:

```bash
uv run pytest -q 2>&1 | tee "runs/${RUN_ID}/logs/15_pytest.log"
uv run python -m compileall -q src tests
git status --porcelain=v1 | tee "runs/${RUN_ID}/environment/git_status_after.txt"
```

Record final configs, commit, environment, and file hashes:

```bash
cp config/config.yaml "runs/${RUN_ID}/config/config.final.yaml"
cp config/prediction_models_config.yaml \
  "runs/${RUN_ID}/config/prediction_models_config.final.yaml"
git rev-parse HEAD | tee "runs/${RUN_ID}/environment/git_commit_final.txt"
uv pip freeze | tee "runs/${RUN_ID}/environment/python_packages_final.txt"

RUN_ID="${RUN_ID}" uv run python - <<'PY'
import hashlib
import json
import os
from pathlib import Path

run_id = os.environ["RUN_ID"]
roots = [
    Path("data/1_5_horizon_recommendation"),
    Path("data/1_6_sd2_optimization"),
    Path("data/3_missing_data/reports"),
    Path("reconstruction_experiments_results"),
    Path("prediction_experiment_results"),
    Path("models/sd2_windowed"),
]
records = []
for root in roots:
    if not root.exists():
        continue
    for path in sorted(p for p in root.rglob("*") if p.is_file()):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        records.append({
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": digest.hexdigest(),
        })
target = Path("runs") / run_id / "manifests" / "generated_files.json"
target.write_text(json.dumps(records, indent=2) + "\n")
print("wrote", len(records), "records to", target)
PY
```

Archive the complete run record:

```bash
tar -czf "runs/rebuttal_run_${RUN_ID}.tar.gz" "runs/${RUN_ID}"
sha256sum "runs/rebuttal_run_${RUN_ID}.tar.gz" \
  > "runs/rebuttal_run_${RUN_ID}.tar.gz.sha256"
```



## 20. Expected final deliverables

Do not declare the rebuttal experiment complete until all are present:

- frozen, committed `config.yaml` and `prediction_models_config.yaml`;
- tracked `uv.lock`, commit hash, clean Git status, and dependency freeze;
- cleaned/split data metadata with correct horizons;
- per-realization and per-gap missingness reports;
- empirical mask-coverage report;
- round-trip and oracle representation ablations;
- synthetic-to-real features, PCA, grouped classifier, and report;
- corrected SD2 training manifest and training report;
- local corrected fine-tuned SD2 pipeline;
- explicit zero-shot vs fine-tuned results for GAF/MTF/RP/SPEC;
- SD2 seed-sensitivity results from 3--5 seeds on the preregistered subset;
- reconstruction metrics for every complete experimental condition;
- rolling-origin cumulative and lead-time metrics with all required baselines;
- explicit SARIMAX fallback report;
- batch Friedman/post-hoc statistical outputs with Holm correction, effect
sizes, and 95% confidence intervals;
- generated-file checksum manifest;
- updated manuscript figures/tables generated only from this run ID.



## 21. Commands that must not be used for final rebuttal results

```bash
# Leakage-prone legacy SD2 optimizer:
# make optimize

# Methodologically asymmetric forecasting path while XGBoost is global:
# make pipeline-full
# make train-prediction-models
# make predict-datasets
# make calculate-prediction-error

# Old fine-tuned Hub weights with corrected encoders:
# unset SD2_FINETUNED_MODEL
```

These commands may remain useful for legacy reproduction or diagnostics, but
their outputs must not be mixed with the final rebuttal run.