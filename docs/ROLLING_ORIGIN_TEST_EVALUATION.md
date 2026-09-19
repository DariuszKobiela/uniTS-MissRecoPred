# Rolling-origin evaluation on the fixed test split

## What the implementation does

The original train/test boundary is preserved. For each later forecast origin,
the model is refitted on:

    reconstructed or original train + real test values before the origin

The forecast target is always the next part of the same fixed test split. No
test value at or after the current origin enters model fitting.

This is an online expanding-history evaluation. It deliberately reuses the
test set for sequential model updates, so its results must be reported
separately from a one-shot untouched-holdout result.

The implementation supports:

- local XGBoost refitted from scratch for every source series and origin;
- SARIMAX/ARIMA refitted for every source series and origin;
- original training series and every reconstructed training series;
- all configured forecast horizons;
- origin-level predictions, metrics and timing.

## Origin placement

For test length N and forecast horizon H, valid origin offsets range from zero
to N-H. Up to five unique offsets are distributed evenly over that range.

With the current horizons this gives:

| dataset | test length | horizon | origins |
| --- | ---: | ---: | ---: |
| boiler | 720 | 12 | 5 |
| boiler | 720 | 96 | 5 |
| boiler | 720 | 180 | 5 |
| boiler | 720 | 720 | 1 |
| pump | 1440 | 12 | 5 |
| pump | 1440 | 96 | 5 |
| pump | 1440 | 360 | 5 |
| pump | 1440 | 1440 | 1 |

A horizon equal to the complete test set has only one possible origin. Five
origins for that horizon would require a longer test set or forecasts extending
past the available ground truth.

Forecast windows may overlap. Later origins are nevertheless leakage-safe with
respect to their own future because they use only the test prefix before the
origin.

## Running

After reconstruction, run:

    make evaluate-rolling-origins

No separate step-7 XGBoost training is required for this evaluation. XGBoost is
local and is refitted at every origin. SARIMAX is also fitted per origin.

To override the number of origins or models:

    uv run python src/8_evaluate_rolling_origins.py \
      --n-origins 5 --models sarimax xgboost

Configuration is under prediction.rolling_origins in config/config.yaml.
Boiler and pump are selected by default.

## Outputs

Origin-level metrics are written to:

    prediction_experiment_results/prediction_results_rolling_origins_TIMESTAMP.csv

Across-origin mean, median and standard deviation are written to:

    prediction_experiment_results/rolling_origins/origin_summary_TIMESTAMP.csv

Individual actual/predicted vectors are written to:

    prediction_experiment_results/rolling_origins/predictions/

A run manifest, including failed tasks, is written to:

    prediction_experiment_results/rolling_origins/run_summary_TIMESTAMP.json

Important result columns include:

- forecast_horizon;
- origin and origin_offset;
- revealed_test_samples;
- base_train_samples and effective_train_samples;
- origin_count_for_horizon;
- evaluation_scheme;
- all configured prediction metrics.

Aggregate results by the complete paired key, including dataset, missingness
mechanism, missingness rate, reconstruction iteration, reconstruction model,
prediction model, forecast horizon and origin.
