.PHONY: help setup clean-datasets recommend-horizons create-split degrade-datasets analyze-missingness ingest-external optimize optimize-quick analyze-sd2-design analyze-sd2-ablation analyze-synthetic-real-gap optimize-sd2-design generate-sd2-training-data train-sd2-windowed reconstruct-datasets calculate-reconstruction-error calculate-mad visualize-reconstruction-error visualize-mad train-prediction-models predict-datasets evaluate-rolling-origins batch-statistics rebuttal-preflight rebuttal-forecast rebuttal-validate run-manifest train-prediction-models predict-datasets calculate-prediction-error visualize-prediction pipeline pipeline-full pipeline-external pipeline-rebuttal clean clean-all test test-prediction test-unit

# Default target
help:
	@echo "==================================================================="
	@echo "Time Series Reconstruction & Prediction Framework - Makefile"
	@echo "uniTS-MissRecoPred"
	@echo "==================================================================="
	@echo ""
	@echo "Setup commands:"
	@echo "  make setup                    - Install dependencies with uv"
	@echo ""
	@echo "Pipeline commands (run in order):"
	@echo "  make clean-datasets              - Step 1:  Clean and validate raw datasets"
	@echo "  make recommend-horizons          - Step 1.5: Recommend forecast horizons (metadata)"
	@echo "  make create-split                - Step 2:  Split data into train/test sets"
	@echo "  make degrade-datasets            - Step 3:  Introduce mechanism × structure missingness"
	@echo "  make analyze-missingness         - Report per-realization and per-gap diagnostics"
	@echo "  make optimize                    - Optional: SD hyperparameters (Optuna)"
	@echo "  make reconstruct-datasets        - Step 4:  Reconstruct missing training values"
	@echo "  make calculate-reconstruction-error - Step 5:  Reconstruction error metrics (CSV)"
	@echo "  make visualize-reconstruction-error - Step 6:  Reconstruction results (Streamlit)"
	@echo "  make train-prediction-models     - Step 7:  Train prediction models"
	@echo "  make predict-datasets            - Step 8:  Run predictions"
	@echo "  make evaluate-rolling-origins    - Authoritative rolling-origin forecast evaluation"
	@echo "  make batch-statistics            - Friedman + Holm post-hoc export (rebuttal)"
	@echo "  make rebuttal-preflight          - Preflight checks before full rebuttal rerun"
	@echo "  make rebuttal-forecast           - Rolling-origin evaluation + batch statistics"
	@echo "  make pipeline-rebuttal           - Full rebuttal pipeline (1-6 + forecast + stats)"
	@echo "  make calculate-prediction-error  - Step 9 (legacy): prediction error metrics"
	@echo "  make visualize-prediction        - Step 10: Prediction results (Streamlit)"
	@echo "  make analyze-sd2-design         - Analyze 512/1024/2048 windows and image sizes"
	@echo "  make analyze-sd2-ablation       - Run round-trip and clean-image oracle controls"
	@echo "  make optimize-sd2-design        - GPU tune window/image/prompt/steps/guidance"
	@echo "  make generate-sd2-training-data - Generate corrected inpainting triplets"
	@echo "  make analyze-synthetic-real-gap - Quantify synthetic vs real signal gap"
	@echo "  make train-sd2-windowed         - Fine-tune the SD2 inpainting UNet (CUDA)"
	@echo ""
	@echo "Aliases: calculate-mad -> calculate-reconstruction-error, visualize-mad -> visualize-reconstruction-error"
	@echo ""
	@echo "Full pipelines:"
	@echo "  make pipeline            - Reconstruction only (steps 1-5, no dashboards)"
	@echo "  make pipeline-full       - Reconstruction + prediction train/predict/eval (1-5, 7-9)"
	@echo "  make ingest-external     - Manifest → missing_dir + test (external missingness; set pipeline.entry in config)"
	@echo "  make pipeline-external   - ingest-external + 4 + 7 + 8 + 9 (skip 1-3, 5-6; see README)"
	@echo ""
	@echo "Cleanup commands:"
	@echo "  make clean       - Remove generated datasets (keep results)"
	@echo "  make clean-all   - Remove all generated files including results"
	@echo ""
	@echo "==================================================================="

# Setup - install dependencies with uv
setup:
	@echo "Installing dependencies with uv..."
	uv sync
	@echo "✓ Dependencies installed"

# Step 1: Clean datasets
clean-datasets:
	@echo "==================================================================="
	@echo "Step 1: Cleaning and validating raw datasets"
	@echo "==================================================================="
	uv run python src/1_clean_datasets.py
	@echo "✓ Datasets cleaned"

# Step 1.5: Recommend forecast horizons (analyze_forecast_horizons.py)
recommend-horizons:
	@echo "==================================================================="
	@echo "Step 1.5: Analyzing series and recommending forecast horizons"
	@echo "==================================================================="
	uv run python src/analyze_forecast_horizons.py
	@echo "✓ Horizon recommendations written"

# Step 2: Split datasets into train/test (2_create_split.py)
create-split:
	@echo "==================================================================="
	@echo "Step 2: Splitting datasets into train/test sets"
	@echo "==================================================================="
	uv run python src/2_create_split.py
	@echo "✓ Datasets split into train/test"

# Ingest external missingness manifest (ingest_external_missing.py)
ingest-external:
	@echo "==================================================================="
	@echo "Ingest external missingness (manifest → 3_missing_data + 2_splitted_data/test)"
	@echo "==================================================================="
	uv run python src/ingest_external_missing.py
	@echo "✓ Ingest complete (set pipeline.entry: external_missing for predict behavior)"

# Step 3: Degrade training datasets (3_degrade_datasets.py)
degrade-datasets:
	@echo "==================================================================="
	@echo "Step 3: Creating degraded training datasets with missing data"
	@echo "==================================================================="
	uv run python src/3_degrade_datasets.py
	@echo "✓ Degraded datasets created"

# Reviewer-facing gap diagnostics (also written automatically by step 3)
analyze-missingness:
	uv run python src/analyze_missingness.py

# Optional: Optimize Stable Diffusion hyperparameters (Full)
optimize:
	@echo "==================================================================="
	@echo "Optimizing Stable Diffusion hyperparameters (Bayesian/Optuna)"
	@echo "==================================================================="
	@echo "⚠️  This may take several hours..."
	uv run python src/optimization/optimize_sd_hyperparams.py
	@echo "✓ Optimization complete"

# Optional: Quick optimization test
optimize-quick:
	@echo "==================================================================="
	@echo "Optimizing Stable Diffusion hyperparameters (Quick Test)"
	@echo "==================================================================="
	uv run python src/optimization/optimize_sd_hyperparams.py --n-trials 20 --max-files 5
	@echo "✓ Quick optimization complete"

# Analyze window/image-size trade-offs without GPU inference
analyze-sd2-design:
	@echo "Analyzing SD2 windows, image resolutions, masks, and legacy training data"
	uv run python src/optimization/analyze_sd2_design.py

# CPU-only encode/decode ceiling and ideal clean-image oracle
analyze-sd2-ablation:
	@echo "Running SD2 representation round-trip and oracle ablations"
	uv run python src/optimization/analyze_sd2_design.py --run-ablation

# Empirical GPU optimization: window, image, prompt, steps, and guidance
optimize-sd2-design:
	@echo "Running GPU validation of the SD2 design (potentially expensive)"
	uv run python src/optimization/analyze_sd2_design.py --run-inference

# Generate explicit clean/conditioning/mask triplets for windowed SD2
generate-sd2-training-data:
	@echo "Generating the corrected windowed SD2 training dataset"
	uv run python src/training/generate_sd2_windowed_dataset.py --samples 2000

analyze-synthetic-real-gap:
	@echo "Analyzing the synthetic-to-real signal gap"
	uv run python src/analysis/analyze_synthetic_real_gap.py

# Fine-tune the inpainting UNet; requires CUDA
train-sd2-windowed:
	@echo "Fine-tuning SD2 on the corrected windowed dataset"
	uv run python src/training/finetune_sd2_windowed.py


# Step 4: Reconstruct datasets (4_reconstruct_datasets.py)
reconstruct-datasets:
	@echo "==================================================================="
	@echo "Step 4: Reconstructing missing values"
	@echo "==================================================================="
	@echo "⚠️  This may take 1-4 hours depending on hardware..."
	uv run python src/4_reconstruct_datasets.py
	@echo "✓ Reconstruction complete"

# Step 5: Reconstruction error metrics (5_calculate_reconstruction_error.py)
calculate-reconstruction-error:
	@echo "==================================================================="
	@echo "Step 5: Calculating reconstruction error metrics (MAD, MAE, RMSE, R², SMAPE, …)"
	@echo "==================================================================="
	uv run python src/5_calculate_reconstruction_error.py
	@echo "✓ Reconstruction error metrics saved"

calculate-mad: calculate-reconstruction-error

# Step 6: Reconstruction dashboard (6_visualize_reconstruction_error.py)
visualize-reconstruction-error:
	@echo "==================================================================="
	@echo "Step 6: Launching reconstruction error dashboard (Streamlit)"
	@echo "==================================================================="
	@echo "Dashboard will open at http://localhost:8501"
	uv run streamlit run src/6_visualize_reconstruction_error.py

visualize-mad: visualize-reconstruction-error

# Step 7: Train prediction models (7_train_prediction_models.py)
train-prediction-models:
	@echo "==================================================================="
	@echo "Step 7: Training prediction models"
	@echo "==================================================================="
	@echo "⚠️  This may take 1-4 hours depending on models..."
	uv run python src/7_train_prediction_models.py
	@echo "✓ Model training complete"

# Step 8: Predict datasets (8_predict_datasets.py)
predict-datasets:
	@echo "==================================================================="
	@echo "Step 8: Predicting with trained models"
	@echo "==================================================================="
	uv run python src/8_predict_datasets.py
	@echo "✓ Prediction complete"


# Authoritative rolling-origin evaluation on the fixed test split
evaluate-rolling-origins:
	@echo "==================================================================="
	@echo "Rolling-origin evaluation (rebuttal authoritative path)"
	@echo "==================================================================="
	SD2_FAIL_CLOSED=1 uv run python src/8_evaluate_rolling_origins.py
	@echo "✓ Rolling-origin evaluation complete"

batch-statistics:
	@echo "==================================================================="
	@echo "Batch Friedman / Holm statistical export"
	@echo "==================================================================="
	uv run python src/11_batch_statistics.py \
	  --rolling-results "$$(ls -t prediction_experiment_results/prediction_results_rolling_origins_*.csv 2>/dev/null | head -1)"
	@echo "✓ Batch statistics exported"

rebuttal-preflight:
	@echo "==================================================================="
	@echo "Rebuttal preflight checks"
	@echo "==================================================================="
	uv run pytest -q
	uv run python -m compileall -q src tests
	test -f config/config.yaml
	test -f uv.lock || (echo "Missing uv.lock — run uv lock" && exit 1)
	test -f data/2_splitted_data/split_manifest.json || (echo "Missing split manifest — run make create-split" && exit 1)
	test -s models/sd2_windowed/best_model/model_index.json || (echo "Missing local fine-tuned SD2 model" && exit 1)
	@echo "✓ Preflight checks passed"

rebuttal-forecast: evaluate-rolling-origins batch-statistics
	@echo "✓ Rebuttal forecast path complete"

run-manifest:
	@RUN_ID="$${RUN_ID:-rebuttal_$$(date -u +%Y%m%dT%H%M%SZ)}"; \
	mkdir -p "runs/$$RUN_ID/manifests"; \
	RUN_ID="$$RUN_ID" uv run python src/generate_run_manifest.py --run-id "$$RUN_ID"

rebuttal-validate: rebuttal-preflight
	@echo "✓ Final rebuttal validation passed"

pipeline-rebuttal: clean-datasets recommend-horizons create-split degrade-datasets analyze-missingness analyze-sd2-design analyze-sd2-ablation generate-sd2-training-data analyze-synthetic-real-gap train-sd2-windowed reconstruct-datasets calculate-reconstruction-error rebuttal-forecast run-manifest
	@echo "==================================================================="
	@echo "✓ REBUTTAL PIPELINE COMPLETE"
	@echo "==================================================================="
# Step 9: Calculate prediction error (9_calculate_prediction_error.py)
calculate-prediction-error:
	@echo "==================================================================="
	@echo "Step 9: Calculating prediction error (MAPE)"
	@echo "==================================================================="
	uv run python src/9_calculate_prediction_error.py
	@echo "✓ Prediction error calculation complete"

# Step 10: Visualize prediction results
visualize-prediction:
	@echo "==================================================================="
	@echo "Step 10: Launching Prediction Streamlit dashboard"
	@echo "==================================================================="
	@echo "Dashboard will open at http://localhost:8501"
	uv run streamlit run src/10_visualize_prediction.py

# Run reconstruction pipeline (steps 1-5)
pipeline: clean-datasets recommend-horizons create-split degrade-datasets reconstruct-datasets calculate-reconstruction-error
	@echo "==================================================================="
	@echo "✓ RECONSTRUCTION PIPELINE COMPLETE"
	@echo "==================================================================="
	@echo "Results saved to: reconstruction_experiments_results/"
	@echo "Run 'make visualize-reconstruction-error' to open the dashboard"
	@echo "Run 'make train-prediction-models' to train prediction models"

# External missingness: ingest + reconstruct + prediction (no 1-3, 5-6)
pipeline-external: ingest-external reconstruct-datasets train-prediction-models predict-datasets calculate-prediction-error
	@echo "==================================================================="
	@echo "✓ EXTERNAL-MISSING PIPELINE COMPLETE (4, 7-9)"
	@echo "==================================================================="

# Run full pipeline including prediction (steps 1-5, 7-9)
pipeline-full: clean-datasets recommend-horizons create-split degrade-datasets reconstruct-datasets calculate-reconstruction-error train-prediction-models predict-datasets calculate-prediction-error
	@echo "==================================================================="
	@echo "✓ FULL PIPELINE COMPLETE"
	@echo "==================================================================="
	@echo "Reconstruction results: reconstruction_experiments_results/"
	@echo "Prediction results: prediction_experiment_results/"
	@echo "Trained models: trained_prediction_models/"
	@echo "Run 'make visualize-prediction' to view prediction results"

# Clean generated data (keep results)
clean:
	@echo "Cleaning generated datasets..."
	rm -rf data/1_cleaned_data/*
	rm -rf data/1_5_horizon_recommendation/*
	rm -rf data/2_splitted_data/train/*
	rm -rf data/2_splitted_data/sd2_validation/*
	rm -rf data/2_splitted_data/test/*
	rm -f data/2_splitted_data/split_manifest.json
	rm -f data/2_splitted_data/external_missing_ingest_state.json
	rm -rf data/3_missing_data/*
	rm -rf data/4_fixed_data/*
	@echo "✓ Generated datasets removed (results preserved)"

# Clean everything including results
clean-all:
	@echo "Cleaning all generated files..."
	rm -rf data/1_cleaned_data/*
	rm -rf data/1_5_horizon_recommendation/*
	rm -rf data/2_splitted_data/train/*
	rm -rf data/2_splitted_data/sd2_validation/*
	rm -rf data/2_splitted_data/test/*
	rm -f data/2_splitted_data/split_manifest.json
	rm -rf data/3_missing_data/*
	rm -rf data/4_fixed_data/*
	rm -rf reconstruction_experiments_results/*.csv
	rm -rf reconstruction_experiments_results/performance_metrics/*.csv
	rm -rf prediction_experiment_results/*.csv
	rm -rf prediction_experiment_results/predictions/*
	rm -rf prediction_experiment_results/performance_metrics/*.csv
	rm -rf trained_prediction_models/*
	rm -rf hyperparameter_optimization/*
	@echo "✓ All generated files removed"

test-unit:
	uv run pytest -q

# Quick test (for development)
test:
	@echo "Running quick test with limited data..."
	uv run python src/3_degrade_datasets.py --techniques MCAR --rates 0.10 --iterations 1
	uv run python src/4_reconstruct_datasets.py --models interpolate_linear interpolate_cubic
	uv run python src/5_calculate_reconstruction_error.py
	@echo "✓ Quick test complete"

# Quick prediction test
test-prediction:
	@echo "Running quick prediction test..."
	uv run python src/7_train_prediction_models.py --models xgboost --iterations 1
	uv run python src/8_predict_datasets.py --models xgboost holt_winters
	uv run python src/9_calculate_prediction_error.py
	@echo "✓ Quick prediction test complete"
