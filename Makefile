.PHONY: help setup clean-datasets create-split degrade-datasets optimize optimize-quick reconstruct-datasets calculate-mad visualize-mad train-prediction-models predict-datasets calculate-prediction-error visualize-prediction pipeline pipeline-full clean clean-all test test-prediction

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
	@echo "  make clean-datasets           - Step 1:  Clean and validate raw datasets"
	@echo "  make create-split             - Step 2:  Split data into train/test sets"
	@echo "  make degrade-datasets         - Step 3:  Create degraded training data"
	@echo "  make optimize                 - Optional: Optimize SD hyperparameters"
	@echo "  make reconstruct-datasets     - Step 4:  Reconstruct missing values"
	@echo "  make calculate-mad            - Step 5:  Calculate MAD metrics"
	@echo "  make visualize-mad            - Step 6:  Launch reconstruction dashboard"
	@echo "  make train-prediction-models  - Step 7:  Train prediction models"
	@echo "  make predict-datasets         - Step 8:  Predict using trained models"
	@echo "  make calculate-prediction-error - Step 9:  Calculate prediction MAPE"
	@echo "  make visualize-prediction     - Step 10: Launch prediction dashboard"
	@echo ""
	@echo "Full pipelines:"
	@echo "  make pipeline            - Reconstruction pipeline (steps 1-5)"
	@echo "  make pipeline-full       - Full pipeline (steps 1-5, 7-9)"
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

# Step 2: Split datasets into train/test (2_create_split.py)
create-split:
	@echo "==================================================================="
	@echo "Step 2: Splitting datasets into train/test sets"
	@echo "==================================================================="
	uv run python src/2_create_split.py
	@echo "✓ Datasets split into train/test"

# Step 3: Degrade training datasets (3_degrade_datasets.py)
degrade-datasets:
	@echo "==================================================================="
	@echo "Step 3: Creating degraded training datasets with missing data"
	@echo "==================================================================="
	uv run python src/3_degrade_datasets.py
	@echo "✓ Degraded datasets created"

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

# Step 4: Reconstruct datasets (4_reconstruct_datasets.py)
reconstruct-datasets:
	@echo "==================================================================="
	@echo "Step 4: Reconstructing missing values"
	@echo "==================================================================="
	@echo "⚠️  This may take 1-4 hours depending on hardware..."
	uv run python src/4_reconstruct_datasets.py
	@echo "✓ Reconstruction complete"

# Step 5: Calculate MAD (5_calculate_mad.py)
calculate-mad:
	@echo "==================================================================="
	@echo "Step 5: Calculating MAD metrics"
	@echo "==================================================================="
	uv run python src/5_calculate_mad.py
	@echo "✓ MAD calculation complete"

# Step 6: Visualize reconstruction results (6_visualize_mad_comparison.py)
visualize-mad:
	@echo "==================================================================="
	@echo "Step 6: Launching Reconstruction Streamlit dashboard"
	@echo "==================================================================="
	@echo "Dashboard will open at http://localhost:8501"
	uv run streamlit run src/6_visualize_mad_comparison.py

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
pipeline: clean-datasets create-split degrade-datasets reconstruct-datasets calculate-mad
	@echo "==================================================================="
	@echo "✓ RECONSTRUCTION PIPELINE COMPLETE"
	@echo "==================================================================="
	@echo "Results saved to: reconstruction_experiments_results/"
	@echo "Run 'make visualize-mad' to view results in dashboard"
	@echo "Run 'make train-prediction-models' to train prediction models"

# Run full pipeline including prediction (steps 1-5, 7-9)
pipeline-full: clean-datasets create-split degrade-datasets reconstruct-datasets calculate-mad train-prediction-models predict-datasets calculate-prediction-error
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
	rm -rf data/2_splitted_data/train/*
	rm -rf data/2_splitted_data/test/*
	rm -rf data/3_missing_data/*
	rm -rf data/4_fixed_data/*
	@echo "✓ Generated datasets removed (results preserved)"

# Clean everything including results
clean-all:
	@echo "Cleaning all generated files..."
	rm -rf data/1_cleaned_data/*
	rm -rf data/2_splitted_data/train/*
	rm -rf data/2_splitted_data/test/*
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

# Quick test (for development)
test:
	@echo "Running quick test with limited data..."
	uv run python src/3_degrade_datasets.py --techniques MCAR --rates 0.10 --iterations 1
	uv run python src/4_reconstruct_datasets.py --models interpolate_linear interpolate_cubic
	uv run python src/5_calculate_mad.py
	@echo "✓ Quick test complete"

# Quick prediction test
test-prediction:
	@echo "Running quick prediction test..."
	uv run python src/7_train_prediction_models.py --models xgboost --iterations 1
	uv run python src/8_predict_datasets.py --models xgboost holt_winters
	uv run python src/9_calculate_prediction_error.py
	@echo "✓ Quick prediction test complete"
