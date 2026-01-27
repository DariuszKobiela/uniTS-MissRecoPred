.PHONY: help setup install clean-datasets split degrade optimize reconstruct calculate predict visualize pipeline pipeline-full clean clean-all test

# Default target
help:
	@echo "==================================================================="
	@echo "Time Series Reconstruction & Prediction Framework - Makefile"
	@echo "uniTS-MissRecoPred"
	@echo "==================================================================="
	@echo ""
	@echo "Setup commands:"
	@echo "  make setup       - Create virtual environment"
	@echo "  make install     - Install dependencies"
	@echo ""
	@echo "Pipeline commands (run in order):"
	@echo "  make clean-datasets  - Step 1: Clean and validate raw datasets"
	@echo "  make split           - Step 2: Split data into train/test sets"
	@echo "  make degrade         - Step 3: Create degraded training data"
	@echo "  make optimize        - Optional: Optimize Stable Diffusion hyperparameters"
	@echo "  make optimize-quick  - Optional: Quick optimization test"
	@echo "  make reconstruct     - Step 4: Reconstruct missing values"
	@echo "  make calculate       - Step 5: Calculate MAD metrics"
	@echo "  make predict         - Step 7: Predict future values"
	@echo "  make visualize       - Step 6: Launch Streamlit dashboard"
	@echo ""
	@echo "Full pipelines:"
	@echo "  make pipeline        - Reconstruction pipeline (steps 1-5)"
	@echo "  make pipeline-full   - Full pipeline including prediction (steps 1-5, 7)"
	@echo ""
	@echo "Cleanup commands:"
	@echo "  make clean       - Remove generated datasets (keep results)"
	@echo "  make clean-all   - Remove all generated files including results"
	@echo ""
	@echo "==================================================================="

# Setup virtual environment
setup:
	@echo "Creating virtual environment..."
	python3 -m venv experiment
	@echo "✓ Virtual environment created in ./experiment"
	@echo "Activate with: source experiment/bin/activate"

# Install dependencies
install:
	@echo "Installing dependencies..."
	experiment/bin/pip install -r requirements.txt
	@echo "✓ Dependencies installed"

# Step 1: Clean datasets
clean-datasets:
	@echo "==================================================================="
	@echo "Step 1: Cleaning and validating raw datasets"
	@echo "==================================================================="
	experiment/bin/python src/1_clean_datasets.py
	@echo "✓ Datasets cleaned"

# Step 2: Split datasets into train/test
split:
	@echo "==================================================================="
	@echo "Step 2: Splitting datasets into train/test sets"
	@echo "==================================================================="
	experiment/bin/python src/2_create_split.py
	@echo "✓ Datasets split into train/test"

# Step 3: Degrade training datasets
degrade:
	@echo "==================================================================="
	@echo "Step 3: Creating degraded training datasets with missing data"
	@echo "==================================================================="
	experiment/bin/python src/3_degrade_datasets.py
	@echo "✓ Degraded datasets created"

# Optional: Optimize Stable Diffusion hyperparameters (Full)
optimize:
	@echo "==================================================================="
	@echo "Optimizing Stable Diffusion hyperparameters (Bayesian/Optuna)"
	@echo "==================================================================="
	@echo "⚠️  This may take several hours..."
	@echo "ℹ️  Using default: 30 random files"
	experiment/bin/python src/optimization/optimize_sd_hyperparams.py
	@echo "✓ Optimization complete"

# Optional: Optimize Stable Diffusion hyperparameters (Quick Test)
optimize-quick:
	@echo "==================================================================="
	@echo "Optimizing Stable Diffusion hyperparameters (Quick Test)"
	@echo "==================================================================="
	experiment/bin/python src/optimization/optimize_sd_hyperparams.py --n-trials 20 --max-files 5
	@echo "✓ Quick optimization complete"

# Step 4: Reconstruct datasets
reconstruct:
	@echo "==================================================================="
	@echo "Step 4: Reconstructing missing values"
	@echo "==================================================================="
	@echo "⚠️  This may take 1-4 hours depending on hardware..."
	experiment/bin/python src/4_reconstruct_datasets.py
	@echo "✓ Reconstruction complete"

# Step 5: Calculate MAD
calculate:
	@echo "==================================================================="
	@echo "Step 5: Calculating MAD metrics"
	@echo "==================================================================="
	experiment/bin/python src/5_calculate_mad.py
	@echo "✓ MAD calculation complete"

# Step 6: Visualize results
visualize:
	@echo "==================================================================="
	@echo "Step 6: Launching Streamlit dashboard"
	@echo "==================================================================="
	@echo "Dashboard will open at http://localhost:8501"
	experiment/bin/streamlit run src/6_visualize_mad_comparison.py

# Step 7: Predict future values
predict:
	@echo "==================================================================="
	@echo "Step 7: Predicting future values"
	@echo "==================================================================="
	@echo "⚠️  This may take 1-4 hours depending on models..."
	experiment/bin/python src/7_predict_datasets.py
	@echo "✓ Prediction complete"

# Run reconstruction pipeline (steps 1-5)
pipeline: clean-datasets split degrade reconstruct calculate
	@echo "==================================================================="
	@echo "✓ RECONSTRUCTION PIPELINE COMPLETE"
	@echo "==================================================================="
	@echo "Results saved to: reconstruction_experiments_results/"
	@echo "Run 'make visualize' to view results in dashboard"
	@echo "Run 'make predict' to run prediction on reconstructed data"

# Run full pipeline including prediction (steps 1-5, 7)
pipeline-full: clean-datasets split degrade reconstruct calculate predict
	@echo "==================================================================="
	@echo "✓ FULL PIPELINE COMPLETE"
	@echo "==================================================================="
	@echo "Reconstruction results: reconstruction_experiments_results/"
	@echo "Prediction results: prediction_experiment_results/"
	@echo "Run 'make visualize' to view results in dashboard"

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
	rm -rf hyperparameter_optimization/*
	@echo "✓ All generated files removed"

# Quick test (for development)
test:
	@echo "Running quick test with limited data..."
	experiment/bin/python src/3_degrade_datasets.py --techniques MCAR --rates 0.10 --iterations 1
	experiment/bin/python src/4_reconstruct_datasets.py --models interpolate_linear interpolate_cubic
	experiment/bin/python src/5_calculate_mad.py
	@echo "✓ Quick test complete"

# Quick prediction test
test-predict:
	@echo "Running quick prediction test..."
	experiment/bin/python src/7_predict_datasets.py --models holt_winters xgboost
	@echo "✓ Quick prediction test complete"
