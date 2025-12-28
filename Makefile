.PHONY: help setup install clean-datasets degrade optimize reconstruct calculate visualize pipeline clean clean-all

# Default target
help:
	@echo "==================================================================="
	@echo "Time Series Reconstruction Framework - Makefile"
	@echo "==================================================================="
	@echo ""
	@echo "Setup commands:"
	@echo "  make setup       - Create virtual environment"
	@echo "  make install     - Install dependencies"
	@echo ""
	@echo "Pipeline commands (run in order):"
	@echo "  make clean-datasets  - Clean and validate raw datasets"
	@echo "  make degrade         - Create degraded datasets with missing data"
	@echo "  make optimize        - Optimize Stable Diffusion hyperparameters (optional)"
	@echo "  make reconstruct     - Reconstruct missing values with all models"
	@echo "  make calculate       - Calculate MAD metrics"
	@echo "  make visualize       - Launch Streamlit dashboard"
	@echo ""
	@echo "Full pipeline:"
	@echo "  make pipeline    - Run complete pipeline (clean → degrade → reconstruct → calculate)"
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

# Step 2: Degrade datasets
degrade:
	@echo "==================================================================="
	@echo "Step 2: Creating degraded datasets with missing data"
	@echo "==================================================================="
	experiment/bin/python src/2_degrade_datasets.py
	@echo "✓ Degraded datasets created"

# Optional: Optimize Stable Diffusion hyperparameters
optimize:
	@echo "==================================================================="
	@echo "Optimizing Stable Diffusion hyperparameters"
	@echo "==================================================================="
	@echo "⚠️  This may take several hours..."
	experiment/bin/python src/optimization/optimize_sd_hyperparams.py
	@echo "✓ Optimization complete"

# Step 3: Reconstruct datasets
reconstruct:
	@echo "==================================================================="
	@echo "Step 3: Reconstructing missing values"
	@echo "==================================================================="
	@echo "⚠️  This may take 1-4 hours depending on hardware..."
	experiment/bin/python src/3_reconstruct_datasets.py
	@echo "✓ Reconstruction complete"

# Step 4: Calculate MAD
calculate:
	@echo "==================================================================="
	@echo "Step 4: Calculating MAD metrics"
	@echo "==================================================================="
	experiment/bin/python src/4_calculate_mad.py
	@echo "✓ MAD calculation complete"

# Step 5: Visualize results
visualize:
	@echo "==================================================================="
	@echo "Step 5: Launching Streamlit dashboard"
	@echo "==================================================================="
	@echo "Dashboard will open at http://localhost:8501"
	experiment/bin/streamlit run src/5_visualize_mad_comparison.py

# Run complete pipeline
pipeline: clean-datasets degrade reconstruct calculate
	@echo "==================================================================="
	@echo "✓ PIPELINE COMPLETE"
	@echo "==================================================================="
	@echo "Results saved to: experiments_results/"
	@echo "Run 'make visualize' to view results in dashboard"

# Clean generated data (keep results)
clean:
	@echo "Cleaning generated datasets..."
	rm -rf data/1_cleaned_data/*
	rm -rf data/2_missing_data/*
	rm -rf data/3_fixed_data/*
	@echo "✓ Generated datasets removed (results preserved)"

# Clean everything including results
clean-all:
	@echo "Cleaning all generated files..."
	rm -rf data/1_cleaned_data/*
	rm -rf data/2_missing_data/*
	rm -rf data/3_fixed_data/*
	rm -rf experiments_results/*.csv
	rm -rf experiments_results/performance_metrics/*.csv
	@echo "✓ All generated files removed"

# Quick test (for development)
test:
	@echo "Running quick test with limited data..."
	experiment/bin/python src/2_degrade_datasets.py --techniques MCAR --rates 0.10 --iterations 1
	experiment/bin/python src/3_reconstruct_datasets.py --models interpolate_linear interpolate_cubic
	experiment/bin/python src/4_calculate_mad.py
	@echo "✓ Quick test complete"

