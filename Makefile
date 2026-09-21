.PHONY: help setup test-unit clean-datasets recommend-horizons create-split analyze-sd2-design analyze-sd2-ablation optimize-sd2-design generate-sd2-training-data analyze-synthetic-real-gap train-sd2-windowed degrade-datasets analyze-missingness reconstruct-datasets calculate-reconstruction-error evaluate-rolling-origins batch-statistics summarize-run visualize-reconstruction-error visualize-prediction-error pipeline-rebuttal

help:
	@echo "uniTS-MissRecoPred — aktywna ścieżka eksperymentu"
	@echo "1 clean-datasets | 2 recommend-horizons | 3 create-split"
	@echo "4 analyze-sd2-design | 5 generate-sd2-training-data | 6 analyze-synthetic-real-gap"
	@echo "7 train-sd2-windowed | 8 degrade-datasets | 9 analyze-missingness"
	@echo "10 reconstruct-datasets | 11 calculate-reconstruction-error"
	@echo "12 evaluate-rolling-origins | 13 batch-statistics | 14 summarize-run"
	@echo "Dashboardy: visualize-reconstruction-error | visualize-prediction-error"
	@echo "Pełny przebieg: make pipeline-rebuttal"

setup:
	uv sync

test-unit:
	uv run pytest -q

clean-datasets:
	uv run python src/1_clean_datasets.py

recommend-horizons:
	uv run python src/2_analyze_forecast_horizons.py

create-split:
	uv run python src/3_create_split.py

analyze-sd2-design:
	uv run python src/4_analyze_sd2_design.py

analyze-sd2-ablation:
	uv run python src/4_analyze_sd2_design.py --run-ablation

optimize-sd2-design:
	uv run python src/4_analyze_sd2_design.py --run-inference

generate-sd2-training-data:
	uv run python src/5_generate_sd2_training_data.py --samples 2000 --workers 20 --overwrite

analyze-synthetic-real-gap:
	uv run python src/6_analyze_synthetic_real_gap.py

train-sd2-windowed:
	uv run python src/7_finetune_sd2.py

degrade-datasets:
	uv run python src/8_degrade_datasets.py

analyze-missingness:
	uv run python src/9_analyze_missingness.py

reconstruct-datasets:
	uv run python src/10_reconstruct_datasets.py

calculate-reconstruction-error:
	uv run python src/11_calculate_reconstruction_error.py

evaluate-rolling-origins:
	SD2_FAIL_CLOSED=1 uv run python src/12_evaluate_rolling_origins.py

batch-statistics:
	@rolling_file="$$(ls -t prediction_experiment_results/prediction_results_rolling_origins_*.csv 2>/dev/null | head -1)"; \
	reconstruction_file="$$(ls -t reconstruction_experiments_results/reconstruction_results_*.csv 2>/dev/null | head -1)"; \
	test -n "$$rolling_file" || (echo "Brak wyników rolling-origin" && exit 1); \
	test -n "$$reconstruction_file" || (echo "Brak wyników rekonstrukcji" && exit 1); \
	uv run python src/13_batch_statistics.py --rolling-results "$$rolling_file" --reconstruction-results "$$reconstruction_file"

summarize-run:
	uv run python src/14_summarize_run.py

visualize-prediction-error:
	uv run streamlit run src/B_visualize_prediction_error.py

visualize-reconstruction-error:
	uv run streamlit run src/A_visualize_reconstruction_error.py

pipeline-rebuttal: clean-datasets recommend-horizons create-split analyze-sd2-design analyze-sd2-ablation generate-sd2-training-data analyze-synthetic-real-gap train-sd2-windowed degrade-datasets analyze-missingness reconstruct-datasets calculate-reconstruction-error evaluate-rolling-origins batch-statistics summarize-run
	@echo "Pełny eksperyment zakończony. Podsumowanie: experiment_run_summary.json"
