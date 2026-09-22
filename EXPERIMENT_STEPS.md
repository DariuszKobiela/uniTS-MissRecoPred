# Kroki eksperymentu

Wszystkie komendy uruchamiaj z katalogu głównego repozytorium:

```bash
cd /home/darek/uniTS-MissRecoPred
```

Pliki źródłowe muszą znajdować się w `data/0_source_data/`. Eksperyment SD2
wymaga GPU z CUDA. Kolejność poniżej odpowiada numerom aktywnych skryptów.

## Sesja tmux

Długie kroki (oznaczone **tmux**) uruchamiaj w nazwanej sesji, żeby proces
przeżył zamknięcie terminala SSH.

Nowa sesja o nazwie `rebuttal`:

```bash
tmux new -s rebuttal
```

Wewnątrz sesji przejdź do repozytorium (`cd /home/darek/uniTS-MissRecoPred`)
i uruchamiaj kolejne komendy. Odłączenie bez zatrzymywania procesów:
`Ctrl+b`, potem `d`.

Powrót do tej samej sesji:

```bash
tmux attach -t rebuttal
```

Lista sesji: `tmux ls`. Zamknięcie sesji i wszystkich procesów w niej:

```bash
tmux kill-session -t rebuttal
```

Z wewnątrz sesji to samo robi `exit` w ostatnim oknie.

## Profil maszyny i równoległość

Eksperyment jest skonfigurowany dla jednej NVIDIA RTX 6000 Ada (48 GiB VRAM),
AMD Ryzen 9 9950X (16 rdzeni fizycznych, 32 logiczne) i 123 GiB RAM. Ciężkie
zadania CPU używają 20 procesów. Równoległość wewnętrzna XGBoost wynosi 1, aby
20 procesów rolling-origin nie tworzyło kolejnych pul wątków. Zadania SD2 są
sekwencyjne na jednej karcie, używają FP16 i TF32; fine-tuning używa batch 4,
akumulacji 2 oraz 20 workerów DataLoadera.

## Przygotowanie

```bash
uv sync --frozen --extra dev
uv run pytest -q
uv run python -m compileall -q src tests
```

`uv sync --frozen --extra dev` instaluje zależności z `uv.lock` oraz pytest i ruff.
Samo `uv sync --frozen` usuwa te narzędzia, bo są w dodatku `dev`. Dwie następne
komendy sprawdzają testy i składnię przed kosztownym przebiegiem.

## 1. Czyszczenie danych źródłowych

```bash
make clean-datasets
```

Skrypt: `src/1_clean_datasets.py`.

Wczytuje CSV z `data/0_source_data/`, rozpoznaje separator, normalizuje indeks,
konwertuje wartości na liczby, usuwa duplikaty i sortuje serię. Jeżeli w danych
źródłowych są braki, uzupełnia je poprzednią obserwacją (`ffill`). Początkowe
braki, dla których poprzednia wartość nie istnieje, są jawnie raportowane i
usuwane.

Wyniki:

- oczyszczone CSV w `data/1_cleaned_data/`;
- `data/1_cleaned_data/reports/cleaning_report.csv` z liczbą braków, liczbą
  wartości uzupełnionych przez ffill i liczbą usuniętych wierszy początkowych.

## 2. Analiza horyzontów

```bash
make recommend-horizons
```

Skrypt: `src/2_analyze_forecast_horizons.py`.

Analizuje długość, regularność i interwał próbkowania każdej serii. Zwraca
`dataset_metadata.json`, `horizon_recommendations.csv` i raport Markdown w
`data/1_5_horizon_recommendation/`. Metadane są wejściem splitu i ewaluacji.
Dla vibration horyzonty 12 i 24 oznaczają kroki zdarzeń, nie godziny.

## 3. Podział danych

```bash
make create-split
```

Skrypt: `src/3_create_split.py`.

Tworzy rozłączne czasowo partycje:

- `data/2_splitted_data/train/` — degradacja i rekonstrukcja;
- `data/2_splitted_data/sd2_validation/` — dobór ustawień SD2;
- `data/2_splitted_data/test/` — końcowa ewaluacja rolling-origin.

Zwraca także `split_manifest.json` z granicami splitów. Test końcowy nie jest
używany do wyboru ustawień.

## 4. Kontrola projektu SD2

```bash
make analyze-sd2-design
make analyze-sd2-ablation
```

Skrypt: `src/4_analyze_sd2_design.py`.

Pierwsza komenda analizuje długości okien, rozdzielczości, kontekst i maski.
Druga wykonuje kontrolę round-trip kodowanie/dekodowanie oraz clean-image
oracle dla GAF, MTF, RP i spektrogramu.

Wyniki znajdują się w `data/1_6_sd2_optimization/`, przede wszystkim:
`sd2_design_analysis.csv`, `sd2_representation_ablations.csv` i
`sd2_design_report.md`.

Jeżeli ustawienia inferencji nie są z góry ustalone, HPO wolno wykonać tylko na
`sd2_validation`. **tmux:** krok długotrwały (GPU).

```bash
make optimize-sd2-design
```

## 5. Generowanie zbioru treningowego SD2

**tmux:** krok długotrwały.

```bash
uv run python src/5_generate_sd2_training_data.py \
  --samples 2000 \
  --output data/sd2_windowed_training \
  --image-size 512 \
  --window-sizes 512 \
  --rates 0.02,0.05,0.20,0.50 \
  --mechanisms MCAR,MAR,MNAR \
  --structures scattered,contiguous,mixed \
  --source synthetic \
  --seed 42 \
  --workers 20 \
  --overwrite
```

Dla 2000 bazowych okien generuje cztery reprezentacje obrazu oraz komplet
clean/conditioning/mask. Zwraca 8000 przykładów, `manifest.jsonl`, serie
źródłowe i `dataset_summary.json` w `data/sd2_windowed_training/`.

## 6. Analiza synthetic-to-real

**tmux:** krok długotrwały.

```bash
make analyze-synthetic-real-gap
```

Skrypt: `src/6_analyze_synthetic_real_gap.py`.

Porównuje cechy syntetycznych okien treningowych z rzeczywistymi seriami.
Zwraca cechy, PCA, wynik klasyfikatora i raport w
`data/1_6_sd2_optimization/synthetic_real_gap/`. Służy do oceny, czy dane
syntetyczne są wystarczająco zbliżone do domeny docelowej.

## 7. Fine-tuning SD2

**tmux:** krok długotrwały (GPU, wiele godzin).

```bash
uv run python src/7_finetune_sd2.py \
  --data-dir data/sd2_windowed_training \
  --output-dir models/sd2_windowed \
  --encodings gaf,mtf,rp,spec \
  --validation-share 0.15 \
  --epochs 30 \
  --early-stop-patience 5 \
  --learning-rate 1e-5 \
  --batch-size 4 \
  --gradient-accumulation-steps 2 \
  --mixed-precision fp16 \
  --num-workers 20 \
  --seed 42
```

Dostraja UNet SD2 z grupowym podziałem po `series_id`. Zwraca najlepszy lokalny
model w `models/sd2_windowed/best_model/` i `training_report.json`.

## 8. Generowanie kontrolowanych braków

**tmux:** krok długotrwały.

```bash
make degrade-datasets
```

Skrypt: `src/8_degrade_datasets.py`.

Na partycji train generuje kombinacje dataset × MCAR/MAR/MNAR ×
scattered/contiguous/mixed × stopa braków × iteracja. Zwraca zdegradowane CSV z
NaN w `data/3_missing_data/`. Są to braki eksperymentalne, odrębne od braków
źródłowych uzupełnionych w kroku 1.

## 9. Raport braków i masek

```bash
make analyze-missingness
```

Skrypt: `src/9_analyze_missingness.py`.

Oblicza liczbę i długości luk, singletony, pokrycie czasowe oraz empiryczne
pokrycie masek reprezentacji. Zwraca w `data/3_missing_data/reports/` pliki
`missingness_realizations.csv`, `missingness_gaps.csv` i
`empirical_mask_coverage.csv`.

## 10. Rekonstrukcja

**tmux:** krok długotrwały, zwłaszcza warianty SD2 (GPU).

Najpierw metody klasyczne:

```bash
uv run python src/10_reconstruct_datasets.py \
  --models impute_mean impute_median impute_ffill impute_bfill \
  interpolate_linear interpolate_cubic interpolate_pchip knn sarimax
```

Następnie SD2 zero-shot:

```bash
uv run python src/10_reconstruct_datasets.py \
  --models stable_diffusion_2_gaf stable_diffusion_2_mtf \
  stable_diffusion_2_rp stable_diffusion_2_spec
```

Na końcu SD2 fine-tuned:

```bash
SD2_FINETUNED_MODEL="$PWD/models/sd2_windowed/best_model" \
uv run python src/10_reconstruct_datasets.py \
  --models stable_diffusion_2_gaf_finetuned stable_diffusion_2_mtf_finetuned \
  stable_diffusion_2_rp_finetuned stable_diffusion_2_spec_finetuned
```

Skrypt wypełnia wyłącznie NaN, zachowuje wartości obserwowane i zapisuje
rekonstrukcje w `data/4_fixed_data/` w układzie katalogów:

`{dataset}/{MCAR-scattered|…}/{2p|5p|…}/{model}/{iteracja}.csv`

(w folderze modelu leżą wszystkie iteracje danego modelu). Metryki czasu i
zasobów trafiają do `reconstruction_experiments_results/performance_metrics/`.

Jeśli masz jeszcze płaskie pliki `dataset_…_model.csv` w korzeniu
`4_fixed_data/` (stary zapis), po zakończeniu bieżącej rekonstrukcji:

```bash
uv run python scripts/migrate_fixed_data_layout.py --dry-run
uv run python scripts/migrate_fixed_data_layout.py
```

## 11. Metryki rekonstrukcji

```bash
make calculate-reconstruction-error
```

Skrypt: `src/11_calculate_reconstruction_error.py`.

Porównuje rekonstrukcje z pełnym train wyłącznie na pozycjach sztucznie
usuniętych. Zwraca `reconstruction_results_*.csv` z metrykami jakości,
metadanymi warunku eksperymentalnego i pomiarami wykonania.

## 12. Prognozy rolling-origin

**tmux:** krok długotrwały.

```bash
make evaluate-rolling-origins
```

Skrypt: `src/12_evaluate_rolling_origins.py`.

Dopasowuje lokalnie XGBoost i SARIMAX oraz liczy persistence, seasonal-naive i
XGBoost bezpośrednio na brakach. Każdy origin daje jedną prognozę do `H_max`, z
której powstają miary skumulowane i biny lead-time. Używa pięciu originów dla
boiler/pump i trzech dla vibration.

Zwraca prognozy, `prediction_results_rolling_origins_*.csv`,
`origin_summary_*.csv` oraz `run_summary_*.json` w
`prediction_experiment_results/`.

## 13. Statystyka zbiorcza

**tmux:** krok długotrwały (bootstrap wielu porównań).

```bash
make batch-statistics
```

Skrypt: `src/13_batch_statistics.py`.

Pobiera najnowszy wynik rolling-origin i eksportuje test Friedmana, porównania
parowe, test normalności różnic, korektę Holma, wielkości efektu oraz 95%
przedziały bootstrapowe. Wyniki zapisuje w
`prediction_experiment_results/batch_statistics/`.

## 14. Jedno podsumowanie runu

```bash
make summarize-run
```

Skrypt: `src/14_summarize_run.py`.

Zbiera ścieżki oraz liczby wierszy najważniejszych artefaktów: raportu
czyszczenia, splitu, treningu SD2, realizacji braków, rekonstrukcji,
rolling-origin i statystyki. Zwraca jeden plik `experiment_run_summary.json` ze
statusem `complete` lub `incomplete`.

## Opcjonalne wizualizacje

**tmux:** dashboard zostaje włączony po odłączeniu sesji.

```bash
make visualize-reconstruction-error
make visualize-prediction-error
```

`src/A_visualize_reconstruction_error.py` pokazuje jakość rekonstrukcji.
`src/B_visualize_prediction_error.py` czyta wyniki rolling-origin i pozwala
filtrować m.in. model prognozy, rekonstruktor, horyzont, origin i zakres metryki.
Dashboardy nie są częścią obliczeń ani statystyki finalnej.
