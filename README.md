# uniTS-MissRecoPred

Framework do oceny metod rekonstrukcji braków w jednowymiarowych szeregach
czasowych i wpływu rekonstrukcji na lokalne prognozy rolling-origin.

## Aktywna ścieżka

Jedyną aktualną instrukcją wykonania eksperymentu jest
[EXPERIMENT_STEPS.md](EXPERIMENT_STEPS.md). Skrócona wersja:

```bash
uv sync --frozen --extra dev
uv run pytest -q
make pipeline-rebuttal
```

Pełny pipeline wykonuje kolejno:

1. czyszczenie danych źródłowych i jawny forward fill;
2. analizę horyzontów;
3. rozłączny split train / SD2-validation / test;
4. kontrolę projektu i reprezentacji SD2;
5. generowanie zbioru treningowego SD2;
6. analizę synthetic-to-real;
7. fine-tuning SD2;
8. generowanie braków MCAR/MAR/MNAR × scattered/contiguous/mixed;
9. raport luk i masek;
10. rekonstrukcję;
11. metryki rekonstrukcji;
12. prognozy rolling-origin;
13. statystykę Friedmana i porównania z korektą Holma;
14. jeden plik `experiment_run_summary.json` podsumowujący run.

## Struktura

- `src/1_*.py` … `src/14_*.py` — wykonywalne etapy eksperymentu w kolejności;
- `src/A_*.py`, `src/B_*.py` — opcjonalne narzędzia pomocnicze;
- `src/reconstruction_models/` — aktywne modele rekonstrukcji;
- `src/prediction_models/` — modele SARIMAX i XGBoost używane przez rolling-origin;
- `src/utils/` — moduły współdzielone;
- `config/` — aktywna konfiguracja;
- `tests/` — testy aktywnej ścieżki;
- `legacy/` — wycofany kod zachowany wyłącznie referencyjnie.

## Dane i wyniki

Dane wejściowe umieszcza się w `data/0_source_data/`. Czyszczenie zapisuje
kompletne szeregi do `data/1_cleaned_data/` i raport uzupełnień do
`data/1_cleaned_data/reports/cleaning_report.csv`.

Najważniejsze wyniki trafiają do:

- `reconstruction_experiments_results/` — metryki rekonstrukcji;
- `prediction_experiment_results/` — prognozy i metryki rolling-origin;
- `prediction_experiment_results/batch_statistics/` — statystyka zbiorcza;
- `experiment_run_summary.json` — zbiorcze podsumowanie całego przebiegu.

## Komendy

```bash
make help
make test-unit
make pipeline-rebuttal
make visualize-reconstruction-error  # opcjonalny dashboard rekonstrukcji
make visualize-prediction-error      # opcjonalny dashboard prognoz
```

Szczegóły metodologiczne znajdują się w [REBUTTAL_README.md](REBUTTAL_README.md).
