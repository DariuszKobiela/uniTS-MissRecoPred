# Protokół eksperymentu rebuttal

Repozytorium zawiera jedną aktywną ścieżkę: rekonstrukcja braków oraz lokalna
ocena prognoz rolling-origin. Dokładne polecenia i artefakty opisuje
[EXPERIMENT_STEPS.md](EXPERIMENT_STEPS.md).

## Najważniejsze decyzje metodologiczne

- Dane są dzielone czasowo na `train`, `sd2_validation` i końcowy `test`.
- MCAR, MAR i MNAR są krzyżowane ze strukturami `scattered`, `contiguous` i
  `mixed`.
- Braki obecne już w surowych CSV są przed eksperymentem uzupełniane poprzednią
  obserwacją (`ffill`) i raportowane w `cleaning_report.csv`.
- Zero-shot i fine-tuned SD2 są oceniane dla GAF, MTF, RP i spektrogramu.
- Prognozowanie dopasowuje model lokalnie dla każdego źródła i originu.
- Jedna prognoza do `H_max` jest dzielona na horyzonty skumulowane i biny
  lead-time.
- Baseline'y to persistence, seasonal-naive, SARIMAX, XGBoost i bezpośredni
  XGBoost na danych z brakami oraz wskaźnikami missingness.
- Boiler i pump używają pięciu originów, vibration trzech.
- Podstawowe metryki prognozy to MAE, RMSE i MASE; MAPE oraz sMAPE są wtórne.
- Statystyka uwzględnia pełny klucz parowania i eksportuje test Friedmana,
  porównania parowe, korektę Holma, wielkość efektu i przedziały bootstrapowe.

## Oczekiwane artefakty

- raport czyszczenia i forward fill;
- metadane horyzontów i manifest splitu;
- kontrole round-trip/oracle reprezentacji SD2;
- zbiór i raport treningu fine-tuned SD2;
- raport synthetic-to-real;
- raport realizacji braków, luk i empirycznego pokrycia masek;
- kompletne rekonstrukcje i metryki;
- wyniki oraz podsumowania rolling-origin;
- pliki statystyki zbiorczej;
- `experiment_run_summary.json`.

Wycofana implementacja znajduje się w `legacy/` i nie jest wywoływana przez
aktywny Makefile.
