# Rekonstrukcja szeregów czasowych — dotrenowany SD2 inpainting

Pakiet zawiera 1081 uszkodzonych plików CSV, kompletny dotrenowany model oraz
kod rekonstrukcji dla kodowań GAF, MTF, RP i spektrogramu (SPEC). Model jest
ładowany wyłącznie z katalogu lokalnego; rekonstrukcja nie pobiera wag z sieci.

## Instalacja na serwerze GPU

Wymagany jest Python 3.11 lub 3.12, sterownik NVIDIA zgodny z instalowaną wersją
PyTorch oraz około 10 GB miejsca po rozpakowaniu. Utwórz środowisko i zainstaluj
zależności (jeżeli serwer wymaga konkretnej wersji CUDA, zainstaluj najpierw
odpowiedni pakiet `torch` zgodnie z dokumentacją PyTorch):

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Uruchamianie

Jedno kodowanie:

```bash
python run_inpainting.py --encodings gaf
```

Kilka kodowań w jednym przebiegu:

```bash
python run_inpainting.py --encodings gaf mtf spec
```

Wszystkie cztery:

```bash
python run_inpainting.py --encodings all
```

Można ograniczyć pracę, np. do jednego zbioru i braków 20%:

```bash
python run_inpainting.py --encodings rp spec \
  --filter-dataset vibration_sensor_S1 --filter-rate 20
```

Pozostałe filtry: `--filter-technique MCAR MAR MNAR`,
`--filter-structure scattered contiguous mixed` oraz
`--filter-iteration 1 2 3`. Opcja `--force` nadpisuje istniejące wyniki.

Wyniki trafiają do `data/4_fixed_data/<zbiór>/<mechanizm>/<struktura>/<poziom>/`,
a metryki czasu i zasobów do `reconstruction_experiments_results/performance_metrics/`.
Każde kodowanie wykonywane jest kolejno na GPU, więc można bezpiecznie wybrać
kilka naraz bez równoczesnego ładowania kilku kopii modelu.

## Zawartość modelu

`models/sd2_windowed/best_model` to pełny snapshot pipeline'u Diffusers:
dotrenowany UNet oraz dołączone VAE, encoder tekstu, tokenizer i scheduler.
Informacje o treningu znajdują się w `models/sd2_windowed/training_report.json`.
