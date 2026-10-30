#!/usr/bin/env python3
"""Build upload-ready Hugging Face dataset and model directories."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE_DATASET = ROOT / "data" / "sd2_windowed_training"
SOURCE_MODEL = ROOT / "models" / "sd2_windowed"
RELEASE_ROOT = ROOT / "release" / "huggingface" / datetime.now(timezone.utc).date().isoformat()
DATASET_NAME = "stable-diffusion-2-inpainting-gaf-mtf-rp-spec-training-data"
MODEL_NAME = "stable-diffusion-2-inpainting-gaf-mtf-rp-spec"
HF_NAMESPACE = "dariusz-kobiela"
DATASET_REPO = f"{HF_NAMESPACE}/{DATASET_NAME}"
MODEL_REPO = f"{HF_NAMESPACE}/{MODEL_NAME}"
BASE_MODEL = "stabilityai/stable-diffusion-2-inpainting"


DATASET_CARD = """---
pretty_name: SD2 Time-Series Inpainting Triplets (GAF, MTF, RP, SPEC)
license: cc-by-4.0
task_categories:
- image-to-image
size_categories:
- 1K<n<10K
tags:
- time-series
- inpainting
- stable-diffusion
- gramian-angular-field
- markov-transition-field
- recurrence-plot
- spectrogram
configs:
- config_name: default
  data_files:
  - split: train
    path: metadata/train.parquet
  - split: validation
    path: metadata/validation.parquet
---

# SD2 time-series inpainting training data

This is the final synthetic training dataset used to fine-tune the shared
windowed Stable Diffusion 2 inpainting UNet reported in the accompanying
industrial time-series reconstruction study.

## Dataset summary

- 2,000 independently generated base series, each with 512 samples.
- 10 signal families: harmonic, trend, seasonal, autoregressive, random walk,
  steps, spikes, chirp, heteroskedastic, and mixed.
- Four encodings per series: GAF, MTF, continuous RP, and spectrogram.
- 8,000 aligned `(clean, conditioning, mask)` triplets at 512 x 512 pixels.
- Missingness grid: MCAR/MAR/MNAR, scattered/contiguous/mixed, and requested
  rates 2%, 5%, 20%, and 50%.
- Binary masks use white pixels for inpainting and black pixels for preservation.
- The conditioning image is built from a linearly filled corrupted signal.
- Seed: 42.

This release replaces the earlier dataset version based on random, block,
periodic, and edge masks.

## Files

```text
clean/                 8,000 target PNG images
conditioning/          8,000 conditioning PNG images
masks/                 8,000 binary mask PNG images
series/                2,000 source arrays in NumPy format
manifest.jsonl         one record per triplet
metadata.csv           tabular subset of the manifest for inspection
metadata/              train/validation metadata in Parquet for the HF viewer
splits.json            exact grouped train/validation series IDs
dataset_summary.json   dataset contract and aggregate counts
release_manifest.json  release provenance and file counts
SHA256SUMS             integrity hashes
```

Paths in `manifest.jsonl` are relative to the repository root. Each record
contains the series family and generator parameters, encoding, missingness
mechanism and structure, requested and realized rates, gap diagnostics, seed,
prompt, and paths to the clean/conditioning/mask files.

## Train/validation split

The published split is grouped by `series_id`: all four encodings of a base
series remain in the same subset. NumPy `default_rng(42)` shuffles 2,000 series
IDs; 300 series (1,200 triplets) form validation and 1,700 series (6,800
triplets) form training. Exact IDs are stored in `splits.json`.

## Loading example

```python
import json
from pathlib import Path
from PIL import Image

root = Path("/path/to/download")
with (root / "manifest.jsonl").open() as handle:
    record = json.loads(next(handle))
clean = Image.open(root / record["clean"]).convert("RGB")
conditioning = Image.open(root / record["conditioning"]).convert("RGB")
mask = Image.open(root / record["mask"]).convert("L")
```

The metadata table can also be loaded without downloading all images:

```python
from datasets import load_dataset

metadata = load_dataset(
    "dariusz-kobiela/stable-diffusion-2-inpainting-gaf-mtf-rp-spec-training-data"
)
print(metadata)
```

For streaming or large-scale use, read `manifest.jsonl` line by line instead of
loading it entirely into memory.

## Intended use and limitations

The dataset is intended for research on image-mediated reconstruction of
univariate time series. It contains synthetic signals rather than natural
sensor histories. A classifier distinguishes these synthetic windows from the
three real evaluation datasets with ROC AUC 0.904, so the dataset must not be
described as domain-equivalent to industrial data. MNAR is a controlled
value-dependent mechanism, not a sensor-specific failure process.

## Reproducibility

Generation code is in
<https://github.com/DariuszKobiela/uniTS-MissRecoPred>:

```bash
uv run python src/5_generate_sd2_training_data.py \
  --samples 2000 --workers 20 --overwrite
```

The scientific manuscript uses the dated frozen results from 2026-10-10.

## Citation

Until a DOI for the manuscript is assigned, cite the repository and this
dataset revision, including its Hugging Face commit hash.
"""


MODEL_CARD = """---
license: openrail++
library_name: diffusers
pipeline_tag: image-to-image
base_model: stabilityai/stable-diffusion-2-inpainting
datasets:
- dariusz-kobiela/stable-diffusion-2-inpainting-gaf-mtf-rp-spec-training-data
tags:
- stable-diffusion
- diffusers
- inpainting
- time-series
- gramian-angular-field
- markov-transition-field
- recurrence-plot
- spectrogram
---

# Windowed SD2 inpainting for time-series image representations

This repository contains the final shared Stable Diffusion 2 inpainting
pipeline used in the 2026 windowed time-series reconstruction experiment. It
replaces the earlier public checkpoint trained with four heuristic mask types.

The UNet was fine-tuned jointly on GAF, MTF, RP, and spectrogram images. The VAE
and CLIP text encoder were frozen. The repository is a complete Diffusers
pipeline and can be loaded directly.

## Training

| Setting | Value |
|---|---:|
| Base model | `stabilityai/stable-diffusion-2-inpainting` |
| Base series | 2,000 |
| Training triplets | 6,800 |
| Validation triplets | 1,200 |
| Image/window size | 512 pixels / 512 samples |
| Batch size | 4 |
| Gradient accumulation | 2 |
| Learning rate | 1e-5 |
| Optimizer | AdamW, weight decay 0.01 |
| Precision | FP16 |
| Seed | 42 |
| Best epoch | 3 |
| Early stop | after epoch 8, patience 5 |
| Best validation loss | 0.0446368243 |

The split is grouped by base `series_id`, so representations of one synthetic
series cannot cross train and validation subsets.

## Loading

```python
import torch
from diffusers import StableDiffusionInpaintPipeline

pipe = StableDiffusionInpaintPipeline.from_pretrained(
    "dariusz-kobiela/stable-diffusion-2-inpainting-gaf-mtf-rp-spec",
    torch_dtype=torch.float16,
    safety_checker=None,
).to("cuda")

result = pipe(
    prompt="high quality recurrence plot mathematical visualization",
    image=conditioning_image.convert("RGB").resize((512, 512)),
    mask_image=binary_mask.convert("L").resize((512, 512)),
    num_inference_steps=42,
    guidance_scale=7.5,
    generator=torch.Generator("cuda").manual_seed(42),
).images[0]
```

The output is an inpainted representation image. Decoding it back to a time
series requires the representation-specific inverse and metadata implemented
in <https://github.com/DariuszKobiela/uniTS-MissRecoPred>.

## Runtime contract used in the paper

- input windows contain at most 512 samples;
- interior windows have 64 samples of context on each side and own a disjoint
  384-sample output core;
- each missing sample is committed exactly once;
- observed time-series values are never overwritten;
- GAF/MTF/RP masks cover the affected image rows and columns;
- spectrogram masks cover affected time-frequency columns;
- inference uses 42 steps, guidance 7.5, and deterministic per-window seeds.

These runtime settings were fixed by design and representation ablations. The
available HPO trace was incomplete and is not evidence of exhaustive empirical
optimization.

## Evaluation summary

On the final balanced reconstruction grid (three datasets, three mechanisms,
three temporal structures, four rates, and ten realizations):

- fine-tuned MTF ranked 3rd of 17 reconstructors at 10.095% sMAPE;
- zero-shot MTF ranked 4th at 10.219%;
- fine-tuning improved GAF, MTF, and RP modestly;
- fine-tuned spectrogram ranked 17th at 90.407% sMAPE;
- transform-only spectrogram controls already had about 3.5% sMAPE and RMSE
  near 21;
- the synthetic-to-real classifier reached ROC AUC 0.904.

The model is therefore representation- and dataset-dependent. These results do
not establish superiority of image inpainting or diffusion models in general.

## Intended use

Research on reconstruction of incomplete univariate time series encoded as GAF,
MTF, RP, or spectrogram images. The model is not intended for direct use on raw
time-series arrays or natural photographs and has not been validated for
safety-critical control.

## Limitations

- trained exclusively on synthetic signals;
- one shared checkpoint for four visually different representations;
- only three industrial univariate datasets in the reported evaluation;
- approximate inversion for MTF and spectrogram;
- nonlocal two-dimensional masks for GAF, MTF, and RP;
- no claim of optimized inference hyperparameters.

## License

This fine-tuned model is a derivative of Stable Diffusion 2 inpainting and is
distributed under the CreativeML Open RAIL++-M terms associated with the base
model. Review `LICENSE_NOTICE.md` and the upstream license before use.

## Citation

Until a DOI for the manuscript is assigned, cite the code repository, this
model revision, the training dataset revision, and the Stable Diffusion latent
diffusion paper.
"""


LICENSE_NOTICE = """# License notice

This repository contains weights derived from Stable Diffusion 2 inpainting.
Use and redistribution are subject to the CreativeML Open RAIL++-M terms of the
base model:

https://huggingface.co/stabilityai/stable-diffusion-2-inpainting

The repository metadata uses the Hugging Face identifier `openrail++`. This
notice is not a substitute for the full upstream license text.
"""


UPLOAD_INSTRUCTIONS = """# Upload instructions

The two directories next to this file are the repository roots to upload.
Authenticate interactively; do not place tokens in command lines, Git remotes,
scripts, or this directory.

```bash
hf auth login
hf auth whoami

# Preserve the old public revisions before replacing their working trees.
hf repo tag create dariusz-kobiela/stable-diffusion-2-inpainting-gaf-mtf-rp-spec \\
  legacy-v1-before-windowed-2026 \\
  --message="Legacy checkpoint before the windowed v2 release"

hf repo tag create dariusz-kobiela/stable-diffusion-2-inpainting-gaf-mtf-rp-spec-training-data \\
  legacy-v1-before-windowed-2026 \\
  --repo-type=dataset \\
  --message="Legacy dataset before the windowed v2 release"

HF_XET_HIGH_PERFORMANCE=1 hf upload \\
  dariusz-kobiela/stable-diffusion-2-inpainting-gaf-mtf-rp-spec-training-data \\
  ./stable-diffusion-2-inpainting-gaf-mtf-rp-spec-training-data . \\
  --repo-type=dataset \\
  --delete="*" \\
  --commit-message="Final 2026 windowed SD2 training dataset"

HF_XET_HIGH_PERFORMANCE=1 hf upload \\
  dariusz-kobiela/stable-diffusion-2-inpainting-gaf-mtf-rp-spec \\
  ./stable-diffusion-2-inpainting-gaf-mtf-rp-spec . \\
  --delete="*" \\
  --commit-message="Final 2026 windowed SD2 checkpoint"

hf repo tag create dariusz-kobiela/stable-diffusion-2-inpainting-gaf-mtf-rp-spec \\
  windowed-v2-2026

hf repo tag create dariusz-kobiela/stable-diffusion-2-inpainting-gaf-mtf-rp-spec-training-data \\
  windowed-v2-2026 \\
  --repo-type=dataset

hf models info dariusz-kobiela/stable-diffusion-2-inpainting-gaf-mtf-rp-spec \\
  --expand=sha,lastModified

hf datasets info dariusz-kobiela/stable-diffusion-2-inpainting-gaf-mtf-rp-spec-training-data \\
  --expand=sha,lastModified
```

`hf upload` uses the current Hugging Face Xet-backed uploader and can be
restarted if interrupted. Run the commands from this release directory.

Before upload:

1. Review the dataset CC BY 4.0 choice and the model's inherited Open RAIL++-M
   terms.
2. Run `python validate_packages.py`.
3. Confirm that both legacy tags were created. The `--delete="*"` option makes
   the remote working tree match the prepared package and removes incompatible
   legacy files; it does not delete the tagged legacy revisions.
4. After upload, record both Hugging Face commit hashes in the manuscript.
"""


VALIDATOR = """#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def validate(directory: Path) -> None:
    sums = directory / "SHA256SUMS"
    if not sums.is_file():
        raise SystemExit(f"Missing {sums}")
    checked = 0
    for line in sums.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        path = directory / relative
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected:
            raise SystemExit(f"Checksum mismatch: {path}")
        checked += 1
    manifest = json.loads((directory / "release_manifest.json").read_text(encoding="utf-8"))
    if checked != manifest["payload_files"] + 1:
        raise SystemExit(f"Unexpected checksummed-file count in {directory}")
    print(
        f"{directory.name}: {checked} checksums OK "
        f"({manifest['payload_files']} payload files + release manifest)"
    )


dataset = ROOT / "stable-diffusion-2-inpainting-gaf-mtf-rp-spec-training-data"
model = ROOT / "stable-diffusion-2-inpainting-gaf-mtf-rp-spec"
validate(dataset)
validate(model)

dataset_release = json.loads((dataset / "release_manifest.json").read_text(encoding="utf-8"))
model_release = json.loads((model / "release_manifest.json").read_text(encoding="utf-8"))
if not dataset_release["repository"].startswith("dariusz-kobiela/"):
    raise SystemExit("Dataset release points to the wrong HF namespace")
if not model_release["repository"].startswith("dariusz-kobiela/"):
    raise SystemExit("Model release points to the wrong HF namespace")

records = sum(1 for line in (dataset / "manifest.jsonl").open(encoding="utf-8") if line.strip())
if records != 8_000:
    raise SystemExit(f"Unexpected manifest row count: {records}")
with (dataset / "metadata.csv").open(encoding="utf-8", newline="") as handle:
    split_counts = {"train": 0, "validation": 0}
    train_ids, validation_ids = set(), set()
    for row in csv.DictReader(handle):
        split_counts[row["split"]] += 1
        (validation_ids if row["split"] == "validation" else train_ids).add(int(row["series_id"]))
if split_counts != {"train": 6_800, "validation": 1_200}:
    raise SystemExit(f"Unexpected split counts: {split_counts}")
if train_ids & validation_ids:
    raise SystemExit("Train/validation series_id leakage detected")

for directory in (dataset, model):
    for card in (directory / "README.md", directory / "release_manifest.json"):
        if "Daro77" in card.read_text(encoding="utf-8"):
            raise SystemExit(f"Legacy namespace remains in {card}")
print("Release metadata and grouped split contract OK")
"""


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def hardlink_file(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def hardlink_tree(source: Path, target: Path) -> None:
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        destination = target / relative
        if path.is_dir():
            destination.mkdir(parents=True, exist_ok=True)
        elif path.is_file():
            hardlink_file(path, destination)
        else:
            raise ValueError(f"Unsupported source entry: {path}")


def payload_stats(directory: Path, excluded: set[str] | None = None) -> tuple[int, int]:
    excluded = excluded or set()
    paths = [
        path
        for path in directory.rglob("*")
        if path.is_file() and path.relative_to(directory).as_posix() not in excluded
    ]
    return len(paths), sum(path.stat().st_size for path in paths)


def write_checksums(directory: Path) -> None:
    output = []
    for path in sorted(directory.rglob("*")):
        if not path.is_file() or path.name == "SHA256SUMS":
            continue
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        output.append(f"{digest.hexdigest()}  {path.relative_to(directory).as_posix()}")
    (directory / "SHA256SUMS").write_text("\n".join(output) + "\n", encoding="utf-8")


def validate_sources() -> tuple[list[dict], dict, dict]:
    dataset_summary = read_json(SOURCE_DATASET / "dataset_summary.json")
    training_report = read_json(SOURCE_MODEL / "training_report.json")
    records = [
        json.loads(line)
        for line in (SOURCE_DATASET / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if dataset_summary["base_series"] != 2_000 or dataset_summary["training_triplets"] != 8_000:
        raise ValueError("Unexpected dataset summary")
    if len(records) != 8_000 or len({record["series_id"] for record in records}) != 2_000:
        raise ValueError("Unexpected manifest cardinality")
    if {record["encoding"] for record in records} != {"gaf", "mtf", "rp", "spec"}:
        raise ValueError("Unexpected encoding set")
    if {record["mechanism"] for record in records} != {"MCAR", "MAR", "MNAR"}:
        raise ValueError("Unexpected mechanism set")
    if {record["structure"] for record in records} != {"scattered", "contiguous", "mixed"}:
        raise ValueError("Unexpected structure set")
    for folder, expected in [
        ("clean", 8_000),
        ("conditioning", 8_000),
        ("masks", 8_000),
        ("series", 2_000),
    ]:
        actual = sum(1 for path in (SOURCE_DATASET / folder).iterdir() if path.is_file())
        if actual != expected:
            raise ValueError(f"Unexpected file count in {folder}: {actual}")
    if training_report["train_triplets"] != 6_800 or training_report["validation_triplets"] != 1_200:
        raise ValueError("Unexpected training split")
    if abs(training_report["best_validation_loss"] - 0.044636824319604786) > 1e-12:
        raise ValueError("Unexpected validation loss")
    required = [
        "model_index.json",
        "scheduler/scheduler_config.json",
        "text_encoder/model.safetensors",
        "tokenizer/tokenizer.json",
        "unet/diffusion_pytorch_model.safetensors",
        "vae/diffusion_pytorch_model.safetensors",
    ]
    for relative in required:
        if not (SOURCE_MODEL / "best_model" / relative).is_file():
            raise FileNotFoundError(relative)
    return records, dataset_summary, training_report


def build_dataset(target: Path, records: list[dict], summary: dict) -> None:
    target.mkdir(parents=True)
    for name in ["clean", "conditioning", "masks", "series"]:
        hardlink_tree(SOURCE_DATASET / name, target / name)
    hardlink_file(SOURCE_DATASET / "manifest.jsonl", target / "manifest.jsonl")
    hardlink_file(SOURCE_DATASET / "dataset_summary.json", target / "dataset_summary.json")

    groups = np.array(sorted({int(record["series_id"]) for record in records}), dtype=int)
    rng = np.random.default_rng(42)
    rng.shuffle(groups)
    validation = {int(value) for value in groups[:300]}
    train = sorted(set(map(int, groups)) - validation)
    split = {
        "schema_version": "1.0",
        "method": "grouped by series_id",
        "seed": 42,
        "validation_share": 0.15,
        "train_series_ids": train,
        "validation_series_ids": sorted(validation),
        "train_triplets": 6_800,
        "validation_triplets": 1_200,
    }
    write_json(target / "splits.json", split)

    columns = [
        "series_id",
        "encoding",
        "source_kind",
        "source_name",
        "pattern",
        "window_samples",
        "image_size",
        "mechanism",
        "structure",
        "missing_rate_requested",
        "missing_rate_actual",
        "n_gaps",
        "gap_length_samples_mean",
        "gap_length_samples_median",
        "gap_length_samples_p90",
        "gap_length_samples_max",
        "singleton_gap_percent",
        "seed",
        "prompt",
        "clean",
        "conditioning",
        "mask",
        "series_path",
        "split",
    ]
    with (target / "metadata.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        metadata_rows = []
        for record in records:
            row = {column: record.get(column) for column in columns}
            row["split"] = "validation" if int(record["series_id"]) in validation else "train"
            writer.writerow(row)
            metadata_rows.append(row)

    metadata = pd.DataFrame(metadata_rows, columns=columns)
    metadata_dir = target / "metadata"
    metadata_dir.mkdir()
    for split_name in ("train", "validation"):
        split_metadata = metadata.loc[metadata["split"] == split_name].reset_index(drop=True)
        split_metadata.to_parquet(
            metadata_dir / f"{split_name}.parquet",
            index=False,
            compression="zstd",
        )

    (target / "README.md").write_text(DATASET_CARD, encoding="utf-8")
    (target / "LICENSE.md").write_text(
        "# License\n\nThis dataset release is provided under CC BY 4.0: "
        "https://creativecommons.org/licenses/by/4.0/\n",
        encoding="utf-8",
    )
    count, size = payload_stats(target, {"release_manifest.json", "SHA256SUMS"})
    write_json(
        target / "release_manifest.json",
        {
            "schema_version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "repository": DATASET_REPO,
            "source_artifact": "data/sd2_windowed_training",
            "base_series": summary["base_series"],
            "triplets": summary["training_triplets"],
            "payload_files": count,
            "payload_bytes": size,
            "large_files_are_hardlinked_locally": True,
        },
    )
    write_checksums(target)


def build_model(target: Path, training_report: dict, dataset_summary: dict) -> None:
    target.mkdir(parents=True)
    hardlink_tree(SOURCE_MODEL / "best_model", target)

    model_index_path = target / "model_index.json"
    model_index = read_json(model_index_path)
    model_index["_name_or_path"] = BASE_MODEL
    model_index_path.unlink()
    write_json(model_index_path, model_index)

    public_report = dict(training_report)
    public_report["base_model"] = BASE_MODEL
    public_report["data_dir"] = f"hf://datasets/{DATASET_REPO}"
    public_report["best_model"] = f"hf://{MODEL_REPO}"
    write_json(target / "training_report.json", public_report)
    write_json(target / "dataset_summary.json", dataset_summary)
    (target / "README.md").write_text(MODEL_CARD, encoding="utf-8")
    (target / "LICENSE_NOTICE.md").write_text(LICENSE_NOTICE, encoding="utf-8")
    count, size = payload_stats(target, {"release_manifest.json", "SHA256SUMS"})
    write_json(
        target / "release_manifest.json",
        {
            "schema_version": "1.0",
            "created_at": datetime.now(timezone.utc).isoformat(),
            "repository": MODEL_REPO,
            "base_model": BASE_MODEL,
            "source_artifact": "models/sd2_windowed/best_model",
            "best_epoch": 3,
            "best_validation_loss": training_report["best_validation_loss"],
            "payload_files": count,
            "payload_bytes": size,
            "large_files_are_hardlinked_locally": True,
        },
    )
    write_checksums(target)


def main() -> None:
    records, dataset_summary, training_report = validate_sources()
    if RELEASE_ROOT.exists():
        raise FileExistsError(
            f"{RELEASE_ROOT} already exists; inspect or move it before rebuilding"
        )
    RELEASE_ROOT.parent.mkdir(parents=True, exist_ok=True)
    staging = RELEASE_ROOT.with_name(f".{RELEASE_ROOT.name}.staging-{os.getpid()}")
    staging.mkdir()
    try:
        build_dataset(staging / DATASET_NAME, records, dataset_summary)
        build_model(staging / MODEL_NAME, training_report, dataset_summary)
        (staging / "UPLOAD_INSTRUCTIONS.md").write_text(UPLOAD_INSTRUCTIONS, encoding="utf-8")
        (staging / "validate_packages.py").write_text(VALIDATOR, encoding="utf-8")
        staging.rename(RELEASE_ROOT)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(f"Prepared Hugging Face release in {RELEASE_ROOT.relative_to(ROOT)}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
