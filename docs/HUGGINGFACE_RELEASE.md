# Hugging Face release: windowed SD2 v2

This project publishes the current windowed Stable Diffusion 2 inpainting
artifacts to:

- model: `dariusz-kobiela/stable-diffusion-2-inpainting-gaf-mtf-rp-spec`;
- dataset: `dariusz-kobiela/stable-diffusion-2-inpainting-gaf-mtf-rp-spec-training-data`.

The source artifacts are `models/sd2_windowed/best_model` and
`data/sd2_windowed_training`. Do not upload those directories directly. Build
the release packages first so that public paths, cards, split metadata,
checksums, and provenance are normalized.

## Build and validate

Run from the repository root:

```bash
source .venv/bin/activate
python scripts/prepare_huggingface_release.py
cd release/huggingface/$(date -u +%F)
python validate_packages.py
```

The generated `UPLOAD_INSTRUCTIONS.md` contains the exact upload commands. The
release builder refuses to overwrite an existing dated directory. Move an old
directory aside if a clean rebuild is required.

## Release contract

The dataset package contains 8,000 aligned triplets (`clean`, `conditioning`,
`mask`) from 2,000 grouped base series, plus the exact grouped split: 6,800
training and 1,200 validation triplets. Raw assets remain PNG/NumPy for exact
training reproducibility; train and validation metadata are additionally
published as Zstandard-compressed Parquet for the Hugging Face dataset viewer.

The model package is a complete Diffusers inpainting pipeline. Only the UNet
was fine-tuned; the VAE and text encoder are the frozen components used during
training.

Before synchronizing `main`, create the two `legacy-v1-before-windowed-2026`
tags. Upload uses `--delete="*"` because the repositories currently contain
files from an incompatible legacy dataset/checkpoint. The legacy tags preserve
that state, while the deletion pattern makes the new working tree match the
validated release package exactly.
