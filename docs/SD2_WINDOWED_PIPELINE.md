# Windowed Stable Diffusion 2 workflow

## Reconstruction contract

Stable Diffusion no longer receives one image made from the complete training
series. Each built-in SD2 reconstruction model processes local windows.

The default configuration uses:

- 512 samples per context window;
- 64 context samples on each side;
- a disjoint 384-sample output core;
- 512 x 512 input images;
- writes only missing values from the output core;
- preserves every observed value exactly.

Dataset-specific window sizes can be configured under
`computation.stable_diffusion.windowing.by_dataset`.

`window_samples` and `image_size` are different:

- `window_samples` controls the physical time span and samples per pixel;
- `image_size` controls the SD2 tensor resolution.

SD2 was trained natively at 512 x 512. A 1024 image has about four times as many
pixels and a 2048 image about sixteen times as many. These resolutions are
experimental and may run out of GPU memory. Increasing the sample window while
keeping a 512 image is normally the safer way to cover longer cycles.

## Design analysis and optimization

Generate the analytical report without GPU inference:

    make analyze-sd2-design

Outputs:

- `data/1_6_sd2_optimization/sd2_design_analysis.csv`;
- `data/1_6_sd2_optimization/sd2_recommendations.json`;
- `data/1_6_sd2_optimization/sd2_design_report.md`.

The analytical score combines retained image context (55%), coverage of the
selected long forecasting horizon (35%), and relative effective-window size
(10%). A conservative penalty is applied outside SD2's native 512-pixel
resolution; memory is a tie-breaker, while potential calls per file are
reported separately. This is a provisional engineering recommendation, not an
accuracy result.

Run model-based optimization on CUDA:

    make optimize-sd2-design

The GPU mode uses clean fragments from the real training series, generates
MCAR/MAR/MNAR validation gaps, and uses Optuna to select:

- sample window: 512, 1024, or 2048;
- image resolution: 512, 1024, or 2048;
- representation-specific prompt;
- inference steps: 20, 30, 42, or 50;
- guidance scale: 1.0, 3.0, 5.0, or 7.5.

For a safer first run, keep native image resolution:

    uv run python src/optimization/analyze_sd2_design.py \
      --run-inference --image-sizes 512 --n-trials 12 --cases-per-dataset 3

The GPU run writes:

- detailed cases to `data/1_6_sd2_optimization/sd2_inference_trials.csv`;
- winners to `data/1_6_sd2_optimization/sd2_runtime_overrides.json`.

The main reconstruction pipeline reads the latter path from `config.yaml` and
automatically applies the winning settings for each dataset and SD2 model.

After fine-tuning, optimize the new local model variants separately:

    uv run python src/optimization/analyze_sd2_design.py --run-inference \
      --image-sizes 512 --models \
      stable_diffusion_2_gaf_finetuned stable_diffusion_2_mtf_finetuned \
      stable_diffusion_2_rp_finetuned stable_diffusion_2_spec_finetuned

Repeated GPU runs merge model-specific winners into the same overrides file.

## Existing training data

The old `stdiff_training_data/` contains 2,000 base series, 8,000 original
images and 8,000 corrupted images. It must not be mixed with the corrected
dataset because:

- it uses the superseded encoders and inverse assumptions;
- image sizes vary with the generated series length;
- there are no explicit binary inpainting masks;
- masks were inferred later from differences between two images;
- its examples do not use the same local-window contract as inference.

Its metadata and pattern distribution remain useful as a legacy description and
as inspiration for the new signal generator.

## Generate the corrected training dataset

Generate 2,000 synthetic base windows and four encodings per window:

    make generate-sd2-training-data

Output structure:

    data/sd2_windowed_training/
      clean/
      conditioning/
      masks/
      manifest.jsonl
      dataset_summary.json

Each manifest row describes one explicit training triplet. The conditioning
image is produced from the corrupted series after the same linear fill used by
inference. The mask is a real binary PNG: white is regenerated and black is
preserved.

A small test dataset can be generated with:

    uv run python src/training/generate_sd2_windowed_dataset.py \
      --samples 20 --output /tmp/sd2-training-smoke --image-size 512

The default source is synthetic to avoid target-series leakage. The optional
`--source mixed --real-share 0.25` mode adds real windows, but it should be
used only in an explicitly described domain-adaptation experiment. Training on
the clean target series before evaluating their corrupted copies would otherwise
make the comparison optimistic.

## Fine-tuning

Run:

    make train-sd2-windowed

The trainer:

- trains only the inpainting UNet;
- freezes VAE and text encoder;
- uses explicit clean/conditioning/mask triplets;
- splits by `series_id`, preventing the four encodings of one signal from
  crossing train and validation;
- supports fp16/bf16, gradient accumulation, gradient checkpointing and
  optional xFormers;
- applies early stopping on validation diffusion loss;
- saves a complete pipeline in `models/sd2_windowed/best_model`.

After training, the reconstruction loader automatically prefers that local
pipeline. A different full pipeline can be selected with:

    SD2_FINETUNED_MODEL=/path/to/model make reconstruct-datasets

Do not compare results produced by the old fine-tuned weights with results from
the corrected encodings as if they were the same model.
