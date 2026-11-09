# Experiment dataset release: source licenses corrected 2026-11-09

Target: `dariusz-kobiela/univariate-time-series-with-missingness-and-reconstructed`.
This release publishes Boiler under CC BY 4.0 and Vibration under CC0 1.0.
See [the license notice](experiment_dataset_release/LICENSE.md) for attribution,
source links and transformation notices. Pump has no stated redistribution
license and is excluded, including metric records. Its source remains linked.

To correct the existing local frozen package:

```bash
.venv/bin/python scripts/correct_experiment_dataset_licenses.py
.venv/bin/python release/huggingface/experiment-data-2026-10-10-licensed/validate_package.py --deep
```

The correction streams and verifies the original archives against their member
hashes, discards Pump, filters reports and metric tables, and regenerates split
metadata, inventories, counts and checksums. Retained series bytes and Parquet
metric values stay unchanged. Mixed batch statistics and provenance/source
bundles are omitted. Local experimental data is preserved for private research.
The builder refuses to overwrite an existing output; use `--output` for another
location. Dependencies: the project environment and `zstd`.

A full rebuild from working experiment artifacts is also available:

```bash
.venv/bin/python scripts/prepare_experiment_dataset_release.py
```

It first validates the complete frozen experiment in a sibling directory ending
in `-private-staging`, then creates the filtered `-licensed` public package.
**Only the `-licensed/repo` payload may be published.** Existing older local
packages still contain Pump and must not be uploaded. Their old validators and
upload scripts are superseded by this workflow.

Publication uses:

```bash
bash release/huggingface/experiment-data-2026-10-10-licensed/upload.sh
```

The publisher validates the corrected package and makes a single Hugging Face
commit replacing current release files and deleting mixed legacy archives
(`data_0_source.tar.gz`, `data_1_missing.tar.gz`, `data_2_fixed.tar.gz`,
`data_images.tar.gz`) and superseded release bundles. It uses the inspected
remote commit as its parent to reject concurrent changes. An additive upload
alone would leave Pump publicly downloadable and is insufficient.

The correction removes files from the current revision. Previous Hugging Face
Git revisions can still contain withdrawn files. Purging history is a separate,
irreversible operation and is not performed by this workflow.

Templates are in `docs/experiment_dataset_release/`; maintain those and rebuild
so future uploads preserve the licensing and filtering policy.
