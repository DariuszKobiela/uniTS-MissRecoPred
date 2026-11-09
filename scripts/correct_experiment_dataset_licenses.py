#!/usr/bin/env python3
"""Build a redistribution-safe package from the frozen experiment package."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import subprocess
import tarfile
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES = ROOT / 'docs/experiment_dataset_release'
ALLOWED = {'boiler_outlet_temp_univ', 'vibration_sensor_S1'}
RELATIVE = Path('releases/2026-10-10')
LEGACY = ['data_0_source.tar.gz', 'data_1_missing.tar.gz', 'data_2_fixed.tar.gz', 'data_images.tar.gz']


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n')


def select(table):
    names = set(table['dataset_name'].to_pylist())
    if not names <= ALLOWED | {'pump_sensor_28_univ'}:
        raise ValueError(f'Unexpected datasets: {names}')
    return table.filter(pc.is_in(table['dataset_name'], value_set=pa.array(sorted(ALLOWED))))


def filter_split(data):
    data['datasets'] = {k: v for k, v in data['datasets'].items() if k.removesuffix('.csv') in ALLOWED}
    return data


def build(source: Path, output: Path):
    if output.exists():
        raise ValueError(f'Output exists: {output}')
    release = output / 'repo' / RELATIVE
    original = source / 'repo' / RELATIVE
    release.mkdir(parents=True)
    inventory = []
    original_inventory = pq.read_table(original / 'metadata/archive_members.parquet').to_pandas()
    known = {(r.archive, r.path): r for r in original_inventory.itertuples(index=False)}
    archives = {}
    for name in ['source_cleaned_split.tar.zst', 'missing.tar.zst', 'reconstructed.tar.zst']:
        print(f'Filtering archive {name}', flush=True)
        destination = release / 'archives' / name
        destination.parent.mkdir(exist_ok=True)
        reader = subprocess.Popen(['zstd', '-dcq', str(original / 'archives' / name)], stdout=subprocess.PIPE)
        with destination.open('wb') as compressed:
            writer = subprocess.Popen(['zstd', '-T4', '-3', '-q', '-c'], stdin=subprocess.PIPE, stdout=compressed)
            try:
                with tarfile.open(fileobj=reader.stdout, mode='r|') as src, tarfile.open(fileobj=writer.stdin, mode='w|') as dst:
                    for member in src:
                        if not member.isfile() or member.name.startswith('/') or '..' in Path(member.name).parts:
                            raise ValueError(f'Unsafe member: {member.name}')
                        if 'pump' in member.name.lower():
                            continue
                        is_split = member.name.endswith('/split_manifest.json')
                        if not is_split and not any(stem in member.name for stem in ALLOWED):
                            raise ValueError(f'Unclassified archive member: {member.name}')
                        data = src.extractfile(member).read()
                        old = known[(f'archives/{name}', member.name)]
                        if hashlib.sha256(data).hexdigest() != old.sha256:
                            raise ValueError(f'Source hash mismatch: {member.name}')
                        if is_split:
                            data = (json.dumps(filter_split(json.loads(data)), indent=2) + '\n').encode()
                        info = tarfile.TarInfo(member.name)
                        info.size, info.mode, info.mtime = len(data), 0o644, 0
                        dst.addfile(info, io.BytesIO(data))
                        inventory.append({'archive': f'archives/{name}', 'path': member.name,
                                          'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                                          'series_rows': old.series_rows})
                while reader.stdout.read(8 * 1024 * 1024):
                    pass
                reader.stdout.close()
                writer.stdin.close()
                if reader.wait() or writer.wait():
                    raise RuntimeError('Archive compression/decompression failed')
            except BaseException:
                reader.terminate(); writer.terminate()
                reader.wait(); writer.wait()
                raise
        archives[name] = {'files': sum(r['archive'] == f'archives/{name}' for r in inventory), 'bytes': destination.stat().st_size}
    (release / 'metadata').mkdir()
    frame = pd.DataFrame(inventory)
    frame['series_rows'] = frame.series_rows.astype('Int64')
    frame.to_parquet(release / 'metadata/archive_members.parquet', index=False, compression='zstd')
    write_json(release / 'metadata/split_manifest.json', filter_split(json.loads((original / 'metadata/split_manifest.json').read_text())))
    for name in ['missingness_realizations.csv', 'empirical_mask_coverage.csv']:
        table = pd.read_csv(original / 'metadata' / name)
        table[table.dataset_name.isin(ALLOWED)].to_csv(release / 'metadata' / name, index=False)
    (release / 'reports').mkdir()
    reader = subprocess.Popen(['zstd', '-dcq', str(original / 'reports/missingness_gaps.csv.zst')], stdout=subprocess.PIPE)
    with (release / 'reports/missingness_gaps.csv.zst').open('wb') as compressed:
        writer = subprocess.Popen(['zstd', '-T4', '-3', '-q', '-c'], stdin=subprocess.PIPE, stdout=compressed)
        with io.TextIOWrapper(writer.stdin, encoding='utf-8', newline='') as text:
            first = True
            for chunk in pd.read_csv(reader.stdout, chunksize=100_000):
                chunk[chunk.dataset_name.isin(ALLOWED)].to_csv(text, index=False, header=first)
                first = False
        reader.stdout.close()
        if reader.wait() or writer.wait():
            raise RuntimeError('Gap report compression/decompression failed')
    (release / 'results/prediction_by_model').mkdir(parents=True)
    forecast_manifest = json.loads((original / 'results/prediction_by_model/manifest.json').read_text())
    for model, meta in forecast_manifest['models'].items():
        target = release / 'results/prediction_by_model' / f'{model}.parquet'
        table = select(pq.read_table(original / 'results/prediction_by_model' / f'{model}.parquet'))
        pq.write_table(table, target, compression='zstd')
        meta.update(path=target.name, rows=table.num_rows, bytes=target.stat().st_size)
    forecast_manifest['selection'] = sorted(ALLOWED)
    write_json(release / 'results/prediction_by_model/manifest.json', forecast_manifest)
    table = select(pq.read_table(original / 'results/reconstruction_metrics.parquet'))
    pq.write_table(table, release / 'results/reconstruction_metrics.parquet', compression='zstd')
    # Preserve frozen numeric values; filter the original CSV rather than reserializing Parquet.
    csv = pd.read_csv(original / 'results/reconstruction_results_merged_20261010_202224.csv')
    csv[csv.dataset_name.isin(ALLOWED)].to_csv(release / 'results/reconstruction_results_merged_20261010_202224.csv', index=False)
    manifest = json.loads((original / 'release_manifest.json').read_text())
    per_model = {m: v['rows'] for m, v in forecast_manifest['models'].items()}
    manifest.update(packaged_at=datetime.now(timezone.utc).isoformat(),
                    redistribution_correction_date='2026-11-09', published_datasets=sorted(ALLOWED),
                    reconstruction_rows=table.num_rows, reconstructed_series=table.num_rows,
                    missing_realizations=720, forecast_rows=sum(per_model.values()),
                    forecast_rows_per_model=next(iter(per_model.values())),
                    checks={'missing_csvs': 720, 'reconstructed_csvs': 12240, 'recomputed_metric_checks': 24480},
                    archives=archives, historical_train_lengths={k:v for k,v in manifest['historical_train_lengths'].items() if k in ALLOWED},
                    split_resolution='Historical split retained for Boiler and Vibration; see metadata/split_manifest.json.',
                    legacy_policy='Mixed legacy archives are removed from the current revision.',
                    excluded_material='Pump series and metric records; mixed batch statistics, provenance bundles and old source snapshot omitted. Frozen source hashes identify private full-experiment inputs only.',
                    licenses={'boiler_outlet_temp_univ': 'cc-by-4.0', 'vibration_sensor_S1': 'cc0-1.0'})
    write_json(release / 'release_manifest.json', manifest)
    for name, target in [('DATASET_CARD.md', output/'repo/README.md'), ('LICENSE.md', output/'repo/LICENSE.md'),
                         ('validate_package.py', output/'validate_package.py'), ('upload.sh', output/'upload.sh'),
                         ('UPLOAD_INSTRUCTIONS.md', output/'UPLOAD_INSTRUCTIONS.md')]:
        shutil.copy2(TEMPLATES/name, target)
    shutil.copy2(ROOT/'scripts/publish_experiment_dataset_correction.py', output/'publish_experiment_dataset_correction.py')
    write_json(output/'deletions.json', {'legacy_files': LEGACY, 'removed_release_directories': ['provenance', 'results/batch_statistics', 'reproducibility']})
    with (output/'repo/SHA256SUMS').open('w') as handle:
        for path in sorted((output/'repo').rglob('*')):
            if path.is_file() and path.name != 'SHA256SUMS':
                with path.open('rb') as src:
                    checksum = hashlib.file_digest(src, 'sha256').hexdigest()
                handle.write(f'{checksum}  {path.relative_to(output/"repo").as_posix()}\n')
    print(f'Prepared {output}; forecast rows per model: {per_model}', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=ROOT/'release/huggingface/experiment-data-2026-10-10-ready')
    parser.add_argument('--output', type=Path, default=ROOT/'release/huggingface/experiment-data-2026-10-10-licensed')
    args = parser.parse_args()
    build(args.source.resolve(), args.output.resolve())
