#!/usr/bin/env python3
"""Publish the reviewed license correction and remove mixed archives atomically."""
from pathlib import Path
import subprocess
import sys

from huggingface_hub import HfApi, CommitOperationAdd, CommitOperationDelete
from huggingface_hub.utils import disable_progress_bars

REPO = 'dariusz-kobiela/univariate-time-series-with-missingness-and-reconstructed'
LEGACY = {'data_0_source.tar.gz', 'data_1_missing.tar.gz', 'data_2_fixed.tar.gz', 'data_images.tar.gz'}


def main():
    package = Path(__file__).resolve().parent
    subprocess.run([sys.executable, str(package/'validate_package.py')], check=True)
    disable_progress_bars()
    api = HfApi()
    info = api.dataset_info(REPO)
    payload = {p.relative_to(package/'repo').as_posix(): p for p in (package/'repo').rglob('*') if p.is_file()}
    remote = {s.rfilename for s in info.siblings}
    removals = remote - set(payload) - {'.gitattributes'}
    unexpected = {p for p in removals if p not in LEGACY and not p.startswith('releases/2026-10-10/')}
    if unexpected:
        raise ValueError(f'Unexpected remote files; review required: {unexpected}')
    operations = [CommitOperationDelete(path_in_repo=p) for p in sorted(removals)]
    operations += [CommitOperationAdd(path_in_repo=p, path_or_fileobj=str(local)) for p, local in sorted(payload.items())]
    result = api.create_commit(repo_id=REPO, repo_type='dataset', parent_commit=info.sha,
                               operations=operations,
                               commit_message='Correct source licenses; remove Pump data and mixed legacy archives')
    print(result.commit_url)
    published = api.dataset_info(REPO, revision=result.oid)
    if {s.rfilename for s in published.siblings} != set(payload) | {'.gitattributes'}:
        raise ValueError('Published file list differs from validated payload')


if __name__ == '__main__':
    main()
