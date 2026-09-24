#!/usr/bin/env python3
"""Verify the anonymous code package and run its numerical regression tests."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--hashes-only', action='store_true',
                        help='check packaged files without running the tests')
    args = parser.parse_args()
    manifest_path = ROOT / 'manifest.json'
    if not manifest_path.is_file():
        parser.error('Run this command from an extracted submission ZIP, which contains manifest.json.')
    manifest = json.loads(manifest_path.read_text())
    failures = []
    for name, record in manifest['files'].items():
        path = ROOT / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != record['sha256']:
            failures.append(name)
    if failures:
        raise SystemExit('Missing or modified packaged files; extract a fresh ZIP:\n' + '\n'.join(failures))
    print(f"Verified {len(manifest['files'])} packaged file hashes.", flush=True)
    if args.hashes_only:
        return
    env = os.environ.copy()
    env.update(PYTHONPATH=str(ROOT / 'src'), OMP_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', MPLBACKEND='Agg')
    command = [sys.executable, '-m', 'pytest', 'tests', '-q']
    print('Running numerical and reporting regression tests; no image data is needed.', flush=True)
    process = subprocess.run(command, cwd=ROOT, env=env, text=True,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (ROOT / 'validation.log').write_text(process.stdout)
    print(process.stdout, end='', flush=True)
    if process.returncode:
        raise SystemExit(process.returncode)
    print('Code-package validation passed. Full training reproduction is described in REPRODUCING.md.', flush=True)


if __name__ == '__main__':
    main()
