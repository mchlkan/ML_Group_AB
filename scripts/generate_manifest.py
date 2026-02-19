#!/usr/bin/env python3
import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def list_files_with_hashes(root: Path):
    items = []
    for p in sorted(root.rglob('*')):
        if p.is_file():
            items.append(
                {
                    'path': str(p),
                    'size_bytes': p.stat().st_size,
                    'sha256': sha256_file(p),
                }
            )
    return items


def read_row_accounting(path: Path):
    metrics = {}
    if not path.exists():
        return metrics
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            metrics[row['metric']] = int(row['value'])
    return metrics


def read_profile_overview(path: Path):
    if not path.exists():
        return {}
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        row = next(reader, None)
        return row or {}


def main():
    parser = argparse.ArgumentParser(description='Generate dataset manifest for processed artifacts.')
    parser.add_argument('--source-csv', required=True)
    parser.add_argument('--output-root', required=True)
    parser.add_argument('--manifest-path', required=True)
    parser.add_argument('--schema-version', required=True)
    parser.add_argument('--dataset-version', default='v1')
    args = parser.parse_args()

    source_csv = Path(args.source_csv)
    output_root = Path(args.output_root)
    manifest_path = Path(args.manifest_path)

    full_root = output_root / 'full'
    slim_root = output_root / 'slim'
    quarantine_file = output_root / 'quarantine.parquet'

    row_accounting = read_row_accounting(output_root / 'row_accounting.csv')
    profile_overview = read_profile_overview(output_root / 'profile_overview.csv')

    manifest = {
        'dataset_version': args.dataset_version,
        'schema_version': args.schema_version,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'source': {
            'path': str(source_csv),
            'size_bytes': source_csv.stat().st_size,
            'sha256': sha256_file(source_csv),
        },
        'profile_overview': profile_overview,
        'row_accounting': row_accounting,
        'artifacts': {
            'full': {
                'root': str(full_root),
                'files': list_files_with_hashes(full_root) if full_root.exists() else [],
            },
            'slim': {
                'root': str(slim_root),
                'files': list_files_with_hashes(slim_root) if slim_root.exists() else [],
            },
            'quarantine': {
                'path': str(quarantine_file),
                'exists': quarantine_file.exists(),
                'size_bytes': quarantine_file.stat().st_size if quarantine_file.exists() else 0,
                'sha256': sha256_file(quarantine_file) if quarantine_file.exists() else None,
            },
            'reports': {
                'source_schema_csv': str(output_root / 'source_schema.csv'),
                'profile_overview_csv': str(output_root / 'profile_overview.csv'),
                'profile_null_rates_csv': str(output_root / 'profile_null_rates.csv'),
                'row_accounting_csv': str(output_root / 'row_accounting.csv'),
            },
        },
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open('w', encoding='utf-8') as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
        f.write('\n')


if __name__ == '__main__':
    main()
