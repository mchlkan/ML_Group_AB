#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
COUNTRIES_CSV="${1:-${ROOT_DIR}/datasets/Countries Data By Aadarsh Vani.csv}"
CULTURAL_MATRIX_CSV="${2:-${ROOT_DIR}/datasets/cultural_distance_matrix.csv}"
OUTPUT_ROOT="${3:-${ROOT_DIR}/datasets/processed/v1_aux}"

python3 "${ROOT_DIR}/scripts/prepare_auxiliary_datasets.py" \
  --countries-csv "${COUNTRIES_CSV}" \
  --cultural-matrix-csv "${CULTURAL_MATRIX_CSV}" \
  --output-root "${OUTPUT_ROOT}"
