#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATASET_VERSION="${DATASET_VERSION:-v1}"
INPUT_CSV="${1:-${ROOT_DIR}/datasets/merged_data.csv}"
OUTPUT_ROOT="${2:-${ROOT_DIR}/datasets/processed/${DATASET_VERSION}}"
DB_FILE="${3:-${OUTPUT_ROOT}/processing.sqlite}"
MANIFEST_PATH="${MANIFEST_PATH:-${ROOT_DIR}/datasets/manifest.${DATASET_VERSION}.json}"
SCHEMA_VERSION="${SCHEMA_VERSION:-${DATASET_VERSION}}"
CHUNKSIZE="${CHUNKSIZE:-150000}"
JOIN_AUX="${JOIN_AUX:-0}"
AUX_ROOT="${AUX_ROOT:-${ROOT_DIR}/datasets/processed/${DATASET_VERSION}_aux}"

if ! command -v python3 >/dev/null 2>&1; then
  echo "python3 is required but not installed."
  exit 1
fi

if [[ ! -f "${INPUT_CSV}" ]]; then
  echo "Input CSV not found: ${INPUT_CSV}"
  exit 1
fi

mkdir -p "${OUTPUT_ROOT}/full" "${OUTPUT_ROOT}/slim"

PROCESS_CMD=(
  python3 "${ROOT_DIR}/scripts/process_first_dataset_pandas.py"
  --input-csv "${INPUT_CSV}"
  --output-root "${OUTPUT_ROOT}"
  --db-file "${DB_FILE}"
  --chunksize "${CHUNKSIZE}"
)

if [[ "${JOIN_AUX}" == "1" ]]; then
  PROCESS_CMD+=(--join-aux --aux-root "${AUX_ROOT}")
fi

"${PROCESS_CMD[@]}"

python3 "${ROOT_DIR}/scripts/generate_manifest.py" \
  --source-csv "${INPUT_CSV}" \
  --output-root "${OUTPUT_ROOT}" \
  --manifest-path "${MANIFEST_PATH}" \
  --schema-version "${SCHEMA_VERSION}" \
  --dataset-version "${DATASET_VERSION}"

echo "Dataset processing finished."
echo "Processed data root: ${OUTPUT_ROOT}"
echo "Manifest: ${MANIFEST_PATH}"
