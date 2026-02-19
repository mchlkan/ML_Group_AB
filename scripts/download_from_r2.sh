#!/usr/bin/env bash
set -euo pipefail

# Required config values:
#   R2_ENDPOINT   e.g. https://<account-id>.r2.cloudflarestorage.com
#   R2_BUCKET     e.g. ml-group-ab-datasets
#   AWS_ACCESS_KEY_ID
#   AWS_SECRET_ACCESS_KEY
#
# Optional values:
#   DATASET_VERSION (default: v1)
#   DOWNLOAD_ROOT (default: ./datasets/v1)
#   R2_CONFIG_FILE (default: ./scripts/r2.env)
#   SKIP_R2_PREFLIGHT (default: 0)

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
R2_CONFIG_FILE="${R2_CONFIG_FILE:-${ROOT_DIR}/scripts/r2.env}"

if [[ -f "${R2_CONFIG_FILE}" ]]; then
  set -a
  # shellcheck disable=SC1090
  source "${R2_CONFIG_FILE}"
  set +a
fi

R2_ENDPOINT="${R2_ENDPOINT:-}"
R2_BUCKET="${R2_BUCKET:-}"
DATASET_VERSION="${DATASET_VERSION:-v1}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-${ROOT_DIR}/datasets/${DATASET_VERSION}}"

if ! command -v aws >/dev/null 2>&1; then
  echo "aws CLI is required. Install with: brew install awscli"
  exit 1
fi

if [[ -z "${R2_ENDPOINT}" || -z "${R2_BUCKET}" ]]; then
  echo "Missing required env vars."
  echo "Set R2_ENDPOINT and R2_BUCKET."
  exit 1
fi

if [[ -z "${AWS_ACCESS_KEY_ID:-}" || -z "${AWS_SECRET_ACCESS_KEY:-}" ]]; then
  echo "Missing AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY."
  exit 1
fi

AWS_REGION="${AWS_REGION:-${AWS_DEFAULT_REGION:-auto}}"
AWS_DEFAULT_REGION="${AWS_DEFAULT_REGION:-${AWS_REGION}}"

export AWS_REGION AWS_DEFAULT_REGION AWS_EC2_METADATA_DISABLED=true
R2_PREFIX="s3://${R2_BUCKET}/dataset/${DATASET_VERSION}"

mkdir -p "${DOWNLOAD_ROOT}"

echo "Using config file: ${R2_CONFIG_FILE}"
echo "Dataset version: ${DATASET_VERSION}"
echo "Download root: ${DOWNLOAD_ROOT}"
echo "R2 bucket: ${R2_BUCKET}"

if [[ "${SKIP_R2_PREFLIGHT:-0}" != "1" ]]; then
  echo "Running R2 preflight check ..."
  if ! aws --endpoint-url "${R2_ENDPOINT}" s3api head-bucket --bucket "${R2_BUCKET}" >/dev/null 2>&1; then
    echo "R2 preflight failed. Verify endpoint, bucket, and credentials in ${R2_CONFIG_FILE}."
    echo "If your permissions do not allow head-bucket, rerun with SKIP_R2_PREFLIGHT=1."
    exit 1
  fi
fi

echo "Downloading ${R2_PREFIX}/ to ${DOWNLOAD_ROOT} ..."
aws --endpoint-url "${R2_ENDPOINT}" s3 sync \
  "${R2_PREFIX}/" \
  "${DOWNLOAD_ROOT}/"

echo "Download complete."
echo "Local path: ${DOWNLOAD_ROOT}/"
