#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PROJECT_PARENT="$(cd "${PROJECT_ROOT}/.." && pwd)"

IMAGE="${IMAGE:-cosyvoice2-ascend:manual}"
BASE_IMAGE="${BASE_IMAGE:-swr.cn-southwest-2.myhuaweicloud.com/aslp/cosyvoice2:62g}"
MODEL_DIR="${MODEL_DIR:-${PROJECT_PARENT}/weight/CosyVoice2-0.5B_sft_shenhu_25_60}"
STAGE_DIR="${STAGE_DIR:-${PROJECT_PARENT}/.cosyvoice2_ascend_build_context}"

if ! command -v docker >/dev/null 2>&1; then
  echo "[ERROR] docker is required" >&2
  exit 1
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "[ERROR] rsync is required to create a clean build context" >&2
  exit 1
fi

if [ ! -d "${MODEL_DIR}" ]; then
  echo "[ERROR] model directory not found: ${MODEL_DIR}" >&2
  exit 1
fi

case "${STAGE_DIR}" in
  ""|"/"|"/tmp"|"/data"|"/data/"*)
    if [ "${STAGE_DIR}" = "/data" ] || [ "${STAGE_DIR}" = "/data/" ]; then
      echo "[ERROR] unsafe STAGE_DIR: ${STAGE_DIR}" >&2
      exit 1
    fi
    ;;
esac

if [ -e "${STAGE_DIR}" ] && [ ! -f "${STAGE_DIR}/.cosyvoice2_ascend_build_context" ]; then
  echo "[ERROR] refusing to delete non-build-context directory: ${STAGE_DIR}" >&2
  exit 1
fi

rm -rf "${STAGE_DIR}"
mkdir -p "${STAGE_DIR}/CosyVoice-claude" "${STAGE_DIR}/weight/CosyVoice2-0.5B_sft_shenhu_25_60"
touch "${STAGE_DIR}/.cosyvoice2_ascend_build_context"

rsync -a --delete \
  --exclude='.git' \
  --exclude='.pytest_cache' \
  --exclude='__pycache__' \
  --exclude='*.pyc' \
  --exclude='logs' \
  --exclude='testout' \
  --exclude='docker_outputs' \
  "${PROJECT_ROOT}/" "${STAGE_DIR}/CosyVoice-claude/"

rsync -a --delete "${MODEL_DIR}/" "${STAGE_DIR}/weight/CosyVoice2-0.5B_sft_shenhu_25_60/"
cp "${PROJECT_ROOT}/docker/Dockerfile.ascend" "${STAGE_DIR}/Dockerfile"

echo "[INFO] build context: ${STAGE_DIR}"
echo "[INFO] image: ${IMAGE}"
echo "[INFO] base image: ${BASE_IMAGE}"
du -sh "${STAGE_DIR}" "${STAGE_DIR}/weight/CosyVoice2-0.5B_sft_shenhu_25_60"

docker build \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  -t "${IMAGE}" \
  "${STAGE_DIR}"

docker image inspect "${IMAGE}" --format '[INFO] built {{.RepoTags}} size={{.Size}}'
