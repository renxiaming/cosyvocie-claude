#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

IMAGE="${IMAGE:-cosyvoice2-ascend:manual}"
PROJECT_IN_CONTAINER="/data/xmren/work/work/test/model/CosyVoice-claude"
ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/docker_outputs}"
CONTAINER_NAME="${CONTAINER_NAME:-cosyvoice2-ascend-run}"

mkdir -p "${OUTPUT_ROOT}/logs" "${OUTPUT_ROOT}/testout"

docker_args=(
  run --rm
  --name "${CONTAINER_NAME}"
  --network host
  --ipc host
  --shm-size 16g
  --privileged
  --ulimit memlock=-1:-1
  -e "ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}"
  -e "TOKENIZERS_PARALLELISM=${TOKENIZERS_PARALLELISM:-false}"
  -v "${OUTPUT_ROOT}/logs:${PROJECT_IN_CONTAINER}/logs"
  -v "${OUTPUT_ROOT}/testout:${PROJECT_IN_CONTAINER}/testout"
  -w "${PROJECT_IN_CONTAINER}"
)

if [ -t 0 ] && [ -t 1 ]; then
  docker_args+=(-it)
fi

for dev in davinci_manager devmm_svm hisi_hdc; do
  if [ -e "/dev/${dev}" ]; then
    docker_args+=(--device "/dev/${dev}")
  fi
done

IFS=',' read -r -a npu_ids <<< "${ASCEND_RT_VISIBLE_DEVICES}"
for npu_id in "${npu_ids[@]}"; do
  if [ -e "/dev/davinci${npu_id}" ]; then
    docker_args+=(--device "/dev/davinci${npu_id}")
  else
    echo "[WARN] /dev/davinci${npu_id} does not exist on host" >&2
  fi
done

if [ -d /usr/local/Ascend/driver ]; then
  docker_args+=(-v /usr/local/Ascend/driver:/usr/local/Ascend/driver:ro)
fi
if [ -f /usr/local/bin/npu-smi ]; then
  docker_args+=(-v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi:ro)
fi
if [ -f /usr/local/sbin/npu-smi ]; then
  docker_args+=(-v /usr/local/sbin/npu-smi:/usr/local/sbin/npu-smi:ro)
fi
if [ -f /etc/ascend_install.info ]; then
  docker_args+=(-v /etc/ascend_install.info:/etc/ascend_install.info:ro)
fi

cmd=("$@")
if [ "${#cmd[@]}" -eq 0 ]; then
  cmd=(bash)
fi

exec docker "${docker_args[@]}" "${IMAGE}" "${cmd[@]}"
