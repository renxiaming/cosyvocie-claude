#!/usr/bin/env bash
set -euo pipefail

IMAGE="${IMAGE:-cosyvoice2-ascend:manual}"
OUT="${OUT:-cosyvoice2-ascend-manual.tar}"

docker image inspect "${IMAGE}" >/dev/null
docker save -o "${OUT}" "${IMAGE}"
sha256sum "${OUT}" > "${OUT}.sha256"

ls -lh "${OUT}" "${OUT}.sha256"
