#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# CosyVoice chunk streaming web probe
# ============================================================

export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export PYTHONPATH=third_party/Matcha-TTS:${PYTHONPATH:-}
export PYTHONPATH=transformers/src:${PYTHONPATH:-}

export ASCEND_GEO_W8A16="${ASCEND_GEO_W8A16:-1}"
export DYNAMIC_QUANT="${DYNAMIC_QUANT:-1}"

export ACLNN_CACHE_LIMIT="${ACLNN_CACHE_LIMIT:-100000}"
export ENABLE_DYNAMIC_SHAPE_MULTI_STREAM="${ENABLE_DYNAMIC_SHAPE_MULTI_STREAM:-1}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-2}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

export COSYVOICE2_FIRST_CHUNK_SIZE="${COSYVOICE2_FIRST_CHUNK_SIZE:-25}"
export COSYVOICE2_TOKEN_HOP_LEN="${COSYVOICE2_TOKEN_HOP_LEN:-60}"
export COSYVOICE2_FLOW_CONTEXT_TOKENS="${COSYVOICE2_FLOW_CONTEXT_TOKENS:-25}"
export COSYVOICE2_FLOW_GEARS="${COSYVOICE2_FLOW_GEARS:-50,74,98,122,146,170,200,230,260,290,320,350,380,410}"
export COSYVOICE2_DEBUG_TIMING="${COSYVOICE2_DEBUG_TIMING:-0}"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH:-}
export CPLUS_INCLUDE_PATH=/usr/local/Ascend/ascend-toolkit/8.1.RC1/toolkit/toolchain/hcc/aarch64-target-linux-gnu/include/c++/7.3.0:${CPLUS_INCLUDE_PATH:-}
export CPLUS_INCLUDE_PATH=/usr/local/Ascend/ascend-toolkit/8.1.RC1/toolkit/toolchain/hcc/aarch64-target-linux-gnu/include/c++/7.3.0/aarch64-target-linux-gnu:${CPLUS_INCLUDE_PATH:-}
export CPLUS_INCLUDE_PATH=/usr/local/Ascend/ascend-toolkit/8.1.RC1/toolkit/toolchain/hcc/aarch64-target-linux-gnu/sys-include:${CPLUS_INCLUDE_PATH:-}

rm -rf ~/.cache/modelscope/

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-50080}"
MODEL_PATH="${MODEL_PATH:-../weight/CosyVoice2-0.5B_sft_shenhu_25_60}"
DEFAULT_SPK="${DEFAULT_SPK:-03729}"
WARM_UP_TIMES="${WARM_UP_TIMES:-2}"

echo "[INFO] Starting chunk stream web probe on ${HOST}:${PORT}"
echo "[INFO] If this is inside Docker, publish the same port, for example: -p ${PORT}:${PORT}"

python3 stream_chunk_web.py \
  --model_path="${MODEL_PATH}" \
  --host="${HOST}" \
  --port="${PORT}" \
  --default_spk="${DEFAULT_SPK}" \
  --warm_up_times="${WARM_UP_TIMES}"
