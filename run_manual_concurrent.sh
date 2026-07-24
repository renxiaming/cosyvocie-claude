# ============================================================
# 多进程并发推理优化配置
# ============================================================
# HiFT decode OM：只替换 hift.decode，未设置时自动走原 PyTorch 路径
export COSYVOICE2_HIFT_DECODE_OM="${COSYVOICE2_HIFT_DECODE_OM:-experiments/hift_decode_om_20260706_230701/hift_decode_static_v2.om}"
export COSYVOICE2_HIFT_DECODE_GEARS="${COSYVOICE2_HIFT_DECODE_GEARS:-30,50,128,130,160}"

# 严格 10 进程正式推理同步：所有进程 warmup 完成后统一开始正式推理
export SYNC_START="${SYNC_START:-1}"
export SYNC_START_TIMEOUT="${SYNC_START_TIMEOUT:-1800}"

# 性能压测默认不保存音频，避免 torchaudio/save 影响 RTF；需要保存时 NO_SAVE_AUDIO=0
export NO_SAVE_AUDIO="${NO_SAVE_AUDIO:-1}"
export LOG_DIR="${LOG_DIR:-logs/manual_hift_om_v2_sync_run}"
export TEXT_FILE="${TEXT_FILE:-data/manual_transcript_20260720.txt}"

# NPU 设备绑定
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"
export PYTHONPATH=third_party/Matcha-TTS:$PYTHONPATH
export PYTHONPATH=transformers/src:$PYTHONPATH

# --- Ascend NPU 低精度加速 (减少 NPU 计算压力和显存占用) ---
export ASCEND_GEO_W8A16=1
export DYNAMIC_QUANT=1

# --- ACL 算子缓存 (避免多进程重复编译算子) ---
export ACLNN_CACHE_LIMIT="${ACLNN_CACHE_LIMIT:-100000}"

# --- Ascend 多 Stream 优化 (允许不同进程的算子提交到不同 Stream 队列) ---
export ENABLE_DYNAMIC_SHAPE_MULTI_STREAM="${ENABLE_DYNAMIC_SHAPE_MULTI_STREAM:-1}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-2}"

# --- Qwen LLM hidden-only 推理 ---
# 只跳过 CosyVoice 未使用的 Qwen lm_head，独立使用 safe hidden-only 编译缓存。
export COSYVOICE2_QWEN_HIDDEN_ONLY="${COSYVOICE2_QWEN_HIDDEN_ONLY:-1}"
export TORCHAIR_CACHE_HOME="${TORCHAIR_CACHE_HOME:-experiments/torchair_cache_hidden_safe}"
# --- Qwen LLM decode 轻量化 ---
# fast_topk 避免原始 RAS/top-p 全量排序；device-token decode 减少逐 token .item() Host 同步。
export COSYVOICE2_SAMPLING_MODE="${COSYVOICE2_SAMPLING_MODE:-fast_topk}"
export COSYVOICE2_FAST_TOPK_K="${COSYVOICE2_FAST_TOPK_K:-25}"
export COSYVOICE2_DEVICE_TOKEN_DECODE="${COSYVOICE2_DEVICE_TOKEN_DECODE:-1}"
# 复用 decode 阶段的 position/KV 元数据，避免每个 token 重建小张量。
export COSYVOICE2_CACHE_DECODE_METADATA="${COSYVOICE2_CACHE_DECODE_METADATA:-1}"
# 复用 speech embedding 输出，避免逐 token 的无必要 clone。
export COSYVOICE2_REUSE_EMBEDDING_OUTPUT="${COSYVOICE2_REUSE_EMBEDDING_OUTPUT:-1}"
export COSYVOICE2_MARK_STATIC_INPUTS="${COSYVOICE2_MARK_STATIC_INPUTS:-1}"

# --- 限制每个进程的 CPU 线程数，减少 10 进程下 CPU/BLAS 竞争 ---
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"

# --- 流式推理性能调优 ---
# 首包大小（默认20，新模型用25）
export COSYVOICE2_FIRST_CHUNK_SIZE="${COSYVOICE2_FIRST_CHUNK_SIZE:-25}"
# 中间包推进步长（默认50，新模型用60）
export COSYVOICE2_TOKEN_HOP_LEN="${COSYVOICE2_TOKEN_HOP_LEN:-60}"
# Flow 上下文长度（默认25）
export COSYVOICE2_FLOW_CONTEXT_TOKENS="${COSYVOICE2_FLOW_CONTEXT_TOKENS:-25}"
# Flow 档位列表（逗号分隔，必须与 OM 编译时一致！）
# mel_len = token数×2 - 6(pre_lookahead_trim)
# 首包(25+3)×2-6=50, 中间包(25+60+3)×2-6=170
export COSYVOICE2_FLOW_GEARS="${COSYVOICE2_FLOW_GEARS:-50,74,98,122,146,170,200,230,260,290,320,350,380,410}"
# 生产环境关闭 debug 计时（同步屏障/打印均有额外开销，调试时设为1）
export COSYVOICE2_DEBUG_TIMING="${COSYVOICE2_DEBUG_TIMING:-0}"
# 跳过推理期 mask 全 false 防御检查，避免每次 mask 后 .item() 强制同步；如需排查 mask 问题设为0。
export COSYVOICE2_SKIP_MASK_SANITY="${COSYVOICE2_SKIP_MASK_SANITY:-1}"
# 关闭额外 Flow/HiFT NPU stream，严格 10 进程下 p95 更稳
export COSYVOICE2_FLOW_HIFT_STREAM="${COSYVOICE2_FLOW_HIFT_STREAM:-0}"
# CPU/NPU 拓扑亲和。NPU0 在当前机器上的就近 CPU 是 144-167；
# 为空时仍按全机器 CPU 均分。
if [ -z "${CPU_AFFINITY_CPUS+x}" ]; then
  NPU_ID="${ASCEND_RT_VISIBLE_DEVICES%%,*}"
  AUTO_CPU_AFFINITY=""
  if command -v npu-smi >/dev/null 2>&1; then
    AUTO_CPU_AFFINITY="$(npu-smi info -t topo -i 0 2>/dev/null | awk -v npu="NPU${NPU_ID}" '$1 == npu && $NF ~ /^[0-9]+-[0-9]+$/ {print $NF; exit}')"
  fi
  case "${AUTO_CPU_AFFINITY}" in
    ''|*[!0-9,-]*)
      # Fallback for this host's 8-card topology if npu-smi is unavailable.
      case "${NPU_ID}" in
        0|2) CPU_AFFINITY_CPUS="144-167" ;;
        1|3) CPU_AFFINITY_CPUS="0-23" ;;
        4|6) CPU_AFFINITY_CPUS="96-119" ;;
        5|7) CPU_AFFINITY_CPUS="48-71" ;;
        *)
          CPU_AFFINITY_CPUS=""
          echo "[WARN] unknown NPU ${NPU_ID}; set CPU_AFFINITY_CPUS explicitly" >&2
          ;;
      esac
      ;;
    *) CPU_AFFINITY_CPUS="${AUTO_CPU_AFFINITY}" ;;
  esac
fi
export CPU_AFFINITY_CPUS
export CPU_AFFINITY_SHARE="${CPU_AFFINITY_SHARE:-1}"

# 性能压测时可设 NO_SAVE_AUDIO=1，只消费推理输出，不拼接/保存 wav
NO_SAVE_ARG=""
if [ "${NO_SAVE_AUDIO:-0}" = "1" ]; then
  NO_SAVE_ARG="--no_save_audio"
fi

SYNC_START_ARG=""
if [ "${SYNC_START:-1}" = "1" ]; then
  SYNC_START_ARG="--sync_start --sync_start_timeout ${SYNC_START_TIMEOUT:-1200}"
fi

CPU_AFFINITY_ARG=()
if [ -n "${CPU_AFFINITY_CPUS:-}" ]; then
  CPU_AFFINITY_ARG=(--cpu_affinity_cpus "${CPU_AFFINITY_CPUS}")
fi
CPU_AFFINITY_SHARE_ARG=""
if [ "${CPU_AFFINITY_SHARE:-0}" = "1" ]; then
  CPU_AFFINITY_SHARE_ARG="--cpu_affinity_share"
fi

# 使能环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 规避找不到ttsfrd
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
# 规避找不到cstdint
export CPLUS_INCLUDE_PATH=/usr/local/Ascend/ascend-toolkit/8.1.RC1/toolkit/toolchain/hcc/aarch64-target-linux-gnu/include/c++/7.3.0:${CPLUS_INCLUDE_PATH}
export CPLUS_INCLUDE_PATH=/usr/local/Ascend/ascend-toolkit/8.1.RC1/toolkit/toolchain/hcc/aarch64-target-linux-gnu/include/c++/7.3.0/aarch64-target-linux-gnu:${CPLUS_INCLUDE_PATH}
export CPLUS_INCLUDE_PATH=/usr/local/Ascend/ascend-toolkit/8.1.RC1/toolkit/toolchain/hcc/aarch64-target-linux-gnu/sys-include:${CPLUS_INCLUDE_PATH}

# 默认保留 modelscope 缓存，避免每次压测都重新构建前端/FST 缓存。
# 如需强制冷启动，可设置 CLEAR_MODELSCOPE_CACHE=1。
if [ "${CLEAR_MODELSCOPE_CACHE:-0}" = "1" ]; then
  rm -rf ~/.cache/modelscope/
fi

python3 infer_manual_concurrent.py \
  --model_path="${MODEL_PATH:-../weight/CosyVoice2-0.5B_sft_shenhu_25_60}" \
  --stream \
  --concurrency="${CONCURRENCY:-10}" \
  --infer_count="${INFER_COUNT:-1}" \
  --warm_up_times="${WARM_UP_TIMES:-5}" \
  --text_file="${TEXT_FILE}" \
  --warmup_full \
  --log_dir="${LOG_DIR:-logs/manual}" \
  --enable_cpu_affinity \
  "${CPU_AFFINITY_ARG[@]}" \
  $CPU_AFFINITY_SHARE_ARG \
  $SYNC_START_ARG \
  $NO_SAVE_ARG
