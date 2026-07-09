# conda activate cosyvoice2_copy
# ============================================================
# 单进程推理优化配置
# ============================================================
# HiFT decode OM：只替换 hift.decode，未设置时自动走原 PyTorch 路径
export COSYVOICE2_HIFT_DECODE_OM="${COSYVOICE2_HIFT_DECODE_OM:-experiments/hift_decode_om_20260706_230701/hift_decode_static_v2.om}"
export COSYVOICE2_HIFT_DECODE_GEARS="${COSYVOICE2_HIFT_DECODE_GEARS:-30,50,128,130,160}"

# 单进程默认保存音频，性能压测可设 NO_SAVE_AUDIO=1
export NO_SAVE_AUDIO="${NO_SAVE_AUDIO:-0}"

# 避免 HuggingFace tokenizer 在线程池初始化后 fork 引发死锁风险。
export TOKENIZERS_PARALLELISM="${TOKENIZERS_PARALLELISM:-false}"

# 指定使用NPU ID，默认为0
export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0}"

# 防止同一张 NPU 上重复启动单进程脚本，共享 CANN/torchair 运行时资源后容易卡在 warmup。
LOCK_FILE="/tmp/cosyvoice2_single_${ASCEND_RT_VISIBLE_DEVICES}.lock"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  echo "[ERROR] another run.sh is already running on ASCEND_RT_VISIBLE_DEVICES=${ASCEND_RT_VISIBLE_DEVICES}. lock=${LOCK_FILE}" >&2
  exit 1
fi

export PYTHONPATH=third_party/Matcha-TTS:$PYTHONPATH
export PYTHONPATH=transformers/src:$PYTHONPATH

# --- Ascend NPU 低精度加速 (减少 NPU 计算压力和显存占用) ---
export ASCEND_GEO_W8A16=1
export DYNAMIC_QUANT=1

# --- ACL 算子缓存 ---
export ACLNN_CACHE_LIMIT="${ACLNN_CACHE_LIMIT:-100000}"

# --- Ascend 多 Stream 优化 ---
export ENABLE_DYNAMIC_SHAPE_MULTI_STREAM="${ENABLE_DYNAMIC_SHAPE_MULTI_STREAM:-1}"
export TASK_QUEUE_ENABLE="${TASK_QUEUE_ENABLE:-2}"

# --- Qwen LLM hidden-only 推理 ---
# 只跳过 CosyVoice 未使用的 Qwen lm_head，保留 hidden_states，降低 LLM 首包/中间包耗时。
# 如需回退原始 Qwen forward，可显式设置 COSYVOICE2_QWEN_HIDDEN_ONLY=0。
export COSYVOICE2_QWEN_HIDDEN_ONLY="${COSYVOICE2_QWEN_HIDDEN_ONLY:-1}"
export TORCHAIR_CACHE_HOME="${TORCHAIR_CACHE_HOME:-experiments/torchair_cache_hidden_safe}"

# --- CPU 线程限制 ---
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

# --- 流式推理参数（对齐新模型 25_60） ---
export COSYVOICE2_FIRST_CHUNK_SIZE="${COSYVOICE2_FIRST_CHUNK_SIZE:-25}"
export COSYVOICE2_TOKEN_HOP_LEN="${COSYVOICE2_TOKEN_HOP_LEN:-60}"
export COSYVOICE2_FLOW_CONTEXT_TOKENS="${COSYVOICE2_FLOW_CONTEXT_TOKENS:-25}"
# Flow 档位列表（逗号分隔，必须与 OM 编译时一致！）
# mel_len = token数×2 - 6(pre_lookahead_trim)
# 首包(25+3)×2-6=50, 中间包(25+60+3)×2-6=170
export COSYVOICE2_FLOW_GEARS="${COSYVOICE2_FLOW_GEARS:-50,74,98,122,146,170,200,230,260,290,320,350,380,410}"
export COSYVOICE2_DEBUG_TIMING="${COSYVOICE2_DEBUG_TIMING:-0}"
export COSYVOICE2_FLOW_HIFT_STREAM="${COSYVOICE2_FLOW_HIFT_STREAM:-0}"

# 性能压测时可设 NO_SAVE_AUDIO=1，只消费推理输出，不拼接/保存 wav
NO_SAVE_ARG=""
if [ "${NO_SAVE_AUDIO:-0}" = "1" ]; then
  NO_SAVE_ARG="--no_save_audio"
fi

# 使能环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 规避找不到ttsfrd
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
# 规避找不到cstdint
export CPLUS_INCLUDE_PATH=/usr/local/Ascend/ascend-toolkit/8.1.RC1/toolkit/toolchain/hcc/aarch64-target-linux-gnu/include/c++/7.3.0:${CPLUS_INCLUDE_PATH}
export CPLUS_INCLUDE_PATH=/usr/local/Ascend/ascend-toolkit/8.1.RC1/toolkit/toolchain/hcc/aarch64-target-linux-gnu/include/c++/7.3.0/aarch64-target-linux-gnu:${CPLUS_INCLUDE_PATH}
export CPLUS_INCLUDE_PATH=/usr/local/Ascend/ascend-toolkit/8.1.RC1/toolkit/toolchain/hcc/aarch64-target-linux-gnu/sys-include:${CPLUS_INCLUDE_PATH}

# 默认保留 modelscope 缓存，避免每次直跑都重新构建索引。
# 如需强制冷启动，可设置 CLEAR_MODELSCOPE_CACHE=1。
if [ "${CLEAR_MODELSCOPE_CACHE:-0}" = "1" ]; then
  rm -rf ~/.cache/modelscope/
fi

python3 infer.py \
  --model_path="${MODEL_PATH:-../weight/CosyVoice2-0.5B_sft_shenhu_25_60}" \
  --stream \
  --warm_up_times="${WARM_UP_TIMES:-5}" \
  --infer_count="${INFER_COUNT:-5}" \
  --output_dir="${OUTPUT_DIR:-testout/run_single}" \
  $NO_SAVE_ARG
# python3 register_wav.py
