# ============================================================
# 多进程并发推理优化配置
# ============================================================

# NPU 设备绑定
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTHONPATH=third_party/Matcha-TTS:$PYTHONPATH
export PYTHONPATH=transformers/src:$PYTHONPATH

# --- Ascend NPU 低精度加速 (减少 NPU 计算压力和显存占用) ---
export ASCEND_GEO_W8A16=1
export DYNAMIC_QUANT=1

# --- ACL 算子缓存 (避免多进程重复编译算子) ---
export ACLNN_CACHE_LIMIT="${ACLNN_CACHE_LIMIT:-100000}"

# --- Ascend 多 Stream 优化 (允许不同进程的算子提交到不同 Stream 队列) ---
export ENABLE_DYNAMIC_SHAPE_MULTI_STREAM=1
export TASK_QUEUE_ENABLE=2

# --- 限制每个进程的 CPU 线程数 (192 核 / 10 进程 ≈ 19 核/进程) ---
# 设为 16 留出余量给系统和其他组件
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

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

# 使能环境变量
source /usr/local/Ascend/ascend-toolkit/set_env.sh
# 规避找不到ttsfrd
export LD_LIBRARY_PATH=/usr/local/lib:$LD_LIBRARY_PATH
# 规避找不到cstdint
export CPLUS_INCLUDE_PATH=/usr/local/Ascend/ascend-toolkit/8.1.RC1/toolkit/toolchain/hcc/aarch64-target-linux-gnu/include/c++/7.3.0:${CPLUS_INCLUDE_PATH}
export CPLUS_INCLUDE_PATH=/usr/local/Ascend/ascend-toolkit/8.1.RC1/toolkit/toolchain/hcc/aarch64-target-linux-gnu/include/c++/7.3.0/aarch64-target-linux-gnu:${CPLUS_INCLUDE_PATH}
export CPLUS_INCLUDE_PATH=/usr/local/Ascend/ascend-toolkit/8.1.RC1/toolkit/toolchain/hcc/aarch64-target-linux-gnu/sys-include:${CPLUS_INCLUDE_PATH}

# 清理modelscope缓存
rm -rf ~/.cache/modelscope/

python3 infer_manual_concurrent.py \
  --model_path="${MODEL_PATH:-../weight/CosyVoice2-0.5B_sft_shenhu_25_60}" \
  --stream \
  --concurrency="${CONCURRENCY:-1}" \
  --infer_count=5 \
  --warm_up_times=5 \
  --log_dir="${LOG_DIR:-logs/manual}" \
  --enable_cpu_affinity
