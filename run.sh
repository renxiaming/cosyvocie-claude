# 指定使用NPU ID，默认为0
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTHONPATH=third_party/Matcha-TTS:$PYTHONPATH
export PYTHONPATH=transformers/src:$PYTHONPATH

# --- Ascend NPU 低精度加速 ---
export ASCEND_GEO_W8A16=1
export DYNAMIC_QUANT=1

# --- ACL 算子缓存 ---
export ACLNN_CACHE_LIMIT="${ACLNN_CACHE_LIMIT:-100000}"

# --- Ascend 多 Stream 优化 ---
export ENABLE_DYNAMIC_SHAPE_MULTI_STREAM=1
export TASK_QUEUE_ENABLE=2

# --- CPU 线程限制 ---
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-16}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-8}"

# --- 流式推理参数（对齐新模型 25_60） ---
export COSYVOICE2_FIRST_CHUNK_SIZE="${COSYVOICE2_FIRST_CHUNK_SIZE:-25}"
export COSYVOICE2_TOKEN_HOP_LEN="${COSYVOICE2_TOKEN_HOP_LEN:-60}"
export COSYVOICE2_FLOW_CONTEXT_TOKENS="${COSYVOICE2_FLOW_CONTEXT_TOKENS:-25}"
# mel_len = token×2-6, 首包50/中间包170
export COSYVOICE2_FLOW_GEARS="${COSYVOICE2_FLOW_GEARS:-50,74,98,122,146,170,200,230,260,290,320,350,380,410}"

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

python3 infer.py --model_path="${MODEL_PATH:-../weight/CosyVoice2-0.5B_sft_shenhu_25_60}" --stream
# python3 register_wav.py
