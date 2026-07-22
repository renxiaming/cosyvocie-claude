export PYTHONPATH=third_party/Matcha-TTS:transformers/src:${PYTHONPATH:-}

MODEL_DIR="${MODEL_DIR:-../weight/CosyVoice2-0.5B_sft_shenhu_25_60}"
mkdir -p "$MODEL_DIR"

python3 cosyvoice/bin/export_onnx.py \
  --model_dir "$MODEL_DIR" \
  2>&1 | tee "$MODEL_DIR/export_flow_onnx.log"
