export PYTHONPATH=third_party/Matcha-TTS:transformers/src:${PYTHONPATH:-}

MODEL_DIR="/home/ma-user/work/test/model/weight/CosyVoice2-0.5B_sft_shenhu_25_60_vnpu212"

python3 cosyvoice/bin/export_onnx.py \
  --model_dir "$MODEL_DIR" \
  2>&1 | tee "$MODEL_DIR/export_flow_onnx.log"