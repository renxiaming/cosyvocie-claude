#!/usr/bin/bash
set -euo pipefail

ROOT="/data/xmren/work/work/test/model/CosyVoice-claude"
cd "$ROOT"

source /usr/local/Ascend/ascend-toolkit/set_env.sh
export LD_LIBRARY_PATH=/usr/local/lib:${LD_LIBRARY_PATH:-}
export ASCEND_GEO_W8A16=1
export DYNAMIC_QUANT=1
export ACLNN_CACHE_LIMIT=100000
export ASCEND_RT_VISIBLE_DEVICES=0
export PYTHONPATH=third_party/Matcha-TTS:transformers/src:$PYTHONPATH
export COSYVOICE2_HIFT_DECODE_OM="${COSYVOICE2_HIFT_DECODE_OM:-experiments/hift_decode_om_20260706_230701/hift_decode_static_v2.om}"
export NO_SAVE_AUDIO=1
export COSYVOICE2_NO_CPU_OUTPUT=1

RUN_ID="$(date +%Y%m%d_%H%M%S)"
LOG_DIR="logs/hbm_monitor_npu0/run_${RUN_ID}"
mkdir -p "$LOG_DIR"

hbm_npu0() {
  npu-smi info 2>/dev/null | awk '/^\| 0     910B1/{getline; match($0, /([0-9]+)[[:space:]]+\/ 65536/, a); print a[1]}'
}

snapshot() {
  local tag="$1"
  local ts
  ts="$(date '+%H:%M:%S')"
  local hbm proc_info
  hbm="$(hbm_npu0)"
  proc_info="$(npu-smi info 2>/dev/null | awk '/^\| 0     910B1/,/^\+\=/{print}')"
  echo "[$ts] tag=$tag hbm_mb=$hbm" | tee -a "$LOG_DIR/timeline.log"
  {
    echo "=== $tag @ $ts hbm=${hbm}MB ==="
    echo "$proc_info"
  } >> "$LOG_DIR/npu0_detail.log"
}

snapshot "baseline_before_run"

MONITOR_PID=""
monitor_loop() {
  while true; do
    echo "$(date '+%H:%M:%S') hbm=$(hbm_npu0)" >> "$LOG_DIR/hbm_poll.log"
    sleep 2
  done
}
monitor_loop &
MONITOR_PID=$!

INFER_LOG="$LOG_DIR/infer.log"
echo "[INFO] starting infer, log=$INFER_LOG" | tee -a "$LOG_DIR/timeline.log"

set +e
echo "是的，您现在还有大概1个G的流量。" > "$LOG_DIR/one_line.txt"
/data/zhbai/mimo/miniconda3/envs/voxcpm/bin/python3 infer.py \
  --model_path="../weight/CosyVoice2-0.5B_sft_shenhu_25_60" \
  --stream \
  --infer_count=1 \
  --warm_up_times=1 \
  --no_save_audio \
  --text_file="$LOG_DIR/one_line.txt" \
  > "$INFER_LOG" 2>&1
INFER_RC=$?
set -e

kill "$MONITOR_PID" 2>/dev/null || true
wait "$MONITOR_PID" 2>/dev/null || true

sleep 3
snapshot "after_process_exit"

echo "$INFER_RC" > "$LOG_DIR/infer.rc"

python3 - <<'PY' "$LOG_DIR" "$INFER_RC"
import re, sys
from pathlib import Path
log_dir = Path(sys.argv[1])
rc = sys.argv[2]
infer = (log_dir / "infer.log").read_text(errors="replace").splitlines()
poll = []
for line in (log_dir / "hbm_poll.log").read_text(errors="replace").splitlines():
    m = re.match(r"(\d\d:\d\d:\d\d) hbm=(\d+)", line.strip())
    if m:
        poll.append((m.group(1), int(m.group(2))))
timeline = (log_dir / "timeline.log").read_text(errors="replace")
base_m = re.search(r"baseline_before_run hbm_mb=(\d+)", timeline)
exit_m = re.search(r"after_process_exit hbm_mb=(\d+)", timeline)
base = int(base_m.group(1)) if base_m else (poll[0][1] if poll else None)
exit_h = int(exit_m.group(1)) if exit_m else (poll[-1][1] if poll else None)
lines = []
lines.append("=== SUMMARY ===")
lines.append(f"infer_exit_code={rc}")
lines.append(f"baseline_hbm={base}MB after_exit_hbm={exit_h}MB retained_delta={exit_h-base if base and exit_h else 'NA'}MB")
lines.append(f"poll_range={min(x[1] for x in poll)}-{max(x[1] for x in poll)}MB" if poll else "poll_range=NA")
keys = ["acl init success", "open device 0 success", "load model", "flow_linux", "flow_static", "speech_linux", "hift decode om loaded", "warm up end", "unload model success"]
for key in keys:
    for ln in infer:
        if key in ln:
            lines.append(f"LOG: {ln.strip()[:140]}")
            break
(log_dir / "summary.txt").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY

echo "[DONE] artifacts in $LOG_DIR"
cat "$LOG_DIR/summary.txt"
