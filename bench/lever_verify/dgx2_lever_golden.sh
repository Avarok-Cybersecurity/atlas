#!/bin/bash
# Golden gate run: ST-995 (perf) + ST-996 (BFCL) --mode both, IoU-safe lever config.
# See docs/lever-folding/GOLDEN_LEVER_CONFIG.md for the full knob rationale + A/B findings.
# Usage: DGX2=spark-43fa MODEL=centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf bash dgx2_lever_golden.sh
set -eu
DGX2="${DGX2:-spark-43fa}"
HOST="${DGX2_HOST:-10.10.10.2}"
PORT="${PORT:-8888}"
MODEL="${MODEL:-centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf}"
IMG="${IMG:-atlas-gb10:midchunk-adapk-ldmab}"
CACHE="${CACHE:-/workspace/.cache/huggingface}"
NUM_DRAFTS="${NUM_DRAFTS:-2}"   # K=2 is the sweet spot: 88.75% BFCL @ 5.74 s/sample (K=1=6.72, K=3=87.5%@6.29)
FFN_LEVER="${FFN_LEVER:-ATLAS_BF16_TC_PREFILL=1}"   # IoU-safe default; use ATLAS_FFN_NVFP4_MMQ=1 for max-perf (lossy)
MMQ_DISABLE="${MMQ_DISABLE:-1}"                      # 1=disable MMQ (needed for BF16_TC to engage — MMQ is default-on); 0 for max-perf (MMQ)
GRAMMAR="${GRAMMAR:-true}"                            # true=OFF (IoU-safe); false=ON (max-perf, IoU-drop suspect)
KV="${KV:-bf16}"
ROOT=/workspace/endpoints-fresh
EXDIR=examples/11_Edge_Agentic_Example
STAMP=$(date +%Y%m%d_%H%M%S)
DST="$ROOT/$EXDIR/golden_run_${STAMP}.yaml"
RPT="results/golden_run_${STAMP}"
LOG=/workspace/e2e_golden_${STAMP}.log
echo "=== golden gate run  model=$MODEL  K=$NUM_DRAFTS  FFN=$FFN_LEVER  grammar_disable=$GRAMMAR  KV=$KV ==="
echo "=== ship image $IMG -> $DGX2 ==="
sudo docker save "$IMG" 2>/dev/null | sudo -u claude ssh -o ConnectTimeout=10 -o BatchMode=yes "$DGX2" 'sudo docker load' 2>&1 | tail -2
sudo -u claude ssh -o ConnectTimeout=8 -o BatchMode=yes "$DGX2" "sudo docker rm -f atlas-golden 2>/dev/null; sleep 4; sudo docker run -d --name atlas-golden --network host --gpus all --ipc=host \
  -e $FFN_LEVER -e ATLAS_NO_FFN_NVFP4_MMQ=$MMQ_DISABLE -e ATLAS_SSM_TAIL_MIDCHUNK=1 -e ATLAS_MTP_DRAFTER_PREFILL=1 \
  -v $CACHE:/root/.cache/huggingface:ro \
  $IMG serve $MODEL --host 0.0.0.0 --port $PORT --model-name $MODEL \
    --max-seq-len 32768 --max-batch-size 1 --kv-cache-dtype $KV --gpu-memory-utilization 0.70 \
    --enable-prefix-caching --ssm-cache-slots 128 --ssm-checkpoint-interval 32 \
    --speculative --num-drafts $NUM_DRAFTS --mtp-quantization bf16 \
    --tool-call-parser qwen3_xml --disable-tool-grammar $GRAMMAR --disable-thinking" 2>&1 | tail -1
for i in $(seq 1 72); do curl -sf -m4 "http://$HOST:$PORT/v1/models" 2>/dev/null | grep -q Qwen && break; sleep 5; done
echo "=== serve ready on $HOST:$PORT — launching --mode both e2e ==="
python3 - "$ROOT/$EXDIR/online_edge_full_run.yaml" "$DST" "$MODEL" "$HOST" "$PORT" "$RPT" <<'PY'
import re, sys
src, dst, served, host, port, rpt = sys.argv[1:7]
s = open(src).read()
s = re.sub(r'(\n  name: )"Qwen3\.6-27B-Q4_K_M".*', r'\1"%s"' % served, s, count=1)
s = re.sub(r'(\n    - )"http://localhost:8080".*', r'\1"http://%s:%s"' % (host, port), s, count=1)
s = re.sub(r'(\nreport_dir: ).*', r'\1%s/' % rpt, s, count=1)
open(dst, "w").write(s)
PY
( cd "$ROOT" && source .venv/bin/activate 2>/dev/null && inference-endpoint benchmark from-config --config "$DST" ) > "$LOG" 2>&1
RC=$?
echo "=== golden e2e DONE rc=$RC  report=$RPT  log=$LOG ==="
grep -aE 'Score for bfcl|TPS|TTFT|Completed in|performance:' "$LOG" 2>/dev/null | tail -8
