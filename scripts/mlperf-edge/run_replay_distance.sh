#!/bin/bash
# Serve the golden config and replay a REAL golden-run conversation through it,
# capturing the server's own Marconi replay accounting.
#
# Answers: on the actual MLPerf-edge workload, how many SSM tokens are replayed
# per warm turn? The TTFT law's 879 ms intercept back-solves to ~251 re-prefilled
# tokens at delta=0, which two-block tail-checkpoint rounding (<=32 tok) cannot
# explain. The dgx2 root cause was measured in CHAT mode (assistant output echoed
# back verbatim); the harness instead drives a flat prefix-extended prompt in which
# the generated tokens never reappear, so that explanation may not transfer.
# The server logs the answer, so measure instead of arguing.
#
# Usage: run_replay_distance.sh <atlas_bin> <outdir> [conversation_id] [max_turns]
set -u
BIN="${1:?path to the built spark binary}"
OUT="${2:?output dir}"
CID="${3:-django__django-16899}"
NT="${4:-25}"
HERE="$(cd "$(dirname "$0")" && pwd)"
EVENTS=/workspace/endpoints-fresh/results/chainK_golden_e2e_20260724_131209/events.jsonl
MODEL=centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf
PORT=8888
mkdir -p "$OUT"

sudo docker rm -f atlas-replaydist >/dev/null 2>&1; sleep 3
sudo docker run -d --name atlas-replaydist --network host --gpus all --ipc=host \
  -e ATLAS_NO_FFN_NVFP4_MMQ=1 -e ATLAS_SSM_TAIL_MIDCHUNK=0 -e ATLAS_MTP_CATCHUP=0 \
  -e ATLAS_MTP_DRAFT_CONF=0.0 -e ATLAS_MTP_GATE_FORCE=1 \
  -e ATLAS_SSM_TAIL_LEASE_TTL=128 -e ATLAS_BF16_TC_PREFILL=1 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
  -v "$BIN:/usr/local/bin/spark:ro" \
  atlas-gb10:followups serve "$MODEL" \
  --host 0.0.0.0 --port $PORT --model-name "$MODEL" \
  --max-seq-len 32768 --max-batch-size 1 --kv-cache-dtype bf16 --gpu-memory-utilization 0.70 \
  --enable-prefix-caching --ssm-cache-slots 128 --ssm-checkpoint-interval 32 \
  --speculative --num-drafts 3 --mtp-quantization bf16 \
  --tool-call-parser qwen3_xml --disable-tool-grammar true --disable-thinking >/dev/null 2>&1

for _ in $(seq 1 180); do
  curl -sf -m4 http://localhost:$PORT/v1/models 2>/dev/null | grep -q Qwen && break
  sudo docker ps --format '{{.Names}}' | grep -q atlas-replaydist || { echo SERVE_DIED; exit 1; }
  sleep 5
done
echo "=== serve up; replaying real conversation $CID ==="

python3 -u "$HERE/replay_distance_probe.py" $PORT "$EVENTS" "$CID" "$OUT/turns.json" \
    --max-turns "$NT" 2>&1 | tee "$OUT/probe.log"

# The measurement. Keep the raw lines AND a parsed histogram of the replay field.
sudo docker logs atlas-replaydist 2>&1 | grep -a "Marconi" > "$OUT/marconi.log" || true
wc -l < "$OUT/marconi.log" | xargs echo "Marconi log lines:"
python3 - "$OUT/marconi.log" <<'PY'
import re, sys, statistics
txt = open(sys.argv[1], errors="ignore").read()
# "restored from checkpoint at token N (skipping S tokens, replaying R SSM tokens to reach M)"
inter = [int(m) for m in re.findall(r"replaying (\d+) SSM tokens", txt)]
skip  = [int(m) for m in re.findall(r"skipping (\d+) tokens", txt)]
print(f"\nintermediate-hit events: {len(inter)}")
if inter:
    s = sorted(inter)
    print(f"  SSM tokens REPLAYED per hit: p50={statistics.median(s):.0f} "
          f"mean={statistics.mean(s):.0f} p90={s[int(.9*(len(s)-1))]} max={max(s)}")
    print(f"  total replayed = {sum(inter)}")
if skip:
    print(f"  tokens SKIPPED per hit: p50={statistics.median(skip):.0f} max={max(skip)}")
print("\nINTERPRETATION: replay p50 ~0-32 => the re-prefill lever does NOT exist on")
print("this workload and the 879 ms intercept is something else. replay p50 in the")
print("hundreds => it does, and end-of-generation checkpointing is worth pursuing.")
PY
sudo docker rm -f atlas-replaydist >/dev/null 2>&1
echo "REPLAY_DISTANCE_DONE out=$OUT"
