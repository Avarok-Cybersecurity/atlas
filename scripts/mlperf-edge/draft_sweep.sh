#!/bin/bash
# L6-viability: full-serve draft-count sweep. Does MORE drafts net-win TPOT now that
# carry-context + lm_head-batchm are in main? Legs: --num-drafts 2 (shipping) vs 3.
# GATE_FORCE=1 pins greedy output; capture real per-position draft-accept counters.
set -u
IMG=atlas-gb10:followups
BIN=/workspace/.wt-decode-fold/target/release/spark
MODEL=centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf
HFCACHE=/workspace/.cache/huggingface
PORT=8888
OUTDIR=/workspace/.wt-decode-fold/draft_sweep
mkdir -p "$OUTDIR"

BASE_ENV=(
  -e ATLAS_NO_FFN_NVFP4_MMQ=1 -e ATLAS_SSM_TAIL_MIDCHUNK=0 -e ATLAS_MTP_CATCHUP=0
  -e ATLAS_MTP_DRAFT_CONF=0.0 -e ATLAS_MTP_GATE_FORCE=1 \
  -e ATLAS_SSM_TAIL_LEASE_TTL=128 -e ATLAS_BF16_TC_PREFILL=1
  -e ATLAS_MTP_CARRY_DEBUG=1   # surfaces draft/accept counters in the log
)

leg() {
  local nd="$1" tag="d${1}"
  local CN="atlas-sweep-$tag"
  for c in $(sudo docker ps -q --filter "name=atlas-sweep-"); do sudo docker rm -f "$c" >/dev/null 2>&1; done
  sleep 4
  sudo docker run -d --name "$CN" --network host --gpus all --ipc=host \
    "${BASE_ENV[@]}" \
    -v "$HFCACHE:/root/.cache/huggingface:ro" -v "$BIN:/usr/local/bin/spark:ro" \
    "$IMG" serve "$MODEL" --host 0.0.0.0 --port "$PORT" --model-name qwen \
    --max-seq-len 32768 --max-batch-size 1 --kv-cache-dtype bf16 --gpu-memory-utilization 0.70 \
    --enable-prefix-caching --ssm-cache-slots 128 --ssm-checkpoint-interval 32 \
    --speculative --num-drafts "$nd" --mtp-quantization bf16 \
    --tool-call-parser qwen3_xml --disable-tool-grammar true --disable-thinking >/dev/null 2>&1
  local up=0
  for i in $(seq 1 150); do
    curl -sf -m4 "http://0.0.0.0:$PORT/v1/models" 2>/dev/null | grep -q qwen && { up=1; break; }
    sudo docker ps --format '{{.Names}}' | grep -q "^${CN}$" || { echo "[$tag] DIED"; sudo docker logs "$CN" 2>&1 | tail -25; return 1; }
    sleep 5
  done
  [ "$up" = 1 ] || { echo "[$tag] no serve"; sudo docker logs "$CN" 2>&1 | tail -25; return 1; }
  echo "===== LEG $tag (num-drafts=$nd) up after ~$((i*5))s ====="
  python3 /workspace/.wt-decode-fold/ab_probe.py "$PORT" "$tag" "$OUTDIR/${tag}.json"
  # capture per-position draft-accept counters emitted during the probe
  sudo docker logs "$CN" 2>&1 | grep -aiE 'draft|accept|n_acc|spec|match|verify' | tail -40 > "$OUTDIR/${tag}_serve_accept.log"
  echo "  [accept counters -> $OUTDIR/${tag}_serve_accept.log ($(wc -l <"$OUTDIR/${tag}_serve_accept.log") lines)]"
  sudo docker rm -f "$CN" >/dev/null 2>&1
  sleep 4
}

echo "### draft-count sweep — $(date)"
leg 2 || exit 1
leg 3 || exit 1

python3 - <<'PY'
import json
d2=json.load(open("/workspace/.wt-decode-fold/draft_sweep/d2.json"))
d3=json.load(open("/workspace/.wt-decode-fold/draft_sweep/d3.json"))
print("\n===== DRAFT-COUNT SWEEP SUMMARY =====")
for t,d in (("num-drafts=2 (shipping)",d2),("num-drafts=3",d3)):
    print(f"  {t:24}: TPOT warm-med {d['tpot_med_warm']:.2f}ms  sha={d['combined_sha']}")
ident = d2['combined_sha']==d3['combined_sha']
print(f"  output byte-identical across legs: {'YES (clean A/B)' if ident else 'NO — trajectories diverged, TPOT is ms/tok-normalized but interpret with care'}")
if d2['tpot_med_warm']:
    delta=(d3['tpot_med_warm']-d2['tpot_med_warm'])/d2['tpot_med_warm']*100
    print(f"  TPOT delta (nd3 vs nd2): {delta:+.1f}%  ({'nd=3 WINS' if delta<-1 else 'nd=3 loses/neutral'})")
print("  vLLM target TPOT: 31.39ms")
PY
echo "SWEEP_DONE"
