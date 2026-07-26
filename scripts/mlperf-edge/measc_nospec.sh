#!/bin/bash
# qwen Measurement C: non-spec (K=1) TPOT baseline. If T_no_spec < 31.39ms the spec path
# is HURTING (fix spec overhead, not acceptance). If > 31.39ms, base decode is the floor.
# Also a --num-drafts 1 (K=2) leg to see the spec-scaling from K=1->K=2->K=3.
set -u
IMG=atlas-gb10:followups
BIN=/workspace/.wt-decode-fold/target/release/spark
MODEL=centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf
HFCACHE=/workspace/.cache/huggingface
PORT=8888
OUTDIR=/workspace/.wt-decode-fold/measc
mkdir -p "$OUTDIR"
BASE_ENV=(
  -e ATLAS_NO_FFN_NVFP4_MMQ=1 -e ATLAS_SSM_TAIL_MIDCHUNK=0 -e ATLAS_MTP_CATCHUP=0
  -e ATLAS_MTP_DRAFT_CONF=0.0 -e ATLAS_MTP_GATE_FORCE=1 -e ATLAS_SSM_TAIL_PROTECT=1
  -e ATLAS_SSM_TAIL_LEASE_TTL=128 -e ATLAS_BF16_TC_PREFILL=1
)
COMMON="--host 0.0.0.0 --port $PORT --model-name qwen --max-seq-len 32768 --max-batch-size 1 --kv-cache-dtype bf16 --gpu-memory-utilization 0.70 --enable-prefix-caching --ssm-cache-slots 128 --ssm-checkpoint-interval 32 --mtp-quantization bf16 --tool-call-parser qwen3_xml --disable-tool-grammar true --disable-thinking"

leg() {  # tag  <spec-flags>
  local tag="$1"; shift
  local CN="atlas-measc-$tag"
  for c in $(sudo docker ps -q --filter "name=atlas-measc-"); do sudo docker rm -f "$c" >/dev/null 2>&1; done
  sleep 4
  sudo docker run -d --name "$CN" --network host --gpus all --ipc=host \
    "${BASE_ENV[@]}" -v "$HFCACHE:/root/.cache/huggingface:ro" -v "$BIN:/usr/local/bin/spark:ro" \
    "$IMG" serve "$MODEL" $COMMON "$@" >/dev/null 2>&1
  local up=0
  for i in $(seq 1 150); do
    curl -sf -m4 "http://0.0.0.0:$PORT/v1/models" 2>/dev/null | grep -q qwen && { up=1; break; }
    sudo docker ps --format '{{.Names}}' | grep -q "^${CN}$" || { echo "[$tag] DIED"; sudo docker logs "$CN" 2>&1|tail -20; return 1; }
    sleep 5
  done
  [ "$up" = 1 ] || { echo "[$tag] no serve"; sudo docker logs "$CN" 2>&1|tail -20; return 1; }
  echo "===== LEG $tag up ~$((i*5))s ====="
  python3 /workspace/.wt-decode-fold/ab_probe.py "$PORT" "$tag" "$OUTDIR/${tag}.json"
  sudo docker rm -f "$CN" >/dev/null 2>&1; sleep 4
}

echo "### Measurement C — non-spec baseline — $(date)"
leg nospec                             # no --speculative => pure autoregressive K=1
leg k2 --speculative --num-drafts 1    # K=2 (1 draft)
python3 - <<'PY'
import json,os
def load(t):
    p=f"/workspace/.wt-decode-fold/measc/{t}.json"
    return json.load(open(p)) if os.path.exists(p) else None
ns=load("nospec"); k2=load("k2")
print("\n===== MEASUREMENT C SUMMARY =====")
if ns: print(f"  non-spec (K=1) TPOT: {ns['tpot_med_warm']:.2f}ms")
if k2: print(f"  K=2 (1 draft)  TPOT: {k2['tpot_med_warm']:.2f}ms")
print("  K=3 (shipping) TPOT: ~43.13ms (draft_sweep d2, same probe)")
print("  vLLM target        : 31.39ms")
if ns:
    t=ns['tpot_med_warm']
    print(f"  -> non-spec {'<' if t<31.39 else '>'} 31.39ms: " +
          ("SPEC IS HURTING — fix spec overhead, not acceptance" if t<31.39 else
           "base decode is the floor; spec is needed; gap = drafter efficiency"))
PY
echo "MEASC_DONE"
