#!/bin/bash
# L6 A/B: fuse the K=3 MTP-verify conv/norm epilogue. num-drafts=2 (verify
# width = K=3) FIXED. Only toggled var: ATLAS_GDN_FUSED_VERIFY.
#   leg A (unset) = per-token conv/norm epilogue (shipping baseline)
#   leg B (=1)    = fused K=3 conv (gdn_verify_fused_conv_kn) + fused norm
# Byte-identical is MANDATORY (cost-only fusion). GATE_FORCE=1 pins greedy.
set -u
IMG=atlas-gb10:followups
BIN=/workspace/.wt-decode-fold/target/release/spark
MODEL=centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf
HFCACHE=/workspace/.cache/huggingface
PORT=8888
OUTDIR=/workspace/.wt-decode-fold/ab_l6
mkdir -p "$OUTDIR"

BASE_ENV=(
  -e ATLAS_NO_FFN_NVFP4_MMQ=1 -e ATLAS_SSM_TAIL_MIDCHUNK=0 -e ATLAS_MTP_CATCHUP=0
  -e ATLAS_MTP_DRAFT_CONF=0.0 -e ATLAS_MTP_GATE_FORCE=1 \
  -e ATLAS_SSM_TAIL_LEASE_TTL=128 -e ATLAS_BF16_TC_PREFILL=1
  -e ATLAS_MTP_CARRY_DEBUG=1
)

# leg <tag> <extra -e args...>
leg() {
  local tag="$1"; shift
  local CN="atlas-l6-$tag"
  for c in $(sudo docker ps -q --filter "name=atlas-l6-"); do sudo docker rm -f "$c" >/dev/null 2>&1; done
  sleep 4
  sudo docker run -d --name "$CN" --network host --gpus all --ipc=host \
    "${BASE_ENV[@]}" "$@" \
    -v "$HFCACHE:/root/.cache/huggingface:ro" -v "$BIN:/usr/local/bin/spark:ro" \
    "$IMG" serve "$MODEL" --host 0.0.0.0 --port "$PORT" --model-name qwen \
    --max-seq-len 32768 --max-batch-size 1 --kv-cache-dtype bf16 --gpu-memory-utilization 0.70 \
    --enable-prefix-caching --ssm-cache-slots 128 --ssm-checkpoint-interval 32 \
    --speculative --num-drafts 2 --mtp-quantization bf16 \
    --tool-call-parser qwen3_xml --disable-tool-grammar true --disable-thinking >/dev/null 2>&1
  local up=0 i
  for i in $(seq 1 150); do
    curl -sf -m4 "http://0.0.0.0:$PORT/v1/models" 2>/dev/null | grep -q qwen && { up=1; break; }
    sudo docker ps --format '{{.Names}}' | grep -q "^${CN}$" || { echo "[$tag] DIED"; sudo docker logs "$CN" 2>&1 | tail -30; return 1; }
    sleep 5
  done
  [ "$up" = 1 ] || { echo "[$tag] no serve"; sudo docker logs "$CN" 2>&1 | tail -30; return 1; }
  echo "===== LEG $tag up after ~$((i*5))s ====="
  python3 /workspace/.wt-decode-fold/ab_probe.py "$PORT" "$tag" "$OUTDIR/${tag}.json" || return 1
  sudo docker rm -f "$CN" >/dev/null 2>&1
  sleep 4
}

echo "### L6 K=3 conv/norm-fusion A/B — $(date)"
leg A_unset || exit 1
leg B_fused -e ATLAS_GDN_FUSED_VERIFY=1 || exit 1

python3 - <<'PY'
import json
a=json.load(open("/workspace/.wt-decode-fold/ab_l6/A_unset.json"))
b=json.load(open("/workspace/.wt-decode-fold/ab_l6/B_fused.json"))
print("\n===== L6 K=3 CONV/NORM FUSION A/B =====")
for t,d in (("A per-token (unset)",a),("B fused K=3 (=1)",b)):
    print(f"  {t:22}: TPOT warm-med {d['tpot_med_warm']:.2f}ms  ttft-med {d['ttft_med']:.0f}ms  sha={d['combined_sha']}")
ident = a['combined_sha']==b['combined_sha']
print(f"  BYTE-IDENTICAL: {'YES' if ident else 'NO — CORRECTNESS BUG'}")
if a['tpot_med_warm']:
    delta=(b['tpot_med_warm']-a['tpot_med_warm'])/a['tpot_med_warm']*100
    print(f"  TPOT delta (B vs A): {delta:+.2f}%  ({'B FASTER' if delta<0 else 'B slower/neutral'})")
if ident and a['tpot_med_warm'] and b['tpot_med_warm']<a['tpot_med_warm']:
    print("  VERDICT: FOLD (byte-identical AND TPOT improves)")
elif not ident:
    print("  VERDICT: DO-NOT-FOLD (output diverged — correctness bug)")
else:
    print("  VERDICT: DO-NOT-FOLD (byte-identical but no TPOT win)")
PY
echo "L6_DONE"
