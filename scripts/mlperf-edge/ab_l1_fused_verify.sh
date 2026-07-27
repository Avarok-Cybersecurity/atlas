#!/bin/bash
# L1 decode A/B: ATLAS_GDN_FUSED_VERIFY (fuse K=2 conv/norm verify epilogue).
# Leg A = base (flag off) ; Leg B = ATLAS_GDN_FUSED_VERIFY=1. Same binary/model/config.
# Bit-identical toggle -> combined_sha MUST match; TPOT delta is the signal.
set -u
IMG=atlas-gb10:followups
BIN=/workspace/.wt-decode-fold/target/release/spark
MODEL=centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf
HFCACHE=/workspace/.cache/huggingface
PORT=8888
OUTDIR=/workspace/.wt-decode-fold/ab_l1
mkdir -p "$OUTDIR"

# Frozen c2final serve env (ARM=bare / drafter defaults on).
BASE_ENV=(
  -e ATLAS_NO_FFN_NVFP4_MMQ=1 -e ATLAS_SSM_TAIL_MIDCHUNK=0 -e ATLAS_MTP_CATCHUP=0
  -e ATLAS_MTP_DRAFT_CONF=0.0 -e ATLAS_MTP_GATE_FORCE=1 -e ATLAS_SSM_TAIL_PROTECT=1
  -e ATLAS_SSM_TAIL_LEASE_TTL=128 -e ATLAS_BF16_TC_PREFILL=1
)
SERVE_FLAGS=(
  --host 0.0.0.0 --port "$PORT" --model-name qwen
  --max-seq-len 32768 --max-batch-size 1 --kv-cache-dtype bf16 --gpu-memory-utilization 0.70
  --enable-prefix-caching --ssm-cache-slots 128 --ssm-checkpoint-interval 32
  --speculative --num-drafts 2 --mtp-quantization bf16
  --tool-call-parser qwen3_xml --disable-tool-grammar true --disable-thinking
)

leg() {
  local tag="$1"; shift
  local extra=("$@")
  local CN="atlas-l1-$tag"
  for c in $(sudo docker ps -q --filter "name=atlas-l1-"); do sudo docker rm -f "$c" >/dev/null 2>&1; done
  # never touch the neighbour ollama on :8000
  if pgrep -f 'release/spark serve' | grep -qv $$; then :; fi
  sleep 4
  sudo docker run -d --name "$CN" --network host --gpus all --ipc=host \
    "${BASE_ENV[@]}" "${extra[@]}" \
    -v "$HFCACHE:/root/.cache/huggingface:ro" -v "$BIN:/usr/local/bin/spark:ro" \
    "$IMG" serve "$MODEL" "${SERVE_FLAGS[@]}" >/dev/null 2>&1
  local up=0
  for i in $(seq 1 150); do
    curl -sf -m4 "http://0.0.0.0:$PORT/v1/models" 2>/dev/null | grep -q qwen && { up=1; break; }
    sudo docker ps --format '{{.Names}}' | grep -q "^${CN}$" || { echo "[$tag] CONTAINER DIED"; sudo docker logs "$CN" 2>&1 | tail -25; return 1; }
    sleep 5
  done
  [ "$up" = 1 ] || { echo "[$tag] never came up"; sudo docker logs "$CN" 2>&1 | tail -25; return 1; }
  echo "===== LEG $tag up after ~$((i*5))s ====="
  python3 /workspace/.wt-decode-fold/ab_probe.py "$PORT" "$tag" "$OUTDIR/${tag}.json"
  sudo docker rm -f "$CN" >/dev/null 2>&1
  sleep 4
}

echo "### L1 A/B: fused K=2 verify epilogue — $(date)"
leg base_off                                   || exit 1
leg fused_on -e ATLAS_GDN_FUSED_VERIFY=1       || exit 1

python3 - <<'PY'
import json
a=json.load(open("/workspace/.wt-decode-fold/ab_l1/base_off.json"))
b=json.load(open("/workspace/.wt-decode-fold/ab_l1/fused_on.json"))
print("\n===== L1 SUMMARY =====")
print(f"  base_off : TPOT warm-med {a['tpot_med_warm']:.2f}ms  sha={a['combined_sha']}")
print(f"  fused_on : TPOT warm-med {b['tpot_med_warm']:.2f}ms  sha={b['combined_sha']}")
ident = a['combined_sha']==b['combined_sha']
print(f"  byte-identical output: {'YES' if ident else 'NO — A/B INVALID (bug!)'}")
if a['tpot_med_warm']:
    d=(b['tpot_med_warm']-a['tpot_med_warm'])/a['tpot_med_warm']*100
    print(f"  TPOT delta: {d:+.1f}%  ({'WIN' if d<0 else 'no win'})")
print("  VERDICT:", "FOLD" if (ident and b['tpot_med_warm']<a['tpot_med_warm']) else "DO NOT FOLD")
PY
echo "L1_AB_DONE"
