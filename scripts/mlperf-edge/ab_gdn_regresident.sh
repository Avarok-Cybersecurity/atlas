#!/bin/bash
# A/B for ATLAS_GDN_REGRESIDENT — the register-resident warm-replay GDN recurrence.
#
# WHAT THIS TESTS
# On a warm turn the Marconi SSM snapshot is restored and the suffix after the
# match point is replayed. That replay deliberately does NOT use the FLA chunked
# path (FLA's 64-token grid is only token-equal when it matches the pass that
# produced the cached K/V; replaying from an arbitrary snapshot offset regroups
# the recurrence and its bf16 intermediates drift into SHARED prefix-cache
# blocks — the 2026-06-10 token-stutter corruption). Warm replay therefore falls
# back to WY4, which keeps H in FP32 smem token-sequentially.
#
# `gated_delta_rule_prefill_regresident` is a drop-in for WY4 on that path: H
# lives in registers (one warp per v-column) instead of 64 KB of smem, so >=2
# CTA/SM and no per-token barriers. Its author measured cosine 1.0 / max|dH|~1e-8
# vs WY4 (same acceptance class) and ~2.9x in isolation, ~24% warm TTFT — then
# left it behind a default-OFF flag "until serve-validated". Nobody validated it.
#
# WHY IT SHOULD MATTER HERE
# The MLPerf-edge agentic wall decomposes (2026-07-25, events.jsonl, 1007
# samples) as: decode 59.6% / fixed per-turn TTFT 21.1% / marginal prefill 18.8%.
# That 18.8% (771 s) IS the warm-replay path, run over a p50 of 210 new tokens
# through 48 GDN layers every turn.
#
# TRAP: there is no `ATLAS_GDN_REGRESIDENT` variable. PR #369 folded this
# lever default-ON and the only thing Rust reads is the NEGATIVE kill switch
# (model_levers.rs: `var("ATLAS_NO_GDN_REGRESIDENT").as_deref() != Ok("1")`).
# Setting the positive spelling does NOTHING, so until this was fixed both legs
# below ran the identical configuration — regresident ON — and any delta they
# reported was noise. The legs are therefore inverted relative to the original
# script: the DEFAULT leg is the one with the feature, and the control leg is
# the one that has to switch it off.
#
# Note the kill switch is an exact `== "1"` test, not a presence test:
# `ATLAS_NO_GDN_REGRESIDENT=0` leaves the feature ON.
#
# Usage: ab_gdn_regresident.sh <atlas_bin> <outdir> [reps]
set -u
BIN="${1:?path to the built spark binary}"
OUT="${2:?output dir}"
REPS="${3:-7}"
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL=centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf
PORT=8888
mkdir -p "$OUT"

serve() { # $1 = leg name, $2 = extra -e args (may be empty)
  sudo docker rm -f atlas-rr-ab >/dev/null 2>&1; sleep 3
  # shellcheck disable=SC2086
  sudo docker run -d --name atlas-rr-ab --network host --gpus all --ipc=host \
    -e ATLAS_NO_FFN_NVFP4_MMQ=1 -e ATLAS_SSM_TAIL_MIDCHUNK=0 -e ATLAS_MTP_CATCHUP=0 \
    -e ATLAS_MTP_DRAFT_CONF=0.0 -e ATLAS_MTP_GATE_FORCE=1 \
    -e ATLAS_SSM_TAIL_LEASE_TTL=128 -e ATLAS_BF16_TC_PREFILL=1 $2 \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
    -v "$BIN:/usr/local/bin/spark:ro" \
    atlas-gb10:followups serve "$MODEL" \
    --host 0.0.0.0 --port $PORT --model-name "$MODEL" \
    --max-seq-len 32768 --max-batch-size 1 --kv-cache-dtype bf16 --gpu-memory-utilization 0.70 \
    --enable-prefix-caching --ssm-cache-slots 128 --ssm-checkpoint-interval 32 \
    --speculative --num-drafts 3 --mtp-quantization bf16 \
    --tool-call-parser qwen3_xml --disable-tool-grammar true --disable-thinking >/dev/null 2>&1
  for _ in $(seq 1 180); do
    curl -sf -m4 http://localhost:$PORT/v1/models 2>/dev/null | grep -q Qwen && return 0
    sudo docker ps --format '{{.Names}}' | grep -q atlas-rr-ab || { echo "SERVE_DIED leg=$1"; return 1; }
    sleep 5
  done
  echo "SERVE_TIMEOUT leg=$1"; return 1
}

# Which GDN prefill path is live is printed once by a `Once` guard. Capture it
# per leg: this is the only proof the flag actually took effect, and the
# regresident banner fires on the first warm REPLAY, not at startup.
banners() {
  sudo docker logs atlas-rr-ab 2>&1 \
    | grep -aE 'GDN prefill: (FLA chunked|REGISTER-RESIDENT)' | sort -u | tee "$OUT/$1.banner.txt"
}

for leg in control regresident; do
  case $leg in
    control)     EXTRA="-e ATLAS_NO_GDN_REGRESIDENT=1" ;;  # kill switch => WY4
    regresident) EXTRA="" ;;                               # default => ON, see TRAP
  esac
  serve "$leg" "$EXTRA" || exit 1
  echo "=== leg=$leg serve up ==="
  python3 -u "$HERE/warm_replay_probe.py" $PORT "$leg" \
      "$OUT/$leg.json" --reps "$REPS" 2>&1 | tee "$OUT/$leg.log"
  banners "$leg"
  # Snapshot the raw completions so the two legs can be diffed for token equality.
  cp "$OUT/$leg.json" "$OUT/$leg.raw.json" 2>/dev/null
  sudo docker rm -f atlas-rr-ab >/dev/null 2>&1
done

echo "=== BANNERS ==="
for leg in control regresident; do echo "--- $leg"; cat "$OUT/$leg.banner.txt" 2>/dev/null; done
python3 "$HERE/warm_replay_compare.py" "$OUT/control.json" "$OUT/regresident.json" | tee "$OUT/VERDICT.txt"
echo "RR_AB_DONE out=$OUT"
