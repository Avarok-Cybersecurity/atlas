#!/bin/bash
# Serve-validation sweep for three measured-but-never-validated levers that sit
# in-tree behind default-OFF flags on the shipping GB10 golden config.
#
# Why these three, and what each should move (wall decomposition of the 4104 s
# golden perf phase, measured 2026-07-25 from events.jsonl over 1007 samples:
# decode 59.6% / fixed per-turn TTFT 21.1% / marginal prefill 18.8%):
#
#  (ATLAS_DECODE_GRAPHS_MULTISEQ was investigated and REJECTED before costing a
#   leg, 2026-07-25. Its comment advertises "the dominant lever for n>=2 decode
#   (~1500 kernel launches/step)", but `decode_a2.rs` iterates over `seqs` —
#   concurrent SEQUENCES contributing one token each — not the K verify tokens.
#   This config is --max-batch-size 1 single-stream and MTP is gated to
#   active.len()==1, so that path is never entered. Separately, the K=4 verify
#   path ALREADY captures graphs by default: verify_c.rs:170
#   `use_graphs = comm.is_none() && !hss_engaged && !lora_eager`. Nothing to win.)
#
#  noregres ATLAS_NO_GDN_REGRESIDENT=1   (NOTE: this leg REMOVES the lever)
#           Register-resident GDN recurrence, a drop-in for WY4 on the warm
#           Marconi replay path (FLA cannot take that path: its chunked algebra
#           is not token-equal at an arbitrary snapshot offset and would poison
#           shared prefix blocks). Author measured cos 1.0 / max|dH|~1e-8 vs WY4
#           and ~2.9x in isolation, then left it OFF "until serve-validated".
#           Target: warm-replay TTFT -> the 18.8% marginal-prefill slice.
#           SIGN INVERTED vs the rest of this sweep: PR #369 folded this lever
#           default-ON, so `control` already HAS it and the only way to measure
#           it is to take it away. A positive `ATLAS_GDN_REGRESIDENT` does not
#           exist — setting it is a no-op, and this leg used to do exactly that,
#           making it a duplicate of `control`. Read the delta backwards: the
#           lever's benefit is control-minus-noregres, not noregres-minus-control.
#
#  bf16proj ATLAS_BF16_TC_PROJ=1
#           Routes QKV/o projection prefill through the BF16-TC kernel
#           (bit-identical to base w4a16_gemm) instead of the default t_m128,
#           which crushes activations to FP8 E4M3. We ALREADY ship the FFN
#           sibling ATLAS_BF16_TC_PREFILL=1, so today's config is asymmetric:
#           lossless on the FFN, FP8-perturbed on the attention projections.
#           Target: accuracy (removes a perturbation); speed effect unknown.
#
# TRAP: several of these are PRESENCE flags (`var_os(..).is_some()`), so `=0`
# ENABLES them. Each leg is therefore a separate `docker run` with the variable
# entirely ABSENT on the control, never a `-e FLAG=$value` line.
# (ATLAS_DECODE_GRAPHS_MULTISEQ is the exception — it is a strict `== "1"` —
# but it is treated the same way here so the pattern cannot be got wrong.)
# `noregres` is the one leg that INVERTS this: ATLAS_NO_GDN_REGRESIDENT is a
# kill switch (strict `== "1"`), so it is the leg that SETS the variable while
# `control` leaves it absent and keeps the lever on. Before checking any delta,
# confirm the leg's env actually reached the server — a lever whose variable
# name does not exist reports a clean, believable, meaningless zero.
#
# Usage: lever_sweep.sh <atlas_bin> <outdir> [legs...]   default: all four
set -u
BIN="${1:?path to the built spark binary}"
OUT="${2:?output dir}"
shift 2
LEGS=("$@"); [ ${#LEGS[@]} -eq 0 ] && LEGS=(control noregres bf16proj)
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL=centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf
PORT=8888
mkdir -p "$OUT"

extra_for() {
  case "$1" in
    control)  echo "" ;;
    graphs)   echo "-e ATLAS_DECODE_GRAPHS_MULTISEQ=1" ;;
    noregres) echo "-e ATLAS_NO_GDN_REGRESIDENT=1" ;;
    bf16proj) echo "-e ATLAS_BF16_TC_PROJ=1" ;;
    *) echo "UNKNOWN_LEG" ;;
  esac
}

for leg in "${LEGS[@]}"; do
  EXTRA="$(extra_for "$leg")"
  [ "$EXTRA" = "UNKNOWN_LEG" ] && { echo "unknown leg: $leg" >&2; exit 2; }
  sudo docker rm -f atlas-lever >/dev/null 2>&1; sleep 3
  # shellcheck disable=SC2086
  sudo docker run -d --name atlas-lever --network host --gpus all --ipc=host \
    -e ATLAS_NO_FFN_NVFP4_MMQ=1 -e ATLAS_SSM_TAIL_MIDCHUNK=0 -e ATLAS_MTP_CATCHUP=0 \
    -e ATLAS_MTP_DRAFT_CONF=0.0 -e ATLAS_MTP_GATE_FORCE=1 \
    -e ATLAS_SSM_TAIL_LEASE_TTL=128 -e ATLAS_BF16_TC_PREFILL=1 $EXTRA \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
    -v "$BIN:/usr/local/bin/spark:ro" \
    atlas-gb10:followups serve "$MODEL" \
    --host 0.0.0.0 --port $PORT --model-name "$MODEL" \
    --max-seq-len 32768 --max-batch-size 1 --kv-cache-dtype bf16 --gpu-memory-utilization 0.70 \
    --enable-prefix-caching --ssm-cache-slots 128 --ssm-checkpoint-interval 32 \
    --speculative --num-drafts 3 --mtp-quantization bf16 \
    --tool-call-parser qwen3_xml --disable-tool-grammar true --disable-thinking >/dev/null 2>&1

  ok=0
  for _ in $(seq 1 180); do
    curl -sf -m4 http://localhost:$PORT/v1/models 2>/dev/null | grep -q Qwen && { ok=1; break; }
    sudo docker ps --format '{{.Names}}' | grep -q atlas-lever || { echo "SERVE_DIED leg=$leg"; break; }
    sleep 5
  done
  [ $ok -eq 1 ] || { sudo docker logs atlas-lever 2>&1 | tail -30 > "$OUT/$leg.died.txt"; continue; }
  echo "=== leg=$leg serve up (env: ${EXTRA:-<none>}) ==="

  # Gate C2 FIRST: an NVFP4 build can pass timing gates while emitting garbage.
  curl -sf -m60 "http://localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a Python function that returns the nth Fibonacci number iteratively.\"}],\"max_tokens\":150,\"temperature\":0,\"seed\":42}" \
    | python3 -c 'import sys,json;print(json.load(sys.stdin)["choices"][0]["message"]["content"])' \
    > "$OUT/$leg.coherence.txt" 2>&1
  curl -sf -m60 "http://localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Paris? Use the get_weather tool.\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get current weather for a location\",\"parameters\":{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\"}},\"required\":[\"location\"]}}}],\"max_tokens\":120,\"temperature\":0,\"seed\":42}" \
    | python3 -c 'import sys,json;print(json.dumps(json.load(sys.stdin)["choices"][0]["message"].get("tool_calls")))' \
    > "$OUT/$leg.toolcall.txt" 2>&1

  python3 -u "$HERE/warm_tpot_probe.py"   $PORT "$leg" "$OUT/$leg.tpot.json" --turns 8 2>&1 | tee "$OUT/$leg.tpot.log"
  python3 -u "$HERE/warm_replay_probe.py" $PORT "$leg" "$OUT/$leg.ttft.json" --reps 5 2>&1 | tee "$OUT/$leg.ttft.log"

  # Proof the flag took effect. The GDN banners fire on first prefill/replay,
  # NOT at startup, so they must be read AFTER the probes, not before.
  sudo docker logs atlas-lever 2>&1 \
    | grep -aE 'GDN prefill: (FLA chunked|REGISTER-RESIDENT)|CUDA graph|graph capture|BF16.?TC' \
    | sort -u | head -20 | tee "$OUT/$leg.banner.txt"
  sudo docker rm -f atlas-lever >/dev/null 2>&1
done
echo "LEVER_SWEEP_DONE out=$OUT"
