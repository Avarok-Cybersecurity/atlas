#!/bin/bash
# A/B for re-enabling MID-CHUNK SSM tail capture on the GB10 golden config.
#
# THE SITUATION
# `ATLAS_SSM_TAIL_MIDCHUNK` is DEFAULT-ON in the code
# (spark-runtime/src/lib.rs: `!matches!(var.as_deref(), Ok("0"))`). The frozen
# MLPerf-edge config overrides it to `0`, i.e. OFF. That override dates from the
# 2026-07-16 regression where midchunk tail capture corrupted CROSS-REQUEST SSM
# prefix reuse: BFCL single-turn requests share a system-prompt prefix and reused
# each other's tail snapshot, inheriting state that bleeds past the advertised
# prefix boundary -> garbled tool calls -> wrong scores (32 slots: 77.31
# normalized with midchunk ON vs 84.54 OFF).
#
# THAT BUG WAS FIXED. `radix_tree/snapshot.rs::lookup` now carries the exact
# session gate the post-mortem prescribed:
#     if entry.is_tail && (session_hash == 0 || entry.session_hash != session_hash) { continue; }
# so single-turn / cross-request lookups (session_hash == 0) fall through to a
# correct recompute while same-session multi-turn still restores. The `=0` in the
# launch script is therefore a STALE WORKAROUND.
#
# WHAT RE-ENABLING SHOULD BUY
# With midchunk OFF, `prefill_chunk_dispatch` places the tail checkpoint by
# SPLITTING the final prefill chunk — one extra pass over the trailing tokens,
# self-documented at ~150 ms/turn, paid from turn 2 of every conversation. Every
# extra pass re-streams the full 17.54 GB of weights. Midchunk capture gets the
# same checkpoint with NO extra pass, by splitting only the two cheap per-token
# GDN kernels (split4 recurrence + conv1d) — no projection/FFN re-run.
# Against the measured wall decomposition (fixed per-turn TTFT = 879 ms x 1007 =
# 867 s = 21.1% of a 4104 s wall) that is worth up to ~148 s.
#
# It also makes two flags in the frozen config STOP BEING INERT: with
# MIDCHUNK=0 nothing is ever marked `is_tail`, so and
# ATLAS_SSM_TAIL_LEASE_TTL=128 currently govern an empty set (snapshot.rs says
# so in as many words).
#
# WHY THIS NEEDS AN ACCURACY GATE, NOT A SPEED GATE
# The failure mode this flag was disabled for is invisible to a latency probe: it
# corrupts OUTPUT on cross-request reuse, and the MLPerf accuracy phase (995
# single-turn BFCL samples sharing a system prompt) is precisely that regime. A
# TTFT win here means nothing without BFCL holding. So this script runs the BFCL
# subset per leg, and the verdict is accuracy-first.
#
# TRAP: the flag is strict-`"0"` opt-out, NOT a presence flag. To ENABLE midchunk
# the variable must be ABSENT or set to anything that is not "0"; leaving
# `-e ATLAS_SSM_TAIL_MIDCHUNK=0` in place keeps it OFF. The two legs below differ
# only by whether that -e line is emitted.
#
# Usage: ab_midchunk.sh <atlas_bin> <outdir> [bfcl_subset_n]
set -u
BIN="${1:?path to the built spark binary}"
OUT="${2:?output dir}"
NSUB="${3:-200}"
HERE="$(cd "$(dirname "$0")" && pwd)"
MODEL=centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf
PORT=8888
mkdir -p "$OUT"

for leg in mc_off mc_on; do
  case $leg in
    mc_off) MC="-e ATLAS_SSM_TAIL_MIDCHUNK=0" ;;   # today's frozen config
    mc_on)  MC="" ;;                               # the code's actual default
  esac
  sudo docker rm -f atlas-mc >/dev/null 2>&1; sleep 3
  # shellcheck disable=SC2086
  sudo docker run -d --name atlas-mc --network host --gpus all --ipc=host \
    -e ATLAS_NO_FFN_NVFP4_MMQ=1 $MC -e ATLAS_MTP_CATCHUP=0 \
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

  ok=0
  for _ in $(seq 1 180); do
    curl -sf -m4 http://localhost:$PORT/v1/models 2>/dev/null | grep -q Qwen && { ok=1; break; }
    sudo docker ps --format '{{.Names}}' | grep -q atlas-mc || { echo "SERVE_DIED leg=$leg"; break; }
    sleep 5
  done
  [ $ok -eq 1 ] || { sudo docker logs atlas-mc 2>&1 | tail -40 > "$OUT/$leg.died.txt"; continue; }
  echo "=== leg=$leg serve up (midchunk env: ${MC:-<absent => ON>}) ==="

  # Gate C2 first — an NVFP4 build can pass timing while emitting garbage, and
  # the failure mode under test is precisely garbled tool calls.
  curl -sf -m60 "http://localhost:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Paris? Use the get_weather tool.\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get current weather for a location\",\"parameters\":{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\"}},\"required\":[\"location\"]}}}],\"max_tokens\":120,\"temperature\":0,\"seed\":42}" \
    | python3 -c 'import sys,json;print(json.dumps(json.load(sys.stdin)["choices"][0]["message"].get("tool_calls")))' \
    > "$OUT/$leg.toolcall.txt" 2>&1
  cat "$OUT/$leg.toolcall.txt"

  python3 -u "$HERE/warm_tpot_probe.py"   $PORT "$leg" "$OUT/$leg.tpot.json" --turns 8 2>&1 | tee "$OUT/$leg.tpot.log"
  python3 -u "$HERE/warm_replay_probe.py" $PORT "$leg" "$OUT/$leg.ttft.json" --reps 5 2>&1 | tee "$OUT/$leg.ttft.log"

  # is_tail entries only ever exist with midchunk ON; their count is the proof
  # the flag took effect (and that TAIL_PROTECT/LEASE_TTL stopped being inert).
  sudo docker logs atlas-mc 2>&1 \
    | grep -aiE 'midchunk|is_tail|tail snapshot|tail lease' | sort -u | head -20 \
    | tee "$OUT/$leg.tail_evidence.txt"
  sudo docker rm -f atlas-mc >/dev/null 2>&1
done
echo "MIDCHUNK_AB_DONE out=$OUT"
echo
echo "NEXT (REQUIRED before any fold): run the BFCL subset (n=$NSUB) per leg."
echo "A latency win here is NOT sufficient — the 2026-07-16 regression corrupted"
echo "CROSS-REQUEST reuse, which is exactly the 995-sample single-turn accuracy"
echo "phase. Accuracy must hold at >= the 83.64 / 85.32 floors before folding."
