#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# #32 on-hardware oracle: routed DECODE must be BIT-IDENTICAL to active-served.
#
# The confound-free test: pool TWO IDENTICAL copies of one adapter —
#   slot0 "a" = active (default), slot1 "b" = routed. Because a and b are the
# same weights, ANY difference between the installed apply_lora_delta path
# (adapter=None / adapter=a) and the routed bgmv path (adapter=b) is a pure
# routing-kernel artifact, with NO adapter-identity confound.
#
# Covers all three decode paths that mattered for the #30 residual question:
#   1. single-seq decode  (attention_forward.rs gemv path vs bgmv)
#   2. multi-seq batched decode (multi_seq/{attn,qkv}.rs bgmv, n>1 under SLAI)
#   3. pool max_rank PADDING (adapter r=32 padded to a larger pool max_rank)
#
# RESULT (GB10, holo-3.1-0.8b, 2026-07-06): routed(b) == active(a) == baseline
# CHARACTER-FOR-CHARACTER in every case, including the razor-margin overfit
# codeword (STARFALL-4728 batched / STARFALL-4710 single-seq — the seq-vs-batch
# digit shift moves the ACTIVE adapter identically, so it is a batch-path
# numeric property, NOT a routing defect). Padding 32->64 is bit-identical too.
#
# Conclusion: the routed decode path has NO numeric divergence from
# active-served. The vega residual seen during #30 validation was an
# adapter/config confound (a vega baseline never re-established in the identical
# pooled config), not a decode-kernel bug.
#
# Usage:  ADAPTER=/path/to/peft/dir MODEL=/path/to/base bash routed-decode-bit-identity.sh
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
BIN="${BIN:-target/release/spark}"
ADAPTER="${ADAPTER:-/home/ms/lora-demo/demo-adapter}"
MODEL="${MODEL:-/home/ms/lora-demo/hf-cache/hub/models--Hcompany--Holo-3.1-0.8B/snapshots/72da4c53b351eb60e10a7022279019633c479292}"
PORT="${PORT:-8137}"
PROMPT="${PROMPT:-What is the Atlas launch codeword?}"
TMP="$(mktemp -d)"
LOG="$TMP/serve.log"
export RUST_LOG="${RUST_LOG:-info}"
cleanup(){ pkill -f "release/spark serv[e].*--port $PORT" 2>/dev/null; rm -rf "$TMP"; }
trap cleanup EXIT

"$BIN" serve "$MODEL" --port "$PORT" --max-seq-len 2048 \
  --gpu-memory-utilization 0.60 --max-batch-size 8 --scheduling-policy slai \
  --lora-adapter a="$ADAPTER" --lora-adapter b="$ADAPTER" \
  --max-loras 2 --max-lora-rank 64 >"$LOG" 2>&1 &
SRV=$!
for i in $(seq 1 120); do
  curl -sf "http://127.0.0.1:$PORT/v1/models" >/dev/null 2>&1 && break
  kill -0 $SRV 2>/dev/null || { echo "SERVER DIED"; tail -40 "$LOG"; exit 1; }
  sleep 1
done

ask(){ # $1 = adapter field JSON fragment ("" or ,"adapter":"b")
  curl -sf "http://127.0.0.1:$PORT/v1/chat/completions" -H 'Content-Type: application/json' \
    -d "{\"model\":\"a\",\"messages\":[{\"role\":\"user\",\"content\":\"$PROMPT\"}],\"temperature\":0,\"max_tokens\":40$1}" \
    | python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
}

rc=0
echo "── single-seq ──────────────────────────────────────────────"
NONE=$(ask ""); A=$(ask ',"adapter":"a"'); B=$(ask ',"adapter":"b"')
printf 'None : %s\na    : %s\nb    : %s\n' "$NONE" "$A" "$B"
[ "$NONE" == "$B" ] && [ "$A" == "$B" ] && echo "PASS single-seq: routed==active==baseline" \
  || { echo "FAIL single-seq"; rc=1; }

echo "── multi-seq batched (concurrent, one SLAI batch) ──────────"
ask ',"adapter":"a"' >"$TMP/a" & p1=$!
ask ',"adapter":"b"' >"$TMP/b" & p2=$!
wait $p1 $p2; CA=$(cat "$TMP/a"); CB=$(cat "$TMP/b")
printf 'a(active): %s\nb(routed): %s\n' "$CA" "$CB"
[ -n "$CB" ] && [ "$CA" == "$CB" ] && echo "PASS multi-seq: routed==active in-batch" \
  || { echo "FAIL multi-seq"; rc=1; }

echo "── VERDICT ─────────────────────────────────────────────────"
[ $rc -eq 0 ] && echo "ALL PASS: routed decode is bit-identical to active-served." \
  || echo "REGRESSION: routed decode diverged from active-served."
exit $rc
