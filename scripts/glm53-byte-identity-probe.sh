#!/bin/bash
# GLM-5.3 byte-identity gate — the six probes every change on this port must match.
#
# 🔴 ORDER MATTERS AND THE SERVER MUST BE FRESH. Prefix caching makes a completion's hash
# history-dependent: the same prompt issued second can produce a different continuation than
# it does first. Run this as the FIRST six requests after a start, in this order, always.
#
# Prints `<name> <sha256[0:8]> <tok/s>` per probe. Compare the hash column across arms; the
# tok/s column is indicative only (one rep, no warmup discipline).
#
# Sealed t47 reference (2026-08-28, world=2 TP=2 EP=2, fp8 KV, ATLAS_EP_GRAPHS=1):
#   de4e9745 / 5f16d368 / 04c73e90 / 2a7c7286 / 015083e7 / 44597015
set -uo pipefail
# 🪤 The server binds LOOPBACK only. Every request has to originate inside the node, so this
# runs curl over ssh rather than reaching the port directly from the workstation.
NODE="${NODE:-10.10.10.1}"
PORT="${PORT:-8888}"
SSH=(sudo -n -u cluster ssh -o BatchMode=yes "cluster@$NODE")

probe() {
  local name="$1" max="$2" prompt="$3"
  local body
  body=$(jq -nc --arg p "$prompt" --argjson m "$max" \
    '{model:"glm", prompt:$p, max_tokens:$m, temperature:0, stream:false}')
  local resp text n tps
  resp=$("${SSH[@]}" "curl -sS -m 600 -X POST http://127.0.0.1:$PORT/v1/completions \
    -H 'Content-Type: application/json' -d '$body'")
  text=$(printf '%s' "$resp" | jq -r '.choices[0].text // empty')
  if [ -z "$text" ]; then
    printf '%-14s ERROR %s\n' "$name" "$(printf '%s' "$resp" | head -c 200)"
    return 1
  fi
  n=$(printf '%s' "$resp" | jq -r '.usage.completion_tokens // 0')
  # Server-reported rate: excludes ssh + HTTP round trip, which at 15 tok/s would otherwise
  # cost a whole percent.
  tps=$(printf '%s' "$resp" | jq -r '.usage["response_token/s"] // 0')
  printf '%-14s %s  %3s tok  %6.2f tok/s\n' "$name" \
    "$(printf '%s' "$text" | sha256sum | cut -c1-8)" "$n" "$tps"
}

probe france  32  'The capital of France is'
probe 2plus2  32  '2+2='
probe pyadd   64  'Write a Python function add(a, b) that returns the sum.'
probe counting 32 '1, 2, 3, 4,'
probe open128 128 'Explain how a transformer language model works.'
probe open512 512 'Explain how a transformer language model works.'
