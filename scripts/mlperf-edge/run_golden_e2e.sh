#!/bin/bash
# Golden end-to-end run for the MLCommons edge-agentic harness on GB10 (DGX Spark).
#
# This is the ONE reproduce entry point for the numbers in
# docs/campaigns/gb10-decode-fold-2026-07/. It serves Atlas with the frozen
# "c2final" configuration and runs both harness phases (1007 perf + 995 BFCL).
#
# The serve flags and env below are the frozen submission config -- pinned values,
# not tunables. Changing any of them invalidates comparison with the recorded results.
#
# Required:
#   ATLAS_BIN     path to a `spark` built with ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL=qwen3.6-27b
#   HARNESS_DIR   path to the inference-endpoint (MLCommons edge-agentic) checkout
#   BASE_CONFIG   path to the harness config.yaml to derive from
# Optional:
#   EXTRA_ENV     additional `-e VAR=VAL` docker args, for A/B-ing ONE candidate
#                 flag against the frozen config. Leave UNSET to reproduce the
#                 submitted numbers -- anything passed here makes the run a
#                 candidate, not a reproduction, so record it with the result.
#   ND            speculative draft count; verify width K = ND + 1.
#                 Default 3 => K=4, the width selected by the K-ladder.
#   TAG           report-dir prefix. Default "golden".
#   IMAGE         serve container. Default atlas-gb10:followups.
#   PORT          serve port. Default 8888.
#
# Example:
#   ATLAS_BIN=$PWD/target/release/spark \
#   HARNESS_DIR=/workspace/endpoints-fresh \
#   BASE_CONFIG=/workspace/endpoints-fresh/results/defaults_20260721_173342/config.yaml \
#     bash scripts/mlperf-edge/run_golden_e2e.sh
set -euo pipefail

ATLAS_BIN="${ATLAS_BIN:?path to the built spark binary}"
HARNESS_DIR="${HARNESS_DIR:?path to the inference-endpoint checkout}"
BASE_CONFIG="${BASE_CONFIG:?path to the harness base config.yaml}"
ND="${ND:-3}"
TAG="${TAG:-golden}"
IMAGE="${IMAGE:-atlas-gb10:followups}"
PORT="${PORT:-8888}"
# SSM snapshot pool size. 128 is the submitted golden value. Slots cost ~151.5 MB
# each and come straight out of the KV budget (128 -> 20.2 GB KV, 192 -> 10.8 GB),
# so this is a genuine tradeoff, not a free dial -- see SSM_SLOTS_AB.md.
SLOTS="${SLOTS:-128}"
# Extra `-e VAR=VAL` docker args for validating a candidate flag against the frozen
# config. Empty by default, so the default invocation stays byte-identical to the
# submitted run. Anything passed here is by definition NOT part of the frozen
# config and must be stated whenever the resulting numbers are quoted.
EXTRA_ENV="${EXTRA_ENV:-}"
MODEL="centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf"
CONTAINER="atlas-${TAG}-e2e"

[ -x "$ATLAS_BIN" ] || { echo "ATLAS_BIN is not executable: $ATLAS_BIN" >&2; exit 1; }
[ -r "$BASE_CONFIG" ] || { echo "BASE_CONFIG is not readable: $BASE_CONFIG" >&2; exit 1; }

cd "$HARNESS_DIR"
TS=$(date +%Y%m%d_%H%M%S)
RD="results/${TAG}_e2e_${TS}"
CFG="$(mktemp -t "${TAG}_e2e_XXXX.yaml")"
python3 - "$RD" "$CFG" "$BASE_CONFIG" "$PORT" <<'PY'
import sys, yaml
rd, cfg, base, port = sys.argv[1:5]
c = yaml.safe_load(open(base))
c["report_dir"] = rd
c.setdefault("endpoint_config", {})["endpoints"] = [f"http://localhost:{port}"]

# MLPerf submission-checker runtime lock. These are NOT free parameters: the
# official checker compares them against the loadgen constants and rejects the
# submission outright if they differ ("sample_index_rng_seed is wrong,
# expected=2747215439041700203, found=42"), and it rejects a run whose
# min_duration is 0 ("Test duration less than 600s in user config").
#
# The harness's own default template ships 42/42/0, which passes the endpoints
# `check_compliance.py` but FAILS the official submission checker -- the two
# rulesets disagree, and the official one is the gate for an actual submission.
#
# The seeds pick WHICH samples are drawn and in WHAT order, so they cannot be
# edited into an already-recorded config after the fact: doing that makes the
# artifact describe a run that never happened. Set them here, before the run.
runtime = c.setdefault("settings", {}).setdefault("runtime", {})
runtime["min_duration_ms"] = 600_000
runtime["max_duration_ms"] = 14_400_000
runtime["scheduler_random_seed"] = 16159082839903944936
runtime["dataloader_random_seed"] = 2747215439041700203

yaml.safe_dump(c, open(cfg, "w"), sort_keys=False)
print("e2e config ->", rd)
print("runtime lock ->", {k: runtime[k] for k in sorted(runtime)})
PY

sudo docker rm -f "$CONTAINER" >/dev/null 2>&1 || true
sleep 3
# Frozen c2final env. NOTE on ATLAS_* presence-flags: for those, a value of 0 is
# NOT "off" -- the flag must be absent. The ones meant to be off are simply not set.
sudo docker run -d --name "$CONTAINER" --network host --gpus all --ipc=host \
  -e ATLAS_NO_FFN_NVFP4_MMQ=1 -e ATLAS_SSM_TAIL_MIDCHUNK=0 -e ATLAS_MTP_CATCHUP=0 \
  -e ATLAS_MTP_DRAFT_CONF=0.0 -e ATLAS_MTP_GATE_FORCE=1 \
  -e ATLAS_SSM_TAIL_LEASE_TTL=128 -e ATLAS_BF16_TC_PREFILL=1 $EXTRA_ENV \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
  -v "$ATLAS_BIN:/usr/local/bin/spark:ro" \
  "$IMAGE" serve "$MODEL" \
  --host 0.0.0.0 --port "$PORT" --model-name "$MODEL" \
  --max-seq-len 32768 --max-batch-size 1 --kv-cache-dtype bf16 --gpu-memory-utilization 0.70 \
  --enable-prefix-caching --ssm-cache-slots "$SLOTS" --ssm-checkpoint-interval 32 \
  --speculative --num-drafts "$ND" --mtp-quantization bf16 \
  --tool-call-parser qwen3_xml --disable-tool-grammar true --disable-thinking >/dev/null

for _ in $(seq 1 180); do
  curl -sf -m4 "http://localhost:${PORT}/v1/models" 2>/dev/null | grep -q Qwen && break
  sudo docker ps --format '{{.Names}}' | grep -q "$CONTAINER" || { echo "SERVE_DIED"; exit 1; }
  sleep 5
done
echo "serve up (nd=${ND}, K=$((ND + 1)), slots=${SLOTS}, bin=${ATLAS_BIN})"

# Gate C2 runs FIRST: an NVFP4 build can pass the correctness gates while emitting
# garbage, so coherence + a real tool call are checked before committing to the
# multi-hour benchmark.
echo "--- gate C2: coherence ---"
curl -sf -m60 "http://localhost:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"Write a Python function that returns the nth Fibonacci number iteratively.\"}],\"max_tokens\":150,\"temperature\":0,\"seed\":42}" \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["choices"][0]["message"]["content"])'
echo "--- gate C2: tool call ---"
curl -sf -m60 "http://localhost:${PORT}/v1/chat/completions" -H 'Content-Type: application/json' \
  -d "{\"model\":\"${MODEL}\",\"messages\":[{\"role\":\"user\",\"content\":\"What is the weather in Paris? Use the get_weather tool.\"}],\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"description\":\"Get current weather for a location\",\"parameters\":{\"type\":\"object\",\"properties\":{\"location\":{\"type\":\"string\"}},\"required\":[\"location\"]}}}],\"max_tokens\":120,\"temperature\":0,\"seed\":42}" \
  | python3 -c 'import sys,json; print(json.dumps(json.load(sys.stdin)["choices"][0]["message"].get("tool_calls")))'
echo "--- gate C2 output above must be coherent code + a get_weather call on Paris ---"

./.venv/bin/inference-endpoint benchmark from-config -c "$CFG" --mode both -v
echo "GOLDEN_E2E_DONE rd=${HARNESS_DIR}/${RD}"
