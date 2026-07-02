#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Merged-stack model smoke test — load + coherence-generate each
# stack-relevant model on real weights, in the cuda13.2 GB10 container,
# using the all-targets `spark` binary. Verifies the Holo enablement +
# FP8 loading + dense-FP8 + batched-decode stack actually loads and
# generates coherent text across every code path (MoE NVFP4, MoE FP8
# compressed-tensors, dense FP8, dense NVFP4, dense-small, vision/ViT).
#
# Completely repeatable. From the repo root on a build host:
#
#   # 1. Build the all-targets binary (one binary serves every target):
#   CUTLASS_HOME=/home/ms/cutlass FLASHINFER_HOME=/home/ms/flashinfer \
#   CUDARC_CUDA_VERSION=13000 LIBRARY_PATH=/home/ms/nccl/build/lib:/usr/local/cuda/lib64 \
#   ATLAS_TARGET_HW=gb10 ATLAS_TARGET_MODEL='*' ATLAS_TARGET_QUANT='*' \
#     cargo build --release -p spark-server
#
#   # 2. Deploy + run on the GPU host (binary mounted into the cuda13.2 image):
#   scp target/release/spark $GPU_HOST:~/spark-test
#   scp tests/stack_model_smoke.sh $GPU_HOST:~/
#   ssh $GPU_HOST 'ATLAS_SPARK=~/spark-test ~/stack_model_smoke.sh'
#
# Or, if invoked directly on the GPU host with the binary already present:
#   ATLAS_SPARK=~/spark-test tests/stack_model_smoke.sh
#
# Tunables (env):
#   ATLAS_SPARK    path to the all-targets spark binary           (default ~/spark-test)
#   ATLAS_IMG      cuda13.2 runtime image (CUTLASS/cuBLASLt/NCCL)  (default atlas-holo:cuda13.2)
#   KV             --target-kv-tokens pool size (small-ctx smoke)  (default 100000)
#   OVERCOMMIT     ATLAS_KV_OVERCOMMIT — relax the max-batch-size   (default 1)
#                  vs pool startup check so the lean pool loads
#   PORT           server port                                     (default 8891)
#   LOAD_TIMEOUT   per-model load budget, seconds                  (default 600)
#   HFCACHE        HuggingFace cache dir                           (default ~/.cache/huggingface)
#   MODELS_FILE    optional file: "HF_ID|description" per line, overrides the default set
#   OUT            results markdown path                           (default ~/stack_model_smoke_results.md)
#
# Models load SEQUENTIALLY (one container at a time, stopped before the next),
# so peak GPU = the single largest model + lean KV; with the default 100K-token
# pool each stays well under ~70GB and coexists with other GPU tenants (ComfyUI).
set -uo pipefail

IMG="${ATLAS_IMG:-atlas-holo:cuda13.2}"
SPARK="${ATLAS_SPARK:-$HOME/spark-test}"
HFCACHE="${HFCACHE:-$HOME/.cache/huggingface}"
PORT="${PORT:-8891}"
KV="${KV:-100000}"
# Smoke prompts are small-context, so a 100K-token pool is ample. Overcommit
# (ATLAS_KV_OVERCOMMIT=1) relaxes the worst-case "pool must hold max-batch-size
# sequences at max-seq-len" startup check, so the lean pool serves the default
# batch size — safe here since the test issues one short request at a time.
OVERCOMMIT="${OVERCOMMIT:-1}"
LOAD_TIMEOUT="${LOAD_TIMEOUT:-600}"
OUT="${OUT:-$HOME/stack_model_smoke_results.md}"

# Default stack-relevant set — each entry exercises a distinct stack path.
# Override with MODELS_FILE for a different fleet.
default_models=(
  "Hcompany/Holo-3.1-35B-A3B-NVFP4|Holo MoE NVFP4 (#203)"
  "Hcompany/Holo-3.1-35B-A3B-FP8|Holo MoE FP8 compressed-tensors (#210)"
  "deepreinforce-ai/Ornith-1.0-35B-FP8|dense FP8 attn/FFN (#215)"
  "Kbenkhaled/Qwen3.5-27B-NVFP4|dense NVFP4 / #211 dequant"
  "empero-ai/Qwythos-9B-Claude-Mythos-5-1M|dense small (ornith-1.0-9b)"
  "ig1/Qwen3-VL-30B-A3B-Instruct-NVFP4|vision/ViT MoE NVFP4"
)
if [[ -n "${MODELS_FILE:-}" && -f "${MODELS_FILE}" ]]; then
  mapfile -t models < "$MODELS_FILE"
else
  models=("${default_models[@]}")
fi

[[ -x "$SPARK" ]] || { echo "ERROR: spark binary not found/executable at $SPARK" >&2; exit 2; }
command -v docker >/dev/null || { echo "ERROR: docker not on PATH" >&2; exit 2; }

{
  echo "# Merged-stack model smoke test"
  echo ""
  echo "binary: \`$SPARK\` ($(stat -c%s "$SPARK" 2>/dev/null) bytes)  image: \`$IMG\`  KV target: $KV"
  echo ""
  echo "| Model | Stack path | Load | Generate | Sample / Error |"
  echo "|---|---|---|---|---|"
} > "$OUT"

overall=0
for entry in "${models[@]}"; do
  [[ -z "$entry" || "$entry" == \#* ]] && continue
  model="${entry%%|*}"; path="${entry##*|}"
  name="t6-$(echo "$model" | tr '/' '-' | tr '[:upper:]' '[:lower:]' | cut -c1-50)"
  echo "════════════════════════════════════════════════════════════"
  echo ">>> $model  ($path)"
  docker rm -f "$name" >/dev/null 2>&1

  docker run -d --name "$name" --gpus all --ipc host \
    -v "$SPARK":/usr/local/bin/spark:ro \
    -v "$HFCACHE":/root/.cache/huggingface \
    -e HF_HUB_OFFLINE=1 -e RUST_LOG=info -e ATLAS_KV_OVERCOMMIT="$OVERCOMMIT" \
    -p 127.0.0.1:"$PORT":"$PORT" \
    "$IMG" serve "$model" --port "$PORT" --bind 0.0.0.0 --target-kv-tokens "$KV" >/dev/null 2>&1

  load="FAIL"; gen="—"; sample=""
  t0=$SECONDS
  while (( SECONDS - t0 < LOAD_TIMEOUT )); do
    if ! docker ps -q -f name="^${name}$" | grep -q .; then
      sample="container exited: $(docker logs --tail 4 "$name" 2>&1 | tr '\n' ' ' | tr '|' '/' | cut -c1-180)"
      break
    fi
    if curl -s -m 3 "http://127.0.0.1:$PORT/v1/models" 2>/dev/null | grep -q '"id"'; then
      load="OK ($(( SECONDS - t0 ))s)"
      break
    fi
    sleep 3
  done

  if [[ "$load" == OK* ]]; then
    resp=$(curl -s -m 120 "http://127.0.0.1:$PORT/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d "{\"model\":\"$model\",\"messages\":[{\"role\":\"user\",\"content\":\"In two sentences, explain why the sky is blue.\"}],\"max_tokens\":256,\"temperature\":0.7}" 2>/dev/null)
    sample=$(echo "$resp" | python3 -c 'import sys,json
try:
    d=json.load(sys.stdin); c=d["choices"][0]["message"]["content"].strip()
    print(c[:180] if c else "__EMPTY__")
except Exception as e:
    print("__PARSE_ERR__: "+str(e)[:90])' 2>/dev/null)
    if [[ "$sample" == __EMPTY__ || "$sample" == __PARSE_ERR__* || -z "$sample" ]]; then
      gen="FAIL"; overall=1
    else
      gen="OK"
    fi
  else
    overall=1
  fi

  echo "    load=$load  gen=$gen"
  echo "    sample: $sample"
  echo "| \`$model\` | $path | $load | $gen | $(echo "$sample" | tr '|' '/' | cut -c1-130) |" >> "$OUT"
  docker rm -f "$name" >/dev/null 2>&1
  sleep 3
done

echo "════════════════════════════════════════════════════════════"
echo "DONE (exit $overall). Results → $OUT"
cat "$OUT"
exit "$overall"
