#!/usr/bin/env bash
# Phase 2c KV-precision sweep runner.
#
# Usage: ./runner.sh <config_name> <kv_cache_dtype> <kv_high_precision_layers> <fp8_kv_calibration_tokens>
# Example: ./runner.sh turbo8 turbo8 0 0
#
# Output:
#   /workspace/atlas-dumps/numdrift/phase2c-<config_name>/atlas_L*.bin
#   /workspace/atlas-dumps/numdrift/phase2c-<config_name>/atlas_final_norm.bin
#   /workspace/atlas-dumps/numdrift/phase2c-<config_name>/atlas_logits.bin
#   /workspace/atlas-dumps/numdrift/phase2c-<config_name>/cosine.txt   <- summary table
#
# Run sequentially per machine. Multi-config parallelism is across dgx1+dgx2.

set -euo pipefail

CONFIG_NAME="${1:?missing config name}"
KV_DTYPE="${2:?missing kv_cache_dtype}"
KV_HP="${3:-0}"
KV_CALIB="${4:-0}"

OUT_DIR="/workspace/atlas-dumps/numdrift/phase2c-${CONFIG_NAME}"
PROBE="/workspace/atlas-dumps/numdrift/atlas_turn11_probe.json"
IMAGE="atlas-gb10:realfix2"

mkdir -p "$OUT_DIR"

echo "=== [${CONFIG_NAME}] config: dtype=${KV_DTYPE} hp=${KV_HP} calib=${KV_CALIB} ==="
echo "=== [${CONFIG_NAME}] start: $(date -u +%H:%M:%S) ==="

# Stop+rm any existing atlas-qwen
sudo docker stop atlas-qwen 2>/dev/null || true
sudo docker rm atlas-qwen 2>/dev/null || true

# Bounce with the target KV config + ATLAS_NEMO_DUMP env
sudo docker run -d --name atlas-qwen \
  --network host --gpus all --ipc=host \
  -e RUST_LOG=info \
  -e ATLAS_NEMO_DUMP="$OUT_DIR" \
  -v /workspace/.cache/huggingface:/root/.cache/huggingface \
  -v /workspace/atlas-dumps:/workspace/atlas-dumps \
  "$IMAGE" \
  serve Qwen/Qwen3.6-35B-A3B-FP8 \
    --port 8888 --max-seq-len 65536 --max-batch-size 8 \
    --gpu-memory-utilization 0.88 \
    --kv-cache-dtype "$KV_DTYPE" \
    --kv-high-precision-layers "$KV_HP" \
    --fp8-kv-calibration-tokens "$KV_CALIB" \
    --scheduling-policy slai --enable-prefix-caching \
    --speculative --mtp-quantization bf16 \
    --dump "$OUT_DIR/dump.jsonl" \
    >/dev/null 2>&1

# Wait for server ready
for i in {1..120}; do
    if sudo docker logs atlas-qwen 2>&1 | grep -q "Listening on 127.0.0.1:8888"; then
        echo "=== [${CONFIG_NAME}] ready in ${i}s ==="
        break
    fi
    if sudo docker logs atlas-qwen 2>&1 | grep -qE "panic|FATAL|Error:"; then
        echo "=== [${CONFIG_NAME}] BOOT FAILED ==="
        sudo docker logs atlas-qwen 2>&1 | tail -20 > "$OUT_DIR/boot-failure.log"
        exit 1
    fi
    sleep 1
done

# Fire the probe (18920-token agentic prompt)
curl -s -m 600 -X POST http://localhost:8888/v1/chat/completions \
    -H "Content-Type: application/json" \
    --data-binary @"$PROBE" \
    -o "$OUT_DIR/response.jsonl" \
    || { echo "[${CONFIG_NAME}] probe request failed"; exit 1; }

# Verify dump completed (40 layers)
N_LAYERS=$(ls "$OUT_DIR"/atlas_L*.bin 2>/dev/null | wc -l)
if [[ "$N_LAYERS" != "40" ]]; then
    echo "=== [${CONFIG_NAME}] WARN: only ${N_LAYERS}/40 layers dumped ==="
fi

# Run cosine compare against HF reference at /workspace/atlas-dumps/numdrift/hf_*.bin
python3 - <<'PY' "$OUT_DIR" "$CONFIG_NAME" > "$OUT_DIR/cosine.txt"
import sys, pathlib, numpy as np
out_dir = pathlib.Path(sys.argv[1])
cfg = sys.argv[2]
hf = pathlib.Path("/workspace/atlas-dumps/numdrift")

def load(p):
    return np.frombuffer(p.read_bytes(), dtype="<f4")

def cos(a, b):
    a = a.astype(np.float64); b = b.astype(np.float64)
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))

coses = []
for i in range(40):
    ap = out_dir / f"atlas_L{i}.bin"
    hp = hf / f"hf_L{i}.bin"
    if ap.exists() and hp.exists():
        coses.append(cos(load(ap), load(hp)))
    else:
        coses.append(float("nan"))

fn = float("nan")
if (out_dir / "atlas_final_norm.bin").exists() and (hf / "hf_final_norm.bin").exists():
    fn = cos(load(out_dir / "atlas_final_norm.bin"), load(hf / "hf_final_norm.bin"))

print(f"{'config':25} {'mean':>7} {'min':>7} {'L0':>7} {'L20':>7} {'L31':>7} {'L35':>7} {'L37':>7} {'L39':>7} {'final':>7}")
mean = float(np.nanmean(coses))
mn = float(np.nanmin(coses))
print(f"{cfg:25} {mean:7.4f} {mn:7.4f} {coses[0]:7.4f} {coses[20]:7.4f} {coses[31]:7.4f} {coses[35]:7.4f} {coses[37]:7.4f} {coses[39]:7.4f} {fn:7.4f}")

# Also raw table
print()
print("Layer | cos")
for i, c in enumerate(coses):
    print(f"L{i:02} | {c:.4f}")
PY

echo "=== [${CONFIG_NAME}] DONE: $(date -u +%H:%M:%S) ==="
cat "$OUT_DIR/cosine.txt" | head -3
