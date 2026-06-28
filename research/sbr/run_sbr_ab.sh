#!/usr/bin/env bash
# SBR M1 A/B benchmark orchestration — run ON dgx2.
# For each config (baseline tail-pin OFF, fix tail-pin ON) start Atlas with the
# SBR binary bind-mounted into the runtime image, run the warm-hit TTFT harness,
# capture server logs (replay distances), and stop.
set -uo pipefail

IMAGE="${IMAGE:-atlas-gb10:ornith-perf}"
BIN="${BIN:-$HOME/sbr_bench/spark}"
MODEL="${MODEL:-nvidia/Qwen3-Next-80B-A3B-Instruct-NVFP4}"
OUT="$HOME/sbr_bench"
TURNS="${TURNS:-16}"
DISTRACTORS="${DISTRACTORS:-24}"
BASE_TOKENS="${BASE_TOKENS:-8000}"
MAXLEN="${MAXLEN:-24576}"
SLOTS="${SLOTS:-16}"

run_one() {
  local label="$1" pin="$2"
  echo "=== [$label] ATLAS_SBR_TAIL_PIN=$pin ==="
  sudo docker rm -f sbr-srv >/dev/null 2>&1
  sudo docker run -d --name sbr-srv --gpus all --ipc=host --network host \
    -e ATLAS_SBR_TAIL_PIN="$pin" -e ATLAS_SNAP_LOOKUP_DBG=1 -e RUST_LOG=info \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
    -v "$BIN:/usr/local/bin/spark:ro" \
    "$IMAGE" serve "$MODEL" \
      --port 8888 --host 0.0.0.0 --max-seq-len "$MAXLEN" --ssm-cache-slots "$SLOTS" \
      >/dev/null 2>&1

  echo "[$label] waiting for server ready (model load ~6min)..."
  local ready=0
  for i in $(seq 1 180); do
    if curl -sf http://localhost:8888/v1/models >/dev/null 2>&1; then ready=1; break; fi
    if ! sudo docker ps --format '{{.Names}}' | grep -q sbr-srv; then
      echo "[$label] container died during load"; sudo docker logs sbr-srv 2>&1 | tail -30; return 1
    fi
    sleep 5
  done
  if [ "$ready" -ne 1 ]; then echo "[$label] server never ready"; sudo docker logs sbr-srv 2>&1 | tail -30; sudo docker rm -f sbr-srv >/dev/null 2>&1; return 1; fi
  echo "[$label] server ready."

  python3 "$OUT/sbr_bench.py" --label "$label" --out "$OUT/$label.json" \
    --model "$MODEL" --turns "$TURNS" --distractors "$DISTRACTORS" \
    --base-tokens "$BASE_TOKENS" --distractor-tokens 400 --resp-tokens 200

  sudo docker logs sbr-srv > "$OUT/$label.serverlog" 2>&1
  grep -E "Marconi intermediate hit|snap-lookup" "$OUT/$label.serverlog" | tail -40 > "$OUT/$label.replaylog" 2>/dev/null
  sudo docker rm -f sbr-srv >/dev/null 2>&1
  echo "[$label] done -> $OUT/$label.json"
}

run_one baseline_tailpin_off 0
run_one sbr_tailpin_on 1

echo "=== A/B SUMMARY ==="
python3 - "$OUT/baseline_tailpin_off.json" "$OUT/sbr_tailpin_on.json" <<'PY'
import json, sys
for p in sys.argv[1:]:
    try:
        d = json.load(open(p))
    except Exception as e:
        print(f"{p}: MISSING ({e})"); continue
    t = [r["ttft_s"] for r in d["rows"] if r.get("ttft_s")]
    if not t: print(f"{d['label']}: no ttft"); continue
    print(f"{d['label']:24s} turns={d['turns']} TTFT first={t[0]:.3f}s last={t[-1]:.3f}s "
          f"max={max(t):.3f}s mean={sum(t)/len(t):.3f}s")
PY
