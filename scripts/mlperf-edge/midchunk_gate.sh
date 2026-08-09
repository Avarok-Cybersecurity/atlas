#!/bin/bash
# Decisive gate for re-enabling ATLAS_SSM_TAIL_MIDCHUNK on the GB10 golden config.
#
# WHY A SECOND ROUND
# The first A/B (ab_midchunk.sh) produced a split verdict that neither leg's
# numbers can settle on their own:
#   * MONOTONIC conversation (warm_tpot_probe, the representative pattern):
#     midchunk ON is BETTER — mean warm TTFT 991 -> 921 ms (-7.1%) and, more
#     tellingly, it converges to a stable ~882 ms floor from turn 4 while OFF
#     oscillates 869-1135 ms. That is the signature of hitting the tail
#     checkpoint every turn instead of intermittently falling further back.
#   * REPLAY probe (warm_replay_probe): midchunk ON is WORSE, 0.76-0.77x at the
#     larger deltas. But that probe alternates back to BASE between reps and
#     there is only ONE tail slot per session (`reserve_tail_slot`), so the
#     alternation thrashes it in a way a forward-only conversation never does.
#     Treated as a probe artifact, not evidence.
#   * TPOT looked +2.2% worse, but the two legs emitted different token counts
#     (286/293/239/... vs 229/210/215/...), so it is trajectory-confounded and
#     is NOT quotable. Spec-decode output is trajectory-dependent by design.
# All of that was N=1 per leg. This script fixes both defects: N reps of the
# monotonic probe, and the accuracy gate that actually matters.
#
# WHY ACCURACY IS THE GATE, NOT LATENCY
# midchunk was disabled on 2026-07-16 for CORRUPTING CROSS-REQUEST SSM prefix
# reuse: BFCL single-turn samples share a system-prompt prefix and reused each
# other's tail snapshot, whose state bleeds past the advertised prefix boundary
# -> garbled tool calls -> wrong scores (32 slots: 77.31 normalized ON vs 84.54
# OFF). The prescribed fix (session-gating `is_tail` to the same NON-ZERO
# session) is now present verbatim in radix_tree/snapshot.rs::lookup, which is
# why this is worth retesting at all. But the ONLY way to know the fix holds is
# to run the regime that broke: many independent single-turn requests sharing a
# prefix. A latency probe cannot see this failure mode at all.
#
# The BFCL subset keeps the golden 62/10/10 category MIX (do-not-change) and
# scales all three down by the same factor, so the subset stays representative
# rather than reweighted.
#
# Usage: midchunk_gate.sh <atlas_bin> <outdir> [reps] [pct_scale]
set -u
BIN="${1:?path to the built spark binary}"
OUT="${2:?output dir}"
REPS="${3:-3}"
SCALE="${4:-4}"        # divide each category pct by this -> ~1/4 of the 995
HERE="$(cd "$(dirname "$0")" && pwd)"
HARNESS=/workspace/endpoints-fresh
MODEL=centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf
PORT=8888
mkdir -p "$OUT"

for leg in mc_off mc_on; do
  case $leg in
    mc_off) MC="-e ATLAS_SSM_TAIL_MIDCHUNK=0" ;;   # today's frozen config
    mc_on)  MC="" ;;                               # the code's actual default
  esac
  sudo docker rm -f atlas-mcg >/dev/null 2>&1; sleep 3
  # shellcheck disable=SC2086
  sudo docker run -d --name atlas-mcg --network host --gpus all --ipc=host \
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
    sudo docker ps --format '{{.Names}}' | grep -q atlas-mcg || { echo "SERVE_DIED leg=$leg"; break; }
    sleep 5
  done
  [ $ok -eq 1 ] || { sudo docker logs atlas-mcg 2>&1 | tail -40 > "$OUT/$leg.died.txt"; continue; }
  echo "=== leg=$leg serve up (midchunk: ${MC:-<absent => ON>}) ==="

  # N reps of the MONOTONIC probe (forward-only conversation, the real pattern).
  for r in $(seq 1 "$REPS"); do
    python3 -u "$HERE/warm_tpot_probe.py" $PORT "${leg}_r${r}" \
        "$OUT/${leg}.tpot.r${r}.json" --turns 8 2>&1 | tee "$OUT/${leg}.tpot.r${r}.log"
  done

  # ACCURACY GATE — the regime midchunk actually broke.
  RD="results/mcgate_${leg}_$(date +%H%M%S)"
  CFG="$(mktemp -t mcgate_XXXX.yaml)"
  python3 - "$RD" "$CFG" "$HARNESS/results/defaults_20260721_173342/config.yaml" "$PORT" "$SCALE" <<'PY'
import sys, yaml
rd, cfg, base, port, scale = sys.argv[1:6]
scale = int(scale)
c = yaml.safe_load(open(base))
c["report_dir"] = rd
c.setdefault("endpoint_config", {})["endpoints"] = [f"http://localhost:{port}"]
# Keep ONLY the bfcl accuracy dataset: the perf trajectories are measured by the
# monotonic probe above and would cost hours here for no extra signal.
c["datasets"] = [d for d in c["datasets"] if d.get("type") == "accuracy"]
for d in c["datasets"]:
    gp = d.setdefault("generate_params", {})
    pct = gp.get("category_sample_pct") or {}
    # Scale every category by the SAME factor so the golden 62/10/10 mix is
    # preserved. Reweighting the mix would make the subset non-comparable to the
    # full run and to the 83.64 / 85.32 floors.
    gp["category_sample_pct"] = {k: max(1, v // scale) for k, v in pct.items()}
r = c.setdefault("settings", {}).setdefault("runtime", {})
r["min_duration_ms"] = 0
yaml.safe_dump(c, open(cfg, "w"), sort_keys=False)
print("accuracy subset ->", rd, gp.get("category_sample_pct"))
PY
  # `--mode acc` — the choices are perf|acc|both. `--mode accuracy` is rejected
  # with a bare "Required: --mode", which reads like a MISSING argument rather
  # than an invalid value, so a failed accuracy leg is easy to miss in a log.
  ( cd "$HARNESS" && ./.venv/bin/inference-endpoint benchmark from-config -c "$CFG" --mode acc -v ) \
      2>&1 | tail -40 | tee "$OUT/$leg.bfcl.log"
  cp "$HARNESS/$RD/report.txt" "$OUT/$leg.bfcl.report.txt" 2>/dev/null

  sudo docker logs atlas-mcg 2>&1 | grep -c -aiE 'midchunk' > "$OUT/$leg.midchunk_hits.txt" || true
  sudo docker rm -f atlas-mcg >/dev/null 2>&1
done

echo "=== MIDCHUNK GATE SUMMARY ==="
for leg in mc_off mc_on; do
  echo "--- $leg  (midchunk log hits: $(cat "$OUT/$leg.midchunk_hits.txt" 2>/dev/null))"
  grep -E 'Average:|bfcl_v4' "$OUT/$leg.bfcl.report.txt" 2>/dev/null | head -3
done
python3 "$HERE/midchunk_summarize.py" "$OUT" | tee "$OUT/VERDICT.txt"
echo "MIDCHUNK_GATE_DONE out=$OUT"
