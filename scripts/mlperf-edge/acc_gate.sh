#!/bin/bash
# Generic BFCL accuracy gate for a single env-flag candidate.
#
# Latency A/Bs cannot clear a flag for folding when the flag can change emitted
# tokens. ATLAS_GDN_REGRESIDENT is the live case: it is advertised token-equal to
# WY4 (cos 1.0, max|dH| ~1e-8) and DID match on the three shorter replay cells,
# but DIFFERED on the longest (4320-char delta). Over a ~1200-token recurrence a
# different accumulation order can tip razor-margin greedy tokens even at cos 1.0,
# so "same acceptance class" is not the same as "output-neutral" and the fold
# needs accuracy, not just speed.
#
# The subset scales all three golden BFCL categories by the SAME factor so the
# 62/10/10 mix (do-not-change) stays representative and remains comparable to the
# 83.64 / 85.32 floors. Reweighting would make the number meaningless.
#
# TRAP: `--mode` takes perf|acc|both. `--mode accuracy` is rejected with a bare
# "Required: --mode", which reads like a MISSING argument, so a mistyped value
# fails the accuracy leg silently while everything else completes.
#
# Usage: acc_gate.sh <atlas_bin> <outdir> <flag_env_or_NONE> [pct_scale]
#   e.g. acc_gate.sh .../spark out/ ATLAS_NO_GDN_REGRESIDENT=1 4
#
# NOTE on that example: the regresident lever is default-ON since PR #369 and
# only the NEGATIVE spelling is read, so the candidate leg is the one that
# switches the lever OFF and the accuracy question runs backwards — the gate is
# "does removing it change BFCL", not "does adding it". A positive
# `ATLAS_GDN_REGRESIDENT=1` (which this example used to pass) is read by nothing
# and would have scored the default against itself.
set -u
BIN="${1:?path to the built spark binary}"
OUT="${2:?output dir}"
FLAG="${3:?env assignment, or NONE for the control leg}"
SCALE="${4:-4}"
HARNESS=/workspace/endpoints-fresh
MODEL=centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf
PORT=8888
mkdir -p "$OUT"

for leg in control cand; do
  EXTRA=""
  [ "$leg" = cand ] && [ "$FLAG" != NONE ] && EXTRA="-e $FLAG"
  sudo docker rm -f atlas-acc >/dev/null 2>&1; sleep 3
  # shellcheck disable=SC2086
  sudo docker run -d --name atlas-acc --network host --gpus all --ipc=host \
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
    sudo docker ps --format '{{.Names}}' | grep -q atlas-acc || { echo "SERVE_DIED leg=$leg"; break; }
    sleep 5
  done
  [ $ok -eq 1 ] || { sudo docker logs atlas-acc 2>&1 | tail -40 > "$OUT/$leg.died.txt"; continue; }
  echo "=== leg=$leg serve up (${EXTRA:-<control>}) ==="

  RD="results/accgate_${leg}_$(date +%H%M%S)"
  CFG="$(mktemp -t accgate_XXXX.yaml)"
  python3 - "$RD" "$CFG" "$HARNESS/results/defaults_20260721_173342/config.yaml" "$PORT" "$SCALE" <<'PY'
import sys, yaml
rd, cfg, base, port, scale = sys.argv[1:6]
scale = int(scale)
c = yaml.safe_load(open(base))
c["report_dir"] = rd
c.setdefault("endpoint_config", {})["endpoints"] = [f"http://localhost:{port}"]
# Do NOT drop the performance dataset. `--mode acc` already skips the perf phase,
# and removing it fails config validation outright:
#   "load_pattern.type=agentic_inference requires the performance dataset to have
#    agentic_inference config"
# which kills the run before a single request is issued (banner files come back
# empty, which is the tell).
for d in c["datasets"]:
    if d.get("type") != "accuracy":
        continue
    gp = d.setdefault("generate_params", {})
    pct = gp.get("category_sample_pct") or {}
    gp["category_sample_pct"] = {k: max(1, v // scale) for k, v in pct.items()}
    print("subset pct ->", gp["category_sample_pct"])
c.setdefault("settings", {}).setdefault("runtime", {})["min_duration_ms"] = 0
yaml.safe_dump(c, open(cfg, "w"), sort_keys=False)
PY
  ( cd "$HARNESS" && ./.venv/bin/inference-endpoint benchmark from-config -c "$CFG" --mode acc -v ) \
      2>&1 | tail -30 | tee "$OUT/$leg.bfcl.log"
  cp "$HARNESS/$RD/report.txt" "$OUT/$leg.report.txt" 2>/dev/null

  # Banner: proof the flag engaged. Fires on first prefill/replay, not at startup.
  sudo docker logs atlas-acc 2>&1 \
    | grep -aE 'GDN prefill: (FLA chunked|REGISTER-RESIDENT)' | sort -u | tee "$OUT/$leg.banner.txt"
  sudo docker rm -f atlas-acc >/dev/null 2>&1
done

echo "=== ACCURACY GATE: $FLAG ==="
for leg in control cand; do
  echo "--- $leg"
  grep -E 'Average:|bfcl_v4::function_calling' "$OUT/$leg.report.txt" 2>/dev/null | head -2
done
echo "MLPerf floors: 83.64 raw / 85.32 normalized. Candidate must hold at or above the"
echo "control, not merely above the floor -- the subset is smaller than the 995 the"
echo "floors were set on, so a near-floor reading here is not a pass."
echo "ACC_GATE_DONE out=$OUT"
