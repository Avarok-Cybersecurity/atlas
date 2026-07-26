#!/bin/bash
# Phase C driver: A/B the multi-sequence batched SSM recurrent decode path.
#
# Root cause this tests: `kernels/gb10/qwen3.6-27b/nvfp4/gated_delta_rule.cu`
# SHADOWS `kernels/gb10/common/gated_delta_rule.cu` (build.rs collect_cu_files
# overrides common by file stem). The shadow was forked before common gained
# the four decode kernels `_f32_norm`, `_f32_conv_norm`, `_f32_strided` and
# `_f32_strided_norm`, so on the 27B `try_kernel` returned 0 for all four and
# `ATLAS_SSM_BATCHED_RECURRENT=1` could never engage — every concurrent decode
# fell back to the per-sequence loop. Proven at runtime: with the pr369 image
# all four log "Optional kernel ... not loaded"; with :msdecode none do.
#
# Both legs run the SAME image (:msdecode, which carries the ported kernels),
# so the ONLY difference is the env flag — the strictest possible A/B, and it
# also proves the fast path actually engages rather than silently falling back.
# Serve geometry matches Phase A/B EXACTLY (bs=16 / nd=3 / slots=32 /
# seq-len 4096 / util 0.70) or the comparison against them means nothing.
set -u
WT=/workspace/.wt-golden
CS=$WT/conc_sweep
RESULTS=$CS/results
STATE=$WT/docs/campaigns/gb10-concurrency-2026-07/STATE.md
BENCH=$WT/bench/bench-atlas-concurrency.py
MODEL=centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf
PORT=8888

note() { echo "- $(date -u +%FT%TZ) $*" >> "$STATE"; echo "STATE: $*"; }
teardown() { sudo docker rm -f atlas-csweep >/dev/null 2>&1; sleep 3; }

run_leg() { # $1 leg name, $2 extra -e args
  local leg="$1" extra="$2"
  if [ -s "$RESULTS/$leg.json" ]; then echo "SKIP $leg (results exist)"; return 0; fi
  teardown
  # shellcheck disable=SC2086
  sudo docker run -d --name atlas-csweep --network host --gpus all --ipc=host \
    -e ATLAS_NO_FFN_NVFP4_MMQ=1 -e ATLAS_SSM_TAIL_MIDCHUNK=0 -e ATLAS_MTP_CATCHUP=0 \
    -e ATLAS_MTP_DRAFT_CONF=0.0 -e ATLAS_MTP_GATE_FORCE=1 -e ATLAS_SSM_TAIL_PROTECT=1 \
    -e ATLAS_SSM_TAIL_LEASE_TTL=128 -e ATLAS_BF16_TC_PREFILL=1 $extra \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
    atlas-gb10:msdecode serve "$MODEL" \
    --host 0.0.0.0 --port $PORT --model-name "$MODEL" \
    --max-seq-len 4096 --max-batch-size 16 --kv-cache-dtype bf16 \
    --gpu-memory-utilization 0.70 \
    --enable-prefix-caching --ssm-cache-slots 32 --ssm-checkpoint-interval 32 \
    --speculative --num-drafts 3 --mtp-quantization bf16 \
    --tool-call-parser qwen3_xml --disable-tool-grammar true --disable-thinking >/dev/null 2>&1
  local ok=0
  for _ in $(seq 1 200); do
    curl -sf -m4 "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q Qwen && { ok=1; break; }
    sudo docker ps --format '{{.Names}}' | grep -q atlas-csweep || break
    sleep 5
  done
  if [ $ok -eq 1 ]; then
    echo "=== LEG $leg: serve up, benching (env: ${extra:-<none>}) ==="
    if BENCH_PORT=$PORT BENCH_MAX_SEQ_LEN=4096 BENCH_RESULTS_FILE="$RESULTS/$leg.json" \
        python3 -u "$BENCH" 2>&1 | tail -20; [ -s "$RESULTS/$leg.json" ]; then
      note "LEG $leg DONE -> results/$leg.json"
    else
      note "LEG $leg BENCH FAILED"
    fi
    # Engagement evidence: the four kernels must NOT be reported missing.
    sudo docker logs atlas-csweep 2>&1 \
      | grep -ac "not loaded.*\(strided\|f32_norm\|conv_norm\)" > "$CS/$leg.missingkernels" || true
  else
    sudo docker logs atlas-csweep 2>&1 | tail -60 > "$CS/$leg.deathlog" || true
    note "LEG $leg SERVE_DIED (deathlog: conc_sweep/$leg.deathlog)"
  fi
  teardown
  echo "LEG_DONE $leg"
}

run_leg atlasC_perseq ""
run_leg atlasC_batched "-e ATLAS_SSM_BATCHED_RECURRENT=1"
note "PHASEC_DONE"
echo "PHASEC_DONE"
