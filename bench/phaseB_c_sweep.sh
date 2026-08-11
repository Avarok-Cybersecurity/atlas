#!/bin/bash
# Phase B driver: A/B the structural concurrency fixes against the Phase A
# baseline, same box, same bench, same C=[1,2,4,8,16].
#
# Two legs isolate CODE from GRAPHS so the attribution is clean:
#   atlasB_nographs — the Phase B binary (SSOT ladder w/ 12+16, can_mix fusion
#                     fix, vectorized non-MTP sampler picks, bounded sends,
#                     detached terminal frames), graphs still OFF.
#                     vs atlas_synth (Phase A binary): the pure code delta.
#   atlasB_graphs   — same binary + ATLAS_DECODE_GRAPHS_MULTISEQ=1 (strict
#                     =="1"): the n>=2 CUDA-graph lever on top.
#                     vs atlasB_nographs: the pure graphs delta.
#
# Same survivability contract as Phase A: setsid-detached, idempotent legs
# (existing results json => skip), STATE.md appended per leg, deathlog captured
# on serve failure. Serve geometry matches the Phase A atlas leg EXACTLY
# (bs=16 / nd=3 / slots=32 / seq-len 4096 / util 0.70) — only the binary and
# the graphs env differ, or the comparison means nothing.
set -u
WT=/workspace/.wt-golden
CS=$WT/conc_sweep
RESULTS=$CS/results
STATE=$WT/docs/campaigns/gb10-concurrency-2026-07/STATE.md
BENCH=$WT/bench/bench-atlas-concurrency.py
BIN=$CS/spark_phaseB
MODEL=centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf
PORT=8888

note() { echo "- $(date -u +%FT%TZ) $*" >> "$STATE"; echo "STATE: $*"; }
teardown() { sudo docker rm -f atlas-bsweep >/dev/null 2>&1; sleep 3; }

run_leg() { # $1 leg name, $2 extra -e args
  local leg="$1" extra="$2"
  if [ -s "$RESULTS/$leg.json" ]; then echo "SKIP $leg (results exist)"; return 0; fi
  teardown
  # shellcheck disable=SC2086
  sudo docker run -d --name atlas-bsweep --network host --gpus all --ipc=host \
    -e ATLAS_NO_FFN_NVFP4_MMQ=1 -e ATLAS_SSM_TAIL_MIDCHUNK=0 -e ATLAS_MTP_CATCHUP=0 \
    -e ATLAS_MTP_DRAFT_CONF=0.0 -e ATLAS_MTP_GATE_FORCE=1 \
    -e ATLAS_SSM_TAIL_LEASE_TTL=128 -e ATLAS_BF16_TC_PREFILL=1 $extra \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
    -v "$BIN:/usr/local/bin/spark:ro" \
    atlas-gb10:followups serve "$MODEL" \
    --host 0.0.0.0 --port $PORT --model-name "$MODEL" \
    --max-seq-len 4096 --max-batch-size 16 --kv-cache-dtype bf16 \
    --gpu-memory-utilization 0.70 \
    --enable-prefix-caching --ssm-cache-slots 32 --ssm-checkpoint-interval 32 \
    --speculative --num-drafts 3 --mtp-quantization bf16 \
    --tool-call-parser qwen3_xml --disable-tool-grammar true --disable-thinking >/dev/null 2>&1
  local ok=0
  for _ in $(seq 1 200); do
    curl -sf -m4 "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q Qwen && { ok=1; break; }
    sudo docker ps --format '{{.Names}}' | grep -q atlas-bsweep || break
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
    # Graph-capture evidence for the graphs leg (proof the flag engaged).
    sudo docker logs atlas-bsweep 2>&1 | grep -ac "graph" > "$CS/$leg.graphlines" || true
  else
    sudo docker logs atlas-bsweep 2>&1 | tail -60 > "$CS/$leg.deathlog" || true
    note "LEG $leg SERVE_DIED (deathlog: conc_sweep/$leg.deathlog)"
  fi
  teardown
  echo "LEG_DONE $leg"
}

run_leg atlasB_nographs ""
run_leg atlasB_graphs "-e ATLAS_DECODE_GRAPHS_MULTISEQ=1"
note "PHASEB_DONE"
echo "PHASEB_DONE"
