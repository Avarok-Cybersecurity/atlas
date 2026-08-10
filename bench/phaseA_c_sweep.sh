#!/bin/bash
# Phase A driver: the C=[1,2,4,8,16] synthetic scoreboard, Atlas vs vLLM, one box.
#
# SESSION-SURVIVABLE BY DESIGN. This script is launched once via `setsid nohup`
# and then owns the whole phase: serve -> health -> bench -> teardown -> next
# leg, appending to the durable STATE.md after every leg. The interactive
# session that launched it can die at any point; a rejoining session resumes by
# reading STATE.md and tailing conc_sweep/phaseA.log. Re-running this script is
# safe: a leg whose results json already exists is SKIPPED, so a killed driver
# continues instead of repeating finished work.
#
# Scoreboard semantics (user-confirmed): aggregate tok/s must beat vLLM at every
# C AND TTFT/TPOT p50/p99 must not lose. This driver only MEASURES; verdicts are
# drawn from the two json files by the comparison step at the end.
#
# Fairness notes recorded here because they bound what the numbers mean:
#  - Same box, same bench script, same ISL/OSL regimes, sequential legs.
#  - Atlas runs its golden env/flags with --max-batch-size 16 and fifo
#    scheduling (SLAI's should_prefill starves admission whenever any decoder is
#    >80 ms stale — scheduling_policy.rs:101-113 — so it is the wrong policy for
#    a throughput sweep).
#  - vLLM is tried FIRST on the SAME checkpoint Atlas serves (centml W4A4); if
#    it cannot load it, the driver falls back to nvidia/Qwen3.6-27B-NVFP4 and
#    the leg json + STATE record the substitution — a known caveat, not a
#    silent one (the July MLPerf comparison carried the same caveat).
set -u
WT=/workspace/.wt-golden
CS=$WT/conc_sweep
RESULTS=$CS/results
STATE=$WT/docs/campaigns/gb10-concurrency-2026-07/STATE.md
BENCH=$WT/bench/bench-atlas-concurrency.py
ATLAS_BIN=$WT/conc_sweep/spark_phaseA_baseline
MODEL_ATLAS=centml/Qwen3.6-27B-NVFP4-W4A4-mlpinf
VLLM_IMAGE=sparkrun-eugr-vllm:latest
PORT=8888
mkdir -p "$RESULTS"

note() { # append one line to STATE.md's Log section AND the driver log
  echo "- $(date -u +%FT%TZ) $*" >> "$STATE"
  echo "STATE: $*"
}

teardown() { sudo docker rm -f atlas-csweep vllm-csweep >/dev/null 2>&1; sleep 3; }

wait_health() { # $1 container, $2 grep pattern, $3 max tries (x5s)
  for _ in $(seq 1 "$3"); do
    curl -sf -m4 "http://localhost:$PORT/v1/models" 2>/dev/null | grep -q "$2" && return 0
    sudo docker ps --format '{{.Names}}' | grep -q "$1" || return 1
    sleep 5
  done
  return 1
}

run_bench() { # $1 results file tag
  BENCH_PORT=$PORT BENCH_MAX_SEQ_LEN=4096 \
    BENCH_RESULTS_FILE="$RESULTS/$1.json" \
    python3 -u "$BENCH" 2>&1 | tail -30
  [ -s "$RESULTS/$1.json" ]
}

############################ LEG 1: Atlas ############################
if [ -s "$RESULTS/atlas_synth.json" ]; then
  echo "SKIP atlas_synth (results exist)"
else
  teardown
  sudo docker run -d --name atlas-csweep --network host --gpus all --ipc=host \
    -e ATLAS_NO_FFN_NVFP4_MMQ=1 -e ATLAS_SSM_TAIL_MIDCHUNK=0 -e ATLAS_MTP_CATCHUP=0 \
    -e ATLAS_MTP_DRAFT_CONF=0.0 -e ATLAS_MTP_GATE_FORCE=1 \
    -e ATLAS_SSM_TAIL_LEASE_TTL=128 -e ATLAS_BF16_TC_PREFILL=1 \
    -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
    -v "$ATLAS_BIN:/usr/local/bin/spark:ro" \
    atlas-gb10:followups serve "$MODEL_ATLAS" \
    --host 0.0.0.0 --port $PORT --model-name "$MODEL_ATLAS" \
    --max-seq-len 4096 --max-batch-size 16 --kv-cache-dtype bf16 \
    --gpu-memory-utilization 0.70 \
    --enable-prefix-caching --ssm-cache-slots 32 --ssm-checkpoint-interval 32 \
    --speculative --num-drafts 3 --mtp-quantization bf16 \
    --tool-call-parser qwen3_xml --disable-tool-grammar true --disable-thinking >/dev/null 2>&1
  if wait_health atlas-csweep Qwen 200; then
    echo "=== LEG atlas_synth: serve up, benching ==="
    if run_bench atlas_synth; then
      note "LEG atlas_synth DONE -> results/atlas_synth.json"
    else
      note "LEG atlas_synth BENCH FAILED (no results written)"
    fi
    # Slot-leak check (known server bug: pool exhaustion leaks slots).
    sudo docker logs atlas-csweep 2>&1 | grep -aic "pool exhausted" \
      | xargs -I{} echo "pool-exhausted lines: {}" | tee -a "$CS/atlas_synth.notes"
  else
    # Preserve the evidence BEFORE teardown destroys it (first death lost its log).
    sudo docker logs atlas-csweep 2>&1 | tail -60 > "$CS/atlas_synth.deathlog" || true
    note "LEG atlas_synth SERVE_DIED (deathlog: conc_sweep/atlas_synth.deathlog)"
    echo "SERVE_DIED atlas_synth"
  fi
  teardown
  echo "LEG_DONE atlas_synth"
fi

############################ LEG 2: vLLM ############################
if [ -s "$RESULTS/vllm_synth.json" ]; then
  echo "SKIP vllm_synth (results exist)"
else
  for VM in "$MODEL_ATLAS" "nvidia/Qwen3.6-27B-NVFP4"; do
    teardown
    echo "=== LEG vllm_synth: trying checkpoint $VM ==="
    sudo docker run -d --name vllm-csweep --network host --gpus all --ipc=host \
      -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
      -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
      --entrypoint bash "$VLLM_IMAGE" \
      -c "vllm serve $VM --host 0.0.0.0 --port $PORT \
          --max-model-len 32768 --gpu-memory-utilization 0.85 \
          --max-num-seqs 128" >/dev/null 2>&1
    # vLLM load is slow (17+ shards historically ~6 min; NVFP4 27B smaller): 25 min cap.
    if wait_health vllm-csweep Qwen 300; then
      echo "=== vllm up on $VM, benching ==="
      if run_bench vllm_synth; then
        python3 - "$RESULTS/vllm_synth.json" "$VM" <<'PY'
import json, sys
p, m = sys.argv[1], sys.argv[2]
d = json.load(open(p)); d["served_checkpoint"] = m
json.dump(d, open(p, "w"), indent=1)
PY
        note "LEG vllm_synth DONE on $VM -> results/vllm_synth.json"
      else
        note "LEG vllm_synth BENCH FAILED on $VM"
      fi
      break
    else
      note "vllm checkpoint $VM failed to serve (trying fallback if any)"
      sudo docker logs vllm-csweep 2>&1 | tail -15 > "$CS/vllm_${VM##*/}.faillog"
    fi
  done
  teardown
  echo "LEG_DONE vllm_synth"
fi

############################ Compare ############################
if [ -s "$RESULTS/atlas_synth.json" ] && [ -s "$RESULTS/vllm_synth.json" ]; then
  BENCH_RESULTS_FILE="$RESULTS/atlas_synth.json" \
    python3 -u "$BENCH" --compare "$RESULTS/vllm_synth.json" \
    > "$RESULTS/compare.txt" 2>&1 || true
  note "PHASE A compare written -> results/compare.txt"
fi
note "PHASEA_DONE"
echo "PHASEA_DONE"
