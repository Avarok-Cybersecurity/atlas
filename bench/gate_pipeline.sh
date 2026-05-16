#!/usr/bin/env bash
# Autonomous gate pipeline: baseline -> DEER (determinism + N>=10 gate) ->
# ThinkBrake (N>=10 gate) -> final report. Unattended (~6-10h). Each layer
# is kept iff it beats the cf-complete-gate baseline within the harness's
# statistical gate (two-proportion z-test); otherwise its MODEL.toml flag
# is reverted and the image rebuilt. Layer C (streaming Answer-Regen) is a
# separate follow-up after these verdicts (its code is not yet written).
#
# Safety: does NOT touch the server until bench/results_baseline.json
# exists (the in-flight baseline harness owns :8888 until it finishes).
set -uo pipefail

ATLAS=/workspace/atlas
BENCH=$ATLAS/bench
MODEL_TOML=$ATLAS/kernels/gb10/qwen3.6-27b/MODEL.toml
SNAP=/root/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B-FP8/snapshots/e89b16ebf1988b3d6befa7de50abc2d76f26eb09
LOG=/tmp/gate_pipeline.log
REPORT=$BENCH/GATE_REPORT.md
N=10
SEED=1000

cd "$ATLAS"
exec >>"$LOG" 2>&1
ts() { date '+%Y-%m-%d %H:%M:%S'; }
say() { echo "[$(ts)] $*"; }

set_flag() {  # set_flag <key> <true|false|float|int>
  python3 - "$MODEL_TOML" "$1" "$2" <<'PY'
import re, sys
path, key, val = sys.argv[1], sys.argv[2], sys.argv[3]
src = open(path).read()
line = f"{key} = {val}"
if re.search(rf'(?m)^{re.escape(key)}\s*=', src):
    src = re.sub(rf'(?m)^{re.escape(key)}\s*=.*$', line, src)
else:
    # insert just before the first [[model_types]] block
    src = src.replace("\n[[model_types]]", f"\n{line}\n\n[[model_types]]", 1)
open(path, "w").write(src)
print(f"set {key} = {val}")
PY
}

build() {  # build <tag>
  say "docker build -> atlas-gb10:$1"
  docker build -f docker/gb10/Dockerfile -t "atlas-gb10:$1" . >/dev/null 2>&1 \
    && say "build OK: atlas-gb10:$1" || { say "BUILD FAILED: $1"; return 1; }
}

deploy() {  # deploy <tag> [extra docker -e args...]
  local tag=$1; shift
  docker stop atlas-qwen36-27b >/dev/null 2>&1
  docker rm   atlas-qwen36-27b >/dev/null 2>&1
  docker run -d --name atlas-qwen36-27b --gpus all --network host --ipc=host \
    -v /root/.cache/huggingface:/root/.cache/huggingface "$@" \
    "atlas-gb10:$tag" \
    serve "$SNAP" --port 8888 --model-name qwen3.6-27b \
    --num-drafts 1 --max-seq-len 65536 --enable-prefix-caching >/dev/null 2>&1
  say "deploy atlas-gb10:$tag (env: $*) — waiting for ready"
  for _ in $(seq 1 120); do
    curl -sf http://localhost:8888/v1/models >/dev/null 2>&1 && { say "READY"; return 0; }
    sleep 5
  done
  say "DEPLOY TIMEOUT: $tag"; return 1
}

one_gen() {  # one_gen <out_txt> <seed> ; deterministic blocking chess gen
  python3 - "$1" "$2" <<'PY'
import json, sys, urllib.request
out, seed = sys.argv[1], int(sys.argv[2])
fx = json.load(open("/workspace/atlas/bench/fixtures/chess_prompt.json"))
body = {"model": "qwen3.6-27b", "messages": fx["chess"]["messages"],
        "temperature": 0.0, "max_tokens": 4000, "seed": seed}
req = urllib.request.Request("http://localhost:8888/v1/chat/completions",
    data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
r = json.loads(urllib.request.urlopen(req, timeout=420).read())
m = r["choices"][0].get("message", {})
open(out, "w").write((m.get("content") or "") + "\x1e" + (m.get("reasoning_content") or ""))
print("captured", out, len(m.get("content") or ""), "chars")
PY
}

# ---- 0. wait for the in-flight baseline ----
say "=== gate pipeline start; waiting for baseline JSON ==="
while [ ! -f "$BENCH/results_baseline.json" ]; do sleep 60; done
say "baseline JSON present"
echo "# Gate pipeline report ($(ts))" > "$REPORT"
DEER_VERDICT="not-run"; TB_VERDICT="not-run"

# ---- 1. DEER ----
if build deer-on; then
  # 1a. determinism (single image, two envs, same seed)
  if deploy deer-on -e ATLAS_DEER_DISABLE=1; then
    one_gen /tmp/deer_ref.txt 12345
    if deploy deer-on -e ATLAS_DEER_FORCE_ROLLBACK=1; then
      one_gen /tmp/deer_cmp.txt 12345
      if cmp -s /tmp/deer_ref.txt /tmp/deer_cmp.txt; then
        say "DEER determinism: PASS (byte-identical)"
        # 1b. quality gate
        if deploy deer-on; then
          python3 "$BENCH/reasoning_eval.py" run --n $N --seed $SEED \
            --config deer-on --out "$BENCH/results_deer.json"
          if python3 "$BENCH/reasoning_eval.py" gate \
               "$BENCH/results_baseline.json" "$BENCH/results_deer.json"; then
            DEER_VERDICT="KEPT (beat baseline, determinism PASS)"
          else
            DEER_VERDICT="REVERTED (did not beat baseline)"
            set_flag enable_deer false
          fi
        else DEER_VERDICT="REVERTED (deploy fail)"; set_flag enable_deer false; fi
      else
        DEER_VERDICT="REJECTED (determinism FAIL — lossy rollback)"
        set_flag enable_deer false
      fi
    else DEER_VERDICT="REJECTED (determinism deploy fail)"; set_flag enable_deer false; fi
  else DEER_VERDICT="REJECTED (determinism deploy fail)"; set_flag enable_deer false; fi
else DEER_VERDICT="REJECTED (build fail)"; set_flag enable_deer false; fi
say "DEER verdict: $DEER_VERDICT"
echo "- **DEER**: $DEER_VERDICT" >> "$REPORT"

# ---- 2. ThinkBrake (DEER flag now frozen at its gated value) ----
set_flag enable_thinkbrake true
if build thinkbrake-on; then
  if deploy thinkbrake-on; then
    python3 "$BENCH/reasoning_eval.py" run --n $N --seed $SEED \
      --config thinkbrake-on --out "$BENCH/results_thinkbrake.json"
    if python3 "$BENCH/reasoning_eval.py" gate \
         "$BENCH/results_baseline.json" "$BENCH/results_thinkbrake.json"; then
      TB_VERDICT="KEPT (beat baseline)"
    else
      TB_VERDICT="REVERTED (did not beat baseline)"
      set_flag enable_thinkbrake false
    fi
  else TB_VERDICT="REVERTED (deploy fail)"; set_flag enable_thinkbrake false; fi
else TB_VERDICT="REVERTED (build fail)"; set_flag enable_thinkbrake false; fi
say "ThinkBrake verdict: $TB_VERDICT"
echo "- **ThinkBrake**: $TB_VERDICT" >> "$REPORT"

# ---- 3. Answer-Regen (DEER + ThinkBrake flags now frozen) ----
AR_VERDICT="not-run"
set_flag enable_answer_regen true
if build answer-regen-on; then
  if deploy answer-regen-on; then
    python3 "$BENCH/reasoning_eval.py" run --n $N --seed $SEED \
      --config answer-regen-on --out "$BENCH/results_answer_regen.json"
    if python3 "$BENCH/reasoning_eval.py" gate \
         "$BENCH/results_baseline.json" "$BENCH/results_answer_regen.json"; then
      AR_VERDICT="KEPT (beat baseline)"
    else
      AR_VERDICT="REVERTED (did not beat baseline)"
      set_flag enable_answer_regen false
    fi
  else AR_VERDICT="REVERTED (deploy fail)"; set_flag enable_answer_regen false; fi
else AR_VERDICT="REVERTED (build fail)"; set_flag enable_answer_regen false; fi
say "Answer-Regen verdict: $AR_VERDICT"
echo "- **Answer-Regen**: $AR_VERDICT" >> "$REPORT"

# ---- 4. final image reflecting kept flags + report ----
build final-gated && deploy final-gated
{
  echo ""
  echo "## Summary ($(ts))"
  echo "- DEER: $DEER_VERDICT"
  echo "- ThinkBrake: $TB_VERDICT"
  echo "- Answer-Regen: $AR_VERDICT"
  echo "- Results: bench/results_{baseline,deer,thinkbrake,answer_regen}.json"
  echo "- Final deployed image: atlas-gb10:final-gated (kept flags only)"
} >> "$REPORT"
say "=== gate pipeline complete ; report: $REPORT ==="
