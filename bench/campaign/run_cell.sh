#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Run ONE campaign cell end to end and emit its section-10 artifact.
#
# A "cell" is (engine, model, SKU, workload, concurrency, spec, think). This
# script is the day-of driver: preflight, serve, boot gate, coherency gate,
# latency pack, teardown, assemble, validate. It orchestrates the tools that
# already exist -- scripts/start-node-ep.sh, bench/campaign/vllm_control.sh,
# bench/hopper_ab/{time_to_ready.sh,coherency_gate.py},
# bench/ladder38/harness_w55_conc_ladder.py -- and rewrites none of them.
#
# THE RULE THAT SHAPES THE WHOLE FLOW: the artifact is ALWAYS written. A cell
# that fails its boot gate is not a cell with no data; it is a NO-GO with a
# named failing stage, and that is a result the campaign needs as much as a
# CERTIFIED one. So a failing stage records itself, teardown still runs, the
# artifact is still assembled and validated, and only the EXIT CODE says the
# cell failed. The alternative -- die on the first non-zero and leave nothing
# behind -- is how an expensive hour on a rented box produces a shrug.
#
# WHAT IT REFUSES
#
#  * Nothing starts on a box without --yes. --dry-run prints every command and
#    launches nothing; without either flag the script exits 2. A benchmark
#    driver that boots an engine because someone was reading its help text is a
#    bad neighbour on a shared GPU.
#  * A (model, SKU) pair with no recipe exits 3 before anything is created.
#    Reconstructing a serve command is how a campaign measures a guess.
#  * --spec on against a model whose recipe declares no speculative profile
#    exits 4. Both-or-neither is only enforceable if neither side can improvise.
#
# WHAT IT WARNS ABOUT, LOUDLY, AND RECORDS
#
#  * --think on. The ladder sends chat_template_kwargs.enable_thinking=true
#    only when passed --enable-thinking (default false, the GB10 campaign's
#    setting). This script passes it for a think-on cell and the ladder header
#    records which value was sent, so compare.py refuses a think-mismatched
#    pair. Older ladder JSONs without the flag were think-off.
#  * A missing warmup prompt. PRD section 6 pins
#    --warmup-prompt bench/hopper_ab/warmup_1024.txt (a copy of the repo's
#    tests/fixtures/bench_prompt_1024.txt) to kill the 5-30 s first-request
#    autotune. If the file is absent the launcher is given no warmup prompt and
#    the artifact says so rather than silently serving a cold first request.
#
# Usage:
#   run_cell.sh --engine atlas|vllm --model <key> --sku h100|h200|b200|gb10 \
#               --workload lat|agent --concurrency <N> --spec on|off \
#               --think on|off --out <dir> [--dry-run] [--yes] \
#               [--paired-artifact PATH] [--ptx-receipt PATH]
#
# Environment:
#   ATLAS_PORT / VLLM_PORT   client port (default 8888 / 8000)
#   SPARK_BIN                Atlas binary (default ./target/release/spark)
#   IMAGE                    Atlas container image; empty = run SPARK_BIN
#   VLLM_IMAGE_DIGEST        sha256:... -- required for a real vLLM run
#   HF_CACHE                 host HF cache (default ~/.cache/huggingface)
#
# Exit: 0 every gate passed · 1 a gate failed (artifact written anyway) ·
#       2 usage or refusal-to-start · 3 no recipe for the pair · 4 --spec on
#       with no speculative profile
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"

WORKLOADS="$ROOT/bench/hopper_ab/workloads.json"
LADDER="$ROOT/bench/ladder38/harness_w55_conc_ladder.py"
TTR="$ROOT/bench/hopper_ab/time_to_ready.sh"
COHERENCY="$ROOT/bench/hopper_ab/coherency_gate.py"
LAUNCHER="$ROOT/scripts/start-node-ep.sh"
ATLAS_RECIPES="$HERE/atlas_recipes.json"
VLLM_RECIPES="$HERE/vllm_recipes.json"

ENGINE=""; MODEL=""; SKU=""; WORKLOAD=""; CONC=""; SPEC=""; THINK=""; OUT=""
DRY_RUN=0; YES=0; PAIRED=""; PTX_RECEIPT=""

usage() { sed -n '2,68p' "$0"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --engine) ENGINE="${2:-}"; shift 2 ;;
    --model) MODEL="${2:-}"; shift 2 ;;
    --sku) SKU="${2:-}"; shift 2 ;;
    --workload) WORKLOAD="${2:-}"; shift 2 ;;
    --concurrency) CONC="${2:-}"; shift 2 ;;
    --spec) SPEC="${2:-}"; shift 2 ;;
    --think) THINK="${2:-}"; shift 2 ;;
    --out) OUT="${2:-}"; shift 2 ;;
    --paired-artifact) PAIRED="${2:-}"; shift 2 ;;
    --ptx-receipt) PTX_RECEIPT="${2:-}"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    --yes) YES=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

die() { echo "ERROR: $*" >&2; exit 2; }

case "$ENGINE" in atlas|vllm) ;; *) die "--engine must be atlas or vllm (got '${ENGINE}')" ;; esac
case "$SKU" in h100|h200|b200|gb10) ;; *) die "--sku must be h100|h200|b200|gb10 (got '${SKU}')" ;; esac
case "$WORKLOAD" in lat|agent) ;; *) die "--workload must be lat or agent (got '${WORKLOAD}')" ;; esac
case "$SPEC" in on|off) ;; *) die "--spec must be on or off (got '${SPEC}')" ;; esac
case "$THINK" in on|off) ;; *) die "--think must be on or off (got '${THINK}')" ;; esac
case "$CONC" in ''|*[!0-9]*) die "--concurrency must be a positive integer (got '${CONC}')" ;; esac
[ "$CONC" -gt 0 ] || die "--concurrency must be greater than zero"
[ -n "$MODEL" ] || die "--model is required"
[ -n "$OUT" ] || die "--out is required"

if [ "$DRY_RUN" != "1" ] && [ "$YES" != "1" ]; then
  echo "REFUSED: this would start an engine on this box." >&2
  echo "  Re-run with --dry-run to see every command, or --yes to actually run it." >&2
  exit 2
fi

for f in "$WORKLOADS" "$LADDER" "$TTR" "$COHERENCY" "$ATLAS_RECIPES" "$VLLM_RECIPES"; do
  [ -f "$f" ] || die "missing required tool or data file: $f"
done

# ── shape, from the frozen workload file ─────────────────────────────────────
ISL="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["workloads"][sys.argv[2]]["isl"])' "$WORKLOADS" "$WORKLOAD")"
OSL="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["workloads"][sys.argv[2]]["osl"])' "$WORKLOADS" "$WORKLOAD")"
REPS="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["reps"])' "$WORKLOADS")"
WARMUP="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["warmup"])' "$WORKLOADS")"
BOOT_CAP="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["gates"]["boot_s_max"])' "$WORKLOADS")"

# ── recipe resolution, before anything is created ────────────────────────────
if [ "$ENGINE" = "atlas" ]; then
  if ! python3 "$HERE/atlas_render.py" --recipes "$ATLAS_RECIPES" \
        --model "$MODEL" --sku "$SKU" --probe; then
    exit 3
  fi
  HF_ID="$(python3 "$HERE/atlas_render.py" --recipes "$ATLAS_RECIPES" --model "$MODEL" --sku "$SKU" --field hf_id)"
  NGPUS="$(python3 "$HERE/atlas_render.py" --recipes "$ATLAS_RECIPES" --model "$MODEL" --sku "$SKU" --field ngpus)"
  EP_SIZE="$(python3 "$HERE/atlas_render.py" --recipes "$ATLAS_RECIPES" --model "$MODEL" --sku "$SKU" --field ep_size)"
  TP_SIZE="$(python3 "$HERE/atlas_render.py" --recipes "$ATLAS_RECIPES" --model "$MODEL" --sku "$SKU" --field tp_size)"
  EXTRA_ARGS="$(python3 "$HERE/atlas_render.py" --recipes "$ATLAS_RECIPES" --model "$MODEL" \
                  --sku "$SKU" --spec "$SPEC" --think "$THINK" --extra-args)" || exit $?
  PORT="${ATLAS_PORT:-8888}"
else
  if ! python3 "$HERE/vllm_render.py" --recipes "$VLLM_RECIPES" \
        --model "$MODEL" --sku "$SKU" --probe; then
    exit 3
  fi
  HF_ID="$(python3 -c '
import json,sys
d=json.load(open(sys.argv[1]))
print(next(e["hf_id"] for e in d["entries"]
           if e["model_key"]==sys.argv[2] and e["sku"]==sys.argv[3]))' \
    "$VLLM_RECIPES" "$MODEL" "$SKU")"
  PORT="${VLLM_PORT:-8000}"
fi
URL="http://127.0.0.1:$PORT"

CELL_ID="$ENGINE.$MODEL.$SKU.$WORKLOAD.c$CONC.spec$SPEC.think$THINK"
# Docker accepts dots in a name, but a name that reads back as the cell it
# belongs to is what makes a stray container identifiable an hour later. The
# pid is appended because the name must ALSO be this invocation's alone: a
# deterministic name is a name a re-run, or a second operator, already holds,
# and `docker run` then fails with 125 while the teardown below aims at
# somebody else's live server.
CONTAINER="atlas-campaign-$(printf '%s' "$MODEL-$SKU-$WORKLOAD-c$CONC-spec$SPEC-think$THINK" | tr '.' '-')-$$"
# The ownership stamp that survives on the container itself, so a stray one can
# be traced back to the cell and the moment that created it.
RUN_LABEL="atlas-campaign.run=$CELL_ID-$(date +%s)"
# Written by vllm_control.sh only when its `docker run -d` actually created a
# container. Its absence is the proof that there is nothing to tear down.
CONTAINER_ID_FILE="$OUT/container.id"
CONTAINER_ID=""

echo "=== campaign cell $CELL_ID ==="
echo "engine:      $ENGINE"
echo "model:       $MODEL -> $HF_ID"
echo "sku:         $SKU"
echo "workload:    $WORKLOAD  isl=$ISL osl=$OSL  C=$CONC  reps=$REPS warmup=$WARMUP"
echo "spec:        $SPEC     think: $THINK"
echo "boot cap:    ${BOOT_CAP}s"
echo "out:         $OUT"
echo "client url:  $URL"
if [ "$DRY_RUN" = "1" ]; then echo "mode:        DRY RUN (nothing is launched, nothing is written)"; fi
echo ""

# --think on is a client-side setting too: the ladder sends
# chat_template_kwargs.enable_thinking=true only with --enable-thinking, and
# records which it sent in its header (compare.py refuses a mismatched pair).
LADDER_THINK=()
if [ "$THINK" = "on" ]; then LADDER_THINK=(--enable-thinking); fi

WARMUP_PROMPT="$ROOT/bench/hopper_ab/warmup_1024.txt"
WARMUP_NOTE=""
if [ ! -f "$WARMUP_PROMPT" ]; then
  echo "WARNING: $WARMUP_PROMPT does not exist."
  echo "  PRD section 6 pins it to kill the 5-30 s first-request autotune. Serving"
  echo "  without it means the first measured request may carry a capture cost."
  echo "  Recording warmup_prompt=null rather than pretending it was used."
  echo ""
  WARMUP_PROMPT=""
  WARMUP_NOTE="served with no --warmup-prompt: bench/hopper_ab/warmup_1024.txt was not found, so the first request may carry autotune cost"
fi

# ── plumbing ─────────────────────────────────────────────────────────────────
FAILING_STAGE=""
note_fail() { [ -n "$FAILING_STAGE" ] || FAILING_STAGE="$1"; echo "STAGE FAILED: $1"; }

step() { echo ""; echo "--- $* ---"; }

show() { echo "\$ $*"; }

if [ "$DRY_RUN" != "1" ]; then
  mkdir -p "$OUT" || die "cannot create $OUT"
fi

ARTIFACT="$OUT/artifact.json"
SERVE_ARGV="$OUT/serve.argv"
SERVE_ENV="$OUT/serve.env"
SMI_Q="$OUT/nvidia-smi-q.txt"
BOOT_JSON="$OUT/boot.json"
COH_JSON="$OUT/coherency.json"
LADDER_JSON="$OUT/ladder.json"

# ── 1. preflight ─────────────────────────────────────────────────────────────
step "stage 1/7 preflight"
show "nvidia-smi -q > $SMI_Q"
show "df -h $OUT > $OUT/df.txt"
show "git -C $ROOT rev-parse HEAD"
show "sha256sum $LADDER"
GIT_SHA=""
if [ "$DRY_RUN" != "1" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -q > "$SMI_Q" 2>"$OUT/nvidia-smi-q.err" || note_fail preflight
  else
    echo "no nvidia-smi on this host" > "$OUT/nvidia-smi-q.err"
    note_fail preflight
  fi
  df -h "$OUT" > "$OUT/df.txt" 2>&1
  GIT_SHA="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || true)"
  if [ -n "$(git -C "$ROOT" status --porcelain 2>/dev/null)" ]; then
    echo "NOTE: the tree is dirty; git_sha $GIT_SHA does not fully describe it."
  fi
fi

# ── 2. serve ─────────────────────────────────────────────────────────────────
step "stage 2/7 serve"
START_EPOCH=""
if [ "$ENGINE" = "atlas" ]; then
  ATLAS_ENV=(
    "NGPUS=$NGPUS" "EP_SIZE=$EP_SIZE" "TP_SIZE=$TP_SIZE"
    "PORT_BASE=$PORT" "BIND=0.0.0.0" "NCCL_PROFILE=default"
    "BOOT_TIMEOUT_S=$BOOT_CAP" "EXTRA_ARGS=$EXTRA_ARGS"
  )
  [ -n "$WARMUP_PROMPT" ] && ATLAS_ENV+=( "WARMUP_PROMPT=$WARMUP_PROMPT" )
  [ -n "${SPARK_BIN:-}" ] && ATLAS_ENV+=( "SPARK_BIN=$SPARK_BIN" )
  [ -n "${IMAGE:-}" ] && ATLAS_ENV+=( "IMAGE=$IMAGE" )

  show "env ${ATLAS_ENV[*]} bash $LAUNCHER --check-kernels $HF_ID"
  show "env ${ATLAS_ENV[*]} bash $LAUNCHER $HF_ID   # backgrounded; time_to_ready.sh owns the boot clock"
  if [ "$DRY_RUN" = "1" ]; then
    env "${ATLAS_ENV[@]}" bash "$LAUNCHER" --dry-run "$HF_ID" 2>&1 | sed 's/^/  | /'
  else
    printf '%s\n' "${ATLAS_ENV[@]}" > "$SERVE_ENV"
    env "${ATLAS_ENV[@]}" bash "$LAUNCHER" --dry-run "$HF_ID" > "$OUT/serve-dryrun.txt" 2>&1
    python3 - "$OUT/serve-dryrun.txt" "$SERVE_ARGV" <<'PY'
import pathlib, shlex, sys
line = next((l for l in pathlib.Path(sys.argv[1]).read_text().splitlines()
             if l.startswith("rank0_command: ")), None)
argv = shlex.split(line[len("rank0_command: "):]) if line else []
pathlib.Path(sys.argv[2]).write_bytes(b"\0".join(a.encode() for a in argv))
PY
    env "${ATLAS_ENV[@]}" bash "$LAUNCHER" --check-kernels "$HF_ID" > "$OUT/check-kernels.txt" 2>&1 \
      || note_fail serve
    START_EPOCH="$(date +%s)"
    env "${ATLAS_ENV[@]}" bash "$LAUNCHER" "$HF_ID" > "$OUT/serve.log" 2>&1 &
    SERVE_PID=$!
    echo "launcher pid $SERVE_PID; log $OUT/serve.log"
  fi
else
  VC_ARGS=( "$MODEL" "$SKU" --spec "$SPEC" --label "$RUN_LABEL" )
  show "VLLM_CONTAINER=$CONTAINER bash $HERE/vllm_control.sh ${VC_ARGS[*]} --id-file $CONTAINER_ID_FILE"
  if [ "$DRY_RUN" = "1" ]; then
    VLLM_CONTAINER="$CONTAINER" VLLM_RECIPES="$VLLM_RECIPES" \
      bash "$HERE/vllm_control.sh" "${VC_ARGS[@]}" --dry-run 2>&1 | sed 's/^/  | /'
  else
    printf 'VLLM_IMAGE_DIGEST=%s\n' "${VLLM_IMAGE_DIGEST:-}" > "$SERVE_ENV"
    START_EPOCH="$(date +%s)"
    VLLM_CONTAINER="$CONTAINER" VLLM_RECIPES="$VLLM_RECIPES" \
      bash "$HERE/vllm_control.sh" "${VC_ARGS[@]}" --id-file "$CONTAINER_ID_FILE" \
      > "$OUT/serve.log" 2>&1
    serve_rc=$?
    if [ "$serve_rc" -eq 125 ]; then
      echo "docker run exited 125: a container named $CONTAINER already exists and"
      echo "  was NOT created by this invocation. Nothing of it is stopped or removed;"
      echo "  the serve stage fails and the cell records it."
      note_fail serve
    elif [ "$serve_rc" -ne 0 ]; then
      note_fail serve
    fi
    if [ -s "$CONTAINER_ID_FILE" ]; then
      CONTAINER_ID="$(cat "$CONTAINER_ID_FILE")"
      echo "container id: $CONTAINER_ID"
    fi
    sed -n 's/^docker run /docker run /p' "$OUT/serve.log" | head -1 > "$OUT/serve-cmd.txt"
    python3 - "$OUT/serve-cmd.txt" "$SERVE_ARGV" <<'PY'
import pathlib, shlex, sys
text = pathlib.Path(sys.argv[1]).read_text().strip()
argv = shlex.split(text) if text else []
pathlib.Path(sys.argv[2]).write_bytes(b"\0".join(a.encode() for a in argv))
PY
  fi
fi

# ── 3. boot gate ─────────────────────────────────────────────────────────────
step "stage 3/7 boot gate (cap ${BOOT_CAP}s)"
show "bash $TTR --url $URL --model $HF_ID --engine $ENGINE --start-epoch <serve-start> --timeout-s $BOOT_CAP --out $BOOT_JSON"
if [ "$DRY_RUN" != "1" ] && [ -z "$FAILING_STAGE" ]; then
  bash "$TTR" --url "$URL" --model "$HF_ID" --engine "$ENGINE" \
       --start-epoch "$START_EPOCH" --timeout-s "$BOOT_CAP" --out "$BOOT_JSON" \
    || note_fail boot
fi

# ── 4. coherency gate ────────────────────────────────────────────────────────
step "stage 4/7 coherency gate"
show "python3 $COHERENCY --url $URL --model $HF_ID --out $COH_JSON"
if [ "$DRY_RUN" != "1" ] && [ -z "$FAILING_STAGE" ]; then
  python3 "$COHERENCY" --url "$URL" --model "$HF_ID" --out "$COH_JSON" || note_fail coherency
fi

# ── 5. latency pack ──────────────────────────────────────────────────────────
step "stage 5/7 latency pack"
show "python3 $LADDER --url $URL --model $HF_ID --label $CELL_ID --out $LADDER_JSON --concs $CONC --reps $REPS --isl $ISL --osl $OSL --warmup $WARMUP ${LADDER_THINK[*]:-}"
if [ "$DRY_RUN" != "1" ] && [ -z "$FAILING_STAGE" ]; then
  python3 "$LADDER" --url "$URL" --model "$HF_ID" --label "$CELL_ID" \
          --out "$LADDER_JSON" --concs "$CONC" --reps "$REPS" \
          --isl "$ISL" --osl "$OSL" --warmup "$WARMUP" "${LADDER_THINK[@]}" || note_fail ladder
fi

# ── 6. teardown (always) ─────────────────────────────────────────────────────
step "stage 6/7 teardown"
if [ "$ENGINE" = "atlas" ]; then
  show "bash $LAUNCHER --stop"
  if [ "$DRY_RUN" != "1" ]; then
    bash "$LAUNCHER" --stop || note_fail teardown
  fi
else
  # By the container ID this invocation's `docker run -d` returned, and never
  # by `pkill -f`: a `pkill -f vllm` pattern matches this script's own command
  # line, which is how a teardown becomes a self-kill.
  #
  # The ID, not the name, because this block runs even when the serve stage
  # failed. If it failed with 125 the name belongs to a container somebody else
  # is using, and `docker stop <name>` / `docker rm <name>` then deletes their
  # live server -- exactly the sequence a stub Docker recorded: run, stop
  # <same-name>, rm <same-name>. No ID means nothing was created here, and
  # nothing is this cell's to remove.
  if [ "$DRY_RUN" = "1" ]; then
    show "docker stop <id from docker run -d> && docker rm <the same id>"
  elif [ -n "$CONTAINER_ID" ]; then
    show "docker stop $CONTAINER_ID && docker rm $CONTAINER_ID"
    "${DOCKER:-docker}" stop "$CONTAINER_ID" >/dev/null 2>&1 || true
    "${DOCKER:-docker}" rm "$CONTAINER_ID" >/dev/null 2>&1 || true
  else
    echo "no container was created by this invocation ($CONTAINER_ID_FILE is absent):"
    echo "  nothing to stop or remove. A container already holding the name"
    echo "  '$CONTAINER' is not this cell's to delete."
  fi
fi

# ── 7. assemble + validate ───────────────────────────────────────────────────
step "stage 7/7 assemble and validate"
ASSEMBLE=( python3 "$HERE/cell_assemble.py"
  --engine "$ENGINE" --model-key "$MODEL" --sku "$SKU" --workload "$WORKLOAD"
  --concurrency "$CONC" --spec "$SPEC" --think "$THINK" --out "$ARTIFACT"
  --workloads "$WORKLOADS" --atlas-recipes "$ATLAS_RECIPES"
  --vllm-recipes "$VLLM_RECIPES" --client "$LADDER"
  --serve-argv "$SERVE_ARGV" --serve-env "$SERVE_ENV" --nvidia-smi-q "$SMI_Q"
  --boot-json "$BOOT_JSON" --coherency-json "$COH_JSON" --ladder-json "$LADDER_JSON" )
[ -n "$GIT_SHA" ] && ASSEMBLE+=( --git-sha "$GIT_SHA" )
[ -n "${VLLM_IMAGE_DIGEST:-}" ] && [ "$ENGINE" = "vllm" ] && ASSEMBLE+=( --image-digest "$VLLM_IMAGE_DIGEST" )
[ -n "$PAIRED" ] && ASSEMBLE+=( --paired-artifact "$PAIRED" )
[ -n "$PTX_RECEIPT" ] && ASSEMBLE+=( --ptx-receipt "$PTX_RECEIPT" )
[ -n "$WARMUP_NOTE" ] && ASSEMBLE+=( --extra-note "$WARMUP_NOTE" )
[ -n "$FAILING_STAGE" ] && ASSEMBLE+=( --failing-stage "$FAILING_STAGE" )

show "${ASSEMBLE[*]}"
show "python3 $HERE/validate_artifact.py $ARTIFACT"

if [ "$DRY_RUN" = "1" ]; then
  echo ""
  echo "dry-run: nothing launched, nothing written."
  exit 0
fi

"${ASSEMBLE[@]}" || note_fail validate
if [ ! -f "$ARTIFACT" ]; then
  echo "ERROR: the artifact was not written; there is nothing to validate." >&2
  exit 1
fi
python3 "$HERE/validate_artifact.py" "$ARTIFACT" || {
  note_fail validate
  # Re-assemble so the artifact's own verdict admits the validation failure
  # rather than claiming a verdict its shape does not support.
  "${ASSEMBLE[@]}" --failing-stage validate >/dev/null
}

echo ""
if [ -n "$FAILING_STAGE" ]; then
  echo "=== $CELL_ID: FAILED at stage '$FAILING_STAGE'; artifact written to $ARTIFACT ==="
  exit 1
fi
echo "=== $CELL_ID: all gates passed; artifact written to $ARTIFACT ==="
echo "    Verdict is PARTIAL until the paired cell from the other engine exists"
echo "    within 24 h; re-run with --paired-artifact to promote it."
exit 0
