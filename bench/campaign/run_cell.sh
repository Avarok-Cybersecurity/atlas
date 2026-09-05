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
# AND IT IS WRITTEN ON THE WAY OUT OF A SIGNAL, TOO. Teardown that only normal
# control flow reaches is teardown that does not run when an operator Ctrl-Cs
# the campaign or a scheduler sends SIGTERM -- and a detached `docker run -d`
# server then keeps the GPU until somebody notices it by hand. So every way out
# of this script, clean or signalled, goes through ONE idempotent finalizer:
# kill the children this invocation started, stop and remove only what it
# created, and assemble the artifact naming the stage that was interrupted. A
# cell killed at its boot gate is a NO-GO at `boot`, not a silent -15.
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
#  * A thinking mode excluded by the PRD exits 9 before launch. Missing
#    recipes still exit 3; a thinking policy does not supply a recipe.
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
#   ATLAS_NODE_RUN_DIR       PREFIX for this cell's node-EP run directory: the
#                            cell reserves <prefix>-<pid> for itself and stops
#                            only what it recorded (default <out>/node-ep)
#
# Exit: 0 every gate passed · 1 a gate failed (artifact written anyway) ·
#       2 usage or refusal-to-start · 3 no recipe for the pair · 4 --spec on
#       with no speculative profile · 8 invalid vLLM revision identity ·
#       9 excluded thinking mode ·
#       128+N terminated by signal N (130 INT,
#       143 TERM, 129 HUP), with the same teardown run and the artifact naming
#       the stage that was interrupted
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

usage() { sed -n '2,74p' "$0"; }

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

for f in "$WORKLOADS" "$LADDER" "$TTR" "$COHERENCY" "$ATLAS_RECIPES" "$VLLM_RECIPES" \
         "$HERE/thinking_policy.py" "$HERE/thinking_policy.json"; do
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
  VLLM_IMAGE_NAME="$(python3 -c '
import json,sys
d=json.load(open(sys.argv[1]))
print(next(e["image"] for e in d["entries"]
           if e["model_key"]==sys.argv[2] and e["sku"]==sys.argv[3]))' \
    "$VLLM_RECIPES" "$MODEL" "$SKU")"
  PORT="${VLLM_PORT:-8000}"
fi
python3 "$HERE/thinking_policy.py" --model "$MODEL" --think "$THINK" || exit $?
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
# be traced back to the cell and the moment that created it -- and so teardown
# can FIND it. Cell, epoch and pid together make the value this invocation's
# alone: two runs of the same cell, even in the same second, hold different
# pids, so a query by this label can never return somebody else's container.
# That is what makes asking Docker "what did I create?" safe where asking by
# name is not.
RUN_LABEL="atlas-campaign.run=$CELL_ID-$(date +%s)-$$"
# Written by vllm_control.sh only when its `docker run -d` actually created a
# container. Its absence is the proof that there is nothing to tear down.
CONTAINER_ID_FILE="$OUT/container.id"
CONTAINER_ID=""
# What that container is RUNNING, as opposed to what it was asked to run:
# vllm_control.sh writes the created container's resolved image ID here.
CONTAINER_IMAGE_FILE="$OUT/container.image"
# The name and the label are both chosen HERE and written down BEFORE the
# create, because the window that leaks is the one inside `docker run -d`: the
# container exists and its ID has not come back yet. A record written after the
# ID arrives cannot describe that window; this one can.
OWNER_JSON="$OUT/owner.json"
CREATE_ATTEMPTED=0
# The launcher identifies a launch by (run dir, PORT_BASE), and its --stop
# reaches only what its own run directory recorded. So the run directory has to
# be this invocation's ALONE: a directory shared with an earlier run holds that
# run's rank records, and a --stop aimed at it stops ranks this cell never
# started. The pid makes it exclusive, and the serve stage below CREATES it --
# `mkdir`, not `mkdir -p` -- so "this cell reserved it" is a fact rather than an
# assumption. ATLAS_NODE_RUN_DIR therefore names the prefix, not the directory.
NODE_RUN_DIR="${ATLAS_NODE_RUN_DIR:-$OUT/node-ep}-$$"
NODE_RUN_DIR_RESERVED=0

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

# The stage in progress, in the artifact schema's own vocabulary, so a cell
# that is killed mid-flight can name where it was killed. "The runner exited"
# is not a campaign result; "NO-GO at boot" is.
CURRENT_STAGE=""
stage() { CURRENT_STAGE="$1"; shift; step "$@"; }

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
# THIS CHECKOUT, and nothing about the engine. `git rev-parse HEAD` here
# describes run_cell.sh, the assembler and the recipes -- the harness. It used
# to be passed as engine_version.git_sha for either engine, which is how an
# artifact came to name a revision nothing had verified while the digest of the
# image that ran and the hash of the binary that ran were both null. It is
# harness provenance, and the engine's identity is read from the engine below.
HARNESS_GIT_SHA=""
# Filled in by capture_engine_identity, from what actually served the requests.
ENGINE_GIT_SHA=""
ENGINE_IMAGE_DIGEST=""
ENGINE_BINARY=""
ENGINE_VLLM_VERSION=""
IDENTITY_CAPTURED=0

# ── the finalizer: the ONE way out ───────────────────────────────────────────
# Everything this invocation created is released here and nowhere else, so the
# path through a signal is the same path as the path through a clean finish.
# The alternative -- teardown written inline where only normal control flow
# reaches it -- is what leaves a detached vLLM container holding the GPU when
# the campaign is interrupted, with no artifact to say the cell ever ran.
CHILD_PIDS=""          # pids this invocation started, space separated
SERVE_PID=""
FINALIZED=0
MAIN_DONE=0
INTERRUPT_SIG=""
EXTRA_NOTE=""

add_note() {
  if [ -n "$EXTRA_NOTE" ]; then EXTRA_NOTE="$EXTRA_NOTE; $1"; else EXTRA_NOTE="$1"; fi
}

# Kill one pid and the pids it started, by pid. `pgrep -P` walks parent links,
# so the kill stays inside this invocation's own process tree; a process-PATTERN
# kill would match this script's own command line, which is how a teardown
# becomes a self-kill.
kill_tree() {
  local pid="$1" kid
  for kid in $(pgrep -P "$pid" 2>/dev/null); do
    kill_tree "$kid"
  done
  kill "$pid" 2>/dev/null || true
}

forget_child() {
  local keep="" p
  for p in $CHILD_PIDS; do
    [ "$p" = "$1" ] || keep="$keep $p"
  done
  CHILD_PIDS="$keep"
}

# Run one stage's child in the background and wait for it. The backgrounding is
# what makes a signal actionable at all: bash defers a trap until the FOREGROUND
# command returns, and `wait` is the builtin a signal interrupts. Straight-line
# `bash time_to_ready.sh ...` would sit on a SIGTERM until the boot poll gave up
# 1800 s later. The pid is recorded so the finalizer can kill exactly this
# child, by pid.
run_child() {
  "$@" &
  local pid=$! rc=0
  CHILD_PIDS="$CHILD_PIDS $pid"
  wait "$pid" || rc=$?
  forget_child "$pid"
  return "$rc"
}

# Did the launcher record a rank of ours? It writes rank<N>.pid (local binary)
# or rank<N>.container (container mode) as it starts each one, into the run
# directory this cell reserved for itself. Because that directory is this
# invocation's alone, anything recorded in it can only be this invocation's --
# which is why THIS, and not "we backgrounded the launcher", is the ownership
# test. A launcher that refuses (exit 2 on an occupied port, a run directory it
# will not overwrite, a container name it did not create) creates nothing and
# records nothing, and a refusal must never be followed by a --stop.
#
# rank<N>.intent counts as much as rank<N>.container, because the container
# record only appears once `docker run -d` has RETURNED. A cell killed inside
# that create finds no container record and used to conclude that nothing was
# created -- while the container was up and stayed up. The intent is written
# before the create, so its presence is the evidence that a rank may exist, and
# the --stop it triggers is the thing that reconciles it. A refusal still
# writes no intent, so a refusal still stops nothing.
atlas_recorded_ranks() {
  local f
  for f in "$NODE_RUN_DIR"/rank*.pid "$NODE_RUN_DIR"/rank*.container \
           "$NODE_RUN_DIR"/rank*.intent; do
    [ -e "$f" ] && return 0
  done
  return 1
}

# Only what this invocation owns: the pids it started, plus the ranks or the
# container it created. Nothing here is addressed by name or by pattern.
teardown_owned() {
  local pid
  for pid in $CHILD_PIDS; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "killing pid $pid, started by this cell"
      kill_tree "$pid"
    fi
  done
  CHILD_PIDS=""

  if [ "$ENGINE" = "atlas" ]; then
    show "env ATLAS_NODE_RUN_DIR=$NODE_RUN_DIR bash $LAUNCHER --stop"
    if [ "$DRY_RUN" = "1" ]; then return 0; fi
    if [ "$NODE_RUN_DIR_RESERVED" != "1" ]; then
      echo "this invocation reserved no run directory of its own: nothing to stop."
      return 0
    fi
    if ! atlas_recorded_ranks; then
      echo "the launcher recorded no rank or create intent in $NODE_RUN_DIR -- it"
      echo "  refused or failed before creating anything, so this cell owns nothing"
      echo "  to stop."
      return 0
    fi
    # A --stop that could not account for a rank exits non-zero and KEEPS that
    # rank's intent record, so the cell reports the leak rather than closing
    # its teardown stage over it.
    if ! env "ATLAS_NODE_RUN_DIR=$NODE_RUN_DIR" bash "$LAUNCHER" --stop; then
      note_fail teardown
      add_note "the launcher could not account for every rank it recorded in $NODE_RUN_DIR: a rank this cell started may still be running and holding a GPU, and its intent record is kept there for a later --stop"
    fi
    return 0
  fi

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
    return 0
  fi

  # Three places the ID can be, in the order they become true:
  #   1. this shell, once the serve stage read it back;
  #   2. the id file, which vllm_control.sh writes the moment `docker run -d`
  #      returns -- a signal between the write and the read lands here;
  #   3. Docker itself, asked for whatever wears this invocation's label. That
  #      is the interrupted CREATE: the container exists, its ID was never
  #      printed, and the only thing that survives the window is the label
  #      chosen and recorded before the create. It identifies this invocation
  #      alone (cell, epoch, pid), so what comes back is this cell's own.
  local ids="$CONTAINER_ID" cid rc=0 out=""
  if [ -z "$ids" ] && [ -s "$CONTAINER_ID_FILE" ]; then
    ids="$(cat "$CONTAINER_ID_FILE")"
    echo "the id file holds $ids: the create finished, this shell never read it."
  fi
  if [ -z "$ids" ] && [ "$CREATE_ATTEMPTED" = "1" ]; then
    echo "no container ID reached this shell, and a create was attempted:"
    echo "  asking Docker for anything labelled $RUN_LABEL."
    # The lookup's exit status, read apart from its output. `|| true` used to
    # collapse a daemon that could not be reached into a daemon that answered
    # "nothing", and the branch below then reported a confirmed absence for a
    # container that was up. An unsuccessful lookup is neither presence nor
    # absence: nothing is stopped on a guess, and the cell records that a
    # container of its own may still be holding the GPU.
    out="$("${DOCKER:-docker}" ps -aq --filter "label=$RUN_LABEL" 2>&1)" || rc=$?
    if [ "$rc" -ne 0 ]; then
      echo "  the lookup FAILED (docker ps exited $rc): ${out:-(no output)}"
      echo "  Whether this cell's container exists is UNKNOWN, so nothing is stopped"
      echo "  by guesswork -- and it is not reported as absent either."
      note_fail teardown
      add_note "teardown could not reach Docker (docker ps exited $rc) to find the container labelled $RUN_LABEL: a container this cell created may still be running and holding the GPU"
      return 0
    fi
    ids="$out"
  fi

  if [ -z "$ids" ]; then
    echo "no container was created by this invocation ($CONTAINER_ID_FILE is absent"
    echo "  and nothing wears its label): nothing to stop or remove. A container"
    echo "  already holding the name '$CONTAINER' is not this cell's to delete."
    return 0
  fi
  for cid in $ids; do
    show "docker stop $cid && docker rm $cid"
    # A removal that failed is reported, for the same reason: the cell's own
    # artifact is the only place the leak is written down.
    if ! "${DOCKER:-docker}" stop "$cid" >/dev/null 2>&1; then
      echo "  docker stop $cid FAILED: the container may still be running."
      note_fail teardown
      add_note "teardown could not stop the container $cid this cell created: it may still be running and holding the GPU"
      continue
    fi
    if ! "${DOCKER:-docker}" rm "$cid" >/dev/null 2>&1; then
      echo "  docker rm $cid FAILED: the container is stopped but not removed."
      note_fail teardown
      add_note "teardown stopped the container $cid this cell created but could not remove it"
    fi
  done
}

# ── engine identity: what served the requests, asked of the engine ──────────
# One `docker image inspect` of the image that ran, or one hash of the binary
# that ran. Run from the finalizer BEFORE teardown, because teardown removes
# the container and deletes the launcher's record of what it was running --
# after that the only thing left to ask about is the image TAG, and a tag is a
# pointer that may have moved. Running it there also means a cell that was
# interrupted still says what it was running. Every lookup is best-effort --
# no daemon, no image, no label all leave the field null, which is the
# schema's "not measured".
docker_label() {  # docker_label REF TEMPLATE -> the value, or nothing
  local out
  out="$("${DOCKER:-docker}" image inspect --format "$2" "$1" 2>/dev/null || true)"
  case "$out" in
    ""|"<no value>"|"<nil>") ;;
    *) printf '%s\n' "$out" ;;
  esac
}

# The image ID rank 0's create resolved its tag to, written by the launcher
# the moment `docker run -d` returned (scripts/start-node-ep.sh,
# record_rank_image). An ID cannot move under a rebuild the way the tag it was
# resolved from can, so this -- not $IMAGE -- is what the identity is read of.
recorded_rank0_image() {
  local f="$NODE_RUN_DIR/rank0.image" line id=""
  [ -f "$f" ] || return 0
  while IFS= read -r line; do
    case "$line" in id=*) id="${line#id=}" ;; esac
  done < "$f"
  printf '%s' "$id"
}

capture_engine_identity() {
  [ "$IDENTITY_CAPTURED" = "1" ] && return 0
  IDENTITY_CAPTURED=1
  [ "$DRY_RUN" = "1" ] && return 0

  local ref="" digest="" rev=""
  if [ "$ENGINE" = "atlas" ] && [ -z "${IMAGE:-}" ]; then
    # The local binary. `spark --version` prints ATLAS_VERSION, which is
    # env!("CARGO_PKG_VERSION") and carries no revision
    # (crates/spark-server/src/cli.rs), so the hash of the file that ran is the
    # only identity there is to record -- and git_sha stays null rather than
    # being filled in with something that describes a different artefact.
    ENGINE_BINARY="${SPARK_BIN:-./target/release/spark}"
    [ -f "$ENGINE_BINARY" ] || ENGINE_BINARY=""
    return 0
  fi

  if [ "$ENGINE" = "atlas" ]; then
    ref="$(recorded_rank0_image)"
    if [ -z "$ref" ]; then
      # No create ever returned (a launch that was refused, or one interrupted
      # inside `docker run -d`), so there is no resolved ID to name. The tag is
      # the only reference left and it is only as good as the moment it is
      # read -- which the artifact says out loud rather than implying.
      ref="$IMAGE"
      add_note "engine identity read from the image tag $IMAGE: the launcher recorded no resolved image ID for rank 0, so a tag re-pointed during this run would not be visible here"
    fi
  else
    # vLLM's identity is its digest, and the digest is PINNED by the operator:
    # vllm_control.sh refuses to run without VLLM_IMAGE_DIGEST and builds the
    # reference as <repo>@<digest>, so what ran is what that names. The
    # version label is read of the SAME reference -- of the container's
    # resolved image where the create recorded one, else of the pinned
    # <repo>@<digest>, and only as a last resort of the floating tag.
    ENGINE_IMAGE_DIGEST="${VLLM_IMAGE_DIGEST:-}"
    ref="$(recorded_container_image)"
    if [ -z "$ref" ]; then
      local repo="${VLLM_IMAGE:-${VLLM_IMAGE_NAME:-}}"
      [ -n "$repo" ] || return 0
      if [ -n "${VLLM_IMAGE_DIGEST:-}" ]; then
        ref="${repo%%@*}@$VLLM_IMAGE_DIGEST"
      else
        ref="$repo"
        add_note "vLLM engine identity read from the image tag $repo: neither a resolved image ID nor a pinned digest was available"
      fi
    fi
    ENGINE_VLLM_VERSION="$(docker_label "$ref" '{{index .Config.Labels "org.opencontainers.image.version"}}')"
    return 0
  fi

  # RepoDigests reads back as <repo>@sha256:...; only the sha256 half is the
  # image's identity, and a tag-only image (never pushed, or built locally) has
  # no digest at all.
  digest="$(docker_label "$ref" '{{index .RepoDigests 0}}')"
  digest="${digest##*@}"
  case "$digest" in
    sha256:*) ENGINE_IMAGE_DIGEST="$digest" ;;
  esac
  # The revision the IMAGE declares. An image built without the label says
  # nothing, and nothing is what gets recorded.
  rev="$(docker_label "$ref" '{{index .Config.Labels "org.opencontainers.image.revision"}}')"
  if printf '%s' "$rev" | grep -Eq '^[0-9a-f]{7,40}$'; then
    ENGINE_GIT_SHA="$rev"
  fi
  return 0
}

# The same record on the vLLM side: vllm_control.sh knows the container ID the
# moment its create returns, and writes down what that container is running.
recorded_container_image() {
  local line id=""
  [ -f "$CONTAINER_IMAGE_FILE" ] || return 0
  while IFS= read -r line; do
    case "$line" in id=*) id="${line#id=}" ;; esac
  done < "$CONTAINER_IMAGE_FILE"
  printf '%s' "$id"
}

# Built at call time, not once: the failing stage and the interruption note are
# only known when the cell is over, however it got there.
build_assemble() {
  local note="$WARMUP_NOTE"
  ASSEMBLE=( python3 "$HERE/cell_assemble.py"
    --engine "$ENGINE" --model-key "$MODEL" --sku "$SKU" --workload "$WORKLOAD"
    --concurrency "$CONC" --spec "$SPEC" --think "$THINK" --out "$ARTIFACT"
    --workloads "$WORKLOADS" --atlas-recipes "$ATLAS_RECIPES"
    --vllm-recipes "$VLLM_RECIPES" --client "$LADDER"
    --serve-argv "$SERVE_ARGV" --serve-env "$SERVE_ENV" --nvidia-smi-q "$SMI_Q"
    --boot-json "$BOOT_JSON" --coherency-json "$COH_JSON" --ladder-json "$LADDER_JSON" )
  [ -n "$HARNESS_GIT_SHA" ] && ASSEMBLE+=( --harness-git-sha "$HARNESS_GIT_SHA" )
  [ -n "$ENGINE_GIT_SHA" ] && ASSEMBLE+=( --git-sha "$ENGINE_GIT_SHA" )
  [ -n "$ENGINE_IMAGE_DIGEST" ] && ASSEMBLE+=( --image-digest "$ENGINE_IMAGE_DIGEST" )
  [ -n "$ENGINE_BINARY" ] && ASSEMBLE+=( --binary "$ENGINE_BINARY" )
  [ -n "$ENGINE_VLLM_VERSION" ] && ASSEMBLE+=( --vllm-version "$ENGINE_VLLM_VERSION" )
  [ -n "$PAIRED" ] && ASSEMBLE+=( --paired-artifact "$PAIRED" )
  [ -n "$PTX_RECEIPT" ] && ASSEMBLE+=( --ptx-receipt "$PTX_RECEIPT" )
  if [ -n "$EXTRA_NOTE" ]; then note="${note:+$note; }$EXTRA_NOTE"; fi
  [ -n "$note" ] && ASSEMBLE+=( --extra-note "$note" )
  [ -n "$FAILING_STAGE" ] && ASSEMBLE+=( --failing-stage "$FAILING_STAGE" )
  return 0
}

# Idempotent: the EXIT trap fires after a signal handler and after normal
# completion, and neither must tear the same thing down twice.
finalize() {
  local rc=$? sig_rc=""
  if [ "$FINALIZED" = "1" ]; then return "$rc"; fi
  FINALIZED=1

  if [ -n "$INTERRUPT_SIG" ]; then
    case "$INTERRUPT_SIG" in
      INT) sig_rc=130 ;;
      HUP) sig_rc=129 ;;
      *)   sig_rc=143 ;;
    esac
    echo ""
    echo "!!! SIG$INTERRUPT_SIG during stage '${CURRENT_STAGE:-preflight}': releasing what"
    echo "    this cell created, then writing the artifact that says so."
    note_fail "${CURRENT_STAGE:-preflight}"
    add_note "interrupted by SIG$INTERRUPT_SIG during the ${CURRENT_STAGE:-preflight} stage: this cell was terminated before it finished, so its gates are unfinished rather than passed"
  fi

  # Before teardown, not after: teardown removes the container this cell's
  # identity is read of, and takes the launcher's record of it with it.
  capture_engine_identity

  step "stage 6/7 teardown"
  teardown_owned

  step "stage 7/7 assemble and validate"
  build_assemble
  show "${ASSEMBLE[*]}"
  show "python3 $HERE/validate_artifact.py $ARTIFACT"

  if [ "$DRY_RUN" = "1" ]; then
    echo ""
    echo "dry-run: nothing launched, nothing written."
    exit "${sig_rc:-$rc}"
  fi

  "${ASSEMBLE[@]}" || note_fail validate
  if [ ! -f "$ARTIFACT" ]; then
    echo "ERROR: the artifact was not written; there is nothing to validate." >&2
    exit "${sig_rc:-1}"
  fi
  python3 "$HERE/validate_artifact.py" "$ARTIFACT" || {
    note_fail validate
    # Re-assemble so the artifact's own verdict admits the validation failure
    # rather than claiming a verdict its shape does not support.
    build_assemble
    "${ASSEMBLE[@]}" --failing-stage validate >/dev/null
  }

  echo ""
  if [ -n "$sig_rc" ]; then
    echo "=== $CELL_ID: TERMINATED by SIG$INTERRUPT_SIG at stage '$FAILING_STAGE'; artifact written to $ARTIFACT ==="
    exit "$sig_rc"
  fi
  if [ -n "$FAILING_STAGE" ]; then
    echo "=== $CELL_ID: FAILED at stage '$FAILING_STAGE'; artifact written to $ARTIFACT ==="
    exit 1
  fi
  if [ "$MAIN_DONE" != "1" ]; then
    # An exit that came from neither the main flow nor a handled signal: the
    # cell is torn down and recorded, but the original status is its own.
    exit "$rc"
  fi
  echo "=== $CELL_ID: all gates passed; artifact written to $ARTIFACT ==="
  echo "    Verdict is PARTIAL until the paired cell from the other engine exists"
  echo "    within 24 h; re-run with --paired-artifact to promote it."
  exit 0
}

# A handler does none of the work itself: it records WHICH signal and goes
# through the same finalizer a clean finish does.
on_signal() {
  [ -n "$INTERRUPT_SIG" ] || INTERRUPT_SIG="$1"
  finalize
  # Reached only if the finalizer had already run (a second signal during
  # teardown). The process still has to end, and with this signal's status.
  case "$INTERRUPT_SIG" in INT) exit 130 ;; HUP) exit 129 ;; *) exit 143 ;; esac
}

trap 'on_signal TERM' TERM
trap 'on_signal INT' INT
trap 'on_signal HUP' HUP
trap finalize EXIT
# INT is trapped for the interactive Ctrl-C, which is delivered to the
# foreground process group. A shell STARTED in the background inherits SIGINT
# ignored, and a signal ignored on entry cannot be trapped -- bash's rule, not
# this script's. So a campaign driver that backgrounds its cells must terminate
# them with TERM, which is also what the regression in campaign_test.sh sends.

# ── 1. preflight ─────────────────────────────────────────────────────────────
stage preflight "stage 1/7 preflight"
show "nvidia-smi -q > $SMI_Q"
show "df -h $OUT > $OUT/df.txt"
show "git -C $ROOT rev-parse HEAD"
show "sha256sum $LADDER"
if [ "$DRY_RUN" != "1" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -q > "$SMI_Q" 2>"$OUT/nvidia-smi-q.err" || note_fail preflight
  else
    echo "no nvidia-smi on this host" > "$OUT/nvidia-smi-q.err"
    note_fail preflight
  fi
  df -h "$OUT" > "$OUT/df.txt" 2>&1
  HARNESS_GIT_SHA="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || true)"
  if [ -n "$(git -C "$ROOT" status --porcelain 2>/dev/null)" ]; then
    echo "NOTE: the tree is dirty; harness.git_sha $HARNESS_GIT_SHA does not fully"
    echo "  describe it."
  fi
fi

# ── 2. serve ─────────────────────────────────────────────────────────────────
stage serve "stage 2/7 serve"
START_EPOCH=""
if [ "$ENGINE" = "atlas" ]; then
  ATLAS_ENV=(
    "NGPUS=$NGPUS" "EP_SIZE=$EP_SIZE" "TP_SIZE=$TP_SIZE"
    "PORT_BASE=$PORT" "BIND=0.0.0.0" "NCCL_PROFILE=default"
    "BOOT_TIMEOUT_S=$BOOT_CAP" "EXTRA_ARGS=$EXTRA_ARGS"
    "ATLAS_NODE_RUN_DIR=$NODE_RUN_DIR"
  )
  [ -n "$WARMUP_PROMPT" ] && ATLAS_ENV+=( "WARMUP_PROMPT=$WARMUP_PROMPT" )
  [ -n "${SPARK_BIN:-}" ] && ATLAS_ENV+=( "SPARK_BIN=$SPARK_BIN" )
  [ -n "${IMAGE:-}" ] && ATLAS_ENV+=( "IMAGE=$IMAGE" )

  show "env ${ATLAS_ENV[*]} bash $LAUNCHER --check-kernels $HF_ID"
  show "env ${ATLAS_ENV[*]} bash $LAUNCHER $HF_ID   # backgrounded; time_to_ready.sh owns the boot clock"
  if [ "$DRY_RUN" = "1" ]; then
    env "${ATLAS_ENV[@]}" bash "$LAUNCHER" --dry-run "$HF_ID" 2>&1 | sed 's/^/  | /' \
      || exit $?
  else
    # `mkdir`, deliberately without -p: this cell must CREATE the directory it
    # hands the launcher. A directory that already exists may hold an earlier
    # run's rank records, and the finalizer's --stop reaches whatever the
    # directory records -- which is how a launch that was refused before it
    # created anything went on to stop somebody else's ranks.
    if mkdir -p "$(dirname "$NODE_RUN_DIR")" 2>/dev/null \
       && mkdir "$NODE_RUN_DIR" 2>/dev/null; then
      NODE_RUN_DIR_RESERVED=1
      echo "run dir:     $NODE_RUN_DIR (reserved by this cell)"
    else
      echo "REFUSED: $NODE_RUN_DIR could not be reserved for this cell."
      echo "  It already exists, so whatever it records is not this invocation's"
      echo "  to stop later. Nothing is launched."
      note_fail serve
    fi
    printf '%s\n' "${ATLAS_ENV[@]}" > "$SERVE_ENV"
    env "${ATLAS_ENV[@]}" bash "$LAUNCHER" --dry-run "$HF_ID" > "$OUT/serve-dryrun.txt" 2>&1
    python3 - "$OUT/serve-dryrun.txt" "$SERVE_ARGV" <<'PY'
import pathlib, shlex, sys
line = next((l for l in pathlib.Path(sys.argv[1]).read_text().splitlines()
             if l.startswith("rank0_command: ")), None)
argv = shlex.split(line[len("rank0_command: "):]) if line else []
pathlib.Path(sys.argv[2]).write_bytes(b"\0".join(a.encode() for a in argv))
PY
    if [ "$NODE_RUN_DIR_RESERVED" = "1" ]; then
      run_child env "${ATLAS_ENV[@]}" bash "$LAUNCHER" --check-kernels "$HF_ID" \
        > "$OUT/check-kernels.txt" 2>&1 || note_fail serve
      START_EPOCH="$(date +%s)"
      env "${ATLAS_ENV[@]}" bash "$LAUNCHER" "$HF_ID" > "$OUT/serve.log" 2>&1 &
      SERVE_PID=$!
      CHILD_PIDS="$CHILD_PIDS $SERVE_PID"
      # The launcher pid is this invocation's to KILL. What is this invocation's
      # to STOP is decided later, by what the launcher actually recorded in the
      # run directory above -- a launcher that refuses records nothing.
      echo "launcher pid $SERVE_PID; log $OUT/serve.log"
    fi
  fi
else
  VC_ARGS=( "$MODEL" "$SKU" --spec "$SPEC" --label "$RUN_LABEL" )
  show "VLLM_CONTAINER=$CONTAINER bash $HERE/vllm_control.sh ${VC_ARGS[*]} --id-file $CONTAINER_ID_FILE --image-file $CONTAINER_IMAGE_FILE"
  if [ "$DRY_RUN" = "1" ]; then
    VLLM_CONTAINER="$CONTAINER" VLLM_RECIPES="$VLLM_RECIPES" \
      bash "$HERE/vllm_control.sh" "${VC_ARGS[@]}" --dry-run 2>&1 | sed 's/^/  | /' \
      || exit $?
  else
    printf 'VLLM_IMAGE_DIGEST=%s\n' "${VLLM_IMAGE_DIGEST:-}" > "$SERVE_ENV"
    # Ownership is written down BEFORE the create, not after it: from the next
    # line on, a container of this cell's may exist whether or not its ID ever
    # comes back, and the finalizer has to be able to find it either way.
    cat > "$OWNER_JSON" <<OWNER
{
  "cell_id": "$CELL_ID",
  "container_name": "$CONTAINER",
  "run_label": "$RUN_LABEL",
  "container_id_file": "$CONTAINER_ID_FILE",
  "runner_pid": $$
}
OWNER
    CREATE_ATTEMPTED=1
    START_EPOCH="$(date +%s)"
    run_child env VLLM_CONTAINER="$CONTAINER" VLLM_RECIPES="$VLLM_RECIPES" \
      bash "$HERE/vllm_control.sh" "${VC_ARGS[@]}" --id-file "$CONTAINER_ID_FILE" \
      --image-file "$CONTAINER_IMAGE_FILE" \
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
stage boot "stage 3/7 boot gate (cap ${BOOT_CAP}s)"
show "bash $TTR --url $URL --model $HF_ID --engine $ENGINE --start-epoch <serve-start> --timeout-s $BOOT_CAP --out $BOOT_JSON"
if [ "$DRY_RUN" != "1" ] && [ -z "$FAILING_STAGE" ]; then
  run_child bash "$TTR" --url "$URL" --model "$HF_ID" --engine "$ENGINE" \
       --start-epoch "$START_EPOCH" --timeout-s "$BOOT_CAP" --out "$BOOT_JSON" \
    || note_fail boot
fi

# ── 4. coherency gate ────────────────────────────────────────────────────────
stage coherency "stage 4/7 coherency gate"
show "python3 $COHERENCY --url $URL --model $HF_ID --think $THINK --out $COH_JSON"
if [ "$DRY_RUN" != "1" ] && [ -z "$FAILING_STAGE" ]; then
  run_child python3 "$COHERENCY" --url "$URL" --model "$HF_ID" --think "$THINK" --out "$COH_JSON" \
    || note_fail coherency
fi

# ── 5. latency pack ──────────────────────────────────────────────────────────
stage ladder "stage 5/7 latency pack"
show "python3 $LADDER --url $URL --model $HF_ID --label $CELL_ID --out $LADDER_JSON --concs $CONC --reps $REPS --isl $ISL --osl $OSL --warmup $WARMUP ${LADDER_THINK[*]:-}"
if [ "$DRY_RUN" != "1" ] && [ -z "$FAILING_STAGE" ]; then
  run_child python3 "$LADDER" --url "$URL" --model "$HF_ID" --label "$CELL_ID" \
          --out "$LADDER_JSON" --concs "$CONC" --reps "$REPS" \
          --isl "$ISL" --osl "$OSL" --warmup "$WARMUP" "${LADDER_THINK[@]}" || note_fail ladder
fi

# ── 6 + 7. teardown, assemble, validate ─────────────────────────────────────
# Not written out again here: these stages ARE the finalizer above, which is
# the same code a SIGINT/SIGTERM/SIGHUP takes. One teardown, one artifact, one
# way out.
MAIN_DONE=1
finalize
