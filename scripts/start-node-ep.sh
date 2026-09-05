#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Launch Atlas on ONE host with N GPUs: N `spark` processes, one per GPU,
# rank i on GPU i, all bootstrapping NCCL over 127.0.0.1.
#
# WHY THIS EXISTS. Every multi-rank Atlas deployment so far has been two DGX
# Spark nodes, one GB10 each, over RoCE — `scripts/start-ep2.sh`. That script
# pins an NCCL environment (NCCL_SOCKET_IFNAME=enp1s0f0np0, NCCL_IB_HCA,
# NCCL_NVLS_ENABLE=0, NCCL_NET_GDR_LEVEL=0, NCCL_NET_GDR_C2C=0,
# NCCL_DMABUF_ENABLE=0, NCCL_PROTO=Simple, NCCL_ALGO=Ring, MAX_NCHANNELS=2)
# that is correct for GB10-over-RoCE and actively WRONG on an H100/H200/B200
# box: it names a NIC that does not exist there, disables NVLink SHARP on a
# machine that has NVLink, and forces the slowest protocol/algorithm pair onto
# an intra-node transport that would otherwise use NVLink P2P. On one node with
# NVLink the right NCCL configuration is *no* NCCL configuration.
#
# See the "Single node, N GPUs (Hopper / B200)" section of docs/DEPLOYMENT.md
# and the campaign PRD §6.2
# (https://github.com/Avarok-Cybersecurity/atlas/issues/899 — campaign PRD).
#
# TWO DECISIONS WORTH READING BEFORE YOU CHANGE THEM
#
#  1. GPU pinning uses `--gpu-ordinal i`, NOT `CUDA_VISIBLE_DEVICES=i`.
#     `args.gpu_ordinal` is handed straight to `AtlasCudaBackend::new(ordinal)`
#     (crates/spark-server/src/main_modules/serve_phases/preflight.rs:332) and
#     to the arch preflight above it, so it already selects the device. Leaving
#     every GPU visible to every rank keeps NCCL's view of the node complete,
#     so it can pick NVLink/P2P transports between peers instead of falling
#     back. Masking with CUDA_VISIBLE_DEVICES=i is the other common idiom and
#     would also work (the rank would then need `--gpu-ordinal 0`, since the
#     mask renumbers devices), but it hides the topology from NCCL for no gain
#     here. Pick ONE — never both, or rank i lands on the wrong die.
#     UNVERIFIED: no NCCL init has been observed from this script; there is no
#     multi-GPU NVLink box in reach.
#
#  3. OWNERSHIP. A launch is identified by (run dir, PORT_BASE): container
#     names carry that identity, so two launchers with different run
#     directories or ports can never name the same container and `--stop`
#     can only reach what its own run directory recorded. Nothing is ever
#     force-removed to make room -- an already-existing container name, a run
#     directory that still holds live rank records, or a port whose /health
#     already answers are all REFUSALS (exit 2). Each of those is somebody
#     else's live server, and the previous `docker rm -f` deleted it.
#     What a record written AFTER the create cannot describe is the create
#     itself: `docker run -d` makes the container and prints its ID after, and
#     a launch killed in that window wrote no rank<N>.container at all -- so
#     its caller read "nothing was created" and stopped nothing, leaving a
#     rank on the GPU. So the name and a per-launch label are written to
#     rank<N>.intent BEFORE each create, the label goes on the container, and
#     --stop reconciles an intent that has no container record by asking
#     Docker for that exact name wearing that label. Both filters, always:
#     the name alone would match a LATER launch's rank, and the label alone
#     is not a thing a stray container can be found by if the create never
#     reached Docker.
#
#  2. Only rank 0 serves HTTP. This is not a convention, it is the code:
#     `maybe_run_ep_worker` (serve_load.rs:752) returns `Ok(None)` for rank > 0
#     *before* the router is ever built — "An EP worker (rank > 0) never serves
#     HTTP". Ranks 1..N-1 are still given `--port PORT_BASE+i` so that a stray
#     bind can never collide, but that port is not listened on. Point the
#     benchmark client at PORT_BASE only.
#
# Usage:
#   scripts/start-node-ep.sh [OPTIONS] [MODEL]
#
# Options:
#   --dry-run           Print every command that would run, launch nothing.
#   --check-kernels     Run rank 0 alone with --check-kernels --no-tui and exit
#                       with its status (the count of unresolved kernels).
#   --stop              Kill the ranks recorded in $ATLAS_NODE_RUN_DIR and exit.
#                       Only that run directory's own ranks: the names it wrote
#                       carry its (run dir, PORT_BASE) identity. A rank whose
#                       create was interrupted left an INTENT and no container
#                       record; --stop reconciles it by asking Docker for that
#                       exact name wearing that launch's own run label.
#   --stop-on-timeout   On boot timeout, stop the ranks instead of leaving them
#                       up for inspection.
#   -h | --help         This header.
#
# Environment:
#   NGPUS          ranks to start (default: `nvidia-smi -L | wc -l`)
#   EP_SIZE        expert-parallel width  (default: NGPUS)
#   TP_SIZE        tensor-parallel width  (default: 1)
#   PORT_BASE      rank i gets --port PORT_BASE+i (default 8888; only rank 0
#                  actually listens)
#   BIND           rank 0 HTTP bind address (default 127.0.0.1)
#   MASTER_ADDR    NCCL bootstrap address (default 127.0.0.1 — single node)
#   MASTER_PORT    NCCL bootstrap port (default 29500)
#   IMAGE          empty (default) = run $SPARK_BIN directly on the host;
#                  set = run each rank as `docker run` from that image
#   SPARK_BIN      local binary (default ./target/release/spark)
#   DOCKER         docker command (default "docker"; set "sudo docker" if needed)
#   HF_CACHE       HF cache to mount in container mode (default ~/.cache/huggingface)
#   EXTRA_ARGS     appended verbatim to EVERY rank — this is how speculative
#                  flags stay identical across ranks, which they must be
#                  (QUICKSTART.md:328-333). Do NOT put topology flags here.
#   NCCL_PROFILE   default | debug | gb10-roce   (default: default)
#   WARMUP_PROMPT  path to a warmup prompt file -> --warmup-prompt on every rank
#   BOOT_TIMEOUT_S readiness deadline in seconds (default 1800 = the PRD cap)
#   RUST_LOG       log filter passed to every rank (default info)
#   ATLAS_NODE_RUN_DIR   pid files + logs (default /tmp/atlas-node-ep)
#   ATLAS_NODE_HEALTH_URL  override the polled health URL (testing hook)
#
# Examples:
#   # 4×H100, pure EP, local binary, NCCL defaults
#   NGPUS=4 scripts/start-node-ep.sh nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8
#
#   # 8×H200 DeepSeek V4-Flash: MQA means num_key_value_heads=1, so TP is
#   # impossible and EP is the only axis.
#   NGPUS=8 EP_SIZE=8 TP_SIZE=1 EXTRA_ARGS="--kv-cache-dtype fp8 --max-batch-size 1" \
#     scripts/start-node-ep.sh deepseek-ai/DeepSeek-V4-Flash
#
#   scripts/start-node-ep.sh --stop
set -euo pipefail

DRY_RUN=0
CHECK_KERNELS=0
STOP=0
STOP_ON_TIMEOUT=0
MODEL=""

usage() { sed -n '2,114p' "$0"; }

while [ $# -gt 0 ]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --check-kernels) CHECK_KERNELS=1; shift ;;
    --stop) STOP=1; shift ;;
    --stop-on-timeout) STOP_ON_TIMEOUT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    --) shift; break ;;
    -*) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    *)
      if [ -n "$MODEL" ]; then
        echo "unexpected extra positional argument: $1" >&2
        echo "(pass serve flags through EXTRA_ARGS, not on the command line)" >&2
        exit 2
      fi
      MODEL="$1"; shift ;;
  esac
done
if [ $# -gt 0 ] && [ -z "$MODEL" ]; then MODEL="$1"; shift; fi

RUN_DIR="${ATLAS_NODE_RUN_DIR:-/tmp/atlas-node-ep}"
DOCKER="${DOCKER:-docker}"

# ── --stop: kill exactly what this script recorded ────────────────────────────
# By pid file and container name, never `pkill -f`. A `pkill -f spark` here
# would match this script's own command line (and any editor with the word in
# an argument), which is how a stop turns into a self-kill.
# reconcile_intent FILE -> 0 if it stopped a container of this launch's
# An intent is what a create wrote before it ran: the deterministic container
# name and the label this launch stamps on every rank it creates. Asking Docker
# for BOTH is what makes the answer this launch's own -- the name on its own
# belongs to whoever holds it now, which after a crashed run may be a later
# launch, and the label on its own is not enough to find anything if the create
# never reached the daemon. Nothing here is force-removed and nothing is
# matched by pattern.
reconcile_intent() {
  local f="$1" line name="" label="" ids id found=1
  while IFS= read -r line; do
    case "$line" in
      name=*) name="${line#name=}" ;;
      label=*) label="${line#label=}" ;;
    esac
  done < "$f"
  if [ -z "$name" ] || [ -z "$label" ]; then
    echo "intent $(basename "$f") names no container (name='$name' label='$label'):"
    echo "  nothing is stopped by guesswork"
    return 1
  fi
  echo "reconciling the interrupted create of $name (label $label)"
  ids="$("$DOCKER" ps -aq --filter "name=^$name\$" --filter "label=$label" 2>/dev/null || true)"
  if [ -z "$ids" ]; then
    echo "  nothing wears that name and that label: the create never reached Docker"
    return 1
  fi
  for id in $ids; do
    echo "  stopping container $id"
    "$DOCKER" stop "$id" >/dev/null 2>&1 || true
    "$DOCKER" rm "$id" >/dev/null 2>&1 || true
    found=0
  done
  return "$found"
}

stop_ranks() {
  local f pid name stopped=0
  if [ ! -d "$RUN_DIR" ]; then
    echo "no run directory at $RUN_DIR — nothing to stop"
    return 0
  fi
  # Intents first, so an intent whose create DID return is left to the
  # container record below -- that record is the authoritative one.
  for f in "$RUN_DIR"/rank*.intent; do
    [ -e "$f" ] || continue
    if [ -e "${f%.intent}.container" ]; then
      rm -f "$f"
      continue
    fi
    if reconcile_intent "$f"; then
      stopped=$((stopped + 1))
    fi
    rm -f "$f"
  done
  for f in "$RUN_DIR"/rank*.container; do
    [ -e "$f" ] || continue
    name="$(cat "$f")"
    echo "stopping container $name"
    "$DOCKER" stop "$name" >/dev/null 2>&1 || true
    "$DOCKER" rm "$name" >/dev/null 2>&1 || true
    rm -f "$f"
    stopped=$((stopped + 1))
  done
  for f in "$RUN_DIR"/rank*.pid; do
    [ -e "$f" ] || continue
    pid="$(cat "$f")"
    if [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; then
      echo "stopping pid $pid"
      kill "$pid" 2>/dev/null || true
    else
      echo "pid ${pid:-?} already gone"
    fi
    rm -f "$f"
    stopped=$((stopped + 1))
  done
  echo "stopped $stopped rank(s); logs kept in $RUN_DIR"
}

if [ "$STOP" = "1" ]; then
  stop_ranks
  exit 0
fi

# ── Defaults ─────────────────────────────────────────────────────────────────
if [ -z "$MODEL" ]; then
  echo "ERROR: MODEL is required (HF id or local path)." >&2
  usage >&2
  exit 2
fi

detect_ngpus() {
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi -L 2>/dev/null | grep -c '^GPU ' || true
  else
    echo 0
  fi
}

NGPUS="${NGPUS:-$(detect_ngpus)}"
EP_SIZE="${EP_SIZE:-$NGPUS}"
TP_SIZE="${TP_SIZE:-1}"
PORT_BASE="${PORT_BASE:-8888}"
BIND="${BIND:-127.0.0.1}"
MASTER_ADDR="${MASTER_ADDR:-127.0.0.1}"
MASTER_PORT="${MASTER_PORT:-29500}"
IMAGE="${IMAGE:-}"
SPARK_BIN="${SPARK_BIN:-./target/release/spark}"
HF_CACHE="${HF_CACHE:-$HOME/.cache/huggingface}"
EXTRA_ARGS="${EXTRA_ARGS:-}"
NCCL_PROFILE="${NCCL_PROFILE:-default}"
WARMUP_PROMPT="${WARMUP_PROMPT:-}"
BOOT_TIMEOUT_S="${BOOT_TIMEOUT_S:-1800}"
RUST_LOG="${RUST_LOG:-info}"
HEALTH_URL="${ATLAS_NODE_HEALTH_URL:-http://127.0.0.1:$PORT_BASE/health}"

# ── Ownership identity ───────────────────────────────────────────────────────
# (run dir, PORT_BASE) names this invocation. The container name used to be the
# global `atlas-node-ep-rankN` regardless of either, so a second launcher on a
# different port force-removed the first one's live rank, and the first one's
# saved container-name file then pointed at the second one's containers. The
# name is derived rather than random so it stays readable in `docker ps` and so
# a re-launch into the SAME run dir and port collides on purpose -- that is the
# case that should be refused, not silently resolved.
RUN_TAG="$(printf '%s' "$(basename "$RUN_DIR")" | tr -c 'A-Za-z0-9_.-' '-')"
case "$RUN_TAG" in
  atlas-*) ;;
  *) RUN_TAG="atlas-$RUN_TAG" ;;
esac
RUN_ID="$RUN_TAG-$PORT_BASE"
# The value that survives an interrupted create, because it is chosen before
# the create and written down with the name it belongs to. RUN_ID keeps it
# readable in `docker ps`; the epoch and the pid make it THIS invocation's, so
# a later launch that derives the same container name can never answer this
# one's reconciliation query.
RUN_LABEL="atlas-node-ep.run=$RUN_ID-$(date +%s)-$$"

# ── Validation ───────────────────────────────────────────────────────────────
is_positive_int() { case "$1" in ''|*[!0-9]*) return 1 ;; 0) return 1 ;; *) return 0 ;; esac; }

for pair in "NGPUS=$NGPUS" "EP_SIZE=$EP_SIZE" "TP_SIZE=$TP_SIZE" \
            "PORT_BASE=$PORT_BASE" "MASTER_PORT=$MASTER_PORT" \
            "BOOT_TIMEOUT_S=$BOOT_TIMEOUT_S"; do
  name="${pair%%=*}"; value="${pair#*=}"
  if ! is_positive_int "$value"; then
    echo "ERROR: $name must be a positive integer, got '$value'." >&2
    if [ "$name" = "NGPUS" ] && [ "$value" = "0" ]; then
      echo "       No GPU was detected by 'nvidia-smi -L'. Set NGPUS explicitly." >&2
    fi
    exit 2
  fi
done

# The world-size rule, straight out of `resolve_topology`
# (crates/spark-server/src/main_modules/serve_phases/topology.rs:51-63): either
# an orthogonal mesh (world == tp × ep) or overlapping groups (world == tp == ep).
# Checked here so a bad topology costs a second on the shell rather than N
# processes that each load weights and then bail.
if [ "$((TP_SIZE * EP_SIZE))" -ne "$NGPUS" ] && \
   ! { [ "$TP_SIZE" -eq "$EP_SIZE" ] && [ "$TP_SIZE" -eq "$NGPUS" ]; }; then
  echo "ERROR: invalid parallelism topology for one node." >&2
  echo "       NGPUS (world size) = $NGPUS, TP_SIZE = $TP_SIZE, EP_SIZE = $EP_SIZE." >&2
  echo "       Atlas requires world_size == tp_size * ep_size (orthogonal mesh, here" >&2
  echo "       $TP_SIZE * $EP_SIZE = $((TP_SIZE * EP_SIZE))) or world_size == tp_size == ep_size" >&2
  echo "       (overlapping groups). Neither holds." >&2
  echo "       Fix: set EP_SIZE=$NGPUS TP_SIZE=1 (pure EP), or make TP_SIZE*EP_SIZE=$NGPUS." >&2
  exit 2
fi

case "$NCCL_PROFILE" in
  default|debug|gb10-roce) ;;
  *)
    echo "ERROR: NCCL_PROFILE must be one of: default, debug, gb10-roce (got '$NCCL_PROFILE')." >&2
    exit 2 ;;
esac

if [ -n "$WARMUP_PROMPT" ] && [ ! -f "$WARMUP_PROMPT" ] && [ "$DRY_RUN" != "1" ]; then
  echo "ERROR: WARMUP_PROMPT='$WARMUP_PROMPT' is not a readable file." >&2
  exit 2
fi

if [ -z "$IMAGE" ] && [ "$DRY_RUN" != "1" ] && [ ! -x "$SPARK_BIN" ]; then
  echo "ERROR: SPARK_BIN='$SPARK_BIN' is not executable." >&2
  echo "       Build it (cargo build --release) or set IMAGE=<atlas image> for container mode." >&2
  exit 2
fi

# ── NCCL profiles ────────────────────────────────────────────────────────────
# default:   nothing at all. On an NVLink node NCCL's own topology detection
#            beats anything written here, and every variable below is a cap.
# debug:     defaults plus logging, for the FIRST boot on a new box. Read the
#            NET/ section of the log to confirm which transport was chosen.
# gb10-roce: the pessimized GB10 block copied from scripts/start-ep2.sh, kept
#            only so an A/B against the two-Spark deployment is possible. Do not
#            use it on Hopper/Blackwell to make numbers.
NCCL_ENV=()
case "$NCCL_PROFILE" in
  default) ;;
  debug)
    NCCL_ENV=( NCCL_DEBUG=INFO "NCCL_DEBUG_SUBSYS=INIT,NET" )
    ;;
  gb10-roce)
    NCCL_ENV=(
      "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-enp1s0f0np0}"
      NCCL_IB_DISABLE=0
      "NCCL_IB_HCA=${NCCL_IB_HCA:-rocep1s0f0}"
      NCCL_IB_ROCE_VERSION_NUM=2
      NCCL_IB_ADDR_FAMILY=AF_INET
      NCCL_IB_TIMEOUT=22
      NCCL_IB_RETRY_CNT=7
      NCCL_NET_GDR_LEVEL=0
      NCCL_NET_GDR_C2C=0
      NCCL_DMABUF_ENABLE=0
      NCCL_NVLS_ENABLE=0
      NCCL_CUMEM_HOST_ENABLE=0
      NCCL_PROTO=Simple
      NCCL_ALGO=Ring
      NCCL_MIN_NCHANNELS=1
      NCCL_MAX_NCHANNELS=2
      NCCL_DEBUG=INFO
      "NCCL_DEBUG_SUBSYS=INIT,NET"
    )
    ;;
esac

# EXTRA_ARGS is deliberately word-split: it is a flag string, and every rank
# must receive the same tokens for the speculative-flag parity rule to hold.
EXTRA_ARR=()
if [ -n "$EXTRA_ARGS" ]; then
  # shellcheck disable=SC2206  # word splitting is the point
  EXTRA_ARR=( $EXTRA_ARGS )
fi

# Inside a container the warmup file has to exist at a container path.
WARMUP_HOST="$WARMUP_PROMPT"
WARMUP_IN_CONTAINER="/warmup/$(basename "${WARMUP_PROMPT:-none}")"

# ── Command construction ─────────────────────────────────────────────────────
container_name() { printf '%s-rank%s' "$RUN_ID" "$1"; }

# `docker inspect` on a name: exit 0 iff a container by that name exists, and
# `true` on stdout iff it is running. One command answers both questions.
container_exists() { "$DOCKER" inspect --format '{{.State.Running}}' "$1" >/dev/null 2>&1; }
container_running() {
  [ "$("$DOCKER" inspect --format '{{.State.Running}}' "$1" 2>/dev/null)" = "true" ]
}

# Is rank 0 -- the rank this script launched -- still alive?
#
# This is the difference between "the endpoint answers" and "MY endpoint
# answers". A rank that failed to bind while an older server still owns the
# port leaves that server answering 200, and a poll that only looks at the
# HTTP code reports a boot time for a process that is gone.
#
# A local rank is a background job of this shell, so after it exits it is a
# ZOMBIE until reaped: `kill -0` still succeeds on it. The process state is
# what actually distinguishes the two.
rank0_alive() {
  if [ -n "$IMAGE" ]; then
    container_running "$(container_name 0)"
    return $?
  fi
  local pid state
  pid="$(cat "$RUN_DIR/rank0.pid" 2>/dev/null || true)"
  [ -n "$pid" ] || return 1
  kill -0 "$pid" 2>/dev/null || return 1
  state="$(ps -o state= -p "$pid" 2>/dev/null | tr -d ' ')"
  case "$state" in
    Z*) return 1 ;;
    '') return 1 ;;
    *) return 0 ;;
  esac
}

rank_log_tail() {  # rank_log_tail RANK -> 40 lines, wherever they live
  if [ -n "$IMAGE" ]; then
    "$DOCKER" logs --tail 40 "$(container_name "$1")" 2>&1 \
      || echo "(no container logs for $(container_name "$1"))"
  elif [ -f "$RUN_DIR/rank$1.log" ]; then
    tail -n 40 "$RUN_DIR/rank$1.log"
  else
    echo "(no log at $RUN_DIR/rank$1.log)"
  fi
}

# shquote ARGS... -> a single copy-pasteable shell line.
shquote() {
  local arg out=""
  for arg in "$@"; do
    case "$arg" in
      *[!A-Za-z0-9_@%+=:,./-]*|'')
        out="$out '$(printf '%s' "$arg" | sed "s/'/'\\\\''/g")'" ;;
      *)
        out="$out $arg" ;;
    esac
  done
  printf '%s' "${out# }"
}

# build_rank_cmd RANK [--check-kernels-mode]
# Result lands in the global RANK_CMD array (bash 3.2 has no nameref).
RANK_CMD=()
build_rank_cmd() {
  local rank="$1" mode="${2:-serve}" kv port world ep tp
  port=$((PORT_BASE + rank))
  if [ "$mode" = "check" ]; then
    # --check-kernels must run SINGLE-RANK. `init_nccl_comm` runs at
    # serve_load.rs:557, well before the kernel audit at :745 — a rank-0-only
    # process started with --world-size N would block in the NCCL bootstrap
    # waiting for peers that are never coming, and never reach the audit.
    world=1; ep=1; tp=1
  else
    world="$NGPUS"; ep="$EP_SIZE"; tp="$TP_SIZE"
  fi

  RANK_CMD=()
  if [ -n "$IMAGE" ]; then
    RANK_CMD=( "$DOCKER" run )
    if [ "$mode" = "check" ]; then
      RANK_CMD+=( --rm )
    else
      # The label is the only thing that identifies this container while its
      # ID is still inside `docker run -d`; see the OWNERSHIP note above.
      RANK_CMD+=( -d --name "$(container_name "$rank")" --label "$RUN_LABEL" )
    fi
    # No --device=/dev/infiniband, no --cap-add=IPC_LOCK, no memlock ulimit:
    # those exist in start-ep2.sh for RDMA between two chassis. Intra-node
    # NCCL uses NVLink/P2P/shared memory, which --ipc=host already covers.
    RANK_CMD+=( --gpus all --ipc=host --network host )
    for kv in ${NCCL_ENV[@]+"${NCCL_ENV[@]}"}; do
      RANK_CMD+=( -e "$kv" )
    done
    RANK_CMD+=( -e "RUST_LOG=$RUST_LOG" -v "$HF_CACHE:/root/.cache/huggingface" )
    if [ -n "$WARMUP_HOST" ]; then
      RANK_CMD+=( -v "$WARMUP_HOST:$WARMUP_IN_CONTAINER:ro" )
    fi
    RANK_CMD+=( "$IMAGE" serve "$MODEL" )
  else
    RANK_CMD=( env "RUST_LOG=$RUST_LOG" )
    for kv in ${NCCL_ENV[@]+"${NCCL_ENV[@]}"}; do
      RANK_CMD+=( "$kv" )
    done
    RANK_CMD+=( "$SPARK_BIN" serve "$MODEL" )
  fi

  RANK_CMD+=(
    --rank "$rank"
    --world-size "$world"
    --ep-size "$ep"
    --tp-size "$tp"
    --gpu-ordinal "$rank"
    --port "$port"
    --master-addr "$MASTER_ADDR"
    --master-port "$MASTER_PORT"
    --no-tui
  )
  if [ "$mode" = "check" ]; then
    RANK_CMD+=( --check-kernels )
  elif [ "$rank" -eq 0 ]; then
    RANK_CMD+=( --bind "$BIND" )
  fi
  if [ -n "$WARMUP_HOST" ]; then
    if [ -n "$IMAGE" ]; then
      RANK_CMD+=( --warmup-prompt "$WARMUP_IN_CONTAINER" )
    else
      RANK_CMD+=( --warmup-prompt "$WARMUP_HOST" )
    fi
  fi
  RANK_CMD+=( ${EXTRA_ARR[@]+"${EXTRA_ARR[@]}"} )
}

PORT_LAST=$((PORT_BASE + NGPUS - 1))

echo "=== Atlas single-node launch ==="
echo "Model:         $MODEL"
echo "GPUs (ranks):  $NGPUS   (rank i -> --gpu-ordinal i)"
echo "Topology:      TP=$TP_SIZE EP=$EP_SIZE world=$NGPUS"
echo "Ports:         $PORT_BASE..$PORT_LAST  (only rank 0 on $PORT_BASE serves clients)"
echo "NCCL bootstrap: $MASTER_ADDR:$MASTER_PORT"
echo "NCCL profile:  $NCCL_PROFILE"
if [ -n "$IMAGE" ]; then
  echo "Mode:          container ($IMAGE)"
else
  echo "Mode:          local binary ($SPARK_BIN)"
fi
echo "Run dir:       $RUN_DIR"
echo ""

# ── --check-kernels: rank 0 alone, exit with its status ──────────────────────
if [ "$CHECK_KERNELS" = "1" ]; then
  build_rank_cmd 0 check
  echo "# kernel check (single rank — the audit runs after NCCL init, so a"
  echo "#               multi-rank check would hang in the bootstrap)"
  shquote "${RANK_CMD[@]}"; echo ""
  if [ "$DRY_RUN" = "1" ]; then
    echo ""
    echo "dry-run: nothing launched."
    exit 0
  fi
  set +e
  "${RANK_CMD[@]}"
  rc=$?
  set -e
  echo ""
  echo "--check-kernels exited $rc (0 = every kernel lookup resolved)"
  exit "$rc"
fi

# ── Print / launch every rank ────────────────────────────────────────────────
# Workers first, head last: rank 0 is the one whose /health we poll, and the
# NCCL bootstrap is far less confusing when the listeners are already up.
LAUNCH_ORDER=()
i="$NGPUS"
while [ "$i" -gt 1 ]; do
  i=$((i - 1))
  LAUNCH_ORDER+=( "$i" )
done
LAUNCH_ORDER+=( 0 )

if [ "$DRY_RUN" = "1" ]; then
  for rank in "${LAUNCH_ORDER[@]}"; do
    if [ "$rank" -eq 0 ]; then role="head, serves HTTP"; else role="worker"; fi
    echo "# rank $rank ($role)"
    build_rank_cmd "$rank"
    shquote "${RANK_CMD[@]}"; echo ""
    echo ""
  done
  build_rank_cmd 0
  echo "health poll:   $HEALTH_URL (1 s interval, ${BOOT_TIMEOUT_S}s cap)"
  echo "summary: model=$MODEL ngpus=$NGPUS tp=$TP_SIZE ep=$EP_SIZE ports=$PORT_BASE-$PORT_LAST nccl_profile=$NCCL_PROFILE time_to_ready_s=dry-run"
  echo "rank0_command: $(shquote "${RANK_CMD[@]}")"
  echo ""
  echo "dry-run: nothing launched."
  exit 0
fi

# ── Refuse before creating anything ──────────────────────────────────────────
# An endpoint that already answers is somebody else's server. Launching into it
# means the new rank cannot bind, the old one keeps answering 200, and the poll
# below reports a boot time for a process that never served a request. Refusing
# is the only honest answer: this script cannot tell whose server that is.
existing_code="$(curl --silent --output /dev/null --max-time 5 --write-out '%{http_code}' "$HEALTH_URL" || true)"
if [ "$existing_code" = "200" ]; then
  echo "REFUSED: $HEALTH_URL is already answering 200 before anything was started." >&2
  echo "  Some other server owns this port. A rank launched now could not bind, and" >&2
  echo "  the readiness poll below would time that server rather than this one." >&2
  echo "  Stop it first, or pick another PORT_BASE." >&2
  exit 2
fi

# A run directory that still holds rank records belongs to a launch that has
# not been stopped. Its pid/container files are what --stop acts on, so
# overwriting them orphans those ranks. An intent counts: it is the record of a
# create that was never reconciled, and it may name a container that is up.
for f in "$RUN_DIR"/rank*.pid "$RUN_DIR"/rank*.container "$RUN_DIR"/rank*.intent; do
  [ -e "$f" ] || continue
  echo "REFUSED: $RUN_DIR still records a running launch ($(basename "$f"))." >&2
  echo "  Stop it with: ATLAS_NODE_RUN_DIR=$RUN_DIR $0 --stop" >&2
  echo "  or point ATLAS_NODE_RUN_DIR at a fresh directory." >&2
  exit 2
done

# And a container name that already exists is a container this invocation did
# not create. `docker rm -f` used to run here; that is how run B deleted run
# A's live rank.
if [ -n "$IMAGE" ]; then
  for rank in $(seq 0 $((NGPUS - 1))); do
    if container_exists "$(container_name "$rank")"; then
      echo "REFUSED: a container named $(container_name "$rank") already exists." >&2
      echo "  It was not created by this invocation, so it is not this script's to" >&2
      echo "  remove. Stop the launch that owns it, or use a different run directory" >&2
      echo "  or PORT_BASE -- the container name is derived from both." >&2
      exit 2
    fi
  done
fi

mkdir -p "$RUN_DIR"

# `setsid` detaches the rank from this shell's session so a closed terminal
# does not take the whole world down mid-benchmark. It is not present
# everywhere (notably macOS); nohup alone is the fallback.
SETSID=""
if command -v setsid >/dev/null 2>&1; then SETSID="setsid"; fi

START_EPOCH="$(date +%s)"
for rank in "${LAUNCH_ORDER[@]}"; do
  build_rank_cmd "$rank"
  log="$RUN_DIR/rank$rank.log"
  if [ "$rank" -eq 0 ]; then role="head"; else role="worker"; fi
  echo "starting rank $rank ($role) -> $log"
  shquote "${RANK_CMD[@]}"; echo ""
  if [ -n "$IMAGE" ]; then
    # No `docker rm -f` first: the name is refused above if it is taken, and
    # a name this invocation did not create is not this invocation's to delete.
    #
    # The intent goes down BEFORE the create: from the next line on a container
    # of this launch's may exist whether or not `docker run -d` ever returns,
    # and --stop has to be able to find it either way.
    cname="$(container_name "$rank")"
    printf 'name=%s\nlabel=%s\n' "$cname" "$RUN_LABEL" > "$RUN_DIR/rank$rank.intent"
    "${RANK_CMD[@]}" >"$log" 2>&1
    # Written whole or not at all: a reader between the two lines below sees
    # the intent, never half a container name.
    printf '%s\n' "$cname" > "$RUN_DIR/rank$rank.container.tmp"
    mv "$RUN_DIR/rank$rank.container.tmp" "$RUN_DIR/rank$rank.container"
  else
    if [ -n "$SETSID" ]; then
      $SETSID nohup "${RANK_CMD[@]}" >"$log" 2>&1 </dev/null &
    else
      nohup "${RANK_CMD[@]}" >"$log" 2>&1 </dev/null &
    fi
    echo $! > "$RUN_DIR/rank$rank.pid"
  fi
done
echo ""

# ── Readiness poll ───────────────────────────────────────────────────────────
# 503 and connection-refused are both LOADING states, exactly as
# bench/hopper_ab/time_to_ready.sh documents. Only a 200 ends the wait.
echo "polling $HEALTH_URL every 1 s (cap ${BOOT_TIMEOUT_S}s)..."
deadline=$((START_EPOCH + BOOT_TIMEOUT_S))
ready=0
codes_seen=""
died=0
READY_EPOCH=""
while [ "$(date +%s)" -lt "$deadline" ]; do
  if ! rank0_alive; then died=1; break; fi
  code="$(curl --silent --output /dev/null --max-time 5 --write-out '%{http_code}' "$HEALTH_URL" || true)"
  [ -n "$code" ] || code="000"
  case " $codes_seen " in *" $code "*) ;; *) codes_seen="$codes_seen $code" ;; esac
  if [ "$code" = "200" ]; then
    # A 200 is only OUR 200 while our rank 0 is still up, and the check has to
    # survive the race it exists for: a rank that cannot bind exits within
    # milliseconds of launch, while the server that already owns the port
    # answers instantly, so the FIRST poll can land inside that window and see
    # both a 200 and a rank that has not finished dying. One second of settle
    # closes it. The clock is read before the settle so the reported
    # time-to-ready is not inflated by it.
    READY_EPOCH="$(date +%s)"
    sleep 1
    if rank0_alive; then ready=1; break; fi
    died=1; break
  fi
  sleep 1
done

if [ "$died" = "1" ]; then
  echo "" >&2
  echo "FAILED: rank 0 exited before $HEALTH_URL was served by it (codes seen:${codes_seen:- none})." >&2
  echo "  Any 200 on that port is another process; this launch has nothing running." >&2
  for rank in $(seq 0 $((NGPUS - 1))); do
    echo "" >&2
    echo "--- rank $rank (last 40 lines) ---" >&2
    rank_log_tail "$rank" >&2
  done
  echo "" >&2
  stop_ranks >&2
  exit 1
fi
ELAPSED="$(( ${READY_EPOCH:-$(date +%s)} - START_EPOCH ))"

if [ "$ready" != "1" ]; then
  echo ""
  echo "TIMEOUT: $HEALTH_URL never answered 200 within ${BOOT_TIMEOUT_S}s (codes seen:${codes_seen:- none})." >&2
  for rank in $(seq 0 $((NGPUS - 1))); do
    echo "" >&2
    echo "--- rank $rank (last 40 lines) ---" >&2
    rank_log_tail "$rank" >&2
  done
  if [ "$STOP_ON_TIMEOUT" = "1" ]; then
    echo "" >&2
    stop_ranks >&2
  else
    echo "" >&2
    echo "Ranks left running for inspection. Stop them with:" >&2
    echo "  ATLAS_NODE_RUN_DIR=$RUN_DIR $0 --stop" >&2
  fi
  exit 1
fi

build_rank_cmd 0
echo ""
echo "=== ready in ${ELAPSED}s ==="
echo "API:           http://$BIND:$PORT_BASE/v1/chat/completions"
echo "Logs:          $RUN_DIR/rank*.log"
echo "Stop:          ATLAS_NODE_RUN_DIR=$RUN_DIR $0 --stop"
echo "summary: model=$MODEL ngpus=$NGPUS tp=$TP_SIZE ep=$EP_SIZE ports=$PORT_BASE-$PORT_LAST nccl_profile=$NCCL_PROFILE time_to_ready_s=$ELAPSED"
echo "rank0_command: $(shquote "${RANK_CMD[@]}")"
