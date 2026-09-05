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

usage() { sed -n '2,89p' "$0"; }

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
CONTAINER_PREFIX="atlas-node-ep"

# ── --stop: kill exactly what this script recorded ────────────────────────────
# By pid file and container name, never `pkill -f`. A `pkill -f spark` here
# would match this script's own command line (and any editor with the word in
# an argument), which is how a stop turns into a self-kill.
stop_ranks() {
  local f pid name stopped=0
  if [ ! -d "$RUN_DIR" ]; then
    echo "no run directory at $RUN_DIR — nothing to stop"
    return 0
  fi
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
container_name() { printf '%s-rank%s' "$CONTAINER_PREFIX" "$1"; }

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
      RANK_CMD+=( -d --name "$(container_name "$rank")" )
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
    "$DOCKER" rm -f "$(container_name "$rank")" >/dev/null 2>&1 || true
    "${RANK_CMD[@]}" >"$log" 2>&1
    container_name "$rank" > "$RUN_DIR/rank$rank.container"
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
while [ "$(date +%s)" -lt "$deadline" ]; do
  code="$(curl --silent --output /dev/null --max-time 5 --write-out '%{http_code}' "$HEALTH_URL" || true)"
  [ -n "$code" ] || code="000"
  case " $codes_seen " in *" $code "*) ;; *) codes_seen="$codes_seen $code" ;; esac
  if [ "$code" = "200" ]; then ready=1; break; fi
  sleep 1
done
ELAPSED="$(( $(date +%s) - START_EPOCH ))"

if [ "$ready" != "1" ]; then
  echo ""
  echo "TIMEOUT: $HEALTH_URL never answered 200 within ${BOOT_TIMEOUT_S}s (codes seen:${codes_seen:- none})." >&2
  for rank in $(seq 0 $((NGPUS - 1))); do
    echo "" >&2
    echo "--- rank $rank (last 40 lines) ---" >&2
    if [ -n "$IMAGE" ]; then
      "$DOCKER" logs --tail 40 "$(container_name "$rank")" >&2 2>&1 || \
        echo "(no container logs for $(container_name "$rank"))" >&2
    elif [ -f "$RUN_DIR/rank$rank.log" ]; then
      tail -n 40 "$RUN_DIR/rank$rank.log" >&2
    else
      echo "(no log at $RUN_DIR/rank$rank.log)" >&2
    fi
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
