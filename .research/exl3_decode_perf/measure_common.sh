#!/bin/bash
# Shared plumbing for the EXL3 GPU measurement scripts (sourced, not executed).
#
#   source "$(dirname "$0")/measure_common.sh"
#
# Provides: refuse_if_busy, make_run_dir, write_fingerprint, boot_server,
# wait_ready, start_mem_watchdog, stop_server, cleanup (installed on EXIT).
# Every function writes into $RUN_DIR. The serve preset is the ONLY model
# configuration used here; any ATLAS_* variable already in the caller's
# environment reaches the server unchanged (the preset sets its env defaults
# only when unset and logs the deviation), so an A/B arm is "export the kill
# switch, set ARM=<name>, rerun the script" — the switch is the only variable.
#
# House rules baked in: one `spark serve` on the box at a time; check
# `free -g` BEFORE and DURING (the box has hard-rebooted twice under
# unmonitored profiling runs); RUST_LOG=info; full output to files, never
# through head/tail; never `pkill -f spark` (kill our own process group only).

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$SCRIPT_DIR/../.." && pwd)

PORT=${PORT:-8899}
PRESET=${PRESET:-qwen3.8-flash-next-exl3}
CKPT=${CKPT:-/tank/exl3-ckpt/qwen38-flash-next-4.05bpw}
SPARK_BIN=${SPARK_BIN:-$REPO/target/release/spark}
ARM=${ARM:-default}
READY_TIMEOUT_S=${READY_TIMEOUT_S:-1500}
# Pre-boot gate: the preset pledges util 0.72 of ~121 GB (~87 GB) up front.
MIN_AVAIL_GB=${MIN_AVAIL_GB:-95}
# In-run watchdog: kill the server before the box swaps itself to death.
ABORT_AVAIL_GB=${ABORT_AVAIL_GB:-8}
ABORT_SWAP_DELTA_GB=${ABORT_SWAP_DELTA_GB:-4}
MEM_SAMPLE_S=${MEM_SAMPLE_S:-5}
NSYS=${NSYS:-/usr/local/cuda/bin/nsys}

export LD_LIBRARY_PATH=/home/ms/nccl/build/lib:/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export RUST_LOG=${RUST_LOG:-info}
export ATLAS_NO_TUI=1

SERVER_PID=""
WATCHDOG_PID=""
NSYS_SESSION=""
SWAP_USED_AT_BOOT=0
RUN_DIR=""

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { log "FATAL: $*"; exit 1; }

mem_avail_gb() { awk '/MemAvailable/ {printf "%d", $2/1048576}' /proc/meminfo; }
swap_used_gb() { awk '/SwapTotal/ {t=$2} /SwapFree/ {f=$2} END {printf "%d", (t-f)/1048576}' /proc/meminfo; }
gpu_sample() {
    # GB10 is unified memory: nvidia-smi reports memory as [N/A]; util/power/clock still work.
    local s
    s=$(nvidia-smi --query-gpu=utilization.gpu,power.draw,clocks.sm --format=csv,noheader,nounits 2>/dev/null | tr -d ' ')
    echo "${s:-na,na,na}"
}

# Refuse to start on a busy box. Matches any `spark* serve` (including a
# previous nsys-launched one) and any lingering nsys session.
refuse_if_busy() {
    local running avail
    running=$(pgrep -af '[s]park[^ ]* serve' || true)
    if [ -n "$running" ]; then
        printf '%s\n' "$running"
        die "another spark serve is running — one Atlas instance at a time. Kill it BY PID (never pkill -f spark)."
    fi
    if pgrep -x nsys >/dev/null 2>&1; then
        pgrep -ax nsys
        die "an nsys process is alive (orphaned session?). Kill it by PID first."
    fi
    if ss -ltn 2>/dev/null | grep -q ":${PORT} "; then
        die "port ${PORT} is already bound"
    fi
    avail=$(mem_avail_gb)
    if [ "$avail" -lt "$MIN_AVAIL_GB" ]; then
        free -g
        die "MemAvailable ${avail} GB < MIN_AVAIL_GB ${MIN_AVAIL_GB} — the util-0.72 pledge would not fit. Not booting."
    fi
    [ -x "$SPARK_BIN" ] || die "binary not found: $SPARK_BIN (cargo build --release -p spark-server)"
    [ -d "$CKPT" ] || die "checkpoint dir not found: $CKPT"
}

make_run_dir() {
    local name=$1
    RUN_DIR="${EXL3_RUN_ROOT:-$SCRIPT_DIR/runs}/$(date +%Y%m%d_%H%M%S)_${name}_${ARM}"
    mkdir -p "$RUN_DIR"
    export RUN_DIR
}

# Rule 1: fingerprint or it didn't happen. Everything a later reader needs to
# requote the number, in one file next to it.
write_fingerprint() {
    local f=$1
    {
        echo "date=$(date -Is) host=$(hostname) script=$(basename "$0") arm=$ARM port=$PORT"
        echo "repo=$REPO"
        echo "git_head=$(git -C "$REPO" rev-parse HEAD 2>/dev/null) branch=$(git -C "$REPO" rev-parse --abbrev-ref HEAD 2>/dev/null)"
        echo "git_dirty_files=$(git -C "$REPO" status --short 2>/dev/null | wc -l)"
        echo "binary=$SPARK_BIN sha256=$(sha256sum "$SPARK_BIN" | cut -c1-16) mtime=$(date -r "$SPARK_BIN" -Is)"
        echo "checkpoint=$CKPT"
        echo "preset=$PRESET (flags/env from kernels/gb10/qwen3.8-flash-next/MODEL.toml; operator overrides below)"
        echo "extra_serve_args=${EXTRA_SERVE_ARGS:-}"
        echo "caller_env:"; env | grep -E '^ATLAS_' | sort | sed 's/^/  /' || true
        echo "free_g:"; free -g | sed 's/^/  /'
        echo "gpu(util%,W,MHz)=$(gpu_sample)"
        echo "nsys=$("$NSYS" --version 2>/dev/null | head -1)"
        echo "python=$(python3 --version 2>&1)"
    } > "$f"
}

# Boot the preset. $1 = server log path; $2 = "nsys" to wrap in `nsys launch`
# (capture stays off until `nsys start`). Sets SERVER_PID = the process group
# we own (setsid), so cleanup can kill nsys AND the server together.
boot_server() {
    local logf=$1 mode=${2:-plain}
    local -a cmd=("$SPARK_BIN" serve "$PRESET" --model-from-path "$CKPT" --bind 127.0.0.1 --port "$PORT" --no-tui)
    if [ -n "${EXTRA_SERVE_ARGS:-}" ]; then
        # shellcheck disable=SC2206  # operator-supplied extra args are intentionally word-split
        cmd+=(${EXTRA_SERVE_ARGS})
    fi
    if [ "$mode" = nsys ]; then
        NSYS_SESSION="exl3-${ARM}-$$"
        export TMPDIR="$RUN_DIR/nsystmp"
        mkdir -p "$TMPDIR"
        cmd=("$NSYS" launch --trace=cuda,nvtx --session-new="$NSYS_SESSION" "${cmd[@]}")
    fi
    { printf 'serve_cmd='; printf '%q ' "${cmd[@]}"; echo; } >> "$RUN_DIR/fingerprint.txt"
    log "booting: ${cmd[*]}"
    SWAP_USED_AT_BOOT=$(swap_used_gb)
    setsid "${cmd[@]}" > "$logf" 2>&1 < /dev/null &
    SERVER_PID=$!
    log "server pgid=$SERVER_PID log=$logf"
}

wait_ready() {
    local logf=$1 i
    for ((i = 0; i < READY_TIMEOUT_S; i += 2)); do
        if curl -s -m 2 "http://127.0.0.1:${PORT}/v1/models" >/dev/null 2>&1; then
            log "READY after ~${i}s"
            { grep -a -E "Preset (flag|env)|Server live and ready" "$logf" || true; } \
                | sed 's/\x1b\[[0-9;]*m//g' | cut -c1-400 >> "$RUN_DIR/fingerprint.txt"
            return 0
        fi
        if ! kill -0 "$SERVER_PID" 2>/dev/null; then
            log "SERVER EXITED during boot; log tail:"
            tail -30 "$logf" | cut -c1-240
            return 1
        fi
        sleep 2
    done
    log "server not ready after ${READY_TIMEOUT_S}s"
    tail -20 "$logf" | cut -c1-240
    return 1
}

# Background sampler for the server's whole lifetime (start it right after
# boot_server, BEFORE wait_ready: the load phase is where host memory spikes,
# and the subshell needs SERVER_PID to be set). Appends to
# $RUN_DIR/mem_samples.csv and kills the server if the box approaches swap death.
start_mem_watchdog() {
    local csv="$RUN_DIR/mem_samples.csv"
    echo "epoch,mem_avail_gb,swap_used_gb,gpu_util_pct,gpu_power_w,gpu_sm_mhz" > "$csv"
    (
        while true; do
            a=$(mem_avail_gb); s=$(swap_used_gb)
            echo "$(date +%s),$a,$s,$(gpu_sample)" >> "$csv"
            if [ "$a" -lt "$ABORT_AVAIL_GB" ] || [ $((s - SWAP_USED_AT_BOOT)) -gt "$ABORT_SWAP_DELTA_GB" ]; then
                echo "$(date -Is) ABORT: MemAvailable=${a}GB swap_used=${s}GB (at boot ${SWAP_USED_AT_BOOT}GB) — killing server pgid $SERVER_PID" \
                    | tee -a "$RUN_DIR/ABORTED.txt" >&2
                if [ -n "$SERVER_PID" ]; then kill -TERM -- "-$SERVER_PID" 2>/dev/null || true; fi
            fi
            sleep "$MEM_SAMPLE_S"
        done
    ) &
    WATCHDOG_PID=$!
}

stop_server() {
    [ -n "$SERVER_PID" ] || return 0
    if kill -0 "$SERVER_PID" 2>/dev/null; then
        log "stopping server pgid $SERVER_PID (TERM)"
        kill -TERM -- "-$SERVER_PID" 2>/dev/null || true
        local i
        for ((i = 0; i < 45; i++)); do
            kill -0 "$SERVER_PID" 2>/dev/null || break
            sleep 1
        done
        if kill -0 "$SERVER_PID" 2>/dev/null; then
            log "still alive; KILL"
            kill -KILL -- "-$SERVER_PID" 2>/dev/null || true
        fi
    fi
    SERVER_PID=""
}

cleanup() {
    local rc=$? left
    trap - EXIT INT TERM
    if [ -n "$NSYS_SESSION" ]; then
        "$NSYS" stop --session="$NSYS_SESSION" >/dev/null 2>&1 || true
    fi
    stop_server
    [ -n "$WATCHDOG_PID" ] && kill "$WATCHDOG_PID" 2>/dev/null
    sleep 2
    left=$(pgrep -af '[s]park[^ ]* serve|^nsys ' || true)
    if [ -n "$left" ]; then
        log "WARNING: processes still alive after cleanup:"
        printf '%s\n' "$left"
    fi
    if [ -n "$RUN_DIR" ] && [ -f "$RUN_DIR/ABORTED.txt" ]; then
        log "RUN ABORTED by the memory watchdog — see $RUN_DIR/ABORTED.txt"
        rc=3
    fi
    log "done rc=$rc run_dir=$RUN_DIR"
    exit "$rc"
}
trap cleanup EXIT INT TERM
