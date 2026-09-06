#!/bin/bash
# EXL3 concurrency sweep under the named preset (4 sequences x 128K, 2 MTP drafts,
# prefix caching on — the operator's envelope, "not separately validated" by the
# recipe commit). Two arms per boot:
#   short:  ~2K-token prompt per stream, C = 1, 2, 4, REPEATS_SHORT (default 3)
#   long:   ~30K-token prompt per stream, C = 1, 2, 4, REPEATS_LONG (default 1)
#           => C=4 holds ~120K of the 128K x 4 envelope in KV + GDN/PLE state
# and measure_concurrency.py records per-stream decode tok/s (server-attested
# completion_tokens / decode wall), aggregate tok/s, TTFT, prompt tokens, and
# samples MemAvailable + swap + nvidia-smi util/power every 5 s. Two watchdogs
# (this shell's, covering boot; the driver's, covering each cell) TERM the
# server if MemAvailable < ABORT_AVAIL_GB or swap grows > ABORT_SWAP_DELTA_GB —
# the earlier C=4 attempt at util 0.85 swapped the box. The preset's util 0.72
# is NOT raised here; do not raise it to make the long arm fit.
#
# Usage:   ./concurrency_sweep.sh
#          ARM=<name> ATLAS_SOME_SWITCH=1 ./concurrency_sweep.sh   (A/B: the exported
#          switch is the only variable; ARMS="short" to skip the long arm)
# Knobs:   PORT (8899) ARMS ("short long") CONC ("1 2 4") SHORT_TOKENS (2000)
#          LONG_TOKENS (30000) MAX_TOKENS (300) REPEATS_SHORT (3) REPEATS_LONG (1)
#          EXTRA_SERVE_ARGS (e.g. "--max-num-seqs 2") SPARK_BIN CKPT EXL3_RUN_ROOT
#
# Output in $RUN_DIR: sweep_<arm>.txt + .json per arm, results.md (combined
# table), mem_samples.csv (whole run), serve.log, fingerprint.txt. Refuses to
# start beside another `spark serve`; cleans up on any exit. No number printed
# here is a claim until it is read against MEASUREMENT_PLAN.md with n>=3.
set -euo pipefail
# shellcheck source=measure_common.sh
source "$(dirname "$0")/measure_common.sh"

ARMS=${ARMS:-"short long"}
CONC=${CONC:-"1 2 4"}
SHORT_TOKENS=${SHORT_TOKENS:-2000}
LONG_TOKENS=${LONG_TOKENS:-30000}
MAX_TOKENS=${MAX_TOKENS:-300}
REPEATS_SHORT=${REPEATS_SHORT:-3}
REPEATS_LONG=${REPEATS_LONG:-1}
# MTP acceptance is part of the answer (tok/s WITH acceptance, per the decode
# write-up): the accept lines land in serve.log for the summary below.
export ATLAS_MTP_ACCEPT_DEBUG=${ATLAS_MTP_ACCEPT_DEBUG:-1}

refuse_if_busy
make_run_dir concurrency_sweep
write_fingerprint "$RUN_DIR/fingerprint.txt"
echo "harness=measure_concurrency.py arms=$ARMS conc=$CONC short_tokens=$SHORT_TOKENS long_tokens=$LONG_TOKENS max_tokens=$MAX_TOKENS repeats_short=$REPEATS_SHORT repeats_long=$REPEATS_LONG" >> "$RUN_DIR/fingerprint.txt"
log "run dir: $RUN_DIR"

boot_server "$RUN_DIR/serve.log" plain
start_mem_watchdog
wait_ready "$RUN_DIR/serve.log" || die "boot failed"
MODEL=$(curl -s -m 5 "http://127.0.0.1:${PORT}/v1/models" | python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0]["id"])')
log "model id: $MODEL"

# Warm-up: one short request so first-request one-time costs stay out of C=1.
log "warm-up"
python3 -u "$SCRIPT_DIR/measure_decode.py" --port "$PORT" --model "$MODEL" --repeats 1 --max-tokens 64 \
    > "$RUN_DIR/warmup.txt" 2>&1 || { cat "$RUN_DIR/warmup.txt"; die "warm-up request failed"; }
tail -1 "$RUN_DIR/warmup.txt"

RC=0
for arm in $ARMS; do
    case "$arm" in
        short) toks=$SHORT_TOKENS; reps=$REPEATS_SHORT ;;
        long)  toks=$LONG_TOKENS;  reps=$REPEATS_LONG ;;
        *) die "unknown arm '$arm' (short|long)" ;;
    esac
    log "arm=$arm prompt_tokens~$toks C={$CONC} repeats=$reps max_tokens=$MAX_TOKENS"
    # shellcheck disable=SC2086  # CONC is a deliberate word list
    python3 -u "$SCRIPT_DIR/measure_concurrency.py" --port "$PORT" --model "$MODEL" --label "$arm" \
        --concurrency $CONC --prompt-tokens "$toks" --max-tokens "$MAX_TOKENS" --repeats "$reps" \
        --abort-avail-gb "$ABORT_AVAIL_GB" --abort-swap-delta-gb "$ABORT_SWAP_DELTA_GB" \
        --server-pid "$SERVER_PID" --json-out "$RUN_DIR/sweep_${arm}.json" \
        > "$RUN_DIR/sweep_${arm}.txt" 2>&1 || RC=$?
    cat "$RUN_DIR/sweep_${arm}.txt"
    if [ "$RC" -eq 3 ]; then
        log "arm $arm ABORTED by the driver's memory watchdog; not running further arms"
        break
    fi
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        log "server died during arm $arm; log tail:"
        tail -30 "$RUN_DIR/serve.log" | cut -c1-240
        RC=1
        break
    fi
done

# MTP acceptance summary (server-side): mean accepted per step over the run.
{ grep -a "MTP accept" "$RUN_DIR/serve.log" || true; } | sed 's/\x1b\[[0-9;]*m//g' | tail -2000 > "$RUN_DIR/mtp_accept_lines.txt"
log "MTP accept lines: $(wc -l < "$RUN_DIR/mtp_accept_lines.txt") (tail):"
tail -3 "$RUN_DIR/mtp_accept_lines.txt" | cut -c1-200

stop_server

{
    echo "# Concurrency sweep — $(date -Is) arm=$ARM"
    echo
    echo "Fingerprint: \`$RUN_DIR/fingerprint.txt\`. Per-stream decode tok/s = (completion_tokens-1)/decode wall;"
    echo "aggregate decode tok/s = sum tokens / (max last chunk - min first chunk); the incl.-TTFT column divides by the cell wall."
    echo "Host memory from /proc/meminfo; nvidia-smi memory is [N/A] on GB10 (unified memory) — see mem_samples.csv."
    echo
    for arm in $ARMS; do
        [ -f "$RUN_DIR/sweep_${arm}.txt" ] || continue
        echo "## arm: $arm"
        echo
        grep -a "^FINGERPRINT" "$RUN_DIR/sweep_${arm}.txt" || true
        echo
        awk '/^\| arm \|/ {p=1} p' "$RUN_DIR/sweep_${arm}.txt"
        echo
    done
    echo "## Host memory over the whole run (boot included)"
    echo
    awk -F, 'NR>1 {if (min=="" || $2<min) min=$2; if (max=="" || $3>max) max=$3; n++} END {printf "samples=%d min_MemAvailable_GB=%s max_swap_used_GB=%s\n", n, min, max}' "$RUN_DIR/mem_samples.csv"
    [ -f "$RUN_DIR/ABORTED.txt" ] && { echo; echo "**ABORTED by the shell watchdog:**"; cat "$RUN_DIR/ABORTED.txt"; }
} > "$RUN_DIR/results.md"
cat "$RUN_DIR/results.md"
log "artefacts: $RUN_DIR (results.md, sweep_*.json, mem_samples.csv, serve.log, fingerprint.txt)"
exit "$RC"
