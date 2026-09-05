#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
#
# Tests for scripts/start-node-ep.sh. No GPU, no NCCL, no model: every case is
# either a --dry-run (the script prints commands and launches nothing) or runs
# against a local HTTP stub.
#
# What each case is actually defending:
#
#   (a) 4-GPU pure EP, NCCL_PROFILE=default -> exactly four rank commands, each
#       carrying --rank i --world-size 4 --ep-size 4 --tp-size 1
#       --gpu-ordinal i --port 8888+i, and NOT ONE NCCL_* variable. The whole
#       point of this launcher on an NVLink box is that it ships no NCCL
#       config; a stray variable creeping back in is the regression.
#   (b) NCCL_PROFILE=gb10-roce -> the pessimized GB10 block from
#       scripts/start-ep2.sh reappears verbatim, so an A/B against the
#       two-Spark deployment stays possible.
#   (c) NGPUS=4 EP_SIZE=2 TP_SIZE=1 -> REFUSED (exit 2) with a message naming
#       the world-size rule. `resolve_topology` would reject this after every
#       rank had loaded weights; catching it in the shell is free.
#   (d) IMAGE=... -> docker run lines with --gpus all/--ipc=host/--network
#       host, and NO --device=/dev/infiniband (that flag is RDMA-between-
#       chassis, meaningless and privilege-widening on one node).
#   (e) --check-kernels -> single-rank (--world-size 1), because the kernel
#       audit runs AFTER the NCCL bootstrap (serve_load.rs:557 vs :745): a
#       rank-0-only process at --world-size 4 would hang, not report.
#   (f) source grep: the script must never contain `pkill -f`. A `pkill -f`
#       pattern matches the killing shell's own command line, which has cost
#       this project real hours; --stop uses pid files and `kill`.
#   (g) health poll against a stub that answers 503 until it is ready -> the
#       reported time-to-ready must be >= 2 s. A poll loop that treats a
#       loading 503 as ready would report ~0 and make every boot number in the
#       campaign a fiction.
#   (h) two invocations with different run directories and ports, against a
#       fake Docker: neither may stop or remove the other's rank, and --stop
#       from one run directory must leave the other's container alone. The
#       container name used to be the global `atlas-node-ep-rankN`, so run B
#       force-removed run A's live rank before starting its own.
#   (i) a rank that dies while a FOREIGN server answers 200 on the port must
#       fail the launch, not report a boot time. Otherwise the benchmark that
#       follows measures the wrong process.
#   (j) an endpoint that already answers before launch is refused up front.
#
# Usage: bash scripts/start_node_ep_test.sh
set -uo pipefail

SCRIPT="$(cd "$(dirname "$0")" && pwd)/start-node-ep.sh"
MODEL="nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-FP8"
tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

asserts=0
fail() { echo "ASSERT FAILED [$1]: $2" >&2; exit 1; }
ok() { asserts=$((asserts + 1)); echo "  ok [$1] $2"; }

have() { grep -Fq -- "$2" <<<"$1"; }

# ── (a) 4 GPUs, pure EP, NCCL defaults ───────────────────────────────────────
out="$(NGPUS=4 EP_SIZE=4 TP_SIZE=1 NCCL_PROFILE=default \
        bash "$SCRIPT" --dry-run "$MODEL" 2>&1)"; rc=$?
[ $rc -eq 0 ] || fail a "dry-run exited $rc: $out"

n="$(grep -c '^env RUST_LOG=info \./target/release/spark serve ' <<<"$out")"
[ "$n" -eq 4 ] || fail a "expected 4 rank commands, got $n:
$out"
ok a "prints exactly 4 rank commands"

for i in 0 1 2 3; do
  line="$(grep "^env RUST_LOG=info \./target/release/spark serve .* --rank $i " <<<"$out")"
  [ -n "$line" ] || fail a "no command line for rank $i:
$out"
  have "$line" "--rank $i --world-size 4 --ep-size 4 --tp-size 1" \
    || fail a "rank $i topology flags wrong: $line"
  have "$line" "--gpu-ordinal $i" || fail a "rank $i missing --gpu-ordinal $i: $line"
  have "$line" "--port $((8888 + i))" || fail a "rank $i missing --port $((8888 + i)): $line"
  have "$line" "--master-addr 127.0.0.1 --master-port 29500" \
    || fail a "rank $i missing master addr/port: $line"
done
ok a "each rank carries --rank/--world-size/--ep-size/--tp-size/--gpu-ordinal/--port/--master-*"

grep -q 'NCCL_' <<<"$out" && fail a "NCCL_PROFILE=default must emit NO NCCL variables:
$(grep 'NCCL_' <<<"$out")"
ok a "NCCL_PROFILE=default emits no NCCL_* variable"

have "$out" "only rank 0 on 8888 serves clients" \
  || fail a "summary must say only rank 0 serves: $out"
have "$out" "summary: model=$MODEL ngpus=4 tp=1 ep=4 ports=8888-8891 nccl_profile=default" \
  || fail a "missing one-line summary: $out"
grep -q '^rank0_command: env RUST_LOG=info \./target/release/spark serve .* --rank 0 ' <<<"$out" \
  || fail a "missing pasteable rank0_command: $out"
ok a "prints the port layout, the one-line summary and a pasteable rank0_command"

# Workers before the head: the head is the rank whose /health is polled.
order="$(grep -o '^# rank [0-9]' <<<"$out" | tr -d '\n')"
[ "$order" = "# rank 3# rank 2# rank 1# rank 0" ] \
  || fail a "launch order must be workers-then-head, got: $order"
ok a "launch order is ranks N-1..1 then rank 0"

# ── (b) gb10-roce profile reproduces start-ep2.sh's block ────────────────────
out_gb="$(NGPUS=4 EP_SIZE=4 TP_SIZE=1 NCCL_PROFILE=gb10-roce \
           bash "$SCRIPT" --dry-run "$MODEL" 2>&1)"; rc=$?
[ $rc -eq 0 ] || fail b "gb10-roce dry-run exited $rc: $out_gb"
for kv in NCCL_SOCKET_IFNAME=enp1s0f0np0 NCCL_IB_HCA=rocep1s0f0 \
          NCCL_IB_ROCE_VERSION_NUM=2 NCCL_IB_ADDR_FAMILY=AF_INET \
          NCCL_NET_GDR_LEVEL=0 NCCL_NET_GDR_C2C=0 NCCL_DMABUF_ENABLE=0 \
          NCCL_NVLS_ENABLE=0 NCCL_PROTO=Simple NCCL_ALGO=Ring; do
  have "$out_gb" "$kv" || fail b "gb10-roce profile missing $kv: $out_gb"
done
ok b "gb10-roce reproduces the start-ep2.sh NCCL block"

# ── (c) world-size rule is enforced ──────────────────────────────────────────
out_bad="$(NGPUS=4 EP_SIZE=2 TP_SIZE=1 bash "$SCRIPT" --dry-run "$MODEL" 2>&1)"; rc=$?
[ $rc -eq 2 ] || fail c "NGPUS=4 EP=2 TP=1 must exit 2, got $rc: $out_bad"
have "$out_bad" "invalid parallelism topology" || fail c "message must name the problem: $out_bad"
have "$out_bad" "world_size == tp_size * ep_size" || fail c "message must state the rule: $out_bad"
have "$out_bad" "EP_SIZE=4 TP_SIZE=1" || fail c "message must offer the fix: $out_bad"
ok c "NGPUS=4 EP_SIZE=2 TP_SIZE=1 is refused with the rule and a fix"

out_ok="$(NGPUS=4 EP_SIZE=2 TP_SIZE=2 bash "$SCRIPT" --dry-run "$MODEL" 2>&1)"; rc=$?
[ $rc -eq 0 ] || fail c "orthogonal mesh 2x2=4 must be accepted, got $rc: $out_ok"
ok c "orthogonal mesh TP=2 EP=2 on 4 GPUs is accepted"

# ── (d) container mode carries no RDMA flags ─────────────────────────────────
out_img="$(NGPUS=4 EP_SIZE=4 TP_SIZE=1 IMAGE=avarok/atlas-gb10:latest \
            bash "$SCRIPT" --dry-run "$MODEL" 2>&1)"; rc=$?
[ $rc -eq 0 ] || fail d "container dry-run exited $rc: $out_img"
n="$(grep -c '^docker run -d --name atlas-node-ep-8888-rank[0-9] ' <<<"$out_img")"
[ "$n" -eq 4 ] || fail d "expected 4 docker run lines, got $n:
$out_img"
have "$out_img" "--gpus all --ipc=host --network host" \
  || fail d "docker run missing the required container flags: $out_img"
have "$out_img" ":/root/.cache/huggingface" || fail d "HF cache not mounted: $out_img"
for banned in "--device=/dev/infiniband" "--cap-add=IPC_LOCK" "memlock"; do
  grep -Fq -- "$banned" <<<"$out_img" && fail d "container mode must not carry $banned:
$out_img"
done
ok d "container mode prints 4 docker run lines with no RDMA/IB flags"

# ── (e) --check-kernels is single-rank ───────────────────────────────────────
out_ck="$(NGPUS=4 EP_SIZE=4 TP_SIZE=1 bash "$SCRIPT" --dry-run --check-kernels "$MODEL" 2>&1)"; rc=$?
[ $rc -eq 0 ] || fail e "--check-kernels dry-run exited $rc: $out_ck"
have "$out_ck" "--check-kernels" || fail e "the flag itself is missing: $out_ck"
have "$out_ck" "--no-tui" || fail e "--check-kernels run must pass --no-tui: $out_ck"
have "$out_ck" "--rank 0 --world-size 1 --ep-size 1 --tp-size 1" \
  || fail e "--check-kernels must run single-rank (NCCL init precedes the audit): $out_ck"
n="$(grep -c '^env RUST_LOG=info \./target/release/spark serve ' <<<"$out_ck")"
[ "$n" -eq 1 ] || fail e "--check-kernels must print exactly 1 command, got $n:
$out_ck"
ok e "--check-kernels runs rank 0 alone at --world-size 1"

# ── (f) the launcher must never grow a `pkill -f` ────────────────────────────
if grep -n 'pkill -f' "$SCRIPT" | grep -qv '^[0-9]*:#'; then
  fail f "scripts/start-node-ep.sh contains an executable 'pkill -f':
$(grep -n 'pkill -f' "$SCRIPT")"
fi
ok f "no executable 'pkill -f' in the launcher (--stop uses pid files)"

# ── (g) health poll: 503, 503, 200 must measure >= 2 s ───────────────────────
# Stub kept inside this script on purpose: bench/hopper_ab/time_to_ready.sh has
# an equivalent one for its own --selftest, and this test must not depend on
# (or modify) that file.
port="$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')"
cat > "$tmp/stub.py" <<'PY'
import json, sys
from http.server import BaseHTTPRequestHandler, HTTPServer

STATE = {"hits": 0}
LOADING = int(sys.argv[2])


class H(BaseHTTPRequestHandler):
    def _send(self, code, body):
        raw = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path != "/health":
            return self._send(404, {"error": "not found"})
        STATE["hits"] += 1
        # Atlas's shape: loading for a while, then ready. The count is an
        # argument because the launcher's pre-launch occupancy probe consumes
        # the FIRST answer -- a run that starts against a port already serving
        # 200 is refused, so a test that wants a successful launch has to hand
        # back at least one loading answer before the poll begins.
        if STATE["hits"] <= LOADING:
            return self._send(503, {"status": "loading"})
        self._send(200, {"status": "ready"})

    def log_message(self, *a):
        pass


HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
PY
python3 "$tmp/stub.py" "$port" 3 & stub_pid=$!
python3 - "$port" <<'PY' || fail g "health stub never bound"
import socket, sys, time
deadline = time.time() + 10
while time.time() < deadline:
    try:
        socket.create_connection(("127.0.0.1", int(sys.argv[1])), 0.2).close()
        sys.exit(0)
    except OSError:
        time.sleep(0.05)
sys.exit(1)
PY

# A stand-in for the spark binary: the launcher's job here is the poll loop and
# the pid bookkeeping, not the engine. It must stay alive so --stop has
# something to kill.
cat > "$tmp/stub-spark" <<'PY'
#!/usr/bin/env bash
echo "stub spark: $*"
sleep 120
PY
chmod +x "$tmp/stub-spark"

run_dir="$tmp/run"
out_poll="$(NGPUS=1 EP_SIZE=1 TP_SIZE=1 SPARK_BIN="$tmp/stub-spark" \
            ATLAS_NODE_RUN_DIR="$run_dir" \
            ATLAS_NODE_HEALTH_URL="http://127.0.0.1:$port/health" \
            BOOT_TIMEOUT_S=30 bash "$SCRIPT" "$MODEL" 2>&1)"; rc=$?
kill "$stub_pid" 2>/dev/null; wait "$stub_pid" 2>/dev/null

[ $rc -eq 0 ] || fail g "launcher exited $rc against the health stub:
$out_poll"
elapsed="$(sed -n 's/^=== ready in \([0-9][0-9]*\)s ===$/\1/p' <<<"$out_poll")"
[ -n "$elapsed" ] || fail g "no 'ready in Ns' line:
$out_poll"
[ "$elapsed" -ge 2 ] || fail g "two loading polls must cost >= 2 s, reported ${elapsed}s:
$out_poll"
have "$out_poll" "time_to_ready_s=$elapsed" || fail g "summary must carry the same number: $out_poll"
ok g "one occupancy probe plus 503,503,200 measures ${elapsed}s (>= 2) and reports it"

[ -f "$run_dir/rank0.pid" ] || fail g "no pid file written to $run_dir"
pid="$(cat "$run_dir/rank0.pid")"
kill -0 "$pid" 2>/dev/null || fail g "recorded pid $pid is not alive"
out_stop="$(ATLAS_NODE_RUN_DIR="$run_dir" bash "$SCRIPT" --stop 2>&1)"; rc=$?
[ $rc -eq 0 ] || fail g "--stop exited $rc: $out_stop"
have "$out_stop" "stopping pid $pid" || fail g "--stop must name the pid it kills: $out_stop"
[ -f "$run_dir/rank0.pid" ] && fail g "--stop must remove the pid file"
sleep 1
kill -0 "$pid" 2>/dev/null && { kill -9 "$pid" 2>/dev/null; fail g "pid $pid survived --stop"; }
ok g "--stop kills the recorded pid by pid file and clears it"

# ── shared helpers for (h)-(j) ──────────────────────────────────────────────
# One stub HTTP server per port, and a fake Docker that records every call and
# keeps a set of the containers it "created". Both are files so the launcher
# reaches them the way it reaches the real thing: SPARK_BIN / DOCKER.
start_health_stub() {  # start_health_stub PORT LOADING -> echoes the pid
  # stdout redirected, or the command substitution around this function would
  # hold the stub's inherited pipe open and never return.
  python3 "$tmp/stub.py" "$1" "$2" >/dev/null 2>&1 &
  local pid=$!
  python3 - "$1" <<'PY' || fail helper "health stub on port $1 never bound"
import socket, sys, time
deadline = time.time() + 10
while time.time() < deadline:
    try:
        socket.create_connection(("127.0.0.1", int(sys.argv[1])), 0.2).close()
        sys.exit(0)
    except OSError:
        time.sleep(0.05)
sys.exit(1)
PY
  echo "$pid"
}

cat > "$tmp/fake-docker" <<'SH'
#!/usr/bin/env bash
# Records every call, and keeps one line per live container name so `inspect`
# can answer whether a name exists. No Docker involved.
printf '%s\n' "$*" >> "$DOCKER_CALLS"
cmd="${1:-}"; shift || true
name=""
case "$cmd" in
  run)
    while [ $# -gt 0 ]; do
      case "$1" in --name) name="$2"; shift 2 ;; *) shift ;; esac
    done
    printf '%s\n' "$name" >> "$DOCKER_STATE"
    echo "cid-$name"
    ;;
  inspect)
    for a in "$@"; do name="$a"; done
    grep -Fxq -- "$name" "$DOCKER_STATE" 2>/dev/null || exit 1
    echo "true"
    ;;
  stop|rm)
    for a in "$@"; do name="$a"; done
    grep -Fxv -- "$name" "$DOCKER_STATE" > "$DOCKER_STATE.next" 2>/dev/null || true
    mv "$DOCKER_STATE.next" "$DOCKER_STATE"
    ;;
  logs) echo "(fake logs for ${*: -1})" ;;
esac
exit 0
SH
chmod +x "$tmp/fake-docker"

export DOCKER_CALLS="$tmp/docker.calls"
export DOCKER_STATE="$tmp/docker.state"
: > "$DOCKER_CALLS"
: > "$DOCKER_STATE"

# ── (h) two runs cannot touch each other's containers ───────────────────────
port_a="$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')"
port_b="$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')"
stub_a="$(start_health_stub "$port_a" 1)"
stub_b="$(start_health_stub "$port_b" 1)"

run_a="$tmp/run-a"
out_a="$(NGPUS=1 EP_SIZE=1 TP_SIZE=1 IMAGE=avarok/atlas-gb10:latest \
          DOCKER="$tmp/fake-docker" PORT_BASE="$port_a" \
          ATLAS_NODE_RUN_DIR="$run_a" \
          ATLAS_NODE_HEALTH_URL="http://127.0.0.1:$port_a/health" \
          BOOT_TIMEOUT_S=30 bash "$SCRIPT" "$MODEL" 2>&1)"; rc=$?
[ $rc -eq 0 ] || fail h "run A exited $rc:
$out_a"
name_a="$(cat "$run_a/rank0.container")"
grep -Fxq -- "$name_a" "$DOCKER_STATE" || fail h "run A's container is not live: $name_a"

: > "$DOCKER_CALLS"
run_b="$tmp/run-b"
out_b="$(NGPUS=1 EP_SIZE=1 TP_SIZE=1 IMAGE=avarok/atlas-gb10:latest \
          DOCKER="$tmp/fake-docker" PORT_BASE="$port_b" \
          ATLAS_NODE_RUN_DIR="$run_b" \
          ATLAS_NODE_HEALTH_URL="http://127.0.0.1:$port_b/health" \
          BOOT_TIMEOUT_S=30 bash "$SCRIPT" "$MODEL" 2>&1)"; rc=$?
[ $rc -eq 0 ] || fail h "run B exited $rc:
$out_b"
name_b="$(cat "$run_b/rank0.container")"
[ "$name_a" != "$name_b" ] || fail h "two runs must not share the container name $name_a"
ok h "two run directories on two ports get distinct container names ($name_a / $name_b)"

touched="$(grep -E "^(stop|rm) .*(^| )$name_a( |$)" "$DOCKER_CALLS" || true)"
[ -z "$touched" ] || fail h "run B stopped or removed run A's container:
$touched"
grep -Fxq -- "$name_a" "$DOCKER_STATE" || fail h "run A's container did not survive run B"
ok h "starting run B leaves run A's rank alone"

: > "$DOCKER_CALLS"
out_stop_a="$(DOCKER="$tmp/fake-docker" ATLAS_NODE_RUN_DIR="$run_a" \
               bash "$SCRIPT" --stop 2>&1)"; rc=$?
[ $rc -eq 0 ] || fail h "--stop for run A exited $rc: $out_stop_a"
grep -Fq -- "$name_b" "$DOCKER_CALLS" && fail h "--stop from run A touched run B:
$(cat "$DOCKER_CALLS")"
grep -Fxq -- "$name_b" "$DOCKER_STATE" || fail h "run B's container did not survive run A's --stop"
grep -Fxq -- "$name_a" "$DOCKER_STATE" && fail h "--stop must have removed run A's own container"
ok h "--stop from run A's directory removes only run A's container"

kill "$stub_a" "$stub_b" 2>/dev/null
wait "$stub_a" "$stub_b" 2>/dev/null

# ── (i) a dead rank is not readiness, even against a 200 ────────────────────
port_i="$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')"
stub_i="$(start_health_stub "$port_i" 1)"
cat > "$tmp/dying-spark" <<'SH'
#!/usr/bin/env bash
echo "stub spark: refusing to bind, exiting 42" >&2
exit 42
SH
chmod +x "$tmp/dying-spark"

run_i="$tmp/run-i"
out_i="$(NGPUS=1 EP_SIZE=1 TP_SIZE=1 SPARK_BIN="$tmp/dying-spark" \
          PORT_BASE="$port_i" ATLAS_NODE_RUN_DIR="$run_i" \
          ATLAS_NODE_HEALTH_URL="http://127.0.0.1:$port_i/health" \
          BOOT_TIMEOUT_S=15 bash "$SCRIPT" "$MODEL" 2>&1)"; rc=$?
kill "$stub_i" 2>/dev/null; wait "$stub_i" 2>/dev/null
[ $rc -ne 0 ] || fail i "a rank that exited 42 must not be reported ready:
$out_i"
have "$out_i" "rank 0" || fail i "the failure must name the rank that died: $out_i"
have "$out_i" "refusing to bind, exiting 42" \
  || fail i "the failure must carry the dead rank's log tail: $out_i"
grep -Fq -- "=== ready in" <<<"$out_i" && fail i "a dead rank must not print a boot time:
$out_i"
ok i "a rank that exits while a foreign server answers 200 fails the launch"

# ── (j) an endpoint that already answers is refused before launch ───────────
port_j="$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')"
stub_j="$(start_health_stub "$port_j" 0)"
run_j="$tmp/run-j"
out_j="$(NGPUS=1 EP_SIZE=1 TP_SIZE=1 SPARK_BIN="$tmp/stub-spark" \
          PORT_BASE="$port_j" ATLAS_NODE_RUN_DIR="$run_j" \
          ATLAS_NODE_HEALTH_URL="http://127.0.0.1:$port_j/health" \
          BOOT_TIMEOUT_S=15 bash "$SCRIPT" "$MODEL" 2>&1)"; rc=$?
kill "$stub_j" 2>/dev/null; wait "$stub_j" 2>/dev/null
[ $rc -ne 0 ] || fail j "an occupied endpoint must be refused, got 0:
$out_j"
have "$out_j" "already answering" || fail j "the refusal must say what it found: $out_j"
[ -f "$run_j/rank0.pid" ] && fail j "a refused launch must start no rank"
ok j "a port that already answers /health is refused before anything is started"

echo ""
echo "ALL $asserts assertions passed."
