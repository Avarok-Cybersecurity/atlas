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
#       Each create also records the IMAGE it resolved (rank<N>.image): the
#       campaign artifact names the build that served, and a tag inspected
#       after the fact is whatever the tag points at now.
#   (h) two invocations with different run directories and ports, against a
#       fake Docker: neither may stop or remove the other's rank, and --stop
#       from one run directory must leave the other's container alone. The
#       container name used to be the global `atlas-node-ep-rankN`, so run B
#       force-removed run A's live rank before starting its own.
#   (i) a rank that dies while a FOREIGN server answers 200 on the port must
#       fail the launch, not report a boot time. Otherwise the benchmark that
#       follows measures the wrong process.
#   (j) an endpoint that already answers before launch is refused up front.
#   (k) --stop reconciles an interrupted CREATE. `docker run -d` makes the
#       container and prints its ID after, so a launch killed inside that
#       window wrote no rank<N>.container and used to be indistinguishable
#       from a launch that created nothing. The intent record written before
#       the create is what --stop reconciles, and the query it reconciles with
#       carries the exact name AND this launch's label -- a container of a
#       later launch wearing the same name is not this run's to remove.
#   (l) that reconciliation must also survive a lookup that FAILS. `docker ps`
#       exiting non-zero is not "no such container": the intent is the only
#       record of the create, and deleting it on a daemon hiccup is what turns
#       a recoverable leak into an unrecoverable one.
#   (m) the same rule on the path that owns a create which COMPLETED. A
#       rank<N>.container is the authoritative record, and a cleanup that
#       cannot remove the container it names has stopped nothing: the record,
#       the image and the intent all stay, --stop fails, and the retry
#       finishes the job. A container Docker reports as already gone is not a
#       leak, so that one clears its records and succeeds.
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
# Records every call, and keeps one "CID NAME LABEL" line per live container:
# `inspect` answers whether a name exists, and `ps -aq --filter` answers which
# container wears a given name AND a given run label -- the query --stop uses
# to reconcile a create that was interrupted before it could be recorded. No
# Docker involved.
printf '%s\n' "$*" >> "$DOCKER_CALLS"
cmd="${1:-}"; shift || true
name=""; label=""
case "$cmd" in
  run)
    while [ $# -gt 0 ]; do
      case "$1" in
        --name) name="$2"; shift 2 ;;
        --label) label="$2"; shift 2 ;;
        *) shift ;;
      esac
    done
    printf '%s %s %s\n' "cid-$name" "$name" "$label" >> "$DOCKER_STATE"
    echo "cid-$name"
    ;;
  ps)
    # A lookup that FAILED is not a lookup that found nothing, and the launcher
    # has to be able to tell them apart. The marker file named by
    # DOCKER_PS_FAIL_ONCE is CONSUMED by the first `ps`, so exactly one call
    # fails the way a daemon hiccup does and the next one answers normally.
    if [ -n "${DOCKER_PS_FAIL_ONCE:-}" ] && [ -f "$DOCKER_PS_FAIL_ONCE" ]; then
      rm -f "$DOCKER_PS_FAIL_ONCE"
      echo "docker: Cannot connect to the Docker daemon at unix:///var/run/docker.sock." >&2
      exit 1
    fi
    want_name=""; want_label=""; prev=""
    for a in "$@"; do
      if [ "$prev" = "--filter" ]; then
        case "$a" in
          name=*) want_name="${a#name=}"; want_name="${want_name#^}"; want_name="${want_name%\$}" ;;
          label=*) want_label="${a#label=}" ;;
        esac
      fi
      prev="$a"
    done
    while read -r cid nm lab; do
      if [ -n "$want_name" ] && [ "$nm" != "$want_name" ]; then continue; fi
      if [ -n "$want_label" ] && [ "$lab" != "$want_label" ]; then continue; fi
      echo "$cid"
    done < "$DOCKER_STATE"
    ;;
  inspect)
    # Three questions share this verb: does a container by this name exist
    # (the pre-create probe), what IMAGE did its create resolve the tag to,
    # and what reference was that create given. The go template says which.
    fmt=""; prev=""
    for a in "$@"; do
      if [ "$prev" = "--format" ]; then fmt="$a"; fi
      prev="$a"; name="$a"
    done
    case "$fmt" in
      *Config.Image*) echo "${DOCKER_FAKE_IMAGE_REF:-avarok/atlas-gb10:latest}" ;;
      *.Image*) echo "${DOCKER_FAKE_IMAGE_ID:-sha256:feedfacefeedfacefeedfacefeedface}" ;;
      *)
        awk -v n="$name" '$2 == n { found = 1 } END { exit !found }' "$DOCKER_STATE" || exit 1
        echo "true"
        ;;
    esac
    ;;
  stop|rm)
    # Same one-shot marker for the OTHER half of a reconciliation: the lookup
    # answered, and the removal it asked for did not go through.
    if [ -n "${DOCKER_STOP_FAIL_ONCE:-}" ] && [ -f "$DOCKER_STOP_FAIL_ONCE" ]; then
      rm -f "$DOCKER_STOP_FAIL_ONCE"
      echo "docker: Error response from daemon: cannot stop container." >&2
      exit 1
    fi
    # And the STICKY variant, for as long as the marker exists. A one-shot
    # cannot express the cleanup case (m) is about, where `stop` AND `rm` both
    # fail inside the SAME --stop -- which is the shape a leak actually takes:
    # a one-shot `stop` failure is followed by an `rm` that succeeds, and the
    # container is gone for the wrong reason.
    if [ -n "${DOCKER_STOP_FAIL_WHILE:-}" ] && [ -f "$DOCKER_STOP_FAIL_WHILE" ]; then
      echo "docker: Error response from daemon: cannot stop container." >&2
      exit 1
    fi
    for a in "$@"; do name="$a"; done
    # Real Docker does not silently succeed on a container it does not have,
    # and the difference matters to the caller: "no such container" is a
    # CONFIRMED absence -- a container that has vanished is not a leak -- while
    # any other failure leaves the question open.
    if ! awk -v n="$name" '$1 == n || $2 == n { found = 1 } END { exit !found }' "$DOCKER_STATE"; then
      echo "Error response from daemon: No such container: $name" >&2
      exit 1
    fi
    # `stop` ends a container; `rm` is what makes it stop existing. Only the
    # second one may drop the line, or a `stop` that succeeded would make the
    # `rm` after it report an absence that never happened.
    if [ "$cmd" = "rm" ]; then
      awk -v n="$name" '$1 != n && $2 != n' "$DOCKER_STATE" > "$DOCKER_STATE.next"
      mv "$DOCKER_STATE.next" "$DOCKER_STATE"
    fi
    ;;
  logs) echo "(fake logs for ${*: -1})" ;;
esac
exit 0
SH
chmod +x "$tmp/fake-docker"

# The state file is "CID NAME LABEL" per live container, so a test asks about a
# container by the field it means rather than by a whole-line match.
state_has_name() { awk -v n="$1" '$2 == n { found = 1 } END { exit !found }' "$DOCKER_STATE"; }
state_has_cid() { awk -v c="$1" '$1 == c { found = 1 } END { exit !found }' "$DOCKER_STATE"; }

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
state_has_name "$name_a" || fail h "run A's container is not live: $name_a"
[ -f "$run_a/rank0.intent" ] \
  || fail h "run A recorded no intent for the create it made: $(ls "$run_a")"
# The image the create RESOLVED, written down while the container still
# exists. IMAGE is a tag and a tag can be re-pointed; the campaign's artifact
# reads this file so it can name the build that actually served, rather than
# whatever the tag points at once the container is gone.
have "$(cat "$run_a/rank0.image" 2>&1)" "id=sha256:feedfacefeedfacefeedfacefeedface" \
  || fail h "run A recorded no resolved image for its rank: $(ls "$run_a")"
ok h "a create records the image ID it resolved, not just the tag it was given"

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
state_has_name "$name_a" || fail h "run A's container did not survive run B"
ok h "starting run B leaves run A's rank alone"

: > "$DOCKER_CALLS"
out_stop_a="$(DOCKER="$tmp/fake-docker" ATLAS_NODE_RUN_DIR="$run_a" \
               bash "$SCRIPT" --stop 2>&1)"; rc=$?
[ $rc -eq 0 ] || fail h "--stop for run A exited $rc: $out_stop_a"
grep -Fq -- "$name_b" "$DOCKER_CALLS" && fail h "--stop from run A touched run B:
$(cat "$DOCKER_CALLS")"
state_has_name "$name_b" || fail h "run B's container did not survive run A's --stop"
state_has_name "$name_a" && fail h "--stop must have removed run A's own container"
[ -e "$run_a/rank0.intent" ] && fail h "--stop must clear the intent it has reconciled"
[ -e "$run_a/rank0.image" ] && fail h "--stop must clear the image record with the container"
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

# ── (k) --stop reconciles a create that was interrupted ─────────────────────
# The window `docker run -d` opens: the container exists and its name has not
# been recorded yet. All that survives it is what the launcher wrote first.
run_k="$tmp/run-k"
mkdir -p "$run_k"
name_k="atlas-run-k-9999-rank0"
label_k="atlas-node-ep.run=atlas-run-k-9999-1757000000-4242"
printf 'name=%s\nlabel=%s\n' "$name_k" "$label_k" > "$run_k/rank0.intent"
: > "$DOCKER_STATE"; : > "$DOCKER_CALLS"
printf '%s %s %s\n' "cid-$name_k" "$name_k" "$label_k" >> "$DOCKER_STATE"

out_k="$(DOCKER="$tmp/fake-docker" ATLAS_NODE_RUN_DIR="$run_k" \
          bash "$SCRIPT" --stop 2>&1)"; rc=$?
[ $rc -eq 0 ] || fail k "--stop with an intent record exited $rc: $out_k"
state_has_cid "cid-$name_k" && fail k "the container that create made is still live:
$(cat "$DOCKER_STATE")"
have "$out_k" "stopped 1 rank(s)" || fail k "--stop must count what it reconciled: $out_k"
[ -e "$run_k/rank0.intent" ] && fail k "--stop must clear the intent it reconciled"
ok k "an intent with no container record is reconciled by name and run label"

# And the reason that query carries the label as well as the name: a LATER
# launch into the same run directory and port derives the SAME container name,
# and its rank is not this run's to remove.
: > "$DOCKER_STATE"; : > "$DOCKER_CALLS"
printf 'name=%s\nlabel=%s\n' "$name_k" "$label_k" > "$run_k/rank0.intent"
printf '%s %s %s\n' "cid-later" "$name_k" "atlas-node-ep.run=a-later-launch" >> "$DOCKER_STATE"
out_k2="$(DOCKER="$tmp/fake-docker" ATLAS_NODE_RUN_DIR="$run_k" \
           bash "$SCRIPT" --stop 2>&1)"; rc=$?
[ $rc -eq 0 ] || fail k "--stop against another launch's container exited $rc: $out_k2"
state_has_cid "cid-later" || fail k "--stop removed a container another launch owns:
$out_k2"
grep -Eq '^(stop|rm) ' "$DOCKER_CALLS" && fail k "nothing of this run's exists, so nothing
may be stopped:
$(cat "$DOCKER_CALLS")"
ok k "a same-named container wearing another launch's label is left alone"

# ── (l) a lookup that FAILED is not a lookup that found nothing ─────────────
# The intent is the only record of an interrupted create, and --stop used to
# delete it whether the reconciliation query answered "no such container" or
# did not answer at all: `docker ps ... || true` collapses both into an empty
# string. One transient daemon error therefore turned a recoverable leak into
# an unrecoverable one -- the container kept the GPU and the evidence that
# would have found it was gone. An unsuccessful lookup keeps the intent, says
# which rank it could not reconcile, and fails.
run_l="$tmp/run-l"
mkdir -p "$run_l"
name_l="atlas-run-l-9999-rank0"
label_l="atlas-node-ep.run=atlas-run-l-9999-1757000001-4243"
write_intent_l() { printf 'name=%s\nlabel=%s\n' "$name_l" "$label_l" > "$run_l/rank0.intent"; }
write_intent_l
: > "$DOCKER_STATE"; : > "$DOCKER_CALLS"
printf '%s %s %s\n' "cid-$name_l" "$name_l" "$label_l" >> "$DOCKER_STATE"

: > "$tmp/ps-fail-once"
out_l="$(DOCKER="$tmp/fake-docker" DOCKER_PS_FAIL_ONCE="$tmp/ps-fail-once" \
          ATLAS_NODE_RUN_DIR="$run_l" bash "$SCRIPT" --stop 2>&1)"; rc=$?
[ $rc -ne 0 ] || fail l "a --stop whose lookup failed must not report success:
$out_l"
have "$out_l" "$name_l" || fail l "the failure must name the rank it could not reconcile:
$out_l"
[ -f "$run_l/rank0.intent" ] \
  || fail l "a failed lookup must keep the intent it could not act on: $(ls "$run_l")"
state_has_cid "cid-$name_l" \
  || fail l "nothing was confirmed absent, so the container must still be live:
$(cat "$DOCKER_STATE")"
grep -Eq '^(stop|rm) ' "$DOCKER_CALLS" && fail l "a lookup that failed answers nothing to act on:
$(cat "$DOCKER_CALLS")"
ok l "a --stop whose docker ps failed keeps the intent, touches nothing and exits non-zero"

# The other half of the same rule: the lookup answered and the REMOVAL did not.
: > "$DOCKER_CALLS"; : > "$tmp/stop-fail-once"
out_l2="$(DOCKER="$tmp/fake-docker" DOCKER_STOP_FAIL_ONCE="$tmp/stop-fail-once" \
           ATLAS_NODE_RUN_DIR="$run_l" bash "$SCRIPT" --stop 2>&1)"; rc=$?
[ $rc -ne 0 ] || fail l "a --stop whose docker stop failed must not report success:
$out_l2"
[ -f "$run_l/rank0.intent" ] || fail l "a failed removal must keep the intent: $(ls "$run_l")"
state_has_cid "cid-$name_l" || fail l "the container the stop failed on is gone from the state:
$(cat "$DOCKER_STATE")"
ok l "a --stop whose docker stop failed keeps the intent and exits non-zero"

: > "$DOCKER_CALLS"
out_l3="$(DOCKER="$tmp/fake-docker" ATLAS_NODE_RUN_DIR="$run_l" \
           bash "$SCRIPT" --stop 2>&1)"; rc=$?
[ $rc -eq 0 ] || fail l "the retry against a healthy Docker exited $rc: $out_l3"
state_has_cid "cid-$name_l" && fail l "the retry must remove the container the first --stop left:
$(cat "$DOCKER_STATE")"
[ -e "$run_l/rank0.intent" ] && fail l "the retry must clear the intent it reconciled"
ok l "the next --stop, with Docker answering, removes the leaked rank and clears the intent"

# ── (m) a completed create whose cleanup FAILED keeps its records ───────────
# (l) fixed the path for a create that never got to write its container record.
# This is the path for one that did -- and it was the path with no error
# handling at all: `docker stop || true`, `docker rm || true`, then the
# container record AND the image record deleted whatever came back, with the
# intent already thrown away upstream for the sole reason that a container
# record existed. Against a Docker that refuses both halves of the removal,
# --stop therefore exited 0, reported the rank stopped, and destroyed every
# record of a container still holding a GPU. Nothing could find it afterwards:
# the retry has nothing left to read.
run_m="$tmp/run-m"
mkdir -p "$run_m"
name_m="atlas-run-m-9999-rank0"
label_m="atlas-node-ep.run=atlas-run-m-9999-1757000002-4244"
write_records_m() {
  printf '%s\n' "$name_m" > "$run_m/rank0.container"
  printf 'id=%s\nref=%s\n' "sha256:feedfacefeedfacefeedfacefeedface" \
    "avarok/atlas-gb10:latest" > "$run_m/rank0.image"
  printf 'name=%s\nlabel=%s\n' "$name_m" "$label_m" > "$run_m/rank0.intent"
}
write_records_m
: > "$DOCKER_STATE"; : > "$DOCKER_CALLS"
printf '%s %s %s\n' "cid-$name_m" "$name_m" "$label_m" >> "$DOCKER_STATE"

: > "$tmp/docker-down"
out_m="$(DOCKER="$tmp/fake-docker" DOCKER_STOP_FAIL_WHILE="$tmp/docker-down" \
          ATLAS_NODE_RUN_DIR="$run_m" bash "$SCRIPT" --stop 2>&1)"; rc=$?
[ $rc -ne 0 ] || fail m "a --stop that could not remove its container must not report success:
$out_m"
have "$out_m" "$name_m" || fail m "the failure must name the rank it could not clean up:
$out_m"
state_has_name "$name_m" || fail m "the removal failed, so the container must still be live:
$(cat "$DOCKER_STATE")"
for rec in container image intent; do
  [ -f "$run_m/rank0.$rec" ] \
    || fail m "a failed cleanup must keep rank0.$rec: $(ls "$run_m")"
done
ok m "a --stop whose removal failed keeps the container, image and intent records"

rm -f "$tmp/docker-down"
: > "$DOCKER_CALLS"
out_m2="$(DOCKER="$tmp/fake-docker" ATLAS_NODE_RUN_DIR="$run_m" \
           bash "$SCRIPT" --stop 2>&1)"; rc=$?
[ $rc -eq 0 ] || fail m "the retry against a healthy Docker exited $rc: $out_m2"
state_has_name "$name_m" && fail m "the retry must remove the container the first --stop left:
$(cat "$DOCKER_STATE")"
for rec in container image intent; do
  [ -e "$run_m/rank0.$rec" ] && fail m "the retry must clear rank0.$rec"
done
ok m "the next --stop, with Docker answering, removes the container and clears all three records"

# The other half of the rule, so that keeping evidence does not become keeping
# it forever: a container Docker reports as gone is not a leak.
write_records_m
: > "$DOCKER_STATE"; : > "$DOCKER_CALLS"
out_m3="$(DOCKER="$tmp/fake-docker" ATLAS_NODE_RUN_DIR="$run_m" \
           bash "$SCRIPT" --stop 2>&1)"; rc=$?
[ $rc -eq 0 ] || fail m "a container already gone must not fail the --stop: $out_m3"
for rec in container image intent; do
  [ -e "$run_m/rank0.$rec" ] && fail m "a confirmed absence must clear rank0.$rec: $out_m3"
done
ok m "a container Docker answers 'No such container' for is confirmed absent, not unresolved"

echo ""
echo "ALL $asserts assertions passed."
