#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Sourced by campaign_test.sh; uses its temporary fixtures and assertions.
# shellcheck disable=SC2154,SC2034

# ── (l) a TERMINATED cell tears down what it owns and says what happened ────
# The stub server answers 503 forever, which is exactly what a loading engine
# looks like, so the runner is parked in its boot poll when the signal lands --
# the moment a container of its own exists and no gate has finished.
port="$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')"
export STUB_HITS="$tmp/loading.hits"; : > "$STUB_HITS"
cat > "$tmp/loading_stub.py" <<'PY'
import os
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

BODY = b'{"status": "loading"}'


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        with open(os.environ["STUB_HITS"], "a") as fh:
            fh.write(self.path + "\n")
        self.send_response(503)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(BODY)))
        self.end_headers()
        self.wfile.write(BODY)

    def log_message(self, *args):
        pass


HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
PY
python3 "$tmp/loading_stub.py" "$port" & stub_pid=$!

# Poll for a condition rather than sleeping a guessed interval: a fixed sleep
# either flakes on a slow box or wastes the difference on a fast one.
wait_for() {  # wait_for SECONDS WHAT COMMAND...
  local deadline what
  deadline=$(( $(date +%s) + $1 )); shift
  what="$1"; shift
  while [ "$(date +%s)" -lt "$deadline" ]; do
    if "$@"; then return 0; fi
    sleep 0.2
  done
  echo "timed out waiting for $what" >&2
  return 1
}

: > "$DOCKER_CALLS"
DOCKER_FAKE_CID="$CID" PATH="$tmp/bin:$PATH" VLLM_IMAGE_DIGEST="$DIGEST" \
  VLLM_PORT="$port" \
  bash "$RUN" --engine vllm --model nemotron-3-nano-fp8 --sku h100 \
  --workload lat --concurrency 1 --spec off --think off \
  --out "$tmp/killed" --yes > "$tmp/killed.log" 2>&1 &
run_pid=$!

wait_for 120 "the cell's own docker run" grep -q '^run ' "$DOCKER_CALLS" \
  || fail l "the cell never created its container:
$(cat "$tmp/killed.log")"
wait_for 120 "the boot poll to reach the stub" test -s "$STUB_HITS" \
  || fail l "the cell never reached its boot gate:
$(cat "$tmp/killed.log")"

kill -TERM "$run_pid"
wait "$run_pid"; rc=$?
kill "$stub_pid" 2>/dev/null; wait "$stub_pid" 2>/dev/null
log="$(cat "$tmp/killed.log")"

[ $rc -eq 143 ] || fail l "a cell killed with SIGTERM must exit 143, got $rc:
$log"
ok l "SIGTERM during the boot gate exits 143, not a bare -15"

have "$(cat "$DOCKER_CALLS")" "stop $CID" \
  || fail l "the container this cell created was never stopped:
$(cat "$DOCKER_CALLS")"
have "$(cat "$DOCKER_CALLS")" "rm $CID" \
  || fail l "the container this cell created was never removed:
$(cat "$DOCKER_CALLS")"
stray="$(grep -E '^(stop|rm) ' "$DOCKER_CALLS" | grep -v -F -- "$CID" || true)"
[ -z "$stray" ] || fail l "a terminated cell must touch nothing but its own container, saw:
$stray"
ok l "SIGTERM stops and removes exactly the container this invocation created"

have "$log" "killing pid" \
  || fail l "the in-flight boot-gate child must be killed by pid:
$log"
ok l "the boot-gate child this cell started is killed by pid, not left polling"

art="$tmp/killed/artifact.json"
[ -f "$art" ] || fail l "a terminated cell must still write its artifact:
$log"
python3 "$HERE/validate_artifact.py" "$art" >/dev/null \
  || fail l "the interruption artifact must validate:
$(python3 "$HERE/validate_artifact.py" "$art")"
ok l "a terminated cell still writes an artifact, and it validates"

read -r verdict stage_name <<<"$(python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
print(d["verdict"], d["failing_stage"])' "$art")"
[ "$verdict" = "NO-GO" ] || fail l "an interrupted cell is a NO-GO, got $verdict"
[ "$stage_name" = "boot" ] \
  || fail l "an interrupted cell must name the stage it was killed in, got $stage_name"
ok l "the artifact is NO-GO at 'boot' -- the stage that was in flight"

python3 -c '
import json, sys
notes = json.load(open(sys.argv[1]))["notes"]
assert "interrupted by SIGTERM" in notes, notes
assert "boot stage" in notes, notes' "$art" \
  || fail l "the notes must record the interruption, not just a failed stage:
$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))[\"notes\"])" "$art")"
ok l "the notes say it was interrupted by SIGTERM rather than that boot failed on its own"

# ── (m) SIGTERM while the container is still being created ──────────────────
# The narrower window than (l): `docker run -d` has made the container and has
# not yet printed its ID, so the ID this cell tears down by does not exist in
# this shell yet. The stub blocks in exactly that gap. Ownership therefore has
# to be recoverable from something chosen and recorded BEFORE the create -- the
# per-invocation label -- or an interrupted create leaks a live server.
: > "$DOCKER_CALLS"; : > "$DOCKER_RUNNING"
DOCKER_FAKE_CID="$CID" DOCKER_RUN_BLOCK_S=60 PATH="$tmp/bin:$PATH" \
  VLLM_IMAGE_DIGEST="$DIGEST" \
  bash "$RUN" --engine vllm --model nemotron-3-nano-fp8 --sku h100 \
  --workload lat --concurrency 1 --spec off --think off \
  --out "$tmp/creating" --yes > "$tmp/creating.log" 2>&1 &
run_pid=$!

wait_for 120 "docker run to create the container" test -s "$DOCKER_RUNNING" \
  || fail m "the cell never got as far as creating its container:
$(cat "$tmp/creating.log")"

kill -TERM "$run_pid"
wait "$run_pid"; rc=$?
log="$(cat "$tmp/creating.log")"

[ $rc -eq 143 ] || fail m "a cell killed mid-create must exit 143, got $rc:
$log"
ok m "SIGTERM during the create exits 143"

[ -s "$DOCKER_RUNNING" ] && fail m "the container this cell created is still running:
$(cat "$DOCKER_RUNNING")
docker calls:
$(cat "$DOCKER_CALLS")"
ok m "the container the create had already made is gone, not leaked"

have "$(cat "$DOCKER_CALLS")" "stop $CID" \
  || fail m "the interrupted create's container was never stopped:
$(cat "$DOCKER_CALLS")"
have "$(cat "$DOCKER_CALLS")" "rm $CID" \
  || fail m "the interrupted create's container was never removed:
$(cat "$DOCKER_CALLS")"
stray="$(grep -E '^(stop|rm) ' "$DOCKER_CALLS" | grep -v -F -- "$CID" || true)"
[ -z "$stray" ] || fail m "the reconciliation must touch nothing else, saw:
$stray"
ok m "SIGTERM mid-create stops and removes exactly the container that was made"

art="$tmp/creating/artifact.json"
[ -f "$art" ] || fail m "a cell interrupted mid-create must still write its artifact:
$log"
python3 "$HERE/validate_artifact.py" "$art" >/dev/null \
  || fail m "the mid-create interruption artifact must validate:
$(python3 "$HERE/validate_artifact.py" "$art")"
read -r verdict stage_name <<<"$(python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
print(d["verdict"], d["failing_stage"])' "$art")"
[ "$verdict" = "NO-GO" ] || fail m "an interrupted cell is a NO-GO, got $verdict"
[ "$stage_name" = "serve" ] \
  || fail m "a cell killed inside its create was killed at serve, got $stage_name"
ok m "the artifact is a valid NO-GO at 'serve'"

# ── (n) a REFUSED atlas launch owns nothing of a prior run's ────────────────
# The wrapper used to treat "I started the launcher" as "I own what this run
# directory records". The launcher, handed a directory that already held rank
# records and a port whose /health already answered, refused before creating
# anything -- and the wrapper's finalizer then ran --stop against that
# directory and killed the earlier run's rank. Nothing there was ever this
# invocation's, and the fix is that this invocation's run directory is its own.
prior="$tmp/prior-node-ep"
mkdir -p "$prior"
# A container record only: a pid record would make a failing test kill whatever
# process happens to hold that number on this box.
echo "prior-invocation-rank0" > "$prior/rank0.container"

ready_port="$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')"
cat > "$tmp/ready_stub.py" <<'READYPY'
import sys
from http.server import BaseHTTPRequestHandler, HTTPServer

BODY = b'{"status": "ready"}'


class H(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(BODY)))
        self.end_headers()
        self.wfile.write(BODY)

    def log_message(self, *args):
        pass


HTTPServer(("127.0.0.1", int(sys.argv[1])), H).serve_forever()
READYPY
python3 "$tmp/ready_stub.py" "$ready_port" >/dev/null 2>&1 & ready_pid=$!
wait_for 30 "the occupied /health endpoint" \
  curl -sf -o /dev/null "http://127.0.0.1:$ready_port/health" \
  || fail n "the stub endpoint never came up"

# The kernel audit cannot pass on a host with no GPU, and that is the stage
# failure this cell records. What matters is what the launcher does next.
cat > "$tmp/stub-spark" <<'STUBSH'
#!/usr/bin/env bash
echo "stub spark: $*"
echo "stub spark: no CUDA device on this host" >&2
exit 1
STUBSH
chmod +x "$tmp/stub-spark"

: > "$DOCKER_CALLS"; : > "$DOCKER_RUNNING"
out="$(PATH="$tmp/bin:$PATH" ATLAS_PORT="$ready_port" \
        ATLAS_NODE_RUN_DIR="$prior" SPARK_BIN="$tmp/stub-spark" \
        bash "$RUN" --engine atlas --model nemotron-3-nano-fp8 --sku h100 \
        --workload lat --concurrency 1 --spec off --think off \
        --out "$tmp/refused" --yes 2>&1)"; rc=$?
kill "$ready_pid" 2>/dev/null; wait "$ready_pid" 2>/dev/null

[ $rc -eq 1 ] || fail n "a cell whose serve stage failed must exit 1, got $rc:
$out"
have "$(cat "$tmp/refused/serve.log")" "REFUSED" \
  || fail n "the launcher was supposed to refuse the occupied endpoint:
$(cat "$tmp/refused/serve.log")"
ok n "the launcher refuses an occupied /health and the cell records a failed serve"

touched="$(grep -E '^(stop|rm) prior-invocation-rank0' "$DOCKER_CALLS" || true)"
[ -z "$touched" ] || fail n "a refused launch must stop nothing of the prior run, saw:
$touched"
[ -f "$prior/rank0.container" ] \
  || fail n "the prior run's own record was deleted by a launch that created nothing"
ok n "a refused launch leaves the prior run's rank and its record untouched"

art="$tmp/refused/artifact.json"
python3 "$HERE/validate_artifact.py" "$art" >/dev/null \
  || fail n "the refusal artifact must validate:
$(python3 "$HERE/validate_artifact.py" "$art")"
read -r verdict stage_name <<<"$(python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
print(d["verdict"], d["failing_stage"])' "$art")"
[ "$verdict" = "NO-GO" ] || fail n "a cell that never served is a NO-GO, got $verdict"
[ "$stage_name" = "serve" ] || fail n "the failing stage must be serve, got $stage_name"
ok n "the artifact is a valid NO-GO at 'serve'"

# And the other half of the same rule: a launch that DID record its ranks is
# stopped. Ownership by evidence has to keep the case it was always right about.
cat > "$tmp/stub-spark-alive" <<'STUBSH'
#!/usr/bin/env bash
for a in "$@"; do
  if [ "$a" = "--check-kernels" ]; then echo "stub spark: kernels resolved"; exit 0; fi
done
echo "stub spark serve: $*"
sleep 45
STUBSH
chmod +x "$tmp/stub-spark-alive"

live_port="$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')"
: > "$STUB_HITS"
python3 "$tmp/loading_stub.py" "$live_port" >/dev/null 2>&1 & stub_pid=$!
PATH="$tmp/bin:$PATH" ATLAS_PORT="$live_port" SPARK_BIN="$tmp/stub-spark-alive" \
  bash "$RUN" --engine atlas --model nemotron-3-nano-fp8 --sku h100 \
  --workload lat --concurrency 1 --spec off --think off \
  --out "$tmp/live" --yes > "$tmp/live.log" 2>&1 &
run_pid=$!

# The run directory carries the runner's own pid: that is what makes it this
# invocation's alone, and it is where the launcher's rank records land.
live_dir="$tmp/live/node-ep-$run_pid"
wait_for 120 "the launcher to record its rank" test -s "$live_dir/rank0.pid" \
  || fail n "no rank record appeared in $live_dir:
$(cat "$tmp/live.log")"
rank_pid="$(cat "$live_dir/rank0.pid")"

kill -TERM "$run_pid"
wait "$run_pid"; rc=$?
kill "$stub_pid" 2>/dev/null; wait "$stub_pid" 2>/dev/null
log="$(cat "$tmp/live.log")"

[ $rc -eq 143 ] || fail n "a terminated atlas cell must exit 143, got $rc:
$log"
# --stop reached this cell's OWN run directory and the record in it. Whether
# the rank was already down by then (the kill of the launcher's process tree
# can get there first) or --stop killed it is the platform's business; that the
# teardown went through the reserved directory and left the rank dead is not.
have "$log" "stopped 1 rank(s); logs kept in $live_dir" \
  || fail n "--stop never reached the run directory this cell reserved:
$log"
[ -e "$live_dir/rank0.pid" ] && fail n "the rank record this cell owned was not cleared:
$log"
kill -0 "$rank_pid" 2>/dev/null && { kill -9 "$rank_pid" 2>/dev/null; \
  fail n "rank pid $rank_pid survived the teardown"; }
ok n "a launch that recorded a rank in its own run dir is stopped and cleared"

art="$tmp/live/artifact.json"
python3 "$HERE/validate_artifact.py" "$art" >/dev/null \
  || fail n "the interrupted atlas artifact must validate:
$(python3 "$HERE/validate_artifact.py" "$art")"
ok n "the interrupted atlas cell still writes an artifact that validates"

