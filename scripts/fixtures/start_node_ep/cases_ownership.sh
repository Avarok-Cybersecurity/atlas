#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Sourced by start_node_ep_test.sh after its isolated HTTP and Docker fixtures.
: "${tmp:?source this after start_node_ep_test.sh fixture setup}"
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
name_a="$(record_field "$run_a/rank0.container" name)"
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

# And the ID `docker run -d` printed, which is what --stop proves ownership
# with: the name is only on loan for as long as this container holds it.
[ "$(record_field "$run_a/rank0.container" id)" = "cid-$name_a" ] \
  || fail h "run A recorded no created container ID: $(cat "$run_a/rank0.container")"
[ "$(record_field "$run_a/rank0.container" label)" \
  = "$(record_field "$run_a/rank0.intent" label)" ] \
  || fail h "the container record must carry the same run label as the intent:
$(cat "$run_a/rank0.container")"
ok h "a create records the container ID it made and the run label it stamped"

: > "$DOCKER_CALLS"
run_b="$tmp/run-b"
out_b="$(NGPUS=1 EP_SIZE=1 TP_SIZE=1 IMAGE=avarok/atlas-gb10:latest \
          DOCKER="$tmp/fake-docker" PORT_BASE="$port_b" \
          ATLAS_NODE_RUN_DIR="$run_b" \
          ATLAS_NODE_HEALTH_URL="http://127.0.0.1:$port_b/health" \
          BOOT_TIMEOUT_S=30 bash "$SCRIPT" "$MODEL" 2>&1)"; rc=$?
[ $rc -eq 0 ] || fail h "run B exited $rc:
$out_b"
name_b="$(record_field "$run_b/rank0.container" name)"
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
  printf 'id=%s\nname=%s\nlabel=%s\n' "cid-$name_m" "$name_m" "$label_m" \
    > "$run_m/rank0.container"
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

# ── (n) a completed record is owned by ID and label, never by name ──────────
run_n="$tmp/run-n"
mkdir -p "$run_n"
name_n="atlas-run-n-9999-rank0"
label_n="atlas-node-ep.run=atlas-run-n-9999-1757000003-4245"
write_records_n() {
  printf 'id=%s\nname=%s\nlabel=%s\n' "cid-$name_n" "$name_n" "$label_n" \
    > "$run_n/rank0.container"
  printf 'id=%s\nref=%s\n' "sha256:feedfacefeedfacefeedfacefeedface" \
    "avarok/atlas-gb10:latest" > "$run_n/rank0.image"
  printf 'name=%s\nlabel=%s\n' "$name_n" "$label_n" > "$run_n/rank0.intent"
}

# Ours, live, wearing our label: stopped and removed, and addressed by the ID
# the create returned rather than by the name it was given.
write_records_n
: > "$DOCKER_STATE"; : > "$DOCKER_CALLS"
printf '%s %s %s\n' "cid-$name_n" "$name_n" "$label_n" >> "$DOCKER_STATE"
out_n="$(DOCKER="$tmp/fake-docker" ATLAS_NODE_RUN_DIR="$run_n" \
          bash "$SCRIPT" --stop 2>&1)"; rc=$?
[ $rc -eq 0 ] || fail n "--stop against this run's own live container exited $rc: $out_n"
state_has_cid "cid-$name_n" && fail n "--stop must remove the container it recorded:
$(cat "$DOCKER_STATE")"
have "$out_n" "stopped 1 rank(s)" || fail n "--stop must count the rank it stopped: $out_n"
grep -Eq "^rm cid-$name_n\$" "$DOCKER_CALLS" \
  || fail n "the removal must address the container by its ID:
$(cat "$DOCKER_CALLS")"
for rec in container image intent; do
  [ -e "$run_n/rank0.$rec" ] && fail n "a confirmed removal must clear rank0.$rec"
done
ok n "a completed record is stopped and removed by the container ID its create returned"

# The regression. Our container was removed outside this launcher, and a
# REPLACEMENT now answers to the same name with a different ID and a different
# run label. Removing by name destroyed that replacement. Nothing wearing our
# ID and our label exists any more, and "ours is gone" is the whole of what
# this record can still say.
write_records_n
: > "$DOCKER_STATE"; : > "$DOCKER_CALLS"
printf '%s %s %s\n' "cid-replacement" "$name_n" "atlas-node-ep.run=a-later-launch" \
  >> "$DOCKER_STATE"
out_n2="$(DOCKER="$tmp/fake-docker" ATLAS_NODE_RUN_DIR="$run_n" \
           bash "$SCRIPT" --stop 2>&1)"; rc=$?
[ $rc -eq 0 ] || fail n "a rank confirmed gone must not fail the --stop: $out_n2"
grep -Eq '^(stop|rm) ' "$DOCKER_CALLS" && fail n "nothing of this run's exists, so nothing
may be stopped:
$(cat "$DOCKER_CALLS")"
state_has_cid "cid-replacement" || fail n "--stop removed a container a later launch owns:
$out_n2"
have "$out_n2" "$name_n" || fail n "--stop must name the rank it found gone: $out_n2"
for rec in container image intent; do
  [ -e "$run_n/rank0.$rec" ] && fail n "a confirmed absence must clear rank0.$rec: $out_n2"
done
ok n "a same-named container with another ID and label is left alone, ours confirmed gone"
