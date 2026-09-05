#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Sourced by campaign_test.sh; uses its temporary fixtures and assertions.
# shellcheck disable=SC2154,SC2034

# ── (o) SIGTERM inside an ATLAS container create ────────────────────────────
# The Atlas half of (m), and the same window: the launcher writes
# rank<N>.container only once `docker run -d` has returned, and the wrapper
# read an absent record as proof that nothing was created -- so a cell killed
# inside the create stopped nothing and left a rank holding the GPU. What
# survives that window is what the launcher wrote BEFORE the create:
# rank<N>.intent, carrying the deterministic container name and this launch's
# own label. The decoy below is why the reconciliation has to carry both: a
# container wearing the SAME name for a DIFFERENT run is somebody else's, and
# a name-only query would delete it.
atlas_port="$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')"
ACID="1111111111111111111111111111111111111111111111111111111111111111"
DECOY_CID="2222222222222222222222222222222222222222222222222222222222222222"
# What the image itself says it is, read back by (p) below: the digest
# `docker inspect` reports for it, and the revision label it carries.
ADIGEST="sha256:$(printf 'c%.0s' $(seq 64))"
AREV="4b825dc642cb6eb9a060e54bf8d69288fbee4904"
: > "$DOCKER_CALLS"; : > "$DOCKER_RUNNING"
DOCKER_FAKE_CID="$ACID" DOCKER_RUN_BLOCK_S=60 PATH="$tmp/bin:$PATH" \
  DOCKER_FAKE_DIGEST="$ADIGEST" DOCKER_FAKE_REVISION="$AREV" \
  IMAGE="avarok/atlas-fake:campaign-test" ATLAS_PORT="$atlas_port" \
  bash "$RUN" --engine atlas --model nemotron-3-nano-fp8 --sku h100 \
  --workload lat --concurrency 1 --spec off --think off \
  --out "$tmp/creating-atlas" --yes > "$tmp/creating-atlas.log" 2>&1 &
run_pid=$!

wait_for 120 "the launcher's docker run to create rank 0" test -s "$DOCKER_RUNNING" \
  || fail o "the launcher never got as far as creating its container:
$(cat "$tmp/creating-atlas.log")"
created_name="$(awk 'NR==1 {print $NF}' "$DOCKER_RUNNING")"
[ -n "$created_name" ] || fail o "the create recorded no container name:
$(cat "$DOCKER_RUNNING")"
printf '%s %s %s\n' "$DECOY_CID" "atlas-node-ep.run=some-other-launch" "$created_name" \
  >> "$DOCKER_RUNNING"

kill -TERM "$run_pid"
wait "$run_pid"; rc=$?
log="$(cat "$tmp/creating-atlas.log")"

[ $rc -eq 143 ] || fail o "an atlas cell killed mid-create must exit 143, got $rc:
$log"
ok o "SIGTERM during an Atlas container create exits 143"

have "$(cat "$DOCKER_CALLS")" "stop $ACID" \
  || fail o "the rank the create had already made was never stopped:
$(cat "$DOCKER_CALLS")"
have "$(cat "$DOCKER_CALLS")" "rm $ACID" \
  || fail o "the rank the create had already made was never removed:
$(cat "$DOCKER_CALLS")"
ok o "SIGTERM mid-create stops and removes the rank the launcher had created"

left="$(awk '{print $1}' "$DOCKER_RUNNING" | tr '\n' ' ' | sed 's/ $//')"
[ "$left" = "$DECOY_CID" ] || fail o "the running set must hold the other run's container and
nothing else, holds: $left
docker calls:
$(cat "$DOCKER_CALLS")"
stray="$(grep -E "^(stop|rm) $DECOY_CID" "$DOCKER_CALLS" || true)"
[ -z "$stray" ] || fail o "a container of another run wearing the same name was touched:
$stray"
ok o "a same-named container from another run is neither stopped nor removed"

art="$tmp/creating-atlas/artifact.json"
[ -f "$art" ] || fail o "an atlas cell interrupted mid-create must still write its artifact:
$log"
python3 "$HERE/validate_artifact.py" "$art" >/dev/null \
  || fail o "the mid-create interruption artifact must validate:
$(python3 "$HERE/validate_artifact.py" "$art")"
read -r verdict stage_name <<<"$(python3 -c '
import json, sys
d = json.load(open(sys.argv[1]))
print(d["verdict"], d["failing_stage"])' "$art")"
[ "$verdict" = "NO-GO" ] || fail o "an interrupted cell is a NO-GO, got $verdict"
case "$stage_name" in
  serve|boot) ;;
  *) fail o "a cell killed around its create was killed at serve or boot, got $stage_name" ;;
esac
ok o "the artifact is a valid NO-GO at '$stage_name'"

# ── (p) engine_version identifies the ENGINE, not the harness checkout ──────
# The reviewer's second finding: run_cell passed `git rev-parse HEAD` of this
# checkout as --git-sha for either engine and captured a digest only for vLLM,
# so an Atlas container cell reported the campaign's revision with
# image_digest=null and binary_sha256=null -- and validated as CERTIFIED. What
# an artifact has to name is the build that actually served the requests: the
# digest of the image that ran and the revision that image declares, or the
# hash of the binary that ran. The checkout SHA is harness provenance, and it
# now says so.
harness_sha="$(git -C "$ROOT" rev-parse HEAD 2>/dev/null || true)"

atlas_art="$tmp/creating-atlas/artifact.json"
python3 - "$atlas_art" "$ADIGEST" "$AREV" "$harness_sha" <<'PY' || fail p "the Atlas container cell did not record the engine that ran: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["engine_version"], d.get("harness"))' "$atlas_art")"
import json, sys
doc = json.load(open(sys.argv[1]))
digest, rev, harness = sys.argv[2], sys.argv[3], sys.argv[4]
ev = doc["engine_version"]
assert ev["image_digest"] == digest, ev
assert ev["git_sha"] == rev, ev
assert ev["git_sha"] != harness, ev
assert ev["binary_sha256"] is None, ev
assert doc["harness"]["git_sha"] == (harness or None), (doc["harness"], harness)
PY
ok p "an Atlas container cell records the image's digest and revision, not the checkout"

local_art="$tmp/live/artifact.json"
python3 - "$local_art" "$tmp/stub-spark-alive" "$harness_sha" <<'PY' || fail p "the Atlas local-binary cell did not record the binary that ran: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["engine_version"], d.get("harness"))' "$local_art")"
import hashlib, json, sys
doc = json.load(open(sys.argv[1]))
ev = doc["engine_version"]
assert ev["binary_sha256"] == hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest(), ev
# `spark --version` prints CARGO_PKG_VERSION and nothing else
# (crates/spark-server/src/cli.rs), so there is no engine revision to record.
assert ev["git_sha"] is None, ev
assert ev["image_digest"] is None, ev
assert doc["harness"]["git_sha"] == (sys.argv[3] or None), doc["harness"]
PY
ok p "an Atlas local-binary cell hashes the binary it ran and claims no revision"

vllm_art="$tmp/creating/artifact.json"
python3 - "$vllm_art" "$DIGEST" "$harness_sha" <<'PY' || fail p "the vLLM cell's provenance changed: $(python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); print(d["engine_version"], d.get("harness"))' "$vllm_art")"
import json, sys
doc = json.load(open(sys.argv[1]))
ev = doc["engine_version"]
assert ev["image_digest"] == sys.argv[2], ev
assert ev["git_sha"] is None, ev
assert ev["binary_sha256"] is None, ev
assert doc["harness"]["git_sha"] == (sys.argv[3] or None), doc["harness"]
PY
ok p "the vLLM cell keeps the digest it was pinned to and claims no engine revision"

# ── (r) the engine identity is the image that RAN, not the tag afterwards ───
# capture_engine_identity used to run AFTER teardown and inspect $IMAGE -- the
# tag. A tag is a mutable pointer: a rebuild or a pull between the launch and
# the finalizer re-points it, and the artifact then names a build that never
# served a request. What the container started from is its resolved image ID,
# recorded the moment the create returns and inspected before teardown removes
# the container. Here tag T resolves to image A at create time and is
# re-pointed to B during the run; the artifact must say A.
IMG_A="sha256:$(printf 'd%.0s' $(seq 64))"
IMG_B="sha256:$(printf 'e%.0s' $(seq 64))"
DIGEST_A="sha256:$(printf '1%.0s' $(seq 64))"
DIGEST_B="sha256:$(printf '2%.0s' $(seq 64))"
REV_A="aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
REV_B="bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb"
MUTABLE_TAG="avarok/atlas-fake:mutable"
export DOCKER_REGISTRY="$tmp/docker.registry"
export DOCKER_TAG_IMAGE="$tmp/docker.tag-image"
{
  printf '%s %s %s\n' "$IMG_A" "$DIGEST_A" "$REV_A"
  printf '%s %s %s\n' "$IMG_B" "$DIGEST_B" "$REV_B"
  # The TAG's own answers are B's, because by the time anything inspects the
  # tag it has been re-pointed. An artifact carrying these is the bug.
  printf '%s %s %s\n' "$MUTABLE_TAG" "$DIGEST_B" "$REV_B"
} > "$DOCKER_REGISTRY"
printf '%s\n' "$IMG_A" > "$DOCKER_TAG_IMAGE"

mutable_port="$(python3 -c 'import socket;s=socket.socket();s.bind(("127.0.0.1",0));print(s.getsockname()[1]);s.close()')"
: > "$STUB_HITS"
python3 "$tmp/loading_stub.py" "$mutable_port" >/dev/null 2>&1 & stub_pid=$!
: > "$DOCKER_CALLS"; : > "$DOCKER_RUNNING"; : > "$DOCKER_CREATED_IMAGES"
DOCKER_FAKE_CID="3333333333333333333333333333333333333333333333333333333333333333" \
  PATH="$tmp/bin:$PATH" IMAGE="$MUTABLE_TAG" ATLAS_PORT="$mutable_port" \
  bash "$RUN" --engine atlas --model nemotron-3-nano-fp8 --sku h100 \
  --workload lat --concurrency 1 --spec off --think off \
  --out "$tmp/mutable-tag" --yes > "$tmp/mutable-tag.log" 2>&1 &
run_pid=$!

mutable_dir="$tmp/mutable-tag/node-ep-$run_pid"
wait_for 120 "the launcher to record the image its create resolved" \
  test -s "$mutable_dir/rank0.image" \
  || fail r "the launcher recorded no image for the rank it created:
$(cat "$tmp/mutable-tag.log")"
have "$(cat "$mutable_dir/rank0.image")" "id=$IMG_A" \
  || fail r "the launcher must record the resolved image ID, got:
$(cat "$mutable_dir/rank0.image")"
ok r "the launcher writes down the image its detached create actually resolved"

# The rebuild: same tag, different image, while the cell is still running.
printf '%s\n' "$IMG_B" > "$DOCKER_TAG_IMAGE"

kill -TERM "$run_pid"
wait "$run_pid"; rc=$?
kill "$stub_pid" 2>/dev/null; wait "$stub_pid" 2>/dev/null
log="$(cat "$tmp/mutable-tag.log")"

[ $rc -eq 143 ] || fail r "the interrupted cell must exit 143, got $rc:
$log"
art="$tmp/mutable-tag/artifact.json"
[ -f "$art" ] || fail r "an interrupted cell must still write its artifact:
$log"
python3 "$HERE/validate_artifact.py" "$art" >/dev/null \
  || fail r "the artifact must validate:
$(python3 "$HERE/validate_artifact.py" "$art")"
python3 - "$art" "$DIGEST_A" "$REV_A" "$DIGEST_B" "$REV_B" <<'RPY' \
  || fail r "the artifact names the build the tag points at NOW, not the one that ran: $(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["engine_version"])' "$art")"
import json, sys
ev = json.load(open(sys.argv[1]))["engine_version"]
ran_digest, ran_rev, rebuilt_digest, rebuilt_rev = sys.argv[2:6]
assert ev["image_digest"] == ran_digest, ev
assert ev["git_sha"] == ran_rev, ev
assert ev["image_digest"] != rebuilt_digest, ev
assert ev["git_sha"] != rebuilt_rev, ev
RPY
ok r "a tag re-pointed during the run does not change the build the artifact names"

grep -Fq -- "$IMG_A" "$DOCKER_CALLS" \
  || fail r "the identity lookup never inspected the recorded image ID:
$(cat "$DOCKER_CALLS")"
ok r "the identity is read from the recorded image ID, not from the tag"
unset DOCKER_REGISTRY DOCKER_TAG_IMAGE

# ── (q) a teardown lookup that FAILED is not a teardown that found nothing ──
# The launcher's half of this is scripts/start_node_ep_test.sh case (l); this
# is the runner's own label lookup. `docker ps --filter label=... || true`
# collapsed an unreachable daemon into an empty result, and the cell then
# printed "no container was created by this invocation" and finished its
# teardown stage cleanly -- while the container it created in the window the
# signal landed in kept the GPU. The artifact has to say the container may
# still be running; a silent clean teardown is the thing that loses it.
: > "$DOCKER_CALLS"; : > "$DOCKER_RUNNING"; : > "$tmp/ps-fail-once"
DOCKER_FAKE_CID="$CID" DOCKER_RUN_BLOCK_S=60 PATH="$tmp/bin:$PATH" \
  DOCKER_PS_FAIL_ONCE="$tmp/ps-fail-once" VLLM_IMAGE_DIGEST="$DIGEST" \
  bash "$RUN" --engine vllm --model nemotron-3-nano-fp8 --sku h100 \
  --workload lat --concurrency 1 --spec off --think off \
  --out "$tmp/lookup-failed" --yes > "$tmp/lookup-failed.log" 2>&1 &
run_pid=$!

wait_for 120 "docker run to create the container" test -s "$DOCKER_RUNNING" \
  || fail q "the cell never got as far as creating its container:
$(cat "$tmp/lookup-failed.log")"

kill -TERM "$run_pid"
wait "$run_pid"; rc=$?
# The signal lands while the shell is inside `run_child ... > $OUT/serve.log`,
# and that redirection is the whole function call's -- so the finalizer the
# trap runs writes THERE, not to the runner's own log. Both are read.
log="$(cat "$tmp/lookup-failed.log" "$tmp/lookup-failed/serve.log" 2>&1)"

[ $rc -eq 143 ] || fail q "a cell killed mid-create must exit 143, got $rc:
$log"
stray="$(grep -E '^(stop|rm) ' "$DOCKER_CALLS" || true)"
[ -z "$stray" ] || fail q "the lookup answered nothing, so nothing may be stopped by guess:
$stray"
ok q "a teardown whose docker ps failed stops nothing on a guess"

have "$log" "may still be running" \
  || fail q "the teardown must say the container's fate is unknown, not that there was none:
$log"
grep -Fq "no container was created by this invocation" <<<"$log" \
  && fail q "an unreachable daemon was reported as proof that nothing was created:
$log"
ok q "the teardown reports an unknown fate rather than a confirmed absence"

art="$tmp/lookup-failed/artifact.json"
[ -f "$art" ] || fail q "the cell must still write its artifact:
$log"
python3 "$HERE/validate_artifact.py" "$art" >/dev/null \
  || fail q "the artifact must validate:
$(python3 "$HERE/validate_artifact.py" "$art")"
python3 -c '
import json, sys
doc = json.load(open(sys.argv[1]))
notes = doc["notes"]
assert "may still be running" in notes, notes
assert doc["verdict"] == "NO-GO", doc["verdict"]' "$art" \
  || fail q "the notes must carry the leaked container forward:
$(python3 -c "import json,sys;print(json.load(open(sys.argv[1]))[\"notes\"])" "$art")"
ok q "the artifact validates as NO-GO and its notes name the container that may be leaked"
