#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Sourced by campaign_test.sh; uses its temporary fixtures and assertions.
# shellcheck disable=SC2154,SC2034

# ── (j) teardown only touches the container this invocation created ─────────
# The stubs are PATH-shimmed rather than passed through $DOCKER so that a
# hardcoded `docker` anywhere in the chain is caught too. nvidia-smi is stubbed
# because otherwise preflight fails first and the serve stage never runs.
mkdir -p "$tmp/bin"
cat > "$tmp/bin/docker" <<'SH'
#!/usr/bin/env bash
# A Docker whose bookkeeping is a file. A DETACHED `run` puts the container
# into the running set BEFORE it answers, because that is the order the real
# one works in -- `docker run -d` creates the container and then prints its ID
# -- and DOCKER_RUN_BLOCK_S widens the gap between the two until a signal fits
# inside it. A foreground `docker run --rm` (the launcher's kernel check)
# outlives nothing and is recorded nowhere. `ps -aq --filter` answers from that
# set, honouring BOTH a `label=` filter and an exact `name=^...$` one, and
# stop/rm remove from it, so a test can ask the only question that matters
# after an interruption: is the container this cell created still there?
printf '%s\n' "$*" >> "$DOCKER_CALLS"
running="${DOCKER_RUNNING:-}"
# `docker image inspect` and `docker inspect` answer the same questions here;
# the sub-verb is dropped so one branch serves both.
if [ "${1:-}" = "image" ] && [ "${2:-}" = "inspect" ]; then shift; fi
# What the mutable TAG resolves to right now. A tag is a pointer: re-pointing
# this file mid-run is a rebuild under the same tag, which is what makes
# inspecting the tag after a run the wrong question.
tag_image() { [ -s "${DOCKER_TAG_IMAGE:-}" ] 2>/dev/null && cat "$DOCKER_TAG_IMAGE"; }
# "<image-id-or-tag> <digest> <revision>" per line: what each image says about
# ITSELF, so image A's answers stay distinguishable from image B's.
registry() {  # registry TARGET FIELD -> the field, or nothing
  { [ -n "${DOCKER_REGISTRY:-}" ] && [ -f "$DOCKER_REGISTRY" ]; } || return 1
  awk -v t="$1" -v f="$2" '$1 == t { print $f; found = 1 } END { exit !found }' \
    "$DOCKER_REGISTRY"
}
case "${1:-}" in
  run)
    if [ "${DOCKER_RUN_RC:-0}" != "0" ]; then
      echo "docker: Error response from daemon: Conflict. The container name is already in use." >&2
      exit "${DOCKER_RUN_RC}"
    fi
    label=""; name=""; detached=0; prev=""
    for a in "$@"; do
      case "$prev" in
        --label) label="$a" ;;
        --name) name="$a" ;;
      esac
      if [ "$a" = "-d" ]; then detached=1; fi
      prev="$a"
    done
    if [ "$detached" = "1" ]; then
      if [ -n "$running" ]; then
        printf '%s %s %s\n' "${DOCKER_FAKE_CID:?}" "$label" "$name" >> "$running"
      fi
      # The image the create RESOLVED the tag to, recorded at create time the
      # way the daemon does. `inspect {{.Image}}` answers from here.
      if [ -n "${DOCKER_CREATED_IMAGES:-}" ]; then
        printf '%s %s %s\n' "${DOCKER_FAKE_CID:?}" "$name" "$(tag_image)" \
          >> "$DOCKER_CREATED_IMAGES"
      fi
      if [ -n "${DOCKER_RUN_BLOCK_S:-}" ]; then sleep "$DOCKER_RUN_BLOCK_S"; fi
    fi
    echo "${DOCKER_FAKE_CID:?}"
    ;;
  ps)
    # A lookup that FAILED is not a lookup that found nothing. The marker file
    # named by DOCKER_PS_FAIL_ONCE is CONSUMED by the first `ps`, so exactly
    # one call fails the way a daemon hiccup does.
    if [ -n "${DOCKER_PS_FAIL_ONCE:-}" ] && [ -f "$DOCKER_PS_FAIL_ONCE" ]; then
      rm -f "$DOCKER_PS_FAIL_ONCE"
      echo "docker: Cannot connect to the Docker daemon at unix:///var/run/docker.sock." >&2
      exit 1
    fi
    want_label=""; want_name=""; prev=""
    for a in "$@"; do
      if [ "$prev" = "--filter" ]; then
        case "$a" in
          label=*) want_label="${a#label=}" ;;
          name=*) want_name="${a#name=}"; want_name="${want_name#^}"; want_name="${want_name%\$}" ;;
        esac
      fi
      prev="$a"
    done
    { [ -n "$running" ] && [ -f "$running" ]; } || exit 0
    while read -r cid lab nm; do
      if [ -n "$want_label" ] && [ "$lab" != "$want_label" ]; then continue; fi
      if [ -n "$want_name" ] && [ "$nm" != "$want_name" ]; then continue; fi
      echo "$cid"
    done < "$running"
    ;;
  inspect)
    # Three questions share this verb: does a container name exist (the
    # launcher's pre-create probe, where "yes" is a REFUSAL), what digest the
    # image that ran has, and what revision or version that image declares.
    # The go template says which one is being asked.
    fmt=""; target=""; prev=""
    for a in "$@"; do
      if [ "$prev" = "--format" ]; then fmt="$a"; fi
      prev="$a"; target="$a"
    done
    case "$fmt" in
      *RepoDigests*)
        digest="$(registry "$target" 2)" || digest="${DOCKER_FAKE_DIGEST:-}"
        [ -n "$digest" ] || exit 1
        echo "$target@$digest"
        ;;
      *image.revision*)
        registry "$target" 3 || echo "${DOCKER_FAKE_REVISION:-<no value>}"
        ;;
      *image.version*) echo "${DOCKER_FAKE_IMAGE_VERSION:-<no value>}" ;;
      # The reference the create was GIVEN, and the image it resolved to. The
      # second is the immutable one, and the only honest answer to "what ran".
      *Config.Image*) echo "${DOCKER_FAKE_IMAGE_REF:-$target}" ;;
      *.Image*)
        { [ -n "${DOCKER_CREATED_IMAGES:-}" ] && [ -f "$DOCKER_CREATED_IMAGES" ]; } || exit 1
        awk -v t="$target" '$1 == t || $2 == t { print $3; found = 1 } END { exit !found }' \
          "$DOCKER_CREATED_IMAGES"
        ;;
      *)
        { [ -n "$running" ] && [ -f "$running" ]; } || exit 1
        while read -r cid lab nm; do
          if [ "$nm" = "$target" ]; then echo "true"; exit 0; fi
        done < "$running"
        exit 1
        ;;
    esac
    ;;
  stop|rm)
    { [ -n "$running" ] && [ -f "$running" ]; } || exit 0
    kept=""
    while read -r cid lab nm; do
      if [ "$cid" != "${2:-}" ] && [ "$nm" != "${2:-}" ]; then kept="$kept$cid $lab $nm
"; fi
    done < "$running"
    printf '%s' "$kept" > "$running"
    ;;
esac
exit 0
SH
cat > "$tmp/bin/nvidia-smi" <<'SH'
#!/usr/bin/env bash
cat <<'Q'
==============NVSMI LOG==============
Driver Version                            : 999.00
CUDA Version                              : 99.9
Attached GPUs                             : 1
GPU 00000000:01:00.0
    Product Name                          : Stub Hopper
Q
SH
chmod +x "$tmp/bin/docker" "$tmp/bin/nvidia-smi"
DIGEST="sha256:$(printf 'a%.0s' $(seq 64))"
CID="0123456789abcdef0123456789abcdef0123456789abcdef0123456789abcdef"
# The stub's running set: what `docker run` created and stop/rm have not yet
# taken away. A cleanup that misses is visible here as a leftover line.
export DOCKER_RUNNING="$tmp/docker.running"; : > "$DOCKER_RUNNING"
# "<cid> <name> <image-id>": what each create resolved its tag to, at the
# moment it created the container.
export DOCKER_CREATED_IMAGES="$tmp/docker.created-images"; : > "$DOCKER_CREATED_IMAGES"

# (j1) vllm_control.sh hands the container ID back, or reports that it made none.
export DOCKER_CALLS="$tmp/vc.calls"; : > "$DOCKER_CALLS"
# The image this create resolves. The reference vllm_control launches is
# <repo>@<digest> and so cannot drift, but the container's own resolved ID is
# what stays true once the container is gone -- so it is recorded too.
VC_IMAGE="sha256:$(printf 'f%.0s' $(seq 64))"
printf '%s\n' "$VC_IMAGE" > "$tmp/docker.tag-image"
out="$(DOCKER_FAKE_CID="$CID" PATH="$tmp/bin:$PATH" VLLM_IMAGE_DIGEST="$DIGEST" \
        DOCKER_TAG_IMAGE="$tmp/docker.tag-image" \
        VLLM_CONTAINER=atlas-campaign-selftest-1 \
        bash "$VC" nemotron-3-nano-fp8 h100 --spec off \
        --label atlas-campaign.run=selftest-1 --id-file "$tmp/vc.id" \
        --image-file "$tmp/vc.image" 2>&1)"; rc=$?
[ $rc -eq 0 ] || fail j "vllm_control with a stub docker exited $rc:
$out"
[ "$(tail -1 <<<"$out")" = "container_id: $CID" ] \
  || fail j "the container ID must be the last line: $(tail -3 <<<"$out")"
[ "$(cat "$tmp/vc.id")" = "$CID" ] || fail j "--id-file must hold the container ID"
have "$(cat "$DOCKER_CALLS")" "--label atlas-campaign.run=selftest-1" \
  || fail j "the ownership label must reach docker run: $(cat "$DOCKER_CALLS")"
ok j "vllm_control returns the created container ID and labels it as this run's"

have "$(cat "$tmp/vc.image" 2>&1)" "id=$VC_IMAGE" \
  || fail j "--image-file must hold the image the created container resolved, got:
$(cat "$tmp/vc.image" 2>&1)"
rm -f "$tmp/docker.tag-image"
ok j "--image-file records what the created container is running, not the tag"

: > "$DOCKER_CALLS"; rm -f "$tmp/vc.id"
out="$(DOCKER_FAKE_CID="$CID" DOCKER_RUN_RC=125 PATH="$tmp/bin:$PATH" \
        VLLM_IMAGE_DIGEST="$DIGEST" VLLM_CONTAINER=atlas-campaign-selftest-2 \
        bash "$VC" nemotron-3-nano-fp8 h100 --spec off --id-file "$tmp/vc.id" 2>&1)"; rc=$?
[ $rc -eq 125 ] || fail j "a name conflict must propagate exit 125, got $rc:
$out"
[ -e "$tmp/vc.id" ] && fail j "a failed docker run must write no container ID"
ok j "a docker run name conflict exits 125 and records no container ID"

# (j2) the cell: a name conflict stops and removes nothing.
: > "$DOCKER_CALLS"
out="$(DOCKER_FAKE_CID="$CID" DOCKER_RUN_RC=125 PATH="$tmp/bin:$PATH" \
        VLLM_IMAGE_DIGEST="$DIGEST" \
        bash "$RUN" --engine vllm --model nemotron-3-nano-fp8 --sku h100 \
        --workload lat --concurrency 1 --spec off --think off \
        --out "$tmp/collide" --yes 2>&1)"; rc=$?
[ $rc -eq 1 ] || fail j "a cell whose serve stage failed must exit 1, got $rc:
$out"
touched="$(grep -E '^(stop|rm) ' "$DOCKER_CALLS" || true)"
[ -z "$touched" ] || fail j "a name conflict must stop and remove NOTHING, saw:
$touched"
ok j "docker run exit 125 -> no stop, no rm, nothing of another run is touched"

stage="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["failing_stage"])' \
          "$tmp/collide/artifact.json")"
[ "$stage" = "serve" ] || fail j "a name conflict must fail the serve stage, got $stage"
python3 "$HERE/validate_artifact.py" "$tmp/collide/artifact.json" >/dev/null \
  || fail j "the collision artifact must still validate:
$(python3 "$HERE/validate_artifact.py" "$tmp/collide/artifact.json")"
ok j "a name conflict is recorded as a serve-stage failure in a valid artifact"

names="$(grep '^run ' "$DOCKER_CALLS" | grep -o -- '--name [^ ]*' | sort -u)"
have "$names" "atlas-campaign-" || fail j "the container name must be campaign-scoped: $names"
: > "$DOCKER_CALLS"
DOCKER_FAKE_CID="$CID" DOCKER_RUN_RC=125 PATH="$tmp/bin:$PATH" \
  VLLM_IMAGE_DIGEST="$DIGEST" \
  bash "$RUN" --engine vllm --model nemotron-3-nano-fp8 --sku h100 \
  --workload lat --concurrency 1 --spec off --think off \
  --out "$tmp/collide2" --yes >/dev/null 2>&1
names2="$(grep '^run ' "$DOCKER_CALLS" | grep -o -- '--name [^ ]*' | sort -u)"
[ "$names" != "$names2" ] || fail j "two invocations must not share a container name: $names"
ok j "each invocation names its container uniquely ($names vs $names2)"

