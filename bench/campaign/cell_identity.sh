#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Sourced by run_cell.sh; functions share its per-cell state.
# shellcheck disable=SC2154,SC2034

capture_model_launch() {
  [ "$DRY_RUN" = "1" ] && return 0
  local cid="" label="" line=""
  if [ "$ENGINE" = "atlas" ]; then
    # Only a completed create records a container ID. Intent recovery remains
    # the teardown's job; no model identity is inferred from a container name.
    [ -n "${IMAGE:-}" ] && [ -f "$NODE_RUN_DIR/rank0.container" ] || return 0
    while IFS= read -r line; do
      case "$line" in id=*) cid="${line#id=}" ;; label=*) label="${line#label=}" ;; esac
    done < "$NODE_RUN_DIR/rank0.container"
  else
    [ -f "$CONTAINER_ID_FILE" ] || return 0
    IFS= read -r cid < "$CONTAINER_ID_FILE" || true
    label="$RUN_LABEL"
  fi
  [ -n "$cid" ] && [ -n "$label" ] || return 0
  # Retain the actual argv and runtime identity, excluding Config.Env and
  # unrelated labels that may carry credentials. pipefail preserves a failed
  # Docker query; neither failure nor an old output file becomes proof.
  if docker inspect "$cid" 2>"$OUT/model-launch.err" \
      | python3 "$HERE/model_launch_capture.py" --out "$OUT/model-launch.json" \
          --container-id "$cid" --label "$label" 2>>"$OUT/model-launch.err"; then
    MODEL_LAUNCH_JSON="$OUT/model-launch.json"
    MODEL_LAUNCH_CONTAINER_ID="$cid"
    MODEL_LAUNCH_LABEL="$label"
  else
    add_note "model launch inspect unavailable; model revision remains unproven"
  fi
  return 0
}

# ── engine identity: what served the requests, asked of the engine ──────────
# One `docker image inspect` of the image that ran, or one hash of the binary
# that ran. Run from the finalizer BEFORE teardown, because teardown removes
# the container and deletes the launcher's record of what it was running --
# after that the only thing left to ask about is the image TAG, and a tag is a
# pointer that may have moved. Running it there also means a cell that was
# interrupted still says what it was running. Every lookup is best-effort --
# no daemon, no image, no label all leave the field null, which is the
# schema's "not measured".
docker_label() {  # docker_label REF TEMPLATE -> the value, or nothing
  local out
  out="$("${DOCKER:-docker}" image inspect --format "$2" "$1" 2>/dev/null || true)"
  case "$out" in
    ""|"<no value>"|"<nil>") ;;
    *) printf '%s\n' "$out" ;;
  esac
}

# The image ID rank 0's create resolved its tag to, written by the launcher
# the moment `docker run -d` returned (scripts/start-node-ep.sh,
# record_rank_image). An ID cannot move under a rebuild the way the tag it was
# resolved from can, so this -- not $IMAGE -- is what the identity is read of.
recorded_rank0_image() {
  local f="$NODE_RUN_DIR/rank0.image" line id=""
  [ -f "$f" ] || return 0
  while IFS= read -r line; do
    case "$line" in id=*) id="${line#id=}" ;; esac
  done < "$f"
  printf '%s' "$id"
}

capture_engine_identity() {
  [ "$IDENTITY_CAPTURED" = "1" ] && return 0
  IDENTITY_CAPTURED=1
  [ "$DRY_RUN" = "1" ] && return 0

  local ref="" digest="" rev=""
  if [ "$ENGINE" = "atlas" ] && [ -z "${IMAGE:-}" ]; then
    # The local binary. `spark --version` prints ATLAS_VERSION, which is
    # env!("CARGO_PKG_VERSION") and carries no revision
    # (crates/spark-server/src/cli.rs), so the hash of the file that ran is the
    # only identity there is to record -- and git_sha stays null rather than
    # being filled in with something that describes a different artefact.
    ENGINE_BINARY="${SPARK_BIN:-./target/release/spark}"
    [ -f "$ENGINE_BINARY" ] || ENGINE_BINARY=""
    return 0
  fi

  if [ "$ENGINE" = "atlas" ]; then
    ref="$(recorded_rank0_image)"
    if [ -z "$ref" ]; then
      # No create ever returned (a launch that was refused, or one interrupted
      # inside `docker run -d`), so there is no resolved ID to name. The tag is
      # the only reference left and it is only as good as the moment it is
      # read -- which the artifact says out loud rather than implying.
      ref="$IMAGE"
      add_note "engine identity read from the image tag $IMAGE: the launcher recorded no resolved image ID for rank 0, so a tag re-pointed during this run would not be visible here"
    fi
  else
    # vLLM's identity is its digest, and the digest is PINNED by the operator:
    # vllm_control.sh refuses to run without VLLM_IMAGE_DIGEST and builds the
    # reference as <repo>@<digest>, so what ran is what that names. The
    # version label is read of the SAME reference -- of the container's
    # resolved image where the create recorded one, else of the pinned
    # <repo>@<digest>, and only as a last resort of the floating tag.
    ENGINE_IMAGE_DIGEST="${VLLM_IMAGE_DIGEST:-}"
    ref="$(recorded_container_image)"
    if [ -z "$ref" ]; then
      local repo="${VLLM_IMAGE:-${VLLM_IMAGE_NAME:-}}"
      [ -n "$repo" ] || return 0
      if [ -n "${VLLM_IMAGE_DIGEST:-}" ]; then
        ref="${repo%%@*}@$VLLM_IMAGE_DIGEST"
      else
        ref="$repo"
        add_note "vLLM engine identity read from the image tag $repo: neither a resolved image ID nor a pinned digest was available"
      fi
    fi
    ENGINE_VLLM_VERSION="$(docker_label "$ref" '{{index .Config.Labels "org.opencontainers.image.version"}}')"
    return 0
  fi

  # RepoDigests reads back as <repo>@sha256:...; only the sha256 half is the
  # image's identity, and a tag-only image (never pushed, or built locally) has
  # no digest at all.
  digest="$(docker_label "$ref" '{{index .RepoDigests 0}}')"
  digest="${digest##*@}"
  case "$digest" in
    sha256:*) ENGINE_IMAGE_DIGEST="$digest" ;;
  esac
  # The revision the IMAGE declares. An image built without the label says
  # nothing, and nothing is what gets recorded.
  rev="$(docker_label "$ref" '{{index .Config.Labels "org.opencontainers.image.revision"}}')"
  if printf '%s' "$rev" | grep -Eq '^[0-9a-f]{7,40}$'; then
    ENGINE_GIT_SHA="$rev"
  fi
  return 0
}

# The same record on the vLLM side: vllm_control.sh knows the container ID the
# moment its create returns, and writes down what that container is running.
recorded_container_image() {
  local line id=""
  [ -f "$CONTAINER_IMAGE_FILE" ] || return 0
  while IFS= read -r line; do
    case "$line" in id=*) id="${line#id=}" ;; esac
  done < "$CONTAINER_IMAGE_FILE"
  printf '%s' "$id"
}

# Built at call time, not once: the failing stage and the interruption note are
# only known when the cell is over, however it got there.
build_assemble() {
  local note="$WARMUP_NOTE"
  ASSEMBLE=( python3 "$HERE/cell_assemble.py"
    --engine "$ENGINE" --model-key "$MODEL" --sku "$SKU" --workload "$WORKLOAD"
    --concurrency "$CONC" --spec "$SPEC" --think "$THINK" --out "$ARTIFACT"
    --workloads "$WORKLOADS" --atlas-recipes "$ATLAS_RECIPES"
    --vllm-recipes "$VLLM_RECIPES" --client "$LADDER"
    --serve-argv "$SERVE_ARGV" --serve-env "$SERVE_ENV" --nvidia-smi-q "$SMI_Q"
    --boot-json "$BOOT_JSON" --coherency-json "$COH_JSON" --ladder-json "$LADDER_JSON" )
  if [ -n "$MODEL_LAUNCH_JSON" ]; then
    ASSEMBLE+=( --model-launch-json "$MODEL_LAUNCH_JSON"
      --model-launch-container-id "$MODEL_LAUNCH_CONTAINER_ID" --model-launch-label "$MODEL_LAUNCH_LABEL" )
  fi
  [ -n "$HARNESS_GIT_SHA" ] && ASSEMBLE+=( --harness-git-sha "$HARNESS_GIT_SHA" )
  [ -n "$ENGINE_GIT_SHA" ] && ASSEMBLE+=( --git-sha "$ENGINE_GIT_SHA" )
  [ -n "$ENGINE_IMAGE_DIGEST" ] && ASSEMBLE+=( --image-digest "$ENGINE_IMAGE_DIGEST" )
  [ -n "$ENGINE_BINARY" ] && ASSEMBLE+=( --binary "$ENGINE_BINARY" )
  [ -n "$ENGINE_VLLM_VERSION" ] && ASSEMBLE+=( --vllm-version "$ENGINE_VLLM_VERSION" )
  [ -n "$PAIRED" ] && ASSEMBLE+=( --paired-artifact "$PAIRED" )
  [ -n "$PTX_RECEIPT" ] && ASSEMBLE+=( --ptx-receipt "$PTX_RECEIPT" )
  if [ -n "$EXTRA_NOTE" ]; then note="${note:+$note; }$EXTRA_NOTE"; fi
  [ -n "$note" ] && ASSEMBLE+=( --extra-note "$note" )
  [ -n "$FAILING_STAGE" ] && ASSEMBLE+=( --failing-stage "$FAILING_STAGE" )
  return 0
}
