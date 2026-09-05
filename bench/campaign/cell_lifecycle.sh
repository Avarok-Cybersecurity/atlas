#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Sourced by run_cell.sh; functions share its per-cell state.
# shellcheck disable=SC2154,SC2034

add_note() {
  if [ -n "$EXTRA_NOTE" ]; then EXTRA_NOTE="$EXTRA_NOTE; $1"; else EXTRA_NOTE="$1"; fi
}

# Kill one pid and the pids it started, by pid. `pgrep -P` walks parent links,
# so the kill stays inside this invocation's own process tree; a process-PATTERN
# kill would match this script's own command line, which is how a teardown
# becomes a self-kill.
kill_tree() {
  local pid="$1" kid
  for kid in $(pgrep -P "$pid" 2>/dev/null); do
    kill_tree "$kid"
  done
  kill "$pid" 2>/dev/null || true
}

forget_child() {
  local keep="" p
  for p in $CHILD_PIDS; do
    [ "$p" = "$1" ] || keep="$keep $p"
  done
  CHILD_PIDS="$keep"
}

# Run one stage's child in the background and wait for it. The backgrounding is
# what makes a signal actionable at all: bash defers a trap until the FOREGROUND
# command returns, and `wait` is the builtin a signal interrupts. Straight-line
# `bash time_to_ready.sh ...` would sit on a SIGTERM until the boot poll gave up
# 1800 s later. The pid is recorded so the finalizer can kill exactly this
# child, by pid.
run_child() {
  "$@" &
  local pid=$! rc=0
  CHILD_PIDS="$CHILD_PIDS $pid"
  wait "$pid" || rc=$?
  forget_child "$pid"
  return "$rc"
}

# Did the launcher record a rank of ours? It writes rank<N>.pid (local binary)
# or rank<N>.container (container mode) as it starts each one, into the run
# directory this cell reserved for itself. Because that directory is this
# invocation's alone, anything recorded in it can only be this invocation's --
# which is why THIS, and not "we backgrounded the launcher", is the ownership
# test. A launcher that refuses (exit 2 on an occupied port, a run directory it
# will not overwrite, a container name it did not create) creates nothing and
# records nothing, and a refusal must never be followed by a --stop.
#
# rank<N>.intent counts as much as rank<N>.container, because the container
# record only appears once `docker run -d` has RETURNED. A cell killed inside
# that create finds no container record and used to conclude that nothing was
# created -- while the container was up and stayed up. The intent is written
# before the create, so its presence is the evidence that a rank may exist, and
# the --stop it triggers is the thing that reconciles it. A refusal still
# writes no intent, so a refusal still stops nothing.
atlas_recorded_ranks() {
  local f
  for f in "$NODE_RUN_DIR"/rank*.pid "$NODE_RUN_DIR"/rank*.container \
           "$NODE_RUN_DIR"/rank*.intent; do
    [ -e "$f" ] && return 0
  done
  return 1
}

# Only what this invocation owns: the pids it started, plus the ranks or the
# container it created. Nothing here is addressed by name or by pattern.
teardown_owned() {
  local pid
  for pid in $CHILD_PIDS; do
    if kill -0 "$pid" 2>/dev/null; then
      echo "killing pid $pid, started by this cell"
      kill_tree "$pid"
    fi
  done
  CHILD_PIDS=""

  if [ "$ENGINE" = "atlas" ]; then
    show "env ATLAS_NODE_RUN_DIR=$NODE_RUN_DIR bash $LAUNCHER --stop"
    if [ "$DRY_RUN" = "1" ]; then return 0; fi
    if [ "$NODE_RUN_DIR_RESERVED" != "1" ]; then
      echo "this invocation reserved no run directory of its own: nothing to stop."
      return 0
    fi
    if ! atlas_recorded_ranks; then
      echo "the launcher recorded no rank or create intent in $NODE_RUN_DIR -- it"
      echo "  refused or failed before creating anything, so this cell owns nothing"
      echo "  to stop."
      return 0
    fi
    # A --stop that could not account for a rank exits non-zero and KEEPS that
    # rank's intent record, so the cell reports the leak rather than closing
    # its teardown stage over it.
    if ! env "ATLAS_NODE_RUN_DIR=$NODE_RUN_DIR" bash "$LAUNCHER" --stop; then
      note_fail teardown
      add_note "the launcher could not account for every rank it recorded in $NODE_RUN_DIR: a rank this cell started may still be running and holding a GPU, and its intent record is kept there for a later --stop"
    fi
    return 0
  fi

  # By the container ID this invocation's `docker run -d` returned, and never
  # by `pkill -f`: a `pkill -f vllm` pattern matches this script's own command
  # line, which is how a teardown becomes a self-kill.
  #
  # The ID, not the name, because this block runs even when the serve stage
  # failed. If it failed with 125 the name belongs to a container somebody else
  # is using, and `docker stop <name>` / `docker rm <name>` then deletes their
  # live server -- exactly the sequence a stub Docker recorded: run, stop
  # <same-name>, rm <same-name>. No ID means nothing was created here, and
  # nothing is this cell's to remove.
  if [ "$DRY_RUN" = "1" ]; then
    show "docker stop <id from docker run -d> && docker rm <the same id>"
    return 0
  fi

  # Three places the ID can be, in the order they become true:
  #   1. this shell, once the serve stage read it back;
  #   2. the id file, which vllm_control.sh writes the moment `docker run -d`
  #      returns -- a signal between the write and the read lands here;
  #   3. Docker itself, asked for whatever wears this invocation's label. That
  #      is the interrupted CREATE: the container exists, its ID was never
  #      printed, and the only thing that survives the window is the label
  #      chosen and recorded before the create. It identifies this invocation
  #      alone (cell, epoch, pid), so what comes back is this cell's own.
  local ids="$CONTAINER_ID" cid rc=0 out=""
  if [ -z "$ids" ] && [ -s "$CONTAINER_ID_FILE" ]; then
    ids="$(cat "$CONTAINER_ID_FILE")"
    echo "the id file holds $ids: the create finished, this shell never read it."
  fi
  if [ -z "$ids" ] && [ "$CREATE_ATTEMPTED" = "1" ]; then
    echo "no container ID reached this shell, and a create was attempted:"
    echo "  asking Docker for anything labelled $RUN_LABEL."
    # The lookup's exit status, read apart from its output. `|| true` used to
    # collapse a daemon that could not be reached into a daemon that answered
    # "nothing", and the branch below then reported a confirmed absence for a
    # container that was up. An unsuccessful lookup is neither presence nor
    # absence: nothing is stopped on a guess, and the cell records that a
    # container of its own may still be holding the GPU.
    out="$("${DOCKER:-docker}" ps -aq --filter "label=$RUN_LABEL" 2>&1)" || rc=$?
    if [ "$rc" -ne 0 ]; then
      echo "  the lookup FAILED (docker ps exited $rc): ${out:-(no output)}"
      echo "  Whether this cell's container exists is UNKNOWN, so nothing is stopped"
      echo "  by guesswork -- and it is not reported as absent either."
      note_fail teardown
      add_note "teardown could not reach Docker (docker ps exited $rc) to find the container labelled $RUN_LABEL: a container this cell created may still be running and holding the GPU"
      return 0
    fi
    ids="$out"
  fi

  if [ -z "$ids" ]; then
    echo "no container was created by this invocation ($CONTAINER_ID_FILE is absent"
    echo "  and nothing wears its label): nothing to stop or remove. A container"
    echo "  already holding the name '$CONTAINER' is not this cell's to delete."
    return 0
  fi
  for cid in $ids; do
    show "docker stop $cid && docker rm $cid"
    # A removal that failed is reported, for the same reason: the cell's own
    # artifact is the only place the leak is written down.
    if ! "${DOCKER:-docker}" stop "$cid" >/dev/null 2>&1; then
      echo "  docker stop $cid FAILED: the container may still be running."
      note_fail teardown
      add_note "teardown could not stop the container $cid this cell created: it may still be running and holding the GPU"
      continue
    fi
    if ! "${DOCKER:-docker}" rm "$cid" >/dev/null 2>&1; then
      echo "  docker rm $cid FAILED: the container is stopped but not removed."
      note_fail teardown
      add_note "teardown stopped the container $cid this cell created but could not remove it"
    fi
  done
}

