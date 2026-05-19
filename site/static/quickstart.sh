#!/bin/sh
# =============================================================================
# Atlas quickstart — https://atlasinference.io/quickstart.sh
# -----------------------------------------------------------------------------
# Installs sparkrun (only if it is not already present) and runs the default
# Atlas recipe. Intended to be piped from curl:
#
#   curl -fsSL https://atlasinference.io/quickstart.sh | sh
#
# It does NOT install Docker/Podman or the NVIDIA container runtime — sparkrun
# uses whatever container engine you already have (the recipe declares its
# `container:`). Re-running is safe: if sparkrun is already installed it is
# left untouched.
# =============================================================================
set -eu

# The recipe to launch. Keep in lockstep with the site copy + atlas-recipes
# SSOT (https://github.com/Avarok-Cybersecurity/atlas-recipes).
RECIPE="qwen3.6-35b-a3b-fp8-mtp-atlas"

log() { printf '\033[1;36m[atlas]\033[0m %s\n' "$1" >&2; }
err() { printf '\033[1;31m[atlas]\033[0m %s\n' "$1" >&2; }

if command -v sparkrun >/dev/null 2>&1; then
  log "sparkrun already installed ($(command -v sparkrun)) — skipping install."
else
  log "sparkrun not found — installing via uvx ..."
  if ! command -v uvx >/dev/null 2>&1; then
    err "uvx is required to install sparkrun but was not found on PATH."
    err "Install uv first: https://docs.astral.sh/uv/  (then re-run this script)."
    exit 1
  fi
  uvx sparkrun setup install
fi

log "Running @atlas/${RECIPE} ..."
exec sparkrun run "@atlas/${RECIPE}"
