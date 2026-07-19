#!/usr/bin/env bash
# SPDX license-header check — the SAME engine CI runs, locally.
#
# CI (.github/workflows/ci.yml, job `license-headers`) uses the
# apache/skywalking-eyes `header` action against `.licenserc.yaml` (SSOT for the
# checked paths + the required `SPDX-License-Identifier: AGPL-3.0-only` line).
# This wrapper invokes that same engine via its published container image so a
# local pass matches CI exactly — no separate rule to drift.
#
# Usage:
#   bash scripts/check-license-headers.sh          # check (fails on missing headers)
#   bash scripts/check-license-headers.sh fix      # insert missing headers in place
set -euo pipefail

MODE="${1:-check}"          # check | fix
IMAGE="ghcr.io/apache/skywalking-eyes/license-eye"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if ! command -v docker >/dev/null 2>&1; then
  echo "error: docker is required to run the skywalking-eyes license engine." >&2
  echo "       Install docker, or run the check in CI. The rule (SSOT): every" >&2
  echo "       crates/**/*.rs and kernels|cuda_kernels/**/*.{cu,cuh,h,hpp,cpp}" >&2
  echo "       must start with 'SPDX-License-Identifier: AGPL-3.0-only'." >&2
  exit 2
fi

exec docker run --rm -v "${REPO_ROOT}:/repo" -w /repo \
  "${IMAGE}" -c .licenserc.yaml header "${MODE}"
