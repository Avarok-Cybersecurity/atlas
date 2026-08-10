#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Measure workspace line/region coverage with cargo-llvm-cov.
#
# SSOT for the coverage invocation: .github/workflows/coverage.yml calls this
# script rather than inlining the command, so a contributor running it locally
# and CI can never disagree about which files are excluded. Same rationale as
# scripts/ci_gpu_stubs.sh.
#
# Usage:
#   scripts/coverage.sh lcov.info                  # write LCOV + print the summary
#   COVERAGE_HTML=1 scripts/coverage.sh lcov.info  # ...also target/llvm-cov/html
#   scripts/coverage.sh lcov.info -p spark-server  # extra args go to the run
#
# HTML is an env var rather than a passed-through flag on purpose: cargo-llvm-cov
# rejects `--html` alongside `--lcov` ("--html may not be used together with
# --lcov"), so it has to be a second `report` pass over the same profdata.
#
# Requires: `cargo install cargo-llvm-cov` + `rustup component add
# llvm-tools-preview` (for the toolchain pinned in rust-toolchain.toml).
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 <lcov-output-path> [extra cargo-llvm-cov args...]" >&2
  exit 2
fi
OUT="$1"
shift

# Same no-GPU env CI's `test` job uses (.github/workflows/ci.yml): skip nvcc in
# build.rs and short-circuit cudarc's driver probe. Exported here too so the
# script behaves identically when a developer runs it on a GB10 box.
export ATLAS_SKIP_BUILD="${ATLAS_SKIP_BUILD:-1}"
export CUDARC_CUDA_VERSION="${CUDARC_CUDA_VERSION:-13000}"
# Belt-and-braces for developers running this on a machine that DOES have a
# GPU: every test that needs one is `#[ignore]`-gated, but hiding the device
# guarantees a stray CUDA init fails fast instead of quietly allocating VRAM
# out from under a benchmark that happens to be running. On CI this is a no-op.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-}"

# Files that CANNOT be meaningfully covered by a CPU-only test run. Reporting
# them would not make the number "honest by omission" — it would make it
# meaningless, because they are 100% unreachable without a GB10, so their zero
# is a property of the runner and not of the test suite.
#
#   vendor/                  vendored cudarc (upstream code, not ours to test)
#   target/                  build-script output, incl. the generated
#                            OUT_DIR/*_ptx.rs that atlas-kernels/spark-storage
#                            `include!`
#   build.rs                 build scripts run at compile time, not under test
#   crates/atlas-kernels/    entirely build.rs-generated PTX constants
#   crates/cufile-sys/       raw dlopen FFI to libcufile (GDS; dormant on GB10)
#   crates/spark-comm/       raw FFI to libnccl; CI links a fail-fast stub
#   atlas-rdma/src/verbs.rs  ibverbs FFI; `cfg(atlas_rdma_verbs)` is compiled
#                            OUT under ATLAS_SKIP_BUILD. The rest of the crate
#                            (wire codecs, railset, handshake) IS covered and
#                            deliberately stays in the report.
#   spark-model/.../ops/     thin `KernelLaunch` wrappers — one CUDA kernel
#                            launch each, zero host logic
#   tests|benches|examples/  test/bench/dev-harness code, not product code.
#                            (Inline `#[cfg(test)] mod tests` is already
#                            excluded by cargo-llvm-cov itself.)
#
# ★ Any addition here needs a rationale on the line above it. An exclusion is
#   how a coverage number quietly becomes a decoration.
IGNORE_REGEX='(^|/)vendor/|(^|/)target/|/build\.rs$|(^|/)crates/(atlas-kernels|cufile-sys|spark-comm)/|(^|/)crates/atlas-rdma/src/verbs\.rs$|(^|/)crates/spark-model/src/layers/ops/|(^|/)crates/[^/]+/(tests|benches|examples)/'

# Run and report are SPLIT on purpose (`clean` -> `--no-report` -> `report`,
# cargo-llvm-cov's documented multi-step flow). The single-shot form generates
# the report only on a zero exit, so one failing test would leave no lcov.info
# at all — and a red test suite is exactly when you want to see which code the
# surviving tests still cover. `clean` first because `--no-report` does not
# discard the previous run's .profraw the way the single-shot form does; without
# it a report would silently mix two runs.
#
# --no-fail-fast: a failing test must not stop the remaining test binaries from
#                 running, or the report covers only part of the tree.
# --locked:       same lockfile discipline as `cargo test --workspace --locked`.
#
# NOTE: `--no-fail-fast` and `--ignore-run-fail` are mutually exclusive in
# cargo-llvm-cov ("--ignore-run-fail may not be used together with
# --no-fail-fast"), so the test exit status is captured and re-raised at the end
# instead. A test failure therefore still fails this script — the report is
# produced first, not skipped. ci.yml's `test` job remains the real gate; this
# is redundancy, not a second gate.
#
# `--profraw-only` rather than a bare `clean --workspace`: the only thing that
# must not survive between runs is the previous run's raw profile data.
# `clean --workspace` ALSO deletes the compiled objects, which costs a full
# instrumented rebuild of ~340k LoC every single invocation — a real tax
# locally, and unnecessary here because cargo-llvm-cov builds into its own
# dedicated target dir, so every object in it is already instrumented.
cargo llvm-cov clean --profraw-only
status=0
cargo llvm-cov --no-report --workspace --locked --no-fail-fast "$@" || status=$?

# Both reports read the same profdata — no rebuild, no re-run.
cargo llvm-cov report --ignore-filename-regex "$IGNORE_REGEX" \
  --lcov --output-path "$OUT"
cargo llvm-cov report --ignore-filename-regex "$IGNORE_REGEX" --summary-only

# Optional third pass, same profdata: a browsable line-by-line report. Off by
# default because CI has nowhere to display it.
if [[ "${COVERAGE_HTML:-0}" == "1" ]]; then
  cargo llvm-cov report --ignore-filename-regex "$IGNORE_REGEX" --html
  echo "HTML report: target/llvm-cov/html/index.html"
fi

if [[ $status -ne 0 ]]; then
  echo "coverage: report written to $OUT, but the test run exited $status" >&2
fi
exit "$status"
