# Contributing to Atlas

WE thank you for your interest in Atlas! This document explains how to contribute effectively.

## Philosophy

Atlas follows the **AI Kernel HyperCompiling** philosophy: for every `(Hardware, Model_q)` tuple, there exists a set of kernels producing the highest performance such that it performs at the hardware's theoretical peak. Contributions should align with this — we value specialization over generalization.

### AI-First Codebase

Atlas is an **AI-first codebase**. We're reversing the conventional logic:

- **All PRs are expected to be AI-generated.** Use the best AI tools available to write your kernels, Rust code, and benchmarks.
- **Human-written code must be justified.** If you submit code written without AI assistance, you must explicitly denote which parts are human-authored and explain why a human wrote it better than an AI could.
- **Human-only contributions will be reviewed by AI.** We will subject human-written code to scrutiny by higher-intelligence AI systems to verify that the human approach is genuinely superior.

This is not a gimmick — it's the logical extension of our philosophy. If AI can hyperoptimize CUDA kernels for specific hardware targets, it can write the infrastructure too. Prove us wrong, and we'll happily merge your PR.

## Getting Started

### Prerequisites

- CUDA 13.0+ with `nvcc`
- Rust stable (see `rust-toolchain.toml`)
- NVIDIA GB10 GPU (for kernel testing — unit tests run without GPU)

### Build & Test

```bash
export CUDA_HOME=/usr/local/cuda-13.0

# Build (requires CUDA + nvcc)
cargo build --release
```

#### Testing without a GPU

Most contribution paths can be developed and validated without GPU
hardware. CI runs all of these on a standard `ubuntu-latest` runner:

```bash
# Rust correctness — no GPU, no nvcc required.
# scripts/check.sh sets ATLAS_SKIP_BUILD=1 + CUDARC_CUDA_VERSION
# so cudarc skips its driver probe and atlas-kernels emits a stub.
./scripts/check.sh                  # cargo check, ~30s incremental
./scripts/check.sh clippy --tests   # cargo clippy
./scripts/check.sh test             # cargo test (unit tests only)
cargo fmt --all -- --check          # formatting

# File-size cap (≤500 LoC per .rs file in crates/) — local check
find crates -name '*.rs' -not -name '*.bak' -not -path '*/target/*' \
  | xargs wc -l | awk '$1 > 500 && $2 != "total"'
```

Tests that need a GPU are gated behind `--ignored` and require GB10
hardware plus cached HuggingFace model weights:

```bash
# Integration tests — needs GB10 + ~30 GB+ model cache.
cargo test -p spark-server --release -- --ignored

# End-to-end multi-model sweep (the canonical regression suite).
# Requires Docker, NVIDIA Container Toolkit, optionally a 2-node
# DGX Spark cluster. Defaults to localhost for single-node runs.
python3 tests/run_all_models.py

# Microbenchmarks (single GPU).
cargo bench -p atlas-spark-bench
```

CI enforces (all GPU-free): `fmt`, `clippy`, `cargo test --workspace`
(unit tests + non-`#[ignore]` integration tests), license-headers,
typo check, `cargo-deny`, file-size cap (≤500 LoC per `crates/**/*.rs`).
PRs fail without an authoring maintainer needing GPU access — the kernel
work happens locally.

**CI-green is not the same as shippable.** The GPU-free CI proves the code
compiles and is hygienic; it does *not* boot a model. An image is only
"verified" once it passes the **serve matrix** (`tests/run_all_models.py` +
the coherence gate). Deploying on GB10? Read
[`docs/GB10_DEPLOYMENT_GUIDE.md`](docs/GB10_DEPLOYMENT_GUIDE.md) for the model
compatibility matrix, quant selection, and known-issue workarounds. Cutting an
image? The build → verify → publish pipeline is the `atlas-release` skill
(`.claude/skills/atlas-release/`).

### Code Formatting

```bash
# Rust
cargo fmt --all

# CUDA kernels
find cuda_kernels/ -name '*.cu' -print0 | xargs -0 clang-format -i
```

## What to Contribute

### New (H, M<sub>q</sub>) Targets

Each hardware × model × quantization combination is a self-contained body of work. To add a new target:

1. Add kernel variants in `cuda_kernels/` optimized for the target SM architecture
2. Register them in the appropriate `crates/atlas-*` kernel crate
3. Add benchmark shapes to `crates/atlas-spark-bench/`
4. Demonstrate speedup over the baseline (PyTorch, cuBLAS, etc.)

### Kernel Optimization

Profile existing kernels and submit improvements. Every PR should include:

- **Before/after timings** on the target hardware
- **What changed** — tiling strategy, register pressure, shared memory layout, etc.
- **Why it's faster** — brief explanation of the optimization

### Benchmark Coverage

Add new shapes and configurations to `crates/atlas-spark-bench/`. More data points help us find optimization opportunities.

### Bug Reports

Open an issue with:

- Hardware details (GPU model, driver version, CUDA version)
- Reproduction steps
- Expected vs actual behavior
- Kernel timings if applicable

## Code Standards

- **Rust** — `cargo fmt` and `cargo clippy -- -D warnings` must pass
- **CUDA** — `clang-format` with the repo's `.clang-format` config
- **No Python** — Atlas is pure Rust + CUDA. The Python benchmarks in `historical-python/` are archived for reference only.
- **Tests** — Add unit tests for new functionality. Use `MockGpuBackend` for tests that don't need a real GPU.

## Pull Request Process

1. Fork the repo and create a feature branch
2. Make your changes with clear, atomic commits
3. Ensure CI passes: `cargo fmt`, `cargo clippy`, `cargo test`, CUDA format + lint
4. Open a PR with:
   - **What** — Summary of the change
   - **Why** — Motivation and context
   - **Benchmarks** — Before/after numbers for performance-related changes
   - **Authorship** — Indicate whether the PR was AI-generated, human-written, or a mix. Human-written sections must include justification for why AI was not used.
5. **Sign the Contributor License Agreement (CLA)**. Our automated `CLA Assistant` bot will leave a comment on your PR. You must reply to this comment explicitly acknowledging and signing the [CLA](CLA.md) before we can merge your changes.
6. A maintainer (and/or AI reviewer) will review and merge

## License & CLA

By contributing, you agree that your contributions will be governed by our [Contributor License Agreement (CLA)](CLA.md). Your work will be distributed in the Community Edition under the [AGPLv3 License](LICENSE) and you grant us the right to commercially re-license it for the Enterprise Edition.
