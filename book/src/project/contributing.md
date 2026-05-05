# Contributing

The canonical references are [`CONTRIBUTING.md`](https://github.com/Avarok-Cybersecurity/atlas/blob/main/CONTRIBUTING.md) and [`AGENTS.md`](https://github.com/Avarok-Cybersecurity/atlas/blob/main/AGENTS.md). This chapter gives a working overview for anyone reading the book first.

## The AI-first policy

Atlas is explicitly an AI-first codebase. From `CONTRIBUTING.md`:

> - **All PRs are expected to be AI-generated.** Use the best AI tools available to write your kernels, Rust code, and benchmarks.
> - **Human-written code must be justified.** Indicate which parts are human-authored and explain why.
> - **Human-only contributions will be reviewed by AI.**

This is not branding — it's the operational consequence of the specialization thesis. If AI can hyperoptimize CUDA kernels for specific hardware targets, it can write the infrastructure too. Ports to new `(H, M_q)` targets are the clearest example: each is a bounded, well-scoped piece of work, and that's the unit AI-assisted engineering handles best.

## What kinds of PRs are welcome

The [README's Contributing](https://github.com/Avarok-Cybersecurity/atlas/blob/main/README.md#contributing) section lists four categories:

- **New `(H, M_q)` targets.** Porting Atlas kernels to new hardware (H100, B200, MI300X, Apple M4, Intel) or new models. Each target is a self-contained body of work. See the [Adding a new hardware target](https://github.com/Avarok-Cybersecurity/atlas/blob/main/README.md#adding-a-new-hardware-target) and [Adding a new model](https://github.com/Avarok-Cybersecurity/atlas/blob/main/README.md#adding-a-new-model) guides.
- **Kernel optimization.** Profile existing kernels, experiment with tiling strategies, register pressure, shared-memory layouts. If you can beat the numbers in the [Benchmarks](../operations/benchmarks.md) chapter, send the PR.
- **Benchmark coverage.** Add shapes and configurations not yet tested. More data points sharpen the hypercompiler.
- **Bug reports.** Include hardware details, repro steps, and kernel timings.

## Local checks before a PR

These are what CI runs (`.github/workflows/ci.yml`). Run them locally first:

```bash
# 1. Formatting
cargo fmt --all -- --check

# 2. Lints (no CUDA required thanks to ATLAS_SKIP_BUILD)
ATLAS_SKIP_BUILD=1 cargo clippy --workspace --tests --all-features -- -Dwarnings

# 3. License headers
bash scripts/check-license-headers.sh

# 4. Typos
typos     # install once: cargo install typos-cli
```

All four are required to pass. Real CUDA build + test cycles require a GB10 host — not the laptop, the DGX Spark itself.

## Ground rules (from AGENTS.md)

- **SPDX header on every source file.** `// SPDX-License-Identifier: AGPL-3.0-only` on line 1 of every `.rs`, `.cu`, `.cuh`, `.h`, `.hpp`, `.cpp`. Enforced by the `license-headers` CI job.
- **License is AGPL-3.0-only.** Don't mix in permissive-only code without confirming compatibility. `deny.toml` controls allowed dependency licenses.
- **Don't regress supported models.** The matrix in [Supported Models](../getting-started/models.md) — 12 models across 7 families — is the contract. If your PR might touch a hot path, validate against `tests/run_all_models.py` on a GB10 before opening.
- **One logical change per commit.** Don't bundle cleanup with a bug fix.
- **Commit message format.** `<area>: <imperative summary>` — e.g. `spark-server: preserve template-forced thinking through EP=2`.

## Failure modes that cost the project time

These are the classes of bug that have burned days. Know them; avoid introducing them.

- **Protocol drift** between OpenAI (`api.rs`) and Anthropic (`anthropic.rs`) surfaces. A fix on one side often needs a matching change on the other.
- **Template mismatches** subtly breaking tool-calling — different `<tool_call>` vs `<minimax:tool_call>` tokens, `<think>` seeded by the template vs emitted by the model, thinking budget enforcement.
- **FP8 / KV / quantization edge cases** — BF16 paged cache routed into an FP8 kernel → silent NaN. If your change touches numeric paths, verify with a real model before claiming success.
- **Docs drift** — CLI flags, release commands, quick-start snippets. Verify against the current binary, not memory.

## The cardinal rule

> **Never assume the model is at fault.** Always look for the Atlas bug first.

The test matrix has caught many issues that would have looked like "model hallucination" in a lesser codebase. The heuristic is: if the model used to produce coherent output on this input and now doesn't, there's an Atlas bug, not a model bug.

## The CLA

By contributing, you agree to the [Contributor License Agreement](https://github.com/Avarok-Cybersecurity/atlas/blob/main/CLA.md). Your work goes out under AGPL-3.0 in the Community Edition, and you grant Avarok the right to relicense for the Enterprise Edition.

The `CLA Assistant` bot automatically comments on every PR. You must explicitly acknowledge and sign before merge.

## Adding a new hardware target

High-level (full walkthrough in the repo README):

1. `kernels/<hw>/HARDWARE.toml` with `vendor = "..."`.
2. `impl ComputeTarget` in `atlas-core/src/compute.rs` (or inline in your crate).
3. Arm in `atlas-kernels/build.rs::resolve_compute_target()`.
4. `impl GpuBackend` in `spark-runtime/src/<vendor>_backend.rs` — 27 methods, some optional.
5. Kernel source files for ~35 kernels.
6. `MODEL.toml` + `KERNEL.toml` for at least one model.
7. Backend selection branch in `spark-server/src/main.rs`.
8. Dockerfile for the new hardware.

## Adding a new model

The model-specific surface is tiny:

1. `crates/spark-model/src/weight_loader/<your_model>.rs` implementing `ModelWeightLoader` (~200–500 lines depending on architecture complexity).
2. Module declaration + `pub use` in `crates/spark-model/src/weight_loader/mod.rs`.
3. One match arm in `crates/spark-model/src/factory.rs::loader_for_config`.
4. Optional: `kernels/<hw>/<your-model>/MODEL.toml` for sampling / behavior defaults.
5. Optional: tool-call parser in `crates/spark-server/src/tool_parser.rs`.
6. Entry in `tests/run_all_models.py` for regression coverage.
7. Entry in [Supported Models](../getting-started/models.md).

Existing loaders for patterns: `qwen35.rs`, `minimax.rs`, `nemotron.rs` cover dense, SSM+MoE hybrid, and attention+MoE shapes respectively.

## PR process

1. Fork and create a feature branch.
2. Atomic commits. Enforced by reviewers; squash only at the reviewer's request.
3. CI must pass (`fmt`, `clippy`, `license-headers`, `typos`, `cargo-deny`).
4. PR template asks for:
   - **What** — summary of the change.
   - **Why** — motivation and context.
   - **Benchmarks** — before/after numbers for perf-related changes.
   - **Authorship** — AI / human / mixed; justify human-written sections.
5. Sign the CLA when the bot asks.
6. A maintainer (and/or AI reviewer) merges.

## Scope escalation

If a task is ambiguous, ask in the issue/PR before implementing. If scope grows past "one PR", split it. If you're modifying a shared trait, a build script, or CI config, flag it in the PR description so reviewers catch it.

## References

- [`CONTRIBUTING.md`](https://github.com/Avarok-Cybersecurity/atlas/blob/main/CONTRIBUTING.md) — canonical.
- [`AGENTS.md`](https://github.com/Avarok-Cybersecurity/atlas/blob/main/AGENTS.md) — practical contributor guide.
- [`CLA.md`](https://github.com/Avarok-Cybersecurity/atlas/blob/main/CLA.md) — the CLA text.
- [`SECURITY.md`](https://github.com/Avarok-Cybersecurity/atlas/blob/main/SECURITY.md) — disclosure (also this book's [Security chapter](./security.md)).
- [`docs/design/`](https://github.com/Avarok-Cybersecurity/atlas/tree/main/docs/design) — authoritative design docs.
