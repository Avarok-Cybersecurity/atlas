# AGENTS.md

A contributor guide for AI agents (and humans) working on Atlas. Read this
alongside [CONTRIBUTING.md](CONTRIBUTING.md). The tone is practical: paths,
commands, and the invariants that matter.

## What Atlas is

Atlas is an AGPL-3.0 inference stack targeting NVIDIA GB10 / DGX Spark. The
moving parts:

- **`crates/spark-server/`** — OpenAI-compatible HTTP server, request
  scheduling, tool-call parsing, streaming.
- **`crates/spark-model/`** — model assembly (attention, MoE, SSM layers),
  weight loaders per model family, model-type dispatch in
  `src/factory.rs`.
- **`crates/spark-runtime/`** — GPU backend, KV cache, kernel dispatch,
  process-group comms.
- **`crates/atlas-kernels/`** — Rust glue over compiled PTX (one artefact
  per `(hw, model, quant)` target).
- **`kernels/<hw>/<model>/<quant>/`** — CUDA kernels + `MODEL.toml`
  (sampling, behaviour defaults, kernel target registration).
- **`crates/atlas-*`** — smaller shared primitives (quant, gemm, ssm, norm,
  attention, reduce, activation, embed).
- **`crates/atlas-spark-bench/`** — benchmark harness.

Design docs live in `docs/design/`; historical notes in `docs/history/`;
release notes in `docs/releases/`.

## Ground rules

- **SPDX header on every source file.** `// SPDX-License-Identifier:
  AGPL-3.0-only` on line 1 of every `.rs`, `.cu`, `.cuh`, `.h`, `.hpp`,
  `.cpp`. Enforced by Github Pipeline.
- **License is AGPL-3.0-only.** Don't mix in permissive-only code without
  confirming compatibility. `deny.toml` controls what dependency licenses
  are allowed.
- **Don't regress on models already in the support matrix.** 13 models
  across Qwen3/Qwen3.5/Qwen3-VL, Nemotron-H, Mistral-Small-4, Gemma-4,
  MiniMax M2/M2.7. See QUICKSTART.md for the current list.

## Local checks before a PR

The commands CI will run:

```bash
# 1. Formatting
cargo fmt --all -- --check

# 2. Lints (the build-script gate lets clippy run without CUDA on the host)
ATLAS_SKIP_BUILD=1 cargo clippy --workspace --tests --all-features -- -Dwarnings

# 3. License headers
bash scripts/check-license-headers.sh

# 4. Typos
typos  # crate-ci/typos — install once, `cargo install typos-cli`
```

A real build + test cycle requires a CUDA-capable host; see
[CONTRIBUTING.md](CONTRIBUTING.md).

## Adding a new model

High-level walkthrough — the patterns to follow are already in-tree.

1. **Model-type dispatch.** Add a new arm in
   `crates/spark-model/src/factory.rs` that returns a new
   `ModelWeightLoader` impl. Use `crates/spark-model/src/weight_loader/`
   for the loader (study `qwen35.rs`, `minimax.rs`, `nemotron_h.rs` for the
   three major shapes: dense, SSM+MoE hybrid, attention+MoE).
2. **Kernel target.** Create `kernels/<hw>/<model-slug>/<quant>/` with a
   `MODEL.toml` declaring the model-type matches, sampling presets, and
   behaviour defaults. The top-level `kernels/<hw>/HARDWARE.toml` picks up
   the new target automatically if you set
   `ATLAS_TARGET_MODEL=*` at build time (default).
3. **Behavioural knobs.** `MODEL.toml` is the SSOT for per-model
   sampling/thinking/tool-use policy. `build.rs` in `atlas-kernels` parses
   it into `SamplingPresets` + `ModelBehavior` consumed by the server.
4. **Jinja template.** If the model uses a chat template that's not
   covered by `jinja-templates/`, add one. Naming convention matches the
   HF repo.

Concrete recent examples worth reading:

- Mistral-Small-4 integration — `crates/spark-model/src/mistral_loader.rs`
  + `kernels/gb10/mistral-small-4/`.
- MiniMax M2/M2.7 (attention + 256-expert sigmoid-routed MoE) —
  `crates/spark-model/src/weight_loader/minimax.rs` +
  `kernels/gb10/minimax-m2-229b/`.
- Gemma-4 (sliding/full attention alternation) —
  `crates/spark-model/src/weight_loader/gemma4.rs` +
  `kernels/gb10/gemma-4-*/`.

## The kernel target system

Three dimensions: **hardware** × **model** × **quantization**. At build
time, `atlas-kernels/build.rs` enumerates the `ATLAS_TARGET_*` env vars
(with `*` meaning "all matching") and produces one PTX artefact per
target. Runtime selects the correct target based on the model's
`model_type` and loaded config.

- `ATLAS_TARGET_HW=gb10` — currently the only implemented hardware.
- `ATLAS_TARGET_MODEL=*` / `ATLAS_TARGET_QUANT=*` — wildcard compiles all.
- `ATLAS_SKIP_BUILD=1` — emits a stub so clippy/fmt can run without nvcc.

## Writing commits

- One logical change per commit. Don't bundle an unrelated cleanup with a
  bug fix.
- Message format: `<area>: <imperative summary>` — e.g.
  `spark-server: preserve template-forced thinking through EP=2`.
- If the change affects runtime behaviour, rebuild the Docker image from
  scratch and run the relevant slice of the validation suite
  (`tests/single_gpu_suite.py` for most cases) before opening the PR.

## Failure modes that cost us time

These aren't abstract — they're the classes of bug that have burned days:

- **Protocol drift** between OpenAI and Anthropic paths (`api.rs`,
  `anthropic.rs`). A fix on one surface often needs a matching change on
  the other.
- **Template mismatches** that break tool-calling subtly — different
  `<tool_call>` vs `<minimax:tool_call>` tokens, `<think>` seeded by the
  template vs emitted by the model, `thinking_budget` enforcement.
- **FP8 / KV / quantization edge cases** — BF16 paged cache routed into an
  FP8 kernel, silent NaN. If your change touches numeric paths, verify
  with a real model before claiming success.
- **Docs drift** — CLI flags, release commands, quick-start snippets.
  Verify against the current binary, not your memory.

When you hit a regression, **never assume the model is at fault** — always
look for the Atlas bug first.

## Scope and escalation

If a task is ambiguous, stop and ask in the issue/PR before implementing.
If the scope grows past "one PR", split it. If you're about to modify
something shared (a cross-cutting trait, a build script, CI config), flag
it in the PR description so reviewers catch it.

See `CONTRIBUTING.md` for coding style and the CLA expectations,
`SECURITY.md` for disclosure, and `docs/design/` for the authoritative
architecture references.
