# Security Policy

Canonical: [`SECURITY.md`](https://github.com/Avarok-Cybersecurity/atlas/blob/main/SECURITY.md). This chapter summarises the policy and the threat model.

## Reporting a vulnerability

**Do not open a public issue for security vulnerabilities.**

Email **security@avarok.net** with:

1. **Description** — what the vulnerability is and its potential impact.
2. **Reproduction steps** — minimal.
3. **Environment** — OS, CUDA version, GPU model, Rust version.
4. **Affected component** — which crate or kernel.

Receipt acknowledged within **48 hours**. Initial assessment within **7 days**.

## Supported versions

| Version | Supported |
|---|---|
| latest `main` | ✅ |
| older commits | ❌ |

Atlas moves fast; there are no LTS branches. Run from `main` or a recently-tagged release.

## Threat model

Atlas is an inference server that runs locally with GPU access. The primary surfaces:

### 1. CUDA kernel safety

- **Out-of-bounds reads/writes** in kernels.
- **Integer overflow** in kernel grid/block parameter computation.
- **Buffer overflows** in shared-memory layouts.

Automated: `cppcheck` static analysis on CUDA source. Human: kernel reviews require the PR author to document tile shapes and memory accesses.

### 2. HTTP API input

- **Malformed JSON** — axum + serde handles schema validation; unknown fields are rejected by default.
- **Oversized request bodies** — `ATLAS_MAX_BODY_BYTES` (default 8 MiB) caps inbound body size.
- **Prompt injection** via the chat template — the model is the primary defense; Atlas does not attempt content-level filtering.
- **Rate-limit exhaustion** — per-key token bucket with a `MAX_KEYS` DoS guard against cardinality explosion.

### 3. Weight loading

- **Malicious safetensor files** — the `safetensors` crate handles format parsing; Atlas validates shapes against `ModelConfig` before any GPU upload.
- **Path traversal** during model load — paths are resolved through `PathBuf::canonicalize` and checked against the configured cache root.
- **Disk exhaustion** — model downloads from HF can be many GB; operators should size the `HF_HUB_CACHE` volume accordingly.

### 4. Unsafe Rust

Atlas uses `unsafe` blocks for:

- **CUDA FFI** via `cudarc` (driver calls, raw pointer arithmetic).
- **NCCL FFI** via the vendored `nccl_sys` bindings.
- **`MaybeUninit` scratch buffers** in a handful of hot paths.

Every `unsafe` block is annotated with the safety invariant it relies on. Reviewers treat `unsafe` introductions as high-priority and typically block the PR until the invariant is written down.

### 5. Dependency supply chain

- **`cargo deny`** audits dependencies for known advisories (RustSec), license compliance (AGPL-compatible only), and banned crates. Runs on every PR and weekly via cron.
- **`deny.toml`** controls allow/deny lists. Permissive licenses (MIT, Apache-2.0, BSD-3) are allowed; GPL variants incompatible with AGPL-3.0 are denied.

## Automated security in CI

| Check | Frequency | File |
|---|---|---|
| `cargo-deny` advisories | every PR + weekly | `.github/workflows/security.yml` |
| `cppcheck` CUDA static analysis | every PR touching kernels | same |
| SPDX license header check | every PR | `.github/workflows/ci.yml` |
| `cargo clippy -D correctness -D suspicious` | every PR | `.github/workflows/ci.yml` |

The `-D correctness -D suspicious` gate is deliberate: stylistic `clippy` lints churn across toolchain releases and are not worth blocking PRs, but the correctness + suspicious categories map to real bugs and always block.

## Disclosure policy

Coordinated disclosure. On a valid report:

1. Fix lands in `main`.
2. New tagged release.
3. Credit to the reporter unless anonymity is requested.

## Out of scope

Some things are *not* a security concern under this policy — they're bugs, but not security bugs:

- **Slow kernels.** Performance regressions go through the normal PR/bench workflow.
- **Model hallucinations.** The model is not Atlas. `SECURITY.md` does not cover what the model chooses to say.
- **Operator misconfiguration.** `--gpu-memory-utilization 1.0` will OOM; that's not a vulnerability.

## If you found something

Email security@avarok.net. Include what you need, keep the repro minimal, and do not exploit the vulnerability against production deployments you do not own. The team has fixed every credibly-reported issue within the 7-day initial-assessment window; known-good practice gets a prompt response.
