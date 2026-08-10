# Security Policy

## Supported Versions

| Version | Supported |
|---|---|
| latest `main` | ✅ |
| older commits | ❌ |

## Reporting a Vulnerability

**Do not open a public issue for security vulnerabilities.**

Instead, please report vulnerabilities privately by emailing **security@avarok.net** with:

1. **Description** — What the vulnerability is and its potential impact
2. **Reproduction steps** — Minimal steps to reproduce the issue
3. **Environment** — OS, CUDA version, GPU model, Rust version
4. **Affected component** — Which crate or kernel is affected

We will acknowledge receipt within **48 hours** and provide an initial assessment within **7 days**.

## Scope

Atlas is an inference server that runs locally with GPU access. The primary threat surface includes:

- **CUDA kernel safety** — Out-of-bounds memory access, buffer overflows in GPU kernels
- **HTTP API** — Input validation on the OpenAI-compatible endpoint (`spark-server`)
- **Weight loading** — Malicious safetensor files, path traversal during model loading
- **Unsafe Rust** — Atlas uses `unsafe` blocks for CUDA FFI; these are high-priority review targets

## Automated Auditing

Atlas runs one automated security check in CI:

- **`cargo deny`** — Audits dependencies for known advisories, license compliance, and banned crates. Runs on every pull request, on pushes to `main` that touch `Cargo.toml`/`Cargo.lock`/`deny.toml`, and weekly. See `.github/workflows/security.yml`.

There is **no** automated static analysis of the CUDA kernel sources. Kernel
memory safety is covered by human review and by the runtime kernel audit, not
by a CI tool. This section previously advertised a `cppcheck` job that has
never existed in this repository; claiming a control you do not run is worse
than claiming none, because it tells a reader a class of defect is already
being looked for.

## Disclosure Policy

We follow coordinated disclosure. Once a fix is available, we will:

1. Merge the fix to `main`
2. Tag a release
3. Credit the reporter (unless anonymity is requested)
