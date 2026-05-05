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

Atlas runs automated security checks in CI:

- **`cargo deny`** — Audits dependencies for known advisories, license compliance, and banned crates (weekly + on every PR)
- **`cppcheck`** — Static analysis on CUDA kernel source

## Disclosure Policy

We follow coordinated disclosure. Once a fix is available, we will:

1. Merge the fix to `main`
2. Tag a release
3. Credit the reporter (unless anonymity is requested)
