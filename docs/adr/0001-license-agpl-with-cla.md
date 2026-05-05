# ADR-0001: License — AGPL-3.0 with a CLA

**Status:** Accepted
**Date:** 2026-04-17

## Context

Atlas is a from-scratch CUDA inference engine that's competitive with
production stacks (vLLM, TensorRT-LLM) for the model families it covers. We
want the code public so users can audit kernels and contribute new
hardware/model support, but we also want to retain the option to monetize
hosted offerings or relicense to a commercial customer who can't ship AGPL
code.

The candidate license shape was:

1. **MIT / Apache-2.0** — maximally permissive. Anyone can fork, rebrand,
   sell hosted Atlas without contributing back. Disqualified: we want
   improvements made by service providers to flow back upstream.
2. **GPL-3.0** — strong copyleft, but the network-use loophole means a SaaS
   provider can run modified Atlas internally without ever distributing the
   binary, and thus owes nothing back.
3. **AGPL-3.0** — closes the SaaS loophole: if you offer Atlas-as-a-service
   over a network, you must publish your modifications.
4. **Source-available custom license** (e.g. BSL, Elastic) — discourages
   contributions from users who treat license proliferation as a smell. The
   AGPL is widely recognized; a homebrew license is not.

Separately, we want a **Contributor License Agreement** so we can relicense
the codebase (e.g. dual-license to a commercial customer) without chasing
every contributor for re-permission.

## Decision

- Atlas ships under **AGPL-3.0-only** (`LICENSE` in repo root, SPDX headers
  on every source file enforced by `.licenserc.yaml` + a CI job).
- Every contributor signs the Atlas **CLA** (`CLA.md`) before their first
  PR is merged. The CLA grants Atlas an irrevocable license to redistribute
  the contribution under any future Atlas-chosen license, while leaving
  copyright with the contributor.
- The PR template includes a CLA checkbox; merging without it is a
  reviewer-blocking issue.

## Consequences

**Better:**
- Hosted-service forks must contribute their changes back, which is the
  point.
- The CLA gives us a clean path to dual-license (e.g. for a customer that
  can't ship AGPL code in their product).
- AGPL is well-understood by enterprise legal teams — far less friction
  than a custom source-available license.

**Worse:**
- Some contributors and companies refuse to sign CLAs as a matter of
  policy. We will lose those contributions.
- AGPL-3.0 is incompatible with several permissive ecosystems
  (e.g. you cannot embed Atlas into a permissively licensed library
  without re-licensing the whole thing). Users who want Atlas as a
  library, not a service, may need a commercial license from us.
- Enforcement of AGPL §13 (network-use disclosure) requires us to actually
  notice violations — practical enforcement is limited.

**New problems we created:**
- Need to maintain a CLA-status check or bot. Today this is manual; should
  be automated via CLA Assistant or similar before contributor volume
  grows.
