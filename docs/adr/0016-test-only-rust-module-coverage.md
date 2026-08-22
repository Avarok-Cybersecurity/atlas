# ADR-0016: Exempt only proven test-only Rust modules from benchmark invalidation

**Status:** Proposed
**Date:** 2026-08-21
**Builds on:** ADR-0013 (coverage by content, not ancestry)

## Context

The performance boundary contains the whole `crates/` tree. That is safe for
unknown host code, but it also treats a change to a Rust test assertion as a
change to the release program. Twelve open test-audit PRs demonstrate the
result: their only changed files are either
`crates/atlas-core/src/config/tests.rs` or
`crates/atlas-core/src/config/gguf/tests.rs`, yet all ten GPU records are
invalidated.

The files are not merely named like tests. Their parent modules declare them
through module edges guarded by `#[cfg(test)]`, so they are absent from
non-test builds. A benchmark cannot observe code that is not part of the
benchmarked program.

A glob such as `**/tests.rs` would still be unsafe. Rust permits arbitrary
module names and explicit `#[path]` or `include!` edges, and a production file
can be named `tests.rs`. The gate must not infer reachability from naming.

## Decision

Maintain an exact registry of test-only Rust module files. Each entry records:

* the exact file path;
* the exact parent module path;
* the module name declared by that parent;
* the exact `#[path]` value when the module uses one.

An exact registry match does not invalidate benchmark records. No prefix,
suffix, basename, or directory matching is permitted.

CI validates every registry entry against the repository:

* the file and parent both exist;
* the parent contains exactly one module declaration for that name;
* the registered declaration, including any explicit path, is guarded by
  `#[cfg(test)]`;
* repository Rust sources do not create an explicit `include!` or `#[path]`
  edge to the registered file.

The coverage map remains a verdict-defining boundary file. Editing the
registry or its matching logic therefore invalidates all ten records. For the
initial landing only, the existing content-pinned amnesty mechanism covers the
exact final blobs of `coverage.rs`, `check.rs`, and `required.rs`. The latter
two carry the bootstrap hook and corrected ten-gate documentation. Any later
edit changes its blob OID and invalidates normally. The grant expires by test
once every required gate has a record newer than 2026-08-21.

## Fail-closed behavior

Files not in the registry keep the existing `crates/` behavior. This includes:

* files that merely look like tests;
* nested test helpers not separately registered;
* tests embedded in production source files;
* every production neighbour of a registered test module;
* every gate boundary file.

If a PR changes both a registered test file and its production parent, the
parent still invalidates all applicable gates. Removing a `#[cfg(test)]` guard
also changes the parent and fails the structural coverage test.

## Consequences

Test-only PRs can satisfy the benchmark check using existing records because
the benchmarked release inputs did not change. Their Rust tests, formatting,
lint, and platform builds continue to run normally.

Adding another exemption is an explicit policy change with reviewable proof,
not an automatic result of choosing a test-like filename. A future release
dependency-closure fingerprint may replace this registry, but it is not a
precondition for correcting the two demonstrated false invalidations.

## Tests

The policy tests pin seven properties:

1. every registered file has its guarded parent edge;
2. registered files invalidate no GPU records;
3. lookalike and neighbouring paths still invalidate all ten records;
4. a production change cannot hide beside an exempt test change;
5. no gate boundary file appears in the test-only registry.
6. a real git fixture keeps an existing record across a test-only commit and
   invalidates it after a production-parent commit;
7. the bootstrap grant contains exactly three content-pinned boundary blobs,
   fails closed for every other blob, and demands removal after fresh records.
