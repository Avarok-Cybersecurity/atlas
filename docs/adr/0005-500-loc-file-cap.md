# ADR-0005: 500-LoC per-file cap, enforced in CI

**Status:** Accepted
**Date:** 2026-04-17

## Context

Pre-refactor, `crates/spark-server/` and `crates/spark-model/` had several
files in the 1k–4k-LoC range: a single `model.rs` covering prefill +
decode + verify for every architecture, a single `chat_completions.rs`
mixing routing + parsing + grammar + streaming, etc.

These files had three problems:

1. **AI-assistant context blowup.** A 3k-LoC file fills a tool's read
   buffer, so reviewing assistants (and humans) couldn't keep the whole
   thing in working memory. Bugs hid in the parts that scrolled off.
2. **Merge-conflict surface area.** Anything anyone touched in
   `model.rs` collided with anything else.
3. **Discovery friction.** "Where does verify-K=2 happen?" required
   grepping a giant file rather than navigating to a named module.

We considered three responses:

1. **Just write smaller files.** No cap, just a norm. This had been the
   norm and it didn't hold; under deadline pressure, big files grew
   bigger.
2. **Hard cap, manually enforced.** Reviewers nag. Doesn't scale.
3. **Hard cap, CI-enforced.** A workflow fails the build if any
   `crates/**/*.rs` file exceeds 500 lines.

## Decision

Adopt a **500-line hard cap** on every `*.rs` file under `crates/` (and
`*.cu` files under `kernels/`, with looser exemptions for handful of
template-heavy kernels). Enforcement is a CI job
(`.github/workflows/file-size-cap.yml`) that runs on every PR.

There is an allow-list, currently empty post-refactor.

When a file approaches the cap, the response is to:

- **Split by phase**: scheduler tick into `phases/{decode,verify,sample,
  emit,lifecycle}.rs` (Pattern B in ADR-0006).
- **Split by responsibility**: extract API parsing, body building, stream
  emission into siblings.
- **Convert flat `impl` blocks** into multi-file `impl Model for
  TransformerModel` blocks under `model/trait_impl/` siblings.

500 was chosen empirically — large enough to host a self-contained
unit, small enough that you can read the whole thing in a sitting.

## Consequences

**Better:**
- Every file in `crates/` is now reviewable in one screenful or two.
- AI assistants and human reviewers see the whole module, not a
  scrolling fragment.
- File-level merge conflicts are rare — work that touches "the verify
  path" lands in a small file; work that touches "decode" lands in a
  different file.
- New contributors can navigate by file name to a clearly scoped unit.

**Worse:**
- More files, deeper directory trees. Some logic that "wants" to be
  one 700-line file is now three 250-line files plus a `mod.rs`.
- Cross-cutting refactors touch more files. Renaming a type used in 12
  small files is louder than renaming it in 2 big ones.
- 500 is arbitrary. There is occasional friction at 510 lines where
  the right answer is "let it be 510" but CI says no. We pay this cost
  rather than open a "judgment-call" loophole.

**New problems we created:**
- We rely on the CI job to be uncircumventable. The allow-list must be
  treated as a liability — every entry is a TODO.
- A cap encourages "split into 3 files" over "rethink the design."
  Reviewers must occasionally push back: "this file isn't too long, it's
  doing too much; the right fix is a redesign, not a knife."
