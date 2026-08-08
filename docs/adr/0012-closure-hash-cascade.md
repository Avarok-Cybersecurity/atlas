# ADR-0012: Scope a kernel change by its include closure, not its file set

**Status:** Accepted
**Date:** 2026-08-08

## Context

`kernels/gb10/common/` holds 160 `.cu` files. Each model directory shadows
only a handful of them — 11 for `qwen3.6-27b`, 5 for `qwen3.6-35b-a3b`, 18
for `deepseek-v4-flash`. `collect_cu_files` (`atlas-kernels/build.rs`) merges
the two layers into a `HashMap` keyed by file stem, common first, model
second, so the model directory holds *overrides*, not the model's kernels.

The gate (ADR-0009's tuple world, plus the `PERF_PATHS` invalidation set)
treats `kernels` as one path. Any edit under it invalidates every committed
benchmark record for every gate. With two BFCL accuracy legs at ~3.5
GPU-hours each on a three-box fleet, a shared-kernel edit costs most of a
day of hardware — and nothing shadows `paged_decode_attn_fp8.cu`, so a
change there is inherited by all 22 gb10 targets.

We want a shared-kernel change to re-test the targets it can actually
affect, and to prove — not assume — that it cannot affect the rest.

The obvious design is to hash each target's **resolved file set** after
shadowing: if the set is unchanged, the record still covers. It is wrong,
in two ways we found only by reading the tree.

1. **Shadow files textually include what they shadow.**
   `kernels/gb10/qwen3.6-27b/nvfp4/inferspark_prefill_paged_indirect.cu:12`
   is `#include "../../common/inferspark_prefill_paged_indirect.cu"`. Eight
   files do this. A set hash says the 27B shadows that stem and is therefore
   immune — while the edited bytes are compiled straight into its kernel.
   The failure is silent and fails *open*, on exactly the change class the
   scheme exists to scope.

2. **Headers are in no file set at all.** `find_cu_files` is non-recursive
   and matches only `.cu`, so the nine `common/*.cuh` files — including the
   one carrying the `BR64` define — are invisible. Editing a header would
   invalidate nothing.

We also could not reuse the existing `ATLAS_KERNEL_SET_HASH`
(`atlas-kernels/src/lib.rs`): it is global across all targets, hashes the
*generated* `target_ptx.rs` text rather than kernel content, and is a stub
under `ATLAS_SKIP_BUILD=1` — which is how the gate runs.

We considered three responses:

1. **Keep the single `kernels` path.** Correct, and the status quo. Costs a
   day of GPU per shared edit; the cost is what makes people want an
   exception, and exceptions are how gates die.
2. **Hash the resolved file set.** Cheap and wrong, as above.
3. **Hash the transitive include closure.** Follows quoted `#include`s from
   each resolved source, so an included common file and an edited `.cuh` are
   both inside the hash.

## Decision

Scope by **closure hash**: a content hash per `(hardware, model, quant)`
target over the transitive quoted-`#include` closure of its resolved
sources, plus merged nvcc flags, `MODEL.toml`, `KERNEL.toml`,
`HARDWARE.toml`, arch, and nvcc version. One crate computes it, used by both
`build.rs` and the gate checker.

The hash is **two-sided**: the tree-computed value must equal a value baked
into the binary and reported by the serve. A record whose two sides disagree
is refused rather than trusted.

Unchanged hash means the target's compiled device code and baked config are
byte-identical to what the record measured, so — for a diff confined to
`kernels/` — its records still cover. Anything the closure cannot see keeps
today's behaviour: `crates/`, `Cargo.lock`, `jinja-templates`,
`3rdparty_patches` invalidate everything, as does any computation error,
missing field, or unresolvable include.

## Consequences

**Better:**
- A shared-kernel edit re-tests the targets whose compiled code actually
  changed. Targets that genuinely shadow the file — without including it —
  keep their records.
- Header edits are caught for the first time. Today they invalidate nothing.
- The two-sided check closes a second hole: a benchmark can no longer be
  recorded against a binary built from different sources. Today nothing
  connects a record to the binary that produced it beyond a commit sha, and a
  sha does not describe a working tree that was dirty, a stale `target/`, or a
  binary carried between boxes — all of which happen during a gate campaign.

  (An earlier draft of this ADR claimed `MODEL.toml` had no
  `cargo:rerun-if-changed` and was therefore a live staleness bug. That was
  **wrong**: `build_parse.rs:162` emits the directive, and `build_parse.rs` is
  `#[path]`-included from `build.rs:1064`. The two-sided check is justified on
  its own terms above, not by that non-existent bug.)

**Worse:**
- An include walk is a small C preprocessor, and preprocessors are where
  subtle bugs live. It resolves quoted includes only; a `-I` search path or
  a generated header is missed, and `#if`/`#ifdef` are not evaluated, so an
  include in a branch this build never takes is still walked. That
  over-includes, which costs re-runs rather than soundness.

  **An unresolvable include was fatal, and that was reverted after
  measuring.** The rule read well — omitting a file omits compiled bytes —
  but the tree disagreed with it. Of 66 quoted includes under `kernels/`,
  exactly 2 do not resolve: the `GGML_USE_HIP` and `GGML_USE_MUSA` arms of
  one `#if` chain in the 27B's vendored q4k code, whose live `#else` arm
  includes `vendors/cuda.h`. Neither named file exists anywhere in the
  repository, so no `-I` path could supply them and no compiler opens them.

  The cost of the rule was not abstract: it denied an attestation to
  `gb10/qwen3.6-27b/nvfp4` alone — the MLPerf flagship, the 3.5-GPU-hour
  target this whole scheme exists to spare — while the other 21 targets
  kept theirs. A safety rule that switches itself off exactly where the
  cost is highest is not buying safety. Unresolvable includes are now
  hashed by NAME under their own digest label, and `build.rs` emits a
  `cargo:warning` for each, so a third one is noticed without being fatal.

  The residual gap is a header reached through `-I` whose content changes
  with no source edit — already listed as out of scope, not new.
- The hash is one more thing that can be *wrong* rather than merely
  imprecise. A hash that under-specifies its inputs is a cache that returns
  stale results and reports success. The omission list belongs in the
  module doc, kept current.
- Two computations of the same hash must agree forever, across a build
  script and a no-GPU checker. They share a crate for that reason; drift
  between them would be indistinguishable from a real change.

**New problems we created:**
- Nothing the tree can see covers out-of-repo inputs: checkpoint revision,
  recipe content (a separate repository — the record stores a name), serve
  env, driver, container, box state. These are recorded as provenance and
  compared where possible, but no hash sees them, and a reader who trusts
  the hash to mean "nothing changed" will be wrong about them.
- Equal hash proves equal *code*, not equal *outcome* under load. It is
  sound for "this record still describes this binary" and says nothing
  about concurrency-dependent behaviour, which is why bitwise output gating
  remains valid only at C=1.
