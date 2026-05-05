# ADR-0006: Multi-file module idiom (`mod.rs` + sibling files)

**Status:** Accepted
**Date:** 2026-04-17

## Context

ADR-0005's 500-LoC cap forces splits. Without a convention for *how* to
split, we ended up with three different idioms in three crates:

- Anonymous `pub use` re-exports that hid the real source location.
- Flat-file modules that imported a dozen siblings, defeating module
  privacy.
- `impl Foo` blocks in many files but no single canonical "where is the
  trait surface defined" file.

We needed one idiom, applied consistently, that supports two recurring
shapes:

- **Pattern A** — splitting one large function into helper functions that
  share a `Ctx` struct or a parameter pack. Useful for scheduler ticks,
  request handlers, anything with a long linear flow.
- **Pattern B** — splitting one large `impl Trait for Type { ... }` block
  into multiple files, each implementing a coherent subset of the trait
  methods.

## Decision

Atlas's module idiom is:

```
foo/
├── mod.rs                # public surface, type definitions, re-exports
├── inner.rs              # private impl details (Pattern A delegate target)
└── trait_impl/
    ├── mod.rs            # `mod prefill; mod decode; mod verify;`
    ├── prefill.rs        # impl Trait for Type { fn prefill(...) ... }
    ├── decode.rs         # impl Trait for Type { fn decode(...) ... }
    └── verify.rs         # impl Trait for Type { fn verify(...) ... }
```

Rules:

1. **`mod.rs` is the module's public face.** Type/struct/trait
   definitions live here. Sibling files contain `impl` blocks and
   private helpers.
2. **No anonymous `pub use *`.** Every re-export is named explicitly so
   `cargo doc` and grep find the source.
3. **Pattern A (one fn → many helpers):** The public function is a thin
   wrapper that builds a `Ctx` struct, then calls a private
   `something_inner(&mut ctx)`. Phase functions live in siblings and
   take `&mut Ctx`.
4. **Pattern B (one trait impl → many files):** Place each impl block
   under `trait_impl/<phase>.rs`. The trait *signature* stays in one
   place; the *implementation* is split. This is the only case where
   we tolerate `impl Trait for Type { ... }` appearing in more than
   one file — and only inside a `trait_impl/` subdirectory so the
   intent is explicit.
5. **No `lib.rs` mega-modules.** Each crate's `lib.rs` is a flat
   `pub mod` listing plus a few prelude re-exports.

## Consequences

**Better:**
- New contributors can navigate by phase: "where is verify implemented?"
  → `crates/spark-model/src/model/trait_impl/verify.rs`.
- File-size-cap (ADR-0005) compliance falls out naturally — each phase
  file has a hard ceiling and a clear purpose.
- Refactors are smaller blast-radius — moving prefill code touches one
  file, not the whole `Model` impl.

**Worse:**
- More files to open. A reader chasing "the model" has to hop across
  ~6 files instead of scrolling one.
- Some Rust idioms (e.g. cross-method shared private state) are awkward
  when impls are split. We work around this with `Ctx` structs that
  collect the shared bag of state explicitly.

**New problems we created:**
- New patterns proliferate. We've already seen Pattern C ("phase
  functions in siblings, no Ctx") emerge ad-hoc. Reviewers should push
  back when a file uses an idiom not described above.
- IDE rename + go-to-definition occasionally lose the thread when
  `impl` blocks span files. `rust-analyzer` handles it correctly today,
  but rough edges remain.
