# XGrammar → Pure Rust Port

## Goal

Replace the vendored C++ `xgrammar` (v0.1.32) + the `cxx`-FFI
`xgrammar-rs` binding crate with a from-scratch **pure-Rust** crate.
End state: **zero `.c` / `.cc` / `.h` / `.hpp` / `.py` files** anywhere
in the grammar stack; `cargo build` with no nvcc-for-grammar, no
cpptrace submodule, no `cxx` bridge.

Source of truth for the port: `mlc-ai/xgrammar` tag `v0.1.32`,
checked out at `/workspace/.cargo/xgrammar-rs-cache/xgrammar-v0.1.32/`.

## Scope (measured)

- C++ to port: **~20,000 LoC of algorithmic core** across ~31 files,
  plus ~5-6 small `cpp/support/` utility files.
- Skipped: `cpp/nanobind/` (3 files, Python C-ext) and
  `cpp/testing.{cc,h}` (port as `#[cfg(test)]` instead).
- ~11 `cpp/support/` files are *replaced by crates*, not ported:
  `dynamic_bitset`→`bitvec`, `thread_pool`→`rayon`,
  `thread_safe_cache`→`dashmap`, `json_serializer`+`reflection`→`serde`,
  `logging`→`tracing`/`thiserror`.

## Public API contract (what Atlas's `grammar/` module consumes)

These must exist with matching behaviour so `crates/spark-server/src/
grammar/` needs zero changes:

- `Grammar` — `from_structural_tag`, EBNF construction
- `GrammarCompiler` — `new(tokenizer_info, max_threads, cache_enabled,
  cache_limit)`, `compile_grammar`, `compile_json_schema`,
  `compile_builtin_json_grammar`, `compile_grammar_from_ebnf`,
  `compile_structural_tag`
- `CompiledGrammar`
- `GrammarMatcher` — `new`, `is_terminated`, `fill_next_token_bitmask`,
  `accept_token`, `rollback`
- `BatchGrammarMatcher`
- `TokenizerInfo` — `new(vocab, VocabType, stop_tokens, …)`,
  `detect_metadata_from_hf`
- `VocabType` { Raw, ByteFallback, ByteLevel }
- `StructuralTagItem`

## Module layout (`crates/xgrammar/src/`)

```
lib.rs              public re-exports
support/            int_set, compact_2d_array, encoding, union_find  [crate-backed otherwise]
grammar/            data model (Grammar AST), builder, EBNF parser, functors, printer
fsm/                fsm, fsm_builder
earley/             earley_parser
compiler/           grammar_compiler, compiled_grammar
matcher/            grammar_matcher, batch
schema/             json_schema_converter (+ _ext)
structural_tag/     structural_tag
regex/              regex_converter
tokenizer/          tokenizer_info
```

## Dependency DAG — port order (bottom-up)

```
W1  support utilities          (no deps)
W1  grammar data model         (grammar_impl.h)              ← FOUNDATION
W2  EBNF parser                (grammar_parser.cc)           dep: W1
W2  FSM                        (fsm.cc/.h, fsm_builder.cc)   dep: support
W3  grammar functors           (grammar_functor.cc)          dep: W1,W2
W3  regex converter            (regex_converter.cc)          dep: FSM,grammar
W3  tokenizer info             (tokenizer_info.cc)           dep: support
W4  JSON-schema converter      (json_schema_converter*.cc)   dep: grammar builder
W4  Earley parser              (earley_parser.cc)            dep: grammar,FSM
W5  grammar compiler           (grammar_compiler.cc,         dep: W1-W4
                                compiled_grammar.cc)
W5  structural tag             (structural_tag.cc)           dep: grammar,schema
W6  grammar matcher            (grammar_matcher.cc)          dep: compiler,earley
W7  public API + Atlas repoint (lib.rs; Cargo path swap)     dep: all
```

W1's grammar data model defines the shared types every later wave
builds on — it is ported first and by hand (not farmed out) so the
type contract is fixed before parallel work begins.

## Verification methodology

Each subsystem ports the corresponding C++ unit tests
(`tests/cpp/` + `tests/python/`) as Rust `#[cfg(test)]` tests. The
final gate: stand up Atlas with the pure-Rust crate path-swapped in,
run `tool-eval-bench` and compare tool-call pass rate against the
C++-xgrammar baseline — must be at parity.

## Status

- [x] Worktree `feat/xgrammar-pure-rust`, crate scaffold
- [ ] W1 grammar data model — IN PROGRESS
- [ ] W1 support utilities
- [ ] W2 … W7
