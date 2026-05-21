// SPDX-License-Identifier: AGPL-3.0-only
//
// Grammar compiler — port wave W5.
//
// Pure-Rust port of `cpp/grammar_compiler.cc` + `cpp/compiled_grammar
// .cc` + `cpp/compiled_grammar_impl.h`. Turns a grammar / JSON schema /
// structural tag, together with a tokenizer, into a `CompiledGrammar`:
// the optimized FSM-accelerated grammar plus the precomputed
// per-parser-state adaptive token masks the matcher uses to fill the
// logit bitmask fast.
//
// Module map:
//   mask             — AdaptiveTokenMask (accept/reject/uncertain set)
//   mask_gen         — per-state mask computation (EarleyParser scan)
//   compiled_grammar — CompiledGrammar / CompiledGrammarImpl
//   compile          — no-cache compilation core (XGrammar-2 JIT)
//   compiler         — GrammarCompiler with the dashmap-backed cache
//
// SIMPLIFICATIONS vs C++
// ----------------------
//  * The cross-grammar `RuleLevelCache` is omitted — it is a pure
//    speed optimization and the mask computation is fully correct
//    without it (see `mask_gen`).
//  * The grammar-level cache is a `dashmap` keyed by the request
//    parameters; it has no LRU byte-budget eviction (the C++
//    `ThreadSafeLRUCache`). `cache_limit_bytes` is recorded and
//    reported but not enforced — entries are kept until `clear_cache`.
//  * `compile_structural_tag` delegates the tag-JSON -> `GrammarData`
//    conversion to the W5 `src/structural_tag/` module.

mod compile;
mod compiled_grammar;
mod compiler;
mod mask;
mod mask_gen;

pub use compiled_grammar::{CompiledGrammar, CompiledGrammarImpl};
pub use compiler::{CompileError, GrammarCompiler};
pub use mask::{AdaptiveTokenMask, StoreType, USE_BITSET_THRESHOLD};

#[cfg(test)]
mod tests;
