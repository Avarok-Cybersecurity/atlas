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
//   rule_cache       — cross-grammar RuleLevelCache (Tier 2)
//
// SIMPLIFICATIONS vs C++
// ----------------------
//  * The grammar-level cache is a `dashmap` keyed by the request
//    parameters; it has no LRU byte-budget eviction (the C++
//    `ThreadSafeLRUCache`). `cache_limit_bytes` is recorded and
//    reported but not enforced — entries are kept until `clear_cache`.
//  * `compile_structural_tag` delegates the tag-JSON -> `GrammarData`
//    conversion to the W5 `src/structural_tag/` module.
//
// TIER 2 PERF FEATURES (ported from xgrammar-main)
// ------------------------------------------------
//  * Cross-grammar `RuleLevelCache` (`rule_cache`, upstream commit
//    `bfb2a79`): a structurally-keyed, LRU-bounded cache of computed
//    `AdaptiveTokenMask`s, shared across every grammar a `GrammarCompiler`
//    builds. Re-enables the per-rule `GrammarFsmHasher` (its hashes are
//    the cache key). 3-7x on multi-request tool-calling.
//  * Overlapped mask generation — `CompiledGrammar::compile_top_k_masks`:
//    eagerly warms the K most-expensive masks (during prefill, before
//    decode), populating the lazy JIT cache.

mod compile;
mod compiled_grammar;
mod compiler;
mod mask;
mod mask_gen;
mod rule_cache;

pub use compiled_grammar::{CompiledGrammar, CompiledGrammarImpl};
pub use compiler::{CompileError, GrammarCompiler};
pub use mask::{AdaptiveTokenMask, StoreType, USE_BITSET_THRESHOLD};
pub use rule_cache::{RuleLevelCache, RuleMaskKey, UNLIMITED_SIZE};

#[cfg(test)]
mod tests;
