// SPDX-License-Identifier: AGPL-3.0-only
//
// Pure-Rust XGrammar — a from-scratch port of mlc-ai/xgrammar v0.1.32,
// replacing the C++ implementation and the `cxx` FFI bridge. No C/C++
// /header/Python files; builds with plain `cargo build`.
//
// PORT STATUS: wave W1 (grammar AST foundation). The public API
// (`Grammar`, `GrammarCompiler`, `GrammarMatcher`, `CompiledGrammar`,
// `TokenizerInfo`, `StructuralTagItem`, `VocabType`) is introduced
// wave-by-wave per PORT_PLAN.md; until then this crate is not yet a
// drop-in replacement for the vendored `xgrammar-rs`.

pub mod fsm;
pub mod grammar;
pub mod support;

pub use grammar::{GrammarData, GrammarExpr, GrammarExprType, Rule, TagDispatch};
