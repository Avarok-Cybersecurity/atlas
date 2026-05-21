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

pub mod compiler;
pub mod earley;
pub mod fsm;
pub mod grammar;
pub mod regex;
pub mod schema;
pub mod structural_tag;
pub mod support;
pub mod tokenizer;

pub use compiler::{CompiledGrammar, GrammarCompiler};
pub use grammar::{GrammarData, GrammarExpr, GrammarExprType, Rule, TagDispatch};
pub use schema::{
    deepseek_xml_tool_calling_to_ebnf, json_schema_to_ebnf, json_schema_to_grammar,
    minimax_xml_tool_calling_to_ebnf, qwen_xml_tool_calling_to_ebnf, JsonFormat,
    SchemaConverterOptions, SchemaError,
};
pub use structural_tag::{
    structural_tag_from_items, structural_tag_to_grammar, StructuralTag, StructuralTagError,
    StructuralTagItem,
};
pub use tokenizer::{TokenizerInfo, VocabType};
