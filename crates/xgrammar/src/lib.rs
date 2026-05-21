// SPDX-License-Identifier: AGPL-3.0-only
//
// Pure-Rust XGrammar — a from-scratch port of mlc-ai/xgrammar v0.1.32,
// replacing the C++ implementation and the `cxx` FFI bridge. No C/C++
// /header/Python files; builds with plain `cargo build`.
//
// PORT STATUS: COMPLETE (wave W7 — public API + Atlas repoint). The
// `api` module exposes the exact public surface the vendored
// `xgrammar-rs` crate provided, so this crate is a drop-in replacement.
// The algorithmic core lives in the modules below; `api` is the
// vendored-signature façade Atlas's `spark-server` links against.

// Index-based loops and same-named submodules (`fsm::fsm`,
// `compiler::compiler`) are kept where they mirror the C++ source's
// structure 1:1 — that traceability is deliberate for a line-by-line
// port and makes diffing against upstream xgrammar tractable.
#![allow(clippy::needless_range_loop, clippy::module_inception)]

pub mod compiler;
pub mod earley;
pub mod fsm;
pub mod grammar;
pub mod matcher;
pub mod regex;
pub mod schema;
pub mod structural_tag;
pub mod support;
pub mod tokenizer;

mod api;

// ── Public API façade (vendored `xgrammar-rs` surface) ─────────────
//
// These are the names Atlas's `use xgrammar::{...}` resolves against.
// The façade types shadow the core's same-named types at the crate
// root; the core types remain reachable via their module paths
// (`xgrammar::compiler::GrammarCompiler`, etc.) for the crate's own
// tests and any consumer that wants the richer pure-Rust API.
pub use api::{
    BatchGrammarMatcher, CompiledGrammar, DLDataType, DLDataTypeCode, DLDevice, DLDeviceType,
    DLTensor, Grammar, GrammarCompiler, GrammarMatcher, HfMetadata, StructuralTagItem,
    TokenizerInfo, allocate_token_bitmask, detect_metadata_from_hf, get_bitmask_shape,
    reset_token_bitmask,
};
pub use tokenizer::VocabType;

// ── Core algorithmic exports (richer pure-Rust API) ────────────────
//
// Not part of the vendored surface — kept public for the crate's own
// test suite and downstream consumers that want direct AST access.
pub use grammar::{GrammarData, GrammarExpr, GrammarExprType, Rule, TagDispatch};
pub use schema::{
    JsonFormat, SchemaConverterOptions, SchemaError, deepseek_xml_tool_calling_to_ebnf,
    json_schema_to_ebnf, json_schema_to_grammar, minimax_xml_tool_calling_to_ebnf,
    qwen_xml_tool_calling_to_ebnf,
};
pub use structural_tag::{
    StructuralTag, StructuralTagError, structural_tag_from_items, structural_tag_to_grammar,
};

// ── `VocabType` SCREAMING-CASE aliases ─────────────────────────────
//
// The vendored C++ `enum class VocabType` surfaced through autocxx as
// SCREAMING_SNAKE variants (`VocabType::RAW`, `BYTE_FALLBACK`,
// `BYTE_LEVEL`). The pure-Rust port follows Rust convention
// (`Raw`/`ByteFallback`/`ByteLevel`). Atlas's `grammar/engine.rs`
// writes the SCREAMING form, so we add associated constants that map
// onto the new variants — both spellings now resolve.
impl VocabType {
    /// Vendored alias for [`VocabType::Raw`].
    pub const RAW: VocabType = VocabType::Raw;
    /// Vendored alias for [`VocabType::ByteFallback`].
    pub const BYTE_FALLBACK: VocabType = VocabType::ByteFallback;
    /// Vendored alias for [`VocabType::ByteLevel`].
    pub const BYTE_LEVEL: VocabType = VocabType::ByteLevel;
}
