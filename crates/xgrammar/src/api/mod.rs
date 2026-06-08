// SPDX-License-Identifier: AGPL-3.0-only
//
// Public-API façade — port wave W7.
//
// The pure-Rust algorithmic core (`grammar`, `fsm`, `support`,
// `regex`, `tokenizer`, `schema`, `earley`, `compiler`,
// `structural_tag`, `matcher`) has the full functionality but its
// public names/signatures differ from the vendored C++-backed
// `xgrammar-rs` crate Atlas was built against. This module restores
// the exact vendored surface — same type names, same method
// signatures — so Atlas's `crates/spark-server/src/grammar/*.rs`
// resolves `use xgrammar::{Grammar, GrammarCompiler, CompiledGrammar,
// GrammarMatcher, TokenizerInfo, VocabType, ...}` unchanged.
//
// Where the vendored and core signatures differ, this façade adds thin
// newtype wrappers (`GrammarCompiler`, `GrammarMatcher`,
// `TokenizerInfo`, `Grammar`) rather than touching the core or Atlas.
// The single unavoidable Atlas edit — `grammar/state.rs` passing its
// bitmask buffer as `&mut [i32]` instead of an FFI `DLTensor` — is
// documented in the W7 report.
//
// Module map:
//   dlpack    — DLTensor / DLDevice / DLDataType compat structs.
//   bitmask   — `allocate_token_bitmask` / `get_bitmask_shape` /
//               `reset_token_bitmask` free functions.
//   grammar   — `Grammar` handle (`from_ebnf`, `from_structural_tag`…).
//   tokenizer — `TokenizerInfo` newtype (vendored constructor shape).
//   compiler  — `GrammarCompiler` newtype (vendored constructor shape).
//   matcher   — `GrammarMatcher` newtype (vendored `accept_token`
//               arity, slice-based `fill_next_token_bitmask`).

mod bitmask;
mod compiler;
mod dlpack;
mod grammar;
mod matcher;
mod tokenizer;

pub use bitmask::{allocate_token_bitmask, get_bitmask_shape, reset_token_bitmask};
pub use compiler::GrammarCompiler;
pub use dlpack::{DLDataType, DLDataTypeCode, DLDevice, DLDeviceType, DLTensor};
pub use grammar::Grammar;
pub use matcher::GrammarMatcher;
pub use tokenizer::TokenizerInfo;

// `CompiledGrammar` is opaque to Atlas (it never calls methods on it —
// only stores and forwards it), so the pure-Rust core type is
// re-exported directly. `BatchGrammarMatcher` likewise needs no shim.
pub use crate::compiler::CompiledGrammar;
pub use crate::matcher::BatchGrammarMatcher;

// `HfMetadata` / `detect_metadata_from_hf` already match the vendored
// signatures exactly (`Result<HfMetadata, String>`, fields
// `vocab_type` + `add_prefix_space`).
pub use crate::tokenizer::{HfMetadata, detect_metadata_from_hf};

// `StructuralTagItem` is identical (`begin`, `schema`, `end`).
pub use crate::structural_tag::StructuralTagItem;
