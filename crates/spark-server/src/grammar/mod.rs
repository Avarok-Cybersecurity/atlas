// SPDX-License-Identifier: AGPL-3.0-only

//! Grammar-based constrained decoding via xgrammar-rs.
//!
//! Provides two layers:
//! - [`GrammarEngine`]: initialized once at server startup with tokenizer info;
//!   compiles grammars from tool definitions or JSON schemas.
//! - [`GrammarState`]: per-request state that wraps a [`xgrammar::GrammarMatcher`];
//!   fills bitmasks, accepts tokens, supports rollback for MTP speculative decode.

mod compile_misc;
mod compile_tools;
mod engine;
mod schema;
mod state;

#[cfg(test)]
mod tests;

pub use engine::{GrammarEngine, GrammarError};
pub use schema::augment_schema_with_tafc_think;
pub use state::GrammarState;

/// Extract an ordered vocabulary from a HuggingFace tokenizer.
///
/// Returns `vocab[i] = token_string_for_id_i`. Gaps are filled with empty strings.
pub fn extract_ordered_vocab(tokenizer: &tokenizers::Tokenizer) -> Vec<String> {
    let vocab = tokenizer.get_vocab(true);
    let max_id = vocab.values().copied().max().unwrap_or(0) as usize;
    let size = vocab.len().max(max_id + 1);
    let mut ordered = vec![String::new(); size];
    for (token, id) in vocab {
        let idx = id as usize;
        if idx < size {
            ordered[idx] = token;
        }
    }
    ordered
}
