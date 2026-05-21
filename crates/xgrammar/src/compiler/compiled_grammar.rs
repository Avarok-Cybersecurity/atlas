// SPDX-License-Identifier: AGPL-3.0-only
//
// CompiledGrammar — port of `class CompiledGrammar` / `CompiledGrammar
// ::Impl` from `cpp/compiled_grammar.cc` + `cpp/compiled_grammar_impl.h`.
//
// A grammar compiled against a specific tokenizer: it bundles the
// optimized, FSM-accelerated grammar, the tokenizer info, and the
// precomputed per-parser-state adaptive token masks. The matcher (W6)
// consults `adaptive_token_mask` to fill the logit bitmask fast.

use std::collections::HashMap;
use std::sync::Arc;

use crate::earley::ParserState;
use crate::grammar::GrammarData;
use crate::tokenizer::TokenizerInfo;

use super::mask::AdaptiveTokenMask;

/// The inner, shared state of a [`CompiledGrammar`]. Port of
/// `CompiledGrammar::Impl`.
#[derive(Debug)]
pub struct CompiledGrammarImpl {
    /// The optimized, FSM-accelerated grammar.
    pub grammar: GrammarData,
    /// The tokenizer this grammar was compiled against.
    pub tokenizer_info: TokenizerInfo,
    /// Per-parser-state adaptive token mask. Equivalent to the C++
    /// `adaptive_token_mask_cache` (a hash map keyed by `ParserState`).
    pub adaptive_token_mask: HashMap<ParserState, AdaptiveTokenMask>,
}

impl CompiledGrammarImpl {
    /// Approximate heap memory usage in bytes. Port of
    /// `MemorySize(const CompiledGrammar::Impl&)`.
    pub fn memory_size(&self) -> usize {
        let grammar_bytes = self.grammar.complete_fsm.memory_size()
            + self.grammar.num_exprs() as usize * 4
            + self.grammar.num_rules() as usize * 32;
        let mask_bytes: usize = self
            .adaptive_token_mask
            .values()
            .map(AdaptiveTokenMask::memory_size)
            .sum();
        grammar_bytes + mask_bytes
    }
}

/// A grammar compiled against a tokenizer — the result of
/// preprocessing performed by [`super::GrammarCompiler`].
///
/// Cheap to clone: the inner state is shared via [`Arc`], matching the
/// C++ pimpl `shared_ptr` semantics.
#[derive(Debug, Clone)]
pub struct CompiledGrammar {
    pimpl: Arc<CompiledGrammarImpl>,
}

impl CompiledGrammar {
    /// Wrap an already-built [`CompiledGrammarImpl`].
    pub fn from_impl(pimpl: Arc<CompiledGrammarImpl>) -> Self {
        Self { pimpl }
    }

    /// The associated optimized grammar.
    pub fn grammar(&self) -> &GrammarData {
        &self.pimpl.grammar
    }

    /// The associated tokenizer info.
    pub fn tokenizer_info(&self) -> &TokenizerInfo {
        &self.pimpl.tokenizer_info
    }

    /// The precomputed per-state adaptive token masks.
    pub fn adaptive_token_mask(&self) -> &HashMap<ParserState, AdaptiveTokenMask> {
        &self.pimpl.adaptive_token_mask
    }

    /// Look up the adaptive token mask for a parser state, if compiled.
    pub fn mask_for_state(&self, state: &ParserState) -> Option<&AdaptiveTokenMask> {
        self.pimpl.adaptive_token_mask.get(state)
    }

    /// Approximate memory usage in bytes. Port of
    /// `CompiledGrammar::MemorySizeBytes`.
    pub fn memory_size_bytes(&self) -> usize {
        self.pimpl.memory_size()
    }

    /// Shared-pointer access to the inner state — used by the matcher.
    pub fn inner(&self) -> &Arc<CompiledGrammarImpl> {
        &self.pimpl
    }
}
