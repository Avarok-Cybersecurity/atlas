// SPDX-License-Identifier: AGPL-3.0-only
//
// TokenizerInfo — port of `TokenizerInfo` / `TokenizerInfo::Impl` in
// `cpp/tokenizer_info.{cc,h}` + `tokenizer_info_impl.h`.
//
// Holds the decoded vocabulary, vocab type, stop tokens, and special
// tokens used by the grammar matcher to mask the logit bitmask.

use std::collections::HashSet;

use crate::tokenizer::decoder::decode_token;
use crate::tokenizer::hf_metadata::detect_metadata_from_hf;
use crate::tokenizer::vocab_type::VocabType;

/// Tokens used to auto-detect stop tokens when the caller does not
/// supply explicit `stop_token_ids`. Faithful to the C++
/// `DETECTION_STOP_TOKENS` set (LLaMA2/3, Phi-2, Gemma, DeepSeek-V2).
const DETECTION_STOP_TOKENS: &[&str] = &[
    "</s>",
    "<|end_of_text|>",
    "<|eot_id|>",
    "<|endoftext|>",
    "<eos>",
    "<|eos|>",
    "<end_of_turn>",
    "<\u{ff5c}end\u{2581}of\u{2581}sentence\u{ff5c}>",
];

/// Decoded-vocabulary view of a tokenizer for grammar-constrained
/// decoding.
///
/// Construct with [`TokenizerInfo::new`] from a raw vocabulary, or with
/// [`TokenizerInfo::from_huggingface`] to additionally infer the vocab
/// type / prefix-space from a `tokenizer.json`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerInfo {
    vocab_type: VocabType,
    vocab_size: usize,
    add_prefix_space: bool,
    /// Decoded token bytes for every id `0..vocab_size`. Ids beyond the
    /// supplied raw vocab are padding and decode to an empty sequence.
    decoded_vocab: Vec<Vec<u8>>,
    /// `(id, token)` pairs sorted lexicographically by token bytes.
    /// Excludes special tokens and stop tokens — maximizes prefix reuse
    /// during trie matching.
    sorted_decoded_vocab: Vec<(i32, Vec<u8>)>,
    /// Pseudo-trie subtree ranges: `trie_subtree_nodes_range[i]` is the
    /// exclusive end index of the subtree rooted at sorted entry `i`.
    trie_subtree_nodes_range: Vec<i32>,
    /// Ids accepted as end-of-generation by the grammar matcher.
    stop_token_ids: Vec<i32>,
    /// Ids masked out (ignored) during grammar-guided generation.
    special_token_ids: Vec<i32>,
}

impl TokenizerInfo {
    /// Build a `TokenizerInfo` from a raw (encoded) vocabulary.
    ///
    /// Mirrors the C++ `TokenizerInfo` constructor:
    /// * `encoded_vocab` — raw vocabulary strings, indexed by token id.
    /// * `vocab_type` — controls token decoding (see [`VocabType`]).
    /// * `vocab_size` — total id space; `None` means `encoded_vocab.len()`.
    ///   When larger, the trailing ids become padding special tokens.
    /// * `stop_token_ids` — explicit stop ids; `None` triggers
    ///   detection via `DETECTION_STOP_TOKENS`.
    /// * `add_prefix_space` — recorded for metadata, no decoding effect.
    pub fn new(
        encoded_vocab: &[String],
        vocab_type: VocabType,
        vocab_size: Option<usize>,
        stop_token_ids: Option<Vec<i32>>,
        add_prefix_space: bool,
    ) -> Self {
        let vocab_size = vocab_size.unwrap_or(encoded_vocab.len());

        let mut decoded_vocab: Vec<Vec<u8>> = Vec::with_capacity(encoded_vocab.len());
        let mut sorted_decoded_vocab: Vec<(i32, Vec<u8>)> = Vec::new();
        let mut stop_ids: Vec<i32> = Vec::new();
        let mut special_ids: Vec<i32> = Vec::new();

        // The explicit stop set, if provided, is consulted per id.
        let explicit_stop: Option<HashSet<i32>> =
            stop_token_ids.as_ref().map(|v| v.iter().copied().collect());

        for (i, raw) in encoded_vocab.iter().enumerate() {
            let id = i as i32;
            let token = decode_token(raw, vocab_type);

            let is_stop = match &explicit_stop {
                // Explicit ids: this id is a stop iff it is listed.
                Some(set) => set.contains(&id),
                // Auto-detection: a stop iff the decoded token text is
                // one of the well-known end-of-text markers.
                None => token_str_in_detection_set(&token),
            };

            if is_stop {
                stop_ids.push(id);
            } else if is_special_token(&token) {
                special_ids.push(id);
            } else {
                sorted_decoded_vocab.push((id, token.clone()));
            }
            decoded_vocab.push(token);
        }

        // Ids beyond the raw vocab are padding — pure special tokens.
        for i in encoded_vocab.len()..vocab_size {
            special_ids.push(i as i32);
        }
        // Keep `decoded_vocab` length == vocab_size so callers can index
        // the full id space (padding ids decode to empty bytes).
        decoded_vocab.resize(vocab_size, Vec::new());

        // Lexicographic sort by token bytes — stable on (bytes, id).
        sorted_decoded_vocab.sort_by(|a, b| a.1.cmp(&b.1));

        let trie_subtree_nodes_range = build_trie_ranges(&sorted_decoded_vocab);

        TokenizerInfo {
            vocab_type,
            vocab_size,
            add_prefix_space,
            decoded_vocab,
            sorted_decoded_vocab,
            trie_subtree_nodes_range,
            stop_token_ids: stop_ids,
            special_token_ids: special_ids,
        }
    }

    /// Build a `TokenizerInfo` from the raw text of a HuggingFace
    /// `tokenizer.json`, inferring `vocab_type` and `add_prefix_space`.
    ///
    /// Returns `Err` when `tokenizer_json` is not a parseable JSON
    /// object. The caller still supplies the raw vocabulary, optional
    /// `vocab_size`, and optional `stop_token_ids`.
    pub fn from_huggingface(
        encoded_vocab: &[String],
        tokenizer_json: &str,
        vocab_size: Option<usize>,
        stop_token_ids: Option<Vec<i32>>,
    ) -> Result<Self, String> {
        let meta = detect_metadata_from_hf(tokenizer_json)?;
        Ok(Self::new(
            encoded_vocab,
            meta.vocab_type,
            vocab_size,
            stop_token_ids,
            meta.add_prefix_space,
        ))
    }

    /// The vocabulary type.
    pub fn vocab_type(&self) -> VocabType {
        self.vocab_type
    }

    /// The total id space size (raw vocab length, or the override).
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    /// Whether the tokenizer prepends a space to the first token.
    pub fn add_prefix_space(&self) -> bool {
        self.add_prefix_space
    }

    /// Decoded token bytes, indexed by token id (length `vocab_size`).
    pub fn decoded_vocab(&self) -> &[Vec<u8>] {
        &self.decoded_vocab
    }

    /// Ids accepted as end-of-generation by the grammar matcher.
    pub fn stop_token_ids(&self) -> &[i32] {
        &self.stop_token_ids
    }

    /// Ids masked out during grammar-guided generation.
    pub fn special_token_ids(&self) -> &[i32] {
        &self.special_token_ids
    }

    /// `(id, token)` pairs sorted lexicographically by token bytes,
    /// excluding stop and special tokens.
    pub fn sorted_decoded_vocab(&self) -> &[(i32, Vec<u8>)] {
        &self.sorted_decoded_vocab
    }

    /// Pseudo-trie subtree end-indices over [`Self::sorted_decoded_vocab`].
    pub fn trie_subtree_nodes_range(&self) -> &[i32] {
        &self.trie_subtree_nodes_range
    }

    /// Serialize the metadata to the compact JSON form C++
    /// `DumpMetadata` emits:
    /// `{"vocab_type":N,"vocab_size":N,"add_prefix_space":b,"stop_token_ids":[...]}`.
    pub fn dump_metadata(&self) -> String {
        let ids: Vec<String> = self.stop_token_ids.iter().map(|i| i.to_string()).collect();
        format!(
            "{{\"vocab_type\":{},\"vocab_size\":{},\"add_prefix_space\":{},\"stop_token_ids\":[{}]}}",
            self.vocab_type.as_int(),
            self.vocab_size,
            self.add_prefix_space,
            ids.join(",")
        )
    }
}

/// A decoded token is a "special token" iff it is empty.
///
/// Faithful to C++ `IsSpecialToken`, whose comment notes that *only* the
/// empty string is treated as special.
fn is_special_token(token: &[u8]) -> bool {
    token.is_empty()
}

/// True if the decoded token, interpreted as UTF-8, equals one of the
/// auto-detection stop markers. Tokens that are not valid UTF-8 can
/// never match (the C++ stores `std::string` and compares directly,
/// which for our purposes is equivalent to a byte comparison).
fn token_str_in_detection_set(token: &[u8]) -> bool {
    match std::str::from_utf8(token) {
        Ok(s) => DETECTION_STOP_TOKENS.contains(&s),
        Err(_) => false,
    }
}

/// Build the pseudo-trie subtree ranges over the sorted vocabulary.
///
/// Faithful to the C++ prefix-stack algorithm: `trie[i]` is the
/// exclusive end of the subtree rooted at sorted entry `i`. The stack
/// is unwound whenever the current token does *not* contain the stack
/// top as a substring (C++ uses `token.find(top) == npos`).
fn build_trie_ranges(sorted: &[(i32, Vec<u8>)]) -> Vec<i32> {
    let mut ranges = vec![0i32; sorted.len()];
    // Stack of (token_bytes_index_into_sorted, stack_slot).
    let mut prefix_stack: Vec<usize> = Vec::new();

    for (i, (_, token)) in sorted.iter().enumerate() {
        while let Some(&top) = prefix_stack.last() {
            let top_token = &sorted[top].1;
            if contains_subslice(token, top_token) {
                break;
            }
            ranges[top] = i as i32;
            prefix_stack.pop();
        }
        prefix_stack.push(i);
    }
    let end = sorted.len() as i32;
    while let Some(top) = prefix_stack.pop() {
        ranges[top] = end;
    }
    ranges
}

/// True if `needle` occurs as a contiguous subslice of `haystack`
/// (the empty needle always matches), mirroring `std::string::find`.
fn contains_subslice(haystack: &[u8], needle: &[u8]) -> bool {
    if needle.is_empty() {
        return true;
    }
    if needle.len() > haystack.len() {
        return false;
    }
    haystack.windows(needle.len()).any(|w| w == needle)
}

#[cfg(test)]
#[path = "tests.rs"]
mod tests;
