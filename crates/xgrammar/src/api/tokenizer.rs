// SPDX-License-Identifier: AGPL-3.0-only
//
// `TokenizerInfo` façade — W7 compatibility shim.
//
// The vendored `xgrammar-rs` `TokenizerInfo::new` took
// `(&[T: AsRef<str>], VocabType, &Option<Box<[i32]>>, bool)` and
// returned `Result<Self, String>` (the C++ constructor could fail).
// The pure-Rust core's constructor takes `(&[String], VocabType,
// Option<usize>, Option<Vec<i32>>, bool)` and is infallible. This
// newtype restores the vendored signature so Atlas's
// `grammar/engine.rs` compiles unchanged.

use crate::tokenizer::TokenizerInfo as CoreInfo;
use crate::tokenizer::VocabType;

/// Decoded-vocabulary metadata the grammar matcher needs.
///
/// Port of the vendored `xgrammar::TokenizerInfo`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TokenizerInfo {
    inner: CoreInfo,
}

impl TokenizerInfo {
    /// Construct the tokenizer info from a raw (encoded) vocabulary.
    ///
    /// Port of the vendored `TokenizerInfo::new`. The vocab size is
    /// `encoded_vocab.len()`; `stop_token_ids` is `None` for
    /// auto-detection or the explicit stop-id set.
    pub fn new<T: AsRef<str>>(
        encoded_vocab: &[T],
        vocab_type: VocabType,
        stop_token_ids: &Option<Box<[i32]>>,
        add_prefix_space: bool,
    ) -> Result<Self, String> {
        let vocab: Vec<String> = encoded_vocab
            .iter()
            .map(|t| t.as_ref().to_string())
            .collect();
        let stops: Option<Vec<i32>> = stop_token_ids.as_ref().map(|b| b.to_vec());
        Ok(Self {
            inner: CoreInfo::new(&vocab, vocab_type, None, stops, add_prefix_space),
        })
    }

    /// Construct from the raw text of a HuggingFace `tokenizer.json`,
    /// inferring `vocab_type` and `add_prefix_space`.
    ///
    /// Port of the vendored `TokenizerInfo::from_huggingface` (the
    /// metadata-from-JSON form, used to side-step the `tokenizers`
    /// crate version skew — see F68).
    pub fn from_huggingface<T: AsRef<str>>(
        encoded_vocab: &[T],
        tokenizer_json: &str,
        vocab_size: Option<usize>,
        stop_token_ids: Option<&[i32]>,
    ) -> Result<Self, String> {
        let vocab: Vec<String> = encoded_vocab
            .iter()
            .map(|t| t.as_ref().to_string())
            .collect();
        let stops: Option<Vec<i32>> = stop_token_ids.map(<[i32]>::to_vec);
        CoreInfo::from_huggingface(&vocab, tokenizer_json, vocab_size, stops)
            .map(|inner| Self { inner })
    }

    /// The vocabulary type. Port of the vendored `vocab_type`.
    pub fn vocab_type(&self) -> VocabType {
        self.inner.vocab_type()
    }

    /// The vocabulary size. Port of the vendored `vocab_size`.
    pub fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    /// Whether the tokenizer prepends a space. Port of the vendored
    /// `add_prefix_space`.
    pub fn add_prefix_space(&self) -> bool {
        self.inner.add_prefix_space()
    }

    /// The stop token ids. Port of the vendored `stop_token_ids`.
    pub fn stop_token_ids(&self) -> Box<[i32]> {
        self.inner.stop_token_ids().to_vec().into_boxed_slice()
    }

    /// The special token ids. Port of the vendored `special_token_ids`.
    pub fn special_token_ids(&self) -> Box<[i32]> {
        self.inner.special_token_ids().to_vec().into_boxed_slice()
    }

    /// Dump tokenizer metadata to JSON. Port of the vendored
    /// `dump_metadata`.
    pub fn dump_metadata(&self) -> String {
        self.inner.dump_metadata()
    }

    /// Clone the wrapped pure-Rust `TokenizerInfo` — the compiler core
    /// takes it by value.
    pub(crate) fn core_clone(&self) -> CoreInfo {
        self.inner.clone()
    }
}
