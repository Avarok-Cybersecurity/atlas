// SPDX-License-Identifier: AGPL-3.0-only

use xgrammar::{GrammarCompiler, TokenizerInfo, VocabType, detect_metadata_from_hf};

use super::extract_ordered_vocab;

// ── GrammarEngine ──────────────────────────────────────────────────────

/// Server-wide grammar compilation engine.
///
/// Initialized once at startup with the tokenizer vocabulary. Compiles
/// grammars on demand and caches them by schema fingerprint.
pub struct GrammarEngine {
    pub(super) compiler: GrammarCompiler,
    vocab_size: usize,
}

// SAFETY: GrammarEngine is initialized on the main thread and moved to the
// scheduler thread. The xgrammar-rs C++ bindings have raw pointers that don't
// auto-impl Send, but the engine is only accessed from one thread at a time.
unsafe impl Send for GrammarEngine {}

/// Errors from grammar compilation or matching.
#[derive(Debug, thiserror::Error)]
pub enum GrammarError {
    #[error("grammar compilation failed: {0}")]
    Compilation(String),
    #[error("invalid JSON schema: {0}")]
    InvalidSchema(String),
    #[error("no tools provided")]
    NoTools,
}

impl GrammarEngine {
    /// Build a [`GrammarEngine`] from an ordered vocabulary (index = token ID).
    ///
    /// `vocab` must be ordered by token ID: `vocab[i]` is the string for token `i`.
    /// `stop_token_ids` are the EOS token IDs for the model.
    ///
    /// This constructor uses [`VocabType::RAW`] — the vocab strings are
    /// treated as the literal byte sequence each token decodes to. That
    /// is correct for tokenizers whose stored vocab strings already
    /// equal the decoded bytes (e.g. SentencePiece-style with
    /// byte_fallback). For HuggingFace **ByteLevel BPE** tokenizers
    /// (Qwen3.5/3.6, MiniMax M2/M2.7, Mistral 4) the vocab strings are
    /// stored in the GPT-2 ByteLevel-encoded form (`Ġ` for space, `Ċ`
    /// for `\n`, etc.) and `RAW` will misalign every grammar pattern
    /// that contains whitespace or control characters — see F68. For
    /// those tokenizers use [`Self::from_tokenizer`] which delegates to
    /// xgrammar's auto-detection.
    pub fn new(vocab: &[String], stop_token_ids: &[i32]) -> Result<Self, GrammarError> {
        let stop: Option<Box<[i32]>> = if stop_token_ids.is_empty() {
            None
        } else {
            Some(stop_token_ids.to_vec().into_boxed_slice())
        };
        let tokenizer_info = TokenizerInfo::new(vocab, VocabType::RAW, &stop, false)
            .map_err(GrammarError::Compilation)?;
        Self::from_tokenizer_info(tokenizer_info)
    }

    /// Build a [`GrammarEngine`] from a HuggingFace tokenizer with the
    /// vocab type **auto-detected** from the tokenizer's serialized
    /// metadata.
    ///
    /// F68 (2026-04-29): with [`Self::new`] (`VocabType::RAW`), grammar
    /// constraint silently fails on ByteLevel BPE tokenizers because
    /// the matcher checks RAW vocab bytes against the structural-tag
    /// pattern, while the vocab strings are GPT-2 ByteLevel-encoded
    /// (`Ċ` for `\n`, `Ġ` for space, etc.). On MiniMax M2.7 every
    /// pattern containing `\n` or space rejects every token, the
    /// matcher dies silently, and the model freelances into the
    /// `<minimax:tool_call></minimax:tool_call>...` envelope loop.
    ///
    /// We delegate auto-detection to [`detect_metadata_from_hf`] (which
    /// takes the tokenizer's serialized JSON) instead of
    /// `TokenizerInfo::from_huggingface` (which takes a
    /// `tokenizers::Tokenizer` struct directly) — going through the
    /// JSON string side-steps the version skew between Atlas's
    /// `tokenizers = "0.21"` and xgrammar-rs's
    /// `tokenizers = "0.22"` (different `Tokenizer` types).
    pub fn from_tokenizer(
        tokenizer: &tokenizers::Tokenizer,
        vocab_size: Option<usize>,
        stop_token_ids: &[i32],
    ) -> Result<Self, GrammarError> {
        let backend_str = tokenizer
            .to_string(false)
            .map_err(|e| GrammarError::Compilation(format!("tokenizer serialize failed: {e}")))?;
        let metadata = detect_metadata_from_hf(&backend_str).map_err(GrammarError::Compilation)?;
        let detected_label = match metadata.vocab_type {
            VocabType::RAW => "raw",
            VocabType::BYTE_FALLBACK => "byte_fallback",
            VocabType::BYTE_LEVEL => "byte_level",
        };
        tracing::info!(
            "Grammar: detected tokenizer vocab_type={detected_label}, add_prefix_space={}",
            metadata.add_prefix_space,
        );

        let ordered_vocab = extract_ordered_vocab(tokenizer);
        let model_vocab_size = vocab_size.unwrap_or(ordered_vocab.len());
        let mut sized_vocab = if model_vocab_size > ordered_vocab.len() {
            let mut v = ordered_vocab;
            v.resize(model_vocab_size, String::new());
            v
        } else if model_vocab_size < ordered_vocab.len() {
            let mut v = ordered_vocab;
            v.truncate(model_vocab_size);
            v
        } else {
            ordered_vocab
        };
        // Drop the `mut` warning.
        let _ = &mut sized_vocab;

        let stop: Option<Box<[i32]>> = if stop_token_ids.is_empty() {
            None
        } else {
            Some(stop_token_ids.to_vec().into_boxed_slice())
        };
        let tokenizer_info = TokenizerInfo::new(
            &sized_vocab,
            metadata.vocab_type,
            &stop,
            metadata.add_prefix_space,
        )
        .map_err(GrammarError::Compilation)?;
        Self::from_tokenizer_info(tokenizer_info)
    }

    fn from_tokenizer_info(tokenizer_info: TokenizerInfo) -> Result<Self, GrammarError> {
        let vocab_size = tokenizer_info.vocab_size();
        // Single compilation thread, cache enabled, no memory limit.
        let compiler = GrammarCompiler::new(&tokenizer_info, 1, true, -1)
            .map_err(GrammarError::Compilation)?;
        Ok(Self {
            compiler,
            vocab_size,
        })
    }

    /// Vocabulary size (determines bitmask width).
    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}
