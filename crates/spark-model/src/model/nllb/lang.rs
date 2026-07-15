// SPDX-License-Identifier: AGPL-3.0-only

//! Translation language configuration for the served NLLB model.
//!
//! NLLB is not a causal chat model: the source must be encoded as
//! `[src_lang] + subwords + </s>` and generation is seeded with
//! `forced_bos = tgt_lang`. The language *strings* (`eng_Latn`, `gvn_Latn`, …)
//! are resolved to token ids by the server's tokenizer (the model has no
//! tokenizer), so this struct carries only the already-resolved ids plus the
//! architectural special tokens.

/// Resolved per-deployment translation tokens. `src_lang_id`/`tgt_lang_id` come
/// from `--src-lang`/`--tgt-lang` (or a recipe default) resolved through the
/// tokenizer at serve start; the rest are M2M-100/NLLB architectural constants.
#[derive(Debug, Clone, Copy)]
pub struct NllbLang {
    /// Source-language prefix token prepended to the encoder input.
    pub src_lang_id: u32,
    /// Target-language token forced as the first decoded token (`forced_bos`).
    pub tgt_lang_id: u32,
    /// Decoder start token (M2M-100 convention: the eos id).
    pub decoder_start_id: u32,
    /// End-of-sequence / stop token.
    pub eos_id: u32,
    /// Padding token id (M2M-100 `padding_idx = 1`).
    pub pad_id: u32,
}

impl NllbLang {
    /// Format the raw source subword ids into the encoder input
    /// `[src_lang] + tokens + </s>` using the deployment-default source language.
    pub(super) fn encoder_input(&self, tokens: &[u32]) -> Vec<u32> {
        self.encoder_input_with(self.src_lang_id, tokens)
    }

    /// Encoder input with an explicit (per-request) source-language token.
    pub(super) fn encoder_input_with(&self, src_lang_id: u32, tokens: &[u32]) -> Vec<u32> {
        let mut ids = Vec::with_capacity(tokens.len() + 2);
        ids.push(src_lang_id);
        ids.extend_from_slice(tokens);
        ids.push(self.eos_id);
        ids
    }
}
