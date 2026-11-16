// SPDX-License-Identifier: AGPL-3.0-only

//! Tokenizer accessors for the llama.cpp-compatible routes (`/tokenize`,
//! `/props`, `/completion`). Kept out of `chat_impl.rs`, which carries the
//! chat-template rendering and is near the file-size cap.

use anyhow::Result;

use super::ChatTokenizer;

impl ChatTokenizer {
    /// Tokenize `text`, optionally letting the tokenizer's post-processor add
    /// the model's own start-of-text tokens (BOS). [`Self::encode`] never adds
    /// them; llama.cpp's `add_special` does.
    pub fn encode_with_special(&self, text: &str, add_special: bool) -> Result<Vec<u32>> {
        let encoding = self
            .tokenizer
            .encode(text, add_special)
            .map_err(|e| anyhow::anyhow!("Tokenizer encode error: {e}"))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// The Jinja chat template this server renders prompts with: the model's
    /// own, or the per-model override when one is installed. This is the
    /// source `/apply-template` renders, so it is what `/props` reports.
    pub fn chat_template_source(&self) -> &str {
        &self.chat_template
    }

    /// The text of the start-of-text token the tokenizer adds on its own, or
    /// an empty string when it adds none (as for ChatML-style models).
    ///
    /// Read off the tokenizer's own behavior (encode with and without special
    /// tokens and compare), not guessed from a list of names.
    pub fn bos_token_text(&self) -> String {
        let plain = self.tokenizer.encode("a", false);
        let with = self.tokenizer.encode("a", true);
        let (Ok(plain), Ok(with)) = (plain, with) else {
            return String::new();
        };
        let (plain, with) = (plain.get_ids(), with.get_ids());
        if with.len() > plain.len() && with.ends_with(plain) {
            let lead = with[0];
            self.tokenizer.id_to_token(lead).unwrap_or_default()
        } else {
            String::new()
        }
    }

    /// The text of the end-of-sequence token this server stops on.
    pub fn eos_token_text(&self) -> String {
        self.tokenizer
            .id_to_token(self.eos_token_id)
            .unwrap_or_default()
    }
}
