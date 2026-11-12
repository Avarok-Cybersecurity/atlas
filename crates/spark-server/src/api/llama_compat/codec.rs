// SPDX-License-Identifier: AGPL-3.0-only

//! The slice of the tokenizer the llama.cpp routes need, behind a trait so the
//! request and response logic can be unit-tested without loading a tokenizer.

use crate::tokenizer::ChatTokenizer;

/// Text <-> token operations used by `/tokenize`, `/detokenize` and
/// `/completion`.
pub(crate) trait TextCodec {
    /// Tokenize `text`. With `add_special`, the tokenizer's post-processor adds
    /// the model's own start-of-text tokens (llama.cpp's `add_special`).
    /// Special-token text inside `text` is always parsed into its token id.
    fn encode_text(&self, text: &str, add_special: bool) -> Result<Vec<u32>, String>;

    /// Detokenize. `skip_special` drops special tokens from the output.
    fn decode_text(&self, ids: &[u32], skip_special: bool) -> Result<String, String>;

    /// Number of ids the model accepts as input.
    fn vocab_size(&self) -> usize;

    /// The raw bytes one token stands for. A token that holds half of a
    /// multi-byte character yields bytes that are not valid UTF-8, which is
    /// why this is bytes and not a string.
    fn token_bytes(&self, id: u32) -> Vec<u8>;

    /// Incremental detokenizer for streaming: returns only the text that became
    /// stable since the last call. Offsets are the caller's state.
    fn decode_incremental(&self, ids: &[u32], prefix: &mut usize, read: &mut usize) -> String;
}

impl TextCodec for ChatTokenizer {
    fn encode_text(&self, text: &str, add_special: bool) -> Result<Vec<u32>, String> {
        self.encode_with_special(text, add_special)
            .map_err(|e| e.to_string())
    }

    fn decode_text(&self, ids: &[u32], skip_special: bool) -> Result<String, String> {
        let r = if skip_special {
            self.decode(ids)
        } else {
            self.decode_with_special(ids)
        };
        r.map_err(|e| e.to_string())
    }

    fn vocab_size(&self) -> usize {
        self.inner().get_vocab_size(true)
    }

    fn token_bytes(&self, id: u32) -> Vec<u8> {
        let decoded = self.decode_with_special(&[id]).unwrap_or_default();
        if !decoded.contains('\u{FFFD}') {
            return decoded.into_bytes();
        }
        // The lone token is half a character and the lossy decode replaced it.
        // Recover the real byte from the vocabulary entry.
        match self.inner().id_to_token(id) {
            Some(raw) => raw_vocab_bytes(&raw).unwrap_or_else(|| decoded.into_bytes()),
            None => decoded.into_bytes(),
        }
    }

    fn decode_incremental(&self, ids: &[u32], prefix: &mut usize, read: &mut usize) -> String {
        self.incremental_decode(ids, prefix, read)
    }
}

/// Bytes behind a vocabulary string that is not valid text on its own: the
/// SentencePiece byte-fallback form `<0xE2>`, or GPT-2 byte-level BPE where
/// each byte was spelled as a printable character.
pub(crate) fn raw_vocab_bytes(raw: &str) -> Option<Vec<u8>> {
    if let Some(hex) = raw.strip_prefix("<0x").and_then(|r| r.strip_suffix('>'))
        && hex.len() == 2
        && let Ok(b) = u8::from_str_radix(hex, 16)
    {
        return Some(vec![b]);
    }
    raw.chars().map(gpt2_unicode_to_byte()).collect()
}

/// Inverse of GPT-2's `bytes_to_unicode`: printable bytes map to themselves,
/// the rest to code points from U+0100 upward, in byte order.
fn gpt2_unicode_to_byte() -> impl Fn(char) -> Option<u8> {
    let printable = |b: u32| (33..=126).contains(&b) || (161..=172).contains(&b) || b >= 174;
    move |c: char| {
        let cp = c as u32;
        if cp < 256 && printable(cp) {
            return Some(cp as u8);
        }
        if cp < 256 {
            return None;
        }
        let mut next = 256u32;
        for b in 0u32..256 {
            if !printable(b) {
                if next == cp {
                    return Some(b as u8);
                }
                next += 1;
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn byte_fallback_token_names_decode() {
        assert_eq!(raw_vocab_bytes("<0xE2>"), Some(vec![0xE2]));
        assert_eq!(raw_vocab_bytes("<0x0A>"), Some(vec![0x0A]));
        // Not a byte-fallback name: spelled as ordinary byte-level characters.
        assert_eq!(raw_vocab_bytes("<0xZZ>"), Some(b"<0xZZ>".to_vec()));
    }

    #[test]
    fn gpt2_byte_level_spelling_decodes() {
        // 'Ġ' (U+0120) is how GPT-2 BPE spells byte 0x20, 'Ċ' (U+010A) 0x0A.
        assert_eq!(raw_vocab_bytes("\u{0120}"), Some(vec![0x20]));
        assert_eq!(raw_vocab_bytes("\u{010A}"), Some(vec![0x0A]));
        assert_eq!(raw_vocab_bytes("\u{0120}Hi"), Some(vec![0x20, b'H', b'i']));
        // 'â' is byte 0xE2 itself (printable range), the lead byte of '€'.
        assert_eq!(raw_vocab_bytes("\u{00E2}"), Some(vec![0xE2]));
    }

    #[test]
    fn a_character_outside_the_table_is_not_guessed() {
        assert_eq!(raw_vocab_bytes("\u{4E2D}"), None);
        // U+00A0 is a control-range byte spelled elsewhere, never itself.
        assert_eq!(raw_vocab_bytes("\u{00A0}"), None);
    }
}
