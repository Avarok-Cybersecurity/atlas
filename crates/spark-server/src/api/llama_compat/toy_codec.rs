// SPDX-License-Identifier: AGPL-3.0-only

//! A byte-level toy tokenizer for the llama.cpp route tests (test builds only).
//!
//! Id 1 is BOS, id 2 is EOS, id 9 is the special token `<|eot|>`, and id
//! `100 + b` is the single byte `b`. Like byte-level BPE, a multi-byte
//! character spans several ids, so half-character tokens occur.

use super::codec::TextCodec;

pub(crate) const BOS: u32 = 1;
pub(crate) const EOS: u32 = 2;
pub(crate) const EOT: u32 = 9;
const SPECIAL_TEXT: &str = "<|eot|>";

pub(crate) struct ToyCodec;

impl ToyCodec {
    /// Ids for `text`, with no BOS.
    pub(crate) fn ids(text: &str) -> Vec<u32> {
        ToyCodec.encode_text(text, false).expect("toy encode")
    }
}

impl TextCodec for ToyCodec {
    fn encode_text(&self, text: &str, add_special: bool) -> Result<Vec<u32>, String> {
        let mut out = Vec::new();
        if add_special {
            out.push(BOS);
        }
        let mut rest = text;
        while !rest.is_empty() {
            if let Some(tail) = rest.strip_prefix(SPECIAL_TEXT) {
                out.push(EOT);
                rest = tail;
            } else {
                let ch = rest.chars().next().expect("non-empty");
                let mut buf = [0u8; 4];
                out.extend(ch.encode_utf8(&mut buf).bytes().map(|b| 100 + u32::from(b)));
                rest = &rest[ch.len_utf8()..];
            }
        }
        Ok(out)
    }

    fn decode_text(&self, ids: &[u32], skip_special: bool) -> Result<String, String> {
        let mut bytes = Vec::new();
        for &id in ids {
            if id >= 100 {
                bytes.push((id - 100) as u8);
            } else if !skip_special && id == EOT {
                bytes.extend_from_slice(SPECIAL_TEXT.as_bytes());
            }
        }
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    }

    fn vocab_size(&self) -> usize {
        356
    }

    fn token_bytes(&self, id: u32) -> Vec<u8> {
        if id >= 100 {
            vec![(id - 100) as u8]
        } else if id == EOT {
            SPECIAL_TEXT.as_bytes().to_vec()
        } else {
            Vec::new()
        }
    }

    fn decode_incremental(&self, ids: &[u32], prefix: &mut usize, read: &mut usize) -> String {
        let before = self
            .decode_text(&ids[*prefix..*read], true)
            .unwrap_or_default();
        let now = self.decode_text(&ids[*prefix..], true).unwrap_or_default();
        if now.len() > before.len()
            && !now.ends_with('\u{FFFD}')
            && let Some(delta) = now.get(before.len()..)
        {
            let delta = delta.to_string();
            *prefix = *read;
            *read = ids.len();
            return delta;
        }
        String::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn toy_round_trips_ascii_and_splits_multibyte() {
        assert_eq!(ToyCodec::ids("hi"), vec![204, 205]);
        assert_eq!(ToyCodec::ids("\u{20AC}").len(), 3, "euro sign is 3 bytes");
        let ids = ToyCodec::ids("a\u{20AC}b");
        assert_eq!(ToyCodec.decode_text(&ids, true).unwrap(), "a\u{20AC}b");
    }

    #[test]
    fn special_text_becomes_one_id() {
        assert_eq!(ToyCodec::ids("a<|eot|>"), vec![197, EOT]);
        assert_eq!(
            ToyCodec.decode_text(&[197, EOT], false).unwrap(),
            "a<|eot|>"
        );
        assert_eq!(ToyCodec.decode_text(&[197, EOT], true).unwrap(), "a");
    }
}
