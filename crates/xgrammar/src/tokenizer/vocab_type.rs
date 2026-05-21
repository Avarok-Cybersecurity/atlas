// SPDX-License-Identifier: AGPL-3.0-only
//
// VocabType — port of the `enum class VocabType` in
// `include/xgrammar/tokenizer_info.h`.
//
// The vocabulary type controls how raw (encoded) vocabulary strings are
// turned into byte sequences. `ByteLevel` is the F68-critical type:
// ByteLevel-BPE tokenizers (Qwen, MiniMax, Mistral 4) MUST be detected
// as `ByteLevel` or constrained decoding silently corrupts.

use std::fmt;

/// How a tokenizer encodes its vocabulary strings.
///
/// Discriminants match the C++ `enum class VocabType : int` exactly so
/// that serialized metadata (`vocab_type` integer) round-trips.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum VocabType {
    /// The vocabulary string is the literal token bytes (no decoding).
    Raw = 0,
    /// SentencePiece-style: `<0xHH>` byte tokens and `▁` space markers.
    ByteFallback = 1,
    /// GPT-2 byte-level BPE: each byte is mapped to a printable unicode
    /// codepoint via the bytes-to-unicode table.
    ByteLevel = 2,
}

impl VocabType {
    /// Convert from the integer discriminant used in serialized metadata.
    ///
    /// Returns `None` for any value other than `0`, `1`, `2` — faithful
    /// to the C++ `XGRAMMAR_CHECK(vocab_type_int == 0 || 1 || 2)`.
    pub fn from_int(value: i64) -> Option<Self> {
        match value {
            0 => Some(VocabType::Raw),
            1 => Some(VocabType::ByteFallback),
            2 => Some(VocabType::ByteLevel),
            _ => None,
        }
    }

    /// The integer discriminant used in serialized metadata.
    pub fn as_int(self) -> i64 {
        self as i64
    }
}

impl Default for VocabType {
    /// The C++ constructor defaults `vocab_type` to `VocabType::RAW`.
    fn default() -> Self {
        VocabType::Raw
    }
}

impl fmt::Display for VocabType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            VocabType::Raw => "RAW",
            VocabType::ByteFallback => "BYTE_FALLBACK",
            VocabType::ByteLevel => "BYTE_LEVEL",
        };
        f.write_str(s)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn discriminants_match_cpp() {
        assert_eq!(VocabType::Raw as i64, 0);
        assert_eq!(VocabType::ByteFallback as i64, 1);
        assert_eq!(VocabType::ByteLevel as i64, 2);
    }

    #[test]
    fn from_int_round_trip() {
        for v in [
            VocabType::Raw,
            VocabType::ByteFallback,
            VocabType::ByteLevel,
        ] {
            assert_eq!(VocabType::from_int(v.as_int()), Some(v));
        }
    }

    #[test]
    fn from_int_rejects_invalid() {
        assert_eq!(VocabType::from_int(-1), None);
        assert_eq!(VocabType::from_int(3), None);
        assert_eq!(VocabType::from_int(999), None);
    }

    #[test]
    fn default_is_raw() {
        assert_eq!(VocabType::default(), VocabType::Raw);
    }

    #[test]
    fn display_strings() {
        assert_eq!(VocabType::Raw.to_string(), "RAW");
        assert_eq!(VocabType::ByteFallback.to_string(), "BYTE_FALLBACK");
        assert_eq!(VocabType::ByteLevel.to_string(), "BYTE_LEVEL");
    }
}
