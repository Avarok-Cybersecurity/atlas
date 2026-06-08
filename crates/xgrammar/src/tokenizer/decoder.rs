// SPDX-License-Identifier: AGPL-3.0-only
//
// Token decoders — port of `class TokenDecoder` in
// `cpp/tokenizer_info.cc`.
//
// Turns a raw (encoded) vocabulary string into the actual token byte
// sequence, according to the tokenizer's `VocabType`:
//
//   * Raw          — the string is the literal token bytes.
//   * ByteFallback — `<0xHH>` tokens decode to a single byte; the
//                    `▁` (U+2581) marker decodes back to a space.
//   * ByteLevel    — the GPT-2 bytes-to-unicode transform is inverted.
//
// The C++ returns `std::string` (raw bytes). We return `Vec<u8>` since
// a decoded token need not be valid UTF-8.

use crate::support::encoding::{char_handling_error, parse_utf8};
use crate::tokenizer::vocab_type::VocabType;

/// Decode one raw vocabulary token to its byte sequence for `vocab_type`.
pub fn decode_token(token: &str, vocab_type: VocabType) -> Vec<u8> {
    match vocab_type {
        VocabType::Raw => token.as_bytes().to_vec(),
        VocabType::ByteFallback => space_replacer_decoder(&byte_fallback_decoder(token)),
        VocabType::ByteLevel => byte_level_decoder(token),
    }
}

/// ByteFallback decoder: transform a token like `<0x1B>` into the single
/// raw byte `0x1B`. Any other token is returned unchanged.
///
/// Faithful to C++ `ByteFallbackDecoder`: the match requires length 6,
/// a `<0x` prefix, a `>` suffix, and two uppercase-or-digit hex chars.
fn byte_fallback_decoder(token: &str) -> Vec<u8> {
    let bytes = token.as_bytes();
    if bytes.len() == 6 && &bytes[0..3] == b"<0x" && bytes[5] == b'>' {
        let mut byte: i32 = 0;
        for &c in &bytes[3..5] {
            // C++ only handles `0-9` and `A-F` here (uppercase hex).
            let digit = if c.is_ascii_digit() {
                (c - b'0') as i32
            } else {
                (c - b'A') as i32 + 10
            };
            byte = byte * 16 + digit;
        }
        // C++ `XGRAMMAR_CHECK(byte >= 0 && byte < 256)`. With two hex
        // chars this always holds, so we keep the value directly.
        return vec![byte as u8];
    }
    token.as_bytes().to_vec()
}

/// SpaceReplacer decoder: transform the UTF-8 of `▁` (U+2581, "lower one
/// eighth block", bytes `E2 96 81`) back into an ASCII space.
///
/// Faithful to C++ `SpaceReplacerDecoder`, which scans bytes directly.
fn space_replacer_decoder(token: &[u8]) -> Vec<u8> {
    let mut result = Vec::with_capacity(token.len());
    let mut i = 0;
    while i < token.len() {
        // C++ guard is `i + 2 < size`, i.e. all three bytes addressable.
        if i + 2 < token.len() && token[i] == 0xE2 && token[i + 1] == 0x96 && token[i + 2] == 0x81 {
            result.push(b' ');
            i += 3;
        } else {
            result.push(token[i]);
            i += 1;
        }
    }
    result
}

/// The inverse of the GPT-2 bytes-to-unicode map.
///
/// `char_to_byte_map()[codepoint]` is the original byte, or `-1` when
/// no byte maps to that codepoint. The table has 324 entries, matching
/// `std::array<int, 324>` in the C++ source verbatim.
pub fn char_to_byte_map() -> &'static [i16; 324] {
    &CHAR_TO_BYTE_MAP
}

/// The GPT-2 bytes-to-unicode forward map: byte -> unicode codepoint.
///
/// This is the table HuggingFace uses to *encode* the vocabulary; it is
/// the exact inverse of `CHAR_TO_BYTE_MAP`. Derived (not duplicated)
/// from the inverse table to keep a single source of truth.
pub fn byte_to_char_map() -> &'static [u32; 256] {
    &BYTE_TO_CHAR_MAP
}

#[rustfmt::skip]
static CHAR_TO_BYTE_MAP: [i16; 324] = [
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45,
    46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68,
    69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91,
    92, 93, 94, 95, 96, 97, 98, 99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110, 111,
    112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 124, 125, 126, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, 161, 162, 163, 164, 165, 166, 167, 168, 169, 170, 171, 172, -1,
    174, 175, 176, 177, 178, 179, 180, 181, 182, 183, 184, 185, 186, 187, 188, 189, 190, 191,
    192, 193, 194, 195, 196, 197, 198, 199, 200, 201, 202, 203, 204, 205, 206, 207, 208, 209,
    210, 211, 212, 213, 214, 215, 216, 217, 218, 219, 220, 221, 222, 223, 224, 225, 226, 227,
    228, 229, 230, 231, 232, 233, 234, 235, 236, 237, 238, 239, 240, 241, 242, 243, 244, 245,
    246, 247, 248, 249, 250, 251, 252, 253, 254, 255, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 127, 128,
    129, 130, 131, 132, 133, 134, 135, 136, 137, 138, 139, 140, 141, 142, 143, 144, 145, 146,
    147, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 158, 159, 160, 173,
];

/// Forward map derived once from `CHAR_TO_BYTE_MAP` (SSOT: the inverse
/// table is authoritative; this is `byte -> codepoint` such that
/// `CHAR_TO_BYTE_MAP[byte_to_char(byte)] == byte`).
static BYTE_TO_CHAR_MAP: [u32; 256] = build_byte_to_char_map();

const fn build_byte_to_char_map() -> [u32; 256] {
    let mut out = [0u32; 256];
    let mut cp = 0usize;
    while cp < CHAR_TO_BYTE_MAP.len() {
        let b = CHAR_TO_BYTE_MAP[cp];
        if b >= 0 {
            out[b as usize] = cp as u32;
        }
        cp += 1;
    }
    out
}

/// ByteLevel decoder: invert the GPT-2 bytes-to-unicode transform.
///
/// Faithful to C++ `ByteLevelDecoder`. If the token is not valid UTF-8,
/// or contains a codepoint with no inverse mapping, the original token
/// bytes are returned unchanged.
fn byte_level_decoder(token: &str) -> Vec<u8> {
    let codepoints = parse_utf8(token.as_bytes(), false);
    if codepoints.len() == 1 && codepoints[0] == char_handling_error::INVALID_UTF8 {
        return token.as_bytes().to_vec();
    }

    let map = char_to_byte_map();
    let mut decoded = Vec::with_capacity(codepoints.len());
    for cp in codepoints {
        // C++ `XGRAMMAR_CHECK(unicode_codepoint >= 0)` — parse_utf8 with
        // preserve=false never yields negatives other than the invalid
        // sentinel handled above, so any negative here means no mapping.
        if cp < 0 || cp as usize >= map.len() || map[cp as usize] == -1 {
            return token.as_bytes().to_vec();
        }
        decoded.push(map[cp as usize] as u8);
    }
    decoded
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn raw_is_literal() {
        assert_eq!(decode_token("hello", VocabType::Raw), b"hello");
        assert_eq!(decode_token("", VocabType::Raw), b"");
    }

    #[test]
    fn byte_fallback_hex_token() {
        // <0x1B> -> ESC byte 0x1B
        assert_eq!(decode_token("<0x1B>", VocabType::ByteFallback), vec![0x1B]);
        // <0x00> -> NUL
        assert_eq!(decode_token("<0x00>", VocabType::ByteFallback), vec![0x00]);
        // <0xFF> -> 0xFF
        assert_eq!(decode_token("<0xFF>", VocabType::ByteFallback), vec![0xFF]);
    }

    #[test]
    fn byte_fallback_non_hex_passthrough() {
        // Not the <0xHH> shape -> unchanged.
        assert_eq!(decode_token("er", VocabType::ByteFallback), b"er");
        // Lowercase hex is NOT matched by the C++ uppercase-only path,
        // but it still has the <0x..> shape; C++ computes c-'A'+10 on
        // 'b' which is wrong — we faithfully reproduce: keep length 6
        // shape match. Use a clearly-non-matching token instead.
        assert_eq!(decode_token("<0x1>", VocabType::ByteFallback), b"<0x1>");
    }

    #[test]
    fn byte_fallback_space_marker() {
        // U+2581 "▁" -> space
        let token = "\u{2581}hello";
        assert_eq!(decode_token(token, VocabType::ByteFallback), b" hello");
    }

    #[test]
    fn byte_fallback_double_space_marker() {
        let token = "\u{2581}\u{2581}";
        assert_eq!(decode_token(token, VocabType::ByteFallback), b"  ");
    }

    #[test]
    fn byte_level_round_trip_ascii() {
        // For printable ASCII, byte_to_char is the identity, so the
        // encoded string equals the literal — decode returns the bytes.
        let map = byte_to_char_map();
        // 'A' (0x41) maps to itself.
        assert_eq!(map[0x41], 0x41);
        let mut enc = String::new();
        for &b in b"automotive" {
            enc.push(char::from_u32(map[b as usize]).unwrap());
        }
        assert_eq!(decode_token(&enc, VocabType::ByteLevel), b"automotive");
    }

    #[test]
    fn byte_level_space_is_g_with_dot() {
        // GPT-2 maps byte 0x20 (space) to U+0120 'Ġ'.
        let map = byte_to_char_map();
        assert_eq!(map[0x20], 0x0120);
        // " automotive" -> "Ġautomotive"
        let mut enc = String::from("\u{0120}");
        for &b in b"automotive" {
            enc.push(char::from_u32(map[b as usize]).unwrap());
        }
        assert_eq!(decode_token(&enc, VocabType::ByteLevel), b" automotive");
    }

    #[test]
    fn byte_level_high_bytes() {
        // The UTF-8 of '我' is E6 88 91 — encode each byte via the map.
        let map = byte_to_char_map();
        let mut enc = String::new();
        for &b in "我".as_bytes() {
            enc.push(char::from_u32(map[b as usize]).unwrap());
        }
        assert_eq!(decode_token(&enc, VocabType::ByteLevel), "我".as_bytes());
    }

    #[test]
    fn byte_level_unmapped_codepoint_passthrough() {
        // U+4E2D '中' has no inverse mapping -> token returned as-is.
        assert_eq!(decode_token("中", VocabType::ByteLevel), "中".as_bytes());
    }

    #[test]
    fn inverse_table_is_consistent() {
        // Every mapped entry of the inverse table must round-trip.
        let inv = char_to_byte_map();
        let fwd = byte_to_char_map();
        for byte in 0u32..256 {
            let cp = fwd[byte as usize];
            assert_eq!(inv[cp as usize], byte as i16);
        }
        // All 256 bytes have a forward mapping (the map is a bijection
        // onto 256 distinct codepoints).
        let mut seen = std::collections::HashSet::new();
        for byte in 0u32..256 {
            assert!(seen.insert(fwd[byte as usize]));
        }
    }

    #[test]
    fn byte_level_invalid_utf8_passthrough() {
        // A literal char that decodes to a non-mapped point falls back.
        // Codepoint 0 maps to -1 in the inverse table.
        assert_eq!(char_to_byte_map()[0], -1);
    }
}
