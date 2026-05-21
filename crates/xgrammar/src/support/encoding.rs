// SPDX-License-Identifier: AGPL-3.0-only
//
// UTF-8 codepoint encode/decode + hex + Latin-1 — port of the UTF-8
// half of `cpp/support/encoding.h`. Escape handling lives in the
// sibling `escape` module.
//
// The C++ uses `int32_t` codepoints with negative sentinel values for
// errors. We keep the same `TCodepoint` alias and `CharHandlingError`
// values for faithful behavior; decoding APIs additionally expose a
// Rust-idiomatic `Result`-free `Option` form where useful.

/// A Unicode codepoint, matching C++ `TCodepoint = int32_t`.
pub type TCodepoint = i32;

/// Error sentinels returned in-band as negative [`TCodepoint`] values,
/// matching C++ `enum CharHandlingError`.
pub mod char_handling_error {
    use super::TCodepoint;
    /// The UTF-8 string is invalid.
    pub const INVALID_UTF8: TCodepoint = -10;
    /// The escape sequence is invalid.
    pub const INVALID_ESCAPE: TCodepoint = -11;
    /// The Latin-1 string is invalid.
    pub const INVALID_LATIN1: TCodepoint = -12;
}

/// Encode a codepoint into UTF-8 bytes.
///
/// Faithful to C++ `CharToUTF8`: it does not validate surrogates and
/// produces 1–4 bytes for codepoints in `0..=0x10FFFF`. Codepoints
/// above `0x10FFFF` are debug-checked in C++; here we still encode
/// using the 4-byte form (the low 21 bits), never panicking.
pub fn char_to_utf8(codepoint: TCodepoint) -> Vec<u8> {
    let cp = codepoint as u32;
    let mut out = Vec::with_capacity(4);
    if cp <= 0x7F {
        out.push(cp as u8);
    } else if cp <= 0x7FF {
        out.push(0xC0 | ((cp >> 6) & 0x1F) as u8);
        out.push(0x80 | (cp & 0x3F) as u8);
    } else if cp <= 0xFFFF {
        out.push(0xE0 | ((cp >> 12) & 0x0F) as u8);
        out.push(0x80 | ((cp >> 6) & 0x3F) as u8);
        out.push(0x80 | (cp & 0x3F) as u8);
    } else {
        out.push(0xF0 | ((cp >> 18) & 0x07) as u8);
        out.push(0x80 | ((cp >> 12) & 0x3F) as u8);
        out.push(0x80 | ((cp >> 6) & 0x3F) as u8);
        out.push(0x80 | (cp & 0x3F) as u8);
    }
    out
}

/// Lookup table: number of bytes a UTF-8 sequence starting with each
/// possible first byte spans, or `-1` for an invalid first byte.
const UTF8_BYTES: [i8; 256] = build_utf8_bytes();

const fn build_utf8_bytes() -> [i8; 256] {
    let mut t = [0i8; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = if i < 0x80 {
            1
        } else if i < 0xC0 {
            -1
        } else if i < 0xE0 {
            2
        } else if i < 0xF0 {
            3
        } else if i < 0xF8 {
            4
        } else {
            -1
        };
        i += 1;
    }
    t
}

/// Mask applied to the first byte to extract its codepoint bits,
/// indexed by sequence length (`0` slot unused).
const FIRST_BYTE_MASK: [u8; 5] = [0x00, 0x7F, 0x1F, 0x0F, 0x07];

/// Inspect a UTF-8 first byte.
///
/// Returns `(is_valid, total_byte_count, initial_codepoint_bits)`,
/// matching C++ `HandleUTF8FirstByte`.
pub fn handle_utf8_first_byte(byte: u8) -> (bool, i32, TCodepoint) {
    let num_bytes = UTF8_BYTES[byte as usize];
    if num_bytes == -1 {
        return (false, 0, 0);
    }
    (
        true,
        num_bytes as i32,
        (byte & FIRST_BYTE_MASK[num_bytes as usize]) as TCodepoint,
    )
}

/// Decode the first codepoint from a UTF-8 byte slice.
///
/// Returns `(codepoint, bytes_consumed)`. On invalid UTF-8 (bad first
/// byte, missing/invalid continuation byte, or truncated input)
/// returns `(INVALID_UTF8, 0)` — faithful to C++ `ParseNextUTF8`.
pub fn parse_next_utf8(utf8: &[u8]) -> (TCodepoint, i32) {
    if utf8.is_empty() {
        return (char_handling_error::INVALID_UTF8, 0);
    }
    let (mut accepted, num_bytes, mut res) = handle_utf8_first_byte(utf8[0]);
    if accepted {
        for i in 1..num_bytes as usize {
            // C++ treats a NUL byte (string terminator) or any byte
            // whose top bits are not `10` as invalid. A short slice is
            // likewise invalid (the C++ string would have terminated).
            if i >= utf8.len() || utf8[i] == 0 || (utf8[i] & 0xC0) != 0x80 {
                accepted = false;
                break;
            }
            res = (res << 6) | (utf8[i] & 0x3F) as TCodepoint;
        }
    }
    if !accepted {
        return (char_handling_error::INVALID_UTF8, 0);
    }
    (res, num_bytes)
}

/// Decode every codepoint in a UTF-8 byte slice.
///
/// When `preserve_invalid_bytes` is `false` (the C++ default), an
/// invalid byte aborts decoding and the result is the single-element
/// vector `[INVALID_UTF8]`. When `true`, each invalid byte is pushed
/// as its raw `u8` value (widened to `TCodepoint`) and decoding
/// continues. Faithful to C++ `ParseUTF8`. Decoding stops at the first
/// NUL byte, matching the C-string semantics of the original.
pub fn parse_utf8(utf8: &[u8], preserve_invalid_bytes: bool) -> Vec<TCodepoint> {
    let mut codepoints = Vec::new();
    let mut pos = 0;
    while pos < utf8.len() && utf8[pos] != 0 {
        let (codepoint, num_bytes) = parse_next_utf8(&utf8[pos..]);
        if codepoint == char_handling_error::INVALID_UTF8 {
            if preserve_invalid_bytes {
                codepoints.push(utf8[pos] as TCodepoint);
                pos += 1;
                continue;
            } else {
                return vec![char_handling_error::INVALID_UTF8];
            }
        }
        codepoints.push(codepoint);
        pos += num_bytes as usize;
    }
    codepoints
}

/// Convert a hex digit (`0-9`, `a-f`, `A-F`) to its value, or `-1` if
/// not a hex digit. Faithful to C++ `HexCharToInt`.
pub fn hex_char_to_int(c: u8) -> i32 {
    match c {
        b'0'..=b'9' => (c - b'0') as i32,
        b'a'..=b'f' => (c - b'a' + 10) as i32,
        b'A'..=b'F' => (c - b'A' + 10) as i32,
        _ => -1,
    }
}

/// Convert a Latin-1 string (given as bytes that are themselves valid
/// UTF-8) into the raw Latin-1 byte sequence.
///
/// Every codepoint must be in `0x00..=0xFF`; ASCII passes through and
/// `0x80..=0xFF` is expected as a 2-byte UTF-8 sequence. Returns
/// `Err(INVALID_LATIN1)` on any out-of-range or malformed input —
/// faithful to C++ `Latin1ToBytes`.
pub fn latin1_to_bytes(latin1: &[u8]) -> Result<Vec<u8>, TCodepoint> {
    let mut result = Vec::with_capacity(latin1.len());
    let mut i = 0;
    while i < latin1.len() {
        let c1 = latin1[i];
        if c1 < 0x80 {
            result.push(c1);
            i += 1;
        } else {
            if i + 1 >= latin1.len() {
                return Err(char_handling_error::INVALID_LATIN1);
            }
            let c2 = latin1[i + 1];
            if (c2 & 0xC0) != 0x80 {
                return Err(char_handling_error::INVALID_LATIN1);
            }
            let code = (((c1 & 0x1F) as i32) << 6) | (c2 & 0x3F) as i32;
            if !(0x80..=0xFF).contains(&code) {
                return Err(char_handling_error::INVALID_LATIN1);
            }
            result.push(code as u8);
            i += 2;
        }
    }
    Ok(result)
}

/// Convert a raw byte sequence into a Latin-1 string encoded as UTF-8.
///
/// Bytes `0x00..=0x7F` pass through; `0x80..=0xFF` are emitted as the
/// 2-byte UTF-8 encoding of the same codepoint. Faithful to C++
/// `ByteToLatin1`. Stops at the first NUL byte, matching the original.
pub fn byte_to_latin1(bytes: &[u8]) -> Vec<u8> {
    let mut result = Vec::new();
    for &b in bytes {
        if b == 0 {
            break;
        }
        if b <= 0x7F {
            result.push(b);
        } else {
            result.push(0xC0 | (b >> 6));
            result.push(0x80 | (b & 0x3F));
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::char_handling_error::*;
    use super::*;

    #[test]
    fn encode_one_byte() {
        assert_eq!(char_to_utf8(0x41), vec![0x41]);
        assert_eq!(char_to_utf8(0x00), vec![0x00]);
        assert_eq!(char_to_utf8(0x7F), vec![0x7F]);
    }

    #[test]
    fn encode_two_byte() {
        // U+00A9 ©
        assert_eq!(char_to_utf8(0xA9), vec![0xC2, 0xA9]);
        // U+07FF boundary
        assert_eq!(char_to_utf8(0x7FF), vec![0xDF, 0xBF]);
    }

    #[test]
    fn encode_three_byte() {
        // U+2603 ☃
        assert_eq!(char_to_utf8(0x2603), vec![0xE2, 0x98, 0x83]);
        // U+FFFF boundary
        assert_eq!(char_to_utf8(0xFFFF), vec![0xEF, 0xBF, 0xBF]);
    }

    #[test]
    fn encode_four_byte() {
        // U+1F600 😀
        assert_eq!(char_to_utf8(0x1F600), vec![0xF0, 0x9F, 0x98, 0x80]);
        // U+10FFFF max codepoint
        assert_eq!(char_to_utf8(0x10FFFF), vec![0xF4, 0x8F, 0xBF, 0xBF]);
    }

    #[test]
    fn encode_decode_round_trip() {
        for cp in [0x41, 0xA9, 0x2603, 0x1F600, 0x6D4B, 0x8BD5] {
            let bytes = char_to_utf8(cp);
            let (decoded, n) = parse_next_utf8(&bytes);
            assert_eq!(decoded, cp);
            assert_eq!(n as usize, bytes.len());
        }
    }

    #[test]
    fn first_byte_classification() {
        assert_eq!(handle_utf8_first_byte(0x41), (true, 1, 0x41));
        assert!(handle_utf8_first_byte(0xC2).0);
        assert_eq!(handle_utf8_first_byte(0xC2).1, 2);
        assert_eq!(handle_utf8_first_byte(0xE2).1, 3);
        assert_eq!(handle_utf8_first_byte(0xF0).1, 4);
        // Continuation byte cannot start a sequence.
        assert_eq!(handle_utf8_first_byte(0x80), (false, 0, 0));
        // 0xF8.. is invalid.
        assert_eq!(handle_utf8_first_byte(0xFF), (false, 0, 0));
    }

    #[test]
    fn decode_empty_is_invalid() {
        assert_eq!(parse_next_utf8(&[]), (INVALID_UTF8, 0));
    }

    #[test]
    fn decode_invalid_first_byte() {
        assert_eq!(parse_next_utf8(&[0x80]), (INVALID_UTF8, 0));
    }

    #[test]
    fn decode_bad_continuation() {
        // 0xC2 followed by 0x20 (not a continuation byte).
        assert_eq!(parse_next_utf8(&[0xC2, 0x20]), (INVALID_UTF8, 0));
    }

    #[test]
    fn decode_truncated_sequence() {
        // 0xE2 expects 3 bytes but only 1 is present.
        assert_eq!(parse_next_utf8(&[0xE2]), (INVALID_UTF8, 0));
    }

    #[test]
    fn parse_utf8_string() {
        let s = "UTF-8: © ☃ 😀".as_bytes();
        let cps = parse_utf8(s, false);
        assert!(cps.contains(&0xA9));
        assert!(cps.contains(&0x2603));
        assert!(cps.contains(&0x1F600));
    }

    #[test]
    fn parse_utf8_invalid_aborts() {
        let cps = parse_utf8(&[0x41, 0xC2, 0x20], false);
        assert_eq!(cps, vec![INVALID_UTF8]);
    }

    #[test]
    fn parse_utf8_preserve_invalid() {
        let cps = parse_utf8(&[0x41, 0x80, 0x42], true);
        assert_eq!(cps, vec![0x41, 0x80, 0x42]);
    }

    #[test]
    fn parse_utf8_empty() {
        assert_eq!(parse_utf8(&[], false), Vec::<TCodepoint>::new());
    }

    #[test]
    fn parse_utf8_stops_at_nul() {
        let cps = parse_utf8(&[0x41, 0x00, 0x42], false);
        assert_eq!(cps, vec![0x41]);
    }

    #[test]
    fn hex_digits() {
        assert_eq!(hex_char_to_int(b'0'), 0);
        assert_eq!(hex_char_to_int(b'9'), 9);
        assert_eq!(hex_char_to_int(b'a'), 10);
        assert_eq!(hex_char_to_int(b'f'), 15);
        assert_eq!(hex_char_to_int(b'A'), 10);
        assert_eq!(hex_char_to_int(b'F'), 15);
        assert_eq!(hex_char_to_int(b'g'), -1);
        assert_eq!(hex_char_to_int(b' '), -1);
    }

    #[test]
    fn latin1_round_trip() {
        // 0xFF -> two UTF-8 bytes -> back to 0xFF
        let utf8 = byte_to_latin1(&[0xFF]);
        assert_eq!(utf8, vec![0xC3, 0xBF]);
        assert_eq!(latin1_to_bytes(&utf8), Ok(vec![0xFF]));
    }

    #[test]
    fn latin1_ascii_passthrough() {
        assert_eq!(byte_to_latin1(b"Abc"), b"Abc".to_vec());
        assert_eq!(latin1_to_bytes(b"Abc"), Ok(b"Abc".to_vec()));
    }

    #[test]
    fn latin1_invalid_truncated() {
        assert_eq!(latin1_to_bytes(&[0xC3]), Err(INVALID_LATIN1));
    }

    #[test]
    fn latin1_invalid_bad_continuation() {
        assert_eq!(latin1_to_bytes(&[0xC3, 0x20]), Err(INVALID_LATIN1));
    }

    #[test]
    fn latin1_invalid_out_of_range() {
        // 0xC4 0x80 = U+0100, just above the Latin-1 range 0x00..=0xFF.
        // (c1 & 0x1F)=0x04, <<6 = 0x100, | (0x80 & 0x3F)=0 -> 0x100.
        assert_eq!(latin1_to_bytes(&[0xC4, 0x80]), Err(INVALID_LATIN1));
        // 0xC2 0x40: 0x40 is not a valid continuation byte.
        assert_eq!(latin1_to_bytes(&[0xC2, 0x40]), Err(INVALID_LATIN1));
    }

    #[test]
    fn byte_to_latin1_stops_at_nul() {
        assert_eq!(byte_to_latin1(&[0x41, 0x00, 0x42]), vec![0x41]);
    }
}
