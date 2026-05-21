// SPDX-License-Identifier: AGPL-3.0-only
//
// Grammar-literal escape handling — port of the escape half of
// `cpp/support/encoding.h` (`EscapeString` / `ParseNextEscaped` /
// `ParseNextUTF8OrEscaped`).
//
// `print_as_escaped` turns a codepoint (or string) into a printable
// escaped form; `parse_next_escaped` reverses a single `\…` sequence;
// `parse_next_utf8_or_escaped` dispatches between a raw UTF-8 char and
// an escape sequence.

use super::encoding::{
    TCodepoint, char_handling_error, char_to_utf8, hex_char_to_int, parse_next_utf8, parse_utf8,
};
use std::collections::HashMap;

/// C-style escape table: codepoint -> escaped text. Matches the C++
/// `kCodepointToEscape` static map.
fn codepoint_to_escape(cp: TCodepoint) -> Option<&'static str> {
    Some(match cp {
        0x27 => "\\'",  // '
        0x22 => "\\\"", // "
        0x3F => "\\?",  // ?
        0x5C => "\\\\", // backslash
        0x07 => "\\a",
        0x08 => "\\b",
        0x0C => "\\f",
        0x0A => "\\n",
        0x0D => "\\r",
        0x09 => "\\t",
        0x0B => "\\v",
        0x00 => "\\0",
        0x1B => "\\e",
        _ => return None,
    })
}

/// Reverse of [`codepoint_to_escape`]: the escape letter following a
/// backslash -> codepoint. Matches the C++ `kEscapeToCodepoint` map.
fn escape_to_codepoint(c: u8) -> Option<TCodepoint> {
    Some(match c {
        b'\'' => 0x27,
        b'"' => 0x22,
        b'?' => 0x3F,
        b'\\' => 0x5C,
        // `\/` -> `/`: keeps the EBNF lexer in sync with JSON's relaxed
        // escape rule (upstream commit 5d65108, #626).
        b'/' => 0x2F,
        b'a' => 0x07,
        b'b' => 0x08,
        b'f' => 0x0C,
        b'n' => 0x0A,
        b'r' => 0x0D,
        b't' => 0x09,
        b'v' => 0x0B,
        b'0' => 0x00,
        b'e' => 0x1B,
        _ => return None,
    })
}

/// Render a single codepoint as a printable, escaped string.
///
/// `additional_escape_map` is checked first (e.g. `{'-': "\\-"}`),
/// then the C-escape table, then a printable-ASCII passthrough for
/// `0x20..=0x7E`; anything else becomes a `\xHH` / `\uHHHH` / `\UHHHHHHHH`
/// hex escape sized by magnitude. Faithful to C++ `EscapeString`.
pub fn print_as_escaped(
    codepoint: TCodepoint,
    additional_escape_map: &HashMap<TCodepoint, String>,
) -> String {
    if let Some(s) = additional_escape_map.get(&codepoint) {
        return s.clone();
    }
    if let Some(s) = codepoint_to_escape(codepoint) {
        return s.to_string();
    }
    if (0x20..=0x7E).contains(&codepoint) {
        return (codepoint as u8 as char).to_string();
    }
    // Hex escape, width and prefix selected by magnitude.
    let (prefix, width) = if codepoint <= 0xFF {
        ('x', 2)
    } else if codepoint <= 0xFFFF {
        ('u', 4)
    } else {
        ('U', 8)
    };
    format!("\\{}{:0width$x}", prefix, codepoint, width = width)
}

/// Render a single raw byte as a printable escaped string (the
/// `EscapeString(uint8_t)` overload).
pub fn print_byte_as_escaped(raw_char: u8) -> String {
    print_as_escaped(raw_char as TCodepoint, &HashMap::new())
}

/// Render an entire byte string as a printable escaped string (the
/// `EscapeString(std::string)` overload). Invalid UTF-8 bytes are
/// preserved and escaped individually.
pub fn print_str_as_escaped(raw: &[u8]) -> String {
    let mut res = String::new();
    for cp in parse_utf8(raw, true) {
        res.push_str(&print_as_escaped(cp, &HashMap::new()));
    }
    res
}

/// Parse one escape sequence at the start of `data` (which must begin
/// with `\`).
///
/// Returns `(codepoint, bytes_consumed)`. Supports the C-escape table,
/// `additional_escape_map` (checked first), arbitrary-length `\xHH…`,
/// fixed `\uHHHH`, and fixed `\UHHHHHHHH`. On any malformed input
/// returns `(INVALID_ESCAPE, 0)` — faithful to C++ `ParseNextEscaped`.
pub fn parse_next_escaped(
    data: &[u8],
    additional_escape_map: &HashMap<u8, TCodepoint>,
) -> (TCodepoint, i32) {
    if data.is_empty() || data[0] != b'\\' {
        return (char_handling_error::INVALID_ESCAPE, 0);
    }
    // The C++ reads data[1] unconditionally; on a bare trailing '\'
    // the C string's NUL terminator is read. We treat a missing
    // second byte as invalid.
    if data.len() < 2 {
        return (char_handling_error::INVALID_ESCAPE, 0);
    }
    let esc = data[1];
    // C++ rejects escape chars whose value exceeds 128.
    if (esc as i32) > 128 {
        return (char_handling_error::INVALID_ESCAPE, 0);
    }
    if let Some(&cp) = additional_escape_map.get(&esc) {
        return (cp, 2);
    }
    if let Some(cp) = escape_to_codepoint(esc) {
        return (cp, 2);
    }
    match esc {
        b'x' => {
            // Arbitrary-length hex.
            let mut len = 0usize;
            let mut codepoint: TCodepoint = 0;
            while 2 + len < data.len() {
                let digit = hex_char_to_int(data[2 + len]);
                if digit == -1 {
                    break;
                }
                codepoint = codepoint * 16 + digit;
                len += 1;
            }
            if len == 0 {
                return (char_handling_error::INVALID_ESCAPE, 0);
            }
            (codepoint, (len + 2) as i32)
        }
        b'u' | b'U' => {
            let len = if esc == b'u' { 4 } else { 8 };
            let mut codepoint: TCodepoint = 0;
            for i in 0..len {
                if 2 + i >= data.len() {
                    return (char_handling_error::INVALID_ESCAPE, 0);
                }
                let digit = hex_char_to_int(data[2 + i]);
                if digit == -1 {
                    return (char_handling_error::INVALID_ESCAPE, 0);
                }
                codepoint = codepoint * 16 + digit;
            }
            (codepoint, (len + 2) as i32)
        }
        _ => (char_handling_error::INVALID_ESCAPE, 0),
    }
}

/// Parse the first codepoint from `utf8`, transparently decoding an
/// escape sequence when the input starts with `\`.
///
/// Returns `(codepoint, bytes_consumed)`. On invalid input returns
/// `(INVALID_UTF8, 0)` or `(INVALID_ESCAPE, 0)` per the failure mode —
/// faithful to C++ `ParseNextUTF8OrEscaped`.
pub fn parse_next_utf8_or_escaped(
    utf8: &[u8],
    additional_escape_map: &HashMap<u8, TCodepoint>,
) -> (TCodepoint, i32) {
    if utf8.is_empty() || utf8[0] != b'\\' {
        return parse_next_utf8(utf8);
    }
    parse_next_escaped(utf8, additional_escape_map)
}

/// Decode an escaped grammar-literal byte string into its UTF-8 form.
///
/// Convenience helper layered on [`parse_next_utf8_or_escaped`]:
/// walks the whole input, decoding each char or escape and re-encoding
/// the resulting codepoints to UTF-8. Returns `Err` with the offending
/// error sentinel on the first malformed char/escape.
pub fn unescape_string(
    data: &[u8],
    additional_escape_map: &HashMap<u8, TCodepoint>,
) -> Result<Vec<u8>, TCodepoint> {
    let mut out = Vec::new();
    let mut pos = 0;
    while pos < data.len() {
        let (cp, n) = parse_next_utf8_or_escaped(&data[pos..], additional_escape_map);
        if n == 0 {
            return Err(cp);
        }
        out.extend_from_slice(&char_to_utf8(cp));
        pos += n as usize;
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::char_handling_error::*;
    use super::*;

    fn empty_cp_map() -> HashMap<TCodepoint, String> {
        HashMap::new()
    }
    fn empty_esc_map() -> HashMap<u8, TCodepoint> {
        HashMap::new()
    }

    #[test]
    fn escape_c_specials() {
        assert_eq!(print_as_escaped(0x0A, &empty_cp_map()), "\\n");
        assert_eq!(print_as_escaped(0x09, &empty_cp_map()), "\\t");
        assert_eq!(print_as_escaped(0x0D, &empty_cp_map()), "\\r");
        assert_eq!(print_as_escaped(0x5C, &empty_cp_map()), "\\\\");
        assert_eq!(print_as_escaped(0x00, &empty_cp_map()), "\\0");
        assert_eq!(print_as_escaped(0x1B, &empty_cp_map()), "\\e");
    }

    #[test]
    fn escape_printable_passthrough() {
        assert_eq!(print_as_escaped(0x41, &empty_cp_map()), "A");
        assert_eq!(print_as_escaped(0x20, &empty_cp_map()), " ");
        assert_eq!(print_as_escaped(0x7E, &empty_cp_map()), "~");
    }

    #[test]
    fn escape_hex_forms() {
        // Non-printable low byte -> \xHH
        assert_eq!(print_as_escaped(0x01, &empty_cp_map()), "\\x01");
        assert_eq!(print_as_escaped(0xFF, &empty_cp_map()), "\\xff");
        // BMP -> \uHHHH
        assert_eq!(print_as_escaped(0x2603, &empty_cp_map()), "\\u2603");
        // Astral -> \UHHHHHHHH
        assert_eq!(print_as_escaped(0x1F600, &empty_cp_map()), "\\U0001f600");
    }

    #[test]
    fn escape_additional_map_wins() {
        let mut m = empty_cp_map();
        m.insert('-' as TCodepoint, "\\-".to_string());
        assert_eq!(print_as_escaped('-' as TCodepoint, &m), "\\-");
    }

    #[test]
    fn escape_byte_overload() {
        assert_eq!(print_byte_as_escaped(b'A'), "A");
        assert_eq!(print_byte_as_escaped(0x0A), "\\n");
        assert_eq!(print_byte_as_escaped(0xFF), "\\xff");
    }

    #[test]
    fn escape_whole_string() {
        assert_eq!(print_str_as_escaped(b"a\nb"), "a\\nb");
        // Invalid byte preserved + hex-escaped.
        assert_eq!(print_str_as_escaped(&[0x41, 0x80]), "A\\x80");
    }

    #[test]
    fn parse_c_escapes() {
        assert_eq!(parse_next_escaped(b"\\n", &empty_esc_map()), (0x0A, 2));
        assert_eq!(parse_next_escaped(b"\\t", &empty_esc_map()), (0x09, 2));
        assert_eq!(parse_next_escaped(b"\\r", &empty_esc_map()), (0x0D, 2));
        assert_eq!(parse_next_escaped(b"\\\\", &empty_esc_map()), (0x5C, 2));
        assert_eq!(parse_next_escaped(b"\\b", &empty_esc_map()), (0x08, 2));
        assert_eq!(parse_next_escaped(b"\\f", &empty_esc_map()), (0x0C, 2));
        assert_eq!(parse_next_escaped(b"\\\"", &empty_esc_map()), (0x22, 2));
        // `\/` -> `/`: JSON's relaxed escape, accepted by the lexer
        // (upstream commit 5d65108, #626).
        assert_eq!(parse_next_escaped(b"\\/", &empty_esc_map()), (0x2F, 2));
    }

    #[test]
    fn parse_hex_x_escape() {
        assert_eq!(parse_next_escaped(b"\\x41", &empty_esc_map()), (0x41, 4));
        // Arbitrary length.
        assert_eq!(
            parse_next_escaped(b"\\x1F600", &empty_esc_map()),
            (0x1F600, 7)
        );
        // Stops at first non-hex char.
        assert_eq!(parse_next_escaped(b"\\xABg", &empty_esc_map()), (0xAB, 4));
    }

    #[test]
    fn parse_unicode_escapes() {
        assert_eq!(parse_next_escaped(b"\\u00A9", &empty_esc_map()), (0xA9, 6));
        assert_eq!(
            parse_next_escaped(b"\\u2603", &empty_esc_map()),
            (0x2603, 6)
        );
        assert_eq!(
            parse_next_escaped(b"\\U0001F600", &empty_esc_map()),
            (0x1F600, 10)
        );
    }

    #[test]
    fn parse_invalid_escapes() {
        // Not a backslash.
        assert_eq!(
            parse_next_escaped(b"n", &empty_esc_map()),
            (INVALID_ESCAPE, 0)
        );
        // Unknown escape letter.
        assert_eq!(
            parse_next_escaped(b"\\z", &empty_esc_map()),
            (INVALID_ESCAPE, 0)
        );
        // Bare backslash.
        assert_eq!(
            parse_next_escaped(b"\\", &empty_esc_map()),
            (INVALID_ESCAPE, 0)
        );
        // \x with no hex digits.
        assert_eq!(
            parse_next_escaped(b"\\xg", &empty_esc_map()),
            (INVALID_ESCAPE, 0)
        );
        // \u with too few digits.
        assert_eq!(
            parse_next_escaped(b"\\u12", &empty_esc_map()),
            (INVALID_ESCAPE, 0)
        );
        // \u with a non-hex digit.
        assert_eq!(
            parse_next_escaped(b"\\u12zz", &empty_esc_map()),
            (INVALID_ESCAPE, 0)
        );
    }

    #[test]
    fn parse_additional_escape_map() {
        let mut m = empty_esc_map();
        m.insert(b'-', '-' as TCodepoint);
        assert_eq!(parse_next_escaped(b"\\-", &m), ('-' as TCodepoint, 2));
    }

    #[test]
    fn utf8_or_escaped_dispatch() {
        // Raw UTF-8 char.
        assert_eq!(
            parse_next_utf8_or_escaped(b"A", &empty_esc_map()),
            (0x41, 1)
        );
        // Escape sequence.
        assert_eq!(
            parse_next_utf8_or_escaped(b"\\n", &empty_esc_map()),
            (0x0A, 2)
        );
        // Invalid escape.
        assert_eq!(
            parse_next_utf8_or_escaped(b"\\z", &empty_esc_map()),
            (INVALID_ESCAPE, 0)
        );
        // Invalid UTF-8.
        assert_eq!(
            parse_next_utf8_or_escaped(&[0x80], &empty_esc_map()),
            (INVALID_UTF8, 0)
        );
    }

    #[test]
    fn unescape_full_string() {
        // "escaped \"quotes\"" style.
        let out = unescape_string(b"escaped \\\"quotes\\\"", &empty_esc_map()).unwrap();
        assert_eq!(out, b"escaped \"quotes\"");
        // Mixed \n\r\t\\
        let out = unescape_string(b"\\n\\r\\t\\\\", &empty_esc_map()).unwrap();
        assert_eq!(out, b"\n\r\t\\");
    }

    #[test]
    fn unescape_unicode_to_utf8() {
        let out = unescape_string(b"\\u00A9 \\u2603 \\U0001F600", &empty_esc_map()).unwrap();
        assert_eq!(out, "© ☃ 😀".as_bytes());
    }

    #[test]
    fn unescape_propagates_error() {
        assert_eq!(
            unescape_string(b"\\z", &empty_esc_map()),
            Err(INVALID_ESCAPE)
        );
        assert_eq!(
            unescape_string(&[0x80], &empty_esc_map()),
            Err(INVALID_UTF8)
        );
    }

    #[test]
    fn escape_unescape_round_trip() {
        for cp in [0x0A_i32, 0x41, 0xA9, 0x2603, 0x1F600, 0x00, 0x1B] {
            let escaped = print_as_escaped(cp, &empty_cp_map());
            let (back, _) = parse_next_utf8_or_escaped(escaped.as_bytes(), &empty_esc_map());
            assert_eq!(back, cp, "round trip failed for {cp:#x}");
        }
    }
}
