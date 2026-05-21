// SPDX-License-Identifier: AGPL-3.0-only
//
// Escape-sequence handlers for the regex converter — port of
// `RegexConverter::HandleCharEscape`, `HandleEscape` and
// `HandleEscapeInCharClass` from `cpp/regex_converter.cc`.

use super::RegexError;
use super::cursor::Cursor;
use crate::support::encoding::{TCodepoint, char_handling_error, hex_char_to_int};
use crate::support::escape::{parse_next_escaped, print_as_escaped};
use std::collections::HashMap;

/// The custom escape map used inside EBNF string literals — the C++
/// `CUSTOM_ESCAPE_MAP` in `HandleEscape`. Maps the escape letter to
/// the literal codepoint it stands for.
fn escape_map_string() -> HashMap<u8, TCodepoint> {
    let mut m = HashMap::new();
    for c in b"^$.*+?\\()[]{}|/" {
        m.insert(*c, *c as TCodepoint);
    }
    m
}

/// The custom escape map used inside character classes — the C++
/// `CUSTOM_ESCAPE_MAP` in `HandleCharEscape` (it additionally maps
/// `-`).
fn escape_map_char_class() -> HashMap<u8, TCodepoint> {
    let mut m = escape_map_string();
    m.insert(b'-', b'-' as TCodepoint);
    m
}

/// Render a codepoint the way C++ `EscapeString(codepoint)` does
/// (no additional escape map).
fn escape_codepoint(cp: TCodepoint) -> String {
    print_as_escaped(cp, &HashMap::new())
}

/// Port of `RegexConverter::HandleCharEscape`.
///
/// `current_` (the cursor) must point at the leading `\`. The cursor
/// is advanced past the whole escape sequence. Returns the escaped
/// string representation of the matched codepoint.
pub(super) fn handle_char_escape(cur: &mut Cursor) -> Result<String, RegexError> {
    let bytes = cur.rest_bytes();
    // C++ length pre-checks: a sequence must have at least 2 bytes,
    // `\u…` at least 5, `\x…` at least 4, `\c…` at least 3.
    let second = bytes.get(1).copied();
    let too_short = bytes.len() < 2
        || (second == Some(b'u') && bytes.len() < 5)
        || (second == Some(b'x') && bytes.len() < 4)
        || (second == Some(b'c') && bytes.len() < 3);
    if too_short {
        return Err(cur.error("Escape sequence is not finished."));
    }

    let (codepoint, len) = parse_next_escaped(&bytes, &escape_map_char_class());
    if codepoint != char_handling_error::INVALID_ESCAPE {
        cur.advance_bytes(len as usize);
        return Ok(escape_codepoint(codepoint));
    }

    // `\u{HHHH}` — brace-delimited unicode escape.
    if second == Some(b'u') && bytes.get(2) == Some(&b'{') {
        return handle_brace_unicode(cur);
    }

    // `\cX` — control character: takes `X % 32`.
    if second == Some(b'c') {
        let ctrl = bytes[2];
        if !ctrl.is_ascii_alphabetic() {
            // Advance past `\c` so the error position matches C++.
            cur.advance_bytes(2);
            return Err(cur.error("Invalid control character escape sequence."));
        }
        cur.advance_bytes(3);
        return Ok(escape_codepoint((ctrl % 32) as TCodepoint));
    }

    // Unrecognised escape: C++ warns and matches the char literally.
    let lit = bytes[1] as TCodepoint;
    cur.warn(format!(
        "Escape sequence '\\{}' is not recognized. The character itself will be matched",
        escape_codepoint(lit)
    ));
    cur.advance_bytes(2);
    Ok(escape_codepoint(lit))
}

/// `\u{HHHH}` handling, split out of [`handle_char_escape`].
fn handle_brace_unicode(cur: &mut Cursor) -> Result<String, RegexError> {
    let bytes = cur.rest_bytes();
    // bytes[0..3] == "\u{"
    let mut len = 0usize;
    let mut value: TCodepoint = 0;
    while len <= 6 {
        match bytes.get(3 + len) {
            Some(&b) => {
                let d = hex_char_to_int(b);
                if d == -1 {
                    break;
                }
                value = value * 16 + d;
                len += 1;
            }
            None => break,
        }
    }
    if len == 0 || len > 6 || bytes.get(3 + len) != Some(&b'}') {
        return Err(cur.error("Invalid Unicode escape sequence."));
    }
    cur.advance_bytes(3 + len + 1);
    Ok(escape_codepoint(value))
}

/// Port of `RegexConverter::HandleEscape` — escapes that appear at
/// the top level of the regex (outside a character class).
pub(super) fn handle_escape(cur: &mut Cursor) -> Result<String, RegexError> {
    let bytes = cur.rest_bytes();
    if bytes.len() < 2 {
        return Err(cur.error("Escape sequence is not finished."));
    }
    let res = match bytes[1] {
        b'd' => Some("[0-9]"),
        b'D' => Some("[^0-9]"),
        b'w' => Some("[a-zA-Z0-9_]"),
        b'W' => Some("[^a-zA-Z0-9_]"),
        b's' => Some("[\\f\\n\\r\\t\\v\\u0020\\u00a0]"),
        b'S' => Some("[^[\\f\\n\\r\\t\\v\\u0020\\u00a0]"),
        _ => None,
    };
    if let Some(s) = res {
        cur.advance_bytes(2);
        return Ok(s.to_string());
    }
    match bytes[1] {
        b'1'..=b'9' | b'k' => Err(cur.error("Backreference is not supported yet.")),
        b'p' | b'P' => {
            Err(cur.error("Unicode character class escape sequence is not supported yet."))
        }
        b'b' | b'B' => Err(cur.error("Word boundary is not supported yet.")),
        _ => {
            let escaped = handle_char_escape(cur)?;
            Ok(format!("\"{escaped}\""))
        }
    }
}

/// Port of `RegexConverter::HandleEscapeInCharClass` — escapes that
/// appear *inside* a `[...]` character class.
pub(super) fn handle_escape_in_char_class(cur: &mut Cursor) -> Result<String, RegexError> {
    let bytes = cur.rest_bytes();
    if bytes.len() < 2 {
        return Err(cur.error("Escape sequence is not finished."));
    }
    let res: Option<&str> = match bytes[1] {
        b'd' => Some("0-9"),
        b'D' => Some(r"\x00-\x2F\x3A-\U0010FFFF"),
        b'w' => Some("a-zA-Z0-9_"),
        b'W' => Some(r"\x00-\x2F\x3A-\x40\x5B-\x5E\x60\x7B-\U0010FFFF"),
        b's' => Some("\\f\\n\\r\\t\\v\\u0020\\u00a0"),
        b'S' => Some(r"\x00-\x08\x0E-\x1F\x21-\x9F\xA1-\U0010FFFF"),
        _ => None,
    };
    if let Some(s) = res {
        cur.advance_bytes(2);
        return Ok(s.to_string());
    }
    // Otherwise fall back to a plain char escape; `]` and `-` are
    // re-escaped so they do not terminate / split the char class.
    let r = handle_char_escape(cur)?;
    if r == "]" || r == "-" {
        Ok(format!("\\{r}"))
    } else {
        Ok(r)
    }
}
