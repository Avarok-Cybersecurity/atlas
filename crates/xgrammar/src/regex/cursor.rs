// SPDX-License-Identifier: AGPL-3.0-only
//
// Cursor over the regex codepoint stream — split out of
// `converter.rs` to keep each file under the 250-line cap.

use super::RegexError;
use crate::support::encoding::{TCodepoint, char_handling_error, char_to_utf8, parse_utf8};
/// A cursor over the regex codepoint stream.
///
/// The C++ converter decodes the regex into a `std::vector<TCodepoint>`
/// with an appended `0` terminator and walks it with `current_` /
/// `start_` / `end_` raw pointers. We model the same thing with an
/// index — `end` points at the terminator slot, so `pos == end` means
/// "at the null terminator", matching `current_ == end_`.
pub struct Cursor {
    /// Decoded codepoints (no trailing terminator).
    pub(super) codepoints: Vec<TCodepoint>,
    /// Current read position.
    pub(super) pos: usize,
    /// Collected non-fatal warnings.
    pub(super) warnings: Vec<String>,
}

impl Cursor {
    /// Decode `regex` into codepoints. Returns an error if the regex
    /// is not valid UTF-8 — faithful to the C++ `kInvalidUTF8` check.
    pub(super) fn new(regex: &str) -> Result<Self, RegexError> {
        let codepoints = if regex.is_empty() {
            Vec::new()
        } else {
            let cps = parse_utf8(regex.as_bytes(), false);
            if cps.first() == Some(&char_handling_error::INVALID_UTF8) {
                return Err(RegexError {
                    position: 1,
                    message: "The regex is not a valid UTF-8 string.".to_string(),
                });
            }
            cps
        };
        Ok(Cursor {
            codepoints,
            pos: 0,
            warnings: Vec::new(),
        })
    }

    /// `current_ == end_`: at or past the terminator.
    pub(super) fn at_end(&self) -> bool {
        self.pos >= self.codepoints.len()
    }

    /// `current_ == start_`: nothing consumed yet.
    pub(super) fn at_start(&self) -> bool {
        self.pos == 0
    }

    /// Whether `current_` is exactly one before `end_` (the C++
    /// `current_ == end_ - 1` test used for `$`).
    pub(super) fn at_last(&self) -> bool {
        self.pos + 1 == self.codepoints.len()
    }

    /// The current codepoint (`*current_`), or `0` at the terminator —
    /// matching the C-string null terminator the C++ relies on.
    pub(super) fn peek(&self) -> TCodepoint {
        self.codepoints.get(self.pos).copied().unwrap_or(0)
    }

    /// The previous codepoint (`current_[-1]`), or `None` at the start.
    pub(super) fn prev(&self) -> Option<TCodepoint> {
        self.pos.checked_sub(1).map(|i| self.codepoints[i])
    }

    /// Advance one codepoint.
    pub(super) fn bump(&mut self) {
        self.pos += 1;
    }

    /// The remaining input from `current_`, re-encoded as UTF-8 bytes.
    /// Escape handlers use this to call the byte-oriented
    /// `parse_next_escaped`.
    pub(super) fn rest_bytes(&self) -> Vec<u8> {
        let mut out = Vec::new();
        for &cp in &self.codepoints[self.pos..] {
            out.extend_from_slice(&char_to_utf8(cp));
        }
        out
    }

    /// Advance past `n` *bytes* of escape input. Because every escape
    /// sequence `parse_next_escaped` consumes is pure ASCII, one byte
    /// equals one codepoint here.
    pub(super) fn advance_bytes(&mut self, n: usize) {
        self.pos += n;
    }

    /// Build a [`RegexError`] at the current 1-based position — the
    /// C++ `RaiseError` offset is `current_ - start_ + 1`.
    pub(super) fn error(&self, message: impl Into<String>) -> RegexError {
        RegexError {
            position: self.pos + 1,
            message: message.into(),
        }
    }

    /// Record a non-fatal warning (the C++ `RaiseWarning`).
    pub(super) fn warn(&mut self, message: impl Into<String>) {
        self.warnings.push(message.into());
    }
}
