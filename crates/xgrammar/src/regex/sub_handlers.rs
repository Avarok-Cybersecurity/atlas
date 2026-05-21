// SPDX-License-Identifier: AGPL-3.0-only
//
// Regex sub-construct handlers — `[...]` character classes, `{x,y}`
// repetition ranges and `(?...)` group modifiers. Split out of
// `converter.rs` to keep each file under the 250-line cap. Port of
// `HandleCharacterClass`, `HandleRepetitionRange` and
// `HandleGroupModifier` from `cpp/regex_converter.cc`.

use super::RegexError;
use super::converter::RegexConverter;
use super::escape_handlers::handle_escape_in_char_class;
use crate::support::encoding::{TCodepoint, char_to_utf8};

impl RegexConverter {
    /// Port of `HandleCharacterClass`.
    pub(super) fn handle_character_class(&mut self) -> Result<String, RegexError> {
        let mut cls = String::from("[");
        self.cur.bump(); // past '['
        if self.cur.peek() == ']' as TCodepoint {
            return Err(self
                .cur
                .error("Empty character class is not allowed in regex."));
        }
        while self.cur.peek() != ']' as TCodepoint && !self.cur.at_end() {
            if self.cur.peek() == '\\' as TCodepoint {
                cls.push_str(&handle_escape_in_char_class(&mut self.cur)?);
            } else {
                let bytes = char_to_utf8(self.cur.peek());
                // The class accumulator is byte text; push the raw
                // UTF-8 of the codepoint as the C++ `CharToUTF8` does.
                cls.push_str(&String::from_utf8_lossy(&bytes));
                self.cur.bump();
            }
        }
        if self.cur.at_end() {
            return Err(self.cur.error("Unclosed '['"));
        }
        cls.push(']');
        self.cur.bump(); // past ']'
        Ok(cls)
    }

    /// Port of `HandleRepetitionRange` — `{x}`, `{x,}`, `{x,y}`.
    pub(super) fn handle_repetition_range(&mut self) -> Result<String, RegexError> {
        let mut result = String::from("{");
        self.cur.bump(); // past '{'
        if !is_ascii_digit(self.cur.peek()) {
            return Err(self.cur.error("Invalid repetition count."));
        }
        while is_ascii_digit(self.cur.peek()) {
            result.push(self.cur.peek() as u8 as char);
            self.cur.bump();
        }
        if self.cur.peek() != ',' as TCodepoint && self.cur.peek() != '}' as TCodepoint {
            return Err(self.cur.error("Invalid repetition count."));
        }
        let sep = self.cur.peek();
        result.push(sep as u8 as char);
        self.cur.bump();
        if sep == '}' as TCodepoint {
            return Ok(result); // `{x}`
        }
        // sep was ','
        if !is_ascii_digit(self.cur.peek()) && self.cur.peek() != '}' as TCodepoint {
            return Err(self.cur.error("Invalid repetition count."));
        }
        while is_ascii_digit(self.cur.peek()) {
            result.push(self.cur.peek() as u8 as char);
            self.cur.bump();
        }
        if self.cur.peek() != '}' as TCodepoint {
            return Err(self.cur.error("Invalid repetition count."));
        }
        result.push('}');
        self.cur.bump();
        Ok(result)
    }

    /// Port of `HandleGroupModifier` — `(?:…)`, `(?<name>…)`, and the
    /// rejection of lookahead/lookbehind/flag modifiers.
    pub(super) fn handle_group_modifier(&mut self) -> Result<(), RegexError> {
        if self.cur.at_end() {
            return Err(self.cur.error("Group modifier is not finished."));
        }
        let c = self.cur.peek();
        if c == ':' as TCodepoint {
            self.cur.bump(); // non-capturing group
        } else if c == '=' as TCodepoint || c == '!' as TCodepoint {
            return Err(self.cur.error("Lookahead is not supported yet."));
        } else if c == '<' as TCodepoint
            && self
                .peek_at(1)
                .is_some_and(|n| n == '=' as TCodepoint || n == '!' as TCodepoint)
        {
            return Err(self.cur.error("Lookbehind is not supported yet."));
        } else if c == '<' as TCodepoint {
            self.cur.bump();
            while !self.cur.at_end() && is_ascii_alpha(self.cur.peek()) {
                self.cur.bump();
            }
            if self.cur.at_end() || self.cur.peek() != '>' as TCodepoint {
                return Err(self.cur.error("Invalid named capturing group."));
            }
            self.cur.bump(); // ignore the group's name
        } else {
            return Err(self.cur.error("Group modifier flag is not supported yet."));
        }
        Ok(())
    }

    /// Codepoint `offset` positions ahead of `current_`, if in range.
    fn peek_at(&self, offset: usize) -> Option<TCodepoint> {
        self.cur.codepoints.get(self.cur.pos + offset).copied()
    }
}

/// Map a quantifier codepoint back to its ASCII char.
pub(super) fn quant_char(c: TCodepoint) -> char {
    c as u8 as char
}

/// `isdigit` for a codepoint.
pub(super) fn is_ascii_digit(c: TCodepoint) -> bool {
    (b'0' as TCodepoint..=b'9' as TCodepoint).contains(&c)
}

/// `isalpha` for a codepoint.
pub(super) fn is_ascii_alpha(c: TCodepoint) -> bool {
    (b'a' as TCodepoint..=b'z' as TCodepoint).contains(&c)
        || (b'A' as TCodepoint..=b'Z' as TCodepoint).contains(&c)
}
