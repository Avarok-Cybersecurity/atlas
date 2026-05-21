// SPDX-License-Identifier: AGPL-3.0-only
//
// Core regex→EBNF converter — port of `class RegexConverter` and
// `RegexConverter::Convert` from `cpp/regex_converter.cc`.

use super::RegexError;
use super::cursor::Cursor;
use super::escape_handlers::handle_escape;
use super::sub_handlers::quant_char;
use crate::support::encoding::TCodepoint;
use crate::support::escape::print_as_escaped;
use std::collections::HashMap;

/// Result of a successful conversion.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ConvertOutput {
    /// The EBNF rule body (no `root ::=` prefix, no trailing newline).
    pub ebnf: String,
    /// Non-fatal warnings collected during conversion (stray `^`/`$`,
    /// unrecognised escapes). The C++ prints these to stderr.
    pub warnings: Vec<String>,
}

/// The regex→EBNF converter. Construct with [`RegexConverter::new`]
/// then call [`RegexConverter::convert`].
pub struct RegexConverter {
    pub(super) cur: Cursor,
    pub(super) result: String,
    pub(super) parenthesis_level: i32,
}

impl RegexConverter {
    /// Build a converter for `regex`. Fails if `regex` is not valid
    /// UTF-8.
    pub fn new(regex: &str) -> Result<Self, RegexError> {
        Ok(RegexConverter {
            cur: Cursor::new(regex)?,
            result: String::new(),
            parenthesis_level: 0,
        })
    }

    /// Append `element` to the result, inserting a separating space
    /// when the result is non-empty — port of `AddEBNFSegment`.
    fn add_segment(&mut self, element: &str) {
        if !self.result.is_empty() {
            self.result.push(' ');
        }
        self.result.push_str(element);
    }

    /// Run the conversion — port of `RegexConverter::Convert`.
    pub fn convert(mut self) -> Result<ConvertOutput, RegexError> {
        let mut is_empty = true;
        while !self.cur.at_end() {
            let c = self.cur.peek();
            match c {
                _ if c == '^' as TCodepoint => {
                    if !self.cur.at_start() {
                        self.cur.warn(
                            "'^' should be at the start of the regex, but found in the \
                             middle. It is ignored.",
                        );
                    }
                    self.cur.bump();
                }
                _ if c == '$' as TCodepoint => {
                    if !self.cur.at_last() {
                        self.cur.warn(
                            "'$' should be at the end of the regex, but found in the \
                             middle. It is ignored.",
                        );
                    }
                    self.cur.bump();
                }
                _ if c == '[' as TCodepoint => {
                    is_empty = false;
                    let cls = self.handle_character_class()?;
                    self.add_segment(&cls);
                }
                _ if c == '(' as TCodepoint => {
                    is_empty = false;
                    self.cur.bump();
                    self.parenthesis_level += 1;
                    self.add_segment("(");
                    if !self.cur.at_end() && self.cur.peek() == '?' as TCodepoint {
                        self.cur.bump();
                        self.handle_group_modifier()?;
                    }
                }
                _ if c == ')' as TCodepoint => {
                    is_empty = false;
                    if self.parenthesis_level == 0 {
                        return Err(self.cur.error("Unmatched ')'"));
                    }
                    // If the previous char was '|', emit an empty
                    // alternative so `(a|)` becomes `( "a" | "" )`.
                    if self.cur.prev() == Some('|' as TCodepoint) {
                        self.add_segment("\"\"");
                    }
                    self.parenthesis_level -= 1;
                    self.add_segment(")");
                    self.cur.bump();
                }
                _ if c == '*' as TCodepoint || c == '+' as TCodepoint || c == '?' as TCodepoint => {
                    is_empty = false;
                    self.result.push(quant_char(c));
                    self.cur.bump();
                    self.consume_optional_non_greedy();
                    self.reject_consecutive_quantifier()?;
                }
                _ if c == '{' as TCodepoint => {
                    is_empty = false;
                    let rep = self.handle_repetition_range()?;
                    self.result.push_str(&rep);
                    self.consume_optional_non_greedy();
                    self.reject_consecutive_quantifier()?;
                }
                _ if c == '|' as TCodepoint => {
                    is_empty = false;
                    self.add_segment("|");
                    self.cur.bump();
                }
                _ if c == '\\' as TCodepoint => {
                    is_empty = false;
                    let esc = handle_escape(&mut self.cur)?;
                    self.add_segment(&esc);
                }
                _ if c == '.' as TCodepoint => {
                    is_empty = false;
                    self.add_segment("[\\u0000-\\U0010FFFF]");
                    self.cur.bump();
                }
                _ => {
                    is_empty = false;
                    let escaped = print_as_escaped(c, &HashMap::new());
                    self.add_segment(&format!("\"{escaped}\""));
                    self.cur.bump();
                }
            }
        }
        if self.parenthesis_level != 0 {
            return Err(self.cur.error("The parenthesis is not closed."));
        }
        if is_empty {
            self.add_segment("\"\"");
        }
        Ok(ConvertOutput {
            ebnf: self.result,
            warnings: self.cur.warnings,
        })
    }

    /// Skip a `?` non-greedy modifier if present — our grammar treats
    /// all repetition non-deterministically, so it is simply dropped.
    fn consume_optional_non_greedy(&mut self) {
        if !self.cur.at_end() && self.cur.peek() == '?' as TCodepoint {
            self.cur.bump();
        }
    }

    /// Reject a second repetition modifier directly following a
    /// quantifier — port of the "Two consecutive repetition
    /// modifiers" check.
    fn reject_consecutive_quantifier(&self) -> Result<(), RegexError> {
        if self.cur.at_end() {
            return Ok(());
        }
        let c = self.cur.peek();
        if c == '{' as TCodepoint
            || c == '*' as TCodepoint
            || c == '+' as TCodepoint
            || c == '?' as TCodepoint
        {
            return Err(self
                .cur
                .error("Two consecutive repetition modifiers are not allowed."));
        }
        Ok(())
    }
}
