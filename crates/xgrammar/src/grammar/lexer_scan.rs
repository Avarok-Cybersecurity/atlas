// SPDX-License-Identifier: AGPL-3.0-only
//
// EBNFLexer — token-scanning methods (string/char-class/integer
// literals, the `next_token` dispatch and rule-name re-tagging).
// Split out of `lexer.rs` to keep each file under the 250-line cap.
// Port of the corresponding `EBNFLexer::Impl` methods from xgrammar
// `cpp/grammar_parser.cc`.

use std::collections::HashMap;

// INTEGRATION: `support::escape` holds escape-aware UTF-8 parsing;
// `support::encoding` holds codepoint→UTF-8 encoding.
use crate::support::encoding::{self, char_handling_error};
use crate::support::escape;

use super::{EbnfLexer, LexError, MAX_INTEGER_IN_GRAMMAR, Token, TokenType, TokenValue};

impl EbnfLexer {
    /// Parse a `"..."` string literal.
    pub(super) fn parse_string(&mut self) -> Result<Token, LexError> {
        let (line, column) = (self.line, self.column);
        let start = self.pos;
        self.consume(1); // opening quote
        let mut codepoints: Vec<i32> = Vec::new();
        let no_extra: HashMap<u8, i32> = HashMap::new();
        while self.peek(0) != 0 && !matches!(self.peek(0), b'"' | b'\n' | b'\r') {
            // INTEGRATION: support::escape::parse_next_utf8_or_escaped
            let (cp, len) = escape::parse_next_utf8_or_escaped(&self.input[self.pos..], &no_extra);
            match cp {
                char_handling_error::INVALID_UTF8 => {
                    return Err(self.err("Invalid UTF8 sequence"));
                }
                char_handling_error::INVALID_ESCAPE => {
                    return Err(self.err("Invalid escape sequence"));
                }
                _ => {}
            }
            self.consume(len as usize);
            codepoints.push(cp);
        }
        if self.peek(0) != b'"' {
            return Err(self.err("Expect \" in string literal"));
        }
        self.consume(1); // closing quote
        let lexeme = String::from_utf8_lossy(&self.input[start..self.pos]).into_owned();
        let mut value_bytes: Vec<u8> = Vec::new();
        for cp in codepoints {
            // INTEGRATION: support::encoding::char_to_utf8 -> Vec<u8>
            value_bytes.extend_from_slice(&encoding::char_to_utf8(cp));
        }
        let value = String::from_utf8_lossy(&value_bytes).into_owned();
        Ok(Token {
            ty: TokenType::StringLiteral,
            lexeme,
            value: TokenValue::Str(value),
            line,
            column,
        })
    }

    /// Parse a character class `[...]` into a token stream.
    pub(super) fn parse_char_class(&mut self) -> Result<Vec<Token>, LexError> {
        let mut tokens: Vec<Token> = Vec::new();
        let (line, column) = (self.line, self.column);
        tokens.push(self.simple_token(TokenType::LBracket, "[", line, column));
        self.consume(1); // '['
        if self.peek(0) == b'^' {
            let (l, c) = (self.line, self.column);
            tokens.push(self.simple_token(TokenType::Caret, "^", l, c));
            self.consume(1);
        }
        while self.peek(0) != 0 && self.peek(0) != b']' {
            let (l, c) = (self.line, self.column);
            if self.peek(0) == b'\r' || self.peek(0) == b'\n' {
                return Err(self.err("Character class should not contain newline"));
            } else if self.peek(0) == b'-' {
                tokens.push(self.simple_token(TokenType::Dash, "-", l, c));
                self.consume(1);
            } else if self.peek(0) == b'\\' && is_regex_special_escape(self.peek(1)) {
                let lexeme =
                    String::from_utf8_lossy(&self.input[self.pos..self.pos + 2]).into_owned();
                let val = (self.peek(1) as char).to_string();
                tokens.push(Token {
                    ty: TokenType::EscapeInCharClass,
                    lexeme,
                    value: TokenValue::Str(val),
                    line: l,
                    column: c,
                });
                self.consume(2);
            } else {
                // INTEGRATION: support::escape::parse_next_utf8_or_escaped
                let (cp, len) = escape::parse_next_utf8_or_escaped(
                    &self.input[self.pos..],
                    &regex_escape_map(),
                );
                match cp {
                    char_handling_error::INVALID_UTF8 => {
                        return Err(self.err("Invalid UTF8 sequence"));
                    }
                    char_handling_error::INVALID_ESCAPE => {
                        return Err(self.err("Invalid escape sequence"));
                    }
                    _ => {}
                }
                let len = len as usize;
                let lexeme =
                    String::from_utf8_lossy(&self.input[self.pos..self.pos + len]).into_owned();
                tokens.push(Token {
                    ty: TokenType::CharInCharClass,
                    lexeme,
                    value: TokenValue::Codepoint(cp),
                    line: l,
                    column: c,
                });
                self.consume(len);
            }
        }
        if self.peek(0) == 0 {
            return Err(self.err("Unterminated character class"));
        }
        let (l, c) = (self.line, self.column);
        tokens.push(self.simple_token(TokenType::RBracket, "]", l, c));
        self.consume(1); // ']'
        Ok(tokens)
    }

    /// Parse an integer literal (with optional sign).
    pub(super) fn parse_integer(&mut self) -> Result<Token, LexError> {
        let (line, column) = (self.line, self.column);
        let start = self.pos;
        let mut is_negative = false;
        if self.peek(0) == b'-' {
            is_negative = true;
            self.consume(1);
        } else if self.peek(0) == b'+' {
            self.consume(1);
        }
        let mut num: i64 = 0;
        while self.peek(0).is_ascii_digit() {
            num = num * 10 + (self.peek(0) - b'0') as i64;
            self.consume(1);
            if num > MAX_INTEGER_IN_GRAMMAR {
                return Err(self.err(format!(
                    "Integer is too large: parsed {num}, max allowed is {MAX_INTEGER_IN_GRAMMAR}"
                )));
            }
        }
        let lexeme = String::from_utf8_lossy(&self.input[start..self.pos]).into_owned();
        Ok(Token {
            ty: TokenType::IntegerLiteral,
            lexeme,
            value: TokenValue::Int(if is_negative { -num } else { num }),
            line,
            column,
        })
    }

    /// Produce the next token(s). Most calls return one token; a
    /// character class returns several.
    pub(super) fn next_token(&mut self) -> Result<Vec<Token>, LexError> {
        self.consume_space();
        let (line, column) = (self.line, self.column);
        if self.peek(0) == 0 {
            return Ok(vec![self.simple_token(
                TokenType::EndOfFile,
                "",
                line,
                column,
            )]);
        }
        let single = |s: &mut Self, ty, lex: &str, n| {
            s.consume(n);
            vec![s.simple_token(ty, lex, line, column)]
        };
        Ok(match self.peek(0) {
            b'(' if self.peek(1) == b'=' => single(self, TokenType::LookaheadLParen, "(=", 2),
            b'(' => single(self, TokenType::LParen, "(", 1),
            b')' => single(self, TokenType::RParen, ")", 1),
            b'{' => single(self, TokenType::LBrace, "{", 1),
            b'}' => single(self, TokenType::RBrace, "}", 1),
            b'|' => single(self, TokenType::Pipe, "|", 1),
            b',' => single(self, TokenType::Comma, ",", 1),
            b'*' => single(self, TokenType::Star, "*", 1),
            b'+' if !self.next_is_signed_int() => single(self, TokenType::Plus, "+", 1),
            b'?' => single(self, TokenType::Question, "?", 1),
            b'=' => single(self, TokenType::Equal, "=", 1),
            b':' if self.peek(1) == b':' && self.peek(2) == b'=' => {
                single(self, TokenType::Assign, "::=", 3)
            }
            b':' => return Err(self.err("Unexpected character: ':'")),
            b'"' => vec![self.parse_string()?],
            b'[' => self.parse_char_class()?,
            c if Self::is_name_char(c, true) => vec![self.parse_identifier_or_boolean()?],
            c if c.is_ascii_digit() || c == b'-' || c == b'+' => vec![self.parse_integer()?],
            c => return Err(self.err(format!("Unexpected character: {}", c as char))),
        })
    }

    /// Whether the current `+`/`-` begins a signed integer literal
    /// (digit follows). `*`/`+`/`?` quantifiers take priority for `+`.
    fn next_is_signed_int(&self) -> bool {
        self.peek(1).is_ascii_digit()
    }

    /// Re-tag the identifier that precedes a `::=` as a `RuleName`,
    /// validating that it sits at the start of a line.
    pub(super) fn convert_identifier_to_rule_name(tokens: &mut [Token]) -> Result<(), LexError> {
        let make_err = |t: &Token, msg: &str| LexError {
            line: t.line,
            column: t.column,
            msg: msg.to_string(),
        };
        for i in 0..tokens.len() {
            if tokens[i].ty != TokenType::Assign {
                continue;
            }
            if i == 0 {
                return Err(make_err(&tokens[0], "Assign should not be the first token"));
            }
            if tokens[i - 1].ty != TokenType::Identifier {
                return Err(make_err(
                    &tokens[i - 1],
                    "Assign should be preceded by an identifier",
                ));
            }
            if i >= 2 && tokens[i - 2].line == tokens[i - 1].line {
                return Err(make_err(
                    &tokens[i - 1],
                    "The rule name should be at the beginning of the line",
                ));
            }
            tokens[i - 1].ty = TokenType::RuleName;
        }
        Ok(())
    }
}

/// Regex-style escapes with special meaning inside a character class.
fn is_regex_special_escape(c: u8) -> bool {
    matches!(c, b'd' | b'D' | b's' | b'S' | b'w' | b'W')
}

/// Additional escape map for character-class literal escapes — each of
/// these escape sequences (`\^`, `\.`, `\-`, …) resolves to itself.
fn regex_escape_map() -> HashMap<u8, i32> {
    const CHARS: &[u8] = b"^$\\.*+?()[]{}|/-";
    CHARS.iter().map(|&c| (c, c as i32)).collect()
}
