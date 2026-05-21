// SPDX-License-Identifier: AGPL-3.0-only
//
// EBNFLexer — tokenizer for BNF/EBNF grammar text.
// Port of `EBNFLexer` from xgrammar `cpp/grammar_parser.cc`.
//
// The lexer turns grammar source into a flat `Vec<Token>`, classifying
// rule-definition names, identifiers, string/char-class/integer
// literals, quantifiers and structural punctuation.
//
// This file holds the lexer struct + core scanning helpers. Token
// type definitions live in `lexer_token.rs`, the literal-scanning
// methods in `lexer_scan.rs`, tests in `lexer_tests.rs` — split to
// keep each file under the 250-line cap.

#[path = "lexer_scan.rs"]
mod scan;
#[cfg(test)]
#[path = "lexer_tests.rs"]
mod tests;
#[path = "lexer_token.rs"]
mod token;

pub use token::{LexError, Token, TokenType, TokenValue};

/// Largest integer literal accepted in a grammar (`1e15`).
pub(super) const MAX_INTEGER_IN_GRAMMAR: i64 = 1_000_000_000_000_000;

/// The EBNF lexer. Equivalent to xgrammar's `EBNFLexer::Impl`.
pub struct EbnfLexer {
    pub(super) input: Vec<u8>,
    pub(super) pos: usize,
    pub(super) line: i32,
    pub(super) column: i32,
}

impl EbnfLexer {
    /// Tokenize `input` into a flat token vector terminated by `EndOfFile`.
    pub fn tokenize(input: &str) -> Result<Vec<Token>, LexError> {
        let mut lexer = EbnfLexer {
            input: input.as_bytes().to_vec(),
            pos: 0,
            line: 1,
            column: 1,
        };
        let mut tokens: Vec<Token> = Vec::new();
        loop {
            let mut produced = lexer.next_token()?;
            let stop = produced
                .last()
                .is_some_and(|t| t.ty == TokenType::EndOfFile);
            tokens.append(&mut produced);
            if stop {
                break;
            }
        }
        Self::convert_identifier_to_rule_name(&mut tokens)?;
        Ok(tokens)
    }

    /// Byte at `pos + delta`, or `0` (NUL) past the end — mirrors the
    /// C++ NUL-terminated-string `Peek`.
    pub(super) fn peek(&self, delta: usize) -> u8 {
        self.input.get(self.pos + delta).copied().unwrap_or(0)
    }

    /// Byte one position before `pos`, or `0` at the start.
    fn peek_back(&self) -> u8 {
        if self.pos == 0 {
            0
        } else {
            self.input[self.pos - 1]
        }
    }

    /// Consume `cnt` bytes, updating line/column tracking.
    pub(super) fn consume(&mut self, cnt: usize) {
        for _ in 0..cnt {
            let c = self.peek(0);
            if c == b'\n' || (c == b'\r' && self.peek(1) != b'\n') {
                self.line += 1;
                self.column = 1;
            } else {
                self.column += 1;
            }
            self.pos += 1;
        }
    }

    /// Skip whitespace and `#` comments.
    pub(super) fn consume_space(&mut self) {
        while matches!(self.peek(0), b' ' | b'\t' | b'#' | b'\n' | b'\r') {
            self.consume(1);
            if self.peek_back() == b'#' {
                while self.peek(0) != 0 && self.peek(0) != b'\n' && self.peek(0) != b'\r' {
                    self.consume(1);
                }
                if self.peek(0) == 0 {
                    return;
                }
                self.consume(1);
                if self.peek_back() == b'\r' && self.peek(0) == b'\n' {
                    self.consume(1);
                }
            }
        }
    }

    /// Build a [`LexError`] at the current source position.
    pub(super) fn err(&self, msg: impl Into<String>) -> LexError {
        LexError {
            line: self.line,
            column: self.column,
            msg: msg.into(),
        }
    }

    /// Whether `c` can appear in an identifier.
    pub(super) fn is_name_char(c: u8, is_first: bool) -> bool {
        c == b'_'
            || c == b'-'
            || c == b'.'
            || c.is_ascii_lowercase()
            || c.is_ascii_uppercase()
            || (!is_first && c.is_ascii_digit())
    }

    /// Parse a bare identifier string.
    fn parse_identifier_str(&mut self) -> Result<String, LexError> {
        let start = self.pos;
        let mut first = true;
        while self.peek(0) != 0 && Self::is_name_char(self.peek(0), first) {
            self.consume(1);
            first = false;
        }
        if start == self.pos {
            return Err(self.err("Expect identifier"));
        }
        Ok(String::from_utf8_lossy(&self.input[start..self.pos]).into_owned())
    }

    /// Parse an identifier or `true`/`false` boolean.
    pub(super) fn parse_identifier_or_boolean(&mut self) -> Result<Token, LexError> {
        let (line, column) = (self.line, self.column);
        let ident = self.parse_identifier_str()?;
        if ident == "true" || ident == "false" {
            return Ok(Token {
                ty: TokenType::BooleanLiteral,
                value: TokenValue::Bool(ident == "true"),
                lexeme: ident,
                line,
                column,
            });
        }
        Ok(Token {
            ty: TokenType::Identifier,
            value: TokenValue::Str(ident.clone()),
            lexeme: ident,
            line,
            column,
        })
    }

    /// Build a value-less token at `(line, column)`.
    pub(super) fn simple_token(
        &self,
        ty: TokenType,
        lexeme: &str,
        line: i32,
        column: i32,
    ) -> Token {
        Token {
            ty,
            lexeme: lexeme.to_string(),
            value: TokenValue::None,
            line,
            column,
        }
    }
}
