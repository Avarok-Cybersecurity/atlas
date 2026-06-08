// SPDX-License-Identifier: AGPL-3.0-only
//
// EBNFParser — parses BNF/EBNF grammar text into a `GrammarData`.
// Port of `EBNFParser` + `ParseEBNF` from xgrammar `cpp/grammar_parser.cc`.
//
// The parser accepts W3C-style EBNF with xgrammar's extensions:
//   - `#` line comments instead of C-style comments
//   - C-style unicode escapes `ƫ`, `\U000001AB`, `\xAB`
//
// This file holds the error type, parser state, the core peek/consume
// helpers, sequence/choices/rule parsing and the public entry point.
// Element parsing is in `parser_element.rs`, quantifier/repetition
// expansion in `parser_quant.rs`, macro/tag-dispatch parsing in
// `parser_macro.rs` — split to keep each file under the 250-line cap.

use super::builder::{BuilderError, GrammarBuilder};
use super::data::GrammarData;
use super::lexer::{EbnfLexer, LexError, Token, TokenType, TokenValue};

#[path = "parser_element.rs"]
mod element;
#[path = "parser_macro.rs"]
mod macros;
#[path = "parser_quant.rs"]
mod quant;
#[path = "parser_repeat.rs"]
mod repeat;
#[path = "parser_tagdispatch.rs"]
mod tagdispatch;
#[cfg(test)]
#[path = "parser_tests.rs"]
mod tests;

/// Maximum nesting depth of parenthesised groups.
pub(super) const MAX_NEST_LAYER: i32 = 1000;

/// An EBNF parse error.
#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum ParseError {
    /// Lexer-stage error.
    #[error(transparent)]
    Lex(#[from] LexError),
    /// Parser-stage error with source position.
    #[error("EBNF parser error at line {line}, column {column}: {msg}")]
    Parse {
        /// 1-based source line.
        line: i32,
        /// 1-based source column.
        column: i32,
        /// Human-readable error message.
        msg: String,
    },
    /// An error surfaced from the underlying [`GrammarBuilder`].
    #[error(transparent)]
    Builder(#[from] BuilderError),
}

/// The EBNF parser. Equivalent to xgrammar's `EBNFParser`.
pub(super) struct EbnfParser {
    pub(super) builder: GrammarBuilder,
    pub(super) tokens: Vec<Token>,
    /// Index of the current token in `tokens`.
    pub(super) cur: usize,
    pub(super) cur_rule_name: String,
    pub(super) root_rule_name: String,
    pub(super) nest_layer_guard: i32,
}

impl EbnfParser {
    /// Token at `cur + delta`. The token stream is always `EndOfFile`-
    /// terminated, so callers stay in bounds when reading near the end.
    pub(super) fn peek(&self, delta: i32) -> &Token {
        let idx = (self.cur as i32 + delta).clamp(0, self.tokens.len() as i32 - 1) as usize;
        &self.tokens[idx]
    }

    /// Advance `cnt` tokens.
    pub(super) fn consume(&mut self, cnt: usize) {
        self.cur = (self.cur + cnt).min(self.tokens.len() - 1);
    }

    /// Consume a token of `ty`, or report `message`.
    pub(super) fn peek_and_consume(
        &mut self,
        ty: TokenType,
        message: &str,
    ) -> Result<(), ParseError> {
        if self.peek(0).ty != ty {
            return Err(self.parse_error(message, 0));
        }
        self.consume(1);
        Ok(())
    }

    /// Build a [`ParseError::Parse`] anchored at `peek(delta_element)`.
    pub(super) fn parse_error(&self, msg: &str, delta_element: i32) -> ParseError {
        let t = self.peek(delta_element);
        ParseError::Parse {
            line: t.line,
            column: t.column,
            msg: msg.to_string(),
        }
    }

    /// Parse a `|`-free sequence of elements into a `Sequence` expr.
    fn parse_sequence(&mut self) -> Result<i32, ParseError> {
        let mut elements: Vec<i32> = Vec::new();
        loop {
            elements.push(self.parse_element_with_quantifier()?);
            if matches!(
                self.peek(0).ty,
                TokenType::Pipe
                    | TokenType::RParen
                    | TokenType::LookaheadLParen
                    | TokenType::RuleName
                    | TokenType::EndOfFile
            ) {
                break;
            }
        }
        Ok(self.builder.add_sequence(&elements))
    }

    /// Parse `|`-separated sequences into a `Choices` expr.
    pub(super) fn parse_choices(&mut self) -> Result<i32, ParseError> {
        let mut choices: Vec<i32> = vec![self.parse_sequence()?];
        while self.peek(0).ty == TokenType::Pipe {
            self.consume(1);
            choices.push(self.parse_sequence()?);
        }
        Ok(self.builder.add_choices(&choices))
    }

    /// Parse a `(= ...)` lookahead assertion, returning its expr id.
    fn parse_lookahead_assertion(&mut self) -> Result<i32, ParseError> {
        self.peek_and_consume(
            TokenType::LookaheadLParen,
            "Expect (= in lookahead assertion",
        )?;
        let result = self.parse_choices()?;
        self.peek_and_consume(TokenType::RParen, "Expect )")?;
        Ok(result)
    }

    /// Parse one `name ::= body (= lookahead)?` rule definition.
    fn parse_rule(&mut self) -> Result<(String, i32, i32), ParseError> {
        if self.peek(0).ty != TokenType::RuleName {
            return Err(self.parse_error("Expect rule name", 0));
        }
        self.cur_rule_name = match &self.peek(0).value {
            TokenValue::Str(s) => s.clone(),
            _ => return Err(self.parse_error("Expect rule name", 0)),
        };
        self.consume(1);
        self.peek_and_consume(TokenType::Assign, "Expect ::=")?;
        let body_id = self.parse_choices()?;
        let mut lookahead_id = -1;
        if self.peek(0).ty == TokenType::LookaheadLParen {
            lookahead_id = self.parse_lookahead_assertion()?;
        }
        Ok((self.cur_rule_name.clone(), body_id, lookahead_id))
    }

    /// Pre-declare every rule name (so forward references resolve) and
    /// verify the root rule is present.
    fn init_rule_names(&mut self) -> Result<(), ParseError> {
        for delta in 0..self.tokens.len() {
            if self.tokens[delta].ty != TokenType::RuleName {
                continue;
            }
            let name = match &self.tokens[delta].value {
                TokenValue::Str(s) => s.clone(),
                _ => continue,
            };
            if self.builder.get_rule_id(&name) != -1 {
                let t = &self.tokens[delta];
                return Err(ParseError::Parse {
                    line: t.line,
                    column: t.column,
                    msg: format!("Rule \"{name}\" is defined multiple times"),
                });
            }
            self.builder.add_empty_rule(name)?;
        }
        if self.builder.get_rule_id(&self.root_rule_name) == -1 {
            return Err(self.parse_error(
                &format!(
                    "The root rule with name \"{}\" is not found",
                    self.root_rule_name
                ),
                0,
            ));
        }
        Ok(())
    }

    /// Run the full parse and emit the finished grammar.
    fn parse(mut self) -> Result<GrammarData, ParseError> {
        self.init_rule_names()?;
        while self.peek(0).ty != TokenType::EndOfFile {
            let (name, body_id, lookahead_id) = self.parse_rule()?;
            self.builder.update_rule_body_named(&name, body_id)?;
            self.builder
                .update_lookahead_assertion_named(&name, lookahead_id)?;
        }
        let root = self.root_rule_name.clone();
        Ok(self.builder.get(&root)?)
    }
}

/// Parse a BNF/EBNF grammar string into a [`GrammarData`].
///
/// `root_rule_name` names the start rule (xgrammar's default is
/// `"root"`). Returns a [`ParseError`] on malformed input rather than
/// panicking.
pub fn parse_ebnf(ebnf_string: &str, root_rule_name: &str) -> Result<GrammarData, ParseError> {
    let tokens = EbnfLexer::tokenize(ebnf_string)?;
    let parser = EbnfParser {
        builder: GrammarBuilder::new(),
        tokens,
        cur: 0,
        cur_rule_name: String::new(),
        root_rule_name: root_rule_name.to_string(),
        nest_layer_guard: 0,
    };
    parser.parse()
}

/// Parse a grammar string with the default root rule name `"root"`.
pub fn parse_ebnf_default(ebnf_string: &str) -> Result<GrammarData, ParseError> {
    parse_ebnf(ebnf_string, "root")
}
