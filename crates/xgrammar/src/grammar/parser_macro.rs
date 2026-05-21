// SPDX-License-Identifier: AGPL-3.0-only
//
// EBNFParser — macro IR + macro argument parsing. Split out of
// `parser.rs`. Port of `EBNFParser::MacroIR`, `ParseMacroArguments`
// and `ParseMacroValue` from xgrammar `cpp/grammar_parser.cc`. The
// `TagDispatch` macro itself lives in `parser_tagdispatch.rs`.

use super::{EbnfParser, ParseError};
use crate::grammar::lexer::{TokenType, TokenValue};

/// One parsed macro argument value. Port of `EBNFParser::MacroIR::Node`.
/// `Int` is part of the faithful IR; no macro consumes integer args yet.
#[derive(Debug, Clone)]
pub(super) enum MacroValue {
    Str(String),
    #[allow(dead_code)]
    Int(i64),
    Bool(bool),
    Ident(String),
    Tuple(Vec<MacroValue>),
}

/// Parsed macro argument list (positional + named).
#[derive(Debug, Default)]
pub(super) struct MacroArguments {
    pub(super) arguments: Vec<MacroValue>,
    pub(super) named_arguments: Vec<(String, MacroValue)>,
}

impl EbnfParser {
    /// Parse a `(...)` macro argument list.
    pub(super) fn parse_macro_arguments(&mut self) -> Result<MacroArguments, ParseError> {
        let mut args = MacroArguments::default();
        self.peek_and_consume(TokenType::LParen, "Expect ( after macro function name")?;
        if self.peek(0).ty != TokenType::RParen {
            loop {
                if self.peek(0).ty == TokenType::Identifier && self.peek(1).ty == TokenType::Equal {
                    let name = match &self.peek(0).value {
                        TokenValue::Str(s) => s.clone(),
                        _ => return Err(self.parse_error("Expect identifier", 0)),
                    };
                    self.consume(2);
                    let value = self.parse_macro_value()?;
                    args.named_arguments.push((name, value));
                } else {
                    args.arguments.push(self.parse_macro_value()?);
                }
                if self.peek(0).ty == TokenType::Comma {
                    self.consume(1);
                } else if self.peek(0).ty == TokenType::RParen {
                    break;
                } else {
                    return Err(self.parse_error("Expect , or ) in macro arguments", 0));
                }
            }
        }
        self.peek_and_consume(TokenType::RParen, "Expect ) after macro arguments")?;
        Ok(args)
    }

    /// Parse a single macro value (string, int, bool, identifier, tuple).
    fn parse_macro_value(&mut self) -> Result<MacroValue, ParseError> {
        match self.peek(0).ty {
            TokenType::StringLiteral => {
                let v = string_value(&self.peek(0).value);
                self.consume(1);
                Ok(MacroValue::Str(v))
            }
            TokenType::IntegerLiteral => {
                let v = match self.peek(0).value {
                    TokenValue::Int(i) => i,
                    _ => 0,
                };
                self.consume(1);
                Ok(MacroValue::Int(v))
            }
            TokenType::BooleanLiteral => {
                let v = matches!(self.peek(0).value, TokenValue::Bool(true));
                self.consume(1);
                Ok(MacroValue::Bool(v))
            }
            TokenType::Identifier => {
                let v = string_value(&self.peek(0).value);
                self.consume(1);
                Ok(MacroValue::Ident(v))
            }
            TokenType::LParen => {
                self.consume(1);
                let mut elements: Vec<MacroValue> = Vec::new();
                if self.peek(0).ty != TokenType::RParen {
                    loop {
                        elements.push(self.parse_macro_value()?);
                        if self.peek(0).ty == TokenType::Comma {
                            self.consume(1);
                        } else if self.peek(0).ty == TokenType::RParen {
                            break;
                        } else {
                            return Err(self.parse_error("Expect , or ) in tuple", 0));
                        }
                    }
                }
                self.consume(1);
                Ok(MacroValue::Tuple(elements))
            }
            _ => Err(self.parse_error(
                "Expect string, integer, boolean, or tuple in macro argument",
                0,
            )),
        }
    }
}

/// Extract the string payload of a token value, or `""`.
fn string_value(v: &TokenValue) -> String {
    match v {
        TokenValue::Str(s) => s.clone(),
        _ => String::new(),
    }
}
