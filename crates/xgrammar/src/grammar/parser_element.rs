// SPDX-License-Identifier: AGPL-3.0-only
//
// EBNFParser — element-level parsing: identifiers, character classes,
// string literals, rule references, and the element + quantifier
// dispatch. Split out of `parser.rs` to keep each file under the
// 250-line cap. Port of the corresponding `EBNFParser` methods from
// xgrammar `cpp/grammar_parser.cc`.

use super::{EbnfParser, MAX_NEST_LAYER, ParseError};
use crate::grammar::builder::CharacterClassElement;
use crate::grammar::lexer::{TokenType, TokenValue};

impl EbnfParser {
    /// Parse a bare rule-reference identifier and return its name.
    pub(super) fn parse_identifier(&mut self) -> Result<String, ParseError> {
        if self.peek(0).ty != TokenType::Identifier {
            return Err(self.parse_error("Expect identifier", 0));
        }
        let name = match &self.peek(0).value {
            TokenValue::Str(s) => s.clone(),
            _ => return Err(self.parse_error("Expect identifier", 0)),
        };
        self.consume(1);
        Ok(name)
    }

    /// Parse a `[...]` character class into a `CharacterClass` expr.
    fn parse_char_class(&mut self) -> Result<i32, ParseError> {
        self.peek_and_consume(TokenType::LBracket, "Expect [ in character class")?;
        let mut elements: Vec<CharacterClassElement> = Vec::new();
        let mut is_negated = false;
        if self.peek(0).ty == TokenType::Caret {
            is_negated = true;
            self.consume(1);
        }
        while self.peek(0).ty != TokenType::RBracket && self.peek(0).ty != TokenType::EndOfFile {
            if self.peek(0).ty == TokenType::EscapeInCharClass {
                return Err(
                    self.parse_error("Character class escape is not supported yet in EBNF", 0)
                );
            }
            let cp = self.char_class_codepoint(0)?;
            self.consume(1);
            // A range expression: `c - c2`.
            if self.peek(0).ty == TokenType::Dash
                && (self.peek(1).ty == TokenType::CharInCharClass
                    || self.peek(1).ty == TokenType::Dash)
            {
                let cp2 = self.char_class_codepoint(1)?;
                if cp > cp2 {
                    return Err(self.parse_error(
                        "Invalid character class: lower bound is larger than upper bound",
                        -1,
                    ));
                }
                elements.push(CharacterClassElement::new(cp, cp2));
                self.consume(2);
            } else {
                elements.push(CharacterClassElement::new(cp, cp));
            }
        }
        self.peek_and_consume(TokenType::RBracket, "Expect ] in character class")?;
        Ok(self.builder.add_character_class(&elements, is_negated))
    }

    /// Codepoint of the char-class token at `cur + delta` (a
    /// `CharInCharClass` or a `Dash`, which decodes to `-`).
    fn char_class_codepoint(&self, delta: i32) -> Result<i32, ParseError> {
        let t = self.peek(delta);
        match t.ty {
            TokenType::CharInCharClass => match t.value {
                TokenValue::Codepoint(c) => Ok(c),
                _ => Err(self.parse_error("Invalid character in character class", delta)),
            },
            TokenType::Dash => Ok(b'-' as i32),
            _ => Err(self.parse_error(
                &format!("Unexpected character in character class: {}", t.lexeme),
                delta,
            )),
        }
    }

    /// Parse a `"..."` literal into a `ByteString` (or `EmptyStr`).
    fn parse_string(&mut self) -> Result<i32, ParseError> {
        if self.peek(0).ty != TokenType::StringLiteral {
            return Err(self.parse_error("Expect string literal", 0));
        }
        let value = match &self.peek(0).value {
            TokenValue::Str(s) => s.clone(),
            _ => return Err(self.parse_error("Expect string literal", 0)),
        };
        self.consume(1);
        if value.is_empty() {
            Ok(self.builder.add_empty_str())
        } else {
            Ok(self.builder.add_byte_string(&value))
        }
    }

    /// Parse a reference to a previously-declared rule.
    fn parse_rule_ref(&mut self) -> Result<i32, ParseError> {
        let name = self.parse_identifier()?;
        let rule_id = self.builder.get_rule_id(&name);
        if rule_id == -1 {
            return Err(self.parse_error(&format!("Rule \"{name}\" is not defined"), -1));
        }
        Ok(self.builder.add_rule_ref(rule_id))
    }

    /// Parse one element: a parenthesised group, a character class, a
    /// string literal, a macro call, or a rule reference.
    fn parse_element(&mut self) -> Result<i32, ParseError> {
        match self.peek(0).ty {
            TokenType::LParen => {
                self.nest_layer_guard += 1;
                if self.nest_layer_guard > MAX_NEST_LAYER {
                    return Err(self.parse_error("Nest layer exceeded the maximum limit", -1));
                }
                self.consume(1);
                if self.peek(0).ty == TokenType::RParen {
                    self.consume(1);
                    self.nest_layer_guard -= 1;
                    return Ok(self.builder.add_empty_str());
                }
                let id = self.parse_choices()?;
                self.peek_and_consume(TokenType::RParen, "Expect )")?;
                self.nest_layer_guard -= 1;
                Ok(id)
            }
            TokenType::LBracket => self.parse_char_class(),
            TokenType::StringLiteral => self.parse_string(),
            TokenType::Identifier => {
                let id = match &self.peek(0).value {
                    TokenValue::Str(s) => s.clone(),
                    _ => return Err(self.parse_error("Expect element", 0)),
                };
                if id == "TagDispatch" {
                    self.parse_tag_dispatch()
                } else {
                    self.parse_rule_ref()
                }
            }
            _ => Err(self.parse_error(
                &format!("Expect element, but got {}", self.peek(0).lexeme),
                0,
            )),
        }
    }

    /// Parse an element optionally followed by a `*`/`+`/`?`/`{m,n}`
    /// quantifier. Quantifier expansion lives in `parser_quant.rs`.
    pub(super) fn parse_element_with_quantifier(&mut self) -> Result<i32, ParseError> {
        let id = self.parse_element()?;
        match self.peek(0).ty {
            TokenType::Star => {
                self.consume(1);
                self.handle_star_quantifier(id)
            }
            TokenType::Plus => {
                self.consume(1);
                self.handle_plus_quantifier(id)
            }
            TokenType::Question => {
                self.consume(1);
                self.handle_question_quantifier(id)
            }
            TokenType::LBrace => {
                let (lower, upper) = self.parse_repetition_range()?;
                self.handle_repetition_range(id, lower, upper)
            }
            _ => Ok(id),
        }
    }
}
