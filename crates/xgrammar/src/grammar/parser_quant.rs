// SPDX-License-Identifier: AGPL-3.0-only
//
// EBNFParser — quantifier handling (`*` / `+` / `?`), repetition-range
// parsing and small-count repetition expansion. Split out of
// `parser.rs` to keep each file under the 250-line cap. Port of the
// corresponding `EBNFParser` methods from xgrammar
// `cpp/grammar_parser.cc`. The large `handle_repetition_range` lives
// in `parser_repeat.rs`; macro/tag-dispatch parsing in `parser_macro.rs`.

use super::{EbnfParser, ParseError};
use crate::grammar::expr::GrammarExprType;
use crate::grammar::lexer::{TokenType, TokenValue};

/// Repetition counts beyond this are expanded via a recursive rule
/// rather than literal unzipping. Mirrors C++ `kUnzipThreshold`.
pub(super) const UNZIP_THRESHOLD: i64 = 128;

impl EbnfParser {
    /// `a*` — character classes become `CharacterClassStar`; anything
    /// else expands to a recursive rule `rule ::= a rule | ""`.
    pub(super) fn handle_star_quantifier(
        &mut self,
        grammar_expr_id: i32,
    ) -> Result<i32, ParseError> {
        let expr = self.builder.get_grammar_expr(grammar_expr_id);
        if expr.kind == GrammarExprType::CharacterClass {
            let data: Vec<i32> = expr.data.to_vec();
            return Ok(self
                .builder
                .add_grammar_expr(GrammarExprType::CharacterClassStar, &data));
        }
        let new_rule_name = self.builder.get_new_rule_name(&self.cur_rule_name.clone());
        let new_rule_id = self.builder.add_empty_rule(new_rule_name)?;
        let ref_to_new_rule = self.builder.add_rule_ref(new_rule_id);
        let empty = self.builder.add_empty_str();
        let seq = self
            .builder
            .add_sequence(&[grammar_expr_id, ref_to_new_rule]);
        let body = self.builder.add_choices(&[empty, seq]);
        self.builder.update_rule_body(new_rule_id, body)?;
        Ok(self.builder.add_rule_ref(new_rule_id))
    }

    /// `a+` — expands to `rule ::= a rule | a`.
    pub(super) fn handle_plus_quantifier(
        &mut self,
        grammar_expr_id: i32,
    ) -> Result<i32, ParseError> {
        let new_rule_name = self.builder.get_new_rule_name(&self.cur_rule_name.clone());
        let new_rule_id = self.builder.add_empty_rule(new_rule_name)?;
        let ref_to_new_rule = self.builder.add_rule_ref(new_rule_id);
        let seq = self
            .builder
            .add_sequence(&[grammar_expr_id, ref_to_new_rule]);
        let body = self.builder.add_choices(&[seq, grammar_expr_id]);
        self.builder.update_rule_body(new_rule_id, body)?;
        Ok(self.builder.add_rule_ref(new_rule_id))
    }

    /// `a?` — expands to `rule ::= "" | a`.
    pub(super) fn handle_question_quantifier(
        &mut self,
        grammar_expr_id: i32,
    ) -> Result<i32, ParseError> {
        let new_rule_name = self.builder.get_new_rule_name(&self.cur_rule_name.clone());
        let empty = self.builder.add_empty_str();
        let body = self.builder.add_choices(&[empty, grammar_expr_id]);
        let new_rule_id = self.builder.add_rule_named(new_rule_name, body)?;
        Ok(self.builder.add_rule_ref(new_rule_id))
    }

    /// Parse a `{m}` / `{m,}` / `{m,n}` repetition range.
    pub(super) fn parse_repetition_range(&mut self) -> Result<(i64, i64), ParseError> {
        self.peek_and_consume(TokenType::LBrace, "Expect {")?;
        let lower = self.parse_integer()?;
        if lower < 0 {
            return Err(self.parse_error("Lower bound cannot be negative", -1));
        }
        if self.peek(0).ty == TokenType::Comma {
            self.consume(1);
            if self.peek(0).ty == TokenType::RBrace {
                self.consume(1);
                return Ok((lower, -1));
            }
            let upper = self.parse_integer()?;
            if upper < lower {
                return Err(self.parse_error(
                    &format!("Lower bound is larger than upper bound: {lower} > {upper}"),
                    -1,
                ));
            }
            self.peek_and_consume(TokenType::RBrace, "Expect }")?;
            return Ok((lower, upper));
        } else if self.peek(0).ty == TokenType::RBrace {
            self.consume(1);
            return Ok((lower, lower));
        }
        Err(self.parse_error("Expect ',' or '}' in repetition range", 0))
    }

    /// Parse a single integer literal token.
    pub(super) fn parse_integer(&mut self) -> Result<i64, ParseError> {
        if self.peek(0).ty != TokenType::IntegerLiteral {
            return Err(self.parse_error(
                &format!("Expect integer, but got {}", self.peek(0).lexeme),
                0,
            ));
        }
        let num = match self.peek(0).value {
            TokenValue::Int(v) => v,
            _ => return Err(self.parse_error("Expect integer", 0)),
        };
        self.consume(1);
        Ok(num)
    }

    /// Expand `{lower, upper}` by literal unzipping. `upper == -1` means
    /// unbounded. Mirrors C++ `LegacyHandleRepetitionRange`.
    pub(super) fn legacy_handle_repetition_range(
        &mut self,
        grammar_expr_id: i32,
        lower: i64,
        upper: i64,
    ) -> Result<i32, ParseError> {
        let mut elements: Vec<i32> = vec![grammar_expr_id; lower.max(0) as usize];
        // Case {l}.
        if upper == lower {
            return Ok(self.builder.add_sequence(&elements));
        }
        // Case {l,}.
        if upper == -1 {
            let name = self.builder.get_new_rule_name(&self.cur_rule_name.clone());
            let new_rule_id = self.builder.add_empty_rule(name)?;
            let ref_to_new_rule = self.builder.add_rule_ref(new_rule_id);
            let empty = self.builder.add_empty_str();
            let seq = self
                .builder
                .add_sequence(&[grammar_expr_id, ref_to_new_rule]);
            let body = self.builder.add_choices(&[empty, seq]);
            self.builder.update_rule_body(new_rule_id, body)?;
            elements.push(self.builder.add_rule_ref(new_rule_id));
            return Ok(self.builder.add_sequence(&elements));
        }
        // Case {l, r}, r - l >= 1.
        let span = (upper - lower) as usize;
        let mut rest_rule_ids: Vec<i32> = Vec::with_capacity(span);
        for _ in 0..span {
            let name = self.builder.get_new_rule_name(&self.cur_rule_name.clone());
            rest_rule_ids.push(self.builder.add_empty_rule(name)?);
        }
        for i in 0..span.saturating_sub(1) {
            let ref_to_next = self.builder.add_rule_ref(rest_rule_ids[i + 1]);
            let empty = self.builder.add_empty_str();
            let seq = self.builder.add_sequence(&[grammar_expr_id, ref_to_next]);
            let body = self.builder.add_choices(&[empty, seq]);
            self.builder.update_rule_body(rest_rule_ids[i], body)?;
        }
        let empty = self.builder.add_empty_str();
        let last = self.builder.add_choices(&[empty, grammar_expr_id]);
        let last_idx = rest_rule_ids.len() - 1;
        self.builder
            .update_rule_body(rest_rule_ids[last_idx], last)?;
        elements.push(self.builder.add_rule_ref(rest_rule_ids[0]));
        Ok(self.builder.add_sequence(&elements))
    }
}
