// SPDX-License-Identifier: AGPL-3.0-only
//
// EBNFParser — large-count repetition expansion (`handle_repetition_range`).
// Split out of `parser_quant.rs` to keep each file under the 250-line
// cap. Port of `EBNFParser::HandleRepetitionRange` from xgrammar
// `cpp/grammar_parser.cc`.

use super::quant::UNZIP_THRESHOLD;
use super::{EbnfParser, ParseError};
use crate::grammar::builder::CharacterClassElement;
use crate::grammar::expr::GrammarExprType;

impl EbnfParser {
    /// Expand `{lower, upper}` choosing literal unzipping for small
    /// counts and a `Repeat` expr for large ones. Mirrors C++
    /// `HandleRepetitionRange`.
    pub(super) fn handle_repetition_range(
        &mut self,
        grammar_expr_id: i32,
        mut lower: i64,
        mut upper: i64,
    ) -> Result<i32, ParseError> {
        if (upper != -1 && upper <= UNZIP_THRESHOLD) || (upper == -1 && lower <= UNZIP_THRESHOLD) {
            return self.legacy_handle_repetition_range(grammar_expr_id, lower, upper);
        }
        let mut choices: Vec<i32> = Vec::new();
        if lower < UNZIP_THRESHOLD {
            choices.push(self.legacy_handle_repetition_range(
                grammar_expr_id,
                lower,
                UNZIP_THRESHOLD - 1,
            )?);
            lower = UNZIP_THRESHOLD;
        }
        let mut infinite_repetition_id: Option<i32> = None;
        let mut repeated_sequence: Vec<i32> = Vec::new();
        if upper == -1 {
            infinite_repetition_id = Some(self.build_infinite_repetition(grammar_expr_id)?);
            upper = lower;
        }
        if let Some(id) = infinite_repetition_id {
            repeated_sequence.push(id);
        }
        if upper != UNZIP_THRESHOLD {
            let repeat_ref = self.build_large_repeat(grammar_expr_id, lower, upper)?;
            repeated_sequence.push(repeat_ref);
        }
        for _ in 0..UNZIP_THRESHOLD {
            repeated_sequence.push(grammar_expr_id);
        }
        choices.push(self.builder.add_sequence(&repeated_sequence));
        Ok(self.builder.add_choices(&choices))
    }

    /// Build the unbounded-repetition node for `{lower,}` — a
    /// `CharacterClassStar` for character classes, else a recursive
    /// rule `r ::= "" | a r`.
    fn build_infinite_repetition(&mut self, grammar_expr_id: i32) -> Result<i32, ParseError> {
        let rule_expr = self.builder.get_grammar_expr(grammar_expr_id);
        if rule_expr.kind == GrammarExprType::CharacterClass {
            let is_negative = rule_expr.data[0] != 0;
            let mut ranges: Vec<CharacterClassElement> = Vec::new();
            let mut i = 1;
            while i + 1 < rule_expr.data.len() {
                ranges.push(CharacterClassElement::new(
                    rule_expr.data[i],
                    rule_expr.data[i + 1],
                ));
                i += 2;
            }
            return Ok(self.builder.add_character_class_star(&ranges, is_negative));
        }
        let hint = format!("{}_repeat_inf", self.cur_rule_name);
        let name = self.builder.get_new_rule_name(&hint);
        let unbounded_rule_id = self.builder.add_empty_rule(name)?;
        let ref_rule = self.builder.add_rule_ref(unbounded_rule_id);
        let recursion_sequence = self.builder.add_sequence(&[grammar_expr_id, ref_rule]);
        let empty = self.builder.add_empty_str();
        let recursion_choice = self.builder.add_choices(&[empty, recursion_sequence]);
        self.builder
            .update_rule_body(unbounded_rule_id, recursion_choice)?;
        Ok(self.builder.add_rule_ref(unbounded_rule_id))
    }

    /// Build the `{lower, upper}` part where `threshold <= lower <= upper`,
    /// using a `Repeat` expr plus a `threshold`-wide lookahead.
    fn build_large_repeat(
        &mut self,
        grammar_expr_id: i32,
        lower: i64,
        upper: i64,
    ) -> Result<i32, ParseError> {
        let repeat_name = format!("{}_repeat_1", self.cur_rule_name);
        let inner_seq = self.builder.add_sequence(&[grammar_expr_id]);
        let new_grammar_expr_id = self.builder.add_choices(&[inner_seq]);
        let new_rule_id = self
            .builder
            .add_rule_with_hint(&repeat_name, new_grammar_expr_id)?;
        let repeat_expr = self.builder.add_repeat(
            new_rule_id,
            (lower - UNZIP_THRESHOLD) as i32,
            (upper - UNZIP_THRESHOLD) as i32,
        );
        let repeat_seq = self.builder.add_sequence(&[repeat_expr]);
        let repeated_ref = self.builder.add_choices(&[repeat_seq]);
        let inner_name = format!("{repeat_name}_inner");
        let new_repeated_rule_id = self.builder.add_rule_with_hint(&inner_name, repeated_ref)?;
        let lookahead: Vec<i32> = vec![grammar_expr_id; UNZIP_THRESHOLD as usize];
        let la = self.builder.add_sequence(&lookahead);
        self.builder.update_lookahead_assertion(new_rule_id, la)?;
        Ok(self.builder.add_rule_ref(new_repeated_rule_id))
    }
}
