// SPDX-License-Identifier: AGPL-3.0-only
//
// Optimization pipeline — `RepetitionNormalizer` and `GrammarOptimizer`.
// Split out of `optimizer.rs` to keep each file under the 250-line cap.
// Port of the corresponding functors in `cpp/grammar_functor.cc`.

use crate::grammar::data::GrammarData;
use crate::grammar::expr::GrammarExprType;
use crate::grammar::functor::analyzer::AllowEmptyRuleAnalyzer;
use crate::grammar::functor::fsm_builder::GrammarFsmBuilder;
use crate::grammar::functor::lookahead::LookaheadAssertionAnalyzer;
use crate::grammar::functor::optimizer::{ByteStringFuser, DeadCodeEliminator, RuleInliner};

/// Normalize repetition ranges: if the repeated rule is nullable, the
/// minimum repeat count is forced to 0. Operates in place.
pub struct RepetitionNormalizer;

impl RepetitionNormalizer {
    /// Run the pass on `grammar` in place.
    pub fn apply(grammar: &mut GrammarData) {
        for i in 0..grammar.num_exprs() {
            let expr = grammar.expr(i);
            if expr.kind != GrammarExprType::Repeat {
                continue;
            }
            let repeat_rule_id = expr.data[0];
            grammar.rule_mut(repeat_rule_id).is_exact_lookahead = true;
            if grammar
                .allow_empty_rule_ids
                .binary_search(&repeat_rule_id)
                .is_ok()
            {
                grammar.set_expr_data(i, 1, 0);
            }
        }
    }
}

/// Full optimization pipeline. Port of `GrammarOptimizer`.
pub struct GrammarOptimizer;

impl GrammarOptimizer {
    /// Apply byte fusion, inlining, dead-code elimination, lookahead
    /// analysis, allow-empty analysis, repetition normalization and FSM
    /// construction. Always returns a new (optimized) grammar.
    pub fn apply(grammar: GrammarData) -> GrammarData {
        let mut result = ByteStringFuser::apply(grammar);
        result = RuleInliner::apply(result);
        result = DeadCodeEliminator::apply(result);
        result = LookaheadAssertionAnalyzer::apply(result);
        result.allow_empty_rule_ids = AllowEmptyRuleAnalyzer::apply(&result);
        RepetitionNormalizer::apply(&mut result);
        GrammarFsmBuilder::apply(&mut result);
        result.optimized = true;
        result
    }
}
