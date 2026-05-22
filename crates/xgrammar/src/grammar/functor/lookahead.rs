// SPDX-License-Identifier: AGPL-3.0-only
//
// LookaheadAssertionAnalyzer — port of `LookaheadAssertionAnalyzerImpl`
// from `cpp/grammar_functor.cc`.
//
// Detects an exact lookahead assertion for each non-root rule: when a
// rule is referenced in exactly one mid-sequence position across the
// whole grammar, the suffix following that reference becomes the rule's
// lookahead assertion (and is marked exact).
//
// Performance: per-rule lookahead facts are pre-computed in a single
// O(N) pass (`build_rule_lookahead_info`) instead of re-scanning every
// rule body for every rule (which is O(N^2)). Port of upstream
// xgrammar commit 96ae88b. Behavior is unchanged: the per-rule
// predicates and the derived suffix sequence are identical to the old
// per-rule scan.

use crate::grammar::data::GrammarData;
use crate::grammar::expr::GrammarExprType;

/// Lookahead-assertion detection pass. Operates on a *normalized*
/// grammar (rule bodies are choices-of-sequences or TagDispatch).
pub struct LookaheadAssertionAnalyzer;

/// Pre-computed per-rule lookahead facts. Port of upstream
/// `RuleLookaheadInfo`.
#[derive(Debug, Clone, Default)]
struct RuleLookaheadInfo {
    /// The rule is the target of a tag-dispatch trigger.
    is_triggered_by_dispatch: bool,
    /// The rule appears as the last element of some *other* rule's
    /// sequence (a tail reference).
    appears_as_last_in_other_rule: bool,
    /// How many times the rule is referenced in a non-last (mid-)
    /// position across all sequences.
    non_last_occurrence_count: i32,
    /// The element ids following the *first* mid-position reference.
    suffix_after_first_occurrence: Vec<i32>,
}

impl LookaheadAssertionAnalyzer {
    /// Run the pass, returning the updated grammar.
    pub fn apply(grammar: GrammarData) -> GrammarData {
        let root = grammar.root_rule();
        if grammar.expr(root.body_expr_id).kind == GrammarExprType::TagDispatch {
            return grammar;
        }
        let mut g = grammar;
        let root_id = g.root_rule_id();
        let infos = build_rule_lookahead_info(&g);
        for i in 0..g.num_rules() {
            if i == root_id {
                continue;
            }
            if g.rule(i).lookahead_assertion_id != -1 {
                // `is_exact_lookahead` == `can_use_derived_lookahead`.
                g.rule_mut(i).is_exact_lookahead = can_use_derived_lookahead(&infos, i);
                continue;
            }
            if can_use_derived_lookahead(&infos, i) {
                let seq = infos[i as usize].suffix_after_first_occurrence.clone();
                let seq_id = g.append_expr(GrammarExprType::Sequence, &seq);
                g.rule_mut(i).lookahead_assertion_id = seq_id;
                g.rule_mut(i).is_exact_lookahead = true;
            }
        }
        g
    }
}

/// A rule can use a derived lookahead assertion iff it is not triggered
/// by a dispatch, never appears as a tail reference, and is referenced
/// in exactly one mid-sequence position. Port of upstream
/// `CanUseDerivedLookahead`.
fn can_use_derived_lookahead(infos: &[RuleLookaheadInfo], rule_id: i32) -> bool {
    let info = &infos[rule_id as usize];
    !info.is_triggered_by_dispatch
        && !info.appears_as_last_in_other_rule
        && info.non_last_occurrence_count == 1
}

/// Single O(N) pass over all rule bodies computing the lookahead facts
/// for every rule. Port of upstream `BuildRuleLookaheadInfo`.
fn build_rule_lookahead_info(grammar: &GrammarData) -> Vec<RuleLookaheadInfo> {
    let mut infos: Vec<RuleLookaheadInfo> =
        vec![RuleLookaheadInfo::default(); grammar.num_rules() as usize];
    for i in 0..grammar.num_rules() {
        let body_id = grammar.rule(i).body_expr_id;
        let body = grammar.expr(body_id);
        if body.kind == GrammarExprType::TagDispatch {
            for (_, rid) in grammar.tag_dispatch(body_id).tag_rule_pairs {
                infos[rid as usize].is_triggered_by_dispatch = true;
            }
            continue;
        }
        debug_assert_eq!(body.kind, GrammarExprType::Choices);
        let choices: Vec<i32> = body.data.to_vec();
        for seq_id in choices {
            let seq = grammar.expr(seq_id);
            if seq.kind != GrammarExprType::Sequence || seq.data.is_empty() {
                continue;
            }
            let seq_data: Vec<i32> = seq.data.to_vec();
            // Tail reference: last element is a RuleRef in a *different*
            // rule's sequence.
            if let Some(&last) = seq_data.last() {
                let last_e = grammar.expr(last);
                if last_e.kind == GrammarExprType::RuleRef && i != last_e.data[0] {
                    infos[last_e.data[0] as usize].appears_as_last_in_other_rule = true;
                }
            }
            // Mid-position references: every RuleRef before the last
            // element counts toward its target's occurrence count, and
            // the first such reference records the suffix after it.
            for j in 0..seq_data.len().saturating_sub(1) {
                let elem = grammar.expr(seq_data[j]);
                if elem.kind != GrammarExprType::RuleRef {
                    continue;
                }
                let target = elem.data[0] as usize;
                let info = &mut infos[target];
                if info.non_last_occurrence_count == 0 {
                    info.suffix_after_first_occurrence = seq_data[j + 1..].to_vec();
                }
                info.non_last_occurrence_count += 1;
            }
        }
    }
    infos
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::functor::normalizer::GrammarNormalizer;
    use crate::grammar::parse_ebnf_default;

    fn normed(ebnf: &str) -> GrammarData {
        GrammarNormalizer::apply(parse_ebnf_default(ebnf).expect("parse"))
    }

    #[test]
    fn detects_suffix_lookahead() {
        // `sub` is referenced mid-sequence in root, followed by "z".
        let g = normed("root ::= sub \"z\"\nsub ::= \"a\"\n");
        let analyzed = LookaheadAssertionAnalyzer::apply(g);
        let sub_id = (0..analyzed.num_rules())
            .find(|&i| analyzed.rule(i).name == "sub")
            .unwrap();
        assert_ne!(analyzed.rule(sub_id).lookahead_assertion_id, -1);
        assert!(analyzed.rule(sub_id).is_exact_lookahead);
    }

    #[test]
    fn tail_reference_no_lookahead() {
        // `sub` is only at the tail of root — no lookahead.
        let g = normed("root ::= \"z\" sub\nsub ::= \"a\"\n");
        let analyzed = LookaheadAssertionAnalyzer::apply(g);
        let sub_id = (0..analyzed.num_rules())
            .find(|&i| analyzed.rule(i).name == "sub")
            .unwrap();
        assert_eq!(analyzed.rule(sub_id).lookahead_assertion_id, -1);
    }

    #[test]
    fn tag_dispatch_root_passthrough() {
        let g = normed("root ::= TagDispatch((\"a\", sub))\nsub ::= \"x\"\n");
        let analyzed = LookaheadAssertionAnalyzer::apply(g.clone());
        assert_eq!(analyzed.num_rules(), g.num_rules());
    }

    #[test]
    fn multiple_references_no_lookahead() {
        let g = normed("root ::= sub \"y\" | sub \"z\"\nsub ::= \"a\"\n");
        let analyzed = LookaheadAssertionAnalyzer::apply(g);
        let sub_id = (0..analyzed.num_rules())
            .find(|&i| analyzed.rule(i).name == "sub")
            .unwrap();
        assert_eq!(analyzed.rule(sub_id).lookahead_assertion_id, -1);
    }

    #[test]
    fn precompute_pass_matches_predicates() {
        // One mid-position reference of `sub` → derivable lookahead.
        let g = normed("root ::= sub \"z\"\nsub ::= \"a\"\n");
        let infos = build_rule_lookahead_info(&g);
        let sub_id = (0..g.num_rules())
            .find(|&i| g.rule(i).name == "sub")
            .unwrap();
        assert!(can_use_derived_lookahead(&infos, sub_id));
        assert_eq!(infos[sub_id as usize].non_last_occurrence_count, 1);
        assert!(!infos[sub_id as usize].appears_as_last_in_other_rule);
        assert!(!infos[sub_id as usize].is_triggered_by_dispatch);
        assert!(
            !infos[sub_id as usize]
                .suffix_after_first_occurrence
                .is_empty()
        );
    }

    #[test]
    fn precompute_pass_counts_dispatch_trigger() {
        let g = normed("root ::= TagDispatch((\"a\", sub))\nsub ::= \"x\"\n");
        let infos = build_rule_lookahead_info(&g);
        let sub_id = (0..g.num_rules())
            .find(|&i| g.rule(i).name == "sub")
            .unwrap();
        assert!(infos[sub_id as usize].is_triggered_by_dispatch);
        assert!(!can_use_derived_lookahead(&infos, sub_id));
    }
}
