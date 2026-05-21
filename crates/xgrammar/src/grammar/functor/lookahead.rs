// SPDX-License-Identifier: AGPL-3.0-only
//
// LookaheadAssertionAnalyzer — port of `LookaheadAssertionAnalyzerImpl`
// from `cpp/grammar_functor.cc`.
//
// Detects an exact lookahead assertion for each non-root rule: when a
// rule is referenced in exactly one mid-sequence position across the
// whole grammar, the suffix following that reference becomes the rule's
// lookahead assertion (and is marked exact).

use crate::grammar::data::GrammarData;
use crate::grammar::expr::GrammarExprType;

/// Lookahead-assertion detection pass. Operates on a *normalized*
/// grammar (rule bodies are choices-of-sequences or TagDispatch).
pub struct LookaheadAssertionAnalyzer;

impl LookaheadAssertionAnalyzer {
    /// Run the pass, returning the updated grammar.
    pub fn apply(grammar: GrammarData) -> GrammarData {
        let root = grammar.root_rule();
        if grammar.expr(root.body_expr_id).kind == GrammarExprType::TagDispatch {
            return grammar;
        }
        let mut g = grammar;
        let root_id = g.root_rule_id();
        for i in 0..g.num_rules() {
            if i == root_id {
                continue;
            }
            if g.rule(i).lookahead_assertion_id != -1 {
                let exact = is_exact_lookahead(&g, i);
                g.rule_mut(i).is_exact_lookahead = exact;
                continue;
            }
            if let Some(seq) = detect_lookahead(&g, i) {
                let seq_id = g.append_expr(GrammarExprType::Sequence, &seq);
                g.rule_mut(i).lookahead_assertion_id = seq_id;
                g.rule_mut(i).is_exact_lookahead = true;
            }
        }
        g
    }
}

/// Walk every rule body, calling `f` for each sequence expr. Returns
/// `None` to short-circuit (when a rule is referenced in a tail position
/// or by a tag dispatch — both disqualify a lookahead assertion).
fn walk_sequences<F>(grammar: &GrammarData, rule_id: i32, mut f: F) -> Option<()>
where
    F: FnMut(usize, &[i32]) -> Option<()>,
{
    for i in 0..grammar.num_rules() {
        let body_id = grammar.rule(i).body_expr_id;
        let body = grammar.expr(body_id);
        if body.kind == GrammarExprType::TagDispatch {
            for (_, rid) in grammar.tag_dispatch(body_id).tag_rule_pairs {
                if rid == rule_id {
                    return None;
                }
            }
            continue;
        }
        debug_assert_eq!(body.kind, GrammarExprType::Choices);
        let choices: Vec<i32> = body.data.to_vec();
        for seq_id in choices {
            let seq = grammar.expr(seq_id);
            if seq.kind != GrammarExprType::Sequence {
                continue;
            }
            // Tail-position reference disqualifies (unless self-ref).
            if let Some(&last) = seq.data.last() {
                let last_e = grammar.expr(last);
                if last_e.kind == GrammarExprType::RuleRef
                    && last_e.data[0] == rule_id
                    && i != rule_id
                {
                    return None;
                }
            }
            let data: Vec<i32> = seq.data.to_vec();
            f(i as usize, &data)?;
        }
    }
    Some(())
}

/// Whether an existing lookahead assertion is exact: the rule must be
/// referenced in exactly one mid-sequence position.
fn is_exact_lookahead(grammar: &GrammarData, rule_id: i32) -> bool {
    let mut found = false;
    let res = walk_sequences(grammar, rule_id, |_i, seq| {
        for j in 0..seq.len().saturating_sub(1) {
            let elem = grammar.expr(seq[j]);
            if elem.kind != GrammarExprType::RuleRef || elem.data[0] != rule_id {
                continue;
            }
            if found {
                return None;
            }
            found = true;
        }
        Some(())
    });
    res.is_some() && found
}

/// Detect a lookahead assertion: if `rule_id` is referenced in exactly
/// one mid-sequence position, return the suffix element ids after it.
fn detect_lookahead(grammar: &GrammarData, rule_id: i32) -> Option<Vec<i32>> {
    let mut found = false;
    let mut found_sequence: Vec<i32> = Vec::new();
    let res = walk_sequences(grammar, rule_id, |_i, seq| {
        for j in 0..seq.len().saturating_sub(1) {
            let elem = grammar.expr(seq[j]);
            if elem.kind != GrammarExprType::RuleRef || elem.data[0] != rule_id {
                continue;
            }
            if found {
                return None;
            }
            found = true;
            found_sequence.extend_from_slice(&seq[j + 1..]);
        }
        Some(())
    });
    if res.is_none() || !found {
        return None;
    }
    Some(found_sequence)
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
}
