// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use crate::grammar::functor::normalizer::GrammarNormalizer;
use crate::grammar::parse_ebnf_default;

fn normed(ebnf: &str) -> GrammarData {
    GrammarNormalizer::apply(parse_ebnf_default(ebnf).expect("parse"))
}

#[test]
fn used_rules_skips_unreferenced() {
    let g = normed("root ::= \"a\"\norphan ::= \"b\"\n");
    let used = UsedRulesAnalyzer::apply(&g);
    assert_eq!(used, vec![g.root_rule_id()]);
}

#[test]
fn used_rules_follows_refs() {
    let g = normed("root ::= sub\nsub ::= \"x\"\n");
    let used = UsedRulesAnalyzer::apply(&g);
    assert_eq!(used.len(), 2);
}

#[test]
fn ref_graph_is_inverted() {
    let g = normed("root ::= sub\nsub ::= \"x\"\n");
    let graph = RuleRefGraphFinder::apply(&g);
    let sub_id = (0..g.num_rules())
        .find(|&i| g.rule(i).name == "sub")
        .unwrap();
    // sub is referenced by root
    assert!(graph[sub_id as usize].contains(&g.root_rule_id()));
}

#[test]
fn allow_empty_detects_explicit() {
    let g = normed("root ::= \"\" | \"a\"\n");
    let empty = AllowEmptyRuleAnalyzer::apply(&g);
    assert!(empty.contains(&g.root_rule_id()));
}

#[test]
fn allow_empty_detects_indirect() {
    let g = normed("root ::= sub\nsub ::= \"\"\n");
    let empty = AllowEmptyRuleAnalyzer::apply(&g);
    assert_eq!(empty.len(), 2);
}

#[test]
fn non_empty_rule_excluded() {
    let g = normed("root ::= \"abc\"\n");
    let empty = AllowEmptyRuleAnalyzer::apply(&g);
    assert!(empty.is_empty());
}
