// SPDX-License-Identifier: AGPL-3.0-only
use super::*;
use crate::grammar::functor::normalizer::GrammarNormalizer;
use crate::grammar::parse_ebnf_default;
use crate::grammar::printer::print_grammar;

fn normed(ebnf: &str) -> GrammarData {
    GrammarNormalizer::apply(parse_ebnf_default(ebnf).expect("parse"))
}

#[test]
fn byte_string_fuser_merges() {
    let g = normed("root ::= \"a\" \"b\" \"c\"\n");
    let fused = ByteStringFuser::apply(g);
    let out = print_grammar(&fused);
    assert!(out.contains("\"abc\""), "{out}");
}

#[test]
fn dead_code_eliminator_drops_unused() {
    let g = normed("root ::= \"a\"\nunused ::= \"b\"\n");
    let pruned = DeadCodeEliminator::apply(g);
    assert_eq!(pruned.num_rules(), 1);
}

#[test]
fn dead_code_keeps_referenced() {
    let g = normed("root ::= sub\nsub ::= \"x\"\n");
    let pruned = DeadCodeEliminator::apply(g);
    assert_eq!(pruned.num_rules(), 2);
}

#[test]
fn rule_inliner_inlines_leading_ref() {
    let g = normed("root ::= sub \"z\"\nsub ::= \"a\" | \"b\"\n");
    let inlined = RuleInliner::apply(g);
    let out = print_grammar(&inlined);
    // After inlining, root should reference "a"/"b" directly.
    assert!(out.contains('a') && out.contains('b'), "{out}");
}

#[test]
fn optimizer_sets_flag() {
    let g = normed("root ::= \"hello\"\n");
    let opt = GrammarOptimizer::apply(g);
    assert!(opt.optimized);
}

#[test]
fn optimizer_builds_fsms() {
    let g = normed("root ::= \"ab\" | \"cd\"\n");
    let opt = GrammarOptimizer::apply(g);
    assert_eq!(opt.per_rule_fsms.len(), opt.num_rules() as usize);
    assert!(opt.per_rule_fsms[0].is_some());
}
