// SPDX-License-Identifier: AGPL-3.0-only
//
// Grammar constructors — port of `SubGrammarAdderImpl`,
// `GrammarUnionFunctorImpl` and `GrammarConcatFunctorImpl` from
// `cpp/grammar_functor.cc`.
//
// These build a new grammar out of one or more existing grammars
// (sub-grammar splicing, union, concatenation).

use crate::grammar::builder::{GrammarBuilder, TagDispatchSpec};
use crate::grammar::data::GrammarData;
use crate::grammar::expr::GrammarExprType;

/// Splice every rule of `sub_grammar` into `builder`, returning the new
/// rule id of `sub_grammar`'s root rule. Rule references, repeats and
/// tag dispatches are remapped to the freshly-allocated rule ids.
pub fn add_sub_grammar(builder: &mut GrammarBuilder, sub_grammar: &GrammarData) -> i32 {
    let num_rules = sub_grammar.num_rules();
    let mut new_ids: Vec<i32> = Vec::with_capacity(num_rules as usize);
    for i in 0..num_rules {
        let hint = builder.get_new_rule_name(&sub_grammar.rule(i).name);
        new_ids.push(builder.add_empty_rule(hint).expect("unique name"));
    }
    for i in 0..num_rules {
        let rule = sub_grammar.rule(i);
        let new_body = copy_expr(builder, sub_grammar, rule.body_expr_id, &new_ids);
        builder
            .update_rule_body(new_ids[i as usize], new_body)
            .expect("range");
        let new_la = if rule.lookahead_assertion_id == -1 {
            -1
        } else {
            copy_expr(builder, sub_grammar, rule.lookahead_assertion_id, &new_ids)
        };
        builder
            .update_lookahead_assertion(new_ids[i as usize], new_la)
            .expect("range");
    }
    new_ids[sub_grammar.root_rule_id() as usize]
}

/// Recursively copy expr `expr_id` from `src` into `builder`, remapping
/// rule ids through `id_map`.
fn copy_expr(builder: &mut GrammarBuilder, src: &GrammarData, expr_id: i32, id_map: &[i32]) -> i32 {
    let e = src.expr(expr_id);
    match e.kind {
        GrammarExprType::RuleRef => builder.add_rule_ref(id_map[e.data[0] as usize]),
        GrammarExprType::Repeat => {
            builder.add_repeat(id_map[e.data[0] as usize], e.data[1], e.data[2])
        }
        GrammarExprType::TagDispatch => {
            let td = src.tag_dispatch(expr_id);
            let spec = TagDispatchSpec {
                tag_rule_pairs: td
                    .tag_rule_pairs
                    .iter()
                    .map(|(t, r)| (t.clone(), id_map[*r as usize]))
                    .collect(),
                stop_eos: td.stop_eos,
                stop_str: td.stop_str,
                loop_after_dispatch: td.loop_after_dispatch,
                excluded_str: td.excluded_str,
            };
            builder.add_tag_dispatch(&spec)
        }
        GrammarExprType::Sequence => {
            let ids: Vec<i32> = e
                .data
                .iter()
                .map(|&c| copy_expr(builder, src, c, id_map))
                .collect();
            builder.add_sequence(&ids)
        }
        GrammarExprType::Choices => {
            let ids: Vec<i32> = e
                .data
                .iter()
                .map(|&c| copy_expr(builder, src, c, id_map))
                .collect();
            builder.add_choices(&ids)
        }
        _ => builder.add_grammar_expr(e.kind, e.data),
    }
}

/// Build a grammar accepting any string from any of `grammars`.
pub struct GrammarUnion;

impl GrammarUnion {
    /// Run the union construction.
    pub fn apply(grammars: &[GrammarData]) -> GrammarData {
        let mut builder = GrammarBuilder::new();
        let root_id = builder.add_empty_rule("root").expect("root");
        let mut choices: Vec<i32> = Vec::with_capacity(grammars.len());
        for g in grammars {
            let sub_root = add_sub_grammar(&mut builder, g);
            let rr = builder.add_rule_ref(sub_root);
            let seq = builder.add_sequence(&[rr]);
            choices.push(seq);
        }
        let body = builder.add_choices(&choices);
        builder.update_rule_body(root_id, body).expect("root range");
        builder.get_by_id(root_id).expect("root")
    }
}

/// Build a grammar accepting the concatenation of `grammars` in order.
pub struct GrammarConcat;

impl GrammarConcat {
    /// Run the concatenation construction.
    pub fn apply(grammars: &[GrammarData]) -> GrammarData {
        let mut builder = GrammarBuilder::new();
        let root_id = builder.add_empty_rule("root").expect("root");
        let mut seq: Vec<i32> = Vec::with_capacity(grammars.len());
        for g in grammars {
            let sub_root = add_sub_grammar(&mut builder, g);
            seq.push(builder.add_rule_ref(sub_root));
        }
        let seq_id = builder.add_sequence(&seq);
        let body = builder.add_choices(&[seq_id]);
        builder.update_rule_body(root_id, body).expect("root range");
        builder.get_by_id(root_id).expect("root")
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::functor::normalizer::GrammarNormalizer;
    use crate::grammar::parse_ebnf_default;

    fn g(ebnf: &str) -> GrammarData {
        GrammarNormalizer::apply(parse_ebnf_default(ebnf).expect("parse"))
    }

    #[test]
    fn union_of_two_grammars() {
        let a = g("root ::= \"a\"\n");
        let b = g("root ::= \"b\"\n");
        let u = GrammarUnion::apply(&[a, b]);
        assert_eq!(u.root_rule().name, "root");
        assert!(u.num_rules() >= 3);
    }

    #[test]
    fn concat_of_two_grammars() {
        let a = g("root ::= \"a\"\n");
        let b = g("root ::= \"b\"\n");
        let c = GrammarConcat::apply(&[a, b]);
        assert_eq!(c.root_rule().name, "root");
        assert!(c.num_rules() >= 3);
    }

    #[test]
    fn sub_grammar_adder_remaps_refs() {
        let mut builder = GrammarBuilder::new();
        let sub = g("root ::= inner\ninner ::= \"x\"\n");
        let root = add_sub_grammar(&mut builder, &sub);
        let result = builder.get_by_id(root).expect("get");
        assert_eq!(result.num_rules(), 2);
    }

    #[test]
    fn union_single_grammar() {
        let a = g("root ::= \"only\"\n");
        let u = GrammarUnion::apply(std::slice::from_ref(&a));
        assert_eq!(u.root_rule().name, "root");
    }
}
