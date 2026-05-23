// SPDX-License-Identifier: AGPL-3.0-only
//
// StructureNormalizer — port of `StructureNormalizerImpl` from
// `cpp/grammar_functor.cc`.
//
// Normalizes the structure of a grammar so every rule body is a
// choices-of-sequences-of-elements (or a TagDispatch). New rules are
// created where nesting must be flattened.

use super::mutator::{MutatorState, OwnedExpr, rebuild_tag_dispatch};
use super::normalizer::SingleElementExprEliminator;
use crate::grammar::data::GrammarData;
use crate::grammar::expr::GrammarExprType;

/// Structure-normalization pass.
#[derive(Default)]
pub struct StructureNormalizer {
    state: Option<MutatorState>,
}

impl StructureNormalizer {
    /// Run the pass: first `SingleElementExprEliminator`, then normalize.
    pub fn apply(grammar: GrammarData) -> GrammarData {
        let grammar = SingleElementExprEliminator::apply(grammar);
        let mut p = Self::default();
        *p.st() = MutatorState::new(grammar);
        let num_rules = p.st().base.num_rules();
        for i in 0..num_rules {
            let name = p.st().base.rule(i).name.clone();
            p.st().builder.add_empty_rule(name).expect("dup rule");
        }
        for i in 0..num_rules {
            let rule = p.st().base.rule(i).clone();
            p.st().cur_rule_name = rule.name.clone();
            let body = p.st().base.owned_expr(rule.body_expr_id);
            let new_body = p.visit_rule_body(&body);
            p.st().builder.update_rule_body(i, new_body).expect("range");
            let la = p.visit_lookahead(rule.lookahead_assertion_id);
            p.st()
                .builder
                .update_lookahead_assertion(i, la)
                .expect("range");
        }
        let root_name = p.st().base.root_rule().name.clone();
        let builder = std::mem::take(&mut p.st().builder);
        builder.get(&root_name).expect("root")
    }

    fn st(&mut self) -> &mut MutatorState {
        self.state
            .get_or_insert_with(|| MutatorState::new(GrammarData::new()))
    }

    fn visit_lookahead(&mut self, id: i32) -> i32 {
        if id == -1 {
            return -1;
        }
        let e = self.st().base.owned_expr(id);
        match e.kind {
            GrammarExprType::Sequence => {
                let ids = self.visit_sequence_(&e);
                self.st().builder.add_sequence(&ids)
            }
            GrammarExprType::Choices => panic!("Choices in lookahead assertion not supported"),
            GrammarExprType::EmptyStr => panic!("Empty string in lookahead assertion"),
            GrammarExprType::TagDispatch => panic!("TagDispatch in lookahead assertion"),
            _ => {
                let elem = self.st().builder.add_grammar_expr(e.kind, &e.data);
                self.st().builder.add_sequence(&[elem])
            }
        }
    }

    fn visit_rule_body(&mut self, e: &OwnedExpr) -> i32 {
        match e.kind {
            GrammarExprType::Sequence => {
                let seq = self.visit_sequence_(e);
                let s = self.st().builder.add_sequence(&seq);
                self.st().builder.add_choices(&[s])
            }
            GrammarExprType::Choices => {
                let ids = self.visit_choices_(e);
                self.st().builder.add_choices(&ids)
            }
            GrammarExprType::EmptyStr => {
                let empty = self.st().builder.add_empty_str();
                self.st().builder.add_choices(&[empty])
            }
            GrammarExprType::TagDispatch => {
                let td = e.tag_dispatch.clone().expect("td");
                rebuild_tag_dispatch(&mut self.st().builder, &td)
            }
            _ => {
                let elem = self.st().builder.add_grammar_expr(e.kind, &e.data);
                let seq = self.st().builder.add_sequence(&[elem]);
                self.st().builder.add_choices(&[seq])
            }
        }
    }

    /// Returns a list of new choice expr ids.
    fn visit_choices_(&mut self, e: &OwnedExpr) -> Vec<i32> {
        let mut new_choice_ids: Vec<i32> = Vec::new();
        let mut found_empty = false;
        for &i in &e.data {
            let choice = self.st().base.owned_expr(i);
            match choice.kind {
                GrammarExprType::Sequence => {
                    let sub = self.visit_sequence_(&choice);
                    if sub.is_empty() {
                        found_empty = true;
                    } else {
                        let s = self.st().builder.add_sequence(&sub);
                        new_choice_ids.push(s);
                    }
                }
                GrammarExprType::Choices => {
                    let sub = self.visit_choices_(&choice);
                    let contains_empty = self.st().builder.get_grammar_expr(sub[0]).kind
                        == GrammarExprType::EmptyStr;
                    if contains_empty {
                        found_empty = true;
                        new_choice_ids.extend_from_slice(&sub[1..]);
                    } else {
                        new_choice_ids.extend_from_slice(&sub);
                    }
                }
                GrammarExprType::EmptyStr => found_empty = true,
                GrammarExprType::TagDispatch => {
                    let td = choice.tag_dispatch.clone().expect("td");
                    let td_id = rebuild_tag_dispatch(&mut self.st().builder, &td);
                    let hint = self.st().cur_rule_name.clone();
                    let new_rule = self
                        .st()
                        .builder
                        .add_rule_with_hint(&hint, td_id)
                        .expect("hint");
                    let rr = self.st().builder.add_rule_ref(new_rule);
                    let s = self.st().builder.add_sequence(&[rr]);
                    new_choice_ids.push(s);
                }
                _ => {
                    let sub = self
                        .st()
                        .builder
                        .add_grammar_expr(choice.kind, &choice.data);
                    let s = self.st().builder.add_sequence(&[sub]);
                    new_choice_ids.push(s);
                }
            }
        }
        if found_empty {
            let empty = self.st().builder.add_empty_str();
            new_choice_ids.insert(0, empty);
        }
        assert!(!new_choice_ids.is_empty());
        new_choice_ids
    }

    /// Returns a list of new sequence element expr ids (flattened).
    fn visit_sequence_(&mut self, e: &OwnedExpr) -> Vec<i32> {
        let mut new_seq: Vec<i32> = Vec::new();
        for &i in &e.data {
            let elem = self.st().base.owned_expr(i);
            match elem.kind {
                GrammarExprType::Sequence => {
                    let sub = self.visit_sequence_(&elem);
                    new_seq.extend_from_slice(&sub);
                }
                GrammarExprType::Choices => {
                    let sub = self.visit_choices_(&elem);
                    if sub.len() == 1 {
                        let ce = self.st().builder.get_grammar_expr(sub[0]);
                        if ce.kind != GrammarExprType::EmptyStr {
                            let data: Vec<i32> = ce.data.to_vec();
                            new_seq.extend_from_slice(&data);
                        }
                    } else {
                        let c = self.st().builder.add_choices(&sub);
                        let hint = self.st().cur_rule_name.clone();
                        let nr = self
                            .st()
                            .builder
                            .add_rule_with_hint(&hint, c)
                            .expect("hint");
                        new_seq.push(self.st().builder.add_rule_ref(nr));
                    }
                }
                GrammarExprType::EmptyStr => {}
                GrammarExprType::TagDispatch => {
                    let td = elem.tag_dispatch.clone().expect("td");
                    let td_id = rebuild_tag_dispatch(&mut self.st().builder, &td);
                    let hint = self.st().cur_rule_name.clone();
                    let nr = self
                        .st()
                        .builder
                        .add_rule_with_hint(&hint, td_id)
                        .expect("hint");
                    new_seq.push(self.st().builder.add_rule_ref(nr));
                }
                _ => {
                    let s = self.st().builder.add_grammar_expr(elem.kind, &elem.data);
                    new_seq.push(s);
                }
            }
        }
        new_seq
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::parse_ebnf_default;
    use crate::grammar::printer::print_grammar;

    #[test]
    fn body_becomes_choices_of_sequences() {
        let g = parse_ebnf_default("root ::= \"a\" \"b\"\n").unwrap();
        let n = StructureNormalizer::apply(g);
        let out = print_grammar(&n);
        // sequence wrapped in choices: "((\"ab\")...)" or "((\"a\" \"b\")...)"
        assert!(out.starts_with("root ::= ("), "{out}");
    }

    #[test]
    fn nested_choices_flattened() {
        let g = parse_ebnf_default("root ::= \"a\" | (\"b\" | \"c\")\n").unwrap();
        let n = StructureNormalizer::apply(g);
        let out = print_grammar(&n);
        assert!(out.contains(" | "), "{out}");
    }

    #[test]
    fn inner_alternation_makes_new_rule() {
        let g = parse_ebnf_default("root ::= \"a\" (\"b\" | \"c\")\n").unwrap();
        let n = StructureNormalizer::apply(g);
        assert!(n.num_rules() >= 2);
    }

    #[test]
    fn empty_str_body() {
        let g = parse_ebnf_default("root ::= \"\"\n").unwrap();
        let n = StructureNormalizer::apply(g);
        assert!(print_grammar(&n).contains("\"\""));
    }
}
