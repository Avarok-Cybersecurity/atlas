// SPDX-License-Identifier: AGPL-3.0-only
//
// Grammar normalization passes (part 1) — port of the normalizer
// section of `cpp/grammar_functor.cc`:
//   SingleElementExprEliminator, RootRuleRenamer, GrammarNormalizer.
//
// `StructureNormalizer` lives in `structure_normalizer.rs` to keep
// each file under the 250-line cap.

use std::collections::HashSet;

use super::mutator::{GrammarMutator, MutatorState, OwnedExpr};
use super::structure_normalizer::StructureNormalizer;
use crate::grammar::data::GrammarData;
use crate::grammar::printer::codepoint_to_bytes;

/// Eliminates single-element sequences/choices and single-codepoint
/// character classes. `choices("a") -> "a"`, `[a-a] -> "a"`.
#[derive(Default)]
pub struct SingleElementExprEliminator {
    state: Option<MutatorState>,
}

impl SingleElementExprEliminator {
    /// Run the pass on `grammar`.
    pub fn apply(grammar: GrammarData) -> GrammarData {
        let mut p = Self::default();
        GrammarMutator::apply(&mut p, grammar)
    }
}

impl GrammarMutator for SingleElementExprEliminator {
    fn state(&mut self) -> &mut MutatorState {
        self.state
            .get_or_insert_with(|| MutatorState::new(GrammarData::new()))
    }

    fn visit_sequence(&mut self, e: &OwnedExpr) -> i32 {
        let ids: Vec<i32> = e.data.iter().map(|&c| self.visit_expr_id(c)).collect();
        if ids.len() == 1 {
            return ids[0];
        }
        self.state().builder.add_sequence(&ids)
    }

    fn visit_choices(&mut self, e: &OwnedExpr) -> i32 {
        let ids: Vec<i32> = e.data.iter().map(|&c| self.visit_expr_id(c)).collect();
        if ids.len() == 1 {
            return ids[0];
        }
        self.state().builder.add_choices(&ids)
    }

    fn visit_character_class(&mut self, e: &OwnedExpr) -> i32 {
        if e.data.len() == 3 && e.data[0] == 0 && e.data[1] == e.data[2] {
            let bytes = codepoint_to_bytes(e.data[1]);
            return self.state().builder.add_byte_string_bytes(&bytes);
        }
        self.state().builder.add_grammar_expr(e.kind, &e.data)
    }
}

/// Rename the root rule to `"root"`; if another rule is already named
/// `"root"`, give it a fresh `root_N` name.
pub struct RootRuleRenamer;

impl RootRuleRenamer {
    /// Run the pass.
    pub fn apply(grammar: GrammarData) -> GrammarData {
        if grammar.root_rule().name == "root" {
            return grammar;
        }
        let mut names: HashSet<String> = HashSet::new();
        let mut root_name_rule_id = -1;
        for i in 0..grammar.num_rules() {
            let n = &grammar.rule(i).name;
            if n == "root" {
                root_name_rule_id = i;
            }
            names.insert(n.clone());
        }
        let mut g = grammar;
        let root_id = g.root_rule_id();
        g.set_rule_name(root_id, "root");
        if root_name_rule_id != -1 {
            for i in 0..=g.num_rules() {
                let candidate = format!("root_{i}");
                if !names.contains(&candidate) {
                    g.set_rule_name(root_name_rule_id, candidate);
                    break;
                }
            }
        }
        g
    }
}

/// Normalize a grammar: rename the root rule, then structure-normalize.
/// Port of `GrammarNormalizer`.
pub struct GrammarNormalizer;

impl GrammarNormalizer {
    /// Run the full normalization pipeline.
    pub fn apply(grammar: GrammarData) -> GrammarData {
        let renamed = RootRuleRenamer::apply(grammar);
        StructureNormalizer::apply(renamed)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::grammar::parse_ebnf_default;
    use crate::grammar::printer::print_grammar;

    fn norm(ebnf: &str) -> GrammarData {
        GrammarNormalizer::apply(parse_ebnf_default(ebnf).expect("parse"))
    }

    #[test]
    fn normalize_simple_string() {
        let g = norm("root ::= \"abc\"\n");
        let out = print_grammar(&g);
        assert!(out.contains("root ::="));
        assert!(out.contains("\"abc\""));
    }

    #[test]
    fn normalize_choices() {
        let g = norm("root ::= \"a\" | \"b\"\n");
        let out = print_grammar(&g);
        assert!(out.contains(" | "));
    }

    #[test]
    fn root_renamer_keeps_root() {
        let g = parse_ebnf_default("root ::= \"x\"\n").unwrap();
        let renamed = RootRuleRenamer::apply(g);
        assert_eq!(renamed.root_rule().name, "root");
    }

    #[test]
    fn single_element_eliminator_collapses_seq() {
        let g = parse_ebnf_default("root ::= (\"a\")\n").unwrap();
        let out = print_grammar(&SingleElementExprEliminator::apply(g));
        assert!(out.contains("\"a\""));
    }

    #[test]
    fn normalize_nested_rule() {
        let g = norm("root ::= \"a\" | (\"b\" (\"c\" | \"d\"))\n");
        let out = print_grammar(&g);
        assert!(g.num_rules() >= 2, "{out}");
    }

    #[test]
    fn empty_string_rule() {
        let g = norm("root ::= \"\"\n");
        let out = print_grammar(&g);
        assert!(out.contains("\"\""));
    }

    #[test]
    fn normalize_idempotent_shape() {
        let g1 = norm("root ::= \"a\" \"b\"\n");
        let g2 = GrammarNormalizer::apply(g1.clone());
        assert_eq!(print_grammar(&g1), print_grammar(&g2));
    }
}
