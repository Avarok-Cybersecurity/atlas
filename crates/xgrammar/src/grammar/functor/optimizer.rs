// SPDX-License-Identifier: AGPL-3.0-only
//
// Grammar optimization passes — port of `cpp/grammar_functor.cc`:
//   ByteStringFuser, RuleInliner, DeadCodeEliminator.
//
// `RepetitionNormalizer` + `GrammarOptimizer` live in
// `optimizer_pipeline.rs` (250-line cap).

use std::collections::HashMap;

use super::analyzer::UsedRulesAnalyzer;
use super::mutator::{GrammarMutator, MutatorState, OwnedExpr, rebuild_tag_dispatch};
use crate::grammar::data::GrammarData;
use crate::grammar::expr::GrammarExprType;

/// Fuse adjacent byte-string elements inside sequences.
/// `("ab" "cd") -> ("abcd")`.
#[derive(Default)]
pub struct ByteStringFuser {
    state: Option<MutatorState>,
}

impl ByteStringFuser {
    /// Run the pass.
    pub fn apply(grammar: GrammarData) -> GrammarData {
        let mut p = Self::default();
        GrammarMutator::apply(&mut p, grammar)
    }
}

impl GrammarMutator for ByteStringFuser {
    fn state(&mut self) -> &mut MutatorState {
        self.state
            .get_or_insert_with(|| MutatorState::new(GrammarData::new()))
    }

    fn visit_sequence(&mut self, e: &OwnedExpr) -> i32 {
        let mut new_seq: Vec<i32> = Vec::new();
        let mut cur_bytes: Vec<i32> = Vec::new();
        for &i in &e.data {
            let elem = self.state().base.owned_expr(i);
            if elem.kind == GrammarExprType::ByteString {
                cur_bytes.extend_from_slice(&elem.data);
            } else {
                if !cur_bytes.is_empty() {
                    let bs = self.state().builder.add_byte_string_bytes(&cur_bytes);
                    new_seq.push(bs);
                    cur_bytes.clear();
                }
                let s = self.state().builder.add_grammar_expr(elem.kind, &elem.data);
                new_seq.push(s);
            }
        }
        if !cur_bytes.is_empty() {
            let bs = self.state().builder.add_byte_string_bytes(&cur_bytes);
            new_seq.push(bs);
        }
        self.state().builder.add_sequence(&new_seq)
    }
}

/// Inline rule references that start a sequence, when the referenced
/// rule is a non-empty choices-of-sequences with no rule references.
#[derive(Default)]
pub struct RuleInliner {
    state: Option<MutatorState>,
    can_inline: HashMap<i32, bool>,
}

impl RuleInliner {
    /// Run the pass.
    pub fn apply(grammar: GrammarData) -> GrammarData {
        let mut p = Self::default();
        GrammarMutator::apply(&mut p, grammar)
    }

    fn check_can_inline(&self, rule_id: i32) -> bool {
        let base = &self.state_ref().base;
        let rule = base.rule(rule_id);
        let body = base.expr(rule.body_expr_id);
        if body.kind != GrammarExprType::Choices || body.is_empty() {
            return false;
        }
        for &cid in body.data {
            let choice = base.expr(cid);
            if choice.kind == GrammarExprType::EmptyStr {
                return false;
            }
            assert_eq!(choice.kind, GrammarExprType::Sequence);
            for &eid in choice.data {
                if base.expr(eid).kind == GrammarExprType::RuleRef {
                    return false;
                }
            }
        }
        true
    }

    fn state_ref(&self) -> &MutatorState {
        self.state.as_ref().expect("state initialized")
    }
}

impl GrammarMutator for RuleInliner {
    fn state(&mut self) -> &mut MutatorState {
        self.state
            .get_or_insert_with(|| MutatorState::new(GrammarData::new()))
    }

    fn visit_choices(&mut self, e: &OwnedExpr) -> i32 {
        let mut new_choice_ids: Vec<i32> = Vec::new();
        for &cid in &e.data {
            let choice = self.state().base.owned_expr(cid);
            if choice.kind == GrammarExprType::EmptyStr {
                new_choice_ids.push(self.visit_expr_id(cid));
                continue;
            }
            assert_eq!(choice.kind, GrammarExprType::Sequence);
            let first = self.state().base.owned_expr(choice.data[0]);
            if first.kind != GrammarExprType::RuleRef {
                new_choice_ids.push(self.visit_expr(&choice));
                continue;
            }
            let rule_ref_id = first.data[0];
            if !self.can_inline.contains_key(&rule_ref_id) {
                let v = self.check_can_inline(rule_ref_id);
                self.can_inline.insert(rule_ref_id, v);
            }
            if !self.can_inline[&rule_ref_id] {
                new_choice_ids.push(self.visit_expr(&choice));
                continue;
            }

            // Inline: visit the trailing elements of this choice...
            let other: Vec<i32> = choice.data[1..]
                .iter()
                .map(|&x| self.visit_expr_id(x))
                .collect();
            // ...and splice each choice of the referenced rule before them.
            let ref_body_id = self.state().base.rule(rule_ref_id).body_expr_id;
            let ref_choices: Vec<i32> = self.state().base.expr(ref_body_id).data.to_vec();
            for ref_cid in ref_choices {
                let ref_choice = self.state().base.owned_expr(ref_cid);
                assert_eq!(ref_choice.kind, GrammarExprType::Sequence);
                let mut to_add: Vec<i32> = ref_choice
                    .data
                    .iter()
                    .map(|&x| self.visit_expr_id(x))
                    .collect();
                to_add.extend_from_slice(&other);
                let s = self.state().builder.add_sequence(&to_add);
                new_choice_ids.push(s);
            }
        }
        self.state().builder.add_choices(&new_choice_ids)
    }
}

/// Eliminate rules not reachable from the root.
#[derive(Default)]
pub struct DeadCodeEliminator {
    state: Option<MutatorState>,
    rule_id_map: HashMap<i32, i32>,
}

impl DeadCodeEliminator {
    /// Run the pass.
    pub fn apply(grammar: GrammarData) -> GrammarData {
        let mut p = Self::default();
        *p.st() = MutatorState::new(grammar);
        let used = UsedRulesAnalyzer::apply(&p.st().base);
        for &rule_id in &used {
            let name = p.st().base.rule(rule_id).name.clone();
            let new_id = p.st().builder.add_empty_rule(name).expect("dup");
            p.rule_id_map.insert(rule_id, new_id);
        }
        for &rule_id in &used {
            let rule = p.st().base.rule(rule_id).clone();
            let new_id = p.rule_id_map[&rule_id];
            let new_body = p.visit_expr_id(rule.body_expr_id);
            p.st()
                .builder
                .update_rule_body(new_id, new_body)
                .expect("range");
            let new_la = p.visit_lookahead(rule.lookahead_assertion_id);
            p.st()
                .builder
                .update_lookahead_assertion(new_id, new_la)
                .expect("range");
        }
        let root_old = p.st().base.root_rule_id();
        let root_new = p.rule_id_map[&root_old];
        let builder = std::mem::take(&mut p.st().builder);
        builder.get_by_id(root_new).expect("root mapped")
    }

    fn st(&mut self) -> &mut MutatorState {
        self.state
            .get_or_insert_with(|| MutatorState::new(GrammarData::new()))
    }
}

impl GrammarMutator for DeadCodeEliminator {
    fn state(&mut self) -> &mut MutatorState {
        self.st()
    }

    fn visit_tag_dispatch(&mut self, e: &OwnedExpr) -> i32 {
        let mut td = e.tag_dispatch.clone().expect("td");
        for pair in &mut td.tag_rule_pairs {
            pair.1 = self.rule_id_map[&pair.1];
        }
        rebuild_tag_dispatch(&mut self.state().builder, &td)
    }

    fn visit_rule_ref(&mut self, e: &OwnedExpr) -> i32 {
        let new_id = self.rule_id_map[&e.data[0]];
        self.state().builder.add_rule_ref(new_id)
    }

    fn visit_repeat(&mut self, e: &OwnedExpr) -> i32 {
        let new_id = self.rule_id_map[&e.data[0]];
        self.state()
            .builder
            .add_repeat(new_id, e.data[1], e.data[2])
    }
}
#[path = "optimizer_pipeline.rs"]
mod optimizer_pipeline;
pub use optimizer_pipeline::{GrammarOptimizer, RepetitionNormalizer};

#[cfg(test)]
#[path = "optimizer_tests.rs"]
mod tests;
