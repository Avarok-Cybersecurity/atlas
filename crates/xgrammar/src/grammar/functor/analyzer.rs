// SPDX-License-Identifier: AGPL-3.0-only
//
// Grammar analysis passes — port of the visitor-style functors in
// `cpp/grammar_functor.cc`:
//   UsedRulesAnalyzer, RuleRefGraphFinder, AllowEmptyRuleAnalyzer.
//
// These are read-only visitors; they collect information rather than
// producing a new grammar, so they do not use the `GrammarMutator`
// trait. `LookaheadAssertionAnalyzer` (which does mutate) lives in
// `lookahead.rs`.

use std::collections::{BTreeSet, HashSet, VecDeque};

use crate::grammar::data::GrammarData;
use crate::grammar::expr::GrammarExprType;

/// Collect every rule reachable from the root rule (transitive closure
/// over rule refs, repeats and tag dispatches). Used for dead-code
/// elimination. Returns the reachable rule ids in ascending order.
pub struct UsedRulesAnalyzer;

impl UsedRulesAnalyzer {
    /// Run the analysis.
    pub fn apply(grammar: &GrammarData) -> Vec<i32> {
        let mut visited: BTreeSet<i32> = BTreeSet::new();
        let mut queue: VecDeque<i32> = VecDeque::new();
        queue.push_back(grammar.root_rule_id());
        while let Some(rule_id) = queue.pop_front() {
            if !visited.insert(rule_id) {
                continue;
            }
            let rule = grammar.rule(rule_id);
            collect_refs(grammar, rule.body_expr_id, &mut queue);
            if rule.lookahead_assertion_id != -1 {
                collect_refs(grammar, rule.lookahead_assertion_id, &mut queue);
            }
        }
        visited.into_iter().collect()
    }
}

/// Push every rule referenced anywhere under `expr_id` onto `queue`.
fn collect_refs(grammar: &GrammarData, expr_id: i32, queue: &mut VecDeque<i32>) {
    let e = grammar.expr(expr_id);
    match e.kind {
        GrammarExprType::RuleRef | GrammarExprType::Repeat => queue.push_back(e.data[0]),
        GrammarExprType::TagDispatch => {
            for (_, rule_id) in grammar.tag_dispatch(expr_id).tag_rule_pairs {
                queue.push_back(rule_id);
            }
        }
        GrammarExprType::Sequence | GrammarExprType::Choices => {
            for &child in e.data {
                collect_refs(grammar, child, queue);
            }
        }
        _ => {}
    }
}

/// Build the *inverted* rule reference graph: `graph[referee]` is the
/// sorted, deduplicated list of rules that reference `referee`.
pub struct RuleRefGraphFinder;

impl RuleRefGraphFinder {
    /// Run the analysis.
    pub fn apply(grammar: &GrammarData) -> Vec<Vec<i32>> {
        let num_rules = grammar.num_rules() as usize;
        let mut graph: Vec<Vec<i32>> = vec![Vec::new(); num_rules];
        for i in 0..grammar.num_rules() {
            let body = grammar.rule(i).body_expr_id;
            visit_refs(grammar, body, i, &mut graph);
        }
        for row in &mut graph {
            row.sort_unstable();
            row.dedup();
        }
        graph
    }
}

/// Record into `graph` every rule referenced under `expr_id`, attributed
/// to referer `cur_rule`.
fn visit_refs(grammar: &GrammarData, expr_id: i32, cur_rule: i32, graph: &mut [Vec<i32>]) {
    let e = grammar.expr(expr_id);
    match e.kind {
        GrammarExprType::RuleRef | GrammarExprType::Repeat => {
            graph[e.data[0] as usize].push(cur_rule);
        }
        GrammarExprType::TagDispatch => {
            for (_, rule_id) in grammar.tag_dispatch(expr_id).tag_rule_pairs {
                graph[rule_id as usize].push(cur_rule);
            }
        }
        GrammarExprType::Sequence | GrammarExprType::Choices => {
            for &child in e.data {
                visit_refs(grammar, child, cur_rule, graph);
            }
        }
        _ => {}
    }
}

/// Analyze which rules can match the empty string. Returns the sorted
/// list of nullable rule ids.
pub struct AllowEmptyRuleAnalyzer;

impl AllowEmptyRuleAnalyzer {
    /// Run the analysis.
    pub fn apply(grammar: &GrammarData) -> Vec<i32> {
        let mut empty: HashSet<i32> = HashSet::new();
        Self::find_explicit(grammar, &mut empty);
        let graph = RuleRefGraphFinder::apply(grammar);
        Self::find_indirect(grammar, &mut empty, &graph);
        let mut result: Vec<i32> = empty.into_iter().collect();
        result.sort_unstable();
        result
    }

    /// Rules that explicitly allow empty: a leading EmptyStr choice, a
    /// TagDispatch with `stop_eos`, or a sequence of only star classes.
    fn find_explicit(grammar: &GrammarData, empty: &mut HashSet<i32>) {
        for i in 0..grammar.num_rules() {
            let body_id = grammar.rule(i).body_expr_id;
            let body = grammar.expr(body_id);
            if body.kind == GrammarExprType::TagDispatch {
                if grammar.tag_dispatch(body_id).stop_eos {
                    empty.insert(i);
                }
                continue;
            }
            debug_assert_eq!(body.kind, GrammarExprType::Choices);
            if grammar.expr(body.data[0]).kind == GrammarExprType::EmptyStr {
                empty.insert(i);
                continue;
            }
            for &seq_id in body.data {
                let seq = grammar.expr(seq_id);
                if seq.kind == GrammarExprType::Sequence
                    && !seq.data.is_empty()
                    && seq
                        .data
                        .iter()
                        .all(|&x| grammar.expr(x).kind == GrammarExprType::CharacterClassStar)
                {
                    empty.insert(i);
                    break;
                }
            }
        }
    }

    /// True if a choice/sequence expr can collapse to epsilon given the
    /// currently-known empty rule set.
    fn seq_is_epsilon(grammar: &GrammarData, seq_id: i32, empty: &HashSet<i32>) -> bool {
        let seq = grammar.expr(seq_id);
        if seq.kind == GrammarExprType::EmptyStr {
            return true;
        }
        debug_assert_eq!(seq.kind, GrammarExprType::Sequence);
        seq.data.iter().all(|&x| {
            let elem = grammar.expr(x);
            match elem.kind {
                GrammarExprType::RuleRef => empty.contains(&elem.data[0]),
                GrammarExprType::CharacterClassStar => true,
                GrammarExprType::Repeat => empty.contains(&elem.data[0]) || elem.data[1] == 0,
                _ => false,
            }
        })
    }

    /// Propagate emptiness through the reference graph (worklist).
    fn find_indirect(grammar: &GrammarData, empty: &mut HashSet<i32>, graph: &[Vec<i32>]) {
        let mut queue: VecDeque<i32> = empty.iter().copied().collect();
        while let Some(rule_id) = queue.pop_front() {
            for &referer in &graph[rule_id as usize] {
                if empty.contains(&referer) {
                    continue;
                }
                let body = grammar.expr(grammar.rule(referer).body_expr_id);
                debug_assert_ne!(body.kind, GrammarExprType::TagDispatch);
                let is_epsilon = body
                    .data
                    .iter()
                    .any(|&seq_id| Self::seq_is_epsilon(grammar, seq_id, empty));
                if is_epsilon {
                    empty.insert(referer);
                    queue.push_back(referer);
                }
            }
        }
    }
}

#[cfg(test)]
#[path = "analyzer_tests.rs"]
mod tests;
