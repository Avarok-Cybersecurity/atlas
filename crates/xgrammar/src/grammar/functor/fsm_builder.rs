// SPDX-License-Identifier: AGPL-3.0-only
//
// GrammarFsmBuilder — builds per-rule FSMs for an (optimized) grammar.
// Port of `GrammarFSMBuilderImpl` from `cpp/grammar_functor.cc`.
//
// The W2 FSM agent explicitly deferred grammar-aware FSM construction to
// the functor wave: this is it. Each rule body (a choices-of-sequences
// or a TagDispatch) is converted to an `FsmWithStartEnd`, then spliced
// into one shared `complete_fsm`. `GrammarData::per_rule_fsms[i]` holds
// a compact view into that shared FSM for rule `i`.

use crate::fsm::{CompactFsmWithStartEnd, Fsm, FsmWithStartEnd};
use crate::grammar::data::GrammarData;
use crate::grammar::expr::{GrammarExpr, GrammarExprType};

use super::char_range::{add_character_range, codepoint_to_packed_utf8};
use super::tag_dispatch_fsm::build_tag_dispatch_fsm;

/// Per-rule FSM construction pass.
pub struct GrammarFsmBuilder;

impl GrammarFsmBuilder {
    /// Build `complete_fsm` + `per_rule_fsms` for `grammar` in place.
    pub fn apply(grammar: &mut GrammarData) {
        let num_rules = grammar.num_rules();
        let mut complete = Fsm::with_states(0);
        // Each entry: the spliced view plus the *sub-FSM's* node/edge
        // counts captured before splicing — needed so the per-rule
        // `CompactFsmWithStartEnd` reports the sub-FSM size rather than
        // the whole `complete_fsm` (upstream commit 58494db, #600).
        let mut per_rule: Vec<Option<(FsmWithStartEnd, usize, usize)>> =
            Vec::with_capacity(num_rules as usize);

        for i in 0..num_rules {
            let body_id = grammar.rule(i).body_expr_id;
            let body = grammar.expr(body_id);
            let rule_fsm: Option<FsmWithStartEnd> = if body.kind == GrammarExprType::TagDispatch {
                build_tag_dispatch_fsm(&grammar.tag_dispatch(body_id))
            } else {
                debug_assert_eq!(body.kind, GrammarExprType::Choices);
                build_choices(&body, grammar)
            };
            per_rule.push(rule_fsm.map(|f| {
                let node_num = f.num_states();
                let edge_num = f.fsm().num_edges();
                (splice_into(&mut complete, &f), node_num, edge_num)
            }));
        }

        let compact_complete = complete.to_compact();
        let final_per_rule: Vec<Option<CompactFsmWithStartEnd>> = per_rule
            .into_iter()
            .map(|opt| {
                opt.map(|(view, node_num, edge_num)| {
                    CompactFsmWithStartEnd::new_view(
                        compact_complete.clone(),
                        view.start(),
                        view.ends().to_vec(),
                        node_num,
                        edge_num,
                    )
                })
            })
            .collect();

        grammar.complete_fsm = compact_complete;
        grammar.per_rule_fsms = final_per_rule;
    }
}

/// Splice `sub` into `complete`, returning a view whose start/ends point
/// at the spliced-in states.
fn splice_into(complete: &mut Fsm, sub: &FsmWithStartEnd) -> FsmWithStartEnd {
    let mapping = complete.add_fsm(sub.fsm());
    let new_start = mapping[sub.start()];
    let mut new_ends = vec![false; complete.num_states()];
    for s in 0..sub.num_states() {
        if sub.is_end_state(s) {
            new_ends[mapping[s]] = true;
        }
    }
    FsmWithStartEnd::new(complete.clone(), new_start, new_ends, false)
}

/// FSM for a `RuleRef` element: a single rule-ref edge.
pub fn build_rule_ref(expr: &GrammarExpr<'_>) -> FsmWithStartEnd {
    let mut fsm = FsmWithStartEnd::default();
    fsm.add_state();
    fsm.add_state();
    fsm.set_start_state(0);
    fsm.add_end_state(1);
    fsm.fsm_mut().add_rule_edge(0, 1, expr.data[0] as i16);
    fsm
}

/// FSM for a `Repeat` element: a single repeat-ref edge.
pub fn build_repeat(expr: &GrammarExpr<'_>) -> FsmWithStartEnd {
    let mut fsm = FsmWithStartEnd::default();
    fsm.add_state();
    fsm.add_state();
    fsm.set_start_state(0);
    fsm.add_end_state(1);
    fsm.fsm_mut()
        .add_repeat_edge(0, 1, expr.data[0], expr.data[1], expr.data[2]);
    fsm
}

/// FSM for a `ByteString` element: a linear chain of byte edges.
pub fn build_byte_string(expr: &GrammarExpr<'_>) -> FsmWithStartEnd {
    let mut fsm = FsmWithStartEnd::default();
    let mut cur = fsm.add_state();
    fsm.set_start_state(cur);
    for &byte in expr.data {
        let next = fsm.add_state();
        fsm.fsm_mut().add_edge(cur, next, byte as i16, byte as i16);
        cur = next;
    }
    fsm.add_end_state(cur);
    fsm
}

/// FSM for a `CharacterClass` / `CharacterClassStar` element.
pub fn build_character_class(expr: &GrammarExpr<'_>) -> FsmWithStartEnd {
    let is_negative = expr.data[0] != 0;
    if is_negative {
        return build_negative_character_class(expr);
    }
    let mut fsm = FsmWithStartEnd::default();
    let start = fsm.add_state();
    fsm.set_start_state(start);
    let is_star = expr.kind == GrammarExprType::CharacterClassStar;
    let end = if is_star { start } else { fsm.add_state() };
    fsm.add_end_state(end);
    let mut i = 1;
    while i < expr.data.len() {
        let lo = codepoint_to_packed_utf8(expr.data[i] as u32);
        let hi = codepoint_to_packed_utf8(expr.data[i + 1] as u32);
        add_character_range(&mut fsm, start, end, lo, hi);
        i += 2;
    }
    fsm
}

/// FSM for a negated character class — the complement of the ASCII set
/// the class names, plus all multi-byte unicode.
fn build_negative_character_class(expr: &GrammarExpr<'_>) -> FsmWithStartEnd {
    let mut char_set = [false; 128];
    let mut i = 1;
    while i < expr.data.len() {
        let lo = expr.data[i];
        let mut hi = expr.data[i + 1];
        if hi > 128 {
            hi = 127;
        }
        for j in lo..=hi {
            if (0..128).contains(&j) {
                char_set[j as usize] = true;
            }
        }
        i += 2;
    }
    let mut fsm = FsmWithStartEnd::default();
    let start = fsm.add_state();
    fsm.set_start_state(start);
    let is_star = expr.kind == GrammarExprType::CharacterClassStar;
    let end = if is_star { start } else { fsm.add_state() };
    fsm.add_end_state(end);
    let mut i = 0;
    while i < 128 {
        if !char_set[i] {
            let mut right = i + 1;
            while right < 128 && !char_set[right] {
                right += 1;
            }
            fsm.fsm_mut()
                .add_edge(start, end, i as i16, (right - 1) as i16);
            i = right;
        } else {
            i += 1;
        }
    }
    use super::char_range::{MAX_4B, MIN_2B};
    add_character_range(&mut fsm, start, end, MIN_2B, MAX_4B);
    fsm
}

/// FSM for a normalized `Sequence` (a list of element exprs).
/// Returns `None` if any element is not FSM-expressible.
pub fn build_sequence(expr: &GrammarExpr<'_>, grammar: &GrammarData) -> Option<FsmWithStartEnd> {
    let mut parts: Vec<FsmWithStartEnd> = Vec::new();
    for &id in expr.data {
        let e = grammar.expr(id);
        let part = match e.kind {
            GrammarExprType::ByteString => build_byte_string(&e),
            GrammarExprType::RuleRef => build_rule_ref(&e),
            GrammarExprType::CharacterClass | GrammarExprType::CharacterClassStar => {
                build_character_class(&e)
            }
            GrammarExprType::Repeat => build_repeat(&e),
            _ => return None,
        };
        parts.push(part);
    }
    if parts.is_empty() {
        return Some(empty_fsm());
    }
    Some(FsmWithStartEnd::concat(&parts))
}

/// FSM for a normalized rule body — a `Choices` of `Sequence`s (with an
/// optional leading `EmptyStr`). Returns `None` if not FSM-expressible.
pub fn build_choices(expr: &GrammarExpr<'_>, grammar: &GrammarData) -> Option<FsmWithStartEnd> {
    debug_assert_eq!(expr.kind, GrammarExprType::Choices);
    let mut fsm_list: Vec<FsmWithStartEnd> = Vec::new();
    let mut nullable = false;
    for &id in expr.data {
        let e = grammar.expr(id);
        if e.kind == GrammarExprType::EmptyStr {
            nullable = true;
            continue;
        }
        debug_assert_eq!(e.kind, GrammarExprType::Sequence);
        fsm_list.push(build_sequence(&e, grammar)?);
    }
    if fsm_list.is_empty() {
        return Some(empty_fsm());
    }
    if nullable {
        fsm_list.push(empty_fsm());
    }
    let mut result = FsmWithStartEnd::union(&fsm_list);
    result = result.simplify_epsilon();
    result = result.merge_equivalent_successors();
    // Upstream commit 96ae88b (#616) dropped the `MinimizeDFA` call here:
    // Hopcroft minimization does not reduce states on these grammars and
    // scales super-linearly in state count, so it was pure wasted work.
    Some(result)
}

/// A one-state FSM accepting only the empty string.
pub fn empty_fsm() -> FsmWithStartEnd {
    let mut fsm = FsmWithStartEnd::default();
    fsm.add_state();
    fsm.set_start_state(0);
    fsm.add_end_state(0);
    fsm
}

#[cfg(test)]
#[path = "fsm_builder_tests.rs"]
mod tests;
