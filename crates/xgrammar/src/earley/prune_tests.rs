// SPDX-License-Identifier: AGPL-3.0-only
//! ZapFormat dead-state pruning tests — child module of `earley::tests`
//! (see tests.rs); split out to keep that file under the 500-LoC cap.

use super::super::prune::ProductivityTable;
use super::*;
use crate::fsm::{CompactFsmWithStartEnd, Fsm, FsmWithStartEnd};

/// Compact a mutable FSM-with-start-end into the immutable form.
fn compact(f: FsmWithStartEnd) -> CompactFsmWithStartEnd {
    CompactFsmWithStartEnd::new(f.fsm().to_compact(), f.start(), f.ends().to_vec())
}

/// Build a per-rule FSM with a deliberate dead branch:
///   0 --'a'--> 1 (end)        productive
///   0 --'b'--> 2 --'c'--> 3   2 and 3 are NOT end, no path to an end
/// Node 0 is productive (reaches end via 1); nodes 2 and 3 are dead.
fn fsm_with_dead_branch() -> CompactFsmWithStartEnd {
    let mut f = FsmWithStartEnd::new(Fsm::with_states(0), 0, Vec::new(), false);
    f.add_state(); // 0
    f.add_state(); // 1
    f.add_state(); // 2
    f.add_state(); // 3
    f.fsm_mut().add_edge(0, 1, b'a' as i16, b'a' as i16);
    f.fsm_mut().add_edge(0, 2, b'b' as i16, b'b' as i16);
    f.fsm_mut().add_edge(2, 3, b'c' as i16, b'c' as i16);
    f.add_end_state(1);
    compact(f)
}

#[test]
fn productivity_classifies_co_accessible_nodes() {
    let fsm = fsm_with_dead_branch();
    let mut grammar = crate::grammar::GrammarData::new();
    grammar.per_rule_fsms = vec![Some(fsm)];
    let table = ProductivityTable::build(&grammar);

    // Node 0 reaches end 1; node 1 *is* an end — both productive.
    assert!(table.is_node_productive(0, 0));
    assert!(table.is_node_productive(0, 1));
    // Nodes 2 and 3 can never reach an end — provably dead.
    assert!(!table.is_node_productive(0, 2));
    assert!(!table.is_node_productive(0, 3));
}

#[test]
fn out_of_range_ids_are_conservatively_productive() {
    let table = ProductivityTable::default();
    // Unknown rule / node ids must never be classified dead — a
    // missed prune is fine, a wrong prune is a correctness bug.
    assert!(table.is_node_productive(99, 0));
    assert!(table.is_node_productive(0, 0));
    assert!(table.is_node_productive(-1, 5));
}

#[test]
fn prune_drops_only_dead_states_keeps_live_ones() {
    let fsm = fsm_with_dead_branch();
    let mut grammar = crate::grammar::GrammarData::new();
    grammar.per_rule_fsms = vec![Some(fsm)];
    let table = ProductivityTable::build(&grammar);

    // A mixed batch: nodes 0,1 live; nodes 2,3 dead.
    let mut states = vec![
        ParserState::new(0, 0, 0, 0, 0), // node 0 — live
        ParserState::new(0, 0, 2, 0, 0), // node 2 — dead
        ParserState::new(0, 0, 1, 0, 0), // node 1 — live
        ParserState::new(0, 0, 3, 0, 0), // node 3 — dead
    ];
    table.prune(&mut states);
    let nodes: Vec<i32> = states.iter().map(|s| s.element_id).collect();
    assert_eq!(nodes, vec![0, 1], "only the dead nodes 2,3 are dropped");
}

#[test]
fn prune_keeps_fsm_less_states() {
    // `rule_id == -1` states are FSM-less; productivity is not
    // tracked for them, so they must be kept conservatively.
    let table = ProductivityTable::default();
    let mut states = vec![ParserState::new(-1, 0, 7, 0, 0)];
    table.prune(&mut states);
    assert_eq!(states.len(), 1);
}

#[test]
fn prune_is_idempotent() {
    let fsm = fsm_with_dead_branch();
    let mut grammar = crate::grammar::GrammarData::new();
    grammar.per_rule_fsms = vec![Some(fsm)];
    let table = ProductivityTable::build(&grammar);
    let mut states = vec![
        ParserState::new(0, 0, 1, 0, 0),
        ParserState::new(0, 0, 2, 0, 0),
    ];
    table.prune(&mut states);
    let after_first = states.clone();
    table.prune(&mut states);
    assert_eq!(states, after_first, "pruning twice is a no-op");
}

/// End-to-end: pruning must not change accept/reject decisions.
/// These grammars all have alternation / repetition branches that
/// die mid-parse — the exact case where dead states accumulate.
#[test]
fn pruning_preserves_accepted_language() {
    // Divergent alternation: after "ab" one branch is dead.
    let g = optimized("root ::= \"abc\" | \"abd\"\n");
    assert!(accepts(g.clone(), "abc"));
    assert!(accepts(g.clone(), "abd"));
    assert!(!accepts(g.clone(), "abe"));
    assert!(!accepts(g.clone(), "ab"));
    assert!(!accepts(g, "abcd"));

    // Many divergent branches sharing a prefix.
    let g = optimized("root ::= \"cat\" | \"car\" | \"can\" | \"cab\"\n");
    for w in ["cat", "car", "can", "cab"] {
        assert!(accepts(g.clone(), w), "{w} must accept");
    }
    assert!(!accepts(g.clone(), "caz"));
    assert!(!accepts(g.clone(), "ca"));
    assert!(!accepts(g, "cats"));

    // Nested rules with a dead inner branch.
    let g = optimized(
        "root ::= a b\n\
             a ::= \"x\" | \"xy\"\n\
             b ::= \"z\"\n",
    );
    assert!(accepts(g.clone(), "xz"));
    assert!(accepts(g.clone(), "xyz"));
    assert!(!accepts(g.clone(), "xyyz"));
    assert!(!accepts(g, "xy"));
}

/// Pruning must not break rollback: the parser state after
/// advance+rollback is byte-identical to before, even though
/// pruning rewrote the scanable rows in between.
#[test]
fn pruning_does_not_break_rollback() {
    let g = optimized("root ::= \"abc\" | \"abd\"\n");
    let mut p = EarleyParser::from_grammar(g);
    p.advance(b'a');
    let snapshot = p.latest_scanable_states().to_vec();
    let steps = p.num_steps();
    p.advance(b'b');
    p.advance(b'c');
    p.pop_last_states(2);
    assert_eq!(p.num_steps(), steps);
    assert_eq!(p.latest_scanable_states(), snapshot.as_slice());
    // Finish via the other branch from the rolled-back point.
    assert!(p.advance(b'b'));
    assert!(p.advance(b'd'));
    assert!(p.is_completed());
}

/// The acceptable-byte mask must be identical with pruning: a
/// pruned (dead) state can contribute no byte that leads anywhere,
/// so removing it never narrows the *useful* acceptable set.
#[test]
fn pruning_preserves_acceptable_byte_mask() {
    let g = optimized("root ::= \"abc\" | \"abd\"\n");
    let mut p = EarleyParser::from_grammar(g);
    assert!(p.can_accept(b'a'));
    p.advance(b'a');
    p.advance(b'b');
    // After "ab" both 'c' and 'd' remain acceptable — neither
    // branch is dead yet, so nothing was pruned away.
    assert!(p.can_accept(b'c'));
    assert!(p.can_accept(b'd'));
    assert!(!p.can_accept(b'e'));
}
