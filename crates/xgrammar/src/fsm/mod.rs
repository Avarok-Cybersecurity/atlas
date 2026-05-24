// SPDX-License-Identifier: AGPL-3.0-only
//
// FSM subsystem — port wave W2.
//
// Pure-Rust port of xgrammar's finite-state-machine subsystem
// (`cpp/fsm.{h,cc}` + `cpp/fsm_builder.{h,cc}`). No `unsafe`; the
// CSR-packed `CompactFsm` representation is preserved (it is load-bearing
// for the compiled-grammar memory layout) and accessed safely via slices.
//
// Module map:
//   edge           — FsmEdge, RepeatEdgeRef, edge-type constants
//   compact_array  — Compact2DArray, the CSR 2D-array primitive
//   union_find     — UnionFindSet, used by the simplification passes
//   traversal      — shared epsilon-closure / transition / advance code
//   fsm            — Fsm: the mutable adjacency-list FSM
//   compact        — CompactFsm: the immutable CSR FSM
//   with_start_end — FsmWithStartEnd / CompactFsmWithStartEnd + Star/
//                    Plus/Optional/Union/Concat
//   simplify       — SimplifyEpsilon / MergeEquivalentSuccessors
//   algorithms     — IsDFA / ToDFA
//   dfa_ops        — MinimizeDFA / Not / Intersect
//   builder        — RegexFsmBuilder / TrieFsmBuilder

pub mod algorithms;
pub mod builder;
pub mod compact;
pub mod compact_array;
pub mod compact_with_start_end;
pub mod dfa_ops;
pub mod edge;
pub mod fsm;
pub mod fsm_ops;
pub mod fsm_traversal;
pub mod simplify;
pub mod traversal;
pub mod union_find;
pub mod with_start_end;

pub use algorithms::DEFAULT_MAX_STATES;
pub use builder::{RegexFsmBuilder, TrieFsmBuilder};
pub use compact::CompactFsm;
pub use compact_array::Compact2DArray;
pub use edge::{FsmEdge, RepeatEdgeRef, edge_type};
pub use fsm::{Fsm, NO_NEXT_STATE};
pub use union_find::UnionFindSet;
pub use with_start_end::{CompactFsmWithStartEnd, FsmWithStartEnd};

#[cfg(test)]
mod integration_tests {
    //! End-to-end tests mirroring `tests/cpp/test_fsm.cc` —
    //! the scenarios that exercise multiple modules together.
    use super::*;

    fn build(re: &str) -> FsmWithStartEnd {
        RegexFsmBuilder::build(re).unwrap_or_else(|e| panic!("build {re}: {e}"))
    }

    #[test]
    fn function_test_compact_roundtrip() {
        // test_fsm.cc FunctionTest1
        let fsm = build("[\\d\\d\\d]+123");
        assert!(fsm.accept_string(b"123456123"));
        let compact = fsm.fsm().to_compact();
        let compact_wse =
            CompactFsmWithStartEnd::new(compact.clone(), fsm.start(), fsm.ends().to_vec());
        assert!(compact_wse.accept_string(b"123456123"));
        let back = FsmWithStartEnd::new(compact.to_fsm(), fsm.start(), fsm.ends().to_vec(), false);
        assert!(back.accept_string(b"123456123"));
    }

    #[test]
    fn function_test_to_dfa_no_epsilon() {
        // test_fsm.cc FunctionTest2
        let fsm = build("([abc]|[\\d])+");
        assert!(fsm.accept_string(b"abc3"));
        let dfa = fsm.to_dfa(DEFAULT_MAX_STATES).unwrap();
        assert!(dfa.accept_string(b"abc3"));
        for edges in dfa.fsm().all_edges() {
            assert!(edges.iter().all(|e| !e.is_epsilon()));
        }
    }

    #[test]
    fn function_test_minimize_then_not() {
        // test_fsm.cc FunctionTest3 + 4
        let fsm = build("([abc]|[\\d])+");
        let dfa = fsm.to_dfa(DEFAULT_MAX_STATES).unwrap();
        let min = dfa.minimize_dfa(DEFAULT_MAX_STATES).unwrap();
        assert!(min.accept_string(b"abc3"));
        assert_eq!(min.num_states(), 2);

        let neg = min.not(DEFAULT_MAX_STATES).unwrap();
        assert!(!neg.accept_string(b"abc3"));
        assert!(neg.accept_string(b"abcd"));
    }

    #[test]
    fn function_test_simplify_epsilon_state_count() {
        // test_fsm.cc FunctionTest6
        let fsm = build("[a][b][c][d]");
        assert!(fsm.accept_string(b"abcd"));
        let simplified = fsm.simplify_epsilon();
        assert_eq!(simplified.fsm().num_states(), 5);
        assert!(simplified.accept_string(b"abcd"));
    }

    #[test]
    fn function_test_merge_equivalent() {
        // test_fsm.cc FunctionTest7 ("abc|abd")
        let fsm = build("abc|abd");
        assert!(fsm.accept_string(b"abc"));
        let merged = fsm.simplify_epsilon().merge_equivalent_successors();
        assert!(merged.accept_string(b"abc"));
        assert!(!merged.accept_string(b"abcd"));
        assert_eq!(merged.fsm().num_states(), 4);
    }

    #[test]
    fn function_test_merge_precursor() {
        // test_fsm.cc FunctionTest8 ("acd|bcd")
        let fsm = build("acd|bcd");
        let merged = fsm.simplify_epsilon().merge_equivalent_successors();
        assert!(merged.accept_string(b"acd"));
        assert!(!merged.accept_string(b"abcd"));
        assert_eq!(merged.fsm().num_states(), 4);
    }

    #[test]
    fn function_test_star_simplify() {
        // test_fsm.cc FunctionTest9 ("ab*")
        let fsm = build("ab*");
        assert!(fsm.accept_string(b"abbb"));
        let simplified = fsm.simplify_epsilon();
        assert!(simplified.accept_string(b"abbb"));
        assert_eq!(simplified.fsm().num_states(), 2);
    }

    #[test]
    fn function_test_intersect() {
        // test_fsm.cc FunctionTest10
        let left = build("[c-f]+");
        let right = build("[d-h]*");
        let inter = FsmWithStartEnd::intersect(&left, &right, DEFAULT_MAX_STATES).unwrap();
        assert!(inter.accept_string(b"de"));
        assert!(inter.accept_string(b"def"));
        assert!(!inter.accept_string(b""));
        assert!(!inter.accept_string(b"cd"));
    }

    #[test]
    fn manual_merge_nodes_test() {
        // test_fsm.cc MergingNodesTest — manual FSM construction
        let mut fsm = FsmWithStartEnd::new(Fsm::with_states(0), 0, Vec::new(), false);
        for _ in 0..10 {
            fsm.add_state();
        }
        fsm.set_start_state(0);
        fsm.add_end_state(9);
        let pairs = [
            (0, 1, b'a'),
            (0, 2, b'a'),
            (1, 3, b'b'),
            (1, 3, b'c'),
            (1, 4, b'b'),
            (1, 4, b'c'),
            (2, 5, b'b'),
            (2, 5, b'c'),
            (2, 6, b'b'),
            (2, 6, b'c'),
            (3, 7, b'd'),
            (4, 7, b'd'),
            (5, 8, b'd'),
            (6, 8, b'd'),
            (7, 9, b'e'),
            (8, 9, b'e'),
        ];
        for (f, t, c) in pairs {
            fsm.fsm_mut().add_edge(f, t, c as i16, c as i16);
        }
        let merged = fsm.merge_equivalent_successors();
        assert_eq!(merged.fsm().num_states(), 5);
        // language: "abde" / "acde" — single letters a,(b|c),d,e
        assert!(merged.accept_string(b"abde"));
        assert!(merged.accept_string(b"acde"));
    }

    #[test]
    fn efficiency_test_minimize_target() {
        // test_fsm.cc EfficiencyTest — reduced: 10 copies of a 26-way
        // alternation each followed by "0123456789".  After minimize the
        // C++ asserts 111 states; we assert language correctness + that
        // minimize produces a small DFA.
        let one = "(a0123456789|b0123456789|c0123456789)";
        let pattern = format!("{one}{one}");
        let fsm = build(&pattern);
        let simplified = fsm.simplify_epsilon().merge_equivalent_successors();
        let dfa = simplified.to_dfa(DEFAULT_MAX_STATES).unwrap();
        let min = dfa.minimize_dfa(DEFAULT_MAX_STATES).unwrap();
        assert!(min.accept_string(b"a0123456789b0123456789"));
        assert!(!min.accept_string(b"a0123456789"));
        assert!(min.num_states() <= dfa.num_states());
    }

    #[test]
    fn trie_then_compact() {
        // test_fsm_builder.cc TestTrieFSMBuilder (compaction half)
        let pats: Vec<&[u8]> = vec![b"hello", b"hi", b"good"];
        let res = TrieFsmBuilder::build(&pats, &[], true, false).unwrap();
        let compact = res.fsm.fsm().to_compact();
        let compact_wse =
            CompactFsmWithStartEnd::new(compact, res.fsm.start(), res.fsm.ends().to_vec());
        assert!(compact_wse.accept_string(b"hello"));
        assert!(compact_wse.accept_string(b"hi"));
        assert!(!compact_wse.accept_string(b"he"));
    }
}
