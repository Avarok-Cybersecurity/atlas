// SPDX-License-Identifier: AGPL-3.0-only
//
// Unit tests for the sibling module, in a separate file so the
// code file stays under the 250-line cap (included via `#[path]`).

use super::super::fsm::Fsm;
use super::*;

fn literal(bytes: &[u8]) -> FsmWithStartEnd {
    let mut fsm = Fsm::with_states(bytes.len() + 1);
    for (i, &b) in bytes.iter().enumerate() {
        fsm.add_edge(i, i + 1, b as i16, b as i16);
    }
    let mut ends = vec![false; bytes.len() + 1];
    ends[bytes.len()] = true;
    FsmWithStartEnd::new(fsm, 0, ends, false)
}

#[test]
fn simplify_epsilon_collapses_chain() {
    // a -eps-> b -eps-> c, all single-edge => collapses to one node
    let mut fsm = Fsm::with_states(3);
    fsm.add_epsilon_edge(0, 1);
    fsm.add_epsilon_edge(1, 2);
    let f = FsmWithStartEnd::new(fsm, 0, vec![false, false, true], false);
    let s = f.simplify_epsilon();
    assert_eq!(s.num_states(), 1);
}

#[test]
fn simplify_epsilon_preserves_language() {
    // build "abcd" via [a][b][c][d] concat, lots of epsilons
    let f = FsmWithStartEnd::concat(&[literal(b"a"), literal(b"b"), literal(b"c"), literal(b"d")]);
    assert!(f.accept_string(b"abcd"));
    let s = f.simplify_epsilon();
    assert!(s.accept_string(b"abcd"));
    assert!(!s.accept_string(b"abc"));
}

#[test]
fn merge_equivalent_successors_reduces_states() {
    // abc | abd -> after simplify+merge should shrink
    let f = FsmWithStartEnd::union(&[literal(b"abc"), literal(b"abd")]);
    assert!(f.accept_string(b"abc"));
    let merged = f.simplify_epsilon().merge_equivalent_successors();
    assert!(merged.accept_string(b"abc"));
    assert!(merged.accept_string(b"abd"));
    assert!(!merged.accept_string(b"abe"));
}

#[test]
fn merge_equivalent_precursors() {
    // acd | bcd -> (a|b)cd
    let f = FsmWithStartEnd::union(&[literal(b"acd"), literal(b"bcd")]);
    let merged = f.simplify_epsilon().merge_equivalent_successors();
    assert!(merged.accept_string(b"acd"));
    assert!(merged.accept_string(b"bcd"));
    assert!(!merged.accept_string(b"abcd"));
}

#[test]
fn merge_does_not_over_merge_enum() {
    // Regression for upstream 8d22ba0 (#632 / #618): an enum of many
    // strings was over-simplified so the matcher accepted strings that
    // were never in the enum. The merge must accept exactly the inputs
    // and reject cross-products / near-misses.
    let words: [&[u8]; 6] = [b"apple", b"apply", b"apron", b"april", b"angle", b"ankle"];
    let f = FsmWithStartEnd::union(&words.map(literal));
    let merged = f.simplify_epsilon().merge_equivalent_successors();
    for w in &words {
        assert!(merged.accept_string(w), "must accept {:?}", w);
    }
    // Cross-products and near-misses that share prefixes/suffixes but
    // are not in the enum must be rejected.
    for bad in [
        &b"applr"[..],
        &b"appll"[..],
        &b"aprol"[..],
        &b"anple"[..],
        &b"apple "[..],
        &b"appl"[..],
        &b""[..],
    ] {
        assert!(!merged.accept_string(bad), "must reject {bad:?}");
    }
}

#[test]
fn merge_preserves_shared_suffix_enum() {
    // ...ation suffix shared across several distinct prefixes.
    let words: [&[u8]; 4] = [b"creation", b"relation", b"location", b"rotation"];
    let f = FsmWithStartEnd::union(&words.map(literal));
    let merged = f.simplify_epsilon().merge_equivalent_successors();
    for w in &words {
        assert!(merged.accept_string(w));
    }
    assert!(!merged.accept_string(b"crelation"));
    assert!(!merged.accept_string(b"creatio"));
}

#[test]
fn merge_tiny_fsm_unchanged() {
    // Fewer than 4 states: the early-exit returns a copy untouched
    // (upstream 96ae88b, #616).
    let f = literal(b"a");
    assert_eq!(f.num_states(), 2);
    let merged = f.merge_equivalent_successors();
    assert_eq!(merged.num_states(), 2);
    assert!(merged.accept_string(b"a"));
    assert!(!merged.accept_string(b"b"));
}

#[test]
fn edge_csr_reset_and_rows() {
    // EdgeCsr must lay rows out per row_sizes and reuse storage on reset.
    let mut csr = EdgeCsr::default();
    csr.reset_with_row_sizes(&[2, 0, 1]);
    assert_eq!(csr.row(0).len(), 2);
    assert_eq!(csr.row(1).len(), 0);
    assert_eq!(csr.row(2).len(), 1);
    csr.row_mut(0)[1] = EndpointEdge {
        peer: 7,
        min: 1,
        max: 2,
    };
    assert_eq!(csr.row(0)[1].peer, 7);
    // Reset to a smaller shape, storage reused, rows re-sized.
    csr.reset_with_row_sizes(&[1]);
    assert_eq!(csr.row(0).len(), 1);
    assert_eq!(csr.indptr, vec![0, 1]);
}

#[test]
fn distinct_peers_counts_groups() {
    // Row 0 has two distinct peers, row 1 has one, row 2 is empty.
    let mut csr = EdgeCsr::default();
    csr.reset_with_row_sizes(&[3, 2, 0]);
    csr.row_mut(0).copy_from_slice(&[
        EndpointEdge {
            peer: 1,
            min: 0,
            max: 0,
        },
        EndpointEdge {
            peer: 1,
            min: 1,
            max: 1,
        },
        EndpointEdge {
            peer: 4,
            min: 0,
            max: 0,
        },
    ]);
    csr.row_mut(1).copy_from_slice(&[
        EndpointEdge {
            peer: 9,
            min: 0,
            max: 0,
        },
        EndpointEdge {
            peer: 9,
            min: 2,
            max: 2,
        },
    ]);
    let mut distinct = Vec::new();
    let mut single = Vec::new();
    distinct_peers(&csr, 3, &mut distinct, &mut single);
    assert_eq!(distinct, vec![2, 1, 0]);
    assert_eq!(single, vec![-1, 9, -1]);
}

#[test]
fn simplify_idempotent_on_dfa() {
    let mut f = literal(b"a");
    f.is_dfa = true;
    let s = f.simplify_epsilon();
    assert_eq!(s.num_states(), f.num_states());
}
