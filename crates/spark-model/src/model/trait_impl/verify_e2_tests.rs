// SPDX-License-Identifier: AGPL-3.0-only

//! Tests for the batched-verify staging helpers (`verify_e2.rs`).
//!
//! Two proof obligations live here, both load-bearing for correctness rather
//! than for coverage:
//!
//! * [`verify_wy_cache_key`] must be INJECTIVE over everything the staged WY
//!   pointer tables depend on. A collision means a verify step reuses device
//!   tables staged for a DIFFERENT batch — every GDN layer would then read
//!   another sequence's recurrent state, silently.
//! * [`value_switch_armed`] must keep the house VALUE convention for the
//!   three verify switches hoisted out of the per-step hot path. An inverted
//!   or loosened predicate would arm `ATLAS_K4_DIAG` (which forces the verify
//!   path EAGER, destroying the graph replay) for anyone who exports it as
//!   `=0`.

use super::{value_switch_armed, verify_wy_cache_key};

/// The full input set, so a test can vary exactly one axis at a time.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Inputs {
    slots: Vec<u32>,
    k: usize,
    ghosts: Vec<(u32, u32)>,
}

impl Inputs {
    fn key(&self) -> Vec<u64> {
        verify_wy_cache_key(&self.slots, self.k, &self.ghosts)
    }
}

fn base() -> Inputs {
    Inputs {
        slots: vec![3, 7],
        k: 4,
        ghosts: vec![],
    }
}

/// Nothing changed ⇒ the key is reused. Re-deriving the key from freshly
/// built (not merely cloned) inputs must land on the same bytes, or the
/// cache would miss on every step and the fast path would be dead code.
#[test]
fn key_is_reused_when_no_input_changes() {
    let a = Inputs {
        slots: vec![3, 7],
        k: 4,
        ghosts: vec![(11, 2)],
    };
    let b = Inputs {
        slots: vec![3, 7],
        k: 4,
        ghosts: vec![(11, 2)],
    };
    assert_eq!(a.key(), b.key());
}

/// ENUMERATION of every input the staged tables depend on (see
/// `verify_wy_cache_key`'s docs for why the list is complete): `k`, the
/// sequence COUNT, each slot VALUE, the slot ORDER, ghost presence, each
/// ghost slot, each ghost depth, and the ghost ORDER. Changing any one of
/// them must change the key, i.e. must force a re-stage.
#[test]
fn key_changes_when_any_input_changes() {
    let b = base();
    let variants: Vec<(&str, Inputs)> = vec![
        ("k", Inputs { k: 3, ..b.clone() }),
        (
            "sequence count (fewer)",
            Inputs {
                slots: vec![3],
                ..b.clone()
            },
        ),
        (
            "sequence count (more)",
            Inputs {
                slots: vec![3, 7, 9],
                ..b.clone()
            },
        ),
        (
            "a slot value",
            Inputs {
                slots: vec![3, 8],
                ..b.clone()
            },
        ),
        (
            "slot order (batch order is table order)",
            Inputs {
                slots: vec![7, 3],
                ..b.clone()
            },
        ),
        (
            "ghost presence",
            Inputs {
                ghosts: vec![(11, 2)],
                ..b.clone()
            },
        ),
    ];
    for (what, v) in variants {
        assert_ne!(b.key(), v.key(), "key must change when {what} changes");
    }

    // Ghost axes, against a ghost-bearing baseline.
    let g = Inputs {
        ghosts: vec![(11, 2), (13, 3)],
        ..base()
    };
    let ghost_variants: Vec<(&str, Inputs)> = vec![
        (
            "a ghost slot",
            Inputs {
                ghosts: vec![(12, 2), (13, 3)],
                ..base()
            },
        ),
        (
            "a ghost depth",
            Inputs {
                ghosts: vec![(11, 4), (13, 3)],
                ..base()
            },
        ),
        (
            "ghost order",
            Inputs {
                ghosts: vec![(13, 3), (11, 2)],
                ..base()
            },
        ),
        (
            "ghost count",
            Inputs {
                ghosts: vec![(11, 2)],
                ..base()
            },
        ),
    ];
    for (what, v) in ghost_variants {
        assert_ne!(g.key(), v.key(), "key must change when {what} changes");
    }
}

/// INJECTIVITY over the whole reachable small domain: a slot run and a ghost
/// tail are concatenated into one flat `Vec<u64>`, so the encoding has to
/// keep them separable. `n` is written into the key for exactly this reason
/// — without it `slots=[1,2], ghosts=[]` and `slots=[1], ghosts=[(2, ..)]`
/// would collide, which is precisely the drain-tail-borrow shape.
#[test]
fn key_is_injective_over_the_reachable_domain() {
    let mut seen: std::collections::HashMap<Vec<u64>, Inputs> = std::collections::HashMap::new();
    let slot_sets: Vec<Vec<u32>> = vec![
        vec![0],
        vec![1],
        vec![0, 1],
        vec![1, 0],
        vec![0, 2],
        vec![0, 1, 2],
    ];
    let ghost_sets: Vec<Vec<(u32, u32)>> = vec![
        vec![],
        vec![(1, 2)],
        vec![(2, 1)],
        vec![(1, 2), (2, 3)],
        vec![(2, 3), (1, 2)],
    ];
    for slots in &slot_sets {
        for k in 2..=4usize {
            for ghosts in &ghost_sets {
                let inp = Inputs {
                    slots: slots.clone(),
                    k,
                    ghosts: ghosts.clone(),
                };
                if let Some(prev) = seen.insert(inp.key(), inp.clone()) {
                    panic!("WY cache key collision: {prev:?} and {inp:?} encode alike");
                }
            }
        }
    }
}

/// The hoisted verify switches are VALUE switches: only the literal `"1"`
/// arms them. Pins the exact predicate the three per-step `std::env::var`
/// reads used to spell inline, so hoisting them behind a `OnceLock` cannot
/// have widened (`presence`) or inverted the semantics.
#[test]
fn value_switch_is_armed_only_by_a_literal_one() {
    assert!(value_switch_armed(Some("1")));
    for raw in [
        None,
        Some(""),
        Some("0"),
        Some("2"),
        Some("true"),
        Some(" 1"),
    ] {
        assert!(
            !value_switch_armed(raw),
            "{raw:?} must not arm a VALUE switch"
        );
    }
}
