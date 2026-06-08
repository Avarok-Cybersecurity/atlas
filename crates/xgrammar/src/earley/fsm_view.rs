// SPDX-License-Identifier: AGPL-3.0-only
//
// FSM-view helpers for the Earley parser.
//
// The per-rule FSMs (`CompactFsmWithStartEnd`) expose `is_end_state`
// and `edges`, but the Earley parser also needs `IsScanableState` and
// `IsNonTerminalState` (defined in the C++ `FSMWithStartEndBase`). The
// Rust `fsm` module is frozen for this wave, so these predicates are
// re-derived here from a state's outgoing edges — exact ports of the
// inline C++ definitions in `cpp/fsm.h`.

use crate::fsm::CompactFsmWithStartEnd;

/// True if `state` has an outgoing character-range edge — i.e. the
/// parser can `Scan` a byte from it. Port of `IsScanableState`.
pub fn is_scanable_state(fsm: &CompactFsmWithStartEnd, state: i32) -> bool {
    fsm.fsm()
        .edges(state as usize)
        .iter()
        .any(|e| e.is_char_range())
}

/// True if `state` has an outgoing rule-ref, epsilon, or repeat-ref
/// edge — i.e. it must be expanded by prediction. Port of
/// `IsNonTerminalState`.
pub fn is_non_terminal_state(fsm: &CompactFsmWithStartEnd, state: i32) -> bool {
    fsm.fsm()
        .edges(state as usize)
        .iter()
        .any(|e| e.is_rule_ref() || e.is_epsilon() || e.is_repeat_ref())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::fsm::{Fsm, FsmWithStartEnd};

    fn compact(f: FsmWithStartEnd) -> CompactFsmWithStartEnd {
        CompactFsmWithStartEnd::new(f.fsm().to_compact(), f.start(), f.ends().to_vec())
    }

    #[test]
    fn scanable_when_char_edge_present() {
        let mut f = FsmWithStartEnd::new(Fsm::with_states(0), 0, Vec::new(), false);
        f.add_state();
        f.add_state();
        f.fsm_mut().add_edge(0, 1, b'a' as i16, b'z' as i16);
        let c = compact(f);
        assert!(is_scanable_state(&c, 0));
        assert!(!is_scanable_state(&c, 1));
    }

    #[test]
    fn non_terminal_when_epsilon_present() {
        let mut f = FsmWithStartEnd::new(Fsm::with_states(0), 0, Vec::new(), false);
        f.add_state();
        f.add_state();
        f.fsm_mut().add_epsilon_edge(0, 1);
        let c = compact(f);
        assert!(is_non_terminal_state(&c, 0));
        assert!(!is_non_terminal_state(&c, 1));
    }
}
