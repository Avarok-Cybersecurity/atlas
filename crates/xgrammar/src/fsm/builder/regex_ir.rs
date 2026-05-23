// SPDX-License-Identifier: AGPL-3.0-only
//
// RegexIr — the intermediate representation a regex string is parsed
// into, plus its NFA construction (`Build` / `visit`). Port of `RegexIR`
// from `cpp/fsm_builder.cc`.
//
// The C++ uses `std::variant`; Rust uses a plain enum. Leaf FSMs are
// built directly from regex fragments via `build_leaf_fsm`.

use crate::fsm::fsm::Fsm;
use crate::fsm::with_start_end::FsmWithStartEnd;

/// Sentinel for an unbounded repeat upper bound (`{n,}`).
pub const REPEAT_NO_UPPER_BOUND: i32 = -1;

/// A postfix-style regex quantifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RegexSymbol {
    /// `*`
    Star,
    /// `+`
    Plus,
    /// `?`
    Optional,
}

/// A node of the regex IR tree.
#[derive(Debug, Clone)]
pub enum RegexState {
    /// A literal fragment or character class, e.g. `"a"` or `"[a-z]"`.
    /// Stored as raw bytes — the regex grammar is byte-wise, so a
    /// fragment may split a UTF-8 codepoint across multiple `Leaf`s.
    Leaf { regex: Vec<u8> },
    /// A quantifier applied to a single child.
    Symbol {
        symbol: RegexSymbol,
        state: Box<RegexState>,
    },
    /// An alternation of branches.
    Union { states: Vec<RegexState> },
    /// A parenthesized concatenation.
    Bracket { states: Vec<RegexState> },
    /// A counted repetition `{lower,upper}`.
    Repeat {
        states: Vec<RegexState>,
        lower_bound: i32,
        upper_bound: i32,
    },
}

/// The whole parsed regex: a sequence of top-level states.
#[derive(Debug, Default, Clone)]
pub struct RegexIr {
    /// Top-level concatenation of states.
    pub states: Vec<RegexState>,
}

impl RegexIr {
    /// Build the NFA for the entire IR.
    pub fn build(&self) -> Result<FsmWithStartEnd, String> {
        if self.states.is_empty() {
            // The empty regex accepts only the empty string.
            let empty = Fsm::with_states(1);
            return Ok(FsmWithStartEnd::new(empty, 0, vec![true], false));
        }
        let mut fsm_list: Vec<FsmWithStartEnd> = Vec::new();
        for state in &self.states {
            fsm_list.push(visit(state)?);
        }
        if fsm_list.len() > 1 {
            Ok(FsmWithStartEnd::concat(&fsm_list))
        } else {
            Ok(fsm_list.into_iter().next().unwrap())
        }
    }
}

/// Construct the NFA for a single IR node.
pub fn visit(state: &RegexState) -> Result<FsmWithStartEnd, String> {
    match state {
        RegexState::Leaf { regex } => Ok(build_leaf_fsm(regex)),
        // (see build_leaf_fsm — takes a &[u8])
        RegexState::Symbol { symbol, state } => {
            let child = visit(state)?;
            Ok(match symbol {
                RegexSymbol::Plus => child.plus(),
                RegexSymbol::Star => child.star(),
                RegexSymbol::Optional => child.optional(),
            })
        }
        RegexState::Union { states } => {
            let mut list = Vec::new();
            for child in states {
                list.push(visit(child)?);
            }
            if list.len() <= 1 {
                return Err("Invalid union".to_string());
            }
            Ok(FsmWithStartEnd::union(&list))
        }
        RegexState::Bracket { states } => {
            let mut list = Vec::new();
            for child in states {
                list.push(visit(child)?);
            }
            if list.is_empty() {
                return Err("Invalid bracket".to_string());
            }
            Ok(FsmWithStartEnd::concat(&list))
        }
        RegexState::Repeat {
            states,
            lower_bound,
            upper_bound,
        } => {
            if states.len() != 1 {
                return Err("Invalid repeat".to_string());
            }
            visit_repeat(&states[0], *lower_bound, *upper_bound)
        }
    }
}

/// NFA construction for a `{lower,upper}` repetition.
fn visit_repeat(
    child_state: &RegexState,
    lower_bound: i32,
    upper_bound: i32,
) -> Result<FsmWithStartEnd, String> {
    let child = visit(child_state)?;
    let mut result = child.copy();
    let mut new_ends: Vec<usize> = Vec::new();

    if lower_bound == 1 {
        for end in 0..result.num_states() {
            if result.is_end_state(end) {
                new_ends.push(end);
            }
        }
    }

    if upper_bound == REPEAT_NO_UPPER_BOUND {
        // {n,}: repeat n times, then loop back to the n-th copy's start.
        let mut i = 2;
        while i < lower_bound {
            result = FsmWithStartEnd::concat(&[result.clone(), child.clone()]);
            i += 1;
        }
        let mut end_of_lower = -1i64;
        for end in 0..result.num_states() {
            if result.is_end_state(end) {
                end_of_lower = end as i64;
                break;
            }
        }
        debug_assert!(end_of_lower != -1, "No end state in lower-bound FSM.");
        result = FsmWithStartEnd::concat(&[result, child]);
        for end in 0..result.num_states() {
            if result.is_end_state(end) {
                result
                    .fsm_mut()
                    .add_epsilon_edge(end, end_of_lower as usize);
            }
        }
        return Ok(result);
    }

    // {n,m} or {n}
    let mut i = 2;
    while i <= upper_bound {
        result = FsmWithStartEnd::concat(&[result, child.clone()]);
        if i >= lower_bound {
            for end in 0..result.num_states() {
                if result.is_end_state(end) {
                    new_ends.push(end);
                }
            }
        }
        i += 1;
    }
    for end in new_ends {
        result.add_end_state(end);
    }
    Ok(result)
}

// Leaf-FSM construction lives in `regex_leaf.rs`; re-exported here
// so existing call sites (`super::regex_ir::build_leaf_fsm`) keep
// working.
pub use super::regex_leaf::{build_leaf_fsm, handle_escapes};

#[cfg(test)]
#[path = "regex_ir_tests.rs"]
mod tests;
