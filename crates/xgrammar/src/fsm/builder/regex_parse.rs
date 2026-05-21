// SPDX-License-Identifier: AGPL-3.0-only
//
// Regex-parser stack helpers — group-close handling and the final
// stack drain. Split out of `regex.rs` to keep each file under the
// 250-line cap. Port of the `)`-handling and post-loop logic in
// `RegexFSMBuilder::Build` from `cpp/fsm_builder.cc`.

use super::regex::StackItem;
use super::regex_ir::{RegexIr, RegexState};

/// Handle a `)` — pop until the matching `(`, build a `Bracket` or
/// `Union` node, and push it back.
pub(crate) fn parse_group_close(stack: &mut Vec<StackItem>) -> Result<(), String> {
    let mut popped: Vec<StackItem> = Vec::new();
    let mut paired = false;
    let mut unioned = false;
    while let Some(item) = stack.pop() {
        match item {
            StackItem::Ctrl(b'(') => {
                paired = true;
                break;
            }
            StackItem::Ctrl(b'|') => {
                unioned = true;
                popped.push(StackItem::Ctrl(b'|'));
            }
            other => popped.push(other),
        }
    }
    if !paired {
        return Err("Invalid regex: no paired bracket!".to_string());
    }
    if popped.is_empty() {
        return Ok(());
    }
    // `popped` is in reverse source order; reverse to source order.
    popped.reverse();
    if !unioned {
        let states: Vec<RegexState> = popped
            .into_iter()
            .map(|it| match it {
                StackItem::State(s) => Ok(s),
                _ => Err("Invalid regex: no paired bracket!".to_string()),
            })
            .collect::<Result<_, _>>()?;
        stack.push(StackItem::State(RegexState::Bracket { states }));
    } else {
        let mut union_states: Vec<RegexState> = Vec::new();
        let mut bracket: Vec<RegexState> = Vec::new();
        for it in popped {
            match it {
                StackItem::Ctrl(b'|') => {
                    union_states.push(RegexState::Bracket {
                        states: std::mem::take(&mut bracket),
                    });
                }
                StackItem::State(s) => bracket.push(s),
                _ => return Err("Invalid regex: no paired bracket!".to_string()),
            }
        }
        union_states.push(RegexState::Bracket { states: bracket });
        stack.push(StackItem::State(RegexState::Union {
            states: union_states,
        }));
    }
    Ok(())
}

/// Drain the parse stack into a finished [`RegexIr`].
pub(crate) fn finalize_ir(stack: Vec<StackItem>) -> Result<RegexIr, String> {
    let mut ir = RegexIr::default();
    let mut res_states: Vec<RegexState> = Vec::new();
    let mut union_list: Vec<Vec<RegexState>> = Vec::new();
    let mut unioned = false;

    // The stack is processed top-to-bottom (LIFO), as in the C++.
    for item in stack.into_iter().rev() {
        match item {
            StackItem::Ctrl(b'|') => {
                union_list.push(std::mem::take(&mut res_states));
                unioned = true;
            }
            StackItem::Ctrl(_) => {
                return Err("Invalid regex: no paired!".to_string());
            }
            StackItem::State(s) => res_states.push(s),
        }
    }

    if !unioned {
        // res_states is in reverse source order — reverse it.
        for s in res_states.into_iter().rev() {
            ir.states.push(s);
        }
    } else {
        union_list.push(res_states);
        let mut union_state: Vec<RegexState> = Vec::new();
        for branch in union_list {
            let mut bracket: Vec<RegexState> = Vec::new();
            for s in branch.into_iter().rev() {
                bracket.push(s);
            }
            union_state.push(RegexState::Bracket { states: bracket });
        }
        ir.states.push(RegexState::Union {
            states: union_state,
        });
    }
    Ok(ir)
}
