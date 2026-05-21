// SPDX-License-Identifier: AGPL-3.0-only
//
// TagDispatch FSM construction — port of
// `GrammarFSMBuilderImpl::TagDispatch` / `BuildTagDispatchWithEOSStop` /
// `BuildTagDispatchWithStopString` from `cpp/grammar_functor.cc`.
//
// A tag dispatch FSM matches arbitrary text, and on encountering a tag
// follows a rule-ref edge into the tag's rule. With `stop_eos` any state
// outside a tag prefix is accepting; with explicit stop strings only the
// stop-string terminals accept.

use crate::fsm::{Fsm, FsmWithStartEnd, TrieFsmBuilder};
use crate::grammar::data::TagDispatch;

/// Build the FSM for a tag dispatch, or `None` if the trie cannot be
/// built (e.g. overlapping tags).
pub fn build_tag_dispatch_fsm(td: &TagDispatch) -> Option<FsmWithStartEnd> {
    if td.stop_eos {
        build_with_eos_stop(td)
    } else {
        build_with_stop_string(td)
    }
}

/// EOS-stop variant: every non-tag-prefix state accepts.
fn build_with_eos_stop(td: &TagDispatch) -> Option<FsmWithStartEnd> {
    let tag_names: Vec<&[u8]> = td
        .tag_rule_pairs
        .iter()
        .map(|(t, _)| t.as_bytes())
        .collect();
    let excluded: Vec<&[u8]> = td.excluded_str.iter().map(|s| s.as_bytes()).collect();
    let trie = TrieFsmBuilder::build(&tag_names, &excluded, false, true)?;
    let mut fsm: Fsm = trie.fsm.fsm().clone();
    let start = trie.fsm.start();
    let end_states = trie.end_states.clone();

    // The old trie terminals are the *non*-accepting states here.
    let old_ends: Vec<bool> = (0..trie.fsm.num_states())
        .map(|s| trie.fsm.is_end_state(s))
        .collect();
    let mut ends: Vec<bool> = (0..fsm.num_states()).map(|s| !old_ends[s]).collect();

    for (i, (_, rule_id)) in td.tag_rule_pairs.iter().enumerate() {
        let next = if td.loop_after_dispatch {
            start
        } else {
            let s = fsm.add_state();
            ends.push(true);
            s
        };
        fsm.add_rule_edge(end_states[i] as usize, next, *rule_id as i16);
    }
    Some(FsmWithStartEnd::new(fsm, start, ends, false))
}

/// Stop-string variant: only the stop-string terminals accept.
fn build_with_stop_string(td: &TagDispatch) -> Option<FsmWithStartEnd> {
    debug_assert!(!td.stop_str.is_empty());
    // Trie over tags ++ stop strings.
    let mut all_names: Vec<&[u8]> = td
        .tag_rule_pairs
        .iter()
        .map(|(t, _)| t.as_bytes())
        .collect();
    for s in &td.stop_str {
        all_names.push(s.as_bytes());
    }
    let excluded: Vec<&[u8]> = td.excluded_str.iter().map(|s| s.as_bytes()).collect();
    let trie = TrieFsmBuilder::build(&all_names, &excluded, false, true)?;
    let mut fsm: Fsm = trie.fsm.fsm().clone();
    let start = trie.fsm.start();
    let end_states = trie.end_states.clone();
    let num_tags = td.tag_rule_pairs.len();

    let mut ends: Vec<bool> = vec![false; fsm.num_states()];
    // The accepting states are the terminals of the stop strings.
    for &es in &end_states[num_tags..] {
        ends[es as usize] = true;
    }

    if td.loop_after_dispatch {
        for (i, (_, rule_id)) in td.tag_rule_pairs.iter().enumerate() {
            fsm.add_rule_edge(end_states[i] as usize, start, *rule_id as i16);
        }
    } else {
        // Build a separate trie holding only the stop strings; splice it
        // in and route each tag's rule edge to its start.
        let stop_names: Vec<&[u8]> = td.stop_str.iter().map(|s| s.as_bytes()).collect();
        let stop_trie = TrieFsmBuilder::build(&stop_names, &excluded, false, false)?;
        let stop_fsm = stop_trie.fsm.fsm().clone();
        let stop_start = stop_trie.fsm.start();
        let stop_ends: Vec<usize> = (0..stop_trie.fsm.num_states())
            .filter(|&s| stop_trie.fsm.is_end_state(s))
            .collect();

        let mapping = fsm.add_fsm(&stop_fsm);
        ends.resize(fsm.num_states(), false);
        let start_of_stop = mapping[stop_start];
        for s in stop_ends {
            ends[mapping[s]] = true;
        }
        for (i, (_, rule_id)) in td.tag_rule_pairs.iter().enumerate() {
            fsm.add_rule_edge(end_states[i] as usize, start_of_stop, *rule_id as i16);
        }
    }
    Some(FsmWithStartEnd::new(fsm, start, ends, false))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn td_eos(pairs: &[(&str, i32)]) -> TagDispatch {
        TagDispatch {
            tag_rule_pairs: pairs.iter().map(|(t, r)| (t.to_string(), *r)).collect(),
            stop_eos: true,
            stop_str: Vec::new(),
            loop_after_dispatch: false,
            excluded_str: Vec::new(),
        }
    }

    #[test]
    fn eos_stop_builds_fsm() {
        let td = td_eos(&[("<call>", 1)]);
        let fsm = build_tag_dispatch_fsm(&td).expect("fsm");
        // Plain text with no tag prefix is accepted.
        assert!(fsm.accept_string(b"abc"));
    }

    #[test]
    fn eos_stop_loop_variant() {
        let mut td = td_eos(&[("<t>", 2)]);
        td.loop_after_dispatch = true;
        let fsm = build_tag_dispatch_fsm(&td);
        assert!(fsm.is_some());
    }

    #[test]
    fn stop_string_variant() {
        let td = TagDispatch {
            tag_rule_pairs: vec![("<x>".to_string(), 1)],
            stop_eos: false,
            stop_str: vec!["END".to_string()],
            loop_after_dispatch: false,
            excluded_str: Vec::new(),
        };
        let fsm = build_tag_dispatch_fsm(&td).expect("fsm");
        assert!(fsm.accept_string(b"END"));
    }
}
