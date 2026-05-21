// SPDX-License-Identifier: AGPL-3.0-only
//
// Leaf-FSM construction for the regex builder — turns a single
// regex fragment (a literal run or a `[...]` character class) into
// an FSM, and expands escape classes (`\d`, `\w`, …). Split out of
// `regex_ir.rs` to keep each file under the 250-line cap. Port of
// `BuildLeafFSMFromRegex` / `HandleEscapes` from `cpp/fsm_builder.cc`.

use crate::fsm::fsm::Fsm;
use crate::fsm::with_start_end::FsmWithStartEnd;

/// Escape-class expansion for `\n`, `\d`, `\w`, … Returns the
/// `(min, max)` byte ranges the escape at `regex[start]` denotes.
pub fn handle_escapes(regex: &[u8], start: usize) -> Vec<(i32, i32)> {
    let c = regex[start + 1];
    match c {
        b'n' => vec![('\n' as i32, '\n' as i32)],
        b't' => vec![('\t' as i32, '\t' as i32)],
        b'r' => vec![('\r' as i32, '\r' as i32)],
        b'0' => vec![(0, 0)],
        b's' => vec![(0, ' ' as i32)],
        b'S' => vec![(' ' as i32 + 1, 0x00FF)],
        b'd' => vec![('0' as i32, '9' as i32)],
        b'D' => vec![(0, '0' as i32 - 1), ('9' as i32 + 1, 0x00FF)],
        b'w' => vec![
            ('0' as i32, '9' as i32),
            ('a' as i32, 'z' as i32),
            ('A' as i32, 'Z' as i32),
            ('_' as i32, '_' as i32),
        ],
        b'W' => vec![
            (0, '0' as i32 - 1),
            ('9' as i32 + 1, 'A' as i32 - 1),
            ('Z' as i32 + 1, '_' as i32 - 1),
            ('_' as i32 + 1, 'a' as i32 - 1),
            ('z' as i32 + 1, 0x00FF),
        ],
        other => vec![(other as i32, other as i32)],
    }
}

/// Build a leaf FSM from a regex fragment that is either a plain literal
/// (`"abx"`, possibly with escapes / `.`) or a character class `[...]`.
/// `regex` is raw bytes — the builder is byte-wise (matches the C++).
pub fn build_leaf_fsm(regex: &[u8]) -> FsmWithStartEnd {
    let bytes = regex;
    let mut result = FsmWithStartEnd::new(Fsm::with_states(0), 0, Vec::new(), true);

    let is_class = !bytes.is_empty() && bytes[0] == b'[' && bytes[bytes.len() - 1] == b']';

    if !is_class {
        build_literal_leaf(&mut result, bytes);
    } else {
        build_class_leaf(&mut result, bytes);
    }
    result
}

/// Literal fragment: one state per byte, optional `\escape` / `.`.
fn build_literal_leaf(result: &mut FsmWithStartEnd, bytes: &[u8]) {
    result.add_state();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'\\' {
            let from = result.num_states() - 1;
            // INTEGRATION: a UTF-8-aware leaf would decode codepoints via
            // crate::support::encoding::decode_utf8 here; the C++ regex
            // builder is byte-wise, so this port matches it byte-wise.
            if bytes[i] == b'.' {
                result.fsm_mut().add_edge(from, from + 1, 0, 0xFF);
            } else {
                let b = bytes[i] as i16;
                result.fsm_mut().add_edge(from, from + 1, b, b);
            }
            result.add_state();
            i += 1;
            continue;
        }
        let escapes = handle_escapes(bytes, i);
        let from = result.num_states() - 1;
        for (lo, hi) in escapes {
            result
                .fsm_mut()
                .add_edge(from, from + 1, lo as u8 as i16, hi as u8 as i16);
        }
        result.add_state();
        i += 2;
    }
    let last = result.num_states() - 1;
    result.add_end_state(last);
}

/// Character class `[...]`: a two-state FSM whose edges are the
/// (possibly negated, gap-coalesced) byte ranges of the class.
fn build_class_leaf(result: &mut FsmWithStartEnd, bytes: &[u8]) {
    result.add_state();
    result.add_state();
    result.add_end_state(1);
    let reverse = bytes.len() > 1 && bytes[1] == b'^';

    let start_idx = if reverse { 2 } else { 1 };
    let end_idx = bytes.len() - 1; // exclusive (the closing ']')
    let mut i = start_idx;
    while i < end_idx {
        if bytes[i] != b'\\' {
            let is_range = i + 2 < end_idx && bytes[i + 1] == b'-';
            if !is_range {
                let b = bytes[i] as i16;
                result.fsm_mut().add_edge(0, 1, b, b);
                i += 1;
                continue;
            }
            if bytes[i + 2] != b'\\' {
                result
                    .fsm_mut()
                    .add_edge(0, 1, bytes[i] as i16, bytes[i + 2] as i16);
                i += 3;
                continue;
            }
            let escaped = handle_escapes(bytes, i + 2);
            if escaped.len() != 1 || escaped[0].0 != escaped[0].1 {
                let b = bytes[i] as i16;
                result.fsm_mut().add_edge(0, 1, b, b);
                i += 1;
                continue;
            }
            // C++ uses regex[0] here (a known quirk); preserved faithfully.
            result
                .fsm_mut()
                .add_edge(0, 1, bytes[0] as i16, escaped[0].0 as u8 as i16);
            i += 4;
            continue;
        }
        // escape at position i
        let escaped = handle_escapes(bytes, i);
        i += 1;
        if escaped.len() != 1 || escaped[0].0 != escaped[0].1 {
            for (lo, hi) in &escaped {
                result
                    .fsm_mut()
                    .add_edge(0, 1, *lo as u8 as i16, *hi as u8 as i16);
            }
            i += 1;
            continue;
        }
        let is_range = i + 2 < end_idx && bytes[i + 1] == b'-';
        if !is_range {
            result
                .fsm_mut()
                .add_edge(0, 1, escaped[0].0 as u8 as i16, escaped[0].1 as u8 as i16);
            i += 1;
            continue;
        }
        if bytes[i + 2] != b'\\' {
            result
                .fsm_mut()
                .add_edge(0, 1, escaped[0].0 as u8 as i16, bytes[i + 2] as i16);
            i += 3;
            continue;
        }
        let rhs = handle_escapes(bytes, i + 2);
        if rhs.len() != 1 || rhs[0].0 != rhs[0].1 {
            result
                .fsm_mut()
                .add_edge(0, 1, escaped[0].0 as u8 as i16, escaped[0].1 as u8 as i16);
            i += 1;
            continue;
        }
        result
            .fsm_mut()
            .add_edge(0, 1, escaped[0].0 as u8 as i16, rhs[0].0 as u8 as i16);
        i += 4;
    }

    // Coalesce the bitmap of covered bytes into maximal ranges,
    // optionally negated.
    let mut has_edge = [false; 0x100];
    for edge in result.fsm().edges(0) {
        for c in edge.min..=edge.max {
            has_edge[c as usize] = true;
        }
    }
    let mut new_fsm = Fsm::with_states(2);
    let mut last: i64 = -1;
    let want = !reverse;
    for c in 0..0x100usize {
        if has_edge[c] == want {
            if last == -1 {
                last = c as i64;
            }
        } else if last != -1 {
            new_fsm.add_edge(0, 1, last as i16, (c - 1) as i16);
            last = -1;
        }
    }
    if last != -1 {
        new_fsm.add_edge(0, 1, last as i16, 0xFF);
    }
    let mut ends = vec![false; new_fsm.num_states()];
    ends[1] = true;
    *result = FsmWithStartEnd::new(new_fsm, 0, ends, false);
}
