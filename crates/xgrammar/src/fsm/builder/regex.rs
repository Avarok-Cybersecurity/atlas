// SPDX-License-Identifier: AGPL-3.0-only
//
// RegexFsmBuilder — parses a regex string into a `RegexIr` and builds the
// NFA. Port of `RegexFSMBuilder::Build` (+ `RegexIR::CheckRepeat`) from
// `cpp/fsm_builder.cc`.
//
// The parser is a single left-to-right pass over the regex with an
// explicit stack. Stack entries are either a finished IR node or a raw
// control char (`(` or `|`).

use crate::fsm::with_start_end::FsmWithStartEnd;

use super::regex_ir::{REPEAT_NO_UPPER_BOUND, RegexIr, RegexState, RegexSymbol};
use super::regex_parse::{finalize_ir, parse_group_close};

/// A stack entry: a finished IR node, or a raw `(` / `|` control char.
#[derive(Debug, Clone)]
pub(crate) enum StackItem {
    State(RegexState),
    Ctrl(u8),
}

/// Parse `regex` and build its NFA.
pub fn build_regex(regex: &str) -> Result<FsmWithStartEnd, String> {
    let ir = parse_to_ir(regex)?;
    ir.build()
}

/// Parse `{n}` / `{n,}` / `{n,m}` starting at `start` (which must point
/// at `{`). On success returns `(lower, upper, index-of-'}')`.
fn check_repeat(regex: &[u8], mut start: usize) -> Result<(i32, i32, usize), String> {
    if regex[start] != b'{' {
        return Err("Invalid repeat format1".to_string());
    }
    start += 1;
    let mut lower = 0i32;
    let mut upper = REPEAT_NO_UPPER_BOUND;
    let skip_spaces = |s: &mut usize| {
        while *s < regex.len() && regex[*s] == b' ' {
            *s += 1;
        }
    };
    skip_spaces(&mut start);
    let mut num = String::new();
    while start < regex.len() && regex[start].is_ascii_digit() {
        num.push(regex[start] as char);
        start += 1;
    }
    if num.is_empty() {
        return Err("Invalid repeat format2".to_string());
    }
    lower = num.parse().unwrap_or(lower);
    skip_spaces(&mut start);
    if start >= regex.len() {
        return Err("Invalid repeat format5".to_string());
    }
    if regex[start] == b'}' {
        return Ok((lower, lower, start));
    }
    if regex[start] != b',' {
        return Err("Invalid repeat format3".to_string());
    }
    start += 1;
    skip_spaces(&mut start);
    if start < regex.len() && regex[start] == b'}' {
        return Ok((lower, upper, start));
    }
    num.clear();
    while start < regex.len() && regex[start].is_ascii_digit() {
        num.push(regex[start] as char);
        start += 1;
    }
    if num.is_empty() {
        return Err("Invalid repeat format4".to_string());
    }
    upper = num.parse().unwrap_or(upper);
    skip_spaces(&mut start);
    if start >= regex.len() || regex[start] != b'}' {
        return Err("Invalid repeat format5".to_string());
    }
    Ok((lower, upper, start))
}

/// Parse the regex string into a [`RegexIr`].
fn parse_to_ir(regex: &str) -> Result<RegexIr, String> {
    let bytes = regex.as_bytes();
    let mut stack: Vec<StackItem> = Vec::new();
    let mut left_bracket: i64 = -1;
    let n = bytes.len();
    let mut i = 0usize;

    while i < n {
        let c = bytes[i];
        // Anchors `^` (at start) / `$` (at end) are ignored.
        if i == 0 && c == b'^' {
            i += 1;
            continue;
        }
        if i == n - 1 && c == b'$' {
            i += 1;
            continue;
        }
        // Character class `[...]`.
        if c == b'[' {
            if left_bracket != -1 {
                return Err("Nested middle bracket!".to_string());
            }
            left_bracket = i as i64;
            i += 1;
            continue;
        }
        if c == b']' {
            if left_bracket == -1 {
                return Err("Invalid middle bracket!".to_string());
            }
            let lb = left_bracket as usize;
            stack.push(StackItem::State(RegexState::Leaf {
                regex: bytes[lb..=i].to_vec(),
            }));
            left_bracket = -1;
            i += 1;
            continue;
        }
        if left_bracket != -1 {
            // inside `[...]`, skip escaped bytes
            if c == b'\\' {
                i += 1;
            }
            i += 1;
            continue;
        }
        // Quantifiers `+ * ?`.
        if c == b'+' || c == b'*' || c == b'?' {
            let top = stack.pop();
            let child = match top {
                Some(StackItem::State(s)) => s,
                _ => return Err("Invalid regex: no state before operator!".to_string()),
            };
            let symbol = match c {
                b'+' => RegexSymbol::Plus,
                b'*' => RegexSymbol::Star,
                _ => RegexSymbol::Optional,
            };
            stack.push(StackItem::State(RegexState::Symbol {
                symbol,
                state: Box::new(child),
            }));
            i += 1;
            continue;
        }
        // Group open / alternation.
        if c == b'(' || c == b'|' {
            stack.push(StackItem::Ctrl(c));
            // skip `(?:` and `(?!` / `(?=` prefixes
            if c == b'('
                && i + 2 < n
                && bytes[i + 1] == b'?'
                && matches!(bytes[i + 2], b':' | b'!' | b'=')
            {
                i += 3;
                continue;
            }
            i += 1;
            continue;
        }
        // Group close.
        if c == b')' {
            parse_group_close(&mut stack)?;
            i += 1;
            continue;
        }
        // Repeat `{...}`.
        if c == b'{' {
            let top = stack.pop();
            let child = match top {
                Some(StackItem::State(s)) => s,
                _ => return Err("Invalid regex: no state before repeat!".to_string()),
            };
            let (lower, upper, end) = check_repeat(bytes, i)?;
            stack.push(StackItem::State(RegexState::Repeat {
                states: vec![child],
                lower_bound: lower,
                upper_bound: upper,
            }));
            i = end + 1;
            continue;
        }
        // Plain literal byte (possibly escaped). Sliced on `bytes` so a
        // multi-byte UTF-8 codepoint splits cleanly into per-byte leaves.
        let leaf_regex = if c != b'\\' {
            let s = bytes[i..i + 1].to_vec();
            i += 1;
            s
        } else {
            let s = bytes[i..(i + 2).min(n)].to_vec();
            i += 2;
            s
        };
        stack.push(StackItem::State(RegexState::Leaf { regex: leaf_regex }));
    }

    finalize_ir(stack)
}

// Stack helpers (`parse_group_close`, `finalize_ir`) live in
// `regex_parse.rs` — imported above via `use super::regex_parse::*`.

#[cfg(test)]
#[path = "regex_tests.rs"]
mod tests;
