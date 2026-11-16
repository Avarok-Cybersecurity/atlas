// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: tests of the bounded KV view's scheduler side (port lane
//! L32): turn classification of ChatML prompts (new user message, tool result,
//! assistant continuation, no markers), the flag validation, and the
//! `/metrics` series while the view is off.

use super::*;

const IMS: u32 = 900;
const IME: u32 = 901;
const NL: u32 = 10;
const USER: u32 = 20;
const ASSISTANT: u32 = 21;
const SYSTEM: u32 = 22;
const TOOL: u32 = 950;
const THINK: u32 = 960;

fn markers() -> Markers {
    Markers {
        im_start: Some(IMS),
        tool_response: Some(TOOL),
    }
}

/// `<|im_start|> role \n content <|im_end|> \n`
fn block(role: u32, content: &[u32]) -> Vec<u32> {
    let mut v = vec![IMS, role, NL];
    v.extend_from_slice(content);
    v.extend([IME, NL]);
    v
}

fn header(think: bool) -> Vec<u32> {
    let mut v = vec![IMS, ASSISTANT, NL];
    if think {
        v.extend([THINK, NL]);
    }
    v
}

fn conversation(last_role: u32, last: &[u32], think: bool) -> (Vec<u32>, usize, usize) {
    let mut p = block(SYSTEM, &[1, 2, 3]);
    p.extend(block(USER, &(100..400).collect::<Vec<_>>()));
    p.extend(block(ASSISTANT, &[5, 6, 7]));
    let start = p.len() + 3;
    p.extend(block(last_role, last));
    let end = p.len() - 2;
    p.extend(header(think));
    (p, start, end)
}

#[test]
fn a_new_user_message_selects_from_its_own_tokens() {
    let question: Vec<u32> = (1000..1600).collect();
    for think in [false, true] {
        let (p, start, end) = conversation(USER, &question, think);
        let t = classify(&p, markers());
        assert_eq!(t.kind, TurnKind::NewUser);
        assert_eq!(t.query, start as u32..end as u32);
        assert_eq!(&p[start..end], &question[..]);
        // A later turn must repeat everything before the generation header.
        let g = p.len() - header(think).len();
        assert_eq!(t.match_len, g as u32);
    }
}

#[test]
fn a_tool_result_or_an_assistant_message_is_a_continuation() {
    let mut result = vec![TOOL];
    result.extend(2000..2300);
    let (p, _, _) = conversation(USER, &result, false);
    assert_eq!(classify(&p, markers()).kind, TurnKind::Continuation);
    let (p, _, _) = conversation(ASSISTANT, &[1, 2, 3], false);
    assert_eq!(classify(&p, markers()).kind, TurnKind::Continuation);
    // Without a resolved <tool_response> a tool result reads as a user message.
    let no_tool = Markers {
        tool_response: None,
        ..markers()
    };
    let (p, _, _) = conversation(USER, &result, false);
    assert_eq!(classify(&p, no_tool).kind, TurnKind::NewUser);
}

#[test]
fn prompts_without_a_generation_header_are_unmarked() {
    let raw: Vec<u32> = (0..5000).collect();
    let t = classify(&raw, markers());
    assert_eq!(t.kind, TurnKind::Unmarked);
    assert_eq!(t.query, 0..5000);
    assert_eq!(t.match_len, 5000);
    // No im_start id resolved at all.
    let (p, _, _) = conversation(USER, &[1, 2], false);
    assert_eq!(classify(&p, Markers::default()).kind, TurnKind::Unmarked);
    // An im_start far from the end is not a generation header.
    let mut tail = p.clone();
    tail.extend(0..100);
    assert_eq!(classify(&tail, markers()).kind, TurnKind::Unmarked);
    // A lone header with no message before it.
    assert_eq!(classify(&header(false), markers()).kind, TurnKind::Unmarked);
    assert_eq!(classify(&[], markers()).kind, TurnKind::Unmarked);
}

#[test]
fn an_empty_final_message_has_an_empty_query() {
    let (p, _, _) = conversation(USER, &[], false);
    let t = classify(&p, markers());
    assert_eq!(t.kind, TurnKind::NewUser);
    assert!(t.query.is_empty());
}

#[test]
fn markers_come_from_the_tokenizer_limits() {
    let l = SchedLimits {
        im_start_hard_stop: Some(IMS),
        tool_response_hard_stop: Some(TOOL),
        max_seq_len: 0,
    };
    assert_eq!(Markers::from_limits(&l), markers());
}

#[test]
fn flags_validate_without_turning_anything_on() {
    let none = PortArgs::default();
    assert!(validate_args(&none).is_ok());
    let tight = PortArgs {
        kv_view_budget_tokens: Some(256),
        ..PortArgs::default()
    };
    assert!(
        validate_args(&tight).is_err(),
        "budget below sink + recent + one block"
    );
}

#[test]
fn the_series_render_as_zeros_while_off() {
    let mut out = String::new();
    render_prometheus(&mut out);
    for name in [
        "atlas_kv_view_selections_total",
        "atlas_kv_view_select_seconds_total",
        "atlas_kv_view_tokens",
        "atlas_kv_view_skipped_total{reason=\"short_context\"}",
        "atlas_kv_view_recall_mass_sum",
        "atlas_kv_view_recall_mass_count",
        "atlas_kv_view_mtp_dense 0",
    ] {
        assert!(out.contains(name), "{name} missing from:\n{out}");
    }
    assert!(!view::enabled());
}
