// SPDX-License-Identifier: AGPL-3.0-only
//! Tests of the request serializer stand-in, block offsets, row selection, the token report (all without a model),
//! the Python-style JSON dump, the CLI flag parser and the readouts file round trip.

use serde_json::json;

use super::cli::{Flags, run};
use super::error::GResult;
use super::fit_eval::{load_inputs, write_readouts};
use super::goldens::token_report;
use super::serializer::{
    IM_END, IM_START, Selection, TextTokenizer, encode_blocks, flatten_content, py_json_dumps,
    render_blocks, select_row,
};

/// Splits at the two special tokens and otherwise one token per character.
struct FakeTok;

impl TextTokenizer for FakeTok {
    type Id = String;

    fn encode(&self, text: &str) -> GResult<Vec<String>> {
        let mut out = Vec::new();
        let mut rest = text;
        while !rest.is_empty() {
            if let Some(sp) = [IM_START, IM_END].iter().find(|s| rest.starts_with(**s)) {
                out.push((*sp).to_string());
                rest = &rest[sp.len()..];
            } else {
                let c = rest.chars().next().expect("non-empty");
                out.push(c.to_string());
                rest = &rest[c.len_utf8()..];
            }
        }
        Ok(out)
    }
}

#[test]
fn render_and_offsets() {
    let msgs = vec![
        json!({"role": "user", "content": "hi"}),
        json!({"role": "assistant", "content": [{"type": "text", "text": "yo"}]}),
    ];
    let blocks = render_blocks(&msgs);
    assert_eq!(blocks[0], "<|im_start|>user\nhi<|im_end|>\n");
    assert_eq!(blocks[1], "<|im_start|>assistant\nyo<|im_end|>\n");
    let enc = encode_blocks(&FakeTok, &blocks, None).expect("encodes");
    // block 0: IM_START u s e r \n h i IM_END \n = 10 tokens
    assert_eq!(enc.nl[0], 9);
    assert_eq!(enc.im_end[0], 8);
    assert_eq!(enc.ids[enc.nl[0]], "\n");
    assert_eq!(enc.ids[enc.im_end[1]], IM_END);
    assert_eq!(enc.nl[1], enc.ids.len() - 1);
    let shelled = encode_blocks(&FakeTok, &blocks, Some("SHELL")).expect("encodes");
    assert_eq!(shelled.nl[0], enc.nl[0] + 5);
    assert_eq!(enc.pick_offsets("im_end").expect("offsets"), enc.im_end);
    assert!(enc.pick_offsets("bogus").is_err());
    assert!(encode_blocks(&FakeTok, &["no specials".to_string()], None).is_err());
}

#[test]
fn flatten_content_forms() {
    assert_eq!(flatten_content(&json!("plain")), "plain");
    assert_eq!(flatten_content(&json!(null)), "");
    let parts = json!([
        {"type": "text", "text": "a"},
        {"type": "tool_use", "name": "read", "input": {"p": 1}},
        {"type": "tool_result", "content": "out"}
    ]);
    assert_eq!(
        flatten_content(&parts),
        "a\n[tool_use read] {\"p\": 1}\n[tool_result] out"
    );
    assert_eq!(
        flatten_content(&json!([{"text": "bare"}, "str", {"x": 1}])),
        "bare\nstr\n{\"x\": 1}"
    );
}

#[test]
fn python_style_json_dump() {
    assert_eq!(
        py_json_dumps(&json!({"b": 1, "a": [1, "x", null, true]})),
        "{\"a\": [1, \"x\", null, true], \"b\": 1}"
    );
    assert_eq!(
        py_json_dumps(&json!("caf\u{e9} \"q\"\n")),
        "\"caf\\u00e9 \\\"q\\\"\\n\""
    );
    assert_eq!(py_json_dumps(&json!("\u{1f600}")), "\"\\ud83d\\ude00\"");
}

#[test]
fn select_row_forms() {
    let rows = vec![vec![1.0, 2.0], vec![3.0, 4.0], vec![5.0, 6.0]];
    let roles: Vec<String> = ["user", "assistant", "user"]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    assert_eq!(select_row(&rows, &roles, Selection::Last), vec![5.0, 6.0]);
    assert_eq!(select_row(&rows, &roles, Selection::Mean), vec![3.0, 4.0]);
    let roles2: Vec<String> = ["user", "user", "assistant"]
        .iter()
        .map(|s| (*s).to_string())
        .collect();
    assert_eq!(
        select_row(&rows, &roles2, Selection::LastUser),
        vec![3.0, 4.0]
    );
    assert!(Selection::parse("bogus").is_err());
    assert_eq!(
        Selection::parse("last_user").expect("parses"),
        Selection::LastUser
    );
}

#[test]
fn token_report_deltas() {
    let mut goldens = vec![
        json!({"case_id": "a", "request": {"request": {"messages": [{"role": "user", "content": "hi"}]}},
               "expected": {"token_count": 15}}),
        json!({"case_id": "b", "request": {"request": {"messages": [{"role": "user", "content": "hiya"}]}},
               "expected": {"token_count": 18}}),
    ];
    let rep = token_report(&FakeTok, &goldens, None).expect("report");
    // "hiya" is 12 stand-in tokens, "hi" is 10
    assert_eq!(
        rep.cases.iter().map(|c| c.delta).collect::<Vec<_>>(),
        vec![5, 6]
    );
    assert!(!rep.constant_offset);
    goldens[1]["expected"]["token_count"] = json!(17);
    assert!(
        token_report(&FakeTok, &goldens, None)
            .expect("report")
            .constant_offset
    );
}

#[test]
fn flag_parser() {
    let args: Vec<String> = [
        "--out",
        "a.json",
        "--seed=7",
        "--fresh",
        "--shell-file",
        "x",
        "--shell-file",
        "y",
    ]
    .iter()
    .map(|s| (*s).to_string())
    .collect();
    let f = Flags::parse(&args, &["out", "seed", "shell-file"], &["fresh"]).expect("parses");
    assert_eq!(f.get("out"), Some("a.json"));
    assert_eq!(f.num("seed", 1u64).expect("number"), 7);
    assert_eq!(f.num("missing", 3u64).expect("default"), 3);
    assert!(f.has("fresh") && !f.has("nope"));
    assert_eq!(f.many("shell-file"), vec!["x".to_string(), "y".to_string()]);
    assert!(f.req("absent").is_err());
    assert!(Flags::parse(&["--unknown".to_string()], &["out"], &[]).is_err());
    assert!(Flags::parse(&["--out".to_string()], &["out"], &[]).is_err());
    assert!(Flags::parse(&["stray".to_string()], &["out"], &[]).is_err());
}

#[test]
fn cli_exit_codes_for_usage_errors() {
    assert_eq!(run(&[]), 2);
    assert_eq!(run(&["bogus".to_string()]), 2);
    assert_eq!(run(&["summary".to_string()]), 2);
    assert_eq!(run(&["help".to_string()]), 0);
}

#[test]
fn readouts_round_trip_through_load_inputs() {
    let dir = tempfile::tempdir().expect("tempdir");
    let labels = dir.path().join("l.jsonl");
    let rows = [
        json!({"id": "p0", "passed": true, "source": "s", "prompt": "a"}),
        json!({"id": "p1", "passed": false, "source": "s", "prompt": "b"}),
        json!({"id": "p2", "passed": null, "source": "s", "prompt": "c"}),
        json!({"id": "p3", "passed": false, "source": "s", "prompt": "d", "truncated": true}),
        json!({"id": "p4", "passed": false, "source": "s", "prompt": "e", "reason": "no_answer"}),
        json!({"id": "p5", "passed": true, "source": "s", "prompt": "f"}),
    ];
    std::fs::write(
        &labels,
        rows.iter().map(|r| format!("{r}\n")).collect::<String>(),
    )
    .expect("write labels");
    let ids: Vec<String> = (0..6).map(|i| format!("p{i}")).collect();
    let x: Vec<f32> = (0..12).map(|v| v as f32).collect();
    let ok = [true, true, true, true, true, false];
    let path = dir.path().join("r.safetensors");
    write_readouts(
        &path,
        &ids,
        &x,
        2,
        &[1, 2, 3, 4, 5, 6],
        &ok,
        &json!({"k": 1}),
    )
    .expect("write readouts");
    let got = load_inputs(&labels, &path, true, true).expect("load");
    assert_eq!(
        got.rows.iter().map(|r| r.id.as_str()).collect::<Vec<_>>(),
        vec!["p0", "p1"]
    );
    assert_eq!((got.x.rows, got.x.cols), (2, 2));
    assert_eq!(got.x.row(1), &[2.0, 3.0]);
    assert_eq!(
        (
            got.dropped.no_label,
            got.dropped.no_readout,
            got.dropped.truncated,
            got.dropped.no_answer
        ),
        (1, 1, 1, 1)
    );
    assert_eq!(got.meta, json!({"k": 1}));
    let all = load_inputs(&labels, &path, false, false).expect("load");
    assert_eq!(all.rows.len(), 4);
}
