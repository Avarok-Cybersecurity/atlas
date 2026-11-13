// SPDX-License-Identifier: AGPL-3.0-only
//! Tests of the reconstructed `canonical_v2` serializer: the exact rendering of representative golden requests, and the
//! release's token counts reproduced with the release tokenizer (skipped, with a note, when the release directory is not
//! on this machine: set `ROUTE_GATE_RELEASE` or keep it at `~/models/ARC-1.0`).

use std::path::PathBuf;

use serde_json::{Value, json};

use super::canonical::Serializer;
use super::serializer::{HfTokenizer, encode_blocks};

fn release_dir() -> Option<PathBuf> {
    let dir = std::env::var_os("ROUTE_GATE_RELEASE")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|h| PathBuf::from(h).join("models/ARC-1.0")))?;
    dir.join("tokenizer.json").exists().then_some(dir)
}

/// Representative golden requests (case id, request object, `expected.token_count` of the release).
fn golden_requests() -> Vec<(&'static str, Value, usize)> {
    let tools = json!([{"description": "Read a text file.",
        "input_schema": {"properties": {"path": {"type": "string"}}, "required": ["path"], "type": "object"},
        "name": "read_file"}]);
    let steps = |n: usize| -> Vec<Value> {
        let mut m = vec![json!({"content": "What is two plus two?", "role": "user"})];
        for k in 1..=n {
            m.push(json!({"content": [{"text": format!("Working on step {k}."), "type": "text"}], "role": "assistant"}));
            m.push(
                json!({"content": format!("Please continue with part {}.", k + 1), "role": "user"}),
            );
        }
        m
    };
    vec![
        ("synthetic-turns-0:0", json!({"messages": steps(0)}), 23),
        (
            // does not end in "." or "?": the closing of the part does not merge with the last token
            "topic-essay-0:0",
            json!({"messages": [{"content": "Edit this paragraph for clarity: 'The river, which it was long, it ran past the old mill where the town had grew.'", "role": "user"}]}),
            45,
        ),
        ("synthetic-turns-0:2", json!({"messages": steps(2)}), 117),
        (
            "synthetic-tools-0:0",
            json!({"messages": [{"content": "Why does src/app.py print hello?", "role": "user"}], "tools": tools}),
            93,
        ),
        (
            "synthetic-tools-0:1",
            json!({"messages": [
                {"content": "Why does src/app.py print hello?", "role": "user"},
                {"content": [{"text": "I will read the file.", "type": "text"},
                             {"id": "call_1", "input": {"path": "src/app.py"}, "name": "read_file", "type": "tool_use"}],
                 "role": "assistant"},
                {"content": [{"content": "print('hello')", "tool_use_id": "call_1", "type": "tool_result"}], "role": "user"}],
                "tools": tools}),
            190,
        ),
    ]
}

#[test]
fn renders_a_user_message_as_a_json_list_of_typed_parts() {
    let req = json!({"messages": [{"content": "What is two plus two?", "role": "user"}]});
    assert_eq!(
        Serializer::CanonicalV2.render(&req),
        vec![
            "<|im_start|>user\n[{\"text\": \"What is two plus two?\", \"type\": \"text\"}]<|im_end|>\n"
                .to_string()
        ]
    );
    // a list of text parts renders the same as the string it replaces
    let as_list = json!({"messages": [{"content": [{"type": "text", "text": "What is two plus two?"}], "role": "user"}]});
    assert_eq!(
        Serializer::CanonicalV2.render(&as_list),
        Serializer::CanonicalV2.render(&req)
    );
}

#[test]
fn renders_tools_as_one_system_block_and_names_the_tool_in_its_result() {
    let (_, req, _) = golden_requests().remove(4);
    let blocks = Serializer::CanonicalV2.render(&req);
    assert_eq!(blocks.len(), 4, "tools block plus three messages");
    assert_eq!(
        blocks[0],
        r##"<|im_start|>system
[{"text": "{\"name\": \"read_file\", \"description\": \"Read a text file.\", \"input_schema\": {\"type\": \"object\", \"properties\": {\"path\": {\"type\": \"string\"}}, \"required\": [\"path\"]}}", "type": "text"}]<|im_end|>
"##
    );
    assert!(blocks[3].contains(r#""tool_name": "read_file""#));
    assert!(blocks[2].contains(r#"{"id": "call_1", "input": {"path": "src/app.py"}, "name": "read_file", "type": "tool_use"}"#));
    // the stand-in does not know tools
    assert_eq!(Serializer::StandIn.render(&req).len(), 3);
}

#[test]
fn serializer_names_parse() {
    assert_eq!(
        Serializer::parse("v2").expect("v2"),
        Serializer::CanonicalV2
    );
    assert_eq!(
        Serializer::parse("standin").expect("standin"),
        Serializer::StandIn
    );
    assert!(Serializer::parse("v3").is_err());
}

#[test]
fn token_counts_equal_the_release_counts() {
    let Some(dir) = release_dir() else {
        eprintln!("skipped: no ARC-1.0 release directory (set ROUTE_GATE_RELEASE)");
        return;
    };
    let tok = HfTokenizer::from_release(&dir).expect("tokenizer");
    for (id, req, expected) in golden_requests() {
        let blocks = Serializer::CanonicalV2.render(&req);
        let got = encode_blocks(&tok, &blocks, None)
            .expect("encodes")
            .ids
            .len();
        assert_eq!(got, expected, "token count of {id}");
    }
}

#[test]
fn stand_in_is_twelve_tokens_per_message_short() {
    let Some(dir) = release_dir() else {
        eprintln!("skipped: no ARC-1.0 release directory (set ROUTE_GATE_RELEASE)");
        return;
    };
    let tok = HfTokenizer::from_release(&dir).expect("tokenizer");
    for (id, req, expected) in golden_requests().into_iter().take(3) {
        let n = req["messages"].as_array().map_or(0, Vec::len);
        let got = encode_blocks(&tok, &Serializer::StandIn.render(&req), None)
            .expect("encodes")
            .ids
            .len();
        // the essay prompt does not end in "." or "?" and gains one more token from the closing of its part
        let extra = usize::from(id == "topic-essay-0:0");
        assert_eq!(expected - got, 12 * n + extra, "{id}");
    }
}
