// SPDX-License-Identifier: AGPL-3.0-only
//
// Repro for the corrupted token bitmask inside JSON-string values when
// serving grammar-constrained tool calls (structural tag + Qwen3.6
// byte-level BPE tokenizer). Loads the REAL tokenizer.json from the HF
// cache; skips (with a message) when it is not present.

use xgrammar::compiler::GrammarCompiler;
use xgrammar::detect_metadata_from_hf;
use xgrammar::matcher::GrammarMatcher;
use xgrammar::tokenizer::TokenizerInfo;

/// Candidate paths for the Qwen3.6 tokenizer.json (unsloth 27B NVFP4 —
/// the model from the bug report — plus a fallback with the identical
/// tokenizer).
const TOKENIZER_PATHS: &[&str] = &[
    "/workspace/.cache/huggingface/hub/models--unsloth--Qwen3.6-27B-NVFP4/snapshots/890bdef7a42feba6d83b6e17a03315c694112f2a/tokenizer.json",
    "/workspace/.cache/huggingface/hub/models--Qwen--Qwen3.6-27B/snapshots/6a9e13bd6fc8f0983b9b99948120bc37f49c13e9/tokenizer.json",
];

/// The structural tag from the serving path (verbatim from the bug
/// report; ml.predict_house_price with required location+size).
const STRUCTURAL_TAG: &str = r#"{"type":"structural_tag","format":{"type":"triggered_tags","triggers":["<tool_call>"],"tags":[{"type":"tag","begin":"<tool_call>\n{\"name\": \"ml.predict_house_price\", \"arguments\": ","content":{"type":"json_schema","json_schema":{"type":"object","properties":{"location":{"type":"string","description":"Location of the house"},"size":{"type":"integer","description":"Size of the house in square feet"}},"required":["location","size"]}},"end":"}\n</tool_call>"}],"at_least_one":false,"stop_after_first":false}}"#;

fn tokenizer_json() -> Option<String> {
    if let Ok(p) = std::env::var("QWEN_TOKENIZER_JSON")
        && let Ok(s) = std::fs::read_to_string(&p)
    {
        return Some(s);
    }
    TOKENIZER_PATHS
        .iter()
        .find_map(|p| std::fs::read_to_string(p).ok())
}

/// Build the ordered vocab exactly as spark-server's
/// `extract_ordered_vocab` does (index = token id), from the raw
/// tokenizer.json: `model.vocab` plus `added_tokens`.
fn ordered_vocab(json: &serde_json::Value) -> Vec<String> {
    let vocab = json["model"]["vocab"]
        .as_object()
        .expect("model.vocab object");
    let added = json["added_tokens"].as_array();
    let max_id = vocab
        .values()
        .map(|v| v.as_i64().unwrap())
        .chain(
            added
                .into_iter()
                .flatten()
                .map(|t| t["id"].as_i64().unwrap()),
        )
        .max()
        .unwrap_or(0) as usize;
    let mut ordered = vec![String::new(); max_id + 1];
    for (tok, id) in vocab {
        ordered[id.as_i64().unwrap() as usize] = tok.clone();
    }
    for t in added.into_iter().flatten() {
        ordered[t["id"].as_i64().unwrap() as usize] = t["content"].as_str().unwrap().to_string();
    }
    ordered
}

/// Build a `TokenizerInfo` the way spark-server's
/// `GrammarEngine::from_tokenizer` does: metadata auto-detection from
/// the serialized tokenizer JSON + explicit stop ids (`<|im_end|>`).
fn build_tokenizer_info(raw: &str) -> TokenizerInfo {
    let json: serde_json::Value = serde_json::from_str(raw).expect("tokenizer.json parses");
    let vocab = ordered_vocab(&json);
    let metadata = detect_metadata_from_hf(raw).expect("metadata detection");
    let im_end = vocab
        .iter()
        .position(|t| t == "<|im_end|>")
        .expect("<|im_end|> in vocab") as i32;
    TokenizerInfo::new(
        &vocab,
        metadata.vocab_type,
        None,
        Some(vec![im_end]),
        metadata.add_prefix_space,
    )
}

fn compile_matcher(info: &TokenizerInfo) -> GrammarMatcher {
    let compiler = GrammarCompiler::new(info.clone(), 1, true, -1);
    let compiled = compiler
        .compile_structural_tag(STRUCTURAL_TAG)
        .expect("structural tag compiles");
    GrammarMatcher::from_compiled_grammar(compiled)
}

fn is_set(mask: &[i32], token_id: i32) -> bool {
    let idx = token_id as usize;
    (mask[idx / 32] >> (idx % 32)) & 1 == 1
}

/// TEST A (mask walk): tokenize the desired tool-call output with the
/// real tokenizer and step the matcher token by token. At every step
/// the natural next token must be legal in `fill_next_token_bitmask`'s
/// mask, and `accept_token` must succeed.
///
/// The id sequence is the real BPE encoding of
/// `<tool_call>\n{"name": "ml.predict_house_price", "arguments": {"location": "New York", "size": 3000}}\n</tool_call>`
/// produced by `tokenizers` 0.22 for the unsloth/Qwen3.6-27B-NVFP4
/// tokenizer. Each id's decoded bytes are asserted against the
/// expected byte string, so a stale id fails loudly rather than
/// testing the wrong thing.
#[test]
fn structural_tag_natural_tokens_stay_legal() {
    let Some(raw) = tokenizer_json() else {
        eprintln!("SKIP: Qwen3.6 tokenizer.json not found in HF cache");
        return;
    };
    let info = build_tokenizer_info(&raw);
    let mut matcher = compile_matcher(&info);

    // (token_id, decoded bytes) — real BPE encoding, see doc comment.
    let steps: &[(i32, &str)] = &[
        (248058, "<tool_call>"),
        (198, "\n"),
        (4754, "{\""),
        (591, "name"),
        (763, "\":"),
        (328, " \""),
        (980, "ml"),
        (23028, ".predict"),
        (62101, "_house"),
        (8768, "_price"),
        (487, "\","),
        (328, " \""),
        (15889, "arguments"),
        (763, "\":"),
        (5046, " {\""),
        (2447, "location"),
        (763, "\":"),
        (328, " \""),
        (3446, "New"),
        (4121, " York"),
        (487, "\","),
        (328, " \""),
        (2073, "size"),
        (763, "\":"),
        (220, " "),
        (18, "3"),
        (15, "0"),
        (15, "0"),
        (15, "0"),
        (3307, "}}"),
        (198, "\n"),
        (248059, "</tool_call>"),
    ];

    let vocab_size = info.vocab_size();
    let words = vocab_size.div_ceil(32);
    let mut mask = vec![0i32; words];
    let mut failures = Vec::new();

    for (pos, &(id, expect_bytes)) in steps.iter().enumerate() {
        assert_eq!(
            info.decoded_vocab()[id as usize],
            expect_bytes.as_bytes(),
            "token id {id} at step {pos} decodes differently — update the id table",
        );
        mask.fill(0);
        matcher
            .fill_next_token_bitmask(&mut mask, 0, false)
            .expect("fill succeeds");
        if !is_set(&mask, id) {
            failures.push(format!(
                "step {pos}: natural token {id} ({expect_bytes:?}) masked ILLEGAL"
            ));
        }
        assert!(
            matcher.accept_token(id, false),
            "step {pos}: accept_token({id}, {expect_bytes:?}) rejected — matcher desynced",
        );
    }
    assert!(
        failures.is_empty(),
        "natural tokens masked illegal:\n{}",
        failures.join("\n"),
    );
}

/// TEST B (required-property hole): after `{"location":""` the object
/// close `}` must be ILLEGAL — the required property `size` is missing.
#[test]
fn structural_tag_required_property_blocks_close() {
    let Some(raw) = tokenizer_json() else {
        eprintln!("SKIP: Qwen3.6 tokenizer.json not found in HF cache");
        return;
    };
    let info = build_tokenizer_info(&raw);
    let mut matcher = compile_matcher(&info);

    let prefix =
        "<tool_call>\n{\"name\": \"ml.predict_house_price\", \"arguments\": {\"location\":\"\"";
    assert!(
        matcher.accept_string(prefix, false),
        "grammar-valid prefix must be accepted",
    );

    let close = info
        .decoded_vocab()
        .iter()
        .position(|t| t == b"}")
        .expect("`}` in vocab") as i32;
    let vocab_size = info.vocab_size();
    let mut mask = vec![0i32; vocab_size.div_ceil(32)];
    matcher
        .fill_next_token_bitmask(&mut mask, 0, false)
        .expect("fill succeeds");
    assert!(
        !is_set(&mask, close),
        "`}}` is mask-legal after {{\"location\":\"\" — required `size` hole",
    );
    // Belt and braces: the parser itself must also reject it.
    let mut probe = compile_matcher(&info);
    assert!(probe.accept_string(prefix, false));
    assert!(
        !probe.accept_token(close, false),
        "`}}` was ACCEPTED after {{\"location\":\"\" — required `size` hole",
    );
}
