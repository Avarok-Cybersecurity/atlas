// SPDX-License-Identifier: AGPL-3.0-only

//! Oracles for the evidence fingerprints. Every expected digest below was
//! computed once with Python's `hashlib` from the definition in the module
//! docs (SHA-256 over little-endian `u32` ids; SHA-256 of the content digest
//! followed by the reasoning digest), so the Rust code is checked against an
//! independent implementation, not against itself.

use super::stream::{EvidenceTap, wrap};
use super::*;
use crate::ir::{Choice, DeltaStream, FinishReason, StreamDelta, Usage};

const EMPTY_SHA: &str = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
const TOK_123: &str = "4636993d3e1da4e9d6b8f87b79e8f7c6d018580d52661950eabc3845c5897a4d";
const TOK_213: &str = "88a785e43f0814338c5cb47bde7de708c7172eabad671e4511199c78a3b4daa1";
const TOK_LE: &str = "9f64a747e1b97f131fabb6b447296c9b6f0201e79fb3c5356e6c77e89b6a806a";
const OUT_HELLO_EMPTY: &str = "6f56cb7315f2efda3924f966ffee8a0b59057481ffe287876c863db44f2e5683";
const OUT_EMPTY_HELLO: &str = "2cdc406caa00167e2d87877d4db045b28b543bd869cbb69c17764e1f64196560";
const OUT_HI_THERE_PLAN: &str = "c2c5cf71065465278211edba282d15350e74cbe41ba2ebc1d072f7d0669b22c1";
const OUT_EMPTY_EMPTY: &str = "2dba5dbc339e7316aea2683faf839c1b7b1ee2313db792112588118df066aa35";
const OUT_ACCENT: &str = "28b956ecd636805408a70f37a9fb3c76211c02c4eec6ab23398b2b674fc42b48";
const COMBINED_123_4: &str = "9ef83375526351ed0d0ab0431813bbe44140bfa2b29af8950b23ec77a34c4596";

fn content(text: &str, ids: &[u32]) -> StreamDelta {
    StreamDelta::Content {
        text: text.to_string(),
        token_ids: ids.to_vec(),
    }
}

fn reasoning(text: &str, ids: &[u32]) -> StreamDelta {
    StreamDelta::Reasoning {
        text: text.to_string(),
        token_ids: ids.to_vec(),
    }
}

fn finish(ids: &[u32]) -> StreamDelta {
    StreamDelta::Finish {
        reason: FinishReason::Stop,
        usage: Usage::default(),
        token_ids: ids.to_vec(),
    }
}

fn usage_of(d: &StreamDelta) -> Usage {
    match d {
        StreamDelta::Finish { usage, .. } => *usage,
        other => panic!("expected Finish, got {other:?}"),
    }
}

fn choice(content: Option<&str>, reasoning: Option<&str>) -> Choice {
    Choice {
        index: 0,
        content: content.map(str::to_string),
        reasoning: reasoning.map(str::to_string),
        tool_calls: Vec::new(),
        refusal: None,
        finish_reason: FinishReason::Stop,
        matched_stop: None,
        logprobs: None,
    }
}

#[test]
fn token_fingerprint_matches_the_reference_vectors() {
    assert_eq!(to_hex(&token_sha(&[])), EMPTY_SHA);
    assert_eq!(to_hex(&token_sha(&[1, 2, 3])), TOK_123);
    // Order matters.
    assert_eq!(to_hex(&token_sha(&[2, 1, 3])), TOK_213);
    // Little-endian layout: 0x04030201 is the bytes 01 02 03 04.
    assert_eq!(to_hex(&token_sha(&[0x0403_0201])), TOK_LE);
}

#[test]
fn incremental_token_hashing_equals_one_shot() {
    let mut t = TokenHasher::new();
    t.update(&[1]);
    t.update(&[]);
    t.update(&[2, 3]);
    assert_eq!(to_hex(&t.finish()), TOK_123);
}

#[test]
fn output_fingerprint_matches_the_reference_vectors() {
    assert_eq!(to_hex(&output_sha("hello", "")), OUT_HELLO_EMPTY);
    assert_eq!(to_hex(&output_sha("", "hello")), OUT_EMPTY_HELLO);
    assert_eq!(to_hex(&output_sha("hi there", "plan")), OUT_HI_THERE_PLAN);
    assert_eq!(to_hex(&output_sha("", "")), OUT_EMPTY_EMPTY);
    // UTF-8 bytes, not code points.
    assert_eq!(to_hex(&output_sha("h\u{e9}llo", "")), OUT_ACCENT);
}

#[test]
fn content_and_reasoning_are_not_interchangeable() {
    assert_ne!(output_sha("hello", ""), output_sha("", "hello"));
}

#[test]
fn output_fingerprint_ignores_chunk_boundaries() {
    let mut o = OutputHasher::new();
    o.add_content("hi ");
    o.add_reasoning("pl");
    o.add_content("th");
    o.add_reasoning("an");
    o.add_content("ere");
    assert_eq!(to_hex(&o.finish()), OUT_HI_THERE_PLAN);
}

#[test]
fn one_choice_is_its_own_digest_and_several_are_hashed_together() {
    let a = token_sha(&[1, 2, 3]);
    let b = token_sha(&[4]);
    assert_eq!(combine(&[a]), a);
    assert_eq!(to_hex(&combine(&[a, b])), COMBINED_123_4);
    assert_ne!(combine(&[a, b]), combine(&[b, a]), "index order matters");
    assert_eq!(to_hex(&combine(&[])), EMPTY_SHA);
}

#[test]
fn hex_is_64_lowercase_digits() {
    let h = to_hex(&token_sha(&[7]));
    assert_eq!(h.len(), 64);
    assert!(
        h.bytes()
            .all(|c| c.is_ascii_digit() || (b'a'..=b'f').contains(&c))
    );
}

#[test]
fn no_evidence_means_no_extras() {
    assert!(UsageExtras::default().is_empty());
    let c = [choice(Some("x"), None)];
    let x = blocking_extras(false, &[], &c, None);
    assert!(x.is_empty(), "fields absent unless requested");
}

#[test]
fn blocking_extras_fingerprint_the_final_choice_text() {
    let c = [choice(Some("hi there"), Some("plan"))];
    let x = blocking_extras(true, &[token_sha(&[1, 2, 3])], &c, None);
    assert_eq!(to_hex(&x.token_sha.unwrap()), TOK_123);
    assert_eq!(to_hex(&x.output_sha.unwrap()), OUT_HI_THERE_PLAN);
    assert!(x.confidence.is_none());
}

#[test]
fn a_null_content_hashes_as_empty_text() {
    // A tool-call-only or refused choice has `content: None`.
    let c = [choice(None, None)];
    let x = blocking_extras(true, &[token_sha(&[])], &c, None);
    assert_eq!(to_hex(&x.output_sha.unwrap()), OUT_EMPTY_EMPTY);
}

#[test]
fn multiple_choices_combine_in_index_order() {
    let cs = [
        choice(Some("hello"), None),
        choice(Some("hi there"), Some("plan")),
    ];
    let x = blocking_extras(true, &[token_sha(&[1, 2, 3]), token_sha(&[4])], &cs, None);
    assert_eq!(to_hex(&x.token_sha.unwrap()), COMBINED_123_4);
    let expect = combine(&[output_sha("hello", ""), output_sha("hi there", "plan")]);
    assert_eq!(x.output_sha, Some(expect));
}

#[test]
fn the_tap_fingerprints_a_stream_and_strips_ids_the_client_did_not_ask_for() {
    let mut tap = EvidenceTap::new(false);
    let seq = [
        reasoning("pl", &[1]),
        reasoning("an", &[]),
        content("hi ", &[2]),
        content("there", &[3]),
    ];
    for d in seq {
        match tap.apply(d) {
            StreamDelta::Content { token_ids, .. } | StreamDelta::Reasoning { token_ids, .. } => {
                assert!(token_ids.is_empty(), "ids must not reach the wire");
            }
            other => panic!("unexpected {other:?}"),
        }
    }
    // The residual id rides the terminal delta.
    let end = tap.apply(finish(&[4]));
    let u = usage_of(&end);
    // [1, 2, 3, 4] in order, whichever delta carried each id.
    assert_eq!(u.atlas.token_sha, Some(token_sha(&[1, 2, 3, 4])));
    assert_eq!(to_hex(&u.atlas.output_sha.unwrap()), OUT_HI_THERE_PLAN);
    match end {
        StreamDelta::Finish { token_ids, .. } => assert!(token_ids.is_empty()),
        other => panic!("expected Finish, got {other:?}"),
    }
}

#[test]
fn the_tap_leaves_ids_alone_when_the_client_asked_for_them() {
    let mut tap = EvidenceTap::new(true);
    assert_eq!(tap.apply(content("a", &[9, 8])), content("a", &[9, 8]));
    match tap.apply(finish(&[7])) {
        StreamDelta::Finish {
            token_ids, usage, ..
        } => {
            assert_eq!(token_ids, vec![7]);
            assert_eq!(usage.atlas.token_sha, Some(token_sha(&[9, 8, 7])));
        }
        other => panic!("expected Finish, got {other:?}"),
    }
}

#[test]
fn tool_and_error_deltas_pass_through_untouched() {
    let mut tap = EvidenceTap::new(false);
    let start = StreamDelta::ToolCallStart {
        index: 0,
        id: "c1".into(),
        name: "f".into(),
    };
    assert_eq!(tap.apply(start.clone()), start);
    let err = StreamDelta::Error {
        message: "boom".into(),
    };
    assert_eq!(tap.apply(err.clone()), err);
    // Tool arguments carry token ids (they are generated tokens) but no text
    // for the output fingerprint.
    match tap.apply(StreamDelta::ToolCallArgs {
        index: 0,
        fragment: "{}".into(),
        token_ids: vec![5],
    }) {
        StreamDelta::ToolCallArgs { token_ids, .. } => assert!(token_ids.is_empty()),
        other => panic!("unexpected {other:?}"),
    }
    let u = usage_of(&tap.apply(finish(&[])));
    assert_eq!(u.atlas.token_sha, Some(token_sha(&[5])));
    assert_eq!(u.atlas.output_sha, Some(output_sha("", "")));
}

#[test]
fn the_wrapped_stream_yields_the_same_deltas_plus_the_fingerprints() {
    let deltas = vec![content("hi there", &[1, 2, 3]), finish(&[4])];
    let inner: DeltaStream = Box::pin(futures::stream::iter(deltas));
    let out: Vec<StreamDelta> = futures::executor::block_on(async {
        use futures::StreamExt;
        wrap(inner, false).collect().await
    });
    assert_eq!(out.len(), 2);
    assert_eq!(out[0], content("hi there", &[]));
    let u = usage_of(&out[1]);
    assert_eq!(
        to_hex(&u.atlas.token_sha.unwrap()),
        to_hex(&token_sha(&[1, 2, 3, 4]))
    );
    assert_eq!(u.atlas.output_sha, Some(output_sha("hi there", "")));
}

#[test]
fn trimming_returns_exactly_what_the_client_asked_for() {
    use crate::ir::{ChoiceLogprobs, TokenLogprob};
    let lp = |n: usize| TokenLogprob {
        token: "t".into(),
        logprob: -0.1,
        top: (0..n).map(|i| (format!("a{i}"), -(i as f32))).collect(),
    };
    let mut c = choice(Some("x"), None);
    c.logprobs = Some(ChoiceLogprobs {
        content: vec![lp(4), lp(4)],
    });
    // Client asked nothing: the scheduler's extra logprobs vanish.
    let mut none = c.clone();
    trim_choice_logprobs(&mut none, None);
    assert!(none.logprobs.is_none());
    // Client asked for 1 alternative: each token keeps one.
    let mut one = c.clone();
    trim_choice_logprobs(&mut one, Some(1));
    assert!(
        one.logprobs
            .unwrap()
            .content
            .iter()
            .all(|t| t.top.len() == 1)
    );
    // Client asked for 20 (more than present): nothing is invented.
    let mut many = c.clone();
    trim_choice_logprobs(&mut many, Some(20));
    assert!(
        many.logprobs
            .unwrap()
            .content
            .iter()
            .all(|t| t.top.len() == 4)
    );
    // No logprobs at all stays none.
    let mut empty = choice(Some("x"), None);
    trim_choice_logprobs(&mut empty, Some(3));
    assert!(empty.logprobs.is_none());
}

#[test]
fn the_streaming_confidence_is_attached_only_when_asked() {
    let mut acc = ConfAcc::new(8);
    acc.observe_logits(&[2.0, 1.0, 0.0]);

    let mut asked = vec![content("x", &[]), finish(&[])];
    finish_conf(&mut asked, Some(acc), true);
    let s = usage_of(&asked[1]).atlas.confidence.expect("reported");
    assert_eq!(s.tokens, 1);

    // Collected for the metrics but not asked for: the usage block is clean.
    let mut quiet = vec![finish(&[])];
    finish_conf(&mut quiet, Some(acc), false);
    assert!(usage_of(&quiet[0]).atlas.is_empty());

    // Nothing collected (no logprobs arrived): nothing attached.
    let mut none = vec![finish(&[])];
    finish_conf(&mut none, Some(ConfAcc::new(8)), true);
    assert!(usage_of(&none[0]).atlas.is_empty());
    finish_conf(&mut none, None, true);
    assert!(usage_of(&none[0]).atlas.is_empty());
}

#[test]
fn stream_options_follow_the_request_and_the_plan() {
    use crate::scheduler::conf_summary::plan;
    // Lowered from the wire, so the request-field mapping is part of the check.
    let wire: crate::openai::ChatCompletionRequest = serde_json::from_value(serde_json::json!({
        "model": "m",
        "messages": [{"role": "user", "content": "hi"}],
        "stream": true,
        "atlas_speculative": false,
        "atlas_confidence": true
    }))
    .expect("valid chat request");
    let mut req: crate::ir::ChatRequest = wire.into();
    let o = StreamPortOpts::for_request(&req, &plan(false, req.confidence, None));
    assert_eq!(
        o,
        StreamPortOpts {
            no_drafts: true,
            conf_collect: true,
            conf_report: true
        }
    );
    req.no_drafts = false;
    req.confidence = false;
    let o = StreamPortOpts::for_request(&req, &plan(false, req.confidence, None));
    assert_eq!(o, StreamPortOpts::default());
}
