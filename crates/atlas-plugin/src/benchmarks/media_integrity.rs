// SPDX-License-Identifier: AGPL-3.0-only

//! Cross-request and cross-path integrity checks, shared by the image and
//! video benchmarks.
//!
//! Everything here targets machinery that a single well-formed request cannot
//! reach, and whose failure mode is a FLUENT ANSWER TO THE WRONG INPUT rather
//! than an error. That is the class this repository keeps rediscovering —
//! logits-row aliasing across mixed steps, prefix-cache contamination serving
//! another request's completion, and most recently a splice that matched only
//! the image pad token and left video positions holding their raw embeddings.
//! None of those produce an error; all of them produce prose.
//!
//! The four checks:
//!
//! * [`heterogeneous_concurrency`] — several DIFFERENT requests in flight at
//!   once, each required to get its own answer.
//! * [`cache_leak`] — the same prompt text with a different image, back to
//!   back, so a prefix cache that forgot vision is caught.
//! * [`long_prompt_path`] — a prompt long enough to leave the single-chunk
//!   fast path, exercising the other splice.
//! * [`media_in_history`] — media in an earlier turn and in a tool result,
//!   rather than in the message that asks the question.

use std::time::Duration;

use serde_json::{Value, json};

use crate::http;
use crate::plugin::PluginHandle;

/// One concurrent subject: the request, a predicate scoring ITS OWN reply,
/// and a label for the failure message. Named because the tuple is otherwise
/// wide enough that clippy objects — and because "the predicate belongs to
/// this request, not to the batch" is the whole idea of the leg.
pub type Subject = (Value, Box<dyn Fn(&str) -> bool + Send + Sync>, String);

/// Outcome of one integrity check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Cell {
    Pass { id: &'static str, detail: String },
    Fail { id: &'static str, detail: String },
    Skipped { id: &'static str, why: String },
    Error { id: &'static str, msg: String },
}

impl Cell {
    pub fn id(&self) -> &'static str {
        match self {
            Cell::Pass { id, .. }
            | Cell::Fail { id, .. }
            | Cell::Skipped { id, .. }
            | Cell::Error { id, .. } => id,
        }
    }
    pub fn passed(&self) -> bool {
        matches!(self, Cell::Pass { .. })
    }
    pub fn measured(&self) -> bool {
        matches!(self, Cell::Pass { .. } | Cell::Fail { .. })
    }
    pub fn line(&self) -> String {
        match self {
            Cell::Pass { id, detail } => format!("{id}: {detail}"),
            Cell::Fail { id, detail } => format!("{id}: FAILED — {detail}"),
            Cell::Skipped { id, why } => format!("{id}: skipped — {why}"),
            Cell::Error { id, msg } => format!("{id}: {msg}"),
        }
    }
}

/// One image part plus a prompt, as a chat body.
pub fn image_request(
    model: &str,
    mime: &str,
    bytes: &[u8],
    prompt: &str,
    max_tokens: usize,
) -> Value {
    use base64::Engine;
    let mut uri = format!("data:{mime};base64,");
    base64::engine::general_purpose::STANDARD.encode_string(bytes, &mut uri);
    json!({
        "model": model,
        "stream": true,
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": false},
        "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": uri}},
            {"type": "text", "text": prompt},
        ]}],
    })
}

/// ★ SEVERAL DIFFERENT REQUESTS AT ONCE, each scored against its OWN expected
/// answer.
///
/// A concurrency check that fires N copies of the SAME request cannot find the
/// bug worth finding here. Requests share one packed ViT output buffer and one
/// grid vector, sliced by per-request base offsets (`vision_row_base`,
/// `vision_grid_base`, `vision_owned_images`); when every request is identical,
/// every offset is interchangeable and an off-by-one is invisible. Give each
/// request different content — including one with NO image and one with TWO —
/// and a mis-sliced offset hands request A the answer to request B.
///
/// `subjects` is (body, predicate-on-its-own-reply, label).
pub async fn heterogeneous_concurrency(
    handle: &PluginHandle,
    subjects: Vec<Subject>,
    timeout: Duration,
) -> Cell {
    const ID: &str = "heterogeneous-concurrency";
    if subjects.len() < 2 {
        return Cell::Skipped {
            id: ID,
            why: "fewer than two subjects".to_string(),
        };
    }
    let n = subjects.len();
    let futures: Vec<_> = subjects
        .iter()
        .map(|(body, _, _)| http::chat_stream(handle.target(), body, timeout))
        .collect();
    let outs = futures::future::join_all(futures).await;

    let mut wrong = Vec::new();
    let mut errors = Vec::new();
    for (out, (_, want, label)) in outs.into_iter().zip(subjects.iter()) {
        match out {
            Ok(o) => {
                if !want(o.text.trim()) {
                    wrong.push(format!(
                        "{label} got \"{}\"",
                        crate::benchmarks::one_line(o.text.chars().take(60).collect::<String>())
                    ));
                }
            }
            Err(e) => errors.push(format!(
                "{label}: {}",
                crate::benchmarks::one_line(format!("{e:#}"))
            )),
        }
    }
    if !errors.is_empty() {
        return Cell::Error {
            id: ID,
            msg: errors.join("; "),
        };
    }
    if wrong.is_empty() {
        Cell::Pass {
            id: ID,
            detail: format!("{n} different requests in flight, each got its own answer"),
        }
    } else {
        Cell::Fail {
            id: ID,
            detail: format!(
                "{}/{n} replies did not match their own input — {}",
                wrong.len(),
                wrong.join("; ")
            ),
        }
    }
}

/// ★ THE SAME PROMPT TEXT WITH A DIFFERENT IMAGE, back to back.
///
/// Vision prompts must not be served from the prefix cache. The guard is a
/// single predicate (`tokens_have_vision_pad`) that has to recognise every pad
/// token there is — it silently missed the VIDEO token until 2026-08-14. When
/// it misses one, the second request matches the first's cached prefix and is
/// answered from it: fast, fluent, and about the previous image.
///
/// The two requests differ ONLY in the image bytes, which is what makes a hit
/// unambiguous.
pub async fn cache_leak(
    handle: &PluginHandle,
    first: Value,
    second: Value,
    second_want: &(dyn Fn(&str) -> bool + Sync),
    first_marker: &(dyn Fn(&str) -> bool + Sync),
    timeout: Duration,
) -> Cell {
    const ID: &str = "prefix-cache-isolation";
    let a = match http::chat_stream(handle.target(), &first, timeout).await {
        Ok(o) => o,
        Err(e) => {
            return Cell::Error {
                id: ID,
                msg: crate::benchmarks::one_line(format!("{e:#}")),
            };
        }
    };
    let b = match http::chat_stream(handle.target(), &second, timeout).await {
        Ok(o) => o,
        Err(e) => {
            return Cell::Error {
                id: ID,
                msg: crate::benchmarks::one_line(format!("{e:#}")),
            };
        }
    };
    let b_text = b.text.trim();
    if second_want(b_text) {
        return Cell::Pass {
            id: ID,
            detail: format!(
                "identical prompt, different media: second reply is its own ({} then {} prompt \
                 tokens)",
                a.prompt_tokens, b.prompt_tokens
            ),
        };
    }
    // Distinguish "answered the FIRST image" — a cache hit — from merely wrong.
    let detail = if first_marker(b_text) {
        format!(
            "the second request was answered with the FIRST image's content — the prefix cache \
             served a vision prompt. Reply: \"{}\"",
            crate::benchmarks::one_line(b_text.chars().take(80).collect::<String>())
        )
    } else {
        format!(
            "the second reply matched neither expectation: \"{}\"",
            crate::benchmarks::one_line(b_text.chars().take(80).collect::<String>())
        )
    };
    Cell::Fail { id: ID, detail }
}

/// ★ A PROMPT LONG ENOUGH TO LEAVE THE SINGLE-CHUNK PATH.
///
/// `phase_start_prefills` admits a media request to the batched co-dispatch
/// encode only when its whole prompt fits one chunk, because — in the
/// scheduler's own words — "the splice + MRoPE reset img_idx per chunk, so a
/// pad run must not straddle a chunk boundary". A longer prompt falls back to
/// a per-request self-encode with its OWN splice in a different file.
///
/// Both splices exist, so both should be exercised; only the short one ever is
/// otherwise. The check is that the SAME image answers the SAME question the
/// same way with a large block of filler text prepended.
pub async fn long_prompt_path(
    handle: &PluginHandle,
    short: Value,
    long: Value,
    want: &(dyn Fn(&str) -> bool + Sync),
    timeout: Duration,
) -> Cell {
    const ID: &str = "long-prompt-splice";
    let s = match http::chat_stream(handle.target(), &short, timeout).await {
        Ok(o) => o,
        Err(e) => {
            return Cell::Error {
                id: ID,
                msg: crate::benchmarks::one_line(format!("{e:#}")),
            };
        }
    };
    let l = match http::chat_stream(handle.target(), &long, timeout).await {
        Ok(o) => o,
        Err(e) => {
            return Cell::Error {
                id: ID,
                msg: crate::benchmarks::one_line(format!("{e:#}")),
            };
        }
    };
    let short_ok = want(s.text.trim());
    let long_ok = want(l.text.trim());
    if short_ok && long_ok {
        Cell::Pass {
            id: ID,
            detail: format!(
                "same answer on both paths ({} vs {} prompt tokens)",
                s.prompt_tokens, l.prompt_tokens
            ),
        }
    } else {
        Cell::Fail {
            id: ID,
            detail: format!(
                "short path {}, long path {} ({} vs {} prompt tokens) — the two splices disagree",
                if short_ok { "ok" } else { "WRONG" },
                if long_ok { "ok" } else { "WRONG" },
                s.prompt_tokens,
                l.prompt_tokens
            ),
        }
    }
}

/// ★ MEDIA THAT IS NOT IN THE MESSAGE ASKING THE QUESTION.
///
/// Every other leg puts the image and the question in one user turn. Real
/// traffic does not: an agent sees a screenshot in a TOOL RESULT and is asked
/// about it two turns later. `collect_message_images` walks every message and
/// every role precisely so that works (it is the motivating case for #165), and
/// a regression that only scanned the final message would pass every other leg
/// in both benchmarks.
pub async fn media_in_history(
    handle: &PluginHandle,
    body: Value,
    want: &(dyn Fn(&str) -> bool + Sync),
    id: &'static str,
    timeout: Duration,
) -> Cell {
    match http::chat_stream(handle.target(), &body, timeout).await {
        Ok(o) => {
            let text = o.text.trim();
            if want(text) {
                Cell::Pass {
                    id,
                    detail: format!("answered from history ({} prompt tokens)", o.prompt_tokens),
                }
            } else {
                Cell::Fail {
                    id,
                    detail: format!(
                        "did not answer about the earlier media: \"{}\"",
                        crate::benchmarks::one_line(text.chars().take(80).collect::<String>())
                    ),
                }
            }
        }
        Err(e) => Cell::Error {
            id,
            msg: crate::benchmarks::one_line(format!("{e:#}")),
        },
    }
}

/// ★ THE SAME REQUEST STREAMED AND NOT STREAMED.
///
/// Every other leg in both benchmarks sets `stream: true`, so the blocking
/// response path had never been exercised by any of them. It is not a thin
/// wrapper around the streaming one — it assembles the response itself — and a
/// server can stream correctly while assembling a blocking reply wrongly.
///
/// Both must answer correctly AND agree on `prompt_tokens`: the prompt is
/// byte-identical, so a difference there means the two paths built different
/// prompts from the same request, which is a preprocessing divergence rather
/// than a sampling one.
pub async fn stream_parity(
    handle: &PluginHandle,
    mut body: Value,
    want: &(dyn Fn(&str) -> bool + Sync),
    timeout: Duration,
) -> Cell {
    const ID: &str = "stream-blocking-parity";
    body["stream"] = Value::Bool(true);
    let streamed = match http::chat_stream(handle.target(), &body, timeout).await {
        Ok(o) => o,
        Err(e) => {
            return Cell::Error {
                id: ID,
                msg: crate::benchmarks::one_line(format!("streaming: {e:#}")),
            };
        }
    };
    body["stream"] = Value::Bool(false);
    let blocking = match http::chat_blocking(handle.target(), &body, timeout).await {
        Ok(o) => o,
        Err(e) => {
            return Cell::Error {
                id: ID,
                msg: crate::benchmarks::one_line(format!("blocking: {e:#}")),
            };
        }
    };
    let b_text = blocking.choices.first().cloned().unwrap_or_default();
    let s_ok = want(streamed.text.trim());
    let b_ok = want(b_text.trim());
    if !s_ok || !b_ok {
        return Cell::Fail {
            id: ID,
            detail: format!(
                "streaming {}, blocking {} — the two response paths disagree about the same \
                 image",
                if s_ok { "ok" } else { "WRONG" },
                if b_ok { "ok" } else { "WRONG" }
            ),
        };
    }
    if streamed.prompt_tokens != blocking.prompt_tokens {
        return Cell::Fail {
            id: ID,
            detail: format!(
                "both answered correctly but built DIFFERENT prompts: {} tokens streaming vs {} \
                 blocking, from a byte-identical request",
                streamed.prompt_tokens, blocking.prompt_tokens
            ),
        };
    }
    Cell::Pass {
        id: ID,
        detail: format!(
            "both paths correct and agree on {} prompt tokens",
            streamed.prompt_tokens
        ),
    }
}

/// ★ `n > 1` WITH AN IMAGE.
///
/// The blocking path decides PER CHOICE whether that choice carries the
/// request's image pixels — choice 0 takes them, the rest get an empty vector,
/// because the encode is shared rather than repeated. That is correct and it
/// is also index-conditional code that nothing ran. If the later choices lose
/// the image rather than sharing it, they answer about nothing at all.
///
/// Every choice must be present and must answer about the picture.
pub async fn multi_choice(
    handle: &PluginHandle,
    mut body: Value,
    n: usize,
    want: &(dyn Fn(&str) -> bool + Sync),
    timeout: Duration,
) -> Cell {
    const ID: &str = "multi-choice-image";
    body["stream"] = Value::Bool(false);
    body["n"] = Value::from(n);
    match http::chat_blocking(handle.target(), &body, timeout).await {
        Ok(o) => {
            if o.choices.len() != n {
                return Cell::Fail {
                    id: ID,
                    detail: format!("asked for n={n}, got {} choices", o.choices.len()),
                };
            }
            let bad: Vec<usize> = o
                .choices
                .iter()
                .enumerate()
                .filter(|(_, c)| !want(c.trim()))
                .map(|(i, _)| i)
                .collect();
            if bad.is_empty() {
                Cell::Pass {
                    id: ID,
                    detail: format!("all {n} choices answered about the image"),
                }
            } else {
                Cell::Fail {
                    id: ID,
                    detail: format!(
                        "choice(s) {bad:?} did not answer about the image — the later choices \
                         are not seeing it"
                    ),
                }
            }
        }
        // `n > 1` may simply not be supported; that is a capability, not a
        // defect, so it is a skip rather than a failure.
        Err(e) => {
            let msg = crate::benchmarks::one_line(format!("{e:#}"));
            let unsupported = msg.contains("400") || msg.to_lowercase().contains("not supported");
            if unsupported {
                Cell::Skipped { id: ID, why: msg }
            } else {
                Cell::Error { id: ID, msg }
            }
        }
    }
}

/// A Responses-API request carrying one image.
pub fn responses_image_request(model: &str, mime: &str, bytes: &[u8], prompt: &str) -> Value {
    use base64::Engine;
    let mut uri = format!("data:{mime};base64,");
    base64::engine::general_purpose::STANDARD.encode_string(bytes, &mut uri);
    json!({
        "model": model,
        "stream": false,
        "temperature": 0.0,
        // `reasoning.effort` is the OPENAI-STANDARD control for this surface,
        // and Atlas honors it (responses_lowering passes `reasoning` through
        // and `client_reasoning_effort` reads it). `chat_template_kwargs` —
        // the vLLM extension the chat-completions legs use — is explicitly
        // dropped by that lowering, so the standard field is the right lever
        // here and sending the extension would test a path that does not
        // exist.
        //
        // "low", not "none": `"none"` maps to *unspecified* rather than off,
        // so it falls back to the model's own default. The budget stays
        // generous regardless — on a thinking-first checkpoint the reply can
        // arrive as REASONING with an empty `output_text`, and a small budget
        // truncates it mid-thought, which reads as "this surface cannot see
        // images" and is nothing of the sort.
        "max_output_tokens": 600,
        "reasoning": {"effort": "low"},
        "input": [{"role": "user", "content": [
            {"type": "input_image", "image_url": {"url": uri}},
            {"type": "input_text", "text": prompt},
        ]}],
    })
}

/// ★ THE RESPONSES API SEES THE IMAGE, AND SIZES IT THE SAME.
///
/// `/v1/responses` has its own content vocabulary (`input_image`,
/// `input_text`) and its own adapter into the IR — a genuinely separate parse
/// path that nothing else in either benchmark drives.
///
/// WHY THIS IS A DIFFERENCE AND NOT A COMPARISON WITH CHAT-COMPLETIONS. The
/// obvious leg — same image both surfaces, assert equal prompt sizes — is
/// WRONG, and measured wrong: it varies two things at once. The
/// chat-completions legs send `chat_template_kwargs.enable_thinking = false`,
/// which the Responses lowering drops, so that surface renders the template's
/// THINKING branch instead. On qwen3.6 the two branches happen to differ by
/// 2 tokens and the comparison looked fine; on qwen3.8 they differ by 28 and
/// it reported a vision defect that was really a template branch. Comparing
/// across surfaces cannot separate "the image is sized differently" from "the
/// envelope is different".
///
/// So the assertion stays INSIDE one surface: two images whose vision-token
/// counts differ by a known amount, both through Responses. Whatever envelope
/// that surface renders, it renders the same one twice, so it cancels:
///
/// ```text
///   (tokens_b - tokens_a)  ==  (vision_b - vision_a)
/// ```
///
/// Both must also answer correctly, which is what proves the pixels arrived
/// at all rather than merely being counted.
pub async fn responses_parity(
    handle: &PluginHandle,
    smaller: Value,
    larger: Value,
    expect_delta: usize,
    want: &(dyn Fn(&str) -> bool + Sync),
    timeout: Duration,
) -> Cell {
    const ID: &str = "responses-api-parity";
    let a = match http::responses_blocking(handle.target(), &smaller, timeout).await {
        Ok(o) => o,
        Err(e) => {
            let msg = crate::benchmarks::one_line(format!("{e:#}"));
            return if msg.contains("404") {
                Cell::Skipped { id: ID, why: msg }
            } else {
                Cell::Error { id: ID, msg }
            };
        }
    };
    let b = match http::responses_blocking(handle.target(), &larger, timeout).await {
        Ok(o) => o,
        Err(e) => {
            return Cell::Error {
                id: ID,
                msg: crate::benchmarks::one_line(format!("{e:#}")),
            };
        }
    };
    let delta = b.prompt_tokens.abs_diff(a.prompt_tokens);
    if delta != expect_delta {
        return Cell::Fail {
            id: ID,
            detail: format!(
                "via /v1/responses the two images differ by {delta} prompt tokens, expected \
                 {expect_delta} ({} and {}) — the image is sized differently on this surface",
                a.prompt_tokens, b.prompt_tokens
            ),
        };
    }
    let a_text = a.choices.first().cloned().unwrap_or_default();
    let b_text = b.choices.first().cloned().unwrap_or_default();
    if !want(a_text.trim()) || !want(b_text.trim()) {
        return Cell::Fail {
            id: ID,
            detail: format!(
                "sizing is right but the surface did not answer about the image: \"{}\"",
                crate::benchmarks::one_line(a_text.chars().take(70).collect::<String>())
            ),
        };
    }
    Cell::Pass {
        id: ID,
        detail: format!(
            "both images answered, and the image contributes exactly {expect_delta} tokens \
             ({} vs {})",
            a.prompt_tokens, b.prompt_tokens
        ),
    }
}

/// The same image request with thinking ON.
pub fn thinking_image_request(model: &str, mime: &str, bytes: &[u8], prompt: &str) -> Value {
    let mut v = image_request(model, mime, bytes, prompt, 600);
    v["chat_template_kwargs"] = json!({"enable_thinking": true});
    // Generous on purpose: a thinking-first checkpoint spends most of the
    // budget reasoning, and a truncated reply reads as a vision failure when
    // it is only a budget one.
    v["max_tokens"] = json!(600);
    v
}

/// ★ VISION WITH THINKING ON — the configuration these checkpoints ship in.
///
/// Every other leg disables thinking, deliberately: a reasoning block spends
/// the token budget without changing whether the image was encoded correctly,
/// and on a thinking-first model it can consume the whole of `max_tokens` and
/// return empty content, which reads as a vision failure and is not one. But
/// that leaves the DEFAULT configuration untested, and thinking is not inert
/// here — it lengthens the prompt, which changes which prefill chunk the pad
/// run lands in. That is the same boundary `long_prompt_path` guards,
/// approached from the direction real traffic comes from.
///
/// THE ASSERTION IS A DIFFERENCE, so the unknown thinking overhead cancels.
/// Two images whose vision-token counts differ by a known amount are sent,
/// both with thinking on; the gap between their prompt sizes must equal that
/// amount exactly:
///
/// ```text
///   (tokens_b - tokens_a)  ==  (vision_b - vision_a)
/// ```
///
/// Whatever the `<think>` block costs, it costs the same in both, so it
/// vanishes from the subtraction. A thinking-on path that mis-sized, dropped
/// or double-counted the image moves this; a template change does not.
pub async fn thinking_parity(
    handle: &PluginHandle,
    smaller: Value,
    larger: Value,
    expect_delta: usize,
    want: &(dyn Fn(&str) -> bool + Sync),
    timeout: Duration,
) -> Cell {
    const ID: &str = "thinking-on-vision";
    let a = match http::chat_stream(handle.target(), &smaller, timeout).await {
        Ok(o) => o,
        Err(e) => {
            return Cell::Error {
                id: ID,
                msg: crate::benchmarks::one_line(format!("{e:#}")),
            };
        }
    };
    let b = match http::chat_stream(handle.target(), &larger, timeout).await {
        Ok(o) => o,
        Err(e) => {
            return Cell::Error {
                id: ID,
                msg: crate::benchmarks::one_line(format!("{e:#}")),
            };
        }
    };
    let delta = b.prompt_tokens.abs_diff(a.prompt_tokens);
    if delta != expect_delta {
        return Cell::Fail {
            id: ID,
            detail: format!(
                "with thinking ON the two images differ by {delta} prompt tokens, expected \
                 {expect_delta} ({} and {} total) — the image's contribution changes when \
                 thinking is enabled",
                a.prompt_tokens, b.prompt_tokens
            ),
        };
    }
    // The model must still ANSWER. Thinking-first checkpoints put the answer
    // after the block, so an empty reply here means the budget ran out before
    // it got there — reported as its own thing rather than as a vision fault.
    let answered = want(a.text.trim()) && want(b.text.trim());
    if !answered {
        let empty = a.text.trim().is_empty() || b.text.trim().is_empty();
        return Cell::Fail {
            id: ID,
            detail: if empty {
                "geometry is right but a reply came back EMPTY with thinking on — the \
                 reasoning block consumed the whole token budget"
                    .to_string()
            } else {
                format!(
                    "geometry is right but the answer is wrong with thinking on: \"{}\"",
                    crate::benchmarks::one_line(a.text.chars().take(60).collect::<String>())
                )
            },
        };
    }
    Cell::Pass {
        id: ID,
        detail: format!(
            "thinking on: both answered, and the image still contributes exactly \
             {expect_delta} tokens ({} vs {})",
            a.prompt_tokens, b.prompt_tokens
        ),
    }
}

#[cfg(test)]
#[path = "media_integrity_tests.rs"]
mod media_integrity_tests;
