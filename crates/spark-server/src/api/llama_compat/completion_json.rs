// SPDX-License-Identifier: AGPL-3.0-only

//! `/completion` response bodies, in the field order and naming of
//! `server_task_result_cmpl_final::to_json_non_oaicompat` and
//! `server_task_result_cmpl_partial::to_json_non_oaicompat` (llama.cpp b11583).

use serde_json::{Value, json};

use super::generation::StopType;

/// Timing facts the scheduler reports for a finished request.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct Timings {
    /// Prompt tokens served from the prefix cache (`cache_n`).
    pub cached: usize,
    /// Prompt tokens evaluated, cache hits excluded (`prompt_n`).
    pub prompt_n: usize,
    /// Time to the first token (`prompt_ms`).
    pub prompt_ms: f64,
    /// Generated tokens (`predicted_n`).
    pub predicted_n: usize,
    /// Time spent generating (`predicted_ms`).
    pub predicted_ms: f64,
}

fn per_token(ms: f64, n: usize) -> f64 {
    if n == 0 { 0.0 } else { ms / n as f64 }
}

fn per_second(ms: f64, n: usize) -> f64 {
    if ms <= 0.0 { 0.0 } else { 1e3 * n as f64 / ms }
}

/// The `timings` object.
pub(crate) fn timings_json(t: &Timings) -> Value {
    json!({
        "cache_n": t.cached,
        "prompt_n": t.prompt_n,
        "prompt_ms": t.prompt_ms,
        "prompt_per_token_ms": per_token(t.prompt_ms, t.prompt_n),
        "prompt_per_second": per_second(t.prompt_ms, t.prompt_n),
        "predicted_n": t.predicted_n,
        "predicted_ms": t.predicted_ms,
        "predicted_per_token_ms": per_token(t.predicted_ms, t.predicted_n),
        "predicted_per_second": per_second(t.predicted_ms, t.predicted_n),
    })
}

/// One streamed chunk: the text and token that just became available.
/// `stop` is false; the last chunk of a stream is built by [`final_json`].
pub(crate) fn chunk_json(
    delta: &str,
    token: Option<u32>,
    probs: Option<Value>,
    tokens_predicted: usize,
    tokens_evaluated: usize,
) -> Value {
    let mut v = json!({
        "index": 0,
        "content": delta,
        "tokens": token.map(|t| vec![t]).unwrap_or_default(),
        "stop": false,
        "id_slot": 0,
        "tokens_predicted": tokens_predicted,
        "tokens_evaluated": tokens_evaluated,
    });
    if let Some(p) = probs {
        v["completion_probabilities"] = Value::Array(vec![p]);
    }
    v
}

/// Everything the final body reports.
pub(crate) struct FinalBody<'a> {
    pub model: &'a str,
    pub stream: bool,
    pub return_tokens: bool,
    pub content: &'a str,
    pub tokens: Vec<u32>,
    pub probs: Vec<Value>,
    pub stop_type: StopType,
    pub stopping_word: &'a str,
    pub tokens_predicted: usize,
    pub tokens_evaluated: usize,
    pub tokens_cached: usize,
    pub prompt: String,
    pub has_new_line: bool,
    pub timings: Timings,
    pub generation_settings: Value,
}

/// The final body. In stream mode the text and tokens already went out in the
/// chunks, so `content` and `tokens` are empty and probabilities are omitted,
/// as in llama.cpp; otherwise `tokens` is filled only when `return_tokens` is
/// set.
pub(crate) fn final_json(b: &FinalBody<'_>) -> Value {
    let (content, tokens) = if b.stream {
        (String::new(), Vec::new())
    } else if b.return_tokens {
        (b.content.to_string(), b.tokens.clone())
    } else {
        (b.content.to_string(), Vec::new())
    };
    let mut v = json!({
        "index": 0,
        "content": content,
        "tokens": tokens,
        "id_slot": 0,
        "stop": true,
        "model": b.model,
        "tokens_predicted": b.tokens_predicted,
        "tokens_evaluated": b.tokens_evaluated,
        "generation_settings": b.generation_settings,
        "prompt": b.prompt,
        "has_new_line": b.has_new_line,
        "truncated": false,
        "stop_type": b.stop_type.as_str(),
        "stopping_word": b.stopping_word,
        "tokens_cached": b.tokens_cached,
        "timings": timings_json(&b.timings),
    });
    if !b.stream && !b.probs.is_empty() {
        v["completion_probabilities"] = Value::Array(b.probs.clone());
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    fn body<'a>(stream: bool, return_tokens: bool) -> FinalBody<'a> {
        FinalBody {
            model: "m",
            stream,
            return_tokens,
            content: "hello",
            tokens: vec![1, 2],
            probs: vec![json!({"id": 1})],
            stop_type: StopType::Limit,
            stopping_word: "",
            tokens_predicted: 2,
            tokens_evaluated: 7,
            tokens_cached: 4,
            prompt: "P".into(),
            has_new_line: false,
            timings: Timings {
                cached: 4,
                prompt_n: 3,
                prompt_ms: 30.0,
                predicted_n: 2,
                predicted_ms: 100.0,
            },
            generation_settings: json!({"n_ctx": 8}),
        }
    }

    #[test]
    fn non_stream_final_has_the_llama_cpp_fields() {
        let v = final_json(&body(false, false));
        for key in [
            "index",
            "content",
            "tokens",
            "id_slot",
            "stop",
            "model",
            "tokens_predicted",
            "tokens_evaluated",
            "generation_settings",
            "prompt",
            "has_new_line",
            "truncated",
            "stop_type",
            "stopping_word",
            "tokens_cached",
            "timings",
        ] {
            assert!(v.get(key).is_some(), "missing {key}");
        }
        assert_eq!(v["content"], "hello");
        assert_eq!(v["stop"], true);
        assert_eq!(v["stop_type"], "limit");
        assert_eq!(v["tokens_predicted"], 2);
        assert_eq!(v["tokens_evaluated"], 7);
        assert_eq!(v["tokens_cached"], 4);
        assert_eq!(v["completion_probabilities"], json!([{"id": 1}]));
    }

    #[test]
    fn tokens_only_when_return_tokens_is_set() {
        assert_eq!(final_json(&body(false, false))["tokens"], json!([]));
        assert_eq!(final_json(&body(false, true))["tokens"], json!([1, 2]));
    }

    #[test]
    fn stream_final_is_empty_of_text_tokens_and_probabilities() {
        let v = final_json(&body(true, true));
        assert_eq!(v["content"], "");
        assert_eq!(v["tokens"], json!([]));
        assert!(v.get("completion_probabilities").is_none());
        assert_eq!(v["stop"], true);
    }

    #[test]
    fn no_probabilities_key_when_none_were_asked() {
        let mut b = body(false, false);
        b.probs.clear();
        assert!(final_json(&b).get("completion_probabilities").is_none());
    }

    #[test]
    fn timings_derive_rates_from_counts_and_milliseconds() {
        let t = timings_json(&body(false, false).timings);
        assert_eq!(t["cache_n"], 4);
        assert_eq!(t["prompt_n"], 3);
        assert_eq!(t["prompt_per_token_ms"].as_f64().unwrap(), 10.0);
        assert_eq!(t["prompt_per_second"].as_f64().unwrap(), 100.0);
        assert_eq!(t["predicted_per_token_ms"].as_f64().unwrap(), 50.0);
        assert_eq!(t["predicted_per_second"].as_f64().unwrap(), 20.0);
    }

    #[test]
    fn timings_never_divide_by_zero() {
        let t = timings_json(&Timings {
            cached: 0,
            prompt_n: 0,
            prompt_ms: 0.0,
            predicted_n: 0,
            predicted_ms: 0.0,
        });
        for k in [
            "prompt_per_token_ms",
            "prompt_per_second",
            "predicted_per_token_ms",
            "predicted_per_second",
        ] {
            assert_eq!(t[k].as_f64().unwrap(), 0.0, "{k}");
        }
    }

    #[test]
    fn a_chunk_carries_one_token_its_text_and_its_probabilities() {
        let c = chunk_json("hi", Some(9), Some(json!({"id": 9})), 3, 12);
        assert_eq!(c["content"], "hi");
        assert_eq!(c["tokens"], json!([9]));
        assert_eq!(c["stop"], false);
        assert_eq!(c["tokens_predicted"], 3);
        assert_eq!(c["tokens_evaluated"], 12);
        assert_eq!(c["completion_probabilities"], json!([{"id": 9}]));
        let plain = chunk_json("", None, None, 1, 12);
        assert_eq!(plain["tokens"], json!([]));
        assert!(plain.get("completion_probabilities").is_none());
    }
}
