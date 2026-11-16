// SPDX-License-Identifier: AGPL-3.0-only

//! The final `/completion` body, assembled from what the scheduler reported
//! (`Done`) and what the [`Generation`] kept.

use serde_json::Value;

use super::codec::TextCodec;
use super::completion_json::{FinalBody, Timings, final_json};
use super::generation::{Generation, StopType};
use super::params::CompletionSpec;

/// The numbers from the scheduler's `StreamEvent::Done`.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct DoneInfo {
    pub finish_reason: String,
    pub completion_tokens: usize,
    pub ttft_ms: f64,
    pub decode_ms: f64,
    pub cached_prompt_tokens: usize,
}

/// The final response body (non-stream), or the closing chunk of a stream.
pub(crate) fn final_value(
    codec: &dyn TextCodec,
    model: &str,
    n_ctx: usize,
    spec: &CompletionSpec,
    generation: &Generation,
    done: &DoneInfo,
) -> Value {
    let word = generation.stop_hit();
    let prompt_len = spec.prompt_tokens.len();
    let cached = done.cached_prompt_tokens.min(prompt_len);
    final_json(&FinalBody {
        model,
        stream: spec.stream,
        return_tokens: spec.return_tokens,
        content: generation.content(),
        tokens: generation.kept_tokens(),
        probs: generation.kept_probs(),
        stop_type: StopType::from_finish_reason(&done.finish_reason, word.is_some()),
        stopping_word: word.unwrap_or(""),
        tokens_predicted: done.completion_tokens,
        tokens_evaluated: prompt_len,
        tokens_cached: cached,
        prompt: codec
            .decode_text(&spec.prompt_tokens, false)
            .unwrap_or_default(),
        has_new_line: generation.has_new_line(),
        timings: Timings {
            cached,
            prompt_n: prompt_len - cached,
            prompt_ms: done.ttft_ms,
            predicted_n: done.completion_tokens,
            predicted_ms: done.decode_ms,
        },
        generation_settings: spec.generation_settings(model, n_ctx),
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use serde_json::json;

    use super::super::params::{ParseContext, parse_completion};
    use super::super::props::SamplingDefaults;
    use super::super::toy_codec::ToyCodec;
    use super::*;

    fn spec(body: Value) -> CompletionSpec {
        let ctx = ParseContext {
            codec: &ToyCodec,
            defaults: SamplingDefaults {
                temperature: 0.7,
                top_k: 20,
                top_p: 0.8,
                min_p: 0.0,
                top_n_sigma: 0.0,
                repetition_penalty: 1.0,
            },
            max_seq_len: 64,
            grammar_available: true,
        };
        parse_completion(body.as_object().unwrap(), &ctx).expect("valid")
    }

    fn done(reason: &str) -> DoneInfo {
        DoneInfo {
            finish_reason: reason.into(),
            completion_tokens: 5,
            ttft_ms: 40.0,
            decode_ms: 100.0,
            cached_prompt_tokens: 1,
        }
    }

    fn run(spec: &CompletionSpec, text: &str, reason: &str) -> Value {
        let mut g = Generation::new(Arc::new(ToyCodec), &spec.stop, spec.n_probs);
        for t in ToyCodec::ids(text) {
            if g.stop_hit().is_none() {
                g.push(t, None);
            }
        }
        g.finish();
        final_value(&ToyCodec, "m", 64, spec, &g, &done(reason))
    }

    #[test]
    fn ending_on_eos() {
        let s = spec(json!({"prompt": "hi"}));
        let v = run(&s, "ok", "stop");
        assert_eq!(v["content"], "ok");
        assert_eq!(v["stop"], true);
        assert_eq!(v["stop_type"], "eos");
        assert_eq!(v["stopping_word"], "");
        assert_eq!(v["tokens_evaluated"], 3, "BOS + 2");
        assert_eq!(v["tokens_cached"], 1);
        assert_eq!(v["timings"]["prompt_n"], 2, "cache hits are not evaluated");
        assert_eq!(v["timings"]["cache_n"], 1);
        assert_eq!(v["tokens_predicted"], 5);
        assert_eq!(v["truncated"], false);
    }

    #[test]
    fn ending_on_the_token_limit() {
        let s = spec(json!({"prompt": "hi", "n_predict": 2}));
        assert_eq!(run(&s, "ab", "length")["stop_type"], "limit");
    }

    #[test]
    fn ending_on_a_stop_word_excludes_it_and_names_it() {
        let s = spec(json!({"prompt": "hi", "stop": ["END"], "return_tokens": true}));
        let v = run(&s, "ab END tail", "stop");
        assert_eq!(v["content"], "ab ");
        assert_eq!(v["stop_type"], "word");
        assert_eq!(v["stopping_word"], "END");
        assert_eq!(v["tokens"], json!(ToyCodec::ids("ab ")));
    }

    #[test]
    fn the_prompt_is_the_processed_prompt_with_special_tokens_written_out() {
        let s = spec(json!({"prompt": "a<|eot|>"}));
        let v = run(&s, "x", "stop");
        assert_eq!(v["prompt"], "a<|eot|>");
    }

    #[test]
    fn stream_mode_final_is_empty_of_content() {
        let s = spec(json!({"prompt": "hi", "stream": true}));
        let v = run(&s, "text", "stop");
        assert_eq!(v["content"], "");
        assert_eq!(v["stop"], true);
        assert!(v.get("completion_probabilities").is_none());
    }

    #[test]
    fn probabilities_ride_along_when_asked() {
        let s = spec(json!({"prompt": "hi", "n_probs": 2}));
        let v = run(&s, "ab", "stop");
        let probs = v["completion_probabilities"].as_array().expect("probs");
        assert_eq!(probs.len(), 2);
        assert_eq!(probs[0]["token"], "a");
        assert_eq!(
            probs[0]["logprob"],
            Value::Null,
            "no scheduler data in this test"
        );
    }

    #[test]
    fn a_timeout_is_reported_as_the_limit() {
        let s = spec(json!({"prompt": "hi"}));
        assert_eq!(run(&s, "a", "timeout")["stop_type"], "limit");
    }
}
