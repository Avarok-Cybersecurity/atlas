// SPDX-License-Identifier: AGPL-3.0-only

//! Turning the scheduler's token events into `/completion` content.
//!
//! One [`Generation`] consumes tokens (with the per-token logprobs the
//! scheduler returns when `n_probs` is set), detokenizes them incrementally,
//! runs the stop-string matcher, and keeps what a response needs: the released
//! text, the kept tokens, and the `completion_probabilities` entries. It is
//! pure over a [`TextCodec`], so every stop and probability rule is unit-tested
//! without a GPU.

use std::sync::Arc;

use serde_json::{Value, json};

use super::codec::TextCodec;
use super::stop::StopMatcher;
use crate::api::TokenLogprobs;

/// What one token contributed.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Step {
    /// Text that became releasable with this token (may be empty: a
    /// half-character, or text held back as a possible stop string).
    pub delta: String,
    /// The token's `completion_probabilities` entry, when `n_probs` > 0.
    pub probs: Option<Value>,
    /// True on the token whose text completed a stop string.
    pub stopped: bool,
}

/// Why generation ended, in llama.cpp's terms.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum StopType {
    Eos,
    Limit,
    Word,
}

impl StopType {
    pub(crate) fn as_str(self) -> &'static str {
        match self {
            StopType::Eos => "eos",
            StopType::Limit => "limit",
            StopType::Word => "word",
        }
    }

    /// Map the scheduler's `finish_reason` (`stop`, `length`, `timeout`) to a
    /// llama.cpp stop type. A matched stop string is `Word` regardless.
    pub(crate) fn from_finish_reason(finish_reason: &str, word_hit: bool) -> Self {
        if word_hit {
            StopType::Word
        } else if finish_reason == "stop" {
            StopType::Eos
        } else {
            // "length", and "timeout" (llama.cpp has no separate timeout type;
            // its own time limit also reports `limit`).
            StopType::Limit
        }
    }
}

pub(crate) struct Generation {
    codec: Arc<dyn TextCodec>,
    stop: StopMatcher,
    n_probs: u8,
    tokens: Vec<u32>,
    /// Decoded byte length after each token: where its text ends.
    decoded_end: Vec<usize>,
    probs: Vec<Value>,
    decoded_len: usize,
    prefix: usize,
    read: usize,
    /// Everything released so far (the response `content`).
    content: String,
}

impl Generation {
    pub(crate) fn new(codec: Arc<dyn TextCodec>, stops: &[String], n_probs: u8) -> Self {
        Self {
            codec,
            stop: StopMatcher::new(stops),
            n_probs,
            tokens: Vec::new(),
            decoded_end: Vec::new(),
            probs: Vec::new(),
            decoded_len: 0,
            prefix: 0,
            read: 0,
            content: String::new(),
        }
    }

    /// Feed one generated token.
    pub(crate) fn push(&mut self, token: u32, lp: Option<&TokenLogprobs>) -> Step {
        self.tokens.push(token);
        let decoded = self
            .codec
            .decode_incremental(&self.tokens, &mut self.prefix, &mut self.read);
        self.decoded_len += decoded.len();
        self.decoded_end.push(self.decoded_len);
        let probs = (self.n_probs > 0).then(|| prob_entry(self.codec.as_ref(), token, lp));
        if let Some(p) = &probs {
            self.probs.push(p.clone());
        }
        let delta = self.stop.push(&decoded);
        self.content.push_str(&delta);
        Step {
            delta,
            probs,
            stopped: self.stop.hit().is_some(),
        }
    }

    /// End of stream: release text still held as a possible stop string.
    /// Returns it, or an empty string when a stop string matched.
    pub(crate) fn finish(&mut self) -> String {
        let tail = self.stop.finish();
        self.content.push_str(&tail);
        tail
    }

    pub(crate) fn stop_hit(&self) -> Option<&str> {
        self.stop.hit()
    }

    /// The released text so far.
    pub(crate) fn content(&self) -> &str {
        &self.content
    }

    /// Number of tokens fed.
    pub(crate) fn token_count(&self) -> usize {
        self.tokens.len()
    }

    /// How many leading tokens belong to the response. Without a stop match,
    /// all of them. With one, the tokens whose text starts before the cut:
    /// the tokens that spelled the stop string, and anything generated after
    /// it, are dropped (llama.cpp trims the probabilities the same way).
    fn kept(&self) -> usize {
        if self.stop.hit().is_none() {
            return self.tokens.len();
        }
        let cut = self.content.len();
        (0..self.tokens.len())
            .take_while(|&i| {
                let start = if i == 0 { 0 } else { self.decoded_end[i - 1] };
                start < cut
            })
            .count()
    }

    /// The kept token ids (for `tokens`).
    pub(crate) fn kept_tokens(&self) -> Vec<u32> {
        self.tokens[..self.kept()].to_vec()
    }

    /// The kept `completion_probabilities` entries; empty when `n_probs` is 0.
    pub(crate) fn kept_probs(&self) -> Vec<Value> {
        let n = self.kept().min(self.probs.len());
        self.probs[..n].to_vec()
    }

    pub(crate) fn has_new_line(&self) -> bool {
        self.content.contains('\n')
    }
}

/// The longest prefix of `bytes` that is valid UTF-8 (llama.cpp's
/// `validate_utf8`: a trailing half-character is cut from the string, while
/// `bytes` still lists every byte).
fn valid_prefix(bytes: &[u8]) -> String {
    match std::str::from_utf8(bytes) {
        Ok(s) => s.to_string(),
        Err(e) => String::from_utf8_lossy(&bytes[..e.valid_up_to()]).into_owned(),
    }
}

/// llama.cpp writes `-inf` as the most negative float, because JSON has no
/// infinity.
fn finite_logprob(lp: f32) -> f64 {
    if lp.is_finite() {
        f64::from(lp)
    } else {
        f64::from(f32::MIN)
    }
}

fn candidate(codec: &dyn TextCodec, id: u32, logprob: Option<f32>) -> Value {
    let bytes = codec.token_bytes(id);
    json!({
        "id": id,
        "token": valid_prefix(&bytes),
        "bytes": bytes,
        "logprob": logprob.map(finite_logprob),
    })
}

/// One `completion_probabilities` entry:
/// `{id, token, bytes, logprob, top_logprobs: [{id, token, bytes, logprob}]}`.
///
/// When the scheduler returned no logprobs for this token the entry keeps its
/// slot, so entries stay aligned with `tokens`, and says so with a `null`
/// `logprob` and an empty `top_logprobs`.
pub(crate) fn prob_entry(codec: &dyn TextCodec, token: u32, lp: Option<&TokenLogprobs>) -> Value {
    let mut entry = candidate(codec, token, lp.map(|l| l.logprob));
    let top: Vec<Value> = lp
        .map(|l| {
            l.top
                .iter()
                .map(|&(id, p)| candidate(codec, id, Some(p)))
                .collect()
        })
        .unwrap_or_default();
    entry["top_logprobs"] = Value::Array(top);
    entry
}

#[cfg(test)]
mod tests {
    use super::super::toy_codec::ToyCodec;
    use super::*;

    fn lp(token: u32, logprob: f32, top: &[(u32, f32)]) -> TokenLogprobs {
        TokenLogprobs {
            token_id: token,
            logprob,
            top: top.to_vec(),
        }
    }

    fn feed(g: &mut Generation, text: &str) -> Vec<Step> {
        ToyCodec::ids(text)
            .into_iter()
            .map(|t| g.push(t, None))
            .collect()
    }

    #[test]
    fn text_streams_token_by_token_without_stops() {
        let mut g = Generation::new(Arc::new(ToyCodec), &[], 0);
        let steps = feed(&mut g, "hi!");
        let deltas: Vec<_> = steps.iter().map(|s| s.delta.as_str()).collect();
        assert_eq!(deltas, ["h", "i", "!"]);
        assert_eq!(g.content(), "hi!");
        assert_eq!(g.kept_tokens().len(), 3);
        assert!(
            g.kept_probs().is_empty(),
            "n_probs=0 returns no probabilities"
        );
        assert!(steps.iter().all(|s| s.probs.is_none() && !s.stopped));
    }

    #[test]
    fn a_half_character_token_releases_nothing_until_it_completes() {
        let mut g = Generation::new(Arc::new(ToyCodec), &[], 0);
        let steps = feed(&mut g, "\u{20AC}");
        let deltas: Vec<_> = steps.iter().map(|s| s.delta.as_str()).collect();
        assert_eq!(deltas, ["", "", "\u{20AC}"]);
    }

    #[test]
    fn a_stop_string_cuts_content_and_trims_tokens() {
        let mut g = Generation::new(Arc::new(ToyCodec), &["END".to_string()], 0);
        let steps = feed(&mut g, "ab END cd");
        // 'a','b',' ' are released; 'E','N','D' complete the stop on the 'D' token.
        let stopped_at = steps.iter().position(|s| s.stopped).expect("stop");
        assert_eq!(stopped_at, 5, "the D of END");
        assert_eq!(g.stop_hit(), Some("END"));
        assert_eq!(g.content(), "ab ");
        // Kept: a, b, space. The tokens E, N, D spell the stop word and are cut.
        assert_eq!(g.kept_tokens(), ToyCodec::ids("ab "));
        // After the stop, later tokens change nothing.
        assert_eq!(g.content(), "ab ");
    }

    #[test]
    fn without_a_stop_the_held_tail_is_flushed_and_all_tokens_kept() {
        let mut g = Generation::new(Arc::new(ToyCodec), &["END".to_string()], 0);
        feed(&mut g, "xyEN");
        assert_eq!(g.content(), "xy", "EN could still become END");
        assert_eq!(g.finish(), "EN");
        assert_eq!(g.content(), "xyEN");
        assert_eq!(g.kept_tokens().len(), 4);
        assert_eq!(g.stop_hit(), None);
    }

    #[test]
    fn n_probs_gives_one_entry_per_token_in_llama_cpp_shape() {
        let mut g = Generation::new(Arc::new(ToyCodec), &[], 2);
        let a = 100 + u32::from(b'a');
        let b = 100 + u32::from(b'b');
        let s = g.push(a, Some(&lp(a, -0.25, &[(a, -0.25), (b, -1.5)])));
        let e = s.probs.expect("entry");
        assert_eq!(e["id"], a);
        assert_eq!(e["token"], "a");
        assert_eq!(e["bytes"], json!([97]));
        assert_eq!(e["logprob"], -0.25);
        let top = e["top_logprobs"].as_array().unwrap();
        assert_eq!(top.len(), 2);
        assert_eq!(
            top[1],
            json!({"id": b, "token": "b", "bytes": [98], "logprob": -1.5})
        );
        assert_eq!(g.kept_probs().len(), 1);
    }

    #[test]
    fn a_token_with_no_logprobs_keeps_its_slot() {
        let mut g = Generation::new(Arc::new(ToyCodec), &[], 3);
        let a = 100 + u32::from(b'a');
        let first = g.push(a, None).probs.expect("entry even without data");
        assert_eq!(first["logprob"], Value::Null);
        assert_eq!(first["top_logprobs"], json!([]));
        assert_eq!(first["id"], a, "alignment with tokens is preserved");
    }

    #[test]
    fn probabilities_of_stop_word_tokens_are_trimmed_with_the_tokens() {
        let mut g = Generation::new(Arc::new(ToyCodec), &["Z".to_string()], 1);
        feed(&mut g, "aZb");
        assert_eq!(g.content(), "a");
        assert_eq!(g.kept_tokens().len(), 1);
        assert_eq!(g.kept_probs().len(), 1);
    }

    #[test]
    fn half_character_pieces_report_bytes_but_a_valid_prefix_string() {
        // The euro sign's first byte alone is not a character: `token` is the
        // empty valid prefix, `bytes` still carries the real byte.
        let lead = 100 + 0xE2;
        let e = prob_entry(&ToyCodec, lead, Some(&lp(lead, -2.0, &[])));
        assert_eq!(e["token"], "");
        assert_eq!(e["bytes"], json!([0xE2]));
    }

    #[test]
    fn minus_infinity_is_written_as_the_lowest_float() {
        let a = 100 + u32::from(b'a');
        let e = prob_entry(&ToyCodec, a, Some(&lp(a, f32::NEG_INFINITY, &[])));
        assert_eq!(e["logprob"].as_f64().unwrap(), f64::from(f32::MIN));
    }

    #[test]
    fn stop_type_follows_the_finish_reason_and_a_word_hit_wins() {
        assert_eq!(StopType::from_finish_reason("stop", false), StopType::Eos);
        assert_eq!(
            StopType::from_finish_reason("length", false),
            StopType::Limit
        );
        assert_eq!(
            StopType::from_finish_reason("timeout", false),
            StopType::Limit
        );
        assert_eq!(StopType::from_finish_reason("stop", true), StopType::Word);
        assert_eq!(StopType::from_finish_reason("length", true), StopType::Word);
        assert_eq!(StopType::Word.as_str(), "word");
    }

    #[test]
    fn new_line_tracking_matches_released_text() {
        let mut g = Generation::new(Arc::new(ToyCodec), &[], 0);
        feed(&mut g, "a\nb");
        assert!(g.has_new_line());
    }
}
