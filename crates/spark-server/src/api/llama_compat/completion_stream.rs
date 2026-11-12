// SPDX-License-Identifier: AGPL-3.0-only

//! The streaming half of `POST /completion`.
//!
//! llama.cpp streams its native route as bare SSE `data: {json}` frames, one
//! per generated token, and ends with a frame whose `stop` is `true`. There is
//! no `[DONE]` sentinel on this route (that is the OpenAI routes' convention).

use std::convert::Infallible;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use axum::response::sse::{Event, KeepAlive};
use axum::response::{IntoResponse, Response, Sse};
use futures::StreamExt;
use serde_json::Value;
use tokio::sync::mpsc;

use crate::AppState;
use crate::api::StreamEvent;

use super::codec::TextCodec;
use super::completion::{Started, StateCodec};
use super::completion_json::chunk_json;
use super::error::CompatError;
use super::finish::{DoneInfo, final_value};
use super::generation::Generation;
use super::params::CompletionSpec;

/// Turns scheduler events into `/completion` frames.
pub(crate) struct Pump {
    rx: mpsc::Receiver<StreamEvent>,
    generation: Generation,
    cancel: Arc<AtomicBool>,
    spec: CompletionSpec,
    codec: Arc<dyn TextCodec>,
    model: String,
    n_ctx: usize,
}

fn error_frame(message: &str) -> Value {
    CompatError::server(message).body()
}

impl Pump {
    pub(crate) fn new(
        started: Started,
        codec: Arc<dyn TextCodec>,
        model: String,
        n_ctx: usize,
    ) -> Self {
        let Started { spec, rx, cancel } = started;
        let generation = Generation::new(Arc::clone(&codec), &spec.stop, spec.n_probs);
        Self {
            rx,
            generation,
            cancel,
            spec,
            codec,
            model,
            n_ctx,
        }
    }

    /// Wait for the next frames. Returns them with `true` when the stream is
    /// over. A token that produces nothing to send (it only completed a stop
    /// string) is absorbed and the wait continues.
    pub(crate) async fn next_frames(&mut self) -> (Vec<Value>, bool) {
        loop {
            let Some(event) = self.rx.recv().await else {
                return (
                    vec![error_frame(
                        "Inference ended without a result (the request was cancelled)",
                    )],
                    true,
                );
            };
            let (token, lp) = match event {
                StreamEvent::Token(t) => (t, None),
                StreamEvent::TokenWithLogprobs(t, lp) => (t, Some(lp)),
                StreamEvent::PromptLogprobs(_) => continue,
                StreamEvent::Error(msg) => {
                    return (vec![error_frame(&format!("Inference error: {msg}"))], true);
                }
                StreamEvent::Done {
                    finish_reason,
                    completion_tokens,
                    time_to_first_token_ms,
                    decode_time_ms,
                    cached_prompt_tokens,
                    ..
                } => {
                    let mut frames = Vec::new();
                    if self.generation.stop_hit().is_none() {
                        let tail = self.generation.finish();
                        if !tail.is_empty() {
                            frames.push(self.chunk(&tail, None, None));
                        }
                    }
                    let done = DoneInfo {
                        finish_reason,
                        completion_tokens,
                        ttft_ms: time_to_first_token_ms,
                        decode_ms: decode_time_ms,
                        cached_prompt_tokens: cached_prompt_tokens as usize,
                    };
                    frames.push(final_value(
                        self.codec.as_ref(),
                        &self.model,
                        self.n_ctx,
                        &self.spec,
                        &self.generation,
                        &done,
                    ));
                    return (frames, true);
                }
            };
            // Tokens that arrive after a stop string, before the scheduler
            // sees the cancel flag, are dropped.
            if self.generation.stop_hit().is_some() {
                continue;
            }
            let step = self.generation.push(token, lp.as_ref());
            if step.stopped {
                self.cancel.store(true, Ordering::Release);
                if step.delta.is_empty() {
                    continue;
                }
            }
            return (
                vec![self.chunk(&step.delta, Some(token), step.probs)],
                false,
            );
        }
    }

    fn chunk(&self, delta: &str, token: Option<u32>, probs: Option<Value>) -> Value {
        chunk_json(
            delta,
            token,
            probs,
            self.generation.token_count(),
            self.spec.prompt_tokens.len(),
        )
    }
}

/// The SSE response for a queued request.
pub(crate) fn stream_response(state: Arc<AppState>, started: Started) -> Response {
    let codec: Arc<dyn TextCodec> = Arc::new(StateCodec(Arc::clone(&state)));
    let pump = Pump::new(started, codec, state.model_name.clone(), state.max_seq_len);
    let frames = futures::stream::unfold(Some(pump), |pump| async move {
        let mut pump = pump?;
        let (frames, finished) = pump.next_frames().await;
        Some((frames, if finished { None } else { Some(pump) }))
    });
    let events = frames.flat_map(|frames| {
        futures::stream::iter(
            frames
                .into_iter()
                .map(|v| Ok::<Event, Infallible>(Event::default().data(v.to_string()))),
        )
    });
    Sse::new(events)
        .keep_alive(KeepAlive::default())
        .into_response()
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::super::params::{ParseContext, parse_completion};
    use super::super::props::SamplingDefaults;
    use super::super::toy_codec::ToyCodec;
    use super::*;
    use crate::api::TokenLogprobs;

    fn pump(body: Value) -> (Pump, mpsc::Sender<StreamEvent>, Arc<AtomicBool>) {
        let codec = ToyCodec;
        let spec = parse_completion(
            body.as_object().unwrap(),
            &ParseContext {
                codec: &codec,
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
            },
        )
        .expect("valid");
        let (tx, rx) = mpsc::channel(64);
        let cancel = Arc::new(AtomicBool::new(false));
        let started = Started {
            spec,
            rx,
            cancel: Arc::clone(&cancel),
        };
        (
            Pump::new(started, Arc::new(ToyCodec), "m".into(), 64),
            tx,
            cancel,
        )
    }

    fn done(tx: &mpsc::Sender<StreamEvent>, n: usize, reason: &str) {
        tx.try_send(StreamEvent::Done {
            finish_reason: reason.into(),
            prompt_tokens: 3,
            completion_tokens: n,
            time_to_first_token_ms: 10.0,
            decode_time_ms: 20.0,
            reasoning_tokens: 0,
            cached_prompt_tokens: 0,
            accepted_prediction_tokens: 0,
            guard_stop: None,
        })
        .unwrap();
    }

    fn tok(tx: &mpsc::Sender<StreamEvent>, c: char) {
        tx.try_send(StreamEvent::Token(100 + c as u32)).unwrap();
    }

    async fn all(p: &mut Pump) -> Vec<Value> {
        let mut out = Vec::new();
        loop {
            let (frames, finished) = p.next_frames().await;
            out.extend(frames);
            if finished {
                return out;
            }
        }
    }

    #[tokio::test]
    async fn one_frame_per_token_then_a_closing_frame_with_stop_true() {
        let (mut p, tx, _c) = pump(json!({"prompt": "hi", "stream": true}));
        tok(&tx, 'o');
        tok(&tx, 'k');
        done(&tx, 2, "stop");
        let frames = all(&mut p).await;
        assert_eq!(frames.len(), 3);
        assert_eq!(frames[0]["content"], "o");
        assert_eq!(frames[0]["tokens"], json!([100 + 'o' as u32]));
        assert_eq!(frames[0]["stop"], false);
        assert_eq!(frames[1]["content"], "k");
        assert_eq!(frames[1]["tokens_predicted"], 2);
        let last = &frames[2];
        assert_eq!(last["stop"], true);
        assert_eq!(last["content"], "", "text already went out in the frames");
        assert_eq!(last["stop_type"], "eos");
        assert_eq!(last["tokens_evaluated"], 3);
    }

    #[tokio::test]
    async fn a_stop_string_cancels_the_request_and_never_leaks() {
        let (mut p, tx, cancel) = pump(json!({"prompt": "hi", "stream": true, "stop": ["END"]}));
        for c in "abEND more".chars() {
            tok(&tx, c);
        }
        done(&tx, 10, "stop");
        let frames = all(&mut p).await;
        assert!(
            cancel.load(Ordering::Acquire),
            "scheduler must be told to stop"
        );
        let text: String = frames
            .iter()
            .filter(|f| f["stop"] == false)
            .map(|f| f["content"].as_str().unwrap())
            .collect();
        assert_eq!(text, "ab");
        let last = frames.last().unwrap();
        assert_eq!(last["stop_type"], "word");
        assert_eq!(last["stopping_word"], "END");
    }

    #[tokio::test]
    async fn text_held_for_a_possible_stop_is_flushed_when_the_stream_ends() {
        let (mut p, tx, cancel) = pump(json!({"prompt": "hi", "stream": true, "stop": ["END"]}));
        for c in "xEN".chars() {
            tok(&tx, c);
        }
        done(&tx, 3, "length");
        let frames = all(&mut p).await;
        assert!(!cancel.load(Ordering::Acquire));
        let text: String = frames
            .iter()
            .filter(|f| f["stop"] == false)
            .map(|f| f["content"].as_str().unwrap())
            .collect();
        assert_eq!(text, "xEN");
        assert_eq!(frames.last().unwrap()["stop_type"], "limit");
    }

    #[tokio::test]
    async fn each_frame_carries_its_own_probabilities() {
        let (mut p, tx, _c) = pump(json!({"prompt": "hi", "stream": true, "n_probs": 1}));
        let a = 100 + u32::from(b'a');
        tx.try_send(StreamEvent::TokenWithLogprobs(
            a,
            TokenLogprobs {
                token_id: a,
                logprob: -0.5,
                top: vec![(a, -0.5)],
            },
        ))
        .unwrap();
        done(&tx, 1, "stop");
        let frames = all(&mut p).await;
        let probs = frames[0]["completion_probabilities"].as_array().unwrap();
        assert_eq!(probs.len(), 1);
        assert_eq!(probs[0]["logprob"], -0.5);
        assert_eq!(probs[0]["top_logprobs"][0]["token"], "a");
        assert!(frames[1].get("completion_probabilities").is_none());
    }

    #[tokio::test]
    async fn a_scheduler_error_is_an_error_frame() {
        let (mut p, tx, _c) = pump(json!({"prompt": "hi", "stream": true}));
        tx.try_send(StreamEvent::Error("boom".into())).unwrap();
        let frames = all(&mut p).await;
        assert_eq!(frames.len(), 1);
        assert_eq!(frames[0]["error"]["code"], 500);
        assert!(
            frames[0]["error"]["message"]
                .as_str()
                .unwrap()
                .contains("boom")
        );
    }

    #[tokio::test]
    async fn a_channel_that_closes_without_done_is_an_error_not_a_silent_end() {
        let (mut p, tx, _c) = pump(json!({"prompt": "hi", "stream": true}));
        tok(&tx, 'a');
        drop(tx);
        let frames = all(&mut p).await;
        assert_eq!(frames.len(), 2);
        assert!(frames[1].get("error").is_some());
    }
}
