// SPDX-License-Identifier: AGPL-3.0-only

use futures::StreamExt;

use super::{ENDED_WITHOUT_RESULT, terminated};
use crate::api::inference_types::StreamEvent;

fn done() -> StreamEvent {
    StreamEvent::Done {
        finish_reason: "stop".into(),
        prompt_tokens: 1,
        completion_tokens: 1,
        time_to_first_token_ms: 0.0,
        decode_time_ms: 0.0,
        reasoning_tokens: 0,
        cached_prompt_tokens: 0,
        accepted_prediction_tokens: 0,
        guard_stop: None,
    }
}

/// Collapse a stream to comparable labels.
async fn labels(events: Vec<StreamEvent>) -> Vec<String> {
    terminated(futures::stream::iter(events))
        .map(|e| match e {
            StreamEvent::Token(t) => format!("token {t}"),
            StreamEvent::Done { .. } => "done".into(),
            StreamEvent::Error(m) => format!("error {m}"),
            _ => "other".into(),
        })
        .collect()
        .await
}

/// THE G25 shape: tokens (or none), then the sender is dropped with no
/// terminal event. Unwrapped, the SSE body just stopped under HTTP 200.
#[tokio::test]
async fn a_stream_dropped_without_a_terminal_event_ends_in_an_error() {
    let silent = format!("error {ENDED_WITHOUT_RESULT}");
    assert_eq!(labels(vec![]).await, vec![silent.clone()]);
    assert_eq!(
        labels(vec![StreamEvent::Token(7)]).await,
        vec!["token 7".to_string(), silent]
    );
}

/// NEGATIVE CONTROLS: a stream the scheduler ended itself gets nothing added,
/// and whatever follows a terminal event is forwarded untouched.
#[tokio::test]
async fn a_terminated_stream_is_passed_through_unchanged() {
    assert_eq!(
        labels(vec![StreamEvent::Token(1), done()]).await,
        vec!["token 1", "done"]
    );
    assert_eq!(
        labels(vec![StreamEvent::Error("prefill_chunk failed: x".into())]).await,
        vec!["error prefill_chunk failed: x"]
    );
    assert_eq!(
        labels(vec![done(), StreamEvent::Token(2)]).await,
        vec!["done", "token 2"]
    );
}

/// Every streaming surface that reads the scheduler's channel must read it
/// through `terminated`. Structural, like `cancel_guard_tests`: the property
/// is "no call site forgot", which a per-surface behaviour test cannot catch
/// for the next surface.
#[test]
fn every_scheduler_stream_surface_reads_through_terminated() {
    const SURFACES: &[(&str, &str)] = &[
        ("chat_stream/mod.rs", include_str!("chat_stream/mod.rs")),
        ("completions.rs", include_str!("completions.rs")),
    ];
    for (name, src) in SURFACES {
        assert!(
            src.contains("terminated(ReceiverStream::new(token_rx))"),
            "{name} reads the scheduler channel without stream_terminal::terminated: \
             a dropped sender would end its SSE body in silence under HTTP 200"
        );
        assert!(
            !src.contains("ReceiverStream::new(token_rx).flat_map"),
            "{name} still maps the raw receiver"
        );
    }
}
