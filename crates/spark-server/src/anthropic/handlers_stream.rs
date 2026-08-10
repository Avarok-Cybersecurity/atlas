// SPDX-License-Identifier: AGPL-3.0-only

use axum::response::sse::{Event, KeepAlive};
use axum::response::{IntoResponse, Response, Sse};
use futures::StreamExt;

use super::translator::*;

/// Encode the pipeline's neutral delta stream as Anthropic's structured
/// SSE event stream, **per-delta**: each `ir::StreamDelta` produces zero
/// or more Anthropic events emitted to the client before the next delta
/// is processed. (This replaces the old wrapper that re-parsed the
/// serialized OpenAI SSE bytes chunk-by-chunk — one fewer serialize/
/// parse round-trip and one fewer protocol to drift against.)
///
/// `dump` (--dump seq + writer handle) captures the response in
/// Anthropic shape: the typed events are aggregated as they are sent
/// and written as one `stream: true` entry when the stream ends.
pub(super) fn anthropic_sse_from_deltas(
    deltas: crate::ir::DeltaStream,
    req_model: String,
    dump: Option<(u64, crate::request_dumper::DumpHandle)>,
) -> Response {
    // Match the chat_stream sizing — see chat_stream/mod.rs for rationale.
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, std::convert::Infallible>>(1024);

    tokio::spawn(async move {
        let mut translator = AnthropicTranslator::new(req_model);
        let mut pending: Vec<SseEvent> = Vec::new();
        let mut deltas = deltas;
        // --dump: aggregate the typed events (event name + JSON data)
        // so the dump entry records the exact Anthropic wire framing.
        let mut captured: Option<Vec<serde_json::Value>> = dump.as_ref().map(|_| Vec::new());

        // Drain `pending` into the channel, converting each typed
        // `SseEvent` to an axum wire event (and capturing it for the
        // --dump entry when enabled). Returns `false` when the receiver
        // has hung up — caller should abort.
        async fn flush(
            tx: &tokio::sync::mpsc::Sender<Result<Event, std::convert::Infallible>>,
            pending: &mut Vec<SseEvent>,
            captured: &mut Option<Vec<serde_json::Value>>,
        ) -> bool {
            for ev in pending.drain(..) {
                if let Some(events) = captured {
                    events.push(serde_json::json!({"event": ev.event, "data": ev.data}));
                }
                if tx.send(Ok(ev.to_axum_event())).await.is_err() {
                    return false;
                }
            }
            true
        }

        let mut aborted = false;
        while let Some(delta) = deltas.next().await {
            translator.on_delta(&delta, &mut pending);
            if !flush(&tx, &mut pending, &mut captured).await {
                aborted = true;
                break;
            }
        }

        if !aborted {
            // Ensure message_stop fires even if the stream ended without
            // a Finish delta.
            translator.finalize(&mut pending);
            if !flush(&tx, &mut pending, &mut captured).await {
                tracing::warn!("anthropic stream: final flush failed (receiver dropped)");
            }
        }

        // --dump: write the aggregated Anthropic event list under the
        // request's seq (partial when the client hung up mid-stream).
        if let (Some((seq, writer)), Some(events)) = (dump, captured) {
            writer.dump_response("/v1/messages", seq, &events, true);
        }
    });

    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}
