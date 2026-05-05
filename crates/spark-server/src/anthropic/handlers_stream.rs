// SPDX-License-Identifier: AGPL-3.0-only

#[allow(unused_imports)]
use std::sync::Arc;

#[allow(unused_imports)]
use axum::extract::State;
#[allow(unused_imports)]
use axum::extract::rejection::JsonRejection;
#[allow(unused_imports)]
use axum::http::StatusCode;
#[allow(unused_imports)]
use axum::response::sse::{Event, KeepAlive};
#[allow(unused_imports)]
use axum::response::{IntoResponse, Json, Response, Sse};
#[allow(unused_imports)]
use futures::StreamExt;
#[allow(unused_imports)]
use serde::{Deserialize, Serialize};

#[allow(unused_imports)]
use crate::AppState;
#[allow(unused_imports)]
use crate::openai;
#[allow(unused_imports)]
use crate::tool_parser;

#[allow(unused_imports)]
use super::convert::*;
#[allow(unused_imports)]
use super::helpers::*;
#[allow(unused_imports)]
use super::translate::*;
#[allow(unused_imports)]
use super::translator::*;
#[allow(unused_imports)]
use super::types::*;
#[allow(unused_imports)]
/// Translate an OpenAI SSE response into Anthropic's structured event
/// stream, **per-chunk**: each OpenAI `data: {…}` line that arrives
/// produces zero or more Anthropic events emitted to the client
/// before the next OpenAI chunk is processed.
///
/// Implementation: spawn a tokio task that consumes the inner body's
/// `Bytes` stream, splits on `\n\n` event boundaries, parses the
/// `data:` payload as JSON, and feeds it through
/// `AnthropicTranslator::process_openai_chunk`. The translator's
/// emitted events are sent down an `mpsc` channel that's wrapped as
/// a `ReceiverStream` and handed to axum's `Sse::new` — the same
/// pattern api.rs uses for its own SSE response.
pub(super) async fn wrap_chat_sse_for_anthropic(
    chat_resp: Response,
    req_model: String,
) -> Response {
    let (parts, body) = chat_resp.into_parts();
    if !parts.status.is_success() {
        // Forward error envelope verbatim, status preserved.
        return Response::from_parts(parts, body);
    }

    // Match the OpenAI chat_stream sizing — see chat_stream/mod.rs for rationale.
    let (tx, rx) = tokio::sync::mpsc::channel::<Result<Event, std::convert::Infallible>>(1024);

    tokio::spawn(async move {
        let mut translator = AnthropicTranslator::new(req_model);
        let mut buf = String::new();
        let mut data_stream = body.into_data_stream();
        let mut pending: Vec<Event> = Vec::new();

        // Inner helper: drain `pending` into the channel. Returns
        // `false` when the receiver has hung up — caller should
        // abort.
        async fn flush(
            tx: &tokio::sync::mpsc::Sender<Result<Event, std::convert::Infallible>>,
            pending: &mut Vec<Event>,
        ) -> bool {
            for ev in pending.drain(..) {
                if tx.send(Ok(ev)).await.is_err() {
                    return false;
                }
            }
            true
        }

        while let Some(chunk_res) = data_stream.next().await {
            let chunk = match chunk_res {
                Ok(b) => b,
                Err(e) => {
                    tracing::warn!("anthropic stream: inner body error: {e}");
                    break;
                }
            };
            buf.push_str(&String::from_utf8_lossy(&chunk));

            // Process any complete `data: {…}\n\n` records.
            while let Some(boundary) = buf.find("\n\n") {
                let raw_record: String = buf.drain(..boundary + 2).collect();
                for line in raw_record.lines() {
                    let payload = match line.strip_prefix("data:") {
                        Some(s) => s.trim(),
                        None => continue,
                    };
                    if payload.is_empty() || payload == "[DONE]" {
                        continue;
                    }
                    let val: serde_json::Value = match serde_json::from_str(payload) {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    translator.process_openai_chunk(&val, &mut pending);
                }
                if !flush(&tx, &mut pending).await {
                    return;
                }
            }
        }

        // Final flush: drain any tail bytes (no trailing `\n\n`) and
        // ensure message_stop fires even if upstream omitted
        // finish_reason.
        if !buf.is_empty() {
            for line in buf.lines() {
                let payload = match line.strip_prefix("data:") {
                    Some(s) => s.trim(),
                    None => continue,
                };
                if payload.is_empty() || payload == "[DONE]" {
                    continue;
                }
                if let Ok(val) = serde_json::from_str::<serde_json::Value>(payload) {
                    translator.process_openai_chunk(&val, &mut pending);
                }
            }
        }
        translator.finalize(&mut pending);
        if !flush(&tx, &mut pending).await {
            tracing::warn!("anthropic stream: final flush failed (receiver dropped)");
        }
    });

    let stream = tokio_stream::wrappers::ReceiverStream::new(rx);
    Sse::new(stream)
        .keep_alive(KeepAlive::default())
        .into_response()
}
