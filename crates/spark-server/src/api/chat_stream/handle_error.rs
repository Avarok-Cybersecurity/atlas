// SPDX-License-Identifier: AGPL-3.0-only
//
// `StreamEvent::Error(msg)` arm of the streaming `flat_map` closure.

use axum::response::sse::Event;

use super::ctx::StreamCtx;

type SseVec = Vec<Result<Event, std::convert::Infallible>>;

pub(super) fn handle_error(ctx: &StreamCtx, msg: String) -> SseVec {
    crate::metrics::REQUESTS_ACTIVE.dec();
    // Abandoned stream — refund the full reservation.
    if let Some(ref rctx) = ctx.req_ctx {
        ctx.state
            .rate_limiter
            .refund_tokens(&rctx.identity, rctx.reserved_tokens);
    }
    let err = serde_json::json!({
        "error": {"message": msg, "type": "server_error", "code": 500}
    });
    let events: SseVec = vec![Ok(Event::default().data(err.to_string()))];
    events
}
