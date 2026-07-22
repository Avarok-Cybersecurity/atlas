// SPDX-License-Identifier: AGPL-3.0-only
//
// `StreamEvent::Error(msg)` arm of the streaming `flat_map` closure.

use crate::ir::StreamDelta;

use super::ctx::StreamCtx;

type DeltaVec = Vec<StreamDelta>;

pub(super) fn handle_error(ctx: &StreamCtx, msg: String) -> DeltaVec {
    crate::metrics::REQUESTS_ACTIVE.dec();
    // Abandoned stream — refund the full reservation.
    if let Some(ref rctx) = ctx.req_ctx {
        ctx.state
            .rate_limiter
            .refund_tokens(&rctx.identity, rctx.reserved_tokens);
    }
    // Wire-ready OpenAI error envelope; the encoder forwards it
    // verbatim as SSE data.
    let err = serde_json::json!({
        "error": {"message": msg, "type": "server_error", "code": 500}
    });
    vec![StreamDelta::Error {
        message: err.to_string(),
    }]
}
