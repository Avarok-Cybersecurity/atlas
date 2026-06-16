// SPDX-License-Identifier: AGPL-3.0-only
//
// Content-chunk processing extracted from `handle_token`: the shared
// SimHash/token-loop watchdog + emit pipeline and the detector-active
// branch's `Content(text)` sanitize-then-process arm.

use axum::response::sse::Event;

use crate::openai::ChatCompletionChunk;

use super::super::super::sanitizer::sanitize_content_chunk;
use super::super::super::stream_guards::check_loop_watchdog;
use super::super::ctx::StreamCtx;
use super::super::state::StreamState;
use super::SseVec;

/// Common processing for a sanitized content chunk: SimHash semantic
/// guard, token-level loop watchdog, salvage on trip, otherwise
/// emit a `content_chunk`. Returns `Some(events)` when the watchdog
/// fired (caller must short-circuit), else `None` (caller continues).
///
/// Note: when called from the detector-active branch, `sanitized`
/// has already been routed through `sanitize_content_chunk`. When
/// called from the no-detector branch, the caller must pre-sanitize
/// (the no-detector path uses the same sanitizer state).
pub(super) fn process_detector_content(
    state: &mut StreamState,
    ctx: &StreamCtx,
    sanitized_or_raw: &str,
) -> Option<SseVec> {
    // From the detector-active branch the input is the Content(text)
    // payload that still needs sanitization. From the no-detector
    // branch the input is already sanitized. Distinguish via a thin
    // wrapper: detector branch ALSO sanitizes; non-detector branch
    // skips by passing the already-sanitized text. To keep the call
    // site simple, we sanitize here only when the input contains the
    // hallmark of an unfiltered Content payload — which we can't
    // reliably detect. Solution: split into two paths.
    //
    // Inlining: this helper is only called once per branch with the
    // correct input type; it never re-sanitizes. The parameter is the
    // post-sanitizer text in both call sites.
    let sanitized = sanitized_or_raw;

    // F4 SimHash guard.
    let semantic_trip = if !state.loop_watchdog_triggered {
        state.simhash_pending.push_str(sanitized);
        let mut dup = false;
        if crate::loop_simhash::ends_at_sentence_boundary(&state.simhash_pending).is_some()
            || state.simhash_pending.len() >= 1024
        {
            dup = state.simhash_guard.check(&state.simhash_pending);
            state.simhash_pending.clear();
        }
        if state.simhash_pending.len() > 4096 {
            let drop_to = state.simhash_pending.len() / 2;
            state.simhash_pending.drain(..drop_to);
        }
        dup
    } else {
        false
    };

    let token_trip = check_loop_watchdog(
        sanitized,
        &mut state.loop_scan_buf,
        state.loop_watchdog_triggered,
    );

    if semantic_trip || token_trip {
        if semantic_trip {
            tracing::warn!(
                ring_len = state.simhash_guard.len(),
                "SimHash semantic-loop watchdog fired (paraphrased sentence repeat)"
            );
        }
        state.loop_watchdog_triggered = true;
        state.stop_string_triggered = true;
        state
            .cancel_flag
            .store(true, std::sync::atomic::Ordering::Release);

        // Watchdog fired: short-circuit the stream with no further
        // content. The model emitted a degenerate loop; we end the
        // response here rather than salvaging a synthetic tool call.
        return Some(SseVec::new());
    }

    if !sanitized.is_empty() {
        if state.refusal_scan_buf.len() < 16_384 {
            state.refusal_scan_buf.push_str(sanitized);
        }
        let chunk = ChatCompletionChunk::content_chunk(&ctx.model, &ctx.id, sanitized.to_string());
        let json = serde_json::to_string(&chunk).unwrap_or_default();
        let events: SseVec = vec![Ok(Event::default().data(json))];
        return Some(events);
    }
    None
}

/// Detector-active branch's `Content(text)` arm: sanitize first,
/// then run the shared semantic/token watchdog + emit pipeline.
pub(super) fn detector_content_arm(
    state: &mut StreamState,
    ctx: &StreamCtx,
    text: &str,
) -> Option<SseVec> {
    let sanitized = sanitize_content_chunk(
        text,
        &mut state.tag_scan_buf,
        &mut state.suppressing_param_leak,
        &mut state.inside_envelope,
        &ctx.leak_markers,
    );
    process_detector_content(state, ctx, &sanitized)
}
