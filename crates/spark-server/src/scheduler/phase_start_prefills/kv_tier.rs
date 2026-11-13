// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the host KV tier's scheduler hooks (lane L26b), called from
//! `start_new_requests`: the per-tick maintenance, the prefetch that runs before a request's
//! normal prefix lookup (synchronous, or parked through lane L27's `restore_queue` when the
//! `kv_restore_async` row is on), and the prefill timing that feeds the restore cost model.
//!
//! Every hook asks the model first (`Model::port_ext`, then `kv_tier_active`): with the
//! `kv_host_tier` row off, no budget or the prefix cache closed, a request passes through
//! untouched and nothing is timed.

use std::time::Instant;

use spark_model::traits::{Model, SequenceState};

use crate::api::InferenceRequest;

/// Prefills shorter than this many computed tokens do not feed the cost model (their time is
/// dominated by fixed per-request costs).
const MIN_OBSERVED_TOKENS: usize = 64;

/// Once per scheduler tick, before any request starts.
pub(super) fn tick(model: &dyn Model) {
    if let Some(ext) = model.port_ext() {
        ext.kv_tier_tick();
    }
}

/// Restore `req`'s host-held prefix before its prefill. `None`: the request was parked until
/// its restore completes (lane L27 re-admits it); `Some`: run its prefill now.
pub(super) fn before_prefill(model: &dyn Model, req: InferenceRequest) -> Option<InferenceRequest> {
    let Some(ext) = model.port_ext() else {
        return Some(req);
    };
    if !ext.kv_tier_active() || !eligible(&req) {
        return Some(req);
    }
    // v1 stores base-model KV only; an adapter request never restores.
    let Ok(adapter) = u32::try_from(model.adapter_id_for(req.adapter_slot())) else {
        return Some(req);
    };
    if adapter != 0 {
        return Some(req);
    }
    let tokens = req.prompt_tokens_arc();
    let session = req.session_hash();
    if spark_model::layers::ops::port_rows::kv_restore_async::row() {
        match ext.kv_tier_restore_begin(&tokens, adapter, session) {
            Ok(Some(ticket)) => match crate::scheduler::restore_queue::try_park(req, ticket) {
                Ok(()) => return None,
                // Not parkable (queue full or closed): finish the restore here and go on.
                Err((req, ticket)) => {
                    ext.kv_tier_settle(ticket);
                    return Some(req);
                }
            },
            Ok(None) => return Some(req),
            Err(e) => {
                tracing::warn!("kv_host_tier: async restore not started: {e:#}");
                return Some(req);
            }
        }
    }
    match ext.kv_tier_prefetch(&tokens, adapter, session) {
        Ok(p) => tracing::debug!(
            "kv_host_tier prefetch: {:?} (gpu {} tokens, restored {} tokens)",
            p.outcome,
            p.gpu_tokens,
            p.restored_tokens
        ),
        Err(e) => tracing::warn!("kv_host_tier prefetch failed (recomputing): {e:#}"),
    }
    Some(req)
}

/// Requests whose prefill reads the prefix cache the ordinary way. Prompt-logprob requests
/// recompute every position, image requests are keyed by placeholder tokens and beam requests
/// run their own search.
fn eligible(req: &InferenceRequest) -> bool {
    req.prompt_logprobs().is_none() && !req.has_image_pixels() && req.num_beams() <= 1
}

/// Times one request's prefill for the restore cost model.
pub(super) struct PrefillClock {
    start: Option<Instant>,
}

impl PrefillClock {
    pub(super) fn start(model: &dyn Model) -> Self {
        let on = model.port_ext().is_some_and(|e| e.kv_tier_active());
        Self {
            start: on.then(Instant::now),
        }
    }

    /// The prefill of `seq` completed (its first token was sampled).
    pub(super) fn finish(self, model: &dyn Model, seq: &SequenceState) {
        let Some(t0) = self.start else {
            return;
        };
        let computed = seq.prompt_len.saturating_sub(seq.cached_prefix_tokens);
        if computed < MIN_OBSERVED_TOKENS {
            return;
        }
        if let Some(ext) = model.port_ext() {
            ext.kv_tier_observe_prefill(computed, t0.elapsed().as_secs_f64());
        }
    }
}
