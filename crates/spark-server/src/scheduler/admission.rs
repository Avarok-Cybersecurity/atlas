// SPDX-License-Identifier: AGPL-3.0-only

//! Depth-aware KV admission: reserve DECODE room, not just the prompt.
//!
//! Admission used to size a request against its PROMPT only
//! (`blocks_needed = prompt_len / block_size + 1`) and never re-consult the
//! pool, so the pool was overcommitted by exactly each sequence's decode
//! depth. Measured on the C=128 ladder (Qwen3.8-27B/GB10): 128 sequences
//! admitted into a 102k-token pool against a 157k-token true demand, four
//! preemption waves, 171 decode-time preemptions. Preemption now RESUMES
//! victims (see `preempt`), but thrash is still pure overhead — the honest
//! fix is to not admit what cannot fit.
//!
//! Policy: a request reserves `prompt + min(max_tokens, WATERMARK)` tokens
//! of blocks (clamped to the served context ceiling), counted against the
//! TOTAL pool minus the same commitment of everything already in flight
//! (active, prefilling, spilled, requeued). Requests beyond capacity stay
//! in the pending queue — the existing queueing machinery — and admit as
//! earlier sequences finish. When everything genuinely fits, the gate
//! admits exactly what the old code admitted: behavior at C<=64 on the
//! measured ladder is unchanged (pinned in tests below).
//!
//! WATERMARK (PCND — explicit, documented default): default `max_seq_len`
//! (0/unlimited ⇒ no clamp), i.e. the reservation is the request's own
//! `max_tokens` — the honest conservative bound, since a request can never
//! generate past it. Operators who prefer overcommit (betting that real
//! generations stop early) set `ATLAS_KV_ADMIT_WATERMARK=<tokens>` lower;
//! `0` reserves prompt-only, which is the pre-gate legacy behavior, and any
//! override below the honest bound keeps a WARN so the C=128 failure mode
//! is at least attributable. The boot-time `KV OVERCOMMIT` warning in
//! `factory/build.rs` is unchanged and complementary (it sizes the pool;
//! this gates runtime admission).

use super::*;

/// Resolve the admission watermark once at scheduler start.
pub(super) fn resolve_admit_watermark(max_seq_len: usize) -> usize {
    let default = if max_seq_len > 0 {
        max_seq_len
    } else {
        usize::MAX
    };
    match std::env::var("ATLAS_KV_ADMIT_WATERMARK") {
        Err(_) => default,
        Ok(v) => match v.parse::<usize>() {
            Ok(w) => {
                if w < default {
                    tracing::warn!(
                        "ATLAS_KV_ADMIT_WATERMARK={w} < the honest bound ({default}): \
                         admission may OVERCOMMIT the KV pool; sequences past the \
                         watermark depth will hit decode-time preemption (resume, \
                         not kill — but pure overhead). 0 = legacy prompt-only \
                         reservation."
                    );
                }
                w
            }
            Err(_) => {
                tracing::warn!(
                    "ATLAS_KV_ADMIT_WATERMARK={v:?} is not an integer; using default {default}"
                );
                default
            }
        },
    }
}

/// SSOT block-count formula — the same shape admission has always used
/// (`tokens / block_size + 1`), kept so the gate never disagrees with the
/// legacy prompt sizing at watermark 0.
pub(super) fn blocks_for_tokens(tokens: usize, block_size: usize) -> usize {
    tokens / block_size.max(1) + 1
}

/// One in-flight sequence's KV demand, in tokens.
pub(super) struct SeqDemand {
    /// Tokens whose KV exists or must exist at resume (prompt + processed).
    pub current_tokens: usize,
    /// Generation still owed (`remaining` / `max_tokens`).
    pub budget_tokens: usize,
}

/// Blocks to reserve for one sequence: current + min(budget, watermark),
/// clamped to the served context ceiling (a sequence can never grow past
/// `max_seq_len`, so reserving beyond it would be dishonest the other way).
pub(super) fn seq_commitment_blocks(
    d: &SeqDemand,
    watermark: usize,
    max_seq_len: usize,
    block_size: usize,
) -> usize {
    let mut depth = d
        .current_tokens
        .saturating_add(d.budget_tokens.min(watermark));
    if max_seq_len > 0 {
        depth = depth.min(max_seq_len.max(d.current_tokens));
    }
    blocks_for_tokens(depth, block_size)
}

/// Total reserved blocks for everything already in flight.
pub(super) fn committed_blocks(
    demands: &[SeqDemand],
    watermark: usize,
    max_seq_len: usize,
    block_size: usize,
) -> usize {
    demands
        .iter()
        .map(|d| seq_commitment_blocks(d, watermark, max_seq_len, block_size))
        .sum()
}

/// How many of `reqs` (`(prompt_len, max_tokens)`, in admission order) fit.
///
/// Returns `(admit_count, forced_oversize)`. Admission stops at the FIRST
/// request that does not fit (no head-of-line bypass: a small request must
/// not starve a big one that arrived first). LIVENESS: when nothing at all
/// is in flight and even the first request cannot fit, it is admitted
/// anyway (`forced_oversize = true`) — exactly today's behavior, where the
/// block allocator back-pressures at runtime — because queueing it forever
/// against an empty pool serves nobody.
pub(super) fn admit_count(
    total_blocks: usize,
    committed: usize,
    reqs: &[(usize, usize)],
    watermark: usize,
    max_seq_len: usize,
    block_size: usize,
) -> (usize, bool) {
    let mut used = committed;
    let mut n = 0usize;
    for &(prompt, max_tokens) in reqs {
        let need = seq_commitment_blocks(
            &SeqDemand {
                current_tokens: prompt,
                budget_tokens: max_tokens,
            },
            watermark,
            max_seq_len,
            block_size,
        );
        if used.saturating_add(need) <= total_blocks {
            used += need;
            n += 1;
        } else if n == 0 && committed == 0 {
            return (1, true);
        } else {
            break;
        }
    }
    (n, false)
}

/// Defer any new request whose adapter differs from the in-flight cohort's.
///
/// Returns the requests that may join the current batch; the rest are pushed
/// back to the FRONT of the pending queue (preserving their relative order) so
/// they run as soon as the batch drains. A no-op when nothing is in flight
/// (the first admitted request defines the cohort) or when no adapter pool is
/// resident, in which case every id is the base sentinel and all requests
/// compare equal.
fn filter_adapter_cohort(
    model: &dyn Model,
    pending: &std::sync::Arc<(Mutex<PendingQueue>, Condvar)>,
    new_reqs: Vec<InferenceRequest>,
    active: &[ActiveSeq],
    prefilling: &[PrefillInProgress],
) -> Vec<InferenceRequest> {
    let cohort = active
        .first()
        .map(|a| a.seq.adapter_slot)
        .or_else(|| prefilling.first().map(|p| p.seq.adapter_slot));
    // Nothing in flight: the FIRST request of this wave defines the cohort.
    // Without this, two requests naming different adapters that arrive in the
    // same wave are both admitted into an empty batch and poison each other —
    // which is exactly what a concurrent streaming pair does.
    let cohort_slot = match cohort {
        Some(s) => s,
        None => match new_reqs.first() {
            Some(r) => r.adapter_slot(),
            None => return new_reqs,
        },
    };
    let cohort_id = model.adapter_id_for(cohort_slot);
    let (admitted, deferred): (Vec<_>, Vec<_>) = new_reqs
        .into_iter()
        .partition(|r| model.adapter_id_for(r.adapter_slot()) == cohort_id);
    if !deferred.is_empty() {
        tracing::debug!(
            "adapter cohort: holding {} request(s) for a different adapter until \
             the current batch drains (v0 is single-active)",
            deferred.len()
        );
        let mut g = pending.0.lock();
        for (i, req) in deferred.into_iter().enumerate() {
            g.requests.insert(i, req);
        }
    }
    admitted
}

/// Runtime gate: split this tick's drained requests into an admissible
/// prefix (returned) and an overflow tail (pushed back to the FRONT of the
/// pending queue, preserving arrival order ahead of newer requests).
#[allow(clippy::too_many_arguments)]
pub(super) fn gate_admissions(
    model: &dyn Model,
    pending: &std::sync::Arc<(Mutex<PendingQueue>, Condvar)>,
    new_reqs: Vec<InferenceRequest>,
    active: &[ActiveSeq],
    prefilling: &[PrefillInProgress],
    swapped: &[SwappedSeq],
    preempted: &[PreemptedSeq],
    watermark: usize,
    max_seq_len: usize,
    block_size: usize,
) -> Vec<InferenceRequest> {
    if new_reqs.is_empty() {
        return new_reqs;
    }
    // Adapter cohort: a batch may only hold sequences sharing ONE adapter.
    //
    // v0 LoRA is single-active. A decode batch containing a sequence routed to
    // a non-active adapter is refused wholesale by
    // `decode_batch_compute_main` — which kills EVERY request in that batch,
    // including the innocent ones routed to the active adapter. Measured: two
    // concurrent requests naming different resident adapters both died with
    // HTTP 500, while either one ALONE completes normally.
    //
    // So hold the mismatched request instead of admitting it into a batch it
    // will poison. It stays at the head of the pending queue and runs once the
    // current cohort drains, which is the same discipline adapter rotation
    // already follows. Serialised rather than failed.
    //
    // Identity comes from `adapter_id_for`, which resolves the `-1`
    // "defer to active" sentinel the same way the model does — comparing raw
    // slot indices would treat `-1` and the active slot as different cohorts.
    let new_reqs = filter_adapter_cohort(model, pending, new_reqs, active, prefilling);
    if new_reqs.is_empty() {
        return new_reqs;
    }
    let total_blocks = model.num_total_blocks();
    if total_blocks == 0 {
        // Backend without a paged KV pool (or no occupancy info): nothing to
        // reserve against — admit as before.
        return new_reqs;
    }
    let mut demands: Vec<SeqDemand> =
        Vec::with_capacity(active.len() + prefilling.len() + swapped.len() + preempted.len());
    demands.extend(active.iter().map(|a| SeqDemand {
        // +1: the pending `last_token` decode input not yet in seq_len.
        current_tokens: a.seq.seq_len + 1,
        budget_tokens: a.remaining,
    }));
    demands.extend(prefilling.iter().map(|p| SeqDemand {
        current_tokens: p.prompt_tokens.len(),
        budget_tokens: p.max_tokens,
    }));
    demands.extend(swapped.iter().map(|s| SeqDemand {
        current_tokens: s.seq_len + 1,
        budget_tokens: s.remaining,
    }));
    demands.extend(preempted.iter().map(|p| SeqDemand {
        current_tokens: p.tokens.len() + 1,
        budget_tokens: p.a.remaining,
    }));
    let committed = committed_blocks(&demands, watermark, max_seq_len, block_size);
    let infos: Vec<(usize, usize)> = new_reqs
        .iter()
        .map(|r| (r.prompt_len(), r.max_tokens()))
        .collect();
    let (admit, forced) = admit_count(
        total_blocks,
        committed,
        &infos,
        watermark,
        max_seq_len,
        block_size,
    );
    if forced {
        tracing::warn!(
            "admitting a request whose reservation exceeds the whole KV pool \
             ({} prompt + min({}, watermark {}) tokens vs {} blocks); the block \
             allocator will back-pressure at runtime",
            infos[0].0,
            infos[0].1,
            watermark,
            total_blocks,
        );
    }
    // Overflow re-queues cycle through drain→gate every tick while parked;
    // log on CHANGE only, or a parked C=128 burst emits thousands of
    // identical lines per minute.
    static LAST_QUEUED: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    if admit >= new_reqs.len() {
        LAST_QUEUED.store(0, std::sync::atomic::Ordering::Relaxed);
        return new_reqs;
    }
    let mut admitted = new_reqs;
    let overflow = admitted.split_off(admit);
    if LAST_QUEUED.swap(overflow.len(), std::sync::atomic::Ordering::Relaxed) != overflow.len() {
        tracing::info!(
            "KV admission: {} of {} request(s) fit ({} blocks committed of {}); \
             {} queued until capacity frees",
            admitted.len(),
            admitted.len() + overflow.len(),
            committed,
            total_blocks,
            overflow.len(),
        );
    }
    let mut g = pending.0.lock();
    for (i, req) in overflow.into_iter().enumerate() {
        g.requests.insert(i, req);
    }
    admitted
}

#[cfg(test)]
mod tests {
    use super::*;

    // The forensic C=128 ladder shape (2026-08-15, Qwen3.8-27B/GB10):
    // pool 102k tokens, per-request demand ~1226 tokens (prompt ~202 +
    // max_tokens 1024), 128 requests ⇒ 157k demand. block_size 16.
    const BS: usize = 16;
    const POOL_BLOCKS: usize = 102_000 / BS; // 6375
    const PROMPT: usize = 202;
    const MAX_TOK: usize = 1024;
    const MAX_SEQ_LEN: usize = 8192;

    fn req_blocks() -> usize {
        blocks_for_tokens(PROMPT + MAX_TOK, BS) // 77
    }

    #[test]
    fn block_math_matches_legacy_formula() {
        // Same `tokens / block_size + 1` shape admission always used.
        assert_eq!(blocks_for_tokens(0, 16), 1);
        assert_eq!(blocks_for_tokens(15, 16), 1);
        assert_eq!(blocks_for_tokens(16, 16), 2);
        assert_eq!(blocks_for_tokens(1226, 16), 77);
    }

    #[test]
    fn watermark_caps_the_decode_reservation() {
        let d = SeqDemand {
            current_tokens: 200,
            budget_tokens: 4096,
        };
        // Watermark below max_tokens: reserve prompt + watermark.
        assert_eq!(
            seq_commitment_blocks(&d, 512, MAX_SEQ_LEN, BS),
            blocks_for_tokens(200 + 512, BS)
        );
        // Watermark above max_tokens: the request's own budget bounds it.
        assert_eq!(
            seq_commitment_blocks(&d, usize::MAX, MAX_SEQ_LEN, BS),
            blocks_for_tokens(200 + 4096, BS)
        );
        // Watermark 0 = legacy prompt-only reservation.
        assert_eq!(
            seq_commitment_blocks(&d, 0, MAX_SEQ_LEN, BS),
            blocks_for_tokens(200, BS)
        );
        // The served context ceiling clamps the depth: no sequence can grow
        // past max_seq_len, so nothing more is ever reserved.
        let long = SeqDemand {
            current_tokens: 8000,
            budget_tokens: 4096,
        };
        assert_eq!(
            seq_commitment_blocks(&long, usize::MAX, MAX_SEQ_LEN, BS),
            blocks_for_tokens(MAX_SEQ_LEN, BS)
        );
    }

    #[test]
    fn fits_everything_admission_unchanged() {
        // C<=64 rung: 64 × 77 = 4928 blocks ≤ 6375 — ALL admitted, exactly
        // as the pre-gate code admitted them. Pins the no-regression claim.
        let reqs = vec![(PROMPT, MAX_TOK); 64];
        let (n, forced) = admit_count(POOL_BLOCKS, 0, &reqs, MAX_SEQ_LEN, MAX_SEQ_LEN, BS);
        assert_eq!(n, 64);
        assert!(!forced);
    }

    #[test]
    fn c128_overflow_queues_instead_of_admit_then_shoot() {
        // The measured failure: 128 requests whose true demand (157k tokens)
        // exceeds the 102k pool. The gate admits what fits and QUEUES the
        // rest — no admit-then-preempt thrash.
        let reqs = vec![(PROMPT, MAX_TOK); 128];
        let (n, forced) = admit_count(POOL_BLOCKS, 0, &reqs, MAX_SEQ_LEN, MAX_SEQ_LEN, BS);
        assert_eq!(n, POOL_BLOCKS / req_blocks()); // 82: every admitted seq fits fully
        assert!(n < 128);
        assert!(!forced);
        // The admitted set can never exhaust the pool.
        assert!(n * req_blocks() <= POOL_BLOCKS);
    }

    #[test]
    fn in_flight_commitments_reduce_capacity() {
        // 40 active sequences mid-decode still owe their remaining budget.
        let demands: Vec<SeqDemand> = (0..40)
            .map(|_| SeqDemand {
                current_tokens: 600,
                budget_tokens: 700,
            })
            .collect();
        let committed = committed_blocks(&demands, MAX_SEQ_LEN, MAX_SEQ_LEN, BS);
        assert_eq!(committed, 40 * blocks_for_tokens(1300, BS));
        let reqs = vec![(PROMPT, MAX_TOK); 128];
        let (n, _) = admit_count(POOL_BLOCKS, committed, &reqs, MAX_SEQ_LEN, MAX_SEQ_LEN, BS);
        assert_eq!(n, (POOL_BLOCKS - committed) / req_blocks());
    }

    #[test]
    fn admission_stops_at_first_misfit_no_head_of_line_bypass() {
        // A huge request mid-queue blocks later small ones from jumping it.
        let reqs = vec![
            (PROMPT, MAX_TOK),
            (100_000, MAX_TOK), // cannot fit
            (PROMPT, MAX_TOK),  // must NOT bypass
        ];
        let (n, forced) = admit_count(POOL_BLOCKS, 0, &reqs, usize::MAX, 0, BS);
        assert_eq!(n, 1);
        assert!(!forced);
    }

    #[test]
    fn liveness_oversized_lone_request_still_admits() {
        // Nothing in flight + a request bigger than the whole pool: admit it
        // (runtime back-pressure applies), never queue it forever.
        let reqs = vec![(200_000, MAX_TOK)];
        let (n, forced) = admit_count(POOL_BLOCKS, 0, &reqs, usize::MAX, 0, BS);
        assert_eq!(n, 1);
        assert!(forced);
        // But with ANYTHING in flight it waits its turn.
        let (n, forced) = admit_count(POOL_BLOCKS, 10, &reqs, usize::MAX, 0, BS);
        assert_eq!(n, 0);
        assert!(!forced);
    }

    #[test]
    fn watermark_zero_reserves_prompt_only_legacy() {
        // Escape hatch pinned: watermark 0 ⇒ the C=128 set is admitted in
        // full (128 × blocks(202) = 128 × 13 = 1664 ≤ 6375) — byte-for-byte
        // the legacy prompt-only admission decision.
        let reqs = vec![(PROMPT, MAX_TOK); 128];
        let (n, forced) = admit_count(POOL_BLOCKS, 0, &reqs, 0, MAX_SEQ_LEN, BS);
        assert_eq!(n, 128);
        assert!(!forced);
    }
}
