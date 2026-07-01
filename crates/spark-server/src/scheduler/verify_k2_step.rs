// SPDX-License-Identifier: AGPL-3.0-only

//! K=2 verify step.

use super::*;

// Periodic ACCEPT/REJECT summary counters (P4, 2026-05-24).
// Replaces the earlier `is_multiple_of(50)` per-step gate which hid
// accept events behind seq_len happenstance. We now log a single
// summary line every K2_SUMMARY_PERIOD verify steps and reset.
// Atomics are fine: scheduler runs on a single thread today, but
// future multi-scheduler builds remain race-free.
const K2_SUMMARY_PERIOD: u64 = 100;
static K2_ACCEPTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static K2_REJECTS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[inline]
fn k2_record_outcome(accepted: bool, seq_len: usize) {
    let counter = if accepted { &K2_ACCEPTS } else { &K2_REJECTS };
    counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let total = K2_ACCEPTS.load(std::sync::atomic::Ordering::Relaxed)
        + K2_REJECTS.load(std::sync::atomic::Ordering::Relaxed);
    if total >= K2_SUMMARY_PERIOD {
        let accepts = K2_ACCEPTS.swap(0, std::sync::atomic::Ordering::Relaxed);
        let rejects = K2_REJECTS.swap(0, std::sync::atomic::Ordering::Relaxed);
        let total = (accepts + rejects).max(1);
        let pct = 100.0 * (accepts as f64) / (total as f64);
        // A6 (2026-05-26): MTP K=2 acceptance rate as a free drift
        // gauge. The HF z-lab/Qwen3.6-27B-DFlash#2 report and our own
        // research3_spec_verify finding show that FP8 Qwen3.6 with
        // sustained <30% MTP accept indicates the target model's
        // logits have entered the "confidently wrong" attractor — the
        // exact failure mode driving the opencode multi-turn drift.
        // For now we just WARN to surface the signal in production
        // logs; a future Tier-B refinement adds per-sequence state +
        // `finish_reason="drift_detected"` to actually terminate the
        // response when the gauge trips.
        const DRIFT_THRESHOLD_PCT: f64 = 30.0;
        if pct < DRIFT_THRESHOLD_PCT && total >= K2_SUMMARY_PERIOD {
            tracing::warn!(
                "K2 drift gauge: accept rate {pct:.1}% < {DRIFT_THRESHOLD_PCT}% over last {total} steps (seq_len={seq_len}). Model logits likely in 'confidently wrong' attractor."
            );
        } else {
            tracing::info!(
                "K2 summary: {accepts} accept / {rejects} reject in last {total} steps ({pct:.1}% accept) seq_len={seq_len}"
            );
        }
    }
}

/// K=2 verify: [last_token, draft] → [v0, v1]. Two outcomes: ACCEPT or REJECT.
///
/// `verify_ctx` carries tokenizer special-token IDs the pre-sample
/// pipeline needs. Each verify position's logits are now copied D2H
/// and run through the same 8-stage processor pipeline used by the
/// non-MTP path — the fix for MTP-emitted tokens bypassing mid-word
/// defer, post-close mask, forced think-end, grammar bitmask, etc.
/// See `verify_pipeline_helper` for the root-cause writeup.
pub fn step_verify_k2(
    model: &dyn Model,
    a: &mut ActiveSeq,
    drafts: &[u32],
    num_drafts: usize,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
    dflash_verify_raw_argmax: bool,
) {
    let t_sync = Instant::now();
    if let Err(e) = model.sync_secondary() {
        tracing::error!("sync_secondary: {e:#}");
        a.finished = true;
        return;
    }
    let sync_us = t_sync.elapsed().as_micros();

    // EP: broadcast verify K=2 command + tokens so worker runs decode_verify_graphed in lockstep.
    let t_ep = Instant::now();
    let tokens_k2 = [a.last_token, drafts[0]];
    if let Err(e) = model.ep_broadcast_cmd_for_seq(a.seq.slot_idx as u32, 0xFFFFFFF2) {
        tracing::error!("EP broadcast verify_k2 cmd: {e:#}");
        a.finished = true;
        return;
    }
    for &t in &tokens_k2 {
        if let Err(e) = model.ep_broadcast_cmd(t) {
            tracing::error!("EP broadcast verify_k2 token: {e:#}");
            a.finished = true;
            return;
        }
    }

    let ep_us = t_ep.elapsed().as_micros();

    let t_verify = Instant::now();
    let result = match model.decode_verify_graphed(&tokens_k2, &mut a.seq, 0) {
        Ok(r) => r,
        Err(e) => {
            tracing::error!("decode_verify_graphed: {e:#}");
            a.finished = true;
            return;
        }
    };
    let verify_us = t_verify.elapsed().as_micros();
    a.last_token_time = Instant::now();
    let [v0_argmax, v1_argmax] = result;

    // Phase C-2 (2026-05-24): apply the full pre-sample logits-
    // processor pipeline to each verify position. Without this,
    // MTP-emitted tokens escape every mask the non-MTP path applies
    // (grammar desync + mid-word `</think>` cuts + stray `<think>`
    // re-entry + malformed tool calls). The D2H copy is one-shot
    // (~0.8ms for 256k vocab × K=2); not graph-captured.
    //
    // DFlash split: acceptance check uses raw GPU argmax (drafter
    // proposes via raw argmax — compare on the same basis). Emission
    // always uses the pipeline-processed token so the output sequence
    // stays aligned with what baseline decode would have produced.
    // Previously both check and emit used raw argmax, causing sequence
    // drift on rejection and collapsing accept rate from ~2.5 to ~1%.
    let processed = crate::scheduler::verify_pipeline_helper::verify_pick_all_with_pipeline(
        model,
        &[v0_argmax, v1_argmax],
        a,
        verify_ctx,
    );
    let v0_emit = processed.first().copied().unwrap_or(v0_argmax);
    let v1_emit = processed.get(1).copied().unwrap_or(v1_argmax);

    // Acceptance check: DFlash compares against raw argmax; MTP against processed.
    // At T>0 with DFlash: use rejection sampling since the drafter always proposes
    // via argmax (T=0 greedy) while the target has a stochastic distribution.
    // With a point-mass drafter (p_draft = 1 at argmax), the spec-decode acceptance
    // probability simplifies to p_target(draft_token, T), with no floor needed.
    let v0_check = if dflash_verify_raw_argmax {
        v0_argmax
    } else {
        v0_emit
    };
    let accepted = if dflash_verify_raw_argmax && a.temperature > 0.0 {
        dflash_stochastic_accept(model, drafts[0], a.temperature, a.seed, a.seq.seq_len)
    } else {
        drafts[0] == v0_check
    };

    // Extract logprobs from verify logits buffer (K=2 positions) when requested.
    let verify_lps = if let Some(top_logprobs) = a.top_logprobs {
        extract_verify_logprobs(model, &[v0_emit, v1_emit], top_logprobs)
    } else {
        Vec::new()
    };

    // EP: always broadcast accept/reject to worker (prevents deadlock on EOS).
    if let Err(e) = model.ep_broadcast_cmd(accepted as u32) {
        tracing::error!("EP broadcast verify_k2 result: {e:#}");
        a.finished = true;
        return;
    }

    // EASD scaffolding (A.2): track baseline accept rate to decide
    // whether activating entropy-aware-spec-decode (per-step D2H of
    // verify logits + entropy threshold per arXiv:2512.23765) is
    // worth its overhead. Baseline acceptance is the prerequisite
    // signal — high accept rate means EASD has little to gain.
    crate::metrics::SPEC_DECODE_VERIFY
        .with_label_values(&["2", if accepted { "accept" } else { "reject" }])
        .inc();

    if accepted {
        // ── ACCEPTED ──
        emit_token(a, v0_emit, verify_lps.first().cloned());
        if !a.finished {
            emit_token(a, v1_emit, verify_lps.get(1).cloned());
        }
        if a.finished {
            return;
        }
        a.last_token = v1_emit;

        // Item #2 (STree-style in-place K=2 verify commit). Full accept
        // (num_accepted=k=2): the verify kernel already wrote the canonical
        // h_state, so the commit is a no-op.
        if let Err(e) = model.commit_accepted_prefix(&mut a.seq, 2, 2) {
            tracing::error!("commit_accepted_prefix (accept): {e:#}");
            return;
        }
        if let Err(e) = model.save_hidden_for_mtp(1, 0) {
            tracing::error!("save_hidden_for_mtp(1): {e:#}");
            return;
        }
        if let Err(e) = model.trim_proposer_state(&mut a.seq, 1, 0) {
            tracing::error!("trim_proposer_state: {e:#}");
        }
        // EAGLE-fix (ATLAS_DFLASH_EAGLE_FIX=1, K=2 accept only): append row 0 @ N
        // then row 1 @ N+1 BEFORE propose so forward_block conditions on row 1
        // (the hidden that generated bonus_S). This also sets the proposer's
        // skip-flag so propose does NOT re-append row 0. With the flag off, the
        // legacy path runs unchanged (propose appends row 0; row 1 appended
        // post-propose below).
        let eagle_fix = std::env::var("ATLAS_DFLASH_EAGLE_FIX").ok().as_deref() == Some("1");
        if eagle_fix && let Err(e) = model.dflash_eagle_accept_append(&mut a.seq) {
            tracing::error!("dflash_eagle_accept_append: {e:#}");
        }
        let t_propose = Instant::now();
        let _mtp_grammar_mask = mtp_grammar_mask_for(a);
        match model.run_mtp_propose_multi(
            v1_emit,
            a.seq.seq_len,
            num_drafts,
            &mut a.seq,
            0,
            _mtp_grammar_mask.as_deref(),
        ) {
            Ok(d) if !d.is_empty() => a.pending_drafts = d,
            Ok(_) => {}
            Err(e) => {
                tracing::error!("run_mtp_propose_multi: {e:#}");
            }
        }
        // Legacy path only: append the accepted draft's hidden (row 1 of
        // dflash_hidden_save) AFTER propose so ctx_len stays in sync with
        // seq_len. Under EAGLE-fix row 1 was already appended before propose, so
        // skip here to avoid a duplicate slot.
        if !eagle_fix && let Err(e) = model.dflash_accept_append(&mut a.seq) {
            tracing::error!("dflash_accept_append: {e:#}");
        }
        let propose_us = t_propose.elapsed().as_micros();
        // Per-step ACCEPT trace at debug — fires every step during
        // spec-decode. Power-user diagnostics:
        // `RUST_LOG=spark::scheduler::verify_k2_step=debug`. The
        // summary line emitted by `k2_record_outcome` is the
        // production signal.
        tracing::debug!(
            "K2 ACCEPT: ep={ep_us}μs sync={sync_us}μs verify={verify_us}μs propose={propose_us}μs seq_len={}",
            a.seq.seq_len
        );
        k2_record_outcome(true, a.seq.seq_len);
        // #155 iter3: block-aligned Marconi checkpoint (live SSM state is
        // canonical post-commit). Fires only at interval boundaries.
        model.decode_marconi_checkpoint(&mut a.seq);
    } else {
        // ── REJECTED ──
        a.seq.seq_len -= 1;
        a.seq.tokens.pop();

        if let Err(e) = model.trim_proposer_state(&mut a.seq, 0, 0) {
            tracing::error!("trim_proposer_state: {e:#}");
        }
        // Item #2 (STree-style in-place K=2 verify commit). K=2 reject means
        // num_accepted=1 (last_token is always accepted): the in-place
        // commit rewinds live h_state to intermediate[0] (state after the
        // accepted token). Verify draft state is discarded.
        if let Err(e) = model.commit_accepted_prefix(&mut a.seq, 1, 2) {
            tracing::error!("commit_accepted_prefix (reject): {e:#}");
            a.finished = true;
            return;
        }

        emit_token(a, v0_emit, verify_lps.first().cloned());
        if a.finished {
            return;
        }
        a.last_token = v0_emit;

        if let Err(e) = model.save_hidden_for_mtp(0, 0) {
            tracing::error!("save_hidden_for_mtp(0): {e:#}");
            return;
        }
        let t_propose = Instant::now();
        let _mtp_grammar_mask = mtp_grammar_mask_for(a);
        match model.run_mtp_propose_multi(
            v0_emit,
            a.seq.seq_len,
            num_drafts,
            &mut a.seq,
            0,
            _mtp_grammar_mask.as_deref(),
        ) {
            Ok(d) if !d.is_empty() => a.pending_drafts = d,
            Ok(_) => {}
            Err(e) => {
                tracing::error!("run_mtp_propose_multi: {e:#}");
            }
        }
        let propose_us = t_propose.elapsed().as_micros();
        let new_draft = a.pending_drafts.first().copied().unwrap_or(0);
        // REJECT log demoted from `info!` to `debug!` to match the
        // ACCEPT path. Rejection is informative when investigating
        // draft quality but spams Docker logs at production load.
        // Periodic accept-rate summary lives in `k2_record_outcome`.
        tracing::debug!(
            "K2 REJECT: ep={ep_us}μs sync={sync_us}μs verify={verify_us}μs propose={propose_us}μs seq_len={} last_tok={} prev_draft={} v0_verified={} new_draft={}",
            a.seq.seq_len,
            a.last_token,
            drafts[0],
            v0_emit,
            new_draft,
        );
        k2_record_outcome(false, a.seq.seq_len);
        // #155 iter3: block-aligned Marconi checkpoint (live SSM state is
        // canonical post-commit). Fires only at interval boundaries.
        model.decode_marconi_checkpoint(&mut a.seq);
    }
}

/// Probabilistic DFlash acceptance for T>0.
///
/// DFlash drafter always uses greedy argmax (T=0), so p_draft is a point mass
/// at the draft token: p_draft(draft) = 1. The spec-decode acceptance criterion
/// min(1, p_target(d,T) / p_draft(d)) therefore simplifies to p_target(d, T).
///
/// We D2H-copy position-0 target logits (vocab × 2 BF16 bytes), compute softmax
/// at temperature T, and accept with probability equal to the draft token's
/// probability under the target distribution.
///
/// Falls back to argmax-equality check on D2H failure.
fn dflash_stochastic_accept(
    model: &dyn Model,
    draft_tok: u32,
    temperature: f32,
    seed: Option<u64>,
    seq_len: usize,
) -> bool {
    let vocab = model.vocab_size();
    // Copy only position-0 logits (vocab × 2 bytes) from the [K, vocab] BF16 buffer.
    let mut logits_pos0 = vec![0u8; vocab * 2];
    if model
        .copy_logits_to_host(model.logits_buffer_ptr(), &mut logits_pos0)
        .is_err()
    {
        // Fallback: treat as if T=0 (accept only on exact argmax match).
        // This only fires on D2H errors (essentially never in practice).
        let argmax = logits_pos0
            .chunks_exact(2)
            .enumerate()
            .map(|(i, c)| (i, bf16_to_f32(c[0], c[1])))
            .max_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
            .map(|(i, _)| i as u32)
            .unwrap_or(0);
        return draft_tok == argmax;
    }

    // Compute numerically stable softmax(logits / T) and extract p_target(draft).
    let inv_t = 1.0_f32 / temperature;
    let mut max_scaled = f32::NEG_INFINITY;
    for c in logits_pos0.chunks_exact(2) {
        let v = bf16_to_f32(c[0], c[1]) * inv_t;
        if v > max_scaled {
            max_scaled = v;
        }
    }
    let mut sum_exp = 0.0_f32;
    let draft_scaled = bf16_to_f32(
        logits_pos0[draft_tok as usize * 2],
        logits_pos0[draft_tok as usize * 2 + 1],
    ) * inv_t;
    for c in logits_pos0.chunks_exact(2) {
        sum_exp += (bf16_to_f32(c[0], c[1]) * inv_t - max_scaled).exp();
    }
    let p_target = ((draft_scaled - max_scaled).exp()) / sum_exp;

    // Deterministic uniform sample: xorshift64 seeded with (seed, seq_len, draft_tok).
    // This makes acceptance reproducible when a.seed is set (T>0 deterministic mode).
    let raw_seed = seed
        .unwrap_or(0)
        .wrapping_add(seq_len as u64)
        .wrapping_add((draft_tok as u64).wrapping_mul(6364136223846793005));
    let mut x = if raw_seed == 0 {
        2862933555777941757
    } else {
        raw_seed
    };
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    let u = (x >> 11) as f32 / (1u64 << 53) as f32;

    u < p_target
}
