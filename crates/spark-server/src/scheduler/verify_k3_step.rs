// SPDX-License-Identifier: AGPL-3.0-only

//! K=3 verify step.

use super::*;

// Periodic accept-distribution summary (P4, 2026-05-24). K=3 has
// three outcomes (0/1/2 drafts accepted) so we track three counters
// and emit a summary line every K3_SUMMARY_PERIOD verify steps.
const K3_SUMMARY_PERIOD: u64 = 100;
static K3_ACCEPT_2: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static K3_ACCEPT_1: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static K3_ACCEPT_0: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

// UNCONDITIONAL per-position draft-match counters (2026-07-21).
//
// The accept-chain (`num_accepted`) short-circuits: if draft 1 is rejected,
// draft 2 is discarded WITHOUT being scored, so the only rate the chain can
// report for position 2 is the CONDITIONAL p(2|1). Measured p1 ~= 0.70 but
// p(2|1) ~= 0.53, and it is not possible to tell from the chain alone whether
// position 2 is genuinely worse or whether p(2|1) is a survivorship artifact
// (position 2 is only ever scored on contexts where position 1 already
// succeeded, which is a biased sample).
//
// The verify step already computes the target argmax at EVERY position
// (`v0`, `v1`, `v2`) in one batched pass, so `drafts[1] == v1` is observable
// on every step regardless of whether `drafts[0] == v0`. That is the
// unconditional rate. Caveat worth remembering when reading it: `v1` is the
// target's argmax GIVEN `drafts[0]` as the preceding token, so when draft 1
// was wrong this measures the drafter on a counterfactual context — which is
// exactly the comparison we want (same position, unbiased sample of contexts).
static K3_STEPS: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static K3_D1_MATCH: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static K3_D2_MATCH_UNCOND: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static K3_D2_MATCH_COND: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[inline]
fn k3_record_positional(d1_match: bool, d2_match: bool, seq_len: usize) {
    K3_STEPS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    if d1_match {
        K3_D1_MATCH.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if d2_match {
            K3_D2_MATCH_COND.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        }
    }
    if d2_match {
        K3_D2_MATCH_UNCOND.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    }
    if K3_STEPS.load(std::sync::atomic::Ordering::Relaxed) >= K3_SUMMARY_PERIOD {
        let steps = K3_STEPS
            .swap(0, std::sync::atomic::Ordering::Relaxed)
            .max(1);
        let d1 = K3_D1_MATCH.swap(0, std::sync::atomic::Ordering::Relaxed);
        let d2u = K3_D2_MATCH_UNCOND.swap(0, std::sync::atomic::Ordering::Relaxed);
        let d2c = K3_D2_MATCH_COND.swap(0, std::sync::atomic::Ordering::Relaxed);
        let p1 = (d1 as f64) / (steps as f64);
        let p2_uncond = (d2u as f64) / (steps as f64);
        let p2_cond = if d1 > 0 {
            (d2c as f64) / (d1 as f64)
        } else {
            f64::NAN
        };
        tracing::info!(
            "K3 positional: steps={steps} p1={p1:.3} p2_uncond={p2_uncond:.3} \
             p2_cond={p2_cond:.3} (d1={d1} d2u={d2u} d2c={d2c}) seq_len={seq_len} \
             [p2_uncond ~= p2_cond => position 2 genuinely worse; \
              p2_uncond ~= p1 => p2_cond is survivorship]"
        );
    }
}

#[inline]
fn k3_record_outcome(num_accepted: usize, seq_len: usize) {
    let counter = match num_accepted {
        2 => &K3_ACCEPT_2,
        1 => &K3_ACCEPT_1,
        _ => &K3_ACCEPT_0,
    };
    counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let total = K3_ACCEPT_2.load(std::sync::atomic::Ordering::Relaxed)
        + K3_ACCEPT_1.load(std::sync::atomic::Ordering::Relaxed)
        + K3_ACCEPT_0.load(std::sync::atomic::Ordering::Relaxed);
    if total >= K3_SUMMARY_PERIOD {
        let a2 = K3_ACCEPT_2.swap(0, std::sync::atomic::Ordering::Relaxed);
        let a1 = K3_ACCEPT_1.swap(0, std::sync::atomic::Ordering::Relaxed);
        let a0 = K3_ACCEPT_0.swap(0, std::sync::atomic::Ordering::Relaxed);
        let total = (a2 + a1 + a0).max(1);
        let mean = (2 * a2 + a1) as f64 / total as f64;
        tracing::info!(
            "K3 summary: {a2} accept-2 / {a1} accept-1 / {a0} reject in last {total} steps (mean accepted={mean:.2}) seq_len={seq_len}"
        );
    }
}

/// K=3 verify: [last_token, draft1, draft2] → [v0, v1, v2]. Three outcomes.
///
/// `verify_ctx` carries the tokenizer special-token IDs the
/// pre-sample logits-processor pipeline needs. See K=2 docstring +
/// `verify_pipeline_helper` for context.
pub fn step_verify_k3(
    model: &dyn Model,
    a: &mut ActiveSeq,
    drafts: &[u32],
    num_drafts: usize,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
    dflash_verify_raw_argmax: bool,
) {
    if let Err(e) = model.sync_secondary() {
        tracing::error!("sync_secondary: {e:#}");
        a.finished = true;
        return;
    }

    // EP: broadcast verify K=3 command + 3 tokens so worker runs decode_verify_graphed_k3 in lockstep.
    let tokens_k3 = [a.last_token, drafts[0], drafts[1]];
    if let Err(e) = model.ep_broadcast_cmd_for_seq(a.seq.slot_idx as u32, 0xFFFFFFF3) {
        tracing::error!("EP broadcast verify_k3 cmd: {e:#}");
        a.finished = true;
        return;
    }
    for &t in &tokens_k3 {
        if let Err(e) = model.ep_broadcast_cmd(t) {
            tracing::error!("EP broadcast verify_k3 token: {e:#}");
            a.finished = true;
            return;
        }
    }

    let t_verify = Instant::now();
    // Fused single-sweep path: DFlash only AND single-rank only. Under EP
    // (multi-rank) the worker ranks dispatch `decode_verify_graphed_k3` on the
    // broadcast cmd above, so the master MUST run the same method to stay in
    // NCCL lockstep — the fused forward is not EP-coherent. The MTP path
    // (non-raw-argmax) also stays on the legacy graphed verify unchanged.
    let result_vec: Vec<u32> = if dflash_verify_raw_argmax && !model.is_ep() {
        // Fused path: single M=3 forward, DFlash hidden captured at row 0.
        match model.decode_and_verify_fused(&tokens_k3, &mut a.seq, 0) {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("decode_and_verify_fused (k3): {e:#}");
                a.finished = true;
                return;
            }
        }
    } else {
        match model.decode_verify_graphed_k3(&tokens_k3, &mut a.seq, 0) {
            Ok(r) => r.to_vec(),
            Err(e) => {
                tracing::error!("decode_verify_graphed_k3: {e:#}");
                a.finished = true;
                return;
            }
        }
    };
    let verify_us = t_verify.elapsed().as_micros();
    a.last_token_time = Instant::now();
    let (v0_argmax, v1_argmax, v2_argmax) = (result_vec[0], result_vec[1], result_vec[2]);

    let (v0, v1, v2) = if dflash_verify_raw_argmax
        && !crate::scheduler::verify_pipeline_helper::dflash_masked_verify_enabled()
    {
        // DFlash drafter proposes on raw argmax; verify on the SAME (GOLD)
        // basis so verifier/drafter judge identically. No rep_pen/DRY here.
        (v0_argmax, v1_argmax, v2_argmax)
    } else {
        // MTP path: full pre-sample pipeline (rep_pen + DRY) unchanged.
        let processed = crate::scheduler::verify_pipeline_helper::verify_pick_all_with_pipeline(
            model,
            &[v0_argmax, v1_argmax, v2_argmax],
            a,
            verify_ctx,
        );
        (
            processed.first().copied().unwrap_or(v0_argmax),
            processed.get(1).copied().unwrap_or(v1_argmax),
            processed.get(2).copied().unwrap_or(v2_argmax),
        )
    };

    let num_accepted = if drafts[0] != v0 {
        0
    } else if drafts[1] != v1 {
        1
    } else {
        2
    };

    // Unconditional per-position draft match — scored BEFORE the accept chain
    // short-circuits, so position 2 is measured on every step, not only on the
    // steps where position 1 happened to succeed.
    k3_record_positional(drafts[0] == v0, drafts[1] == v1, a.seq.seq_len);

    // ATLAS_MTP_REFEED_ACCEPTED: ring the TARGET's true hidden for every
    // accepted position so the next propose's catch-up feed can rebuild the
    // drafter rows that `after_verify` is about to drop.
    //
    // Label convention (must match the serial-decode hook in scheduler/mod.rs,
    // which calls `save_hidden_for_catchup(0, seq.seq_len - 1)` after
    // appending): label n holds the hidden that PRODUCED the token at
    // position n, i.e. hidden_{n-1} in the drafter's pair-key space.
    // Here `a.seq.seq_len` has already been advanced by k=3 inside verify, so
    // the pre-verify base is `L = a.seq.seq_len - 3`; verify row t is the
    // forward pass at position L+t and its hidden produced the token at
    // L+t+1. Ring row t under label L+t+1 for t in 0..=num_accepted.
    //
    // The bound is INCLUSIVE, and both ends of it are load-bearing:
    //
    //  * rows 0..=num_accepted-1 are the repair itself. Drafts 1.. of the
    //    previous propose wrote pair keys L..L+num_accepted-2 with the
    //    DRAFTER's own hidden; the truth for pair key L+j is hidden_{L+j} =
    //    verify row j, which under the label convention is label L+j+1.
    //  * row `num_accepted` closes the ring. A verify step advances the
    //    sequence by `1 + num_accepted`, so the next step's base is
    //    L' = L + num_accepted + 1 and its first label is L' + 1. Stopping at
    //    `num_accepted - 1` leaves label L + num_accepted + 1 = L' unwritten
    //    EVERY step, and `save_hidden_for_catchup_dispatch` resets its
    //    contiguous (start, count) window on any non-contiguous append — so a
    //    single hole per step collapses coverage. Measured with the exclusive
    //    bound: 458 feeds vs 231 "gap of 1 pairs outside ring coverage".
    //    That row is not wasted either: label L' = the next propose position,
    //    whose content is hidden_{L'-1} = `mtp_hidden_save` — the row the
    //    NEXT step's full-reject case needs.
    //
    // Row `num_accepted` is always in range (K=3 has rows 0..=2 and
    // num_accepted <= 2), and it is always a COMMITTED position: on reject the
    // step still commits v0, so row 0 under label L+1 is the correct single
    // write. Hence no `num_accepted > 0` guard.
    //
    // `mtp_refeed_shift` deliberately perturbs the label; it is a
    // mapping-validation hatch and must stay 0 in any production leg.
    if spark_model::speculative::mtp_refeed_accepted_enabled() {
        let base = a.seq.seq_len.saturating_sub(3);
        let shift = spark_model::speculative::mtp_refeed_shift();
        for t in 0..=num_accepted {
            let label = ((base + t + 1) as isize + shift).max(0) as usize;
            if let Err(e) = model.save_hidden_for_catchup(t, label) {
                tracing::debug!("save_hidden_for_catchup(K=3, t={t}): {e:#} — degrading");
                break;
            }
        }
    }

    // Extract logprobs from verify logits buffer (K=3 positions) when requested.
    let verify_lps = if let Some(top_logprobs) = a.top_logprobs {
        extract_verify_logprobs(model, &[v0, v1, v2], top_logprobs)
    } else {
        Vec::new()
    };

    // EP: always broadcast num_accepted to worker (prevents deadlock on EOS).
    if let Err(e) = model.ep_broadcast_cmd(num_accepted as u32) {
        tracing::error!("EP broadcast verify_k3 result: {e:#}");
        a.finished = true;
        return;
    }

    // Per-verify trace at debug — fires every 1-3 output tokens during
    // spec-decode and spams Docker logs at info level. Power-user
    // diagnostics: `RUST_LOG=spark::scheduler::verify_k3_step=debug`.
    tracing::debug!(
        "K3 verify: tokens=[{},{},{}] → v=[{v0},{v1},{v2}] drafts=[{},{}] accepted={num_accepted} seq_len={}",
        tokens_k3[0],
        tokens_k3[1],
        tokens_k3[2],
        drafts[0],
        drafts[1],
        a.seq.seq_len
    );

    if num_accepted == 2 {
        emit_token(a, drafts[0], verify_lps.first().cloned());
        if !a.finished {
            emit_token(a, drafts[1], verify_lps.get(1).cloned());
        }
        if !a.finished {
            emit_token(a, v2, verify_lps.get(2).cloned());
        }
        if a.finished {
            return;
        }
        a.last_token = v2;

        // Item #2 (STree-style in-place K=3 verify commit). Full accept
        // (num_accepted=k=3): the verify kernel already wrote the canonical
        // h_state, so the commit is a no-op.
        if let Err(e) = model.commit_accepted_prefix(&mut a.seq, 3, 3) {
            // SSM state is no longer trustworthy — terminate, do not continue.
            tracing::error!("commit_accepted_prefix (K=3 accept-3): {e:#}");
            a.finished = true;
            return;
        }
        if let Err(e) = model.save_hidden_for_mtp(2, 0) {
            tracing::error!("save_hidden_for_mtp(2): {e:#}");
            return;
        }
        if let Err(e) = model.trim_proposer_state(&mut a.seq, 2, 0) {
            tracing::error!("trim_proposer_state: {e:#}");
        }
        let t_propose = Instant::now();
        let _mtp_grammar_mask = mtp_grammar_mask_for(a);
        match model.run_mtp_propose_multi(
            v2,
            a.seq.seq_len,
            crate::scheduler::spec_step::effective_drafts_under_grammar(a, num_drafts),
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
        tracing::debug!(
            "K3 ACCEPT-2: verify={verify_us}μs propose={propose_us}μs seq_len={}",
            a.seq.seq_len
        );
        k3_record_outcome(2, a.seq.seq_len);
    } else if num_accepted == 1 {
        a.seq.seq_len -= 1;
        a.seq.tokens.pop();
        if let Err(e) = model.trim_proposer_state(&mut a.seq, 1, 0) {
            tracing::error!("trim_proposer_state: {e:#}");
        }
        // Item #2 (STree-style in-place K=3 verify commit). Partial accept
        // (num_accepted=2 < k=3): rewind live h_state to intermediate[1].
        if let Err(e) = model.commit_accepted_prefix(&mut a.seq, 2, 3) {
            tracing::error!("commit_accepted_prefix (K=3 accept-2): {e:#}");
            a.finished = true;
            return;
        }
        emit_token(a, drafts[0], verify_lps.first().cloned());
        if !a.finished {
            emit_token(a, v1, verify_lps.get(1).cloned());
        }
        if a.finished {
            return;
        }
        a.last_token = v1;
        if let Err(e) = model.save_hidden_for_mtp(1, 0) {
            tracing::error!("save_hidden_for_mtp(1): {e:#}");
            return;
        }
        let t_propose = Instant::now();
        let _mtp_grammar_mask = mtp_grammar_mask_for(a);
        match model.run_mtp_propose_multi(
            v1,
            a.seq.seq_len,
            crate::scheduler::spec_step::effective_drafts_under_grammar(a, num_drafts),
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
        tracing::debug!(
            "K3 ACCEPT-1: verify={verify_us}μs propose={propose_us}μs seq_len={}",
            a.seq.seq_len
        );
        k3_record_outcome(1, a.seq.seq_len);
    } else {
        a.seq.seq_len -= 2;
        a.seq.tokens.pop();
        a.seq.tokens.pop();
        if let Err(e) = model.trim_proposer_state(&mut a.seq, 0, 0) {
            tracing::error!("trim_proposer_state: {e:#}");
        }
        // Item #2 (STree-style in-place K=3 verify commit). Partial accept
        // (num_accepted=1 < k=3): rewind live h_state to intermediate[0].
        if let Err(e) = model.commit_accepted_prefix(&mut a.seq, 1, 3) {
            tracing::error!("commit_accepted_prefix (K=3 accept-1): {e:#}");
            a.finished = true;
            return;
        }
        emit_token(a, v0, verify_lps.first().cloned());
        if a.finished {
            return;
        }
        a.last_token = v0;
        if let Err(e) = model.save_hidden_for_mtp(0, 0) {
            tracing::error!("save_hidden_for_mtp(0): {e:#}");
            return;
        }
        let t_propose = Instant::now();
        let _mtp_grammar_mask = mtp_grammar_mask_for(a);
        match model.run_mtp_propose_multi(
            v0,
            a.seq.seq_len,
            crate::scheduler::spec_step::effective_drafts_under_grammar(a, num_drafts),
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
        tracing::debug!(
            "K3 REJECT: verify={verify_us}μs propose={propose_us}μs seq_len={}",
            a.seq.seq_len
        );
        k3_record_outcome(0, a.seq.seq_len);
    }
}
