// SPDX-License-Identifier: AGPL-3.0-only

//! K=4 verify step.

use super::*;

#[path = "verify_k4_step/stats.rs"]
pub(super) mod stats;
use stats::k4_record_positional;

/// K=4 verify: [last_token, draft1, draft2, draft3] → [v0, v1, v2, v3].
/// Four outcomes: accept 0, 1, 2, or 3 drafts.
///
/// `verify_ctx` plumbs special-token IDs to the pre-sample pipeline.
/// See K=2 docstring + `verify_pipeline_helper`.
pub fn step_verify_k4(
    model: &dyn Model,
    a: &mut ActiveSeq,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    drafts: &[u32],
    num_drafts: usize,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
    dflash_verify_raw_argmax: bool,
) {
    // `ATLAS_MTP_TIMING=1` summary for the K=4 path.
    //
    // The per-phase `record()` calls already fire for K=4 because the picks
    // route through `verify_pipeline_helper`, but NOTHING called `step_done`
    // here — that lived only in `verify_k2_step`. So with `--num-drafts 3`
    // (K=4, the shipped config) the accumulators filled and the summary was
    // never emitted: a probe that generated ~1800 tokens produced zero timing
    // lines. This closes that hole.
    //
    // A Drop guard rather than hand-placed calls: this function has four accept
    // branches and several early error returns, so an explicit call per tail
    // would be one refactor away from silently drifting out of date again.
    let _step_timer = crate::scheduler::mtp_timing::StepTimer::new(&sched.timing, a.seq.seq_len);

    if let Err(e) = model.sync_secondary() {
        tracing::error!("sync_secondary: {e:#}");
        a.finished = true;
        return;
    }

    // Captured before the verify/emit paths advance seq_len (shadow top-k
    // join key; see SHADOW_TGT below).
    let shadow_base = a.seq.seq_len;

    let tokens_k4 = [a.last_token, drafts[0], drafts[1], drafts[2]];

    // EP: broadcast verify K=4 command + 4 tokens.
    if let Err(e) = model.ep_broadcast_cmd_for_seq(a.seq.slot_idx as u32, 0xFFFFFFF4) {
        tracing::error!("EP broadcast verify_k4 cmd: {e:#}");
        a.finished = true;
        return;
    }
    for &t in &tokens_k4 {
        if let Err(e) = model.ep_broadcast_cmd(t) {
            tracing::error!("EP broadcast verify_k4 token: {e:#}");
            a.finished = true;
            return;
        }
    }

    let t_verify = Instant::now();
    // Fused single-sweep path: DFlash only AND single-rank only. Under EP
    // (multi-rank) the worker ranks dispatch `decode_verify_graphed_k4` on the
    // broadcast cmd above, so the master MUST run the same method to stay in
    // NCCL lockstep — the fused forward is not EP-coherent. The MTP path
    // (non-raw-argmax) also stays on the legacy graphed verify unchanged.
    let result_vec: Vec<u32> = if dflash_verify_raw_argmax && !model.is_ep() {
        // Fused path: single M=4 forward, DFlash hidden captured at row 0.
        match model.decode_and_verify_fused(&tokens_k4, &mut a.seq, 0) {
            Ok(r) => r,
            Err(e) => {
                tracing::error!("decode_and_verify_fused (k4): {e:#}");
                a.finished = true;
                return;
            }
        }
    } else {
        match model.decode_verify_graphed_k4(&tokens_k4, &mut a.seq, 0) {
            Ok(r) => r.to_vec(),
            Err(e) => {
                tracing::error!("decode_verify_graphed_k4: {e:#}");
                a.finished = true;
                return;
            }
        }
    };
    let verify_us = t_verify.elapsed().as_micros();
    a.last_token_time = Instant::now();
    let (v0_argmax, v1_argmax, v2_argmax, v3_argmax) =
        (result_vec[0], result_vec[1], result_vec[2], result_vec[3]);

    let (v0, v1, v2, v3) = if dflash_verify_raw_argmax && !sched.levers.dflash_masked_verify {
        // DFlash drafter proposes on raw argmax; verify on the SAME (GOLD)
        // basis so verifier/drafter judge identically. No rep_pen/DRY here.
        (v0_argmax, v1_argmax, v2_argmax, v3_argmax)
    } else {
        // MTP path: full pre-sample pipeline (rep_pen + DRY) unchanged.
        let processed = crate::scheduler::verify_pipeline_helper::verify_pick_all_with_pipeline(
            model,
            &[v0_argmax, v1_argmax, v2_argmax, v3_argmax],
            a,
            verify_ctx,
            0,
            Some(drafts),
        );
        (
            processed.first().copied().unwrap_or(v0_argmax),
            processed.get(1).copied().unwrap_or(v1_argmax),
            processed.get(2).copied().unwrap_or(v2_argmax),
            processed.get(3).copied().unwrap_or(v3_argmax),
        )
    };

    let num_accepted = if drafts[0] != v0 {
        0
    } else if drafts[1] != v1 {
        1
    } else if drafts[2] != v2 {
        2
    } else {
        3
    };

    // Shadow top-k target line (ATLAS_MTP_SHADOW_TOPK): joins offline with
    // the drafter's SHADOW_TOPK lines — draft i (drafter pos base+i) vs v_i.
    if sched.levers.shadow_topk > 0 {
        // raw= is the argmax BEFORE verify_pick_all_with_pipeline (rep_pen +
        // DRY + presence). raw!=v at a position means the PIPELINE flipped the
        // pick; raw==v with a wrong token means the verify FORWARD itself
        // diverges from sequential decode. This is the discriminator for the
        // qwen3.8 temp-0 spec-decode divergence (memory: video-fidelity
        // video-before-image loses ", yellow" to an <|im_end|> pick).
        tracing::info!(
            "SHADOW_TGT base={shadow_base} raw=[{v0_argmax},{v1_argmax},{v2_argmax},{v3_argmax}] v=[{v0},{v1},{v2},{v3}] drafts=[{},{},{}]",
            drafts[0],
            drafts[1],
            drafts[2],
        );
    }

    // Unconditional per-position draft match — scored BEFORE the accept chain
    // short-circuits, so positions 2 and 3 are measured on every step.
    k4_record_positional(
        sched,
        drafts[0] == v0,
        drafts[1] == v1,
        drafts[2] == v2,
        a.seq.seq_len,
    );
    // Width-attributed accept telemetry (ATLAS_MTP_ACCEPT_DEBUG): this is the
    // SINGLE-sequence step, i.e. the n=1 row of the same table the batched
    // step fills for n>=2.
    crate::scheduler::mtp_accept_debug::record(1, 3, drafts[0] == v0, num_accepted);

    // ATLAS_MTP_REFEED_ACCEPTED: same contract as `verify_k3_step` — ring the
    // TARGET's true hidden for verify rows 0..=num_accepted under labels
    // L+1..=L+num_accepted+1 (L = the pre-verify seq_len = seq_len - 4 here).
    // `after_verify`'s extra trim (`mtp_rows_to_trim`) is K-agnostic, so this
    // MUST exist on every width the scheduler can dispatch, or at nd=3 the
    // drafter would lose the accepted rows with nothing rebuilding them.
    if spark_model::speculative::mtp_refeed_accepted_enabled() {
        let base = a.seq.seq_len.saturating_sub(4);
        let shift = spark_model::speculative::mtp_refeed_shift();
        for t in 0..=num_accepted {
            let label = ((base + t + 1) as isize + shift).max(0) as usize;
            if let Err(e) = model.save_hidden_for_catchup(t, label) {
                tracing::debug!("save_hidden_for_catchup(K=4, t={t}): {e:#} — degrading");
                break;
            }
        }
    }

    // Extract logprobs from verify logits buffer (K=4 positions) when requested.
    let verify_lps = if let Some(top_logprobs) = a.top_logprobs {
        extract_verify_logprobs(model, &[v0, v1, v2, v3], top_logprobs, 0)
    } else {
        Vec::new()
    };

    // EP: broadcast num_accepted to worker.
    if let Err(e) = model.ep_broadcast_cmd(num_accepted as u32) {
        tracing::error!("EP broadcast verify_k4 result: {e:#}");
        a.finished = true;
        return;
    }

    // Per-verify trace at debug — fires every 1-4 output tokens during
    // spec-decode and spams Docker logs at info level. Power-user
    // diagnostics: `RUST_LOG=spark::scheduler::verify_k4_step=debug`.
    tracing::debug!(
        "K4 verify: tokens=[{},{},{},{}] → v=[{v0},{v1},{v2},{v3}] drafts=[{},{},{}] accepted={num_accepted} seq_len={}",
        tokens_k4[0],
        tokens_k4[1],
        tokens_k4[2],
        tokens_k4[3],
        drafts[0],
        drafts[1],
        drafts[2],
        a.seq.seq_len
    );

    // Accept/rewind/emit/re-propose: the four verdict branches live in
    // `verify_k4_verdict.rs` (E9 extraction, behavior-identical), shared
    // with the batched multi-seq path. `VerifyRow` reads the accepted
    // hidden off the live verify forward exactly as before.
    k4_apply_verdict(
        model,
        a,
        sched,
        drafts,
        &[v0, v1, v2, v3],
        verify_lps,
        num_drafts,
        num_accepted,
        K4Hidden::VerifyRow,
        verify_us,
    );
}
