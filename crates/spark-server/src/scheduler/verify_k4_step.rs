// SPDX-License-Identifier: AGPL-3.0-only

//! K=4 verify step.

use super::*;

// Periodic accept-distribution summary (P4, 2026-05-24). Mirrors K=3.
const K4_SUMMARY_PERIOD: u64 = 100;
static K4_ACCEPT_3: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static K4_ACCEPT_2: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static K4_ACCEPT_1: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
static K4_ACCEPT_0: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);

#[inline]
fn k4_record_outcome(num_accepted: usize, seq_len: usize) {
    let counter = match num_accepted {
        3 => &K4_ACCEPT_3,
        2 => &K4_ACCEPT_2,
        1 => &K4_ACCEPT_1,
        _ => &K4_ACCEPT_0,
    };
    counter.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let total = K4_ACCEPT_3.load(std::sync::atomic::Ordering::Relaxed)
        + K4_ACCEPT_2.load(std::sync::atomic::Ordering::Relaxed)
        + K4_ACCEPT_1.load(std::sync::atomic::Ordering::Relaxed)
        + K4_ACCEPT_0.load(std::sync::atomic::Ordering::Relaxed);
    if total >= K4_SUMMARY_PERIOD {
        let a3 = K4_ACCEPT_3.swap(0, std::sync::atomic::Ordering::Relaxed);
        let a2 = K4_ACCEPT_2.swap(0, std::sync::atomic::Ordering::Relaxed);
        let a1 = K4_ACCEPT_1.swap(0, std::sync::atomic::Ordering::Relaxed);
        let a0 = K4_ACCEPT_0.swap(0, std::sync::atomic::Ordering::Relaxed);
        let total = (a3 + a2 + a1 + a0).max(1);
        let mean = (3 * a3 + 2 * a2 + a1) as f64 / total as f64;
        tracing::info!(
            "K4 summary: {a3} accept-3 / {a2} accept-2 / {a1} accept-1 / {a0} reject in last {total} steps (mean accepted={mean:.2}) seq_len={seq_len}"
        );
    }
}

/// K=4 verify: [last_token, draft1, draft2, draft3] → [v0, v1, v2, v3].
/// Four outcomes: accept 0, 1, 2, or 3 drafts.
///
/// `verify_ctx` plumbs special-token IDs to the pre-sample pipeline.
/// See K=2 docstring + `verify_pipeline_helper`.
pub fn step_verify_k4(
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

    let (v0, v1, v2, v3) = if dflash_verify_raw_argmax
        && !crate::scheduler::verify_pipeline_helper::dflash_masked_verify_enabled()
    {
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

    // Extract logprobs from verify logits buffer (K=4 positions) when requested.
    let verify_lps = if let Some(top_logprobs) = a.top_logprobs {
        extract_verify_logprobs(model, &[v0, v1, v2, v3], top_logprobs)
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

    if num_accepted == 3 {
        emit_token(a, drafts[0], verify_lps.first().cloned());
        if !a.finished {
            emit_token(a, drafts[1], verify_lps.get(1).cloned());
        }
        if !a.finished {
            emit_token(a, drafts[2], verify_lps.get(2).cloned());
        }
        if !a.finished {
            emit_token(a, v3, verify_lps.get(3).cloned());
        }
        if a.finished {
            return;
        }
        a.last_token = v3;

        // F62/F63 (2026-04-27): SpecMamba commit. K=4 full accept.
        if let Err(e) = model.commit_verify_state_async(&mut a.seq, 4, 4) {
            tracing::error!("commit_verify_state_async (K=4 accept-4): {e:#}");
            return;
        }
        if let Err(e) = model.save_hidden_for_mtp(3, 0) {
            tracing::error!("save_hidden_for_mtp(3): {e:#}");
            return;
        }
        if let Err(e) = model.trim_proposer_state(&mut a.seq, 3, 0) {
            tracing::error!("trim_proposer_state: {e:#}");
        }
        let t_propose = Instant::now();
        let _mtp_grammar_mask = mtp_grammar_mask_for(a);
        match model.run_mtp_propose_multi(
            v3,
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
        tracing::debug!(
            "K4 ACCEPT-3: verify={verify_us}μs propose={propose_us}μs seq_len={}",
            a.seq.seq_len
        );
        k4_record_outcome(3, a.seq.seq_len);
    } else if num_accepted == 2 {
        a.seq.seq_len -= 1;
        a.seq.tokens.pop();
        if let Err(e) = model.trim_proposer_state(&mut a.seq, 2, 0) {
            tracing::error!("trim_proposer_state: {e:#}");
        }
        // F62/F63 (2026-04-27): K=4 partial accept (3 of 4).
        if let Err(e) = model.commit_verify_state_async(&mut a.seq, 3, 4) {
            tracing::error!("commit_verify_state_async (K=4 accept-3): {e:#}");
            a.finished = true;
            return;
        }
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
        if let Err(e) = model.save_hidden_for_mtp(2, 0) {
            tracing::error!("save_hidden_for_mtp(2): {e:#}");
            return;
        }
        let t_propose = Instant::now();
        let _mtp_grammar_mask = mtp_grammar_mask_for(a);
        match model.run_mtp_propose_multi(
            v2,
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
        tracing::debug!(
            "K4 ACCEPT-2: verify={verify_us}μs propose={propose_us}μs seq_len={}",
            a.seq.seq_len
        );
        k4_record_outcome(2, a.seq.seq_len);
    } else if num_accepted == 1 {
        a.seq.seq_len -= 2;
        a.seq.tokens.pop();
        a.seq.tokens.pop();
        if let Err(e) = model.trim_proposer_state(&mut a.seq, 1, 0) {
            tracing::error!("trim_proposer_state: {e:#}");
        }
        // F62/F63 (2026-04-27): K=4 partial accept (2 of 4).
        if let Err(e) = model.commit_verify_state_async(&mut a.seq, 2, 4) {
            tracing::error!("commit_verify_state_async (K=4 accept-2): {e:#}");
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
        tracing::debug!(
            "K4 ACCEPT-1: verify={verify_us}μs propose={propose_us}μs seq_len={}",
            a.seq.seq_len
        );
        k4_record_outcome(1, a.seq.seq_len);
    } else {
        a.seq.seq_len -= 3;
        a.seq.tokens.pop();
        a.seq.tokens.pop();
        a.seq.tokens.pop();
        if let Err(e) = model.trim_proposer_state(&mut a.seq, 0, 0) {
            tracing::error!("trim_proposer_state: {e:#}");
        }
        // F62/F63 (2026-04-27): K=4 partial accept (1 of 4).
        if let Err(e) = model.commit_verify_state_async(&mut a.seq, 1, 4) {
            tracing::error!("commit_verify_state_async (K=4 accept-1): {e:#}");
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
        tracing::debug!(
            "K4 REJECT: verify={verify_us}μs propose={propose_us}μs seq_len={}",
            a.seq.seq_len
        );
        k4_record_outcome(0, a.seq.seq_len);
    }
}
