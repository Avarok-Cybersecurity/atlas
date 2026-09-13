// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! The K=N verify step for the MTP-shaped wide verify (#1060): one verified
//! token plus N-1 drafts through the model's width-generic verify entry, then
//! the same accept, commit, rollback and re-propose as the K=3 and K=4 steps
//! (`verify_mtp_wide::finish`). Reached from `mtp_step` when a non-DFlash
//! sequence holds four or more drafts: lookup drafts at `ATLAS_LOOKUP_WIDTH`,
//! or an MTP head drafting past three.
//!
//! Under expert parallelism this broadcasts `EP_CMD_VERIFY_KN` — the
//! width-generic worker command — rather than narrowing to K=4. It used to
//! narrow, because the workers knew only the K=3/K=4 commands; that cap made
//! `ATLAS_LOOKUP_WIDTH` unreachable on a multi-rank serve, and proposing wide
//! while verifying K=4 discards the extra drafts every fire (measured on the
//! copy task at TP=2 x EP=2: width 7 proposed / K=4 verified ran 53.31 tok/s
//! against width 2's 61.88).
//!
//! The wire is `cmd, k, tokens[k]` and then, after the forward, `num_accepted`
//! — `k` first so both ranks agree on how many words are still coming.

use spark_model::traits::Model;

use super::types::ActiveSeq;
use std::time::Instant;

pub(super) fn step_verify_kn(
    model: &dyn Model,
    a: &mut ActiveSeq,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    drafts: &[u32],
    num_drafts: usize,
    verify_ctx: &crate::scheduler::logit_processors::LogitsContext,
) {
    if let Err(e) = model.sync_secondary() {
        tracing::error!("sync_secondary: {e:#}");
        a.finished = true;
        return;
    }
    let mut tokens = Vec::with_capacity(drafts.len() + 1);
    tokens.push(a.last_token);
    tokens.extend_from_slice(drafts);
    let k = tokens.len();

    // EP: width-generic verify (`EP_CMD_VERIFY_KN`). `k` goes out BEFORE the
    // tokens so this loop and the worker's receive loop are driven by the same
    // word — that is what makes a new EP path safe, and it is why this step no
    // longer narrows to K=4 here. Under EP the worker dispatches
    // `decode_verify_graphed_kn` on the command, so the head must run the same
    // method below to stay in NCCL lockstep.
    if model.is_ep() {
        if let Err(e) =
            model.ep_broadcast_cmd_for_seq(a.seq.slot_idx as u32, spark_model::speculative::EP_CMD_VERIFY_KN)
        {
            tracing::error!("EP broadcast verify_kn cmd: {e:#}");
            super::lifecycle::fail_sequence(a, format!("EP broadcast verify_kn cmd: {e:#}"));
            return;
        }
        if let Err(e) = model.ep_broadcast_cmd(k as u32) {
            tracing::error!("EP broadcast verify_kn width: {e:#}");
            super::lifecycle::fail_sequence(a, format!("EP broadcast verify_kn width: {e:#}"));
            return;
        }
        for &t in &tokens {
            if let Err(e) = model.ep_broadcast_cmd(t) {
                tracing::error!("EP broadcast verify_kn token: {e:#}");
                super::lifecycle::fail_sequence(a, format!("EP broadcast verify_kn token: {e:#}"));
                return;
            }
        }
    }

    let t_verify = Instant::now();
    let rows = match model.decode_verify_graphed_kn(&tokens, &mut a.seq, 0) {
        Ok(r) => r.to_vec(),
        Err(e) => {
            tracing::error!("decode_verify_graphed_kn (K={k}): {e:#}");
            a.finished = true;
            return;
        }
    };
    let verify_us = t_verify.elapsed().as_micros();
    a.last_token_time = Instant::now();
    if rows.len() != k {
        tracing::error!("decode_verify_graphed_kn returned {} rows, want {k}", rows.len());
        a.finished = true;
        return;
    }
    tracing::debug!("K{k} verify: {verify_us}us");
    // `gpu_argmax` is this branch's 7th parameter on `finish` (the fast-greedy
    // path reads the verify's argmax rows instead of copying K*vocab logits to
    // the host). K=3/K=4 pass their own `decode_verify_graphed_k*` rows; the
    // K=N rows are the same thing and are already length-checked against k.
    super::verify_mtp_wide::finish(model, a, sched, drafts, num_drafts, verify_ctx, &rows);
}
