// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! The K=N verify step for the MTP-shaped wide verify (#1060): one verified
//! token plus N-1 drafts through the model's width-generic verify entry, then
//! the same accept, commit, rollback and re-propose as the K=3 and K=4 steps
//! (`verify_mtp_wide::finish`). Reached from `mtp_step` when a non-DFlash
//! sequence holds four or more drafts: lookup drafts at `ATLAS_LOOKUP_WIDTH`,
//! or an MTP head drafting past three.
//!
//! Under expert parallelism the worker ranks know the K=3 and K=4 commands
//! only, so this step narrows to the K=4 path there rather than desynchronise
//! the ranks.

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
    if model.is_ep() {
        super::verify_k4_step::step_verify_k4(
            model,
            a,
            sched,
            &drafts[..3],
            num_drafts,
            verify_ctx,
            false,
        );
        return;
    }
    if let Err(e) = model.sync_secondary() {
        tracing::error!("sync_secondary: {e:#}");
        a.finished = true;
        return;
    }
    let mut tokens = Vec::with_capacity(drafts.len() + 1);
    tokens.push(a.last_token);
    tokens.extend_from_slice(drafts);
    let k = tokens.len();
    let t_verify = Instant::now();
    let rows = match model.decode_verify_graphed_kn(&tokens, &mut a.seq, 0) {
        Ok(r) => r,
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
    super::verify_mtp_wide::finish(model, a, sched, drafts, num_drafts, verify_ctx);
}
