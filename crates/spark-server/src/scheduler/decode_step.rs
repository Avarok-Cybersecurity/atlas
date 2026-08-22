// SPDX-License-Identifier: AGPL-3.0-only

//! step_decode_only: batched decode for active sequences (no MTP).

use super::*;

/// Decode-only step: batched decode for all active sequences (no MTP).
pub fn step_decode_only(
    model: &dyn Model,
    active: &mut Vec<ActiveSeq>,
    think_end_token: Option<u32>,
    think_start_token: Option<u32>,
    code_fence_token: Option<u32>,
    tool_call_start_token: Option<u32>,
    tool_call_end_token: Option<u32>,
    adaptive_sampling: bool,
    sched: &crate::scheduler::sched_ctx::SchedCtx,
    spill: Option<&mut KvSpillManager>,
    swapped: &mut Vec<SwappedSeq>,
    preempted: &mut Vec<PreemptedSeq>,
) {
    let t0 = std::time::Instant::now();
    let n = active.len();
    // Batched decode (CUDA-graph replay + batched-recurrent SSM) requires the
    // active sequences in SSM-pool-slot order, so batch position i maps to a
    // contiguous state address (pool_base + i*stride). The pool assigns
    // consecutive slots but the active list is in reverse-arrival order
    // ([7,6,..,0] for 8 seqs), which fails the contiguity check in
    // ssm_batched_recurrent.rs and the graph-capture slot==i assumption,
    // forcing the eager per-seq loop (no concurrency scaling). Sort ascending
    // by SSM slot (falling back to KV slot for non-SSM models) so the
    // contiguous-slot invariant holds and the batched paths engage. The whole
    // ActiveSeq is reordered, so the post-decode position->seq mapping stays
    // consistent.
    if n > 1 {
        active.sort_by_key(|a| a.seq.ssm_slot_idx().unwrap_or(a.seq.slot_idx));
    }

    // CONCURRENT-DECODE DIAG: per-step batch state (slot, seq_len, etc).
    // Demoted to debug after the 2026-04-22 stride+graph fixes shipped —
    // it was a hot per-decode log line that drowned production traces.
    // Re-enable with `RUST_LOG=spark_server::scheduler=debug`.
    if n > 1 && tracing::enabled!(tracing::Level::DEBUG) {
        let diag: Vec<String> = active
            .iter()
            .enumerate()
            .map(|(i, a)| {
                let bt0 = a.seq.block_table.first().copied().unwrap_or(u32::MAX);
                let btn = a.seq.block_table.len();
                format!(
                    "[{i}: slot={} seq_len={} bt={}/{} last={} out_n={}]",
                    a.seq.slot_idx,
                    a.seq.seq_len,
                    bt0,
                    btn,
                    a.last_token,
                    a.output_tokens.len(),
                )
            })
            .collect();
        tracing::debug!("CONC_DIAG n={n}: {}", diag.join(" "));
    }

    // EP broadcasts (seq_id preamble + cmd per active seq) are emitted
    // inside `decode_batch_dispatch` itself, interleaved with each per-seq
    // `decode()` call. Batching them up-front here would diverge the head's
    // comm-stream op order ([B,B,...,B,AR,AR,...]) from the worker's
    // ([B,AR,...,AR,B,AR,...,AR,...]) and deadlock NCCL — observed
    // empirically as a 51s broadcast timeout on the worker followed by
    // stale comm data reads. See decode_a2.rs for the full rationale.

    // Decode, PREEMPTING on KV exhaustion instead of failing the whole batch.
    //
    // Sequences grow one block per `block_size` tokens as they decode; when
    // admission overcommitted the pool (see `admission`), decode is where
    // the collision lands. `decode_batch_with_preemption` spills or requeues
    // ONE victim per retry — for later RESUME, never a kill — so the rest of
    // the batch makes progress and the victim's stream continues once blocks
    // free up. See the `preempt` module docs for the policy and the
    // measured C=128 evidence that motivated it.
    let Some(logits) =
        super::preempt::decode_batch_with_preemption(model, active, spill, swapped, preempted)
    else {
        return;
    };
    // Preemption may have shrunk the batch; `n` gates the n==1 paths below.
    let n = active.len();
    if n == 0 {
        return;
    }

    // Ctx-holes fix (ATLAS_DFLASH_SERIAL_APPEND=1): think-gated stretches
    // route HERE (mod.rs sends `inside_thinking` seqs to step_decode_only,
    // never the mtp bootstrap), so their captured target hiddens were
    // overwritten and permanently lost — the dominant ctx hole: a 270-token
    // think stretch leaves the drafter conditioned on the prompt alone
    // (observed GAP≈290 at first propose, accept ≤6%).
    //
    // Batched decode (n>1) captures ALL batch rows (`decode_multi_seq` layer
    // loop → `try_dflash_capture_all`), so seq i's per-layer hidden lives in
    // scratch row i — commit each seq from its own row. The old `n == 1`
    // gate silently dropped every batched-decode token's ctx row: the C>=2
    // accept-collapse root cause (84%→31%/22%; CTXLEN_PROBE GAP grew while
    // position advanced, and adaptive-spec suspension routed the starved seq
    // right back here — a self-reinforcing spiral).
    if sched.levers.dflash_unified_ctx {
        for (i, a) in active.iter_mut().enumerate() {
            // Unified ctx commit: serial token at RoPE position seq_len-1
            // (decode advanced seq_len past the token just processed).
            let base_pos = a.seq.seq_len.saturating_sub(1);
            if let Err(e) = model.commit_ctx(&mut a.seq, 1, base_pos, i) {
                tracing::error!("commit_ctx (decode_only, row {i}): {e:#}");
            }
        }
    } else if n == 1
        && sched.levers.dflash_serial_append
        && let Err(e) = model.dflash_serial_ctx_append(&mut active[0].seq)
    {
        tracing::error!("dflash_serial_ctx_append (decode_only): {e:#}");
    }

    process_decode_logits(
        model,
        active,
        logits,
        t0,
        think_end_token,
        think_start_token,
        code_fence_token,
        tool_call_start_token,
        tool_call_end_token,
        adaptive_sampling,
        sched,
    );
}
