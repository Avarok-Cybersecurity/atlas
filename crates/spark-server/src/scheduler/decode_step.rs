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
    let tokens: Vec<u32> = active.iter().map(|a| a.last_token).collect();

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

    let mut refs: Vec<&mut SequenceState> = active.iter_mut().map(|a| &mut a.seq).collect();

    let logits = match model.decode_batch(&tokens, &mut refs, 0) {
        Ok(l) => l,
        Err(e) => {
            tracing::error!("decode_batch error: {e:#}");
            for mut a in active.drain(..) {
                send_error(model, &mut a, &format!("{e:#}"));
            }
            return;
        }
    };

    // Ctx-holes fix (ATLAS_DFLASH_SERIAL_APPEND=1): think-gated stretches
    // route HERE (mod.rs sends `inside_thinking` seqs to step_decode_only,
    // never the mtp bootstrap), so their captured target hiddens were
    // overwritten and permanently lost — the dominant ctx hole: a 270-token
    // think stretch leaves the drafter conditioned on the prompt alone
    // (observed GAP≈290 at first propose, accept ≤6%). Append each decoded
    // token's capture. n==1 only: `try_dflash_capture` stores row 0, which
    // is ambiguous in a multi-seq batch (fine here — DFlash runs
    // --max-batch-size 1).
    if n == 1 {
        if crate::scheduler::adaptive_spec::unified_ctx_enabled() {
            // Unified ctx commit: serial token at RoPE position seq_len-1
            // (decode() advanced seq_len past the token just processed).
            let base_pos = active[0].seq.seq_len.saturating_sub(1);
            if let Err(e) = model.commit_ctx(&mut active[0].seq, 1, base_pos) {
                tracing::error!("commit_ctx (decode_only serial): {e:#}");
            }
        } else if crate::scheduler::adaptive_spec::serial_append_enabled()
            && let Err(e) = model.dflash_serial_ctx_append(&mut active[0].seq)
        {
            tracing::error!("dflash_serial_ctx_append (decode_only): {e:#}");
        }
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
    );
}
