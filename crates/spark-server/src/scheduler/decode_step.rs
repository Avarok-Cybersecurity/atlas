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
    reflection_suppress_ids: &[u32],
    adaptive_sampling: bool,
) {
    let t0 = std::time::Instant::now();
    let n = active.len();
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
        reflection_suppress_ids,
        adaptive_sampling,
    );
}
