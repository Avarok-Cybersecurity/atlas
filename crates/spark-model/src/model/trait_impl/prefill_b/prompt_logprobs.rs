// SPDX-License-Identifier: AGPL-3.0-only

//! Prompt-token logprob collection for legacy `/v1/completions`
//! `echo` + `logprobs` (loglikelihood scoring — lm-eval et al.).
//!
//! During prefill the `hidden` buffer holds one row per processed chunk
//! position, but only the LAST row is normally projected to logits.
//! When `seq.collect_prompt_logprobs` is set, this helper projects EVERY
//! prompt position through final-norm + LM head in ≤32-row batches
//! (`buffers.logits()` holds only `min(m, 32)` rows — see
//! `spark-runtime/src/buffers/sizes.rs`), copies each batch D2H, and
//! extracts `log P(tokens[i+1] | tokens[..=i])` plus top-k alternatives.
//!
//! MUST run inside `prefill_chunk_dispatch` BEFORE `prefill_b_finalize_last`:
//! it clobbers `buffers.logits()` and `buffers.norm_output()`, both of
//! which `finalize_last` re-derives fresh for the first sampled token.
//! Collecting requests bypass the prefix cache (see `prefix_lookup.rs`)
//! so every position is guaranteed a live hidden row.

use anyhow::{Result, ensure};

use super::super::super::types::TransformerModel;
use crate::layers::ops;
use crate::traits::{SequenceState, extract_bf16};

/// Rows projected per LM-head batch. `buffers.logits()` is sized for
/// `min(m, 32)` rows; every real serve config has m ≥ 32.
const BATCH_ROWS: usize = 32;

impl TransformerModel {
    /// Collect prompt-token logprobs for one prefill chunk. No-op unless
    /// `seq.collect_prompt_logprobs` is set. Appends to
    /// `seq.prompt_logprobs` (one entry per position scoring the NEXT
    /// prompt token; the final prompt position — whose target is the
    /// first generated token — is excluded).
    pub(in crate::model) fn collect_prompt_logprobs_chunk(
        &self,
        tokens: &[u32],
        seq: &mut SequenceState,
        chunk_start: usize,
        proc_start: usize,
        proc_count: usize,
        stream: u64,
    ) -> Result<()> {
        let Some(k) = seq.collect_prompt_logprobs else {
            return Ok(());
        };
        // Prefix bypass guarantees the whole chunk was computed: hidden
        // row r ↔ absolute prompt position chunk_start + r. A partial
        // proc range would silently misalign rows — fail fast instead.
        ensure!(
            proc_start == chunk_start,
            "prompt-logprob collection requires a full-recompute chunk \
             (proc_start {proc_start} != chunk_start {chunk_start}); \
             prefix-cache bypass failed"
        );

        let h = self.config.hidden_size;
        let v = self.config.vocab_size;
        let eps = self.config.rms_norm_eps as f32;
        let elem = 2usize; // BF16 hidden rows (matches finalize_last)
        let hidden = self.buffers.hidden_states();

        // Score positions chunk_start..min(chunk_end, prompt_len-1):
        // position p's logits predict tokens[p+1]; the last prompt
        // position's target is the first GENERATED token → excluded.
        let last_scored_excl = tokens.len().saturating_sub(1);
        let rows_to_score = proc_count.min(last_scored_excl.saturating_sub(chunk_start));
        if rows_to_score == 0 {
            return Ok(());
        }

        let mut host = vec![0u8; BATCH_ROWS * v * 2];
        let mut start = 0usize;
        while start < rows_to_score {
            let count = (rows_to_score - start).min(BATCH_ROWS);
            let batch_hidden = hidden.offset((start) * h * elem);
            let normed = self.buffers.norm_output();
            ops::rms_norm(
                self.gpu.as_ref(),
                self.rms_norm_kernel,
                batch_hidden,
                &self.final_norm,
                normed,
                count as u32,
                h as u32,
                eps,
                stream,
            )?;
            self.lm_head_batched(normed, count as u32, self.buffers.logits(), stream)?;
            self.gpu.synchronize(stream)?;
            let bytes = &mut host[..count * v * 2];
            self.gpu.copy_d2h(self.buffers.logits(), bytes)?;
            for j in 0..count {
                let target = tokens[chunk_start + start + j + 1];
                let row = &bytes[j * v * 2..(j + 1) * v * 2];
                seq.prompt_logprobs
                    .push(extract_bf16(row, target, k as usize, v));
            }
            start += count;
        }
        Ok(())
    }
}
