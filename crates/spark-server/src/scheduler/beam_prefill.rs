// SPDX-License-Identifier: AGPL-3.0-only

//! Shared beam-hypothesis resolution for the two prefill entry points
//! (`prefill_a_step`, `prefill_b_step`).

use anyhow::Result;
use spark_model::traits::{BeamReq, Model, SequenceState};

/// Resolve a beam request's winning hypothesis: the scheduler's pre-computed
/// result when present (this request was fused with others into ONE batched
/// search by the co-dispatch pre-pass), else a per-request `generate_beam_batch`.
pub(super) fn resolve_beam_hyp(
    model: &dyn Model,
    precomputed: Option<Vec<u32>>,
    seq: &SequenceState,
    prompt_tokens: &[u32],
    max_tokens: usize,
) -> Result<Vec<u32>> {
    if let Some(h) = precomputed {
        return Ok(h);
    }
    let beam_req = BeamReq {
        prompt_tokens: prompt_tokens.to_vec(),
        src_lang_id: seq.src_lang_id,
        tgt_lang_id: seq.tgt_lang_id,
        adapter_slot: seq.adapter_slot,
        num_beams: seq.num_beams as usize,
        max_new: max_tokens,
        length_penalty: seq.length_penalty,
        early_stopping: seq.early_stopping,
    };
    Ok(model
        .generate_beam_batch(std::slice::from_ref(&beam_req))?
        .pop()
        .unwrap_or_default())
}
