// SPDX-License-Identifier: AGPL-3.0-only
// Portions derived from Cinference (https://github.com/satellitedown/cinference, commit 383e5db), itself derived from NInfer by Neroued and contributors (https://github.com/Neroued/ninfer), Apache License 2.0. Source: tools/convert/proposal.py:44-104 (add_proposal: gather the shortlisted rows of the output head into the proposal head). Changes: translated from Python to Rust, the gather done on the device in chunks into Atlas's NVFP4 draft head with the full head's global scale, by Atlas contributors.

//! Responsibility: build the frequency-ranked draft-only NVFP4 LM head
//! (`mtp_vocab_rank = "freq"`) and register its row -> token-id map.
//!
//! Split from `vocab_rank.rs` (which holds the table, the selection and the
//! checks) for the 500-line cap.

use std::sync::Arc;

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend, KernelHandle};
use spark_runtime::kernel_args::KernelLaunch;

use super::vocab_rank::{
    CHUNK_ROWS, REMAP_KERNEL, REMAP_MODULE, RankMode, RankedHead, membership, register,
    select_rows, shipped, table_applies, tokenizer_fingerprint,
};
use crate::weight_map::{DenseWeight, QuantizedWeight};

/// Build the draft-only NVFP4 head from the ranked rows, or `Ok(None)` when
/// the row is `id`, the table does not apply to this model, or the remap
/// kernel is not in this build (each with one log line; the caller then builds
/// the id-ranked head exactly as before).
///
/// The global scale is taken over the WHOLE `vocab x hidden` head, as the
/// id-ranked head's is, and the quantizer is one block per row, so ranked row
/// r is byte-identical to row `row_to_id[r]` of the full quantization.
#[allow(clippy::too_many_arguments)]
pub(crate) fn build_ranked_draft_head(
    lm_head: &DenseWeight,
    rows: usize,
    vocab: usize,
    hidden: usize,
    gpu: &dyn GpuBackend,
    absmax_k: KernelHandle,
    quantize_k: KernelHandle,
    stream: u64,
) -> Result<Option<QuantizedWeight>> {
    if RankMode::resolved() != RankMode::Freq {
        return Ok(None);
    }
    let Some(table) = shipped() else {
        return Ok(None);
    };
    if let Err(why) = table_applies(table, vocab, tokenizer_fingerprint()) {
        tracing::warn!(
            "mtp_vocab_rank=freq not applied: {why}; the draft head keeps the first-N ids"
        );
        return Ok(None);
    }
    let remap_k = crate::layers::try_target_kernel(gpu, REMAP_MODULE, REMAP_KERNEL);
    let gather_k = crate::layers::try_kernel(gpu, "embed_from_argmax", "batched_embed");
    if remap_k.0 == 0 || gather_k.0 == 0 {
        tracing::warn!(
            "mtp_vocab_rank=freq not applied: kernel {REMAP_MODULE}::{REMAP_KERNEL} or \
             embed_from_argmax::batched_embed is not in this build; the draft head keeps \
             the first-N ids"
        );
        return Ok(None);
    }
    anyhow::ensure!(
        hidden.is_multiple_of(16),
        "ranked draft head: hidden {hidden} is not a multiple of the NVFP4 group (16)"
    );
    let row_to_id = select_rows(&table.ids, vocab, rows, &table.forced);
    let rows = row_to_id.len();

    // Phase 1: the global absmax over the full head (quantize_to_nvfp4_rows's).
    let total = vocab * hidden;
    let max_buf = gpu.alloc(4)?;
    gpu.memset(max_buf, 0, 4)?;
    let grid1 = (total / 256).clamp(1, 1024) as u32;
    KernelLaunch::new(gpu, absmax_k)
        .grid([grid1, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(lm_head.weight)
        .arg_ptr(max_buf)
        .arg_u32(total as u32)
        .launch(stream)?;
    gpu.synchronize(stream)?;
    let mut max_bytes = [0u8; 4];
    gpu.copy_d2h(max_buf, &mut max_bytes)?;
    gpu.free(max_buf)?;
    let global_max = f32::from_le_bytes(max_bytes);
    let scale2 = if global_max > 0.0 {
        global_max / (6.0 * 448.0)
    } else {
        1.0
    };

    // Phase 2: gather CHUNK_ROWS ranked rows into BF16 staging, quantize them
    // into their place in the packed head; repeat.
    let ids_bytes: Vec<u8> = row_to_id.iter().flat_map(|i| i.to_le_bytes()).collect();
    let row_to_id_dev = gpu.alloc(rows * 4)?;
    gpu.copy_h2d(&ids_bytes, row_to_id_dev)?;
    let packed = gpu.alloc(rows * hidden / 2)?;
    let scales = gpu.alloc(rows * hidden / 16)?;
    let chunk = CHUNK_ROWS.min(rows).max(1);
    let staging = gpu.alloc(chunk * hidden * 2)?;
    let mut r0 = 0usize;
    while r0 < rows {
        let m = chunk.min(rows - r0);
        crate::layers::ops::batched_embed(
            gpu,
            gather_k,
            row_to_id_dev.offset(r0 * 4),
            lm_head.weight,
            staging,
            m as u32,
            hidden as u32,
            stream,
        )?;
        KernelLaunch::new(gpu, quantize_k)
            .grid([m as u32, 1, 1])
            .block([256, 1, 1])
            .arg_ptr(staging)
            .arg_ptr(packed.offset(r0 * hidden / 2))
            .arg_ptr(scales.offset(r0 * hidden / 16))
            .arg_f32(scale2)
            .arg_u32(m as u32)
            .arg_u32(hidden as u32)
            .launch(stream)?;
        r0 += m;
    }
    gpu.synchronize(stream)?;
    gpu.free(staging)?;

    register(
        packed,
        Arc::new(RankedHead {
            in_head: membership(&row_to_id, vocab),
            row_to_id,
            row_to_id_dev,
            remap_k,
        }),
    );
    tracing::info!(
        "Draft-only NVFP4 LM head built from the {rows} most frequent tokens \
         (mtp_vocab_rank=freq, vocab={vocab}, Cinference-derived table, {} forced ids)",
        table.forced.len()
    );
    Ok(Some(QuantizedWeight {
        weight: packed,
        weight_scale: scales,
        weight_scale_2: scale2,
        input_scale: DevicePtr::NULL,
        weight_scale_2_vec: DevicePtr::NULL,
    }))
}
