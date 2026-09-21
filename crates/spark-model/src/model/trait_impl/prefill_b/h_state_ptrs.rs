// SPDX-License-Identifier: AGPL-3.0-only

//! Q12 Path B: per-layer h_state_ptrs staging for batched SSM/GDN.
//!
//! Each `SsmLayerState::h_state` is a per-stream-per-layer GPU
//! allocation. The batched GDN kernels take a `float* const* h_state_ptrs`
//! parameter — a device array of per-stream h_state device pointers
//! indexed by `b = blockIdx.y`. This module stages that array.
//!
//! Call site: `prefill_ssm_batched_layer` calls this once per SSM layer
//! within the outer layer loop. The returned `DevicePtr` is the device
//! array that the batched GDN op consumes.
//!
//! Storage strategy: write into a dedicated slot of the model's scratch
//! buffer, after the BatchedAttnMetadata layout (which uses the front of
//! scratch). The h_state_ptrs array is small (`batch_size × 8` bytes ≤ 64 B
//! for typical N≤8) so this is cheap.
//!
//! ── Stage-3 f16-SIZED pool (`--ssm-h-dtype f16-pool`) ──
//!
//! The batched GDN kernels take a `float* const*` table, so the widen/narrow
//! pair the single-stream ladder wraps around itself (see
//! `layers::qwen3_ssm::ssm_h_fp16`) has to happen HERE instead — the table
//! must point at each sequence's FP32 staging blob, not at its 2-byte-sized
//! pool slot. [`TransformerModel::stage_h_state_ptrs`] widens as it stages
//! and [`TransformerModel::narrow_h_state_stages`] is the matching epilogue;
//! they are a PAIR and the caller must run the second one on every path the
//! table was actually consumed by. On an FP32-sized pool both are byte-for-
//! byte what they always were: the table carries slot pointers and the
//! epilogue moves nothing.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::super::types::TransformerModel;
use crate::layer::SsmLayerState;
use crate::traits::SequenceState;

impl TransformerModel {
    /// Stage `h_state_ptrs[batch_size]` device array at the given scratch
    /// offset and return the DevicePtr to the staged array.
    ///
    /// Each entry is the per-stream `SsmLayerState::h_state` device
    /// pointer for the given `layer_idx`. Streams whose
    /// `layer_states[layer_idx]` is not an `SsmLayerState` (e.g. dense
    /// FFN layers in a hybrid stack — shouldn't occur for SSM dispatch)
    /// are filled with `DevicePtr::NULL` and the caller is responsible
    /// for refusing to dispatch in that case.
    pub(in crate::model) fn stage_h_state_ptrs(
        &self,
        layer_idx: usize,
        seqs: &mut [&mut SequenceState],
        scratch_offset_bytes: usize,
        stream: u64,
    ) -> Result<DevicePtr> {
        let n = seqs.len();
        if n == 0 {
            anyhow::bail!("stage_h_state_ptrs called with zero streams");
        }
        let mut h_ptrs: Vec<u64> = Vec::with_capacity(n);
        for (i, seq) in seqs.iter_mut().enumerate() {
            let ssm_state = seq.layer_states[layer_idx]
                .as_any_mut()
                .downcast_mut::<SsmLayerState>()
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "stage_h_state_ptrs: stream {i} layer {layer_idx} \
                         is not an SsmLayerState (got non-SSM layer in \
                         SSM batched dispatch)"
                    )
                })?;
            // Stage-3: the table entry is the FP32 STAGING blob, widened
            // from the narrow slot right here. `None` (FP32-sized pool) keeps
            // the historical behaviour exactly — the slot pointer itself,
            // and no conversion launched.
            match ssm_state.h_prefill_stage {
                None => h_ptrs.push(ssm_state.h_state.0),
                Some(stage) => {
                    if self.ssm_h_f16_to_f32_kernel.0 == 0 {
                        anyhow::bail!(
                            "--ssm-h-dtype f16-pool: ssm_h_dtype::ssm_h_state_f16_to_f32 did \
                             not resolve — refusing to point the batched GDN prefill kernels \
                             at 2-byte-sized pool slots they would write as FP32"
                        );
                    }
                    crate::layers::ops::ssm_h_state_f16_to_f32(
                        self.gpu.as_ref(),
                        self.ssm_h_f16_to_f32_kernel,
                        ssm_state.h_state,
                        stage,
                        (self.ssm_pool.h_bytes / 4) as u64,
                        stream,
                    )?;
                    h_ptrs.push(stage.0);
                }
            }
        }

        let dst = self.buffers.scratch().offset(scratch_offset_bytes);
        // SAFETY: `h_ptrs` was `with_capacity(n)` but is also FILLED to `n` —
        // the loop above iterates `seqs` (`n = seqs.len()`) and pushes exactly
        // once per iteration, and its only early exit is the `?` on the
        // downcast, which returns instead of reaching here. So
        // `h_ptrs.len() == n` and `n * size_of::<u64>()` covers only written
        // elements, never the uninitialised capacity tail.
        let bytes = unsafe {
            std::slice::from_raw_parts(h_ptrs.as_ptr() as *const u8, n * std::mem::size_of::<u64>())
        };
        self.gpu.copy_h2d_async(bytes, dst, stream)?;
        Ok(dst)
    }

    /// Epilogue of [`Self::stage_h_state_ptrs`]: narrow each stream's FP32
    /// staging blob back into its 2-byte-sized pool slot (stage-3 f16-SIZED
    /// pool). No-op — not one launch — on an FP32-sized pool.
    ///
    /// ★ Call it only when the batched GDN kernel actually RAN. Skipping it
    /// after a failure leaves every slot holding its pre-pass f16 value,
    /// which is the recoverable state; the VARLEN fallback loop must skip it
    /// too, because that loop routes through the single-stream ladder, which
    /// widens and narrows each sequence itself.
    pub(in crate::model) fn narrow_h_state_stages(
        &self,
        layer_idx: usize,
        seqs: &mut [&mut SequenceState],
        stream: u64,
    ) -> Result<()> {
        if self.ssm_pool.h_prefill_stage_pool.is_none() {
            return Ok(());
        }
        for (i, seq) in seqs.iter_mut().enumerate() {
            let ssm_state = seq.layer_states[layer_idx]
                .as_any_mut()
                .downcast_mut::<SsmLayerState>()
                .ok_or_else(|| {
                    anyhow::anyhow!(
                        "narrow_h_state_stages: stream {i} layer {layer_idx} is not an \
                         SsmLayerState"
                    )
                })?;
            let Some(stage) = ssm_state.h_prefill_stage else {
                anyhow::bail!(
                    "narrow_h_state_stages: stream {i} layer {layer_idx} has no FP32 staging \
                     blob under an f16-sized h pool — its slot cannot hold the FP32 the \
                     batched GDN kernel just wrote"
                );
            };
            if self.ssm_h_f32_to_f16_kernel.0 == 0 {
                anyhow::bail!(
                    "--ssm-h-dtype f16-pool: ssm_h_dtype::ssm_h_state_f32_to_f16 did not \
                     resolve — the batched prefill h-state cannot be narrowed back into its \
                     2-byte-sized slot"
                );
            }
            crate::layers::ops::ssm_h_state_f32_to_f16(
                self.gpu.as_ref(),
                self.ssm_h_f32_to_f16_kernel,
                stage,
                ssm_state.h_state,
                (self.ssm_pool.h_bytes / 4) as u64,
                stream,
            )?;
        }
        Ok(())
    }
}
