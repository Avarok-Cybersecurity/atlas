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
            h_ptrs.push(ssm_state.h_state.0);
        }

        let dst = self.buffers.scratch().offset(scratch_offset_bytes);
        let bytes = unsafe {
            std::slice::from_raw_parts(h_ptrs.as_ptr() as *const u8, n * std::mem::size_of::<u64>())
        };
        self.gpu.copy_h2d_async(bytes, dst, stream)?;
        Ok(dst)
    }
}
