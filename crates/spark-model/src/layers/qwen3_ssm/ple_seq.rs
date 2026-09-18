// SPDX-License-Identifier: AGPL-3.0-only

//! The PLE per-sequence carry helper, split out of `trait_layer.rs`.
//!
//! `trait_layer.rs` is one `impl TransformerLayer for Qwen3SsmLayer` block
//! plus this free function, so the function is the file's only piecewise
//! seam. Moving it takes the file from 501 to 487 LoC. Exact copy.

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;

use crate::layer::LayerState;

/// The PLE per-seq carry from a sequence's [`SsmLayerState`], lazily created
/// on first use. Errors if the state is not an `SsmLayerState`.
pub(super) fn ple_seq_state<'a>(
    ple: &crate::layers::ple::PleLayer,
    state: &'a mut dyn LayerState,
    gpu: &dyn GpuBackend,
) -> Result<&'a mut crate::layers::ple::PleSeqState> {
    let ssm = state
        .as_any_mut()
        .downcast_mut::<crate::layer::SsmLayerState>()
        .ok_or_else(|| anyhow::anyhow!("PLE host layer state is not SsmLayerState"))?;
    if ssm.ple.is_none() {
        ssm.ple = Some(ple.new_seq_state(gpu)?);
    }
    Ok(ssm.ple.as_mut().expect("just created"))
}
