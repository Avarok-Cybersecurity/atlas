// SPDX-License-Identifier: AGPL-3.0-only

//! BF16 twin layer bind. GPU decode is not this slice (C1 is atlas-core CPU).

use anyhow::{Result, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::PagedKvCache;

use crate::layer::{EmptyLayerState, ForwardContext, LayerState, TransformerLayer};
use crate::weight_map::DenseWeight;

const GPU_WIP: &str = "K3 GPU forward is not this slice; C1 greedy lives in atlas_core::kimi_k3";

/// One decoder layer whose BF16 (or FP32) tensors were bound from the store.
pub struct K3BoundLayer {
    pub index: usize,
    pub weights: Vec<DenseWeight>,
}

impl TransformerLayer for K3BoundLayer {
    fn decode_graph_unsupported(&self) -> bool {
        true
    }

    fn decode_multi_seq_unsupported(&self) -> bool {
        true
    }

    #[allow(clippy::too_many_arguments)]
    fn decode(
        &self,
        _hidden: DevicePtr,
        _residual: DevicePtr,
        _state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        _seq_len: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        _ctx: &ForwardContext,
        _stream: u64,
    ) -> Result<()> {
        bail!("{GPU_WIP}")
    }

    fn alloc_state(&self, _gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        Ok(Box::new(EmptyLayerState))
    }
}
