// SPDX-License-Identifier: AGPL-3.0-only

//! BF16/FP32 twin layer bind. Decode copies hidden D2H, runs mixer+MLP+AttnRes,
//! copies H2D. Default mixer is CPU (C1 aviation greedy). `K3_CUDA_KDA=1`
//! swaps LinearAttention / KDA conv+recurrent onto `kda_decode` CUDA.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use atlas_core::kimi_k3::{
    AttnResStream, K3CpuLayer, K3Graph, K3LayerSpec, KdaConfig, LatentMoeConfig, LayerCache,
    MixerKind, MlaConfig,
};
use parking_lot::Mutex;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::PagedKvCache;
use spark_runtime::weights::WeightDtype;

use super::kda_cuda::K3KdaDecodeKernels;
use super::state::K3CpuFallbackState;
use crate::layer::{ForwardContext, LayerState, TransformerLayer};
use crate::weight_map::DenseWeight;

/// Name + device dtype/numel for a lazy host bind (copy from GPU if needed).
#[derive(Clone, Debug)]
pub struct WeightMeta {
    pub name: String,
    pub dtype: WeightDtype,
    pub numel: usize,
}

/// Shared across every decoder layer of one loaded K3 model.
pub struct K3HostShared {
    pub config: ModelConfig,
    pub graph: K3Graph,
    pub kda: KdaConfig,
    pub mla: MlaConfig,
    pub moe: LatentMoeConfig,
    pub output_res_proj: DenseWeight,
    pub output_res_norm: DenseWeight,
    pub output_res_proj_meta: (WeightDtype, usize),
    pub output_res_norm_meta: (WeightDtype, usize),
    pub output_host: OnceLock<(Vec<f32>, Vec<f32>)>,
    /// Resolved once per loaded model. LinearAttention decode launches these.
    pub kda_kernels: OnceLock<K3KdaDecodeKernels>,
    /// AttnRes is per-token across layers. Keyed by this step's `residual`
    /// pointer so prefill (layer-outer, token-inner) still sees the same
    /// stream as CPU `forward_token` (token-outer, layer-inner).
    pub attnres: Mutex<HashMap<DevicePtr, AttnResStream>>,
}

/// One decoder layer whose BF16 (or FP32) tensors were bound from the store.
pub struct K3BoundLayer {
    pub index: usize,
    pub spec: K3LayerSpec,
    pub weights: Vec<DenseWeight>,
    pub weight_meta: Vec<WeightMeta>,
    pub host: OnceLock<K3CpuLayer>,
    pub shared: Arc<K3HostShared>,
}

impl TransformerLayer for K3BoundLayer {
    fn decode_graph_unsupported(&self) -> bool {
        // Host round-trip cannot live in a CUDA graph.
        true
    }

    fn decode_multi_seq_unsupported(&self) -> bool {
        true
    }

    fn uses_ssm_pool(&self) -> bool {
        // KDA is `linear_attention` in config. Own [`K3CpuFallbackState`], not GDN pool.
        false
    }

    fn has_aux_state(&self) -> bool {
        true
    }

    fn snapshot_aux(
        &self,
        state: &dyn LayerState,
        _gpu: &dyn GpuBackend,
        _stream: u64,
    ) -> Result<Option<Vec<u8>>> {
        let st = state
            .as_any()
            .downcast_ref::<K3CpuFallbackState>()
            .context("K3 snapshot_aux: expected K3CpuFallbackState")?;
        Ok(Some(st.cache.to_bytes()))
    }

    fn restore_aux(
        &self,
        state: &mut dyn LayerState,
        blob: &[u8],
        _gpu: &dyn GpuBackend,
        _stream: u64,
    ) -> Result<()> {
        let st = state
            .as_any_mut()
            .downcast_mut::<K3CpuFallbackState>()
            .context("K3 restore_aux: expected K3CpuFallbackState")?;
        st.cache = LayerCache::from_bytes(blob)?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn decode(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        state: &mut dyn LayerState,
        _kv_cache: &mut PagedKvCache,
        seq_len: usize,
        _block_table: &mut Vec<u32>,
        _disk_block_ids: &mut Vec<u32>,
        _disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        self.decode_host(
            hidden,
            residual,
            state,
            seq_len,
            ctx,
            stream,
            atlas_core::kimi_k3::cuda_kda_enabled(),
        )
    }

    fn alloc_state(&self, _gpu: &dyn GpuBackend) -> Result<Box<dyn LayerState>> {
        let cache = match self.spec.mixer {
            MixerKind::Kda => LayerCache::Kda(atlas_core::kimi_k3::KdaState::new(&self.shared.kda)),
            MixerKind::Mla => LayerCache::Mla(atlas_core::kimi_k3::MlaKv::default()),
        };
        Ok(Box::new(K3CpuFallbackState { cache }))
    }
}
