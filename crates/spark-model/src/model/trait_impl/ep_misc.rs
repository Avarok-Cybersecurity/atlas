// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use super::super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::super::ssm_pool::SsmStatePool;
use super::super::ssm_snapshot::SsmSnapshotPool;
use super::super::types::{PinnedMetaStaging, TransformerModel};
use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

impl TransformerModel {
    pub(super) fn ep_worker_step_dispatch(
        &self,
        slots: &mut [Option<SequenceState>],
    ) -> Result<bool> {
        self.ep_worker_step_impl(slots)
    }

    pub(super) fn is_ep_dispatch(&self) -> bool {
        // "EP" here means "the multi-rank head↔worker command protocol is
        // active" — true for EP sharding AND for pure TP (GDN HeadParallel
        // `--tp-size 2 --ep-size 1`, where rank>0 also runs the command
        // loop). The scheduler consults this to disable single-GPU-only
        // fused paths (mixed_forward / co-dispatch) that have no worker
        // wire protocol; those must be off for ANY multi-rank world.
        self.multi_rank_protocol_active()
    }

    pub(super) fn is_mla_dispatch(&self) -> bool {
        // MLA models have a kv_lora_rank > 0 (kv-compression dimension).
        // Same detection as the existing prefill-vs-decode-detail check at
        // line 2185 (`ctx.config.kv_lora_rank > 0`).
        self.config.kv_lora_rank > 0
    }

    pub(super) fn decode_logits_fp32_dispatch(&self) -> bool {
        // Forward to the inherent method. Gated on `use_fp32_logits`, which is
        // hardcoded false in production (Gemma-4 FP32 lm_head bisection scaffold).
        TransformerModel::decode_logits_fp32(self)
    }

    pub(super) fn decode_logits_ptr_dispatch(&self) -> DevicePtr {
        TransformerModel::decode_logits_ptr(self)
    }

    pub(super) fn ep_broadcast_cmd_dispatch(&self, cmd: u32) -> Result<()> {
        // Gate matches `ep_broadcast_seq_and_cmd`: live for EP AND pure TP
        // (see `multi_rank_protocol_active`). Gating on `ep_world_size`
        // alone dropped the prefill chunk args under `--tp-size 2`.
        if self.multi_rank_protocol_active() {
            self.ep_broadcast_u32(cmd)?;
        }
        Ok(())
    }

    pub(super) fn ep_broadcast_tokens_dispatch(&self, tokens: &[u32]) -> Result<Vec<u32>> {
        // Delegate to the inherent method (TransformerModel::ep_broadcast_tokens)
        // which handles per-token fallback via ep_broadcast_u32.
        TransformerModel::ep_broadcast_tokens(self, tokens)
    }

    pub(super) fn default_stream_dispatch(&self) -> u64 {
        self.gpu.default_stream()
    }

    pub(super) fn create_stream_dispatch(&self) -> Result<u64> {
        self.gpu.create_stream()
    }

    pub(super) fn create_event_dispatch(&self) -> Result<u64> {
        self.gpu.create_event()
    }

    pub(super) fn record_event_dispatch(&self, event: u64, stream: u64) -> Result<()> {
        self.gpu.record_event(event, stream)
    }

    pub(super) fn stream_wait_event_dispatch(&self, stream: u64, event: u64) -> Result<()> {
        self.gpu.stream_wait_event(stream, event)
    }

    pub(super) fn synchronize_dispatch(&self, stream: u64) -> Result<()> {
        self.gpu.synchronize(stream)
    }
}
