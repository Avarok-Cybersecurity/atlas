// SPDX-License-Identifier: AGPL-3.0-only

#![allow(unused_imports, dead_code)]

use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use super::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use super::ssm_pool::SsmStatePool;
use super::ssm_snapshot::SsmSnapshotPool;
use super::types::{PinnedMetaStaging, TransformerModel};
use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

impl Drop for TransformerModel {
    fn drop(&mut self) {
        self.drop_pinned_staging();
        // The SSM spill tier's reusable staging blob is page-locked host memory
        // owned by the snapshot pool, which holds no `gpu` handle of its own —
        // same ownership shape as `drop_pinned_staging`. No-op when the tier
        // never ran (the buffer is allocated on first spill).
        self.ssm_snapshots.free_staging(self.gpu.as_ref());
        // Same shape again for the aux collect's gather blob.
        self.aux_staging.free(self.gpu.as_ref());
        // Drop the EXL3 smem-raise memo with the model whose module handles it
        // caches: CUfunction addresses are recycled across a load/unload, and a
        // stale "already raised" entry makes the next model's first EXL3 GEMM
        // fail for want of its 90KB.
        crate::layers::ops::forget_smem_raises();
    }
}
