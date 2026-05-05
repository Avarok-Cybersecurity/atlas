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
use super::ssm_snapshot::SsmSnapshotPool;
use super::types::{PinnedMetaStaging, TransformerModel};
use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

/// Pre-allocated contiguous GPU memory pool for SSM layer states.
///
/// Each pool slot has fixed GPU addresses for h_state and conv_state across
/// all SSM layers. This enables CUDA graph capture at batch sizes > 1 because
/// the graph embeds memory addresses that remain stable across replays.
pub(crate) struct SsmStatePool {
    pub(super) h_state_pools: Vec<DevicePtr>,
    pub(super) conv_state_pools: Vec<DevicePtr>,
    /// Per-slot K=3 intermediate checkpoint pools (only allocated when has_mtp).
    /// Layout: [num_ssm_layers], each allocation = max_slots * 3 * h_bytes.
    pub(super) h_intermediate_pools: Vec<DevicePtr>,
    pub(super) conv_intermediate_pools: Vec<DevicePtr>,
    /// Per-slot SSM state checkpoint pools (only allocated when has_mtp).
    pub(super) h_checkpoint_pools: Vec<DevicePtr>,
    pub(super) conv_checkpoint_pools: Vec<DevicePtr>,
    pub(super) h_bytes: usize,
    pub(super) conv_bytes: usize,
    /// Number of CLAIMABLE slots (excludes the reserved dummy slot at
    /// index `max_slots`). All claim_slot/release_slot operations work
    /// in `[0, max_slots)`.
    pub(super) max_slots: usize,
    pub(super) num_ssm_layers: usize,
    pub(super) has_mtp: bool,
    pub(super) num_intermediates: usize,
    pub(super) free_slots: Mutex<Vec<usize>>,
}

impl SsmStatePool {
    pub(super) fn new(
        config: &ModelConfig,
        max_slots: usize,
        has_mtp: bool,
        num_intermediates: usize,
        gpu: &dyn GpuBackend,
    ) -> Result<Self> {
        let _d_conv = config.linear_conv_kernel_dim;

        let h_bytes = config.ssm_h_state_bytes();
        let conv_bytes = config.ssm_conv_state_bytes();
        let num_ssm_layers = config.num_ssm_layers();

        // Reserve one extra slot at index `max_slots` as a dedicated
        // dummy used by `decode_batch` / `mixed_forward` padding (see
        // `dummy_slot()` below). Without this, pad positions write to
        // pool slot indices `n..padded_n` which can collide with
        // claimed slots if the scheduler invariant ("active sequences
        // occupy contiguous slots [0..n)") is ever broken — silent SSM
        // state corruption. Costs `(h_bytes + conv_bytes) *
        // num_ssm_layers` extra GPU memory (~kilobytes per pool).
        let total_slots = max_slots + 1;

        let mut h_state_pools = Vec::with_capacity(num_ssm_layers);
        let mut conv_state_pools = Vec::with_capacity(num_ssm_layers);
        let mut h_intermediate_pools = Vec::new();
        let mut conv_intermediate_pools = Vec::new();
        let mut h_checkpoint_pools = Vec::new();
        let mut conv_checkpoint_pools = Vec::new();

        for _ in 0..num_ssm_layers {
            let h_pool = gpu.alloc(total_slots * h_bytes)?;
            gpu.memset(h_pool, 0, total_slots * h_bytes)?;
            h_state_pools.push(h_pool);

            let conv_pool = gpu.alloc(total_slots * conv_bytes)?;
            gpu.memset(conv_pool, 0, total_slots * conv_bytes)?;
            conv_state_pools.push(conv_pool);
        }

        if has_mtp {
            let ni = num_intermediates;
            for _ in 0..num_ssm_layers {
                let h_inter = gpu.alloc(total_slots * ni * h_bytes)?;
                gpu.memset(h_inter, 0, total_slots * ni * h_bytes)?;
                h_intermediate_pools.push(h_inter);

                let conv_inter = gpu.alloc(total_slots * ni * conv_bytes)?;
                gpu.memset(conv_inter, 0, total_slots * ni * conv_bytes)?;
                conv_intermediate_pools.push(conv_inter);

                // 1 checkpoint per slot per layer
                let h_ckpt = gpu.alloc(total_slots * h_bytes)?;
                gpu.memset(h_ckpt, 0, total_slots * h_bytes)?;
                h_checkpoint_pools.push(h_ckpt);

                let conv_ckpt = gpu.alloc(total_slots * conv_bytes)?;
                gpu.memset(conv_ckpt, 0, total_slots * conv_bytes)?;
                conv_checkpoint_pools.push(conv_ckpt);
            }

            let mtp_mb = num_ssm_layers
                * total_slots
                * (ni * h_bytes + ni * conv_bytes + h_bytes + conv_bytes)
                / (1024 * 1024);
            tracing::info!("SSM MTP pools ({ni} intermediates + checkpoints): {mtp_mb} MB");
        }

        // free_slots holds claimable indices only; the dummy at index
        // `max_slots` is permanently reserved.
        let free_slots: Vec<usize> = (0..max_slots).rev().collect();

        let total_mb = num_ssm_layers * max_slots * (h_bytes + conv_bytes) / (1024 * 1024);
        tracing::info!(
            "SSM state pool: {max_slots} slots × {num_ssm_layers} layers = {total_mb} MB",
        );

        Ok(Self {
            h_state_pools,
            conv_state_pools,
            h_intermediate_pools,
            conv_intermediate_pools,
            h_checkpoint_pools,
            conv_checkpoint_pools,
            h_bytes,
            conv_bytes,
            max_slots,
            num_ssm_layers,
            has_mtp,
            num_intermediates,
            free_slots: Mutex::new(free_slots),
        })
    }

    pub(super) fn claim_slot(&self) -> Result<usize> {
        self.free_slots.lock().pop().ok_or_else(|| {
            anyhow::anyhow!("SSM state pool exhausted (max {} slots)", self.max_slots)
        })
    }

    pub(super) fn release_slot(&self, idx: usize) {
        self.free_slots.lock().push(idx);
    }

    /// Reserved pool slot used by `decode_batch` / `mixed_forward` padding.
    /// Never claimed by `claim_slot()`, never released. SSM kernels are
    /// free to read/write this slot's pool memory without affecting any
    /// active sequence.
    #[inline]
    pub(super) fn dummy_slot(&self) -> usize {
        self.max_slots
    }

    /// Zero h_state and conv_state for a slot across all SSM layers.
    /// Must be called on slot allocation to prevent stale SSM state
    /// from prior sequences from corrupting new prefill output.
    pub(super) fn zero_slot(&self, idx: usize, gpu: &dyn GpuBackend, stream: u64) -> Result<()> {
        for i in 0..self.num_ssm_layers {
            gpu.memset_async(self.h_state(i, idx), 0, self.h_bytes, stream)?;
            gpu.memset_async(self.conv_state(i, idx), 0, self.conv_bytes, stream)?;
        }
        Ok(())
    }

    pub(super) fn h_state(&self, ssm_layer_idx: usize, slot: usize) -> DevicePtr {
        self.h_state_pools[ssm_layer_idx].offset(slot * self.h_bytes)
    }

    pub(super) fn conv_state(&self, ssm_layer_idx: usize, slot: usize) -> DevicePtr {
        self.conv_state_pools[ssm_layer_idx].offset(slot * self.conv_bytes)
    }

    /// Get fixed-address intermediate h_state for K=2/3/4 verify.
    /// `token_idx` is 0..3 (which token in the verify pass).
    pub(super) fn h_intermediate(
        &self,
        ssm_layer_idx: usize,
        slot: usize,
        token_idx: usize,
    ) -> DevicePtr {
        let ni = self.num_intermediates;
        self.h_intermediate_pools[ssm_layer_idx].offset((slot * ni + token_idx) * self.h_bytes)
    }

    pub(super) fn conv_intermediate(
        &self,
        ssm_layer_idx: usize,
        slot: usize,
        token_idx: usize,
    ) -> DevicePtr {
        let ni = self.num_intermediates;
        self.conv_intermediate_pools[ssm_layer_idx]
            .offset((slot * ni + token_idx) * self.conv_bytes)
    }

    pub(super) fn h_checkpoint(&self, ssm_layer_idx: usize, slot: usize) -> DevicePtr {
        self.h_checkpoint_pools[ssm_layer_idx].offset(slot * self.h_bytes)
    }

    pub(super) fn conv_checkpoint(&self, ssm_layer_idx: usize, slot: usize) -> DevicePtr {
        self.conv_checkpoint_pools[ssm_layer_idx].offset(slot * self.conv_bytes)
    }

    pub(super) fn reset_slot(&self, slot: usize, gpu: &dyn GpuBackend) -> Result<()> {
        for i in 0..self.num_ssm_layers {
            gpu.memset(self.h_state(i, slot), 0, self.h_bytes)?;
            gpu.memset(self.conv_state(i, slot), 0, self.conv_bytes)?;
            if self.has_mtp {
                for t in 0..self.num_intermediates {
                    gpu.memset(self.h_intermediate(i, slot, t), 0, self.h_bytes)?;
                    gpu.memset(self.conv_intermediate(i, slot, t), 0, self.conv_bytes)?;
                }
                gpu.memset(self.h_checkpoint(i, slot), 0, self.h_bytes)?;
                gpu.memset(self.conv_checkpoint(i, slot), 0, self.conv_bytes)?;
            }
        }
        Ok(())
    }

    pub(super) fn copy_slot(
        &self,
        from: usize,
        to: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        for i in 0..self.num_ssm_layers {
            gpu.copy_d2d_async(
                self.h_state(i, from),
                self.h_state(i, to),
                self.h_bytes,
                stream,
            )?;
            gpu.copy_d2d_async(
                self.conv_state(i, from),
                self.conv_state(i, to),
                self.conv_bytes,
                stream,
            )?;
            if self.has_mtp {
                for t in 0..self.num_intermediates {
                    gpu.copy_d2d_async(
                        self.h_intermediate(i, from, t),
                        self.h_intermediate(i, to, t),
                        self.h_bytes,
                        stream,
                    )?;
                    gpu.copy_d2d_async(
                        self.conv_intermediate(i, from, t),
                        self.conv_intermediate(i, to, t),
                        self.conv_bytes,
                        stream,
                    )?;
                }
                gpu.copy_d2d_async(
                    self.h_checkpoint(i, from),
                    self.h_checkpoint(i, to),
                    self.h_bytes,
                    stream,
                )?;
                gpu.copy_d2d_async(
                    self.conv_checkpoint(i, from),
                    self.conv_checkpoint(i, to),
                    self.conv_bytes,
                    stream,
                )?;
            }
        }
        Ok(())
    }
}
