// SPDX-License-Identifier: AGPL-3.0-only

//! Sequence save/restore state I/O for `TransformerModel` (split from sequence.rs for the 500-LoC cap).
use parking_lot::Mutex;
use std::collections::HashMap;
use std::sync::Arc;

use anyhow::{Result, bail};
use atlas_core::config::{LayerType, ModelConfig};
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::{DevicePtr, GpuBackend, GraphHandle, KernelHandle};
use spark_runtime::kv_cache::PagedKvCache;

use crate::layer::{
    AttnMetadataDev, ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState, TransformerLayer,
};
use crate::layers::ops;
use crate::model::block_mgmt::{
    apply_evicted_blocks, ensure_blocks_through_decode, ensure_blocks_through_prefill,
    extract_layer_refs, reuse_prefix_match_disk_ids,
};
use crate::model::ssm_pool::SsmStatePool;
use crate::model::ssm_snapshot::SsmSnapshotPool;
use crate::model::types::{PinnedMetaStaging, TransformerModel};
use crate::speculative::DraftProposer;
use crate::traits::{ChunkedPrefillPageMetadata, Model, SequenceState};
use crate::weight_map::{DenseWeight, MtpWeights, QuantizedWeight};

impl TransformerModel {
    pub(crate) fn save_sequence_state_dispatch(
        &self,
        seq: &SequenceState,
        writer: &mut dyn std::io::Write,
    ) -> Result<()> {
        let gpu = self.gpu.as_ref();

        // Phase 1: Copy all KV block data from GPU to host buffers under the lock.
        let kv_buffers = {
            let kv = self.kv_cache.lock();
            let mut bufs = Vec::with_capacity(seq.block_table.len() * kv.num_layers());
            for &block_idx in &seq.block_table {
                for layer_idx in 0..kv.num_layers() {
                    bufs.push(kv.read_block(layer_idx, block_idx, gpu)?);
                }
            }
            bufs
        }; // Lock released here.

        // Phase 2: Write KV data to disk (no lock held).
        for (k_data, v_data) in &kv_buffers {
            writer.write_all(k_data)?;
            writer.write_all(v_data)?;
        }

        // Phase 3: Copy SSM states from GPU to host, then write to disk.
        for (i, layer_state) in seq.layer_states.iter().enumerate() {
            if self.config.layer_type(i) == LayerType::LinearAttention {
                let ssm = layer_state
                    .as_any()
                    .downcast_ref::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;

                let mut h_buf = vec![0u8; self.ssm_pool.h_bytes];
                let mut c_buf = vec![0u8; self.ssm_pool.conv_bytes];
                gpu.copy_d2h(ssm.h_state, &mut h_buf)?;
                gpu.copy_d2h(ssm.conv_state, &mut c_buf)?;
                writer.write_all(&h_buf)?;
                writer.write_all(&c_buf)?;
            }
        }

        writer.flush()?;
        Ok(())
    }

    pub(crate) fn restore_sequence_state_dispatch(
        &self,
        seq: &mut SequenceState,
        num_blocks: usize,
        reader: &mut dyn std::io::Read,
    ) -> Result<()> {
        let gpu = self.gpu.as_ref();

        // Phase 1: Read all KV block data from disk into host buffers.
        let (num_layers, layer_strides) = {
            let kv = self.kv_cache.lock();
            let n = kv.num_layers();
            let strides: Vec<usize> = (0..n).map(|i| kv.block_stride_bytes_for_layer(i)).collect();
            (n, strides)
        };

        let mut kv_buffers = Vec::with_capacity(num_blocks * num_layers);
        for _ in 0..num_blocks {
            for layer_idx in 0..num_layers {
                let stride = layer_strides[layer_idx];
                let mut k_data = vec![0u8; stride];
                let mut v_data = vec![0u8; stride];
                reader.read_exact(&mut k_data)?;
                reader.read_exact(&mut v_data)?;
                kv_buffers.push((k_data, v_data));
            }
        }

        // Phase 2: Allocate blocks and write data under the lock.
        {
            let mut kv = self.kv_cache.lock();
            let mut new_block_table = Vec::with_capacity(num_blocks);
            let mut buf_idx = 0;
            for _ in 0..num_blocks {
                let block_idx = kv.alloc_block()?;
                for layer_idx in 0..num_layers {
                    let (ref k_data, ref v_data) = kv_buffers[buf_idx];
                    kv.write_block(layer_idx, block_idx, k_data, v_data, gpu)?;
                    buf_idx += 1;
                }
                new_block_table.push(block_idx);
            }
            seq.block_table = new_block_table;
        } // Lock released here.

        // Phase 3: Read SSM state data from disk and upload to GPU.
        for (i, layer_state) in seq.layer_states.iter_mut().enumerate() {
            if self.config.layer_type(i) == LayerType::LinearAttention {
                let ssm = layer_state
                    .as_any_mut()
                    .downcast_mut::<SsmLayerState>()
                    .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState at layer {i}"))?;

                let mut h_buf = vec![0u8; self.ssm_pool.h_bytes];
                let mut c_buf = vec![0u8; self.ssm_pool.conv_bytes];
                reader.read_exact(&mut h_buf)?;
                reader.read_exact(&mut c_buf)?;
                gpu.copy_h2d(&h_buf, ssm.h_state)?;
                gpu.copy_h2d(&c_buf, ssm.conv_state)?;
            }
        }

        Ok(())
    }
}
