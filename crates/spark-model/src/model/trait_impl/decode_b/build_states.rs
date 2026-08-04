// SPDX-License-Identifier: AGPL-3.0-only

//! Decode phase B — decode layer-state assembly for the fused mixed forward.
//!
//! Extracted from `decode_b.rs` to keep each file under the 500-LoC
//! file-size cap. Builds the per-sequence `seq_lens` / `block_tables`
//! vectors and takes ownership of each decode sequence's `layer_states`,
//! appending dummy padding states (one per padding slot) so the batched
//! decode kernel sees a uniform `padded_n` batch.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;

use super::super::super::types::TransformerModel;
use crate::layer::{LayerState, SsmLayerState};
use crate::traits::SequenceState;
use atlas_core::config::LayerType;

impl TransformerModel {
    /// Build the decode portion's `(seq_lens, block_tables, all_layer_states)`
    /// for the fused mixed forward. Real decode sequences contribute their
    /// own seq_len / block_table / moved-out layer_states; padding slots get
    /// `seq_len=0`, the dummy KV block, and freshly-built dummy layer states
    /// (SSM layers point at the pool's `dummy_slot`).
    pub(super) fn mixed_build_decode_layer_states(
        &self,
        decode_seqs: &mut [&mut SequenceState],
        padded_n: usize,
        n_decode: usize,
    ) -> Result<(Vec<usize>, Vec<Vec<u32>>, Vec<Vec<Box<dyn LayerState>>>)> {
        let seq_lens: Vec<usize> = (0..padded_n)
            .map(|i| {
                if i < n_decode {
                    decode_seqs[i].seq_len
                } else {
                    0
                }
            })
            .collect();
        let block_tables: Vec<Vec<u32>> = (0..padded_n)
            .map(|i| {
                if i < n_decode {
                    decode_seqs[i].block_table.clone()
                } else {
                    vec![self.dummy_kv_block]
                }
            })
            .collect();

        let mut all_layer_states: Vec<Vec<Box<dyn LayerState>>> = decode_seqs
            .iter_mut()
            .map(|s| std::mem::take(&mut s.layer_states))
            .collect();

        // Build dummy layer_states for padding positions. Use the
        // dedicated `dummy_slot()` (see SsmStatePool) so pad SSM kernel
        // writes can never collide with another claimed sequence.
        let dummy_ssm_slot = self.ssm_pool.dummy_slot();
        for _pad_pos in n_decode..padded_n {
            let mut dummy: Vec<Box<dyn LayerState>> = Vec::with_capacity(self.layers.len());
            let mut ssm_idx = 0usize;
            for (li, layer) in self.layers.iter().enumerate() {
                if self.config.layer_type(li) == LayerType::LinearAttention {
                    dummy.push(Box::new(SsmLayerState {
                        h_state: self.ssm_pool.h_state(ssm_idx, dummy_ssm_slot),
                        conv_state: self.ssm_pool.conv_state(ssm_idx, dummy_ssm_slot),
                        h_state_checkpoint: None,
                        conv_state_checkpoint: None,
                        h_state_intermediates: Vec::new(),
                        conv_state_intermediates: Vec::new(),
                    }));
                    ssm_idx += 1;
                } else {
                    dummy.push(layer.alloc_state(self.gpu.as_ref())?);
                }
            }
            all_layer_states.push(dummy);
        }

        Ok((seq_lens, block_tables, all_layer_states))
    }
}
