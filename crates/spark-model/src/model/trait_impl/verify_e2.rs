// SPDX-License-Identifier: AGPL-3.0-only

//! Batched K=4 verify support (verify_e.rs): WY pointer-table staging +
//! CUDA-graph gating helpers. Split out to keep verify_e under the LoC cap.

#![allow(dead_code)]

use anyhow::Result;
use atlas_core::config::LayerType;
use spark_runtime::gpu::DevicePtr;

use super::super::types::TransformerModel;
use crate::layer::{SsmLayerState, VERIFY_WY_LAYER_STRIDE_BYTES, VERIFY_WY_TABLE_SEQS};
use crate::traits::SequenceState;

/// Bound on cached batched-verify graphs (one per distinct ssm-slot vector).
/// Slot vectors churn as sequences finish; past the cap a step with a new
/// vector runs eager instead of capturing (graphs are never evicted
/// mid-serve, so the cap bounds graph memory).
pub(super) const VERIFY_BATCHED_GRAPH_CAP: usize = 32;

/// Batched-verify CUDA graphs: ON by default, disabled by PRESENCE of
/// `ATLAS_NO_MTP_VERIFY_GRAPHS` (house convention — `=0` is NOT off).
/// Read once per process.
pub(super) fn verify_graphs_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ATLAS_NO_MTP_VERIFY_GRAPHS").is_none())
}

impl TransformerModel {
    /// Batched-verify graph key: each sequence's ssm-pool slot in batch
    /// order — every SSM pointer the graph bakes (h/conv state, rollback
    /// intermediates, WY table contents) is a pure function of this vector;
    /// all other captured addresses (hidden/logits/scratch/meta) are fixed
    /// buffers refreshed pre-replay. A wy-tables-present sentinel is
    /// appended so a table-less capture can never replay a table-full step
    /// or vice versa. `None` → no graph (a sequence without a pool slot).
    pub(super) fn verify_batched_graph_key(
        &self,
        seqs: &[&mut SequenceState],
        wy_tables_null: bool,
    ) -> Option<Vec<u32>> {
        let mut key: Vec<u32> = Vec::with_capacity(seqs.len() + 1);
        for s in seqs.iter() {
            key.push(s.ssm_slot_idx()? as u32);
        }
        key.push(u32::MAX - u32::from(wy_tables_null));
        Some(key)
    }

    /// Stage the per-GDN-layer WY pointer tables (`[h|Hi0|Hi1|Hi2]` ×
    /// `VERIFY_WY_TABLE_SEQS` u64 entries per layer, batch entries filled,
    /// tail zero) into the fixed `verify_wy_tables` device buffer. Runs
    /// PRE-graph every batched verify step so a replayed graph reads tables
    /// refreshed for the current batch (contents are constant per slot
    /// vector; refreshing keeps replay correct by construction, not by
    /// invariant).
    ///
    /// Returns NULL — uploading nothing — unless EVERY GDN layer × sequence
    /// provides h_state + ≥3 h intermediates (the layer-side batched arm
    /// re-checks per layer; defense in depth). NULL keeps the per-sequence
    /// WY loop, which is byte-identical math.
    pub(super) fn upload_verify_wy_tables(
        &self,
        seqs: &[&mut SequenceState],
        stream: u64,
    ) -> Result<DevicePtr> {
        let n = seqs.len();
        if self.verify_wy_tables.is_null() || n > VERIFY_WY_TABLE_SEQS {
            return Ok(DevicePtr::NULL);
        }
        let num_ssm = self.config.num_ssm_layers();
        if num_ssm == 0 {
            return Ok(DevicePtr::NULL);
        }
        let entries_per_layer = VERIFY_WY_LAYER_STRIDE_BYTES / 8;
        let mut host = vec![0u64; num_ssm * entries_per_layer];
        let mut ssm_idx = 0usize;
        for layer_idx in 0..self.layers.len() {
            if self.config.layer_type(layer_idx) != LayerType::LinearAttention {
                continue;
            }
            let base = ssm_idx * entries_per_layer;
            for (i, seq) in seqs.iter().enumerate() {
                let Some(st) = seq.layer_states[layer_idx]
                    .as_any()
                    .downcast_ref::<SsmLayerState>()
                else {
                    return Ok(DevicePtr::NULL);
                };
                if st.h_state.is_null() || st.h_state_intermediates.len() < 3 {
                    return Ok(DevicePtr::NULL);
                }
                host[base + i] = st.h_state.0;
                host[base + VERIFY_WY_TABLE_SEQS + i] = st.h_state_intermediates[0].0;
                host[base + 2 * VERIFY_WY_TABLE_SEQS + i] = st.h_state_intermediates[1].0;
                host[base + 3 * VERIFY_WY_TABLE_SEQS + i] = st.h_state_intermediates[2].0;
            }
            ssm_idx += 1;
        }
        // Pageable-source async H2D per house pattern (the driver stages the
        // host bytes before returning, same as the metadata uploads).
        let bytes =
            unsafe { std::slice::from_raw_parts(host.as_ptr() as *const u8, host.len() * 8) };
        self.gpu
            .copy_h2d_async(bytes, self.verify_wy_tables, stream)?;
        Ok(self.verify_wy_tables)
    }
}
