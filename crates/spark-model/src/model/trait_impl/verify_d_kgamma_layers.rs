// SPDX-License-Identifier: AGPL-3.0-only

//! Per-layer forward loop for the K=γ (DFlash) verify path. Extracted from
//! `verify_d.rs` to keep that file under the 500-LoC cap.

#![allow(unused_imports, dead_code, clippy::too_many_arguments)]

use anyhow::Result;
use atlas_core::config::LayerType;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::types::TransformerModel;
use crate::layer::{ForwardContext, GdnPrefillBuffers, LayerState, SsmLayerState};
use crate::traits::SequenceState;

pub(super) struct KgammaLayerTiming {
    pub us_attn: u128,
    pub us_p1: u128,
    pub us_p2: u128,
    pub us_p3: u128,
}

impl TransformerModel {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn run_kgamma_verify_layers(
        &self,
        seq: &mut SequenceState,
        kv_cache: &mut PagedKvCache,
        hidden: DevicePtr,
        residual: DevicePtr,
        k: usize,
        h: usize,
        bf16: usize,
        hss_engaged: bool,
        use_prefill_attn: bool,
        use_prefill_ssm: bool,
        eagle_fix: bool,
        timing: bool,
        seq_lens_vec: &[usize],
        block_tables_vec: &[Vec<u32>],
        gdn_bufs: &GdnPrefillBuffers,
        ssm_conv_dim: usize,
        ssm_gate_stride: usize,
        ssm_value_dim: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<KgammaLayerTiming> {
        let mut us_attn: u128 = 0; // attention prefill (sum over attn layers)
        let mut us_p1: u128 = 0; // SSM phase-1 per-token conv
        let mut us_p2: u128 = 0; // SSM phase-2 GDN (wy16 or wy4+replay)
        let mut us_p3: u128 = 0; // SSM phase-3 norm+proj+FFN

        for (layer_idx, layer) in self.layers.iter().enumerate() {
            let layer_type = self.config.layer_type(layer_idx);

            if layer_type == LayerType::FullAttention {
                if hss_engaged {
                    layer.decode_batched(
                        hidden,
                        residual,
                        k,
                        seq.layer_states[layer_idx].as_mut(),
                        kv_cache,
                        seq.seq_len,
                        &mut seq.block_table,
                        &mut seq.disk_block_ids,
                        &mut seq.disk_last_offloaded_per_layer,
                        ctx,
                        stream,
                    )?;
                } else if use_prefill_attn {
                    debug_assert!(seq.seq_len > 0, "prefill-verify requires non-empty context");
                    let _ts = if timing {
                        self.gpu.synchronize(stream)?;
                        Some(std::time::Instant::now())
                    } else {
                        None
                    };
                    layer.prefill(
                        hidden,
                        residual,
                        k,
                        seq.layer_states[layer_idx].as_mut(),
                        kv_cache,
                        seq.seq_len,
                        &mut seq.block_table,
                        &mut seq.disk_block_ids,
                        &mut seq.disk_last_offloaded_per_layer,
                        0,
                        ctx,
                        stream,
                    )?;
                    if let Some(t) = _ts {
                        self.gpu.synchronize(stream)?;
                        us_attn += t.elapsed().as_micros();
                    }
                } else {
                    let mut dummy_states: Vec<Box<dyn LayerState>> = (0..k)
                        .map(|_| layer.alloc_state(self.gpu.as_ref()))
                        .collect::<Result<_>>()?;
                    let mut refs: Vec<&mut (dyn LayerState + 'static)> =
                        dummy_states.iter_mut().map(|s| s.as_mut()).collect();
                    layer.decode_multi_seq(
                        hidden,
                        residual,
                        k,
                        &mut refs,
                        kv_cache,
                        seq_lens_vec,
                        block_tables_vec,
                        ctx,
                        stream,
                    )?;
                }
            } else if layer.is_ssm_layer() && use_prefill_ssm {
                // Three-phase parallel SSM verify (see SSM_VERIFY_PLAN.md).
                //
                // Phase 1: K serial single-token calls fill gdn_bufs at
                // token_offset=t and slide conv_state one step each, so
                // conv_state_intermediates[t] can be saved after each call.
                let _ts1 = if timing {
                    self.gpu.synchronize(stream)?;
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                // C1 conv-fusion (ATLAS_DFLASH_CONV_FUSION=1): run the entire
                // phase-1 pipeline ONCE over all k tokens — batches the QKVZ
                // projection (k GEMVs → 1 GEMM), gates, conv (writing the k-1
                // conv intermediates inline), and l2_norm. Falls back to the
                // per-token loop when off or the _inter kernel is unavailable.
                let conv_fusion =
                    std::env::var("ATLAS_DFLASH_CONV_FUSION").ok().as_deref() == Some("1");
                let fused_p1 = conv_fusion
                    && layer.prefill_phase1_verify(
                        hidden,
                        residual,
                        k,
                        seq.layer_states[layer_idx].as_mut(),
                        kv_cache,
                        seq.seq_len,
                        &mut seq.block_table,
                        &mut seq.disk_block_ids,
                        &mut seq.disk_last_offloaded_per_layer,
                        0,
                        gdn_bufs,
                        ctx,
                        stream,
                    )?;
                if !fused_p1 {
                    for t in 0..k {
                        layer.prefill_phase1(
                            hidden.offset(t * h * bf16),
                            residual.offset(t * h * bf16),
                            1,
                            seq.layer_states[layer_idx].as_mut(),
                            kv_cache,
                            seq.seq_len + t,
                            &mut seq.block_table,
                            &mut seq.disk_block_ids,
                            &mut seq.disk_last_offloaded_per_layer,
                            0,
                            gdn_bufs,
                            t,
                            ctx,
                            stream,
                        )?;
                        let s = seq.layer_states[layer_idx]
                            .as_any_mut()
                            .downcast_mut::<SsmLayerState>()
                            .ok_or_else(|| {
                                anyhow::anyhow!("expected SsmLayerState at layer {layer_idx}")
                            })?;
                        if t < s.conv_state_intermediates.len() {
                            self.gpu.copy_d2d_async(
                                s.conv_state,
                                s.conv_state_intermediates[t],
                                self.ssm_pool.conv_bytes,
                                stream,
                            )?;
                        }
                    }
                }
                if let Some(t) = _ts1 {
                    self.gpu.synchronize(stream)?;
                    us_p1 += t.elapsed().as_micros();
                }
                // Phase 2: single fused WY16 pass (K=16) writes final h_state
                // AND Hi_0..Hi_14 intermediates inline — eliminating the WY4
                // batch + the per-token replay loop. Falls back to WY4+replay
                // when wy16 is unavailable (other K, or kernel NULL).
                let _ts2 = if timing {
                    self.gpu.synchronize(stream)?;
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                let used_wy16 = layer.prefill_gdn_wy16(
                    seq.layer_states[layer_idx].as_mut(),
                    gdn_bufs,
                    ctx,
                    stream,
                )?;
                if !used_wy16 {
                    // Phase 2: one WY4 batch GDN recurrence over all K tokens.
                    layer.prefill_gdn_full(
                        seq.layer_states[layer_idx].as_mut(),
                        gdn_bufs,
                        ctx,
                        stream,
                    )?;
                    // Fill h_state_intermediates via per-token replay from checkpoint (R7).
                    // Extract DevicePtr copies before the replay loop to avoid borrow
                    // conflicts with the trait method calls inside the loop.
                    let (h_state_ptr, h_ckpt_opt, h_ints_vec) = {
                        let s = seq.layer_states[layer_idx]
                            .as_any_mut()
                            .downcast_mut::<SsmLayerState>()
                            .ok_or_else(|| {
                                anyhow::anyhow!("expected SsmLayerState at layer {layer_idx}")
                            })?;
                        (
                            s.h_state,
                            s.h_state_checkpoint,
                            s.h_state_intermediates.clone(),
                        )
                    };
                    let h_bytes = self.ssm_pool.h_bytes;
                    // Save WY4 final h_state to scratch
                    self.gpu
                        .copy_d2d_async(h_state_ptr, self.ssm_verify_h_tmp, h_bytes, stream)?;
                    // Restore pre-verify checkpoint so replay starts from correct state
                    if let Some(ckpt) = h_ckpt_opt {
                        self.gpu
                            .copy_d2d_async(ckpt, h_state_ptr, h_bytes, stream)?;
                    }
                    // K serial single-token GDN steps; after each step h_state_ptr
                    // device memory holds h_state after tokens 0..=t.
                    for t in 0..k.min(h_ints_vec.len()) {
                        let tok_gdn = GdnPrefillBuffers {
                            qkv: gdn_bufs.qkv.offset(t * ssm_conv_dim * bf16),
                            gate_beta: gdn_bufs.gate_beta.offset(t * ssm_gate_stride * 4),
                            output: gdn_bufs.output.offset(t * ssm_value_dim * bf16),
                            z: gdn_bufs.z.offset(t * ssm_value_dim * bf16),
                            total_len: 1,
                        };
                        layer.prefill_gdn_full(
                            seq.layer_states[layer_idx].as_mut(),
                            &tok_gdn,
                            ctx,
                            stream,
                        )?;
                        self.gpu
                            .copy_d2d_async(h_state_ptr, h_ints_vec[t], h_bytes, stream)?;
                    }
                    // Restore WY4 final h_state
                    self.gpu
                        .copy_d2d_async(self.ssm_verify_h_tmp, h_state_ptr, h_bytes, stream)?;
                } // end !used_wy16 (WY4 batch + per-token replay fallback)
                if let Some(t) = _ts2 {
                    self.gpu.synchronize(stream)?;
                    us_p2 += t.elapsed().as_micros();
                }
                // Phase 3: gated RMS norm + output projection + FFN
                let _ts3 = if timing {
                    self.gpu.synchronize(stream)?;
                    Some(std::time::Instant::now())
                } else {
                    None
                };
                layer.prefill_phase3(hidden, residual, k, gdn_bufs, 0, ctx, stream)?;
                if let Some(t) = _ts3 {
                    self.gpu.synchronize(stream)?;
                    us_p3 += t.elapsed().as_micros();
                }
            } else {
                layer.decode_batched(
                    hidden,
                    residual,
                    k,
                    seq.layer_states[layer_idx].as_mut(),
                    kv_cache,
                    seq.seq_len,
                    &mut seq.block_table,
                    &mut seq.disk_block_ids,
                    &mut seq.disk_last_offloaded_per_layer,
                    ctx,
                    stream,
                )?;
            }

            // DFlash per-layer hidden capture into dflash_hidden_save.
            // EAGLE-fix: capture all k verify rows (row-major) so the
            // scheduler can append one ctx slot per accepted position.
            // Legacy: capture only row 0 (last_token), appended via propose.
            if eagle_fix {
                self.try_dflash_capture_all(layer_idx, k, stream)?;
            } else {
                self.try_dflash_capture(layer_idx, 0, stream)?;
            }
        }

        Ok(KgammaLayerTiming {
            us_attn,
            us_p1,
            us_p2,
            us_p3,
        })
    }
}
