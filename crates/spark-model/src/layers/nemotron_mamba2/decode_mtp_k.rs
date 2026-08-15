// SPDX-License-Identifier: AGPL-3.0-only

//! Milestone B: K-token MTP verify body with the stateless phases batched.
//!
//! The verify rows are sequential TIME STEPS of one sequence, not independent
//! sequences — token `t+1`'s recurrent state depends on token `t` — so the
//! milestone-B strided kernels, which batch across sequences, do nothing for
//! the conv+scan here. That inner stays a `t` loop and always will until a
//! fused K-token conv+scan kernel exists.
//!
//! What DOES flip the economics is weight traffic. Every other phase of the
//! layer is row-independent, so running them at `M = K` collapses K sweeps of
//! the ~38.7M-param in_proj/out_proj pair into one. On bandwidth-bound
//! LPDDR5X that pair IS the verify cost: at K=2 across 23 Mamba layers it is
//! ~0.89 GB per verify step that no longer moves, which is the difference
//! between "verify costs more than 2 serial decodes" and "it costs about
//! one". That was the measured reason MTP was net-negative on Lightning
//! (70.1 tok/s serial vs 55.0 forced) despite a healthy p1 ~ 0.55 acceptance:
//! the drafts were fine, the verify arithmetic was not.
//!
//! ## Why this is safe to switch on unconditionally at K >= proj rung
//!
//! Byte-identical to the sequential `decode()` loop it replaces, phase by
//! phase: `rms_norm_residual` / `gated_rms_norm` run one block per row, so a
//! K-row launch does exactly the K single-row reductions; `residual_add` is
//! elementwise; conv and scan still run per token at `batch = 1` with the
//! same pointers; and the projections route to `w8a16_gemv_batch4`, which
//! milestone B proved byte-identical to M x `w8a16_gemv` at the Lightning
//! shapes (`examples/w8a16_batch_bitparity_microtest.rs`). Verify therefore
//! accepts exactly the tokens it accepted before — a throughput change with
//! no acceptance-rate change, which is what makes the A/B interpretable.
//!
//! Arms whose batched projection twin is NOT proven bit-exact keep the
//! sequential loop, via the same `proj_batch_min()` rung the concurrent
//! decode path uses.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::NemotronMamba2Layer;
use crate::layer::{ForwardContext, LayerState, SsmLayerState};
use crate::layers::ops;

impl NemotronMamba2Layer {
    /// Whether the K-token verify body may batch its stateless phases.
    /// `num_tokens` is the VERIFY WIDTH (drafts + 1).
    pub(super) fn mtp_k_batching_ok(&self, num_tokens: usize) -> bool {
        num_tokens >= 2 && num_tokens >= self.proj_batch_min()
    }

    /// K-token verify body: batched norms + projections, sequential conv/scan,
    /// per-token state snapshots preserved exactly where the rollback
    /// contract expects them.
    pub(super) fn decode_batched_k(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let k = num_tokens;
        let bf16 = 2usize;
        let gs = self.n_groups * self.state_size;

        // 1. Batched input norm + residual save.
        let normed = ctx.buffers.norm_output();
        ops::rms_norm_residual(
            ctx.gpu,
            self.rms_norm_residual_k,
            hidden,
            &self.input_norm,
            normed,
            residual,
            k as u32,
            h as u32,
            eps,
            stream,
        )?;

        // 2. Batched in_proj for all K tokens: ONE weight sweep.
        let proj = ctx.buffers.ssm_qkvz();
        self.batched_in_proj(normed, proj, k as u32, h as u32, true, ctx, stream)?;

        // 3. Sequential conv + scan, one token at a time on the single live
        //    state, plus the per-token snapshots the rollback contract reads.
        //    Snapshotting here (not after the whole body) is correct because
        //    steps 4-6 never touch h_state/conv_state.
        let xbc_base = ctx.buffers.ssm_deinterleaved();
        let y_base = ctx.buffers.attn_output();
        for t in 0..k {
            let ssm = state
                .as_any_mut()
                .downcast_mut::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))?;
            let proj_t = proj.offset(t * self.in_proj_size * bf16);
            let xbc_t = xbc_base.offset(t * self.d_xbc * bf16);
            self.conv1d_update_biased(
                ctx.gpu,
                ssm.conv_state,
                proj_t.offset(self.d_inner * bf16),
                xbc_t,
                self.d_xbc as u32,
                self.d_conv as u32,
                1,
                stream,
            )?;
            self.ssm_decode(
                ctx.gpu,
                ssm.h_state,
                xbc_t,
                xbc_t.offset(self.d_inner * bf16),
                xbc_t.offset((self.d_inner + gs) * bf16),
                proj_t.offset((self.d_inner + self.d_xbc) * bf16),
                y_base.offset(t * self.d_inner * bf16),
                1,
                stream,
            )?;
            // Same snapshot set, same indices, same sizes as the sequential
            // body — see `decode_batched`'s doc comment for the contract.
            if t < ssm.h_state_intermediates.len() {
                ctx.gpu.copy_d2d_async(
                    ssm.h_state,
                    ssm.h_state_intermediates[t],
                    self.h_state_bytes,
                    stream,
                )?;
            }
            if t < ssm.conv_state_intermediates.len() {
                ctx.gpu.copy_d2d_async(
                    ssm.conv_state,
                    ssm.conv_state_intermediates[t],
                    self.conv_state_bytes,
                    stream,
                )?;
            }
        }

        // 4. Batched gated RMS norm. y rows are d_inner-packed; the z gate
        //    lives at in_proj_size stride inside proj (explicit gate_stride).
        let gated = ctx.buffers.norm_output();
        let group_size = (self.d_inner / self.n_groups) as u32;
        ops::gated_rms_norm(
            ctx.gpu,
            self.gated_rms_norm_k,
            y_base,
            proj,
            &self.ssm.ssm_norm,
            gated,
            k as u32,
            self.d_inner as u32,
            self.in_proj_size as u32,
            eps,
            group_size,
            stream,
        )?;

        // 5. Batched out_proj. MUST NOT target ssm_qkvz: proj still holds the
        //    z gate step 4 just read (the documented WAR hazard).
        let out = ctx.buffers.qkv_output();
        self.batched_out_proj(gated, out, k as u32, h as u32, true, ctx, stream)?;

        // 6. Batched residual add over all K rows.
        ops::residual_add(
            ctx.gpu,
            self.residual_add_k,
            hidden,
            out,
            (k * h) as u32,
            stream,
        )?;
        Ok(())
    }
}
