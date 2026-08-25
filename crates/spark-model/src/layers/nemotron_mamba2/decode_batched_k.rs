// SPDX-License-Identifier: AGPL-3.0-only

//! K-token verify body for Nemotron-H Mamba-2 — the layer-side half of the
//! speculative-verify contract (DFlash K=γ+1 and the MTP K=2..4 ladder).
//!
//! Ported from #475's `decode_mtp_k` (milestone B) onto this branch's #545
//! helpers. Without this override the trait's default `decode_batched` loops
//! single-token `decode()` calls: the recurrent state marches through all K
//! tokens with NO per-position snapshots, so a partial accept cannot rewind —
//! the SSM state is poisoned past the accepted boundary and every subsequent
//! token decodes from a future that was rejected. That (plus the 0-byte
//! checkpoint copies fixed alongside in verify_a/async_chkpt) is why the
//! first Lightning + DFlash boot accepted nothing.
//!
//! The verify rows are sequential TIME STEPS of one sequence — token t+1's
//! recurrent state depends on token t — so conv+scan stay a `t` loop at
//! batch=1 on the single live state. What batching flips is WEIGHT traffic:
//! every other phase is row-independent, and running the in/out projections
//! at M=K collapses K sweeps of the ~38.7M-param pair into one. The batched
//! projection arms (`batched_in_proj`/`batched_out_proj`, #545) carry their
//! own measured bit-parity rungs and fall back to per-row loops where a twin
//! is unproven, so acceptance is identical to the sequential body either way.
//!
//! Snapshot contract (same as GDN's): `h_state_intermediates[t]` /
//! `conv_state_intermediates[t]` hold the state AFTER verify token t; the
//! rollback in verify_a/async_chkpt restores index `num_accepted - 1`.
//! Snapshots are pool-backed (bound in meta.rs when `has_mtp`), so the vec
//! lengths are the capacity gates — a slot without pool intermediates simply
//! keeps the (correct, unrewindable) sequential behavior and the rollback
//! pre-validation refuses cleanly.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::NemotronMamba2Layer;
use crate::layer::{ForwardContext, LayerState, SsmLayerState, TransformerLayer};
use crate::layers::ops;

/// `ATLAS_NEMOTRON_VERIFY_BATCHED=1` opts the K-token verify into the
/// batched-phase body below. DEFAULT OFF: the batched projection arms are
/// bit-parity-proven for the multi-seq rungs (#545) but NOT yet for the
/// K-row verify shapes on this target, and with 0 accepts the committed
/// stream IS the verify body's row 0 — any non-parity there rewrites the
/// model's output (measured 2026-08-25: batched-unconditional verify
/// degenerated long generations that the sequential path serves at 75.0
/// tok/s). Sequential-with-snapshots is byte-identical to serial decode
/// by construction; flip this only with a measured A/B against it.
fn verify_batched_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var("ATLAS_NEMOTRON_VERIFY_BATCHED").as_deref() == Ok("1"))
}

impl NemotronMamba2Layer {
    /// K-token verify dispatch: sequential (default, byte-identical to
    /// serial `decode()`) or the batched-phase body under the A/B env.
    /// Both write the per-token snapshots the rollback contract reads.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn decode_verify_k(
        &self,
        hidden: DevicePtr,
        residual: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        kv_cache: &mut spark_runtime::kv_cache::PagedKvCache,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if verify_batched_enabled() && num_tokens >= 2 {
            return self.decode_batched_k(hidden, residual, num_tokens, state, ctx, stream);
        }
        // Sequential body: the same per-token `decode()` loop the trait
        // default runs (same offsets, same scratch reuse), plus the
        // snapshot after each token that the default loop lacks.
        let h = ctx.config.hidden_size;
        for t in 0..num_tokens {
            let off = t * h * 2;
            self.decode(
                hidden.offset(off),
                residual.offset(off),
                state,
                kv_cache,
                seq_len + t,
                block_table,
                disk_block_ids,
                disk_last_offloaded_per_layer,
                ctx,
                stream,
            )?;
            let ssm = state
                .as_any_mut()
                .downcast_mut::<SsmLayerState>()
                .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))?;
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
        Ok(())
    }

    /// K-token verify body: batched norms + projections, sequential
    /// conv + scan with per-token state snapshots.
    #[allow(clippy::too_many_arguments)]
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

        // 1. Batched input norm + residual save — one block per row, exactly
        //    the K single-row reductions the sequential body performs.
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

        // 2. Batched in_proj for all K rows: ONE weight sweep.
        //    Layout per row: [z(d_inner) | xBC(d_xbc) | dt(num_heads)].
        let proj = ctx.buffers.ssm_qkvz();
        self.batched_in_proj(normed, proj, k as u32, h as u32, ctx, stream)?;

        // 3. Sequential conv + scan, one token at a time on the single live
        //    state, snapshotting after each token. Snapshotting here (not
        //    after the whole body) is correct because steps 4-6 never touch
        //    h_state / conv_state.
        let xbc_out_base = ctx.buffers.ssm_deinterleaved();
        let y_base = ctx.buffers.attn_output();
        let ssm = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))?;
        for t in 0..k {
            let proj_t = proj.offset(t * self.in_proj_size * bf16);
            let xbc_ptr = proj_t.offset(self.d_inner * bf16);
            let dt_ptr = proj_t.offset((self.d_inner + self.d_xbc) * bf16);
            let xbc_out = xbc_out_base.offset(t * self.d_xbc * bf16);

            self.conv1d_update_biased(
                ctx.gpu,
                ssm.conv_state,
                xbc_ptr,
                xbc_out,
                self.d_xbc as u32,
                self.d_conv as u32,
                1,
                stream,
            )?;

            let x_ptr = xbc_out;
            let b_ptr = xbc_out.offset(self.d_inner * bf16);
            let c_ptr = xbc_out.offset((self.d_inner + gs) * bf16);
            self.ssm_decode(
                ctx.gpu,
                ssm.h_state,
                x_ptr,
                b_ptr,
                c_ptr,
                dt_ptr,
                y_base.offset(t * self.d_inner * bf16),
                1,
                stream,
            )?;

            // Per-token snapshots — the rollback contract's read side
            // (verify_a / async_chkpt restore index num_accepted-1). The
            // vec lengths gate capacity: the H pool is tiered (K-1 wide),
            // so the final token's H snapshot may be intentionally absent —
            // a full accept never rewinds to it.
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

        // 4. Batched gated RMS norm over all K rows: y rows are
        //    d_inner-packed; the z gate lives at in_proj_size stride inside
        //    proj (gate_stride argument).
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

        // 5. Batched out_proj. MUST NOT target ssm_qkvz: step 4 read the z
        //    gate from `proj` and a same-buffer write is the documented WAR
        //    hazard on this path.
        let out = ctx.buffers.qkv_output();
        self.batched_out_proj(gated, out, k as u32, h as u32, ctx, stream)?;

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
