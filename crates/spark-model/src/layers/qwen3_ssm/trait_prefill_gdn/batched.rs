// SPDX-License-Identifier: AGPL-3.0-only

//! prefill_gdn_full — batched / varlen co-dispatch GDN scans.

use super::super::*;
use super::gdn_batched_fla_enabled;

impl Qwen3SsmLayer {
    /// Q12 Path B: batched GDN recurrence — mirrors prefill_gdn_full_inner
    /// dispatch ladder but routes to the `*_batched` kernel variants and
    /// passes `h_state_ptrs` (device array of N pointers) instead of a
    /// single h_state device pointer.
    ///
    /// Constraint: scheduler-enforced same-chunk-len across all N streams.
    /// `gdn_bufs.qkv` / `gate_beta` / `output` are stacked
    /// `[batch_size, chunk_len, *]` contiguous in memory. Each batch
    /// element's QKV starts at `b * chunk_len * conv_dim` (BF16).
    ///
    /// Validation status: kernels unvalidated against hardware.
    pub(crate) fn prefill_gdn_full_batched_inner(
        &self,
        h_state_ptrs: spark_runtime::gpu::DevicePtr,
        gdn_bufs: &GdnPrefillBuffers,
        batch_size: u32,
        chunk_len: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let conv_dim = key_dim * 2 + value_dim;
        let bf16 = 2usize;
        let fp32 = 4usize;

        let q_ptr = gdn_bufs.qkv;
        let k_ptr = gdn_bufs.qkv.offset(key_dim * bf16);
        let v_ptr = gdn_bufs.qkv.offset(key_dim * 2 * bf16);
        let gate_ptr = gdn_bufs.gate_beta;
        let beta_ptr = gdn_bufs.gate_beta.offset(nv * fp32);
        let gb_stride = (nv * 2) as u32;

        // ── Batched FLA scan (ATLAS_GDN_BATCHED_FLA) ──
        // Route the co-dispatched GDN through the chunk-parallel FLA kernels at
        // batch=N instead of the occupancy-starved wy64 [nv,batch]. FLA's
        // chunk_delta_h grid is [nv,batch] too, but at batch=N gives 32N CTAs
        // (fills GB10's 48 SMs) vs wy64's same count on a slower persistent
        // kernel; recompute_wu/chunk_fwd_o add a num_chunks grid axis. h_state is
        // passed as the per-request POINTER TABLE (is_table=true) — same table
        // wy64 uses, so no gather/scatter. Scratch regions span the whole batch:
        // base=(b*num_chunks+c)*nv, so size by total_nt = batch*num_chunks.
        if gdn_batched_fla_enabled() && kd == 128 && vd == 128 {
            let fla_scratch = ctx.buffers.gdn_fla_scratch();
            if fla_scratch.0 != 0
                && self.gdn_prefill_fla_recompute_wu_k.0 != 0
                && self.gdn_prefill_fla_chunk_delta_h_k.0 != 0
                && self.gdn_prefill_fla_chunk_fwd_o_k.0 != 0
            {
                let num_chunks = chunk_len.div_ceil(64);
                let total_nt = (batch_size * num_chunks) as usize;
                let w_out = fla_scratch;
                let u_out = w_out.offset(total_nt * nv * 64 * kd * bf16);
                let s_out = u_out.offset(total_nt * nv * 64 * vd * bf16);
                let uc_out = s_out.offset(total_nt * nv * kd * vd * bf16);
                let gc_out = uc_out.offset(total_nt * nv * 64 * vd * bf16);
                return ops::gdn_prefill_fla(
                    ctx.gpu,
                    self.gdn_prefill_fla_recompute_wu_k,
                    self.gdn_prefill_fla_chunk_delta_h_k,
                    self.gdn_prefill_fla_chunk_delta_h_tc_vblock_k,
                    self.gdn_prefill_fla_chunk_fwd_o_k,
                    h_state_ptrs, // per-request pointer table
                    q_ptr,
                    k_ptr,
                    v_ptr,
                    gate_ptr,
                    beta_ptr,
                    gdn_bufs.output,
                    w_out,
                    u_out,
                    s_out,
                    uc_out,
                    gc_out,
                    batch_size,
                    chunk_len,
                    num_chunks,
                    nk as u32,
                    nv as u32,
                    kd as u32,
                    vd as u32,
                    conv_dim as u32,
                    conv_dim as u32,
                    gb_stride,
                    true,                                // h_state_is_table
                    spark_runtime::gpu::DevicePtr::NULL, // cu_seqlens (uniform)
                    spark_runtime::gpu::DevicePtr::NULL, // cu_chunks (uniform)
                    false,                               // uniform batched (not varlen)
                    ctx.profile,
                    stream,
                );
            }
        }

        // Mirror the single-stream dispatch ladder. Total tokens per stream
        // is `chunk_len`; the kernel internally processes `batch_size` such
        // streams (grid dim Y).
        if self.gdn_prefill_wy32_batched_k.0 != 0 && chunk_len > 32 {
            // #110: dynamic smem must cover the FULL kernel layout (H + smem_k +
            // smem_q + smem_warp[4] + smem_kd[C*C] + smem_g[C] + smem_bt[C], C=32).
            // The old `+256` slack under-counted the smem_warp(16)+smem_g(128)+
            // smem_bt(128)=272 trailer by 16 B, so the kernel's smem_bt tail wrote
            // past the requested allocation → CUDA illegal access under live
            // occupancy (compute-sanitizer: Invalid __shared__ write at +0xce0).
            let smem =
                (kd * vd * 4 + 32 * kd * 2 + 32 * kd * 2 + 32 * 32 * 4 + (4 + 32 + 32) * 4) as u32;
            ops::gdn_prefill_persistent_smem_batched(
                ctx.gpu,
                self.gdn_prefill_wy32_batched_k,
                h_state_ptrs,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_bufs.output,
                batch_size,
                chunk_len,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                smem,
                stream,
            )?;
        } else if self.gdn_prefill_persistent_wy4_batched_k.0 != 0 {
            let smem = (kd * vd * 4 + 8 * kd * 4 + 56) as u32;
            ops::gdn_prefill_persistent_smem_batched(
                ctx.gpu,
                self.gdn_prefill_persistent_wy4_batched_k,
                h_state_ptrs,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_bufs.output,
                batch_size,
                chunk_len,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                smem,
                stream,
            )?;
        } else if (256..=4096).contains(&chunk_len) && self.gdn_prefill_persistent_batched_k.0 != 0
        {
            ops::gdn_prefill_persistent_batched(
                ctx.gpu,
                self.gdn_prefill_persistent_batched_k,
                h_state_ptrs,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_bufs.output,
                batch_size,
                chunk_len,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                stream,
            )?;
        } else if self.gdn_prefill_split4_batched_k.0 != 0 {
            ops::gdn_prefill_split4_batched(
                ctx.gpu,
                self.gdn_prefill_split4_batched_k,
                h_state_ptrs,
                q_ptr,
                k_ptr,
                v_ptr,
                gate_ptr,
                beta_ptr,
                gdn_bufs.output,
                batch_size,
                chunk_len,
                nk as u32,
                nv as u32,
                kd as u32,
                vd as u32,
                conv_dim as u32,
                conv_dim as u32,
                gb_stride,
                stream,
            )?;
        } else {
            anyhow::bail!(
                "Qwen3SsmLayer::prefill_gdn_full_batched_inner: no batched GDN \
                 kernel handle is loaded for this target — caller should fall \
                 back to per-stream prefill_gdn_full."
            );
        }

        Ok(())
    }

    /// VARLEN batched GDN scan: route the ragged (non-uniform-length) co-dispatch
    /// batch through ONE `gdn_prefill_fla(batch=N, is_varlen)` call instead of the
    /// per-request loop. cu_seqlens (device) gives per-stream token offsets; the
    /// kernel computes per-stream chunk offsets in-kernel. h_state via the pointer
    /// table (is_table). Returns false if not eligible (caller loops).
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn prefill_gdn_full_batched_fla_varlen_inner(
        &self,
        h_state_ptrs: spark_runtime::gpu::DevicePtr,
        gdn_bufs: &GdnPrefillBuffers,
        batch_size: u32,
        cu_seqlens: spark_runtime::gpu::DevicePtr,
        max_num_chunks: u32,
        total_nt: usize,
        max_seqlen: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<bool> {
        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let fla_scratch = ctx.buffers.gdn_fla_scratch();
        if !gdn_batched_fla_enabled()
            || kd != 128
            || vd != 128
            || fla_scratch.0 == 0
            || cu_seqlens.0 == 0
            || self.gdn_prefill_fla_recompute_wu_k.0 == 0
            || self.gdn_prefill_fla_chunk_delta_h_k.0 == 0
            || self.gdn_prefill_fla_chunk_fwd_o_k.0 == 0
        {
            return Ok(false); // not eligible → caller falls back to the per-request loop
        }
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        let conv_dim = key_dim * 2 + value_dim;
        let bf16 = 2usize;
        let fp32 = 4usize;
        let q_ptr = gdn_bufs.qkv;
        let k_ptr = gdn_bufs.qkv.offset(key_dim * bf16);
        let v_ptr = gdn_bufs.qkv.offset(key_dim * 2 * bf16);
        let gate_ptr = gdn_bufs.gate_beta;
        let beta_ptr = gdn_bufs.gate_beta.offset(nv * fp32);
        let gb_stride = (nv * 2) as u32;
        // Scratch regions span the whole batch: base=(choff+c)*nv, sized by total_nt.
        let w_out = fla_scratch;
        let u_out = w_out.offset(total_nt * nv * 64 * kd * bf16);
        let s_out = u_out.offset(total_nt * nv * 64 * vd * bf16);
        let uc_out = s_out.offset(total_nt * nv * kd * vd * bf16);
        let gc_out = uc_out.offset(total_nt * nv * 64 * vd * bf16);
        ops::gdn_prefill_fla(
            ctx.gpu,
            self.gdn_prefill_fla_recompute_wu_k,
            self.gdn_prefill_fla_chunk_delta_h_k,
            self.gdn_prefill_fla_chunk_delta_h_tc_vblock_k,
            self.gdn_prefill_fla_chunk_fwd_o_k,
            h_state_ptrs,
            q_ptr,
            k_ptr,
            v_ptr,
            gate_ptr,
            beta_ptr,
            gdn_bufs.output,
            w_out,
            u_out,
            s_out,
            uc_out,
            gc_out,
            batch_size,
            max_seqlen,     // seq_len: cosmetic (varlen kernel reads cu_seqlens)
            max_num_chunks, // grid x = MAX per-stream chunks
            nk as u32,
            nv as u32,
            kd as u32,
            vd as u32,
            conv_dim as u32,
            conv_dim as u32,
            gb_stride,
            true,                                // h_state_is_table
            cu_seqlens,                          // device cu_seqlens (token offsets)
            spark_runtime::gpu::DevicePtr::NULL, // cu_chunks (computed in-kernel)
            true,                                // is_varlen
            ctx.profile,
            stream,
        )?;
        Ok(true)
    }
}
