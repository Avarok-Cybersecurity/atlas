// SPDX-License-Identifier: AGPL-3.0-only

//! Section 10 of `prefill_attention_paged`: O-projection GEMM
//! `[N, nq*hd] → [N, h]`. 6-way quantization dispatch (FP8 transposed,
//! FP8, FP8 col-scale, NVFP4 transposed, BF16 dense, NVFP4 default).
//! Extracted from `paged.rs` to keep that file under 500 LoC.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_attention_paged_oproj(
        &self,
        attn_out: DevicePtr,
        n: u32,
        h: u32,
        nq: u32,
        hd: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let o_out = ctx.buffers.norm_output();
        let force_w8a8 = ops::fp8_blockscaled_prefill_enabled();
        // Native FP4 o_proj: quantize attn_out to NVFP4 and consume o_proj in
        // its original NVFP4 form. OPT-IN ONLY -- see the QKV path's comment.
        let w4a4 = n >= 256
            && self.w4a4_gemm_k.0 != 0
            && self.quantize_nvfp4_k.0 != 0
            && ctx.buffers.fp8_act_bytes() >= (n as usize) * (nq as usize) * (hd as usize)
            && std::env::var("ATLAS_ATTN_W4A4").is_ok();
        if w4a4 {
            let kd = nq * hd;
            let a4 = ctx.buffers.fp8_act();
            let a4_sf = a4.offset((n as usize) * (kd as usize) / 2);
            ops::quantize_bf16_to_nvfp4(
                ctx.gpu,
                self.quantize_nvfp4_k,
                attn_out,
                a4,
                a4_sf,
                n,
                kd,
                stream,
            )?;
            ops::w4a4_gemm_mfast(
                ctx.gpu,
                self.w4a4_gemm_k,
                a4,
                a4_sf,
                &self.attn.o_proj,
                o_out,
                n,
                h,
                kd,
                stream,
            )?;
        } else if ops::cutlass_nvfp4_attn_o_enabled()
            && let Some(ref nvfp4_t) = self.o_nvfp4_t
        {
            ops::log_cutlass_nvfp4_route("attn_o", n, h, nq * hd);
            ops::cutlass_nvfp4_proj(ctx.gpu, attn_out, nvfp4_t, o_out, n, h, nq * hd, stream)?;
        } else if ops::cutlass_nvfp4_attn_o_enabled()
            && let Some(fp8w) = self.o_weight.as_ref().and_then(|w| w.as_fp8())
        {
            ops::log_cutlass_nvfp4_route("attn_o", n, h, nq * hd);
            ops::cutlass_nvfp4_proj_from_fp8(
                ctx.gpu,
                attn_out,
                fp8w,
                o_out,
                n,
                h,
                nq * hd,
                stream,
            )?;
        } else if ops::cublas_gemm_enabled()
            && let Some(fp8w) = self.o_weight.as_ref().and_then(|w| w.as_fp8())
        {
            // cuBLASLt BF16 (3x the hand-written mma.sync GEMM on GB10).
            ops::cublas_bf16_proj(ctx.gpu, attn_out, fp8w, o_out, n, h, nq * hd, stream)?;
        } else if force_w8a8
            && let Some(fp8w) = self.o_weight.as_ref().and_then(|w| w.as_fp8())
            && self.per_token_group_quant_fp8_k.0 != 0
            && self.fp8_gemm_t_blockscaled_k.0 != 0
        {
            // o_proj GEMM: C[M, N] = A[M, K] @ B[N, K]
            //   A = attn_out  : [num_tokens, nq*hd]  (input,  K = nq*hd)
            //   B = o_weight  : [h, nq*hd]           (stored row-major, N = h)
            //   C = o_out     : [num_tokens, h]      (output, N = h)
            let m = n as usize;
            let k_dim = (nq * hd) as usize; // inner contract dim — input width of o_proj
            let n_out = h as usize; // output width of o_proj
            // Persistent arena scratch (no per-projection alloc/sync/free): the
            // quant→GEMM→next-layer chain is same-stream ordered, so the buffer
            // is safely reused without a host sync.
            let a_fp8_buf = ctx.buffers.fp8_act();
            let a_scale_buf = ctx.buffers.fp8_act_scale();
            debug_assert!(m * k_dim <= ctx.buffers.fp8_act_bytes());
            ops::per_token_group_quant_fp8(
                ctx.gpu,
                self.per_token_group_quant_fp8_k,
                attn_out,
                a_fp8_buf,
                a_scale_buf,
                n,
                nq * hd,
                stream,
            )?;
            ops::fp8_gemm_t_blockscaled(
                ctx.gpu,
                self.fp8_gemm_t_blockscaled_k,
                a_fp8_buf,
                a_scale_buf,
                fp8w.weight,
                fp8w.row_scale,
                o_out,
                n,
                n_out as u32,
                k_dim as u32,
                stream,
            )?;
        } else if let Some(ref fp8t) = self.o_fp8w_t
            && self.w8a16_gemm_t_pipelined_k.0 != 0
        {
            // o_proj via the byte-identical ~4.2x faster pipelined transposed
            // tensor-core kernel where available (NVIDIA). gfx1151/HIP has no
            // cp.async -> pipelined absent -> non-pipelined w8a16_gemm_t below.
            ops::w8a16_gemm_t_pipelined(
                ctx.gpu,
                self.w8a16_gemm_t_pipelined_k,
                attn_out,
                fp8t.weight_t,
                fp8t.scale_t,
                o_out,
                n,
                h,
                nq * hd,
                stream,
            )?;
        } else if let Some(ref fp8t) = self.o_fp8w_t
            && self.w8a16_gemm_t_k.0 != 0
        {
            // cp.async-free fallback (gfx1151/HIP): non-pipelined transposed W8A16.
            ops::w8a16_gemm_t(
                ctx.gpu,
                self.w8a16_gemm_t_k,
                attn_out,
                fp8t.weight_t,
                fp8t.scale_t,
                o_out,
                n,
                h,
                nq * hd,
                stream,
            )?;
        } else if self.o_weight.as_ref().and_then(|w| w.as_fp8()).is_some()
            && self.w8a16_gemm_k.0 != 0
        {
            let fp8w = self.o_weight.as_ref().and_then(|w| w.as_fp8()).unwrap();
            ops::w8a16_gemm(
                ctx.gpu,
                self.w8a16_gemm_k,
                attn_out,
                fp8w.weight,
                fp8w.row_scale,
                o_out,
                n,
                h,
                nq * hd,
                stream,
            )?;
        } else if let Some(fp8) = self.o_fp8 {
            if n > 128 {
                ops::fp8_gemm_n128_m128(
                    ctx.gpu,
                    self.fp8_gemm_t_m128_k,
                    attn_out,
                    fp8,
                    o_out,
                    n,
                    h,
                    nq * hd,
                    stream,
                )?;
            } else {
                ops::fp8_gemm_n128(
                    ctx.gpu,
                    self.fp8_gemm_k,
                    attn_out,
                    fp8,
                    o_out,
                    n,
                    h,
                    nq * hd,
                    stream,
                )?;
            }
        } else if let Some(ref nvfp4_t) = self.o_nvfp4_t {
            if n > 128 {
                self.w4a16_gemm_m128_dispatch(
                    ctx.gpu,
                    attn_out,
                    nvfp4_t,
                    o_out,
                    n,
                    h,
                    nq * hd,
                    stream,
                )?;
            } else {
                ops::w4a16_gemm_n128(
                    ctx.gpu,
                    self.w4a16_gemm_t_k,
                    attn_out,
                    nvfp4_t,
                    o_out,
                    n,
                    h,
                    nq * hd,
                    stream,
                )?;
            }
        } else if let Some(o_bf16) = self.o_dense_bf16.as_ref() {
            // BF16 dense fallback (Gemma-4 dense per Nvidia ModelOpt's
            // ignore list — all self_attn projections must stay BF16).
            // Tensor-core pipelined GEMM (~40× scalar on large-M prefill).
            if self.dense_gemm_pipelined_k.0 != 0 {
                ops::dense_gemm_bf16_pipelined(
                    ctx.gpu,
                    self.dense_gemm_pipelined_k,
                    attn_out,
                    o_bf16,
                    o_out,
                    n,
                    h,
                    nq * hd,
                    stream,
                )?;
            } else {
                ops::dense_gemm(
                    ctx.gpu,
                    self.dense_gemm_k,
                    attn_out,
                    o_bf16,
                    o_out,
                    n,
                    h,
                    nq * hd,
                    stream,
                )?;
            }
        } else {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm_k,
                attn_out,
                &self.attn.o_proj,
                o_out,
                n,
                h,
                nq * hd,
                stream,
            )?;
        }
        // ── LoRA delta on o_proj: o_out[n,h] += scale·(attn_out[n,nq*hd]@Aᵀ)@Bᵀ.
        // attn_out is the exact o_proj input (post-attention), matching HF.
        // Before the op_dump so dumps show the adapted output.
        if let Some(ref lw) = self.lora
            && let Some(ref pair) = lw.o
        {
            debug_assert_eq!(pair.k_in, nq * hd);
            debug_assert_eq!(pair.n_out, h);
            // #30 routed-prefill precision (see paged_qkv.rs): a routed (non-active)
            // prefill selects the REQUEST slot's O pair and folds it through the SAME
            // dense `apply_lora_delta` (dense_gemm_tc) the active adapter uses. Indexed
            // by `lw.layer_idx` (GLOBAL layer index, not attn_layer_idx). MUST win over
            // the bgmv branch (a routed prefill satisfies both). `None` → bgmv/installed.
            let routed_pair = ctx.routed_lora_layers.and_then(|ls| {
                crate::lora::select_routed_pair(ls, lw.layer_idx, crate::lora::LoraModule::OProj)
            });
            // Request-scoped routing (see paged_qkv.rs). attn_out is contiguous
            // [n, nq*hd] and o_out is contiguous [n, h], so the routed bgmv is
            // byte-identical to `n` single-row `apply_lora_delta`. No pool / no
            // route → installed-active-pair path (pre-M2 behaviour).
            let seq_slot = ctx
                .attn_metadata
                .map(|m| m.seq_slot)
                .unwrap_or(DevicePtr(0));
            if let Some(routed_pair) = routed_pair {
                debug_assert_eq!(routed_pair.k_in, nq * hd);
                debug_assert_eq!(routed_pair.n_out, h);
                ops::lora_delta::apply_lora_delta(
                    ctx.gpu,
                    &lw.kernels,
                    routed_pair,
                    attn_out,
                    o_out,
                    n,
                    ctx.buffers.lora_xa(),
                    ctx.buffers.lora_delta(),
                    stream,
                )?;
            } else if seq_slot.0 != 0
                && let Some(ref route) = lw.o_route
            {
                ops::lora_delta::apply_lora_bgmv(
                    ctx.gpu,
                    &lw.kernels,
                    route,
                    attn_out,
                    o_out,
                    seq_slot,
                    n,
                    pair.k_in,
                    pair.n_out,
                    ctx.buffers.lora_xa(),
                    stream,
                )?;
            } else {
                ops::lora_delta::apply_lora_delta(
                    ctx.gpu,
                    &lw.kernels,
                    pair,
                    attn_out,
                    o_out,
                    n,
                    ctx.buffers.lora_xa(),
                    ctx.buffers.lora_delta(),
                    stream,
                )?;
            }
        }
        // ATLAS_OP_DUMP hook: post-O-projection — this is the FULL attention
        // block output (Q*K^T*V * O_proj). Compares 1:1 against the HF
        // module hooked on `full_attention.o_proj.forward` for the last
        // token. Use `n` as token-count so we slice the last token.
        let bf16 = 2usize;
        let num_tokens = n as usize;
        if num_tokens > 0 {
            super::super::op_dump::dump_bf16(
                ctx.gpu,
                o_out,
                (num_tokens - 1) * h as usize * bf16,
                h as usize,
                self.attn_layer_idx,
                "o_proj",
                stream,
            )?;
        }
        Ok(o_out)
    }
}
