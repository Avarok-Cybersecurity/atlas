// SPDX-License-Identifier: AGPL-3.0-only

//! Standard Q/K/V projection branch of `prefill_attention_paged`.
//! 6-way quantization dispatch (transposed-FP8, FP8, FP8 col-scale,
//! NVFP4 transposed, NVFP4, BF16) shared across Q, K, V; extracted to
//! keep `paged.rs` under the 500-LoC budget.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

/// Identifies which projection (Q/K/V) — selects the correct weight bank
/// from `Qwen3AttentionLayer`.
pub(super) enum Proj {
    Q,
    K,
    V,
}

impl Qwen3AttentionLayer {
    /// Run the Q, K, and V GEMMs (in that order) for non-MLA prefill.
    /// Output destinations follow the existing buffer layout in `paged.rs`.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_attention_paged_qkv(
        &self,
        normed: DevicePtr,
        n: u32,
        h: u32,
        nkv: u32,
        hd: u32,
        q_proj_dim: usize,
        kv_dim: usize,
        num_tokens: usize,
        bf16: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        // Native FP4 path: quantize `normed` to NVFP4 ONCE and share it across
        // the Q, K and V GEMMs (same input). OPT-IN ONLY (ATLAS_ATTN_W4A4=1):
        // although the input is normed (the distribution class that gated clean
        // on the SSM projections), the outputs here are attention LOGIT inputs,
        // and same-binary A/B on long prompts showed the hallucinated-multiple-
        // choice signature with this on and clean answers with it off -- small
        // perturbations in q/k reroute attention. It also measured only ~2 ms.
        let w4a4 = n >= 256
            && self.w4a4_gemm_k.0 != 0
            && self.quantize_nvfp4_k.0 != 0
            && self.q_weight.as_ref().and_then(|w| w.as_nvfp4()).is_some()
            && ctx.buffers.fp8_act_bytes() >= (n as usize) * (h as usize)
            && std::env::var("ATLAS_ATTN_W4A4").is_ok();
        let a4 = if w4a4 {
            let a4 = ctx.buffers.fp8_act();
            let a4_sf = a4.offset((n as usize) * (h as usize) / 2);
            ops::quantize_bf16_to_nvfp4(
                ctx.gpu,
                self.quantize_nvfp4_k,
                normed,
                a4,
                a4_sf,
                n,
                h,
                stream,
            )?;
            Some((a4, a4_sf))
        } else {
            None
        };
        let qg_out = ctx.buffers.qkv_output();
        self.prefill_one_proj(
            Proj::Q,
            normed,
            qg_out,
            n,
            q_proj_dim as u32,
            h,
            a4,
            ctx,
            stream,
        )?;
        // ATLAS_OP_DUMP hook: q_proj output (last token, BF16 → f32).
        // For gated Qwen3.6, q_proj_dim = 2*q_dim (Q+Gate interleaved).
        // We dump the FULL Q+Gate buffer; the HF reference will only
        // contain the deinterleaved Q so partial cosine on first half
        // is the comparable metric.
        super::super::op_dump::dump_bf16(
            ctx.gpu,
            qg_out,
            (num_tokens - 1) * q_proj_dim * bf16,
            q_proj_dim,
            self.attn_layer_idx,
            "q_proj_full",
            stream,
        )?;

        let k_contiguous = ctx.buffers.ssm_qkvz();
        self.prefill_one_proj(
            Proj::K,
            normed,
            k_contiguous,
            n,
            nkv * hd,
            h,
            a4,
            ctx,
            stream,
        )?;
        super::super::op_dump::dump_bf16(
            ctx.gpu,
            k_contiguous,
            (num_tokens - 1) * kv_dim * bf16,
            kv_dim,
            self.attn_layer_idx,
            "k_proj",
            stream,
        )?;

        let v_contiguous = k_contiguous.offset(num_tokens * kv_dim * bf16);
        self.prefill_one_proj(
            Proj::V,
            normed,
            v_contiguous,
            n,
            nkv * hd,
            h,
            a4,
            ctx,
            stream,
        )?;
        super::super::op_dump::dump_bf16(
            ctx.gpu,
            v_contiguous,
            (num_tokens - 1) * kv_dim * bf16,
            kv_dim,
            self.attn_layer_idx,
            "v_proj",
            stream,
        )?;
        Ok(())
    }

    #[allow(clippy::too_many_arguments)]
    fn prefill_one_proj(
        &self,
        proj: Proj,
        normed: DevicePtr,
        out: DevicePtr,
        n: u32,
        out_dim: u32,
        h: u32,
        a4: Option<(DevicePtr, DevicePtr)>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let (fp8w_t, weight_opt, fp8, nvfp4_t, dense, label) = match proj {
            Proj::Q => (
                self.q_fp8w_t.as_ref(),
                self.q_weight.as_ref(),
                self.q_fp8,
                self.q_nvfp4_t.as_ref(),
                &self.attn.q_proj,
                "attn_q",
            ),
            Proj::K => (
                self.k_fp8w_t.as_ref(),
                self.k_weight.as_ref(),
                self.k_fp8,
                self.k_nvfp4_t.as_ref(),
                &self.attn.k_proj,
                "attn_k",
            ),
            Proj::V => (
                self.v_fp8w_t.as_ref(),
                self.v_weight.as_ref(),
                self.v_fp8,
                self.v_nvfp4_t.as_ref(),
                &self.attn.v_proj,
                "attn_v",
            ),
        };

        // Native FP4: pre-quantized activations x original NVFP4 weights.
        if let (Some((a4p, a4sf)), Some(nvfp4)) = (a4, weight_opt.and_then(|w| w.as_nvfp4())) {
            let _ = label;
            return ops::w4a4_gemm_mfast(
                ctx.gpu,
                self.w4a4_gemm_k,
                a4p,
                a4sf,
                nvfp4,
                out,
                n,
                out_dim,
                h,
                stream,
            );
        }

        let force_w8a8 = ops::fp8_blockscaled_prefill_enabled();
        // W8A8 + FP32 epilogue: requires NON-transposed FP8 weights with
        // block scales (matches the kernel signature). The attn layer stores
        // those via set_fp8_weights — accessible via weight_opt.as_fp8().
        if ops::cutlass_nvfp4_attn_qkv_enabled(label)
            && let Some(nvfp4_t) = nvfp4_t
        {
            ops::log_cutlass_nvfp4_route(label, n, out_dim, h);
            ops::cutlass_nvfp4_proj(ctx.gpu, normed, nvfp4_t, out, n, out_dim, h, stream)?;
        } else if ops::cutlass_nvfp4_attn_qkv_enabled(label)
            && let Some(fp8w) = weight_opt.and_then(|w| w.as_fp8())
        {
            ops::log_cutlass_nvfp4_route(label, n, out_dim, h);
            ops::cutlass_nvfp4_proj_from_fp8(ctx.gpu, normed, fp8w, out, n, out_dim, h, stream)?;
        } else if force_w8a8
            && let Some(fp8w) = weight_opt.and_then(|w| w.as_fp8())
            && self.per_token_group_quant_fp8_k.0 != 0
            && self.fp8_gemm_t_blockscaled_k.0 != 0
        {
            let m = n as usize;
            let k_dim = h as usize;
            // Persistent arena scratch (no per-projection alloc/sync/free).
            let a_fp8_buf = ctx.buffers.fp8_act();
            let a_scale_buf = ctx.buffers.fp8_act_scale();
            debug_assert!(m * k_dim <= ctx.buffers.fp8_act_bytes());
            ops::per_token_group_quant_fp8(
                ctx.gpu,
                self.per_token_group_quant_fp8_k,
                normed,
                a_fp8_buf,
                a_scale_buf,
                n,
                h,
                stream,
            )?;
            ops::fp8_gemm_t_blockscaled(
                ctx.gpu,
                self.fp8_gemm_t_blockscaled_k,
                a_fp8_buf,
                a_scale_buf,
                fp8w.weight,
                fp8w.row_scale,
                out,
                n,
                out_dim,
                h,
                stream,
            )?;
        } else if let Some(fp8t) = fp8w_t
            && self.w8a16_gemm_t_m128_k.0 != 0
        {
            // Fast transposed FP8 prefill: 128x128 / 8-warp / two-level FP32 fold.
            // Consumes the same B_t[K,N] + block_scale_t the transpose produced.
            ops::w8a16_gemm_n128_m128(
                ctx.gpu,
                self.w8a16_gemm_t_m128_k,
                normed,
                fp8t.weight_t,
                fp8t.scale_t,
                out,
                n,
                out_dim,
                h,
                stream,
            )?;
        } else if let Some(fp8t) = fp8w_t {
            ops::w8a16_gemm_t(
                ctx.gpu,
                self.w8a16_gemm_t_k,
                normed,
                fp8t.weight_t,
                fp8t.scale_t,
                out,
                n,
                out_dim,
                h,
                stream,
            )?;
        } else if weight_opt.and_then(|w| w.as_fp8()).is_some() && self.w8a16_gemm_k.0 != 0 {
            let fp8w = weight_opt.and_then(|w| w.as_fp8()).unwrap();
            ops::w8a16_gemm(
                ctx.gpu,
                self.w8a16_gemm_k,
                normed,
                fp8w.weight,
                fp8w.row_scale,
                out,
                n,
                out_dim,
                h,
                stream,
            )?;
        } else if let Some(fp8p) = fp8 {
            if n > 128 {
                ops::fp8_gemm_n128_m128(
                    ctx.gpu,
                    self.fp8_gemm_t_m128_k,
                    normed,
                    fp8p,
                    out,
                    n,
                    out_dim,
                    h,
                    stream,
                )?;
            } else {
                ops::fp8_gemm_n128(
                    ctx.gpu,
                    self.fp8_gemm_k,
                    normed,
                    fp8p,
                    out,
                    n,
                    out_dim,
                    h,
                    stream,
                )?;
            }
        } else if let Some(nvfp4_t) = nvfp4_t {
            if n > 128 {
                self.w4a16_gemm_m128_dispatch(
                    ctx.gpu, normed, nvfp4_t, out, n, out_dim, h, stream,
                )?;
            } else {
                ops::w4a16_gemm_n128(
                    ctx.gpu,
                    self.w4a16_gemm_t_k,
                    normed,
                    nvfp4_t,
                    out,
                    n,
                    out_dim,
                    h,
                    stream,
                )?;
            }
        } else if let Some(nvfp4) = weight_opt.and_then(|w| w.as_nvfp4()) {
            ops::w4a16_gemm(
                ctx.gpu,
                self.w4a16_gemm_k,
                normed,
                nvfp4,
                out,
                n,
                out_dim,
                h,
                stream,
            )?;
        } else {
            // BF16 dense fallback. For native-FP8 models with
            // ATLAS_FP8_DEQUANT_ATTN_TO_BF16=1, `dense` (= attn.{q,k,v}_proj)
            // holds the FP8→BF16 dequanted weight; otherwise it is the
            // model's native dense weight.
            // Prefer the tensor-core pipelined GEMM (~40× the scalar kernel on
            // these large-M prefill projections; same math) — the scalar
            // `dense_gemm` dominated batched-prefill GPU time (nsys: 60%).
            if self.dense_gemm_pipelined_k.0 != 0 {
                ops::dense_gemm_bf16_pipelined(
                    ctx.gpu,
                    self.dense_gemm_pipelined_k,
                    normed,
                    dense,
                    out,
                    n,
                    out_dim,
                    h,
                    stream,
                )?;
            } else {
                ops::dense_gemm(
                    ctx.gpu,
                    self.dense_gemm_k,
                    normed,
                    dense,
                    out,
                    n,
                    out_dim,
                    h,
                    stream,
                )?;
            }
        }
        // ── LoRA runtime delta: out += scale·(normed@Aᵀ)@Bᵀ (Q/K/V).
        // For gated Q this folds into the RAW interleaved `[Q|gate]` `out`
        // (out_dim = q_proj_dim) BEFORE the caller's `deinterleave_qg_split`
        // (paged.rs / cache_skip.rs) — the PEFT `lora_B` was trained against
        // exactly that interleaved basis, so Q folds like K/V, just wider.
        // Runs before the caller's ATLAS_OP_DUMP, so dumps show ADAPTED
        // outputs (what an HF+PEFT forward hook shows).
        if let Some(ref lw) = self.lora {
            let (pair, route, module) = match proj {
                Proj::Q => (
                    lw.q.as_ref(),
                    lw.q_route.as_ref(),
                    Some(crate::lora::LoraModule::QProj),
                ),
                Proj::K => (
                    lw.k.as_ref(),
                    lw.k_route.as_ref(),
                    Some(crate::lora::LoraModule::KProj),
                ),
                Proj::V => (
                    lw.v.as_ref(),
                    lw.v_route.as_ref(),
                    Some(crate::lora::LoraModule::VProj),
                ),
            };
            if let Some(pair) = pair {
                debug_assert_eq!(pair.k_in, h);
                debug_assert_eq!(pair.n_out, out_dim);
                // #30 routed-prefill precision: when this prefill routes to a
                // NON-active slot (`ctx.routed_lora_layers` Some), select THAT
                // slot's (global_layer, module) pair and fold it through the SAME
                // dense `apply_lora_delta` (dense_gemm_tc for m>1) the ACTIVE
                // adapter's prefill uses — numerically identical to serving that
                // adapter active, unlike the per-row bgmv whose accumulation order
                // tips razor-margin tokens. `lw.layer_idx` is the GLOBAL layer
                // index (not `attn_layer_idx`), matching the pool's GLOBAL-indexed
                // slice. `None` when the routed slot doesn't adapt this module →
                // fall through to the bgmv (base for that module) / installed pair.
                let routed_pair = ctx.routed_lora_layers.and_then(|ls| {
                    module.and_then(|m| crate::lora::select_routed_pair(ls, lw.layer_idx, m))
                });
                // Request-scoped routing: fold THIS request's adapter delta over
                // all `n` prompt tokens via the bgmv when the prefill uploaded a
                // per-request slot buffer (`seq_slot != 0`) and the module has a
                // route. `normed` is contiguous [n, h] and `out` is contiguous
                // [n, out_dim], so the bgmv (all rows = same slot) is
                // byte-identical to `n` single-row `apply_lora_delta`. No pool /
                // no route → the installed-active-pair path (pre-M2 behaviour).
                let seq_slot = ctx
                    .attn_metadata
                    .map(|m| m.seq_slot)
                    .unwrap_or(DevicePtr(0));
                if let Some(routed_pair) = routed_pair {
                    // #30 dense routed path (MUST be checked before the bgmv branch:
                    // a routed prefill satisfies BOTH conditions and the dense path
                    // must win). Same k_in/n_out/max_rank as the installed pair
                    // (uniform pool) — only a/b/scale differ (the request slot's).
                    debug_assert_eq!(routed_pair.k_in, h);
                    debug_assert_eq!(routed_pair.n_out, out_dim);
                    ops::lora_delta::apply_lora_delta(
                        ctx.gpu,
                        &lw.kernels,
                        routed_pair,
                        normed,
                        out,
                        n,
                        ctx.buffers.lora_xa(),
                        ctx.buffers.lora_delta(),
                        stream,
                    )?;
                } else if seq_slot.0 != 0
                    && let Some(route) = route
                {
                    ops::lora_delta::apply_lora_bgmv(
                        ctx.gpu,
                        &lw.kernels,
                        route,
                        normed,
                        out,
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
                        normed,
                        out,
                        n,
                        ctx.buffers.lora_xa(),
                        ctx.buffers.lora_delta(),
                        stream,
                    )?;
                }
            }
        }
        Ok(())
    }
}
