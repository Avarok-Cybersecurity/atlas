// SPDX-License-Identifier: AGPL-3.0-only

//! Split out of `super::super::decode.rs` for file-size budget.

#![allow(unused_imports)]

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kv_cache::{KvCacheDtype, PagedKvCache};
use spark_runtime::kv_dequant::{
    NVFP4_E2M1_LUT, TURBO4_LUT, dequant_4bit_block_to_bf16, dequant_fp8_to_bf16,
    dequant_turbo3_block_to_bf16, dequant_turbo8_block_to_bf16,
};

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    pub(in super::super) fn attention_forward(
        &self,
        normed: DevicePtr,
        seq_len: usize,
        block_table: &mut Vec<u32>,
        disk_block_ids: &mut Vec<u32>,
        disk_last_offloaded_per_layer: &mut Vec<u32>,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let h = ctx.config.hidden_size as u32;
        // Per-layer dimension overrides for heterogeneous models (Gemma-4)
        let nq = self
            .num_q_heads_override
            .unwrap_or(ctx.config.num_attention_heads) as u32;
        let nkv = self
            .num_kv_heads_override
            .unwrap_or(ctx.config.num_key_value_heads) as u32;
        let hd = self.head_dim_override.unwrap_or(ctx.config.head_dim) as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let bs = kv_cache.block_size();

        // Phase 6.3: caller (model.rs::TransformerModel::decode and friends)
        // is responsible for block allocation via `ensure_blocks_through_decode`,
        // which handles HSS sliding-window eviction before this layer-internal
        // entry point. Defensive alloc here is incompatible with rolling-window
        // semantics (no access to disk_block_ids).
        let blocks_needed = (seq_len / bs) + 1;
        let expected_window_size = match kv_cache.config().cache_blocks_per_seq {
            Some(cap) => blocks_needed.min(cap as usize),
            None => blocks_needed,
        };
        debug_assert!(
            block_table.len() >= expected_window_size,
            "Qwen3AttentionLayer::decode entered with under-allocated block_table \
             ({}/{} blocks) — caller must call ensure_blocks_through_decode",
            block_table.len(),
            expected_window_size,
        );

        // Q/K/V projections into separate regions of qkv_output (GEMV for M=1)
        let q_out = ctx.buffers.qkv_output();
        let q_dim = nq * hd; // actual Q dimension
        let q_proj_dim = if self.gated { q_dim * 2 } else { q_dim }; // gated: Q + gate
        let q_proj_bytes = q_proj_dim as usize * 2;
        let k_out = q_out.offset(q_proj_bytes);
        let v_out = k_out.offset((nkv * hd) as usize * 2);
        let meta = ctx
            .attn_metadata
            .expect("attention layer requires pre-uploaded metadata");

        // ── MLA 2-step decode ── (extracted to attention_forward_mla.rs)
        if let Some(ref mla) = self.mla {
            let args = super::attention_forward_mla::DecodeMlaArgs {
                normed,
                q_out,
                k_out,
                v_out,
                q_dim,
                h,
                nq,
                hd,
                eps,
                bs,
                stream,
                // Absolute position of the token being generated (seq_len counts
                // it). Drives the V4 inc-3 compressed-pool decode append.
                pos: Some(seq_len.saturating_sub(1) as u32),
            };
            if mla.o_lora_rank > 0 {
                return self.attention_forward_v4(kv_cache, ctx, &args);
            }
            return self.attention_forward_mla(kv_cache, ctx, &args);
        }

        if self.gated {
            // Q+Gate projection with inline deinterleave (output is [Q_all | Gate_all])
            if let Some(fp8) = self.q_weight.as_ref().and_then(|w| w.as_fp8()) {
                // FP8 native: w8a16_gemv + separate deinterleave (no fused QG variant yet)
                ops::w8a16_gemv(
                    ctx.gpu,
                    self.w8a16_gemv_k,
                    normed,
                    fp8.weight,
                    fp8.row_scale,
                    q_out,
                    q_proj_dim,
                    h,
                    stream,
                )?;
                ops::deinterleave_qg(
                    ctx.gpu,
                    self.deinterleave_qg_k,
                    q_out,
                    1,
                    nq,
                    hd,
                    nq * hd * 2,
                    stream,
                )?;
            } else if let Some(nvfp4) = self.q_weight.as_ref().and_then(|w| w.as_nvfp4()) {
                ops::w4a16_gemv_qg(
                    ctx.gpu,
                    self.w4a16_gemv_qg_k,
                    normed,
                    nvfp4,
                    q_out,
                    q_proj_dim,
                    h,
                    nq,
                    hd,
                    stream,
                )?;
            } else {
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    normed,
                    &self.attn.q_proj,
                    q_out,
                    q_proj_dim,
                    h,
                    stream,
                )?;
                ops::deinterleave_qg(
                    ctx.gpu,
                    self.deinterleave_qg_k,
                    q_out,
                    1,
                    nq,
                    hd,
                    nq * hd * 2,
                    stream,
                )?;
            }
        } else {
            // Ungated: Q projection only (no gate)
            if let Some(fp8) = self.q_weight.as_ref().and_then(|w| w.as_fp8()) {
                ops::w8a16_gemv(
                    ctx.gpu,
                    self.w8a16_gemv_k,
                    normed,
                    fp8.weight,
                    fp8.row_scale,
                    q_out,
                    q_dim,
                    h,
                    stream,
                )?;
            } else if let Some(nvfp4) = self.q_weight.as_ref().and_then(|w| w.as_nvfp4()) {
                ops::w4a16_gemv(
                    ctx.gpu,
                    self.w4a16_gemv_k,
                    normed,
                    nvfp4,
                    q_out,
                    q_dim,
                    h,
                    stream,
                )?;
            } else {
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    normed,
                    &self.attn.q_proj,
                    q_out,
                    q_dim,
                    h,
                    stream,
                )?;
            }
        }

        // DIAG: dump normed input and Q output for L0
        if self.attn_layer_idx == 0 && ctx.profile {
            ctx.gpu.synchronize(stream)?;
            let mut input_buf = vec![0u8; 16]; // first 8 BF16 values
            ctx.gpu.copy_d2h(normed, &mut input_buf)?;
            let input_vals: Vec<f32> = input_buf
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect();
            let mut q_buf = vec![0u8; 16];
            ctx.gpu.copy_d2h(q_out, &mut q_buf)?;
            let q_vals: Vec<f32> = q_buf
                .chunks_exact(2)
                .map(|c| {
                    let bits = u16::from_le_bytes([c[0], c[1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect();
            tracing::info!(
                "GEMV_DIAG L0: input[0:8]={:.4?} q_out[0:8]={:.4?} nq={nq} hd={hd} h={h}",
                input_vals,
                q_vals
            );
        }

        // K+V output after Q projection region
        let k_out = q_out.offset(q_proj_bytes);
        let v_out = k_out.offset((nkv * hd) as usize * 2);

        self.attention_forward_kv(normed, k_out, v_out, nkv, hd, h, ctx, stream)?;

        // Q/K RMS norms — three mutually-exclusive paths:
        //  1. MiniMax M2 style: RMSNorm over full projected hidden
        //     `[nq*hd]` per token, single learned weight of that shape.
        //     Reached only for MiniMax — every other loader leaves
        //     `q_norm_full` as `None` (see `AttentionWeights`).
        //  2. Qwen3-family per-head: rows=nq, cols=hd.
        //  3. Nemotron-H standalone attn: both weights NULL, skip.
        //
        // Applied BEFORE RoPE (MiniMaxM2Attention.forward reference).
        // This codepath never runs for Mistral/DeepSeek-style MLA
        // models — they early-return in the MLA branch above.
        if let Some(ref q_norm_full) = self.attn.q_norm_full {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                q_out,
                q_norm_full,
                q_out,
                1,
                nq * hd,
                eps,
                stream,
            )?;
        } else if !self.attn.q_norm.weight.is_null() {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                q_out,
                &self.attn.q_norm,
                q_out,
                nq,
                hd,
                eps,
                stream,
            )?;
        }
        if let Some(ref k_norm_full) = self.attn.k_norm_full {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                k_out,
                k_norm_full,
                k_out,
                1,
                nkv * hd,
                eps,
                stream,
            )?;
        } else if !self.attn.k_norm.weight.is_null() {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                k_out,
                &self.attn.k_norm,
                k_out,
                nkv,
                hd,
                eps,
                stream,
            )?;
        }

        // Gemma-4 v_norm (applied at EVERY layer, not just K=V). HF
        // `Gemma4TextAttention.forward()` modeling_gemma4.py:1220 applies
        // `value_states = self.v_norm(value_states)` with
        // `Gemma4RMSNorm(with_scale=False)` = pure `x * rms` regardless of
        // K=V mode. For full-attention K=V layers, v_out holds raw K (V
        // GEMV against aliased K weights). For sliding layers, v_out holds
        // V projection output. Either way, normalize in place. V does NOT
        // receive RoPE. Ones (not zeros) because Gemma-4's rms_norm uses
        // the absolute formula `out = x * rms * weight`.
        if let Some(v_norm_w) = self.v_norm_weight.as_ref() {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                v_out,
                v_norm_w,
                v_out,
                nkv,
                hd,
                eps,
                stream,
            )?;
        }

        if self.mla.is_some() {
            // MLA: RoPE already applied inside the MLA block (to rope portions only).
            // Skip the shared RoPE to avoid double-rotation.
        } else if self.rope_proportional && self.rope_proportional_k.0 != 0 {
            // Gemma-4 full-attention: proportional RoPE with rotation pairs
            // (i, i + head_dim/2) for i < rope_angles. rotary_dim_override
            // here holds `rope_angles` (64 for 31B full attn).
            let rope_angles = self
                .rotary_dim_override
                .unwrap_or(ctx.config.rotary_dim() as u32);
            ops::rope_proportional(
                ctx.gpu,
                self.rope_proportional_k,
                q_out,
                k_out,
                meta.positions,
                1,
                nq,
                nkv,
                hd,
                rope_angles,
                self.rope_theta_override
                    .unwrap_or(ctx.config.rope_theta as f32),
                stream,
            )?;
        } else if self.mrope_interleaved && self.rope_mrope_interleaved_k.0 != 0 {
            ops::rope_mrope_interleaved(
                ctx.gpu,
                self.rope_mrope_interleaved_k,
                q_out,
                k_out,
                meta.positions,
                meta.positions_h,
                meta.positions_w,
                1,
                nq,
                nkv,
                hd,
                self.rotary_dim_override
                    .unwrap_or(ctx.config.rotary_dim() as u32),
                self.rope_theta_override
                    .unwrap_or(ctx.config.rope_theta as f32),
                stream,
            )?;
        } else {
            ops::rope(
                ctx.gpu,
                self.rope_k,
                q_out,
                k_out,
                meta.positions,
                1,
                nq,
                nkv,
                hd,
                self.rotary_dim_override
                    .unwrap_or(ctx.config.rotary_dim() as u32),
                self.rope_theta_override
                    .unwrap_or(ctx.config.rope_theta as f32),
                stream,
            )?;
        }

        // K/V are contiguous (separate dense_gemm outputs), stride = nkv * hd
        let kv_stride = nkv * hd;
        self.write_kv_cache(
            ctx.gpu,
            k_out,
            v_out,
            kv_cache,
            meta.slot,
            1,
            nkv,
            hd,
            bs as u32,
            kv_stride,
            kv_stride,
            stream,
            ctx.graph_capture,
        )?;

        // Turbo KV cache: apply WHT to Q before paged decode.
        // KV cache stores WHT(K) and WHT(V). By Parseval's theorem,
        // <WHT(Q), WHT(K)> = <Q, K>, so WHT(Q) gives correct attention scores.
        //
        // Asymmetric K/V (e.g. K=turbo4, V=fp8): each side carries an
        // independent rotation requirement. WHT(Q) fires only when K is a
        // turbo type (we're dotting against rotated K); iWHT(out) below
        // fires only when V is a turbo type (output is in rotated-V basis).
        let (k_dtype, v_dtype) = self.kv_dtype.kv_pair();
        let k_is_turbo = k_dtype.is_wht_rotated();
        let v_is_turbo = v_dtype.is_wht_rotated();
        // InnerQ pre-WHT scale_inv on Q (no-op when d_innerq_active=0 on device).
        // Bypass runtime WHT(Q) when weights are pre-rotated at load (TQ_PLUS_WEIGHT_ROTATION=1).
        let weight_pre_rotated = std::env::var("TQ_PLUS_WEIGHT_ROTATION")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        if k_is_turbo && self.innerq_apply_q_k.0 != 0 && hd == 128 {
            use spark_runtime::kernel_args::KernelLaunch;
            KernelLaunch::new(ctx.gpu, self.innerq_apply_q_k)
                .grid([nq, 1, 1])
                .block([32, 1, 1])
                .arg_ptr(q_out)
                .arg_u32(hd)
                .launch(stream)?;
        }
        if k_is_turbo
            && !weight_pre_rotated
            && self.wht_bf16_k.0 != 0
            && (hd == 128 || hd == 256 || hd == 512)
        {
            use spark_runtime::kernel_args::KernelLaunch;
            KernelLaunch::new(ctx.gpu, self.wht_bf16_k)
                .grid([nq, 1, 1]) // one warp per Q head
                .block([32, 1, 1])
                .arg_ptr(q_out)
                .arg_u32(hd)
                .launch(stream)?;
        }

        let attn_out = ctx.buffers.attn_output();
        let inv_sqrt_d = self.effective_attn_scale(hd);

        // --high-speed-swap dispatch (Phase 6.2.c — proper).
        // Routes attention through the orchestrator's tile-streaming kernel
        // when engaged for this layer. Engagement requires:
        //   • `--high-speed-swap` + `--high-speed-swap-cache-blocks-per-seq`
        //     CLI flags (PagedKvCache::config().cache_blocks_per_seq.is_some()).
        //   • The thread-local orchestrator was installed by the scheduler.
        //   • The layer's KV dtype is one of {BF16, FP8, NVFP4, Turbo3/4/8}
        //     (every supported quant has a host-side dequant path that
        //     produces BF16 for the orchestrator's tiled-attention kernel).
        // For Turbo: WHT(Q) was applied just above (line 1542) and iWHT(out)
        // is applied just below (line 1669); the cache holds WHT(K)/WHT(V)
        // so the streaming kernel sees a self-consistent WHT-domain attention
        // and the bookend kernels recover real-V.
        let use_orchestrator = self.high_speed_swap_engaged(kv_cache);

        if use_orchestrator {
            // Phase 6.3: per-layer K/V offload to disk. The alloc-time
            // helper (`ensure_blocks_through_decode`) already grew
            // `disk_block_ids` and may have already slid the window for
            // this step's new block; here we just push this layer's K/V
            // bytes to the on-disk file under each block's disk_id.
            self.high_speed_swap_offload_new_blocks(
                kv_cache,
                block_table,
                disk_block_ids,
                disk_last_offloaded_per_layer,
                ctx,
                stream,
                nkv,
                hd,
                bs,
            )?;
            // Streaming attention over the full disk-side history.
            spark_storage::with_local(|hss| {
                hss.attend_layer_on_stream(
                    stream,
                    self.attn_layer_idx as u32,
                    disk_block_ids,
                    q_out.0,
                    attn_out.0,
                )
            })
            .expect("local installed checked in high_speed_swap_engaged")?;
        } else {
            self.run_paged_decode(
                ctx.gpu,
                q_out,
                kv_cache,
                attn_out,
                meta.block_table,
                meta.seq_len,
                meta.max_blocks_per_seq,
                1,
                nq,
                nkv,
                hd,
                bs as u32,
                inv_sqrt_d,
                nq * hd,
                ctx.buffers.splitk_workspace(),
                stream,
            )?;
        }

        // Turbo KV cache: apply iWHT to attention output.
        // Output = sum(softmax * WHT(V)) → real_output = iWHT(output).
        // With plain WHT this aliases the forward kernel (self-inverse). With
        // TQ_PLUS_SIGNS the inverse reverses signs1/signs2 order.
        //
        // Guard checks V's turbo-ness (not K's): output sits in V's basis,
        // so iWHT only fires when V is a turbo type. For asym K=turbo, V=non-
        // turbo this branch correctly skips.
        if v_is_turbo
            && !weight_pre_rotated
            && self.wht_bf16_k_inv.0 != 0
            && (hd == 128 || hd == 256 || hd == 512)
        {
            use spark_runtime::kernel_args::KernelLaunch;
            KernelLaunch::new(ctx.gpu, self.wht_bf16_k_inv)
                .grid([nq, 1, 1])
                .block([32, 1, 1])
                .arg_ptr(attn_out)
                .arg_u32(hd)
                .launch(stream)?;
        }

        // Apply sigmoid gate: attn_out = attn_out * sigmoid(gate)
        if self.gated {
            let gate_ptr = q_out.offset(q_dim as usize * 2);
            ops::sigmoid_gate_mul(
                ctx.gpu,
                self.sigmoid_gate_mul_k,
                attn_out,
                gate_ptr,
                attn_out,
                nq * hd,
                stream,
            )?;
        }

        // Per-head attention gate (Step 3.7 g_proj) — decode path.
        // Same logic as prefill: gate[h] = g_proj(normed), apply sigmoid broadcast.
        if let Some(ref g_proj) = self.head_gate_weight {
            // For decode, n=1 (single token). Reuse q_out scratch for gate [1, nq].
            let gate_buf = q_out;
            ops::dense_gemm_tc(
                ctx.gpu,
                self.dense_gemm_tc_k,
                normed,
                g_proj,
                gate_buf,
                1, // decode: single token
                nq,
                h,
                stream,
            )?;
            ops::sigmoid_gate_mul_head_broadcast(
                ctx.gpu,
                self.sigmoid_gate_head_broadcast_k,
                attn_out,
                gate_buf,
                attn_out,
                nq,
                hd,
                1, // decode: single token
                stream,
            )?;
        }

        // O projection ── (extracted to attention_forward_oproj.rs)
        let o_out = self.attention_forward_oproj(attn_out, nq, hd, h, ctx, stream)?;

        Ok(o_out)
    }
}
