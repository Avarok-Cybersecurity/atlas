// SPDX-License-Identifier: AGPL-3.0-only

//! `prefill_attention_with_cache_skip` — prefix-cache-aware prefill.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::Qwen3AttentionLayer;
use crate::layer::{BatchedAttnMetadata, ForwardContext};
use crate::layers::ops;

impl Qwen3AttentionLayer {
    /// Prefill attention with optional KV cache write skip for prefix caching.
    ///
    /// `kv_write_start`: number of token positions whose KV entries are already
    /// in the cache. `reshape_and_cache` is only called for positions >= this value.
    #[allow(unreachable_code, unused_variables, unused_assignments)]
    pub(in crate::layers::qwen3_attention) fn prefill_attention_with_cache_skip(
        &self,
        normed: DevicePtr,
        num_tokens: usize,
        kv_write_start: usize,
        kv_cache: &mut PagedKvCache,
        // `Some` = batched co-dispatch chunk-0: positions/slots come from the
        // stacked metadata and the flash kernel runs with batch=batch_size,
        // seq_len=chunk_len (one independent causal attention per sequence).
        // `None` = single-stream (batch=1, metadata from ctx.attn_metadata).
        batched_meta: Option<&BatchedAttnMetadata>,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let h = ctx.config.hidden_size as u32;
        let nq = self
            .num_q_heads_override
            .unwrap_or(ctx.config.num_attention_heads) as u32;
        let nkv = self
            .num_kv_heads_override
            .unwrap_or(ctx.config.num_key_value_heads) as u32;
        let hd = self.head_dim_override.unwrap_or(ctx.config.head_dim) as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let bs = kv_cache.block_size();
        let n = num_tokens as u32;
        let bf16 = 2usize;

        // Batched flash supports only the standard non-MLA, hd<=256 path (these
        // are already excluded upstream by check_kernel_batched_eligible).
        if batched_meta.is_some() {
            anyhow::ensure!(
                self.mla.is_none() && hd <= 256,
                "batched cache-skip flash: MLA/hd>256 unsupported (gate to paged)"
            );
        }
        // Position + slot sources and flash launch geometry. Single-stream:
        // ctx.attn_metadata, batch=1. Batched: stacked metadata, batch=N,
        // seq_len=chunk_len (the kernel runs N independent causal attentions).
        let (positions, positions_h, positions_w, kv_slot, flash_seq_len, flash_batch) =
            if let Some(mb) = batched_meta {
                // Derive batch from the ACTUAL stacked token count so the flash
                // kernel never over-reads (num_tokens must be a whole number of
                // chunk_len-length sequences).
                anyhow::ensure!(
                    mb.chunk_len > 0 && num_tokens.is_multiple_of(mb.chunk_len as usize),
                    "batched flash: num_tokens {num_tokens} not a multiple of chunk_len {}",
                    mb.chunk_len
                );
                let derived_batch = (num_tokens / mb.chunk_len as usize) as u32;
                (
                    mb.positions_stacked,
                    mb.positions_h_stacked,
                    mb.positions_w_stacked,
                    mb.slot_stacked,
                    mb.chunk_len,
                    derived_batch,
                )
            } else {
                let meta = ctx
                    .attn_metadata
                    .expect("attention prefill requires metadata");
                (
                    meta.positions,
                    meta.positions_h,
                    meta.positions_w,
                    meta.slot,
                    n,
                    1u32,
                )
            };

        let q_dim = (nq * hd) as usize;
        let q_proj_dim = if self.gated { q_dim * 2 } else { q_dim };
        let kv_dim = (nkv * hd) as usize;

        // Pre-declare output buffers (used by both MLA and standard paths)
        let _qg_out = ctx.buffers.qkv_output();
        let _k_contiguous = ctx.buffers.ssm_qkvz();
        let _v_contiguous = ctx.buffers.attn_output();

        // Profiling helper
        macro_rules! aprof {
            ($label:expr, $t0:expr) => {
                if ctx.profile {
                    if let Some(t0) = $t0 {
                        ctx.gpu.synchronize(stream)?;
                        let elapsed = t0.elapsed().as_micros();
                        tracing::info!("  ATTN prefill [{}] N={}: {}µs", $label, n, elapsed);
                    }
                }
            };
        }
        let mut t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ── 0. Convert activations BF16 → FP8 once for all Q/K/V projections ──
        // FP8×FP8 GEMM is ~10% faster than BF16×FP8 for Q proj, more than
        // compensating for the 7.6ms conversion cost.
        let use_fp8_act = self.q_fp8.is_some();
        let normed_fp8 = if use_fp8_act {
            let act_fp8 = ctx.buffers.attn_output();
            ops::bf16_to_fp8(ctx.gpu, self.bf16_to_fp8_k, normed, act_fp8, n * h, stream)?;
            act_fp8
        } else {
            DevicePtr::NULL
        };
        aprof!("bf16_to_fp8", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ── MLA 2-step prefill ── (extracted to cache_skip_mla.rs)
        if let Some(ref mla) = self.mla {
            let args = super::cache_skip_mla::CacheSkipMlaArgs {
                normed,
                num_tokens,
                n,
                h,
                nq,
                nkv,
                hd,
                kv_dim,
                eps,
                bf16,
                stream,
            };
            // DeepSeek-V4: output LoRA (o_lora_rank > 0) uses a dedicated prefill path.
            if mla.o_lora_rank > 0 {
                return self.prefill_attention_cache_skip_v4(kv_cache, ctx, &args);
            }
            return self.prefill_attention_cache_skip_mla(kv_cache, ctx, &args);
        }

        // ── Standard Q/K/V projection (non-MLA models) ──
        if self.mla.is_none() {
            self.prefill_attention_cache_skip_qkv(
                normed, normed_fp8, n, h, nkv, hd, q_proj_dim, kv_dim, num_tokens, bf16, ctx,
                stream,
            )?;
        } // end if self.mla.is_none() (standard projection path)

        // ── 4+5. Deinterleave Q/Gate + per-head Q/K RMS norms ──
        let qg_out = ctx.buffers.qkv_output();
        let k_contiguous = ctx.buffers.ssm_qkvz();
        let v_contiguous = k_contiguous.offset(num_tokens * kv_dim * bf16);
        let q_contiguous = ctx.buffers.ssm_deinterleaved();
        let q_rope_fused = self.gated
            && !self.attn.q_norm.weight.is_null()
            && self.mrope_interleaved
            && !self.rope_proportional
            && std::env::var("ATLAS_ATTN_PREFILL_FUSED_QROPE")
                .ok()
                .as_deref()
                == Some("1")
            && self.deinterleave_qg_split_qnorm_mrope_k.0 != 0
            && self.rope_mrope_interleaved_k_only_k.0 != 0;
        if q_rope_fused {
            ops::deinterleave_qg_split_qnorm_mrope(
                ctx.gpu,
                self.deinterleave_qg_split_qnorm_mrope_k,
                qg_out,
                q_contiguous,
                self.attn.q_norm.weight,
                positions,
                positions_h,
                positions_w,
                n,
                nq,
                hd,
                q_proj_dim as u32,
                self.rotary_dim_override
                    .unwrap_or(ctx.config.rotary_dim() as u32),
                eps,
                self.rope_theta_override
                    .unwrap_or(ctx.config.rope_theta as f32),
                stream,
            )?;
        } else if self.gated && !self.attn.q_norm.weight.is_null() {
            // Fused deinterleave + Q norm: eliminates Q global memory round-trip
            ops::deinterleave_qg_split_qnorm(
                ctx.gpu,
                self.deinterleave_qg_split_qnorm_k,
                qg_out,
                q_contiguous,
                self.attn.q_norm.weight,
                n,
                nq,
                hd,
                q_proj_dim as u32,
                eps,
                stream,
            )?;
        } else if self.gated {
            ops::deinterleave_qg_split(
                ctx.gpu,
                self.deinterleave_qg_split_k,
                qg_out,
                q_contiguous,
                n,
                nq,
                hd,
                q_proj_dim as u32,
                stream,
            )?;
        } else if self.mla.is_some() {
            // DIAGNOSTIC: check V BEFORE Q copy
            if self.attn_layer_idx == 0 && ctx.config.model_type == "mistral" {
                ctx.gpu.synchronize(stream)?;
                let v_chk = k_contiguous.offset(num_tokens * kv_dim * bf16);
                crate::layers::qwen3_attention::trait_impl::diag_norm(
                    ctx.gpu,
                    v_chk,
                    (nkv * hd) as usize,
                    stream,
                    "L0 V BEFORE Q_copy",
                );
            }
            ctx.gpu
                .copy_d2d_async(qg_out, q_contiguous, num_tokens * q_dim * bf16, stream)
                .map_err(|e| anyhow::anyhow!("MLA Q copy failed: {e}"))?;
        } else {
            ctx.gpu
                .copy_d2d_async(qg_out, q_contiguous, num_tokens * q_dim * bf16, stream)
                .map_err(|e| anyhow::anyhow!("Q copy d2d failed: {e}"))?;
            if let Some(ref q_norm_full) = self.attn.q_norm_full {
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_w_k,
                    q_contiguous,
                    q_norm_full,
                    q_contiguous,
                    n,
                    nq * hd,
                    eps,
                    stream,
                )?;
            } else if !self.attn.q_norm.weight.is_null() {
                ops::rms_norm(
                    ctx.gpu,
                    self.rms_norm_w_k,
                    q_contiguous,
                    &self.attn.q_norm,
                    q_contiguous,
                    nq * n,
                    hd,
                    eps,
                    stream,
                )?;
            }
        }
        if let Some(ref k_norm_full) = self.attn.k_norm_full {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                k_contiguous,
                k_norm_full,
                k_contiguous,
                n,
                nkv * hd,
                eps,
                stream,
            )?;
        } else if !self.attn.k_norm.weight.is_null() {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                k_contiguous,
                &self.attn.k_norm,
                k_contiguous,
                nkv * n,
                hd,
                eps,
                stream,
            )
            .map_err(|e| anyhow::anyhow!("k_norm rms_norm failed: nkv={nkv} n={n} hd={hd}: {e}"))?;
        }

        // ATLAS_OP_DUMP: k AFTER k_norm, BEFORE RoPE. Matches vLLM's "k_proj"
        // dump point in qwen3_next.py (which is post-k_norm pre-RoPE).
        if num_tokens > 0 {
            let kv_dim_e = (nkv * hd) as usize;
            super::super::op_dump::dump_bf16(
                ctx.gpu,
                k_contiguous,
                (num_tokens - 1) * kv_dim_e * bf16,
                kv_dim_e,
                self.attn_layer_idx,
                "k_post_norm",
                stream,
            )?;
        }

        // Gemma-4 v_norm — applied at EVERY layer in HF reference
        // (modeling_gemma4.py:1220 `value_states = self.v_norm(value_states)`
        // with `Gemma4RMSNorm(with_scale=False)`). For full-attention K=V
        // layers, v_contiguous holds raw K (aliased v_proj). For sliding
        // layers, v_contiguous holds V_proj output. Either way normalize
        // with pure RMSNorm via the ones-buffer (Gemma-4's absolute-
        // formula rms_norm kernel: `x * rms * 1.0 = x * rms`). V does NOT
        // receive RoPE.
        if let Some(v_norm_w) = self.v_norm_weight.as_ref() {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_w_k,
                v_contiguous,
                v_norm_w,
                v_contiguous,
                nkv * n,
                hd,
                eps,
                stream,
            )?;
        }

        aprof!("deinterleave+norms", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ── 6. RoPE for N tokens ──
        if self.mla.is_some() {
            // MLA: RoPE already applied inside the MLA block to rope portions only.
            // Skip shared RoPE to avoid double-rotation.
        } else if q_rope_fused {
            ops::rope_mrope_interleaved_k_only(
                ctx.gpu,
                self.rope_mrope_interleaved_k_only_k,
                k_contiguous,
                positions,
                positions_h,
                positions_w,
                n,
                nkv,
                hd,
                self.rotary_dim_override
                    .unwrap_or(ctx.config.rotary_dim() as u32),
                self.rope_theta_override
                    .unwrap_or(ctx.config.rope_theta as f32),
                stream,
            )
            .map_err(|e| anyhow::anyhow!("rope_mrope_interleaved_k_only failed: {e}"))?;
        } else if self.rope_proportional && self.rope_proportional_k.0 != 0 {
            let rope_angles = self
                .rotary_dim_override
                .unwrap_or(ctx.config.rotary_dim() as u32);
            ops::rope_proportional(
                ctx.gpu,
                self.rope_proportional_k,
                q_contiguous,
                k_contiguous,
                positions,
                n,
                nq,
                nkv,
                hd,
                rope_angles,
                self.rope_theta_override
                    .unwrap_or(ctx.config.rope_theta as f32),
                stream,
            )
            .map_err(|e| anyhow::anyhow!("rope_proportional failed: {e}"))?;
        } else {
            ops::rope(
                ctx.gpu,
                self.rope_k,
                q_contiguous,
                k_contiguous,
                positions,
                n,
                nq,
                nkv,
                hd,
                self.rotary_dim_override
                    .unwrap_or(ctx.config.rotary_dim() as u32),
                self.rope_theta_override
                    .unwrap_or(ctx.config.rope_theta as f32),
                stream,
            )
            .map_err(|e| anyhow::anyhow!("rope failed: {e}"))?;
        }

        // ATLAS_OP_DUMP: k AFTER RoPE (final K that gets written to KV cache).
        if num_tokens > 0 {
            let kv_dim_e = (nkv * hd) as usize;
            let q_dim_e = (nq * hd) as usize;
            super::super::op_dump::dump_bf16(
                ctx.gpu,
                k_contiguous,
                (num_tokens - 1) * kv_dim_e * bf16,
                kv_dim_e,
                self.attn_layer_idx,
                "k_post_rope",
                stream,
            )?;
            super::super::op_dump::dump_bf16(
                ctx.gpu,
                q_contiguous,
                (num_tokens - 1) * q_dim_e * bf16,
                q_dim_e,
                self.attn_layer_idx,
                "q_post_rope",
                stream,
            )?;
        }

        aprof!("rope", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ── 7. Write K/V to paged cache ──
        let write_start = kv_write_start;
        let write_count = num_tokens.saturating_sub(write_start);
        if write_count > 0 {
            let k_offset = write_start * kv_dim * bf16;
            let v_offset = write_start * kv_dim * bf16;
            let slot_offset = write_start * 8;
            self.write_kv_cache(
                ctx.gpu,
                k_contiguous.offset(k_offset),
                v_contiguous.offset(v_offset),
                kv_cache,
                kv_slot.offset(slot_offset),
                write_count as u32,
                nkv,
                hd,
                bs as u32,
                nkv * hd,
                nkv * hd,
                stream,
                ctx.graph_capture,
            )?;
        }
        aprof!("kv_cache_write", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // DIAGNOSTIC: verify Q/K/V before flash attention + check buffer addresses
        if self.attn_layer_idx == 0 && ctx.config.model_type == "mistral" {
            tracing::info!(
                "DIAG ADDRS: q_contiguous={:?} k_contiguous={:?} v_at_offset={:?} ssm_deinterleaved={:?} ssm_qkvz={:?} attn_output={:?}",
                q_contiguous.0,
                k_contiguous.0,
                k_contiguous.offset(num_tokens * kv_dim * bf16).0,
                ctx.buffers.ssm_deinterleaved().0,
                ctx.buffers.ssm_qkvz().0,
                ctx.buffers.attn_output().0
            );
            crate::layers::qwen3_attention::trait_impl::diag_norm(
                ctx.gpu,
                q_contiguous,
                (nq * hd) as usize,
                stream,
                "L0 Q[0] pre-attn",
            );
            crate::layers::qwen3_attention::trait_impl::diag_norm(
                ctx.gpu,
                k_contiguous,
                (nkv * hd) as usize,
                stream,
                "L0 K[0] pre-attn",
            );
            let v_check = k_contiguous.offset(num_tokens * kv_dim * bf16);
            crate::layers::qwen3_attention::trait_impl::diag_norm(
                ctx.gpu,
                v_check,
                (nkv * hd) as usize,
                stream,
                "L0 V[0] pre-attn",
            );
        }

        // ── 8. Flash Attention on contiguous Q/K/V (BR=64 for long sequences) ──
        let attn_out = ctx.buffers.attn_output();
        let inv_sqrt_d = self.effective_attn_scale(hd);

        // TurboQuant WHT bookends (mirrors prefill/paged.rs). For turbo
        // dtypes, write_kv_cache (section 7) WHT-rotated the written
        // [kv_write_start..] range of k/v_contiguous IN PLACE before
        // quantizing it into the cache — so the contiguous buffers this FA
        // reads already hold WHT(K)/WHT(V) for that range. Bring the rest of
        // the inputs into the same basis: rotate the unwritten prefix
        // [0..kv_write_start) (prefix-cache hits skip the write, so the
        // write-path bookend never touched those rows), rotate Q
        // (<WHT(Q), WHT(K)> = <Q, K>), and rotate the output back after the
        // attention (it sits in the rotated-V basis).
        let (wht_k_dtype, wht_v_dtype) = self.kv_dtype.kv_pair();
        let k_is_turbo = wht_k_dtype.is_wht_rotated();
        let v_is_turbo = wht_v_dtype.is_wht_rotated();
        let weight_pre_rotated = std::env::var("TQ_PLUS_WEIGHT_ROTATION")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let wht_runtime_active = !weight_pre_rotated && (hd == 128 || hd == 256 || hd == 512);
        if wht_runtime_active && kv_write_start > 0 && self.wht_bf16_k.0 != 0 {
            use spark_runtime::kernel_args::KernelLaunch;
            let prefix_heads = kv_write_start as u32 * nkv;
            if k_is_turbo {
                KernelLaunch::new(ctx.gpu, self.wht_bf16_k)
                    .grid([prefix_heads, 1, 1]) // one warp per (token, kv_head)
                    .block([32, 1, 1])
                    .arg_ptr(k_contiguous)
                    .arg_u32(hd)
                    .launch(stream)?;
            }
            if v_is_turbo {
                KernelLaunch::new(ctx.gpu, self.wht_bf16_k)
                    .grid([prefix_heads, 1, 1])
                    .block([32, 1, 1])
                    .arg_ptr(v_contiguous)
                    .arg_u32(hd)
                    .launch(stream)?;
            }
        }
        if k_is_turbo && wht_runtime_active && self.wht_bf16_k.0 != 0 {
            use spark_runtime::kernel_args::KernelLaunch;
            KernelLaunch::new(ctx.gpu, self.wht_bf16_k)
                .grid([n * nq, 1, 1]) // one warp per (token, q_head)
                .block([32, 1, 1])
                .arg_ptr(q_contiguous)
                .arg_u32(hd)
                .launch(stream)?;
        }
        if hd > 256 && self.prefill_attn_512_k.0 != 0 {
            // HDIM=512: use scalar reference kernel (BR=16, correct for any head_dim)
            // Full-attention layers (this path) always pass sliding_window=0.
            ops::prefill_attention(
                ctx.gpu,
                self.prefill_attn_512_k,
                q_contiguous,
                k_contiguous,
                v_contiguous,
                attn_out,
                flash_seq_len,
                flash_batch,
                nq,
                nkv,
                hd,
                inv_sqrt_d,
                true,
                0,
                stream,
            )
            .map_err(|e| {
                anyhow::anyhow!("prefill_512 failed: n={n} nq={nq} nkv={nkv} hd={hd}: {e}")
            })?;
        } else {
            ops::prefill_attention_64(
                ctx.gpu,
                self.prefill_attn_64_k,
                q_contiguous,
                k_contiguous,
                v_contiguous,
                attn_out,
                flash_seq_len,
                flash_batch,
                nq,
                nkv,
                hd,
                inv_sqrt_d,
                true,
                self.sliding_window.unwrap_or(0),
                stream,
            )
            .map_err(|e| {
                anyhow::anyhow!("flash_attn_64 failed: n={n} nq={nq} nkv={nkv} hd={hd}: {e}")
            })?;
        }

        // TurboQuant WHT bookend (output side): attention output is
        // sum(softmax * WHT(V)) — rotate back to the real basis.
        if v_is_turbo && wht_runtime_active && self.wht_bf16_k_inv.0 != 0 {
            use spark_runtime::kernel_args::KernelLaunch;
            KernelLaunch::new(ctx.gpu, self.wht_bf16_k_inv)
                .grid([n * nq, 1, 1])
                .block([32, 1, 1])
                .arg_ptr(attn_out)
                .arg_u32(hd)
                .launch(stream)?;
        }
        aprof!("flash_attn_64", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ATLAS_OP_DUMP: attn_out BEFORE sigmoid gate (raw FlashAttention output).
        // Compares 1:1 against vLLM's "attn_out" dump in qwen3_next.py.
        if num_tokens > 0 {
            let nq_hd = (nq * hd) as usize;
            super::super::op_dump::dump_bf16(
                ctx.gpu,
                attn_out,
                (num_tokens - 1) * nq_hd * bf16,
                nq_hd,
                self.attn_layer_idx,
                "attn_out_pre_gate",
                stream,
            )?;
        }

        // ── 9. Sigmoid gate × attn_out (gated only) — single batched kernel ──
        if self.gated {
            let gate_base = qg_out.offset(q_dim * bf16);
            ops::sigmoid_gate_mul_batched(
                ctx.gpu,
                self.sigmoid_gate_mul_batched_k,
                attn_out,
                gate_base,
                attn_out,
                nq * hd,
                q_proj_dim as u32,
                n,
                stream,
            )?;
        }

        // ── 9b. Per-head attention gate (Step 3.7 g_proj) ──
        // g_proj produces one scalar per head from the normed hidden states.
        // Applied as: attn_out = attn_out * sigmoid(gate).broadcast_over(hd)
        if let Some(ref g_proj) = self.head_gate_weight {
            // Reuse q_contiguous as scratch for gate output [n, nq] BF16.
            // Q buffer is no longer needed after flash attention.
            let gate_buf = q_contiguous;
            // GEMM: normed [n, H] × g_proj^T [H, nq] → gate_buf [n, nq]
            ops::dense_gemm_tc(
                ctx.gpu,
                self.dense_gemm_tc_k,
                normed,
                g_proj,
                gate_buf,
                n,
                nq,
                h,
                stream,
            )?;
            // Sigmoid + broadcast multiply: attn_out[t,h,d] *= sigmoid(gate[t,h])
            ops::sigmoid_gate_mul_head_broadcast(
                ctx.gpu,
                self.sigmoid_gate_head_broadcast_k,
                attn_out,
                gate_buf,
                attn_out,
                nq,
                hd,
                n,
                stream,
            )?;
        }
        aprof!("sigmoid_gate", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ATLAS_OP_DUMP: attn_out AFTER sigmoid gate (input to o_proj linear).
        if num_tokens > 0 {
            let nq_hd = (nq * hd) as usize;
            super::super::op_dump::dump_bf16(
                ctx.gpu,
                attn_out,
                (num_tokens - 1) * nq_hd * bf16,
                nq_hd,
                self.attn_layer_idx,
                "attn_out_post_gate",
                stream,
            )?;
        }

        // ── 10. O projection GEMM ── (extracted to paged_oproj.rs)
        let o_out = self.prefill_attention_paged_oproj(attn_out, n, h, nq, hd, ctx, stream)?;
        aprof!("o_proj", t0);
        Ok(o_out)
    }
}
