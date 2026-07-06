// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4-Flash prefill path using standard GQA FlashAttention.
//! Isolated to V4-Flash (o_lora_rank > 0); no other models reach this code.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    pub(super) fn prefill_attention_cache_skip_v4(
        &self,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        args: &super::cache_skip_mla::CacheSkipMlaArgs,
    ) -> Result<DevicePtr> {
        let super::cache_skip_mla::CacheSkipMlaArgs {
            normed,
            num_tokens: _,
            n,
            h,
            nq,
            nkv,
            hd: _,
            kv_dim: _,
            eps,
            bf16: _,
            stream,
        } = *args;
        let mla = self.mla.as_ref().expect("V4-Flash prefill requires MLA");
        let meta = ctx
            .attn_metadata
            .expect("V4-Flash prefill requires metadata");

        let nope = mla.nope as u32;
        let rope = mla.rope as u32;
        let kv_lora = mla.kv_lora_rank as u32;
        let _v_dim = mla.v_dim as u32;
        let q_lora = mla.q_lora_rank as u32;
        let o_lora = mla.o_lora_rank as u32;
        let mla_cache_dim = kv_lora + rope;
        let hd_mla = nope + rope;
        let use_tc = self.dense_gemm_tc_k.0 != 0;
        let diag_all =
            std::env::var("ATLAS_DIAG_V4_ALL_LAYERS").is_ok_and(|v| v == "1" || v == "true");
        let diag_this = self.attn_layer_idx == 0 || diag_all;

        // Per-token NaN scan of `normed` (post hc_pre + input_norm) — localizes
        // whether the K-FULL NaN originates upstream (hc_pre) or in the kv proj.
        if diag_this {
            let _ = ctx.gpu.synchronize(stream);
            let hh = h as usize;
            let mut buf = vec![0u8; (n as usize) * hh * 2];
            if ctx.gpu.copy_d2h(normed, &mut buf).is_ok() {
                let mut bad_tok = -1i64;
                for t in 0..n as usize {
                    let off = t * hh * 2;
                    if (0..hh).any(|i| {
                        let c = &buf[off + i * 2..off + i * 2 + 2];
                        let v = f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16);
                        !v.is_finite()
                    }) {
                        bad_tok = t as i64;
                        break;
                    }
                }
                tracing::info!(
                    "DIAG V4-prefill L{} NORMED first non-finite (nan/inf) token = {}",
                    self.attn_layer_idx,
                    bad_tok
                );
            }
        }

        // ── 1. Q latent → norm → expand ──
        let q_latent = ctx.buffers.ssm_ba();
        if use_tc {
            ops::dense_gemm_tc(
                ctx.gpu,
                self.dense_gemm_tc_k,
                normed,
                &mla.wq_a,
                q_latent,
                n,
                q_lora,
                h,
                stream,
            )?;
        } else {
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                normed,
                &mla.wq_a,
                q_latent,
                n,
                q_lora,
                h,
                stream,
            )?;
        }
        ctx.gpu
            .synchronize(stream)
            .map_err(|e| anyhow::anyhow!("V4 attn: q_latent gemm sync failed: {e}"))?;
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            q_latent,
            &mla.q_a_norm,
            q_latent,
            n,
            q_lora,
            eps,
            stream,
        )?;
        ctx.gpu
            .synchronize(stream)
            .map_err(|e| anyhow::anyhow!("V4 attn: q_a_norm sync failed: {e}"))?;
        let q_full = ctx.buffers.qkv_output();
        ops::dense_gemm(
            ctx.gpu,
            self.dense_gemm_k,
            q_latent,
            &mla.wq_b,
            q_full,
            n,
            nq * hd_mla,
            q_lora,
            stream,
        )?;
        ctx.gpu
            .synchronize(stream)
            .map_err(|e| anyhow::anyhow!("V4 attn: q_full gemm sync failed: {e}"))?;
        // q_b_norm: per-head unweighted RMSNorm over head_dim (DeepSeek-V4),
        // each of the n*nq head vectors renormalized to unit RMS before rope.
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            q_full,
            &crate::weight_map::DenseWeight {
                weight: ctx.buffers.norm_unit_w(),
            },
            q_full,
            n * nq,
            hd_mla,
            eps,
            stream,
        )?;
        if diag_this {
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                q_full,
                (nq * hd_mla) as usize,
                stream,
                &format!(
                    "V4-prefill L{} Q after q_b_norm token0",
                    self.attn_layer_idx
                ),
            );
            let q_last_off = ((n - 1) * nq * hd_mla * 2) as usize;
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                q_full.offset(q_last_off),
                (nq * hd_mla) as usize,
                stream,
                &format!("V4-prefill L{} Q after q_b_norm last", self.attn_layer_idx),
            );
        }

        // ── 2. Direct KV projection (V4-Flash: K=V, no absorption) ──
        // Layout in qkv_output: [Q | K | V]  (mirrors decode path)
        let q_dim = nq * hd_mla;
        let kv_dim = nkv * hd_mla;
        let k_out = q_full.offset((n * q_dim) as usize * 2);
        let v_out = k_out.offset((n * kv_dim) as usize * 2);
        let kv_latent = ctx.buffers.expert_gate_out(); // Capture latent for cache assembly
        // NOTE: dense_gemm_tc produces NON-DETERMINISTIC NaN for the wkv projection
        // here (varying token position across identical runs) — a latent TC-kernel
        // bug exposed once the upstream norms were corrected. Use the scalar
        // dense_gemm path for wkv until the TC kernel is fixed. wq_a above is
        // unaffected (TC output is correct there).
        #[allow(clippy::overly_complex_bool_expr)]
        if false && use_tc {
            ops::dense_gemm_tc(
                ctx.gpu,
                self.dense_gemm_tc_k,
                normed,
                &mla.wkv_a,
                kv_latent, // Write to kv_latent for cache assembly
                n,
                kv_lora,
                h,
                stream,
            )?;
        } else {
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                normed,
                &mla.wkv_a,
                kv_latent, // Write to kv_latent for cache assembly
                n,
                kv_lora,
                h,
                stream,
            )?;
        }
        if diag_this {
            let _ = ctx.gpu.synchronize(stream);
            let kl = kv_lora as usize;
            let mut wbuf = vec![0u8; kl * 2];
            let _ = ctx.gpu.copy_d2h(mla.kv_a_norm.weight, &mut wbuf);
            let wnan = (0..kl).any(|i| {
                let c = &wbuf[i * 2..i * 2 + 2];
                f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16).is_nan()
            });
            let mut lbuf = vec![0u8; (n as usize) * kl * 2];
            let mut lnan = -1i64;
            if ctx.gpu.copy_d2h(kv_latent, &mut lbuf).is_ok() {
                for t in 0..n as usize {
                    if (0..kl).any(|i| {
                        let c = &lbuf[t * kl * 2 + i * 2..t * kl * 2 + i * 2 + 2];
                        f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16).is_nan()
                    }) {
                        lnan = t as i64;
                        break;
                    }
                }
            }
            tracing::info!(
                "DIAG V4-prefill L{} PRE-kvnorm: kv_a_norm has NaN={}, kv_latent first NaN token={}",
                self.attn_layer_idx,
                wnan,
                lnan
            );
        }
        // kv_norm: weighted RMSNorm over each token's kv latent BEFORE rope and
        // before cache assembly (DeepSeek-V4: kv = kv_norm(kv_proj(h))). Applied to
        // kv_latent so the cached latent, k_out and v_out are all normalized.
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            kv_latent,
            &mla.kv_a_norm,
            kv_latent,
            n * nkv,
            kv_lora,
            eps,
            stream,
        )?;
        // Copy kv_latent → k_out (for attention computation)
        ctx.gpu
            .copy_d2d_async(kv_latent, k_out, n as usize * kv_lora as usize * 2, stream)?;
        ctx.gpu
            .synchronize(stream)
            .map_err(|e| anyhow::anyhow!("V4 attn: k_out gemm sync failed: {e}"))?;
        if diag_this {
            // Full-buffer K NaN/inf check across ALL n tokens (locates a bad token).
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                k_out,
                (n * kv_lora) as usize,
                stream,
                &format!("V4-prefill L{} K FULL ({} tokens)", self.attn_layer_idx, n),
            );
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                k_out,
                kv_dim as usize,
                stream,
                &format!("V4-prefill L{} K after proj", self.attn_layer_idx),
            );
        }
        // Copy K → V (V4-Flash: K and V share the same projection output)
        ctx.gpu
            .copy_d2d_async(k_out, v_out, (n * kv_dim) as usize * 2, stream)?;
        if diag_this {
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                v_out,
                kv_dim as usize,
                stream,
                &format!("V4-prefill L{} V after copy", self.attn_layer_idx),
            );
        }

        // ── 3. RoPE on Q and K (V is NOT RoPE'd) ──
        // V4-Flash: rope dims are at offset `nope` per head (matching MLA layout),
        // not at the beginning. Extract → RoPE → writeback.
        let q_rope_tmp = ctx.buffers.ssm_conv_out_f32();
        let k_rope_tmp = q_latent; // reuse after wq_b is done
        ops::mla_q_rope_extract_batched(
            ctx.gpu,
            self.mla_q_rope_extract_batched_k,
            q_full,
            q_rope_tmp,
            n,
            nq,
            hd_mla,
            nope,
            rope,
            nq * hd_mla,
            stream,
        )?;
        ops::mla_q_rope_extract_batched(
            ctx.gpu,
            self.mla_q_rope_extract_batched_k,
            k_out,
            k_rope_tmp,
            n,
            nkv,
            hd_mla,
            nope,
            rope,
            nkv * hd_mla,
            stream,
        )?;
        ops::rope_yarn(
            ctx.gpu,
            // DeepSeek-V4 INTERLEAVED RoPE (rope_interleave=True): adjacent pairs
            // (2i, 2i+1), matching the HF reference. See attention_forward_v4.rs.
            self.rope_yarn_interleaved_k,
            q_rope_tmp,
            k_rope_tmp,
            meta.positions,
            n,
            nq,
            nkv,
            rope,
            rope,
            mla.yarn_inv_freq,
            super::super::helpers::yarn_rope_mscale(ctx.config),
            stream,
        )?;
        ops::mla_q_rope_writeback_batched(
            ctx.gpu,
            self.mla_q_rope_writeback_batched_k,
            q_rope_tmp,
            q_full,
            n,
            nq,
            hd_mla,
            nope,
            rope,
            nq * hd_mla,
            stream,
        )?;
        ops::mla_q_rope_writeback_batched(
            ctx.gpu,
            self.mla_q_rope_writeback_batched_k,
            k_rope_tmp,
            k_out,
            n,
            nkv,
            hd_mla,
            nope,
            rope,
            nkv * hd_mla,
            stream,
        )?;
        ctx.gpu
            .synchronize(stream)
            .map_err(|e| anyhow::anyhow!("V4 attn: rope_yarn sync failed: {e}"))?;
        if diag_this {
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                k_out,
                kv_dim as usize,
                stream,
                &format!("V4-prefill L{} K after RoPE token0", self.attn_layer_idx),
            );
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                k_out.offset((nope * 2) as usize),
                (kv_dim - nope) as usize,
                stream,
                &format!(
                    "V4-prefill L{} K rope after RoPE token0",
                    self.attn_layer_idx
                ),
            );
            let last_k_offset = ((n - 1) * kv_dim * 2) as usize;
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                k_out.offset(last_k_offset),
                kv_dim as usize,
                stream,
                &format!("V4-prefill L{} K after RoPE last", self.attn_layer_idx),
            );
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                k_out.offset(last_k_offset + (nope * 2) as usize),
                (kv_dim - nope) as usize,
                stream,
                &format!("V4-prefill L{} K rope after RoPE last", self.attn_layer_idx),
            );
        }

        // ── 4. Core attention ──
        // CSA layers (compress_ratios[L]=4): attend over [raw causal KV | compressed
        // windowed KV] + per-head sink (DeepSeek Sparse Attention). For short prompts
        // the indexer is a no-op so only the compressor concat matters. Full-attention
        // (layers 0-1) and HCA-short layers fall back to plain prefill attention.
        use spark_runtime::kernel_args::KernelLaunch;
        let attn_out = ctx.buffers.attn_output();
        // DeepSeek-V4 sliding-window (port item 1): the RAW attention arm is windowed to
        // the last V4_WINDOW keys on EVERY layer (config sliding_window=128). Distant
        // context is carried by the compressed arm (CSA/HCA), never raw. Full-causal raw
        // attention past the window is out-of-distribution → prompt-length salad. Windowing
        // the raw arm; compression stays intact.
        const V4_WINDOW: u32 = 128;
        // Port item 3: admit BOTH CSA (ratio 4, overlap) AND HCA (ratio 128, non-overlap)
        // layers to the compression path. The csa_compress kernel already branches on is_csa
        // for stride/token-map/output-count; only the launch's is_csa flag must be passed
        // through (was hardcoded 1). HCA layers were falling to windowed-raw-only (OOD).
        let csa = match mla.compressor {
            Some(c) if self.csa_compress_k.0 != 0 && (n / c.ratio as u32) > 0 => Some(c),
            _ => None,
        };
        let did_csa = if let Some(comp) = csa {
            let ratio = comp.ratio as u32;
            let proj_dim = comp.proj_dim as u32;
            let n_win = n / ratio;
            // compressor projections kv/gate = W·normed [n, proj_dim]
            let kv_comp = ctx.buffers.expert_up_out();
            let gate_comp = ctx.buffers.expert_down_out();
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                normed,
                &comp.wkv,
                kv_comp,
                n,
                proj_dim,
                h,
                stream,
            )?;
            ops::dense_gemm(
                ctx.gpu,
                self.dense_gemm_k,
                normed,
                &comp.wgate,
                gate_comp,
                n,
                proj_dim,
                h,
                stream,
            )?;
            // window softmax-gated compression → compressed [n_win, hd_mla]
            let compressed = ctx.buffers.moe_output();
            KernelLaunch::new(ctx.gpu, self.csa_compress_k)
                .grid([n_win, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(kv_comp)
                .arg_ptr(gate_comp)
                .arg_ptr(comp.ape)
                .arg_ptr(compressed)
                .arg_u32(n)
                .arg_u32(ratio)
                .arg_u32(hd_mla)
                .arg_u32(proj_dim)
                .arg_u32(if comp.is_csa { 1 } else { 0 })
                .launch(stream)?;
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                compressed,
                &comp.norm,
                compressed,
                n_win,
                hd_mla,
                eps,
                stream,
            )?;
            // compressed = V (pre-rope latent); comp_k = rope(compressed) for scores.
            // RoPE the trailing `rope` dims at the window position w*ratio (compress
            // theta = mla.yarn_inv_freq), interleaved — mirrors the raw K rope.
            let comp_v = compressed;
            let comp_k = compressed.offset((n_win * hd_mla) as usize * 2);
            ctx.gpu
                .copy_d2d_async(comp_v, comp_k, (n_win * hd_mla) as usize * 2, stream)?;
            let comp_pos: Vec<u8> = (0..n_win).flat_map(|w| (w * ratio).to_le_bytes()).collect();
            let comp_positions = ctx.buffers.ssm_ba();
            ctx.gpu.copy_h2d_async(&comp_pos, comp_positions, stream)?;
            let comp_rope_tmp = ctx.buffers.ssm_conv_out_f32();
            ops::mla_q_rope_extract_batched(
                ctx.gpu,
                self.mla_q_rope_extract_batched_k,
                comp_k,
                comp_rope_tmp,
                n_win,
                1,
                hd_mla,
                nope,
                rope,
                hd_mla,
                stream,
            )?;
            ops::rope_yarn(
                ctx.gpu,
                self.rope_yarn_interleaved_k,
                comp_rope_tmp,
                comp_rope_tmp,
                comp_positions,
                n_win,
                0,
                1,
                rope,
                rope,
                mla.yarn_inv_freq,
                super::super::helpers::yarn_rope_mscale(ctx.config),
                stream,
            )?;
            ops::mla_q_rope_writeback_batched(
                ctx.gpu,
                self.mla_q_rope_writeback_batched_k,
                comp_rope_tmp,
                comp_k,
                n_win,
                1,
                hd_mla,
                nope,
                rope,
                hd_mla,
                stream,
            )?;
            // ── 4b increment-1: persist prefill's compressed blocks (FP8) ──
            // comp_k is now final (rope'd, bf16). Quantize the n_win blocks to
            // FP8-E4M3 into the layer's persistent flat pool (blocks [0, n_win))
            // so decode reads raw + compressed arms at ONE dtype/scale (single
            // online softmax). Was ephemeral scratch (moe_output), discarded.
            // No serve behavior change yet — written, not yet read by decode.
            // V4 fp8-KV uses static k_scale=1.0 (fp8_calibration off) → the raw
            // arm's write (reshape_and_cache_fp8, scale 1.0) is a plain e4m3 cast,
            // which bf16_to_fp8 matches exactly. If V4 ever calibrates (scale!=1.0)
            // this needs a scale-aware cast — guarded below.
            let (k_scale, _v_scale) = self.effective_fp8_scales();
            debug_assert!(
                (k_scale - 1.0).abs() < 1e-6,
                "V4 compressed-pool persist assumes k_scale=1.0 (got {k_scale}); add scale-aware cast"
            );
            let n_elems = (n_win * hd_mla) as usize; // hd_mla=576 even → even (bf16_to_fp8 req.)
            debug_assert!(
                (n_win as usize) <= comp.pool_blocks,
                "V4 compressed pool overflow: n_win={n_win} > pool_blocks={}",
                comp.pool_blocks
            );
            ops::bf16_to_fp8(
                ctx.gpu,
                self.bf16_to_fp8_k,
                comp_k,
                comp.pool,
                n_elems as u32,
                stream,
            )?;
            // Record how many compressed blocks prefill wrote → decode's compressed
            // arm attends exactly [0, n_win). inc-2: single-shot prefill, blocks at
            // pool offset 0, no decode-time append. (Chunked prefill would need an
            // offset + accumulate — an inc-3 concern alongside decode-append.)
            self.v4_comp_pool_filled
                .store(n_win, std::sync::atomic::Ordering::Relaxed);
            KernelLaunch::new(ctx.gpu, self.prefill_attn_compressed_k)
                .grid([nq, n.div_ceil(16), 1])
                .block([128, 1, 1])
                .arg_ptr(q_full)
                .arg_ptr(k_out)
                // MLA: V==K (rope in the tail), for both the raw and compressed KV.
                .arg_ptr(k_out)
                .arg_ptr(comp_k)
                .arg_ptr(comp_k)
                .arg_ptr(mla.attn_sink)
                .arg_ptr(attn_out)
                .arg_u32(n)
                .arg_u32(nq)
                .arg_u32(nkv)
                .arg_u32(hd_mla)
                .arg_u32(n_win)
                .arg_u32(ratio)
                .arg_u32(V4_WINDOW)
                .arg_f32(1.0f32 / (hd_mla as f32).sqrt())
                .launch(stream)?;
            true
        } else {
            false
        };
        // 4b inc-3: seed the decode-append ring from prefill's tail so decode-time
        // appends are correct from the FIRST window — mirrors the reference
        // Compressor.forward(start_pos==0) kv_state seed (model.py:330-335):
        //  - CSA overlap: prev_win = the last FULL window's normed-x (= the next
        //    window's Ca half). Without it the first decode block's Ca is masked →
        //    one wrong block attended for the rest of the sequence (the observed
        //    early-loop regression).
        //  - both CSA/HCA: the trailing partial-window (`remainder`) tokens seed the
        //    ring's leading slots so decode COMPLETES that straddle window. HCA with
        //    a prompt shorter than one window has n_win=0 and is gated OUT of the
        //    compress branch above, yet still must seed here (else its window-0
        //    decode append is built from an empty ring).
        // Runs for EVERY compressor layer (uses mla.compressor directly, not the
        // n/ratio-gated `csa`). Reset here = the per-sequence clean start.
        if let Some(comp) = mla.compressor.as_ref() {
            use std::sync::atomic::Ordering::Relaxed;
            let cratio = comp.ratio as u32;
            let cnwin = n / cratio;
            let crem = n % cratio;
            let hbytes = h as usize * 2;
            self.v4_decode_started.store(false, Relaxed);
            if comp.is_csa && cnwin > 0 {
                ctx.gpu.copy_d2d_async(
                    normed.offset(((cnwin - 1) * cratio) as usize * hbytes),
                    comp.prev_win,
                    cratio as usize * hbytes,
                    stream,
                )?;
                self.v4_comp_prev_valid.store(true, Relaxed);
            } else {
                self.v4_comp_prev_valid.store(false, Relaxed);
            }
            if crem > 0 {
                ctx.gpu.copy_d2d_async(
                    normed.offset((cnwin * cratio) as usize * hbytes),
                    comp.ring,
                    crem as usize * hbytes,
                    stream,
                )?;
            }
        }
        if !did_csa {
            // V4 full-attention (non-CSA) is always HDIM=512 → the 512 kernel.
            // MLA: V==K (k_out carries the rope tail; v_out is the plain latent).
            // Pass the per-head attention sink so the softmax denominator matches
            // the decode path (the reference applies the sink on EVERY layer).
            ops::prefill_attention_512_sink(
                ctx.gpu,
                self.prefill_attn_512_k,
                q_full,
                k_out,
                k_out,
                attn_out,
                n,
                1,
                nq,
                nkv,
                hd_mla,
                1.0f32 / (hd_mla as f32).sqrt(),
                true,
                V4_WINDOW,
                mla.attn_sink,
                stream,
            )
            .map_err(|e| anyhow::anyhow!("V4 attn: prefill_attention_512_sink failed: {e}"))?;
        }
        ctx.gpu
            .synchronize(stream)
            .map_err(|e| anyhow::anyhow!("V4 attn: prefill_attention sync failed: {e}"))?;

        // DeepSeek-V4 eq.26: de-rotate the attention output by each query position
        // (inverse interleaved YaRN RoPE on the trailing rope dims) before o_proj.
        {
            let o_rope_tmp = ctx.buffers.ssm_conv_out_f32();
            ops::mla_q_rope_extract_batched(
                ctx.gpu,
                self.mla_q_rope_extract_batched_k,
                attn_out,
                o_rope_tmp,
                n,
                nq,
                hd_mla,
                nope,
                rope,
                nq * hd_mla,
                stream,
            )?;
            ops::rope_yarn(
                ctx.gpu,
                self.rope_yarn_interleaved_inv_k,
                o_rope_tmp,
                o_rope_tmp,
                meta.positions,
                n,
                nq,
                0,
                rope,
                rope,
                mla.yarn_inv_freq,
                super::super::helpers::yarn_rope_mscale(ctx.config),
                stream,
            )?;
            ops::mla_q_rope_writeback_batched(
                ctx.gpu,
                self.mla_q_rope_writeback_batched_k,
                o_rope_tmp,
                attn_out,
                n,
                nq,
                hd_mla,
                nope,
                rope,
                nq * hd_mla,
                stream,
            )?;
        }
        if diag_this {
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                attn_out,
                (nq * hd_mla) as usize,
                stream,
                &format!("V4-prefill L{} attn_out token0", self.attn_layer_idx),
            );
            let last_token_offset = ((n - 1) * nq * hd_mla * 2) as usize;
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                attn_out.offset(last_token_offset),
                (nq * hd_mla) as usize,
                stream,
                &format!("V4-prefill L{} attn_out last", self.attn_layer_idx),
            );
        }

        // ── 5. Assemble KV cache (V4-Flash: requires latent+rope assembly) ──
        // NOTE: k_out is 512-dim (complete K), but cache needs 576-dim (512 latent + 64 rope).
        // We need to extract the latent portion (first 512 dims), reassemble with rope, then write.
        let k_cache_assembled = ctx.buffers.expert_up_out();
        let v_cache_assembled = ctx.buffers.expert_down_out();
        ops::mla_cache_assemble_batched(
            ctx.gpu,
            self.mla_cache_assemble_batched_k,
            kv_latent,  // 512-dim latent (reused from step 2)
            k_rope_tmp, // 64-dim RoPE from K (reused from step 3)
            k_cache_assembled,
            v_cache_assembled,
            n,
            kv_lora,
            rope,
            mla_cache_dim,
            stream,
        )?;
        self.write_kv_cache(
            ctx.gpu,
            k_cache_assembled,
            v_cache_assembled,
            kv_cache,
            meta.slot,
            n,
            1,
            mla_cache_dim,
            kv_cache.block_size() as u32,
            mla_cache_dim,
            mla_cache_dim,
            stream,
            ctx.graph_capture,
        )?;
        ctx.gpu
            .synchronize(stream)
            .map_err(|e| anyhow::anyhow!("V4 attn: write_kv_cache sync failed: {e}"))?;

        // ── 6. Grouped low-rank O projection (block-diagonal wo_a → wo_b) ──
        // wo_a is block-diagonal over o_groups (DeepseekV4GroupedLinear); see
        // decode/attention_forward_v4.rs. Per-token×group GEMVs avoid the
        // strided-input limitation of dense_gemm; wo_b stays one GEMM.
        let o_groups = ctx.config.o_groups.max(1) as u32;
        let group_in = (nq * hd_mla) / o_groups;
        let latent_dim = o_groups * o_lora;
        let o_latent = ctx.buffers.o_latent();
        let o_out = ctx.buffers.qkv_output();
        for t in 0..n {
            for g in 0..o_groups {
                let in_g = attn_out.offset(((t * nq * hd_mla) + g * group_in) as usize * 2);
                let w_g = crate::weight_map::DenseWeight {
                    weight: mla
                        .wo_a
                        .weight
                        .offset((g as usize) * (o_lora as usize) * (group_in as usize) * 2),
                };
                let out_g = o_latent.offset(((t * latent_dim) + g * o_lora) as usize * 2);
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    in_g,
                    &w_g,
                    out_g,
                    o_lora,
                    group_in,
                    stream,
                )?;
            }
        }
        ctx.gpu
            .synchronize(stream)
            .map_err(|e| anyhow::anyhow!("V4 attn: wo_a grouped gemv sync failed: {e}"))?;
        ops::dense_gemm(
            ctx.gpu,
            self.dense_gemm_k,
            o_latent,
            &mla.wo_b,
            o_out,
            n,
            h,
            latent_dim,
            stream,
        )?;
        ctx.gpu
            .synchronize(stream)
            .map_err(|e| anyhow::anyhow!("V4 attn: wo_b gemm sync failed: {e}"))?;
        if diag_this {
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                o_out,
                h as usize,
                stream,
                &format!("V4-prefill L{} o_out token0", self.attn_layer_idx),
            );
            let last_token_offset = ((n - 1) * h * 2) as usize;
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                o_out.offset(last_token_offset),
                h as usize,
                stream,
                &format!("V4-prefill L{} o_out last", self.attn_layer_idx),
            );
        }

        Ok(o_out)
    }
}
