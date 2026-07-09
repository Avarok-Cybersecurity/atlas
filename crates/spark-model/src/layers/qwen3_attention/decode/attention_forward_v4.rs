// SPDX-License-Identifier: AGPL-3.0-only

//! DeepSeek-V4-Flash decode path. Reuses low-rank Q projection (wq_a→norm→wq_b)
//! from the MLA path, but uses direct KV projection (no absorption) and
//! grouped low-rank O projection (wo_a→wo_b).

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::super::Qwen3AttentionLayer;
use crate::layer::ForwardContext;
use crate::layers::ops;

impl Qwen3AttentionLayer {
    /// Run the DeepSeek-V4-Flash decode chain. Returns the O-projection
    /// output (`ctx.buffers.qkv_output()`).
    ///
    /// Visibility is widened to the whole `qwen3_attention` module so the
    /// multi-sequence batched-decode path (`trait_impl::multi_seq::mla`)
    /// can drive this exact single-token chain once per verify token —
    /// the V4-Flash direct-KV algorithm is the SSOT here, NOT the absorbed
    /// MLA chain used by Mistral-Small-4.
    pub(in crate::layers::qwen3_attention) fn attention_forward_v4(
        &self,
        kv_cache: &mut PagedKvCache,
        ctx: &ForwardContext,
        args: &super::attention_forward_mla::DecodeMlaArgs,
    ) -> Result<DevicePtr> {
        let super::attention_forward_mla::DecodeMlaArgs {
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
            pos,
        } = *args;
        let mla = self
            .mla
            .as_ref()
            .expect("attention_forward_v4 called without MLA config");
        let meta = ctx
            .attn_metadata
            .expect("V4-Flash decode requires pre-uploaded metadata");

        let q_lora = mla.q_lora_rank as u32;
        let mla_rope = mla.rope as u32;
        let o_lora = mla.o_lora_rank as u32;
        let nkv = ctx.config.num_key_value_heads as u32;
        let profile = ctx.profile;
        let diag_all =
            std::env::var("ATLAS_DIAG_V4_ALL_LAYERS").is_ok_and(|v| v == "1" || v == "true");
        let diag_this = self.attn_layer_idx == 0 || diag_all;
        macro_rules! prof {
            ($label:expr, $body:expr) => {{
                if profile {
                    let _t = std::time::Instant::now();
                    let _r = $body;
                    ctx.gpu.synchronize(stream)?;
                    tracing::info!("    V4 {}: {:.0}µs", $label, _t.elapsed().as_micros());
                    _r
                } else {
                    $body
                }
            }};
        }

        // ── 4b inc-3: decode-time compressed-block append ──
        // Capture this token's compressor input (`normed`, the layer-input
        // RMSNorm output prefill's `cache_skip_v4` feeds `wkv`/`wgate`) into a
        // per-layer BF16 ring, and at each window boundary rerun prefill's
        // compress pipeline over the ring to append ONE FP8 pool block —
        // restoring the double-representation (raw sliding window + compressed
        // history) that inc-2 froze at the prefill count. Runs BEFORE the Q/K/V
        // compute so the MoE scratch buffers (expert_up_out/…) are free, exactly
        // as prefill uses them. Single-sequence eager decode only: `pos` is None
        // on the batched/MTP path, and a captured graph can't re-run host logic.
        if let (Some(pos), Some(comp)) = (pos, mla.compressor.as_ref())
            && meta.num_seqs == 1
            && !ctx.graph_capture
        {
            {
                use std::sync::atomic::Ordering::Relaxed;
                let ratio = comp.ratio as u32;
                let proj_dim = comp.proj_dim as u32;
                let nope = mla.nope as u32;
                let rope_d = mla_rope;
                let hd_mla = nope + rope_d; // compressed block width (= q head_dim)
                let hb = h as usize * 2; // BF16 bytes per token row

                // Capture normed → ring slot (pos % ratio). At a boundary the
                // ring then holds the completed window's `ratio` tokens in order.
                let slot = (pos % ratio) as usize;
                ctx.gpu
                    .copy_d2d_async(normed, comp.ring.offset(slot * hb), hb, stream)?;

                if (pos + 1) % ratio == 0 {
                    let w = (pos + 1) / ratio - 1;
                    let filled = self.v4_comp_pool_filled.load(Relaxed);
                    // Append the next window we don't already hold. Straddle
                    // windows are handled by the prefill→decode ring seed
                    // (cache_skip_v4), so the ring is always complete here — no
                    // coverage guard needed. prev_win is seeded (CSA) so the very
                    // first decode block gets its real Ca (not masked).
                    if w >= filled {
                        use spark_runtime::kernel_args::KernelLaunch;
                        let prev_valid = self.v4_comp_prev_valid.load(Relaxed);
                        // CSA with a real previous window → 2×ratio overlap
                        // (grid[2], take block 1). HCA, or the first CSA window
                        // (Ca masked = window-0 semantics), → ring only (grid[1],
                        // block 0). See csa_compress.cu for the Ca/Cb layout.
                        let (comp_in, t_rows, launch_win, tgt) = if comp.is_csa && prev_valid {
                            ctx.gpu.copy_d2d_async(
                                comp.prev_win,
                                comp.stage,
                                ratio as usize * hb,
                                stream,
                            )?;
                            ctx.gpu.copy_d2d_async(
                                comp.ring,
                                comp.stage.offset(ratio as usize * hb),
                                ratio as usize * hb,
                                stream,
                            )?;
                            (comp.stage, 2 * ratio, 2u32, 1u32)
                        } else {
                            (comp.ring, ratio, 1u32, 0u32)
                        };

                        // compressor projections kv/gate = W·comp_in [T, proj_dim]
                        let kv_comp = ctx.buffers.expert_up_out();
                        let gate_comp = ctx.buffers.expert_down_out();
                        ops::dense_gemm(
                            ctx.gpu,
                            self.dense_gemm_k,
                            comp_in,
                            &comp.wkv,
                            kv_comp,
                            t_rows,
                            proj_dim,
                            h,
                            stream,
                        )?;
                        ops::dense_gemm(
                            ctx.gpu,
                            self.dense_gemm_k,
                            comp_in,
                            &comp.wgate,
                            gate_comp,
                            t_rows,
                            proj_dim,
                            h,
                            stream,
                        )?;
                        // window softmax-gated compression → [launch_win, hd_mla]
                        let compressed = ctx.buffers.moe_output();
                        KernelLaunch::new(ctx.gpu, self.csa_compress_k)
                            .grid([launch_win, 1, 1])
                            .block([256, 1, 1])
                            .arg_ptr(kv_comp)
                            .arg_ptr(gate_comp)
                            .arg_ptr(comp.ape)
                            .arg_ptr(compressed)
                            .arg_u32(t_rows)
                            .arg_u32(ratio)
                            .arg_u32(hd_mla)
                            .arg_u32(proj_dim)
                            .arg_u32(if comp.is_csa { 1 } else { 0 })
                            .launch(stream)?;
                        // rms_norm the target block in place (matches prefill).
                        let block = compressed.offset(tgt as usize * hd_mla as usize * 2);
                        ops::rms_norm(
                            ctx.gpu,
                            self.rms_norm_k,
                            block,
                            &comp.norm,
                            block,
                            1,
                            hd_mla,
                            eps,
                            stream,
                        )?;
                        // comp_k = rope(block): copy → extract tail → yarn @ w*ratio
                        // → writeback. Uses the window's compress position w*ratio,
                        // theta = yarn_inv_freq, interleaved — mirrors prefill.
                        let comp_k = compressed.offset(launch_win as usize * hd_mla as usize * 2);
                        ctx.gpu
                            .copy_d2d_async(block, comp_k, hd_mla as usize * 2, stream)?;
                        let pos_bytes = (w * ratio).to_le_bytes();
                        let comp_positions = ctx.buffers.ssm_ba();
                        ctx.gpu.copy_h2d_async(&pos_bytes, comp_positions, stream)?;
                        let comp_rope_tmp = ctx.buffers.ssm_conv_out_f32();
                        ops::mla_q_rope_extract_batched(
                            ctx.gpu,
                            self.mla_q_rope_extract_batched_k,
                            comp_k,
                            comp_rope_tmp,
                            1,
                            1,
                            hd_mla,
                            nope,
                            rope_d,
                            hd_mla,
                            stream,
                        )?;
                        ops::rope_yarn(
                            ctx.gpu,
                            self.rope_yarn_interleaved_k,
                            comp_rope_tmp,
                            comp_rope_tmp,
                            comp_positions,
                            1,
                            0,
                            1,
                            rope_d,
                            rope_d,
                            mla.yarn_inv_freq,
                            super::super::helpers::yarn_rope_mscale(ctx.config),
                            stream,
                        )?;
                        ops::mla_q_rope_writeback_batched(
                            ctx.gpu,
                            self.mla_q_rope_writeback_batched_k,
                            comp_rope_tmp,
                            comp_k,
                            1,
                            1,
                            hd_mla,
                            nope,
                            rope_d,
                            hd_mla,
                            stream,
                        )?;
                        // Quantize the rope'd block into pool[w] (FP8, 1 byte/elem,
                        // k_scale=1.0 → plain e4m3 cast, matches the raw KV arm).
                        ops::bf16_to_fp8(
                            ctx.gpu,
                            self.bf16_to_fp8_k,
                            comp_k,
                            comp.pool.offset(w as usize * hd_mla as usize),
                            hd_mla,
                            stream,
                        )?;
                        // Publish: decode's compressed arm now attends [0, w+1).
                        self.v4_comp_pool_filled.store(w + 1, Relaxed);
                        // CSA: this window becomes the next window's Ca source.
                        if comp.is_csa {
                            ctx.gpu.copy_d2d_async(
                                comp.ring,
                                comp.prev_win,
                                ratio as usize * hb,
                                stream,
                            )?;
                            self.v4_comp_prev_valid.store(true, Relaxed);
                        }
                    }
                }
            }
        }

        // ── Step 1: Q latent → norm → expand ──
        let q_latent = ctx.buffers.ssm_ba();
        prof!("wq_a", {
            if let Some(ref wqa_nvfp4) = mla.wq_a_nvfp4 {
                ops::w4a16_gemv(
                    ctx.gpu,
                    self.w4a16_gemv_k,
                    normed,
                    wqa_nvfp4,
                    q_latent,
                    q_lora,
                    h,
                    stream,
                )
            } else if let Some(ref wqa_fp8) = mla.wq_a_fp8 {
                // Native block-scaled FP8 GEMV — half the weight traffic of BF16,
                // lossless (in-kernel F32 dequant).
                ops::w8a16_gemv(
                    ctx.gpu,
                    self.w8a16_gemv_k,
                    normed,
                    wqa_fp8.weight,
                    wqa_fp8.row_scale,
                    q_latent,
                    q_lora,
                    h,
                    stream,
                )
            } else {
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    normed,
                    &mla.wq_a,
                    q_latent,
                    q_lora,
                    h,
                    stream,
                )
            }
        })?;
        prof!("q_norm", {
            ops::rms_norm(
                ctx.gpu,
                self.rms_norm_k,
                q_latent,
                &mla.q_a_norm,
                q_latent,
                1,
                q_lora,
                eps,
                stream,
            )
        })?;
        prof!("wq_b", {
            if let Some(ref wqb_nvfp4) = mla.wq_b_nvfp4 {
                ops::w4a16_gemv(
                    ctx.gpu,
                    self.w4a16_gemv_k,
                    q_latent,
                    wqb_nvfp4,
                    q_out,
                    q_dim,
                    q_lora,
                    stream,
                )
            } else if let Some(ref wqb_fp8) = mla.wq_b_fp8 {
                ops::w8a16_gemv(
                    ctx.gpu,
                    self.w8a16_gemv_k,
                    q_latent,
                    wqb_fp8.weight,
                    wqb_fp8.row_scale,
                    q_out,
                    q_dim,
                    q_lora,
                    stream,
                )
            } else {
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    q_latent,
                    &mla.wq_b,
                    q_out,
                    q_dim,
                    q_lora,
                    stream,
                )
            }
        })?;
        // q_b_norm: per-head unweighted RMSNorm over head_dim (DeepSeek-V4).
        // Reference (DeepseekV4UnweightedRMSNorm) renormalizes each of nq heads'
        // hd-dim Q vector to unit RMS BEFORE rope. Missing this makes Q ~sqrt(hd)x
        // too small → near-flat softmax → incoherent output. Weight = all-ones.
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            q_out,
            &crate::weight_map::DenseWeight {
                weight: ctx.buffers.norm_unit_w(),
            },
            q_out,
            nq,
            hd,
            eps,
            stream,
        )?;
        if diag_this {
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                q_out,
                q_dim as usize,
                stream,
                &format!("V4-decode L{} Q after q_b_norm", self.attn_layer_idx),
            );
        }

        // ── Step 2: Direct KV projection ──
        let kv_dim = nkv * hd;
        prof!("wkv", {
            if let Some(ref wkva_nvfp4) = mla.wkv_a_nvfp4 {
                ops::w4a16_gemv(
                    ctx.gpu,
                    self.w4a16_gemv_k,
                    normed,
                    wkva_nvfp4,
                    k_out,
                    kv_dim,
                    h,
                    stream,
                )
            } else if let Some(ref wkva_fp8) = mla.wkv_a_fp8 {
                ops::w8a16_gemv(
                    ctx.gpu,
                    self.w8a16_gemv_k,
                    normed,
                    wkva_fp8.weight,
                    wkva_fp8.row_scale,
                    k_out,
                    kv_dim,
                    h,
                    stream,
                )
            } else {
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    normed,
                    &mla.wkv_a,
                    k_out,
                    kv_dim,
                    h,
                    stream,
                )
            }
        })?;
        // kv_norm: weighted RMSNorm over the kv latent BEFORE rope (DeepSeek-V4
        // reference: kv = kv_norm(kv_proj(h))). Missing this left K ~8x too large
        // → attention score overflow → NaN. nkv heads × (kv_dim/nkv) each.
        ops::rms_norm(
            ctx.gpu,
            self.rms_norm_k,
            k_out,
            &mla.kv_a_norm,
            k_out,
            nkv,
            kv_dim / nkv,
            eps,
            stream,
        )?;
        // K=V for V4-Flash direct KV projection
        ctx.gpu
            .copy_d2d_async(k_out, v_out, (kv_dim as usize) * 2, stream)?;
        if diag_this {
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                k_out,
                kv_dim as usize,
                stream,
                &format!("V4-decode L{} K after proj", self.attn_layer_idx),
            );
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                v_out,
                kv_dim as usize,
                stream,
                &format!("V4-decode L{} V after copy", self.attn_layer_idx),
            );
        }

        // ── Step 3: RoPE for Q and K ──
        // V4-Flash: rope dims are at offset `nope` per head (matching MLA layout),
        // not at the beginning. Extract → RoPE → writeback.
        let q_rope_tmp = ctx.buffers.ssm_conv_out_f32();
        let k_rope_tmp = q_latent; // reuse after wq_b is done
        prof!("rope_extract", {
            ops::mla_q_rope_extract_batched(
                ctx.gpu,
                self.mla_q_rope_extract_batched_k,
                q_out,
                q_rope_tmp,
                1,
                nq,
                hd,
                mla.nope as u32,
                mla_rope,
                nq * hd,
                stream,
            )
        })?;
        // Extract K's rope channels too (MQA: 1 kv head, stride hd). The decode
        // path previously skipped this — `k_rope_tmp` (= reused q_latent) held
        // stale data, so `rope_yarn` rotated garbage and the cached keys got
        // near-zero positional signal → attention degenerates after a few decode
        // tokens. Mirrors the prefill K extract (cache_skip_v4.rs:304).
        prof!("k_rope_extract", {
            ops::mla_q_rope_extract_batched(
                ctx.gpu,
                self.mla_q_rope_extract_batched_k,
                k_out,
                k_rope_tmp,
                1,
                1,
                hd,
                mla.nope as u32,
                mla_rope,
                hd,
                stream,
            )
        })?;
        prof!("rope", {
            ops::rope_yarn(
                ctx.gpu,
                // DeepSeek-V4 uses INTERLEAVED RoPE (rope_interleave=True): adjacent
                // channel pairs (2i, 2i+1), matching the HF reference's rotate_half
                // over cos.repeat_interleave(2). The non-interleaved (NeoX, i/i+half)
                // kernel scrambles positions -> incoherent output.
                self.rope_yarn_interleaved_k,
                q_rope_tmp,
                k_rope_tmp,
                meta.positions,
                1,
                nq,
                1,
                mla_rope,
                mla_rope,
                // Sliding layers (compressor==None) = reference "main" rope:
                // plain θ=10000, mscale=1 (no yarn). CSA/HCA keep θ=160000 yarn.
                if mla.compressor.is_none() {
                    mla.main_inv_freq
                } else {
                    mla.yarn_inv_freq
                },
                if mla.compressor.is_none() {
                    1.0f32
                } else {
                    super::super::helpers::yarn_rope_mscale(ctx.config)
                },
                stream,
            )
        })?;
        prof!("rope_writeback", {
            ops::mla_q_rope_writeback_batched(
                ctx.gpu,
                self.mla_q_rope_writeback_batched_k,
                q_rope_tmp,
                q_out,
                1,
                nq,
                hd,
                mla.nope as u32,
                mla_rope,
                nq * hd,
                stream,
            )
        })?;
        prof!("k_rope_writeback", {
            ops::mla_q_rope_writeback_batched(
                ctx.gpu,
                self.mla_q_rope_writeback_batched_k,
                k_rope_tmp,
                k_out,
                1,
                1,
                hd,
                mla.nope as u32,
                mla_rope,
                hd,
                stream,
            )
        })?;
        if diag_this {
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                k_out,
                kv_dim as usize,
                stream,
                &format!("V4-decode L{} K after RoPE", self.attn_layer_idx),
            );
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                k_out.offset(mla.nope * 2),
                (kv_dim - mla.nope as u32) as usize,
                stream,
                &format!("V4-decode L{} K rope after RoPE", self.attn_layer_idx),
            );
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                q_out.offset(mla.nope * 2),
                (hd - mla.nope as u32) as usize,
                stream,
                &format!("V4-decode L{} Q rope after RoPE", self.attn_layer_idx),
            );
        }

        // ── Step 3.5: Assemble KV cache (V4-Flash: requires latent+rope assembly) ──
        // Cache needs 576-dim (512 latent + 64 rope), but k_out/v_out are 512-dim.
        // Extract RoPE from Q (which has correct [nope|rope] structure) and reuse for K cache.
        let k_cache_assembled = ctx.buffers.ssm_deinterleaved();
        let v_cache_assembled = ctx.buffers.ssm_qkvz();
        let kv_lora = mla.kv_lora_rank as u32;
        let mla_cache_dim = kv_lora + mla_rope;
        prof!("cache_assemble", {
            ops::mla_cache_assemble_batched(
                ctx.gpu,
                self.mla_cache_assemble_batched_k,
                v_out,      // 512-dim latent K (unmodified copy before RoPE writeback)
                k_rope_tmp, // 64-dim RoPE from K
                k_cache_assembled,
                v_cache_assembled,
                1,
                kv_lora,
                mla_rope,
                mla_cache_dim,
                stream,
            )
        })?;

        // ── Step 4: Write assembled K/V to paged cache ──
        prof!("write_kv_cache", {
            self.write_kv_cache(
                ctx.gpu,
                k_cache_assembled,
                v_cache_assembled,
                kv_cache,
                meta.slot,
                1,
                1,
                mla_cache_dim,
                bs as u32,
                mla_cache_dim,
                mla_cache_dim,
                stream,
                ctx.graph_capture,
            )
        })?;

        // ── Step 5: Paged decode attention ──
        let attn_out = ctx.buffers.attn_output();
        let inv_sqrt_d = self.effective_attn_scale(hd);
        prof!("paged_attn", {
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
            )
        })?;
        if diag_this {
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                attn_out,
                (nq * hd) as usize,
                stream,
                &format!("V4-decode L{} attn_out", self.attn_layer_idx),
            );
        }

        // ── Step 5.5: Attention-output de-rotation (DeepSeek-V4 eq.26) ──
        // The reference de-rotates the attention output by the query position
        // (apply_rotary(attn_out, cos, -sin)) so each value's contribution is
        // relative-distance. Since V==K carries the rotated rope in its trailing
        // `mla_rope` dims, undo that rotation on the output before o_proj. Reuse
        // the Q rope extract/writeback with the conjugate (negated-sin) kernel.
        {
            let o_rope_tmp = ctx.buffers.ssm_conv_out_f32();
            ops::mla_q_rope_extract_batched(
                ctx.gpu,
                self.mla_q_rope_extract_batched_k,
                attn_out,
                o_rope_tmp,
                1,
                nq,
                hd,
                mla.nope as u32,
                mla_rope,
                nq * hd,
                stream,
            )?;
            ops::rope_yarn(
                ctx.gpu,
                self.rope_yarn_interleaved_inv_k,
                o_rope_tmp,
                o_rope_tmp,
                meta.positions,
                1,
                nq,
                0, // no KV heads — de-rotate the query/output heads only
                mla_rope,
                mla_rope,
                // MUST match the Q/K rope inv_freq for this layer type (rope-in
                // == de-rotate-out), else the output is scrambled. Sliding =
                // main θ=10000; CSA/HCA = θ=160000 yarn.
                if mla.compressor.is_none() {
                    mla.main_inv_freq
                } else {
                    mla.yarn_inv_freq
                },
                if mla.compressor.is_none() {
                    1.0f32
                } else {
                    super::super::helpers::yarn_rope_mscale(ctx.config)
                },
                stream,
            )?;
            ops::mla_q_rope_writeback_batched(
                ctx.gpu,
                self.mla_q_rope_writeback_batched_k,
                o_rope_tmp,
                attn_out,
                1,
                nq,
                hd,
                mla.nope as u32,
                mla_rope,
                nq * hd,
                stream,
            )?;
        }

        // ── Step 6: Grouped low-rank O projection (wo_a → wo_b) ──
        // wo_a is BLOCK-DIAGONAL (DeepseekV4GroupedLinear): the n_heads*head_dim
        // attention output is split into `o_groups` independent groups, each
        // projected group_in -> o_lora. Weight layout [o_groups*o_lora, group_in].
        // A single dense GEMV would mix across groups and (with o_lora<latent_dim)
        // read only 1/o_groups of wo_b — producing garbage every layer.
        let o_groups = ctx.config.o_groups.max(1) as u32;
        let group_in = (nq * hd) / o_groups; // 4096 = (64*512)/8
        let latent_dim = o_groups * o_lora; // 8192 = 8*1024
        let o_latent = ctx.buffers.o_latent();
        let o_out = ctx.buffers.qkv_output();
        prof!("wo_a_grouped", {
            for g in 0..o_groups {
                let in_g = attn_out.offset((g * group_in) as usize * 2);
                let out_g = o_latent.offset((g * o_lora) as usize * 2);
                if let Some(ref woa_fp8) = mla.wo_a_fp8 {
                    // Native block-scaled FP8 per group (block-diagonal):
                    // weight rows [g*o_lora:(g+1)*o_lora] (fp8, 1 byte/elem) and the
                    // matching [o_lora/128, group_in/128] block-scale sub-tile.
                    let w_off = (g as usize) * (o_lora as usize) * (group_in as usize); // fp8 bytes
                    let s_off =
                        (g as usize) * (o_lora as usize / 128) * (group_in as usize / 128) * 4; // FP32 block-scale bytes
                    ops::w8a16_gemv(
                        ctx.gpu,
                        self.w8a16_gemv_k,
                        in_g,
                        woa_fp8.weight.offset(w_off),
                        woa_fp8.row_scale.offset(s_off),
                        out_g,
                        o_lora,
                        group_in,
                        stream,
                    )?;
                } else {
                    let w_g = crate::weight_map::DenseWeight {
                        weight: mla
                            .wo_a
                            .weight
                            .offset((g as usize) * (o_lora as usize) * (group_in as usize) * 2),
                    };
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
            Ok::<(), anyhow::Error>(())
        })?;
        prof!("wo_b", {
            if let Some(ref wob_fp8) = mla.wo_b_fp8 {
                ops::w8a16_gemv(
                    ctx.gpu,
                    self.w8a16_gemv_k,
                    o_latent,
                    wob_fp8.weight,
                    wob_fp8.row_scale,
                    o_out,
                    h,
                    latent_dim,
                    stream,
                )
            } else {
                ops::dense_gemv(
                    ctx.gpu,
                    self.dense_gemv_k,
                    o_latent,
                    &mla.wo_b,
                    o_out,
                    h,
                    latent_dim,
                    stream,
                )
            }
        })?;
        if diag_this {
            super::super::trait_impl::diag_norm(
                ctx.gpu,
                o_out,
                h as usize,
                stream,
                &format!("V4-decode L{} o_out", self.attn_layer_idx),
            );
        }

        Ok(o_out)
    }
}
