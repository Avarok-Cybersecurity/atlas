// SPDX-License-Identifier: AGPL-3.0-only

//! MoeLayer::forward (decode).

use super::*;

impl MoeLayer {
    /// True when the ATLAS_FP32_ROUTING path is active: the SSM-side MoE-input
    /// norm should emit an FP32 `router_in` (residual_add_rms_norm_gatef32) which
    /// the gate GEMM then consumes at full precision. Requires the f32 kernels to
    /// be present and the softmax-routed dense-gate config (NVFP4 gate / sigmoid+bias
    /// stay BF16). Default off → BF16 routing unchanged.
    pub fn fp32_routing_active(&self) -> bool {
        self.gate_nvfp4.is_none()
            && self.correction_bias_dev.is_none()
            && self.dense_gemm_f32in.0 != 0
            && self.moe_topk_f32.0 != 0
            && std::env::var("ATLAS_FP32_ROUTING").as_deref() == Ok("1")
    }

    /// Forward pass: gate → top-K routing → batched expert FFN → blend.
    ///
    /// All expert dispatch stays on device — zero D2H synchronization.
    /// 9 kernel launches per MoE layer (down from 58).
    ///
    /// When `gelu_activation` is true, falls back to the sorted prefill path
    /// (which uses separate activation kernel) to avoid fused SiLU decode kernels.
    /// LongCat zero-computation experts: `out[t,:] += zero_accum[t] * x[t,:]`
    /// where `zero_accum` was written by the softmax+bias router kernels
    /// (the folded weights of selected identity experts). MUST run after the
    /// routed blend for the SAME tokens whose routing wrote `zero_accum`.
    /// No-op (no launch) when the model has no zero-experts.
    pub fn apply_zero_expert(
        &self,
        out: spark_runtime::gpu::DevicePtr,
        x: spark_runtime::gpu::DevicePtr,
        n: u32,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<()> {
        if self.router_logits_n as usize == ctx.config.num_experts {
            return Ok(());
        }
        anyhow::ensure!(
            self.moe_zero_expert_add_k.0 != 0,
            "zero-expert model but moe_zero_expert_add kernel is absent from this build"
        );
        ops::moe_zero_expert_add(
            ctx.gpu,
            self.moe_zero_expert_add_k,
            out,
            x,
            self.zero_accum_dev,
            n,
            ctx.config.hidden_size as u32,
            stream,
        )
    }

    pub fn forward(
        &self,
        input: DevicePtr,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        // SOLID Incr-4: a genuine single-token decode (num_seqs == 1) folds the
        // routed expert down_proj LoRA delta below (before the wsum blend). The
        // multi-seq per-token reuse of this fn (num_seqs > 1 — decode_batch's
        // per-token MoE loop, or the attention layers' per-token FFN) shares a
        // `padded_n` CUDA graph across mixed-batch transitions, so host-gating
        // the fold there is capture-unsafe (a Fold-captured graph could replay
        // onto a later base-containing batch): keep the loud refusal until the
        // device per-row `row_adapter` map is plumbed.
        let single_seq_decode = ctx.attn_metadata.as_ref().map_or(1, |m| m.num_seqs) <= 1;
        if !single_seq_decode {
            // Multi-seq per-token reuse shares a `padded_n` CUDA graph across mixed
            // base/adapter batches, so host-gating a fold there is capture-unsafe:
            // keep the loud refusal until the device per-row `row_adapter` map lands.
            self.reject_decode_lora(ctx, "forward")?;
        }
        // Single-seq decode: the router delta folds onto `gate_logits` before top-k
        // (below), and the routed-expert gate/up/down deltas fold onto their
        // intermediates — no bail. A `Refuse`/mixed batch still bails inside each
        // fold's `moe_route_gate`, preserving per-row adapter-identity protection.
        // ── Phase 2.7 Tier C: Frankenstein decode-via-prefill dispatch ──
        // For DFlash capture layers only, when `ATLAS_FRANKENSTEIN_DECODE_VIA_PREFILL=1`
        // is set, route this layer's single-token MoE through `forward_prefill(M=1)`,
        // which uses the tensor-core grouped GEMM kernel (E2M1→E4M3 MMA) instead of
        // the scalar FP32 FMA decode path. Tests whether the numerical recipe of the
        // MoE kernel is the dominant cause of low DFlash drafter acceptance.
        //
        // Other (non-capture) layers fall through to the normal scalar decode path,
        // preserving Atlas's TPS on the bulk of the network. The 5 capture layers
        // pay ~250 µs each (microbench), totalling ≈1.25 ms per token (negligible
        // at Atlas's ~58 ms/token decode latency).
        if self.is_dflash_capture_layer
            && std::env::var("ATLAS_FRANKENSTEIN_DECODE_VIA_PREFILL")
                .ok()
                .as_deref()
                == Some("1")
        {
            // One-time per-process log so we can verify the env-gated route is hit.
            if ctx.stats.once("log:moe_route") {
                tracing::info!(
                    "FRANKENSTEIN: routing DFlash capture-layer MoE decode through forward_prefill(M=1) (one-time log)"
                );
            }
            self.forward_prefill(input, 1, ctx, stream)?;
            return Ok(ctx.buffers.moe_output());
        }

        // GeGLU models: fused kernels now have GELU activation (model-specific override).
        // No longer need to redirect through sorted prefill path.
        // But we still need pre_expert_norm between routing and dispatch.
        // For the fused decode path, apply pre_expert_norm to the input before experts.
        // The gate GEMV already completed on the raw input in the fused path below.

        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.moe_intermediate_size as u32;
        let shared_inter = ctx.config.shared_expert_intermediate_size as u32;
        let num_experts = ctx.config.num_experts as u32;
        let top_k = ctx.config.num_experts_per_tok as u32;
        let profile = ctx.profile;

        macro_rules! prof {
            ($label:expr, $body:expr) => {{
                if profile {
                    let t = std::time::Instant::now();
                    let r = $body;
                    ctx.gpu.synchronize(stream)?;
                    tracing::info!("    MoE {}: {:.0}μs", $label, t.elapsed().as_micros());
                    r
                } else {
                    $body
                }
            }};
        }

        let scratch = ctx.buffers.scratch();
        let indices_dev = scratch;
        let weights_dev = scratch.offset(top_k as usize * 4);

        // Note: moe_gate_topk_fused exists but uses single-CTA design,
        // too slow for 256 experts (serializes computation). Separate path is faster.
        {
            // Gemma-4 router pre-norm (no-op for other models).
            let router_in = self.router_input(input, 1, h, ctx, stream)?;
            let gate_logits = ctx.buffers.gate_logits();
            prof!("gate", {
                if let Some(ref nvfp4) = self.gate_nvfp4 {
                    ops::w4a16_decode_gemv(
                        ctx.gpu,
                        self.w4a16_gemv,
                        self.w4a16_gemv_sw,
                        ctx.levers.gemv_sw,
                        router_in,
                        nvfp4,
                        gate_logits,
                        // = num_experts everywhere except LongCat, whose
                        // router also scores the zero-expert logits.
                        self.router_logits_n,
                        h,
                        stream,
                    )
                } else {
                    ops::dense_gemv(
                        ctx.gpu,
                        self.dense_gemv,
                        router_in,
                        &self.weights.gate,
                        gate_logits,
                        self.router_logits_n,
                        h,
                        stream,
                    )
                }
            })?;

            // Feature-1: fold the router `mlp.gate` LoRA delta onto `gate_logits`
            // BEFORE top-k — the exact decode mirror of the prefill router fold
            // (`apply_router_lora_prefill` is n-generic; here n=1). Device-clean
            // (no D2H) so it captures cleanly. No-op unless a router delta is
            // installed; `Refuse` bails inside the hook. Works for both the
            // NVFP4-gate and dense-gate branches (same `gate_logits` output).
            if single_seq_decode {
                self.apply_router_lora_prefill(router_in, gate_logits, 1, ctx, stream)?;
            }

            prof!("topk", {
                if let Some(tid2eid) = self.tid2eid_dev {
                    // DeepSeek-V4 hash routing (hash_moe layer): expert SELECTION
                    // is the static `tid2eid[token_id]` table; the learned gate
                    // still supplies the sqrtsoftplus scores that weight them.
                    let token_ids = ctx.token_ids.ok_or_else(|| {
                        anyhow::anyhow!(
                            "DeepSeek-V4 hash-MoE layer requires ForwardContext.token_ids (decode)"
                        )
                    })?;
                    ops::moe_hash_route(
                        ctx.gpu,
                        self.moe_hash_route_k,
                        gate_logits,
                        tid2eid,
                        token_ids, // decode: single token at offset 0
                        indices_dev,
                        weights_dev,
                        num_experts,
                        top_k,
                        ctx.config.norm_topk_prob,
                        ctx.config.routed_scaling_factor as f32,
                        stream,
                    )
                } else if let Some(bias) = self.correction_bias_dev {
                    if ctx.config.scoring_func == "sqrtsoftplus" {
                        // DeepSeek-V4 sqrtsoftplus + correction bias:
                        //   scores   = sqrtsoftplus(gate_logits)
                        //   indices  = topk(scores + bias)
                        //   weights  = scores[indices] / sum(scores[indices])
                        ops::moe_topk_sqrtsoftplus(
                            ctx.gpu,
                            self.moe_topk_sqrtsoftplus_k,
                            gate_logits,
                            bias,
                            indices_dev,
                            weights_dev,
                            num_experts,
                            top_k,
                            ctx.config.norm_topk_prob,
                            ctx.config.routed_scaling_factor as f32,
                            stream,
                        )
                    } else if ctx.config.scoring_func == "softmax" {
                        // LongCat-Flash: softmax scores + correction bias for
                        // SELECTION, unbiased softmax * scaling for weights,
                        // zero-expert fold into zero_accum (identity experts
                        // are applied by the caller via apply_zero_expert).
                        ops::moe_topk_softmax_bias(
                            ctx.gpu,
                            self.moe_topk_softmax_bias_k,
                            gate_logits,
                            bias,
                            indices_dev,
                            weights_dev,
                            self.zero_accum_dev,
                            self.router_logits_n,
                            num_experts,
                            top_k,
                            ctx.config.norm_topk_prob,
                            ctx.config.routed_scaling_factor as f32,
                            stream,
                        )
                    } else {
                        // DeepSeek-V3 / MiniMax-M2 sigmoid + correction bias:
                        //   scores   = sigmoid(gate_logits)
                        //   indices  = topk(scores + bias)
                        //   weights  = scores[indices] / sum(scores[indices])
                        // Kernel does all three steps; norm_topk_prob toggles
                        // the final divide. scaling_factor comes from the model
                        // config (e.g., Step 3.7 = 3.0, MiniMax M2 = 1.0).
                        ops::moe_topk_sigmoid(
                            ctx.gpu,
                            self.moe_topk_sigmoid_k,
                            gate_logits,
                            bias,
                            indices_dev,
                            weights_dev,
                            num_experts,
                            top_k,
                            ctx.config.norm_topk_prob,
                            ctx.config.routed_scaling_factor as f32,
                            stream,
                        )
                    }
                } else {
                    ops::moe_topk_softmax(
                        ctx.gpu,
                        self.moe_topk,
                        gate_logits,
                        indices_dev,
                        weights_dev,
                        num_experts,
                        top_k,
                        ctx.config.norm_topk_prob,
                        stream,
                    )
                }
            })?;
        }

        if tracing::enabled!(tracing::Level::DEBUG) && !ctx.graph_capture {
            ctx.gpu.synchronize(stream)?;
            // Read expert indices (u32[top_k]) and weights (f32[top_k])
            let k = top_k as usize;
            let mut idx_buf = vec![0u8; k * 4];
            let mut wt_buf = vec![0u8; k * 4];
            ctx.gpu.copy_d2h(indices_dev, &mut idx_buf)?;
            ctx.gpu.copy_d2h(weights_dev, &mut wt_buf)?;
            let indices: Vec<u32> = (0..k)
                .map(|i| {
                    u32::from_le_bytes([
                        idx_buf[i * 4],
                        idx_buf[i * 4 + 1],
                        idx_buf[i * 4 + 2],
                        idx_buf[i * 4 + 3],
                    ])
                })
                .collect();
            let weights: Vec<f32> = (0..k)
                .map(|i| {
                    f32::from_le_bytes([
                        wt_buf[i * 4],
                        wt_buf[i * 4 + 1],
                        wt_buf[i * 4 + 2],
                        wt_buf[i * 4 + 3],
                    ])
                })
                .collect();
            tracing::info!("  MoE experts: {:?}, weights: {:.4?}", indices, weights);
        }

        // ── Native EXL3 routed experts (ATLAS_EXL3_NATIVE_MOE=1) ──
        // Ahead of every materialized arm: the NVFP4 tables hold nulls when
        // the experts were kept packed. Routing above (GEMV router + top-k)
        // is reused as-is; the arm reads `indices_dev`/`weights_dev` from
        // scratch, folds the probs into the down mgemm, and handles the
        // shared expert + EP tail itself.
        if self.exl3_native_active() {
            return self.forward_exl3_after_routing(
                input,
                1,
                indices_dev,
                weights_dev,
                ctx,
                stream,
            );
        }

        // Apply pre-expert norm AFTER routing, BEFORE expert dispatch (Gemma-4 26B).
        // Write to scratch buffer to preserve original `input` (= residual in caller).
        let expert_input = if let Some(ref norm_w) = self.pre_expert_norm {
            let normed = ctx.buffers.ssm_deinterleaved();
            let eps = ctx.config.rms_norm_eps as f32;
            prof!("pre_expert_norm", {
                ops::rms_norm(
                    ctx.gpu,
                    self.pre_expert_norm_k,
                    input,
                    norm_w,
                    normed,
                    1,
                    h,
                    eps,
                    stream,
                )
            })?;
            normed
        } else {
            input
        };

        // ── Batched expert FFN: 3 GEMV + 1 activation + 1 weighted sum ──
        let expert_gate_out = ctx.buffers.expert_gate_out();
        let expert_up_out = ctx.buffers.expert_up_out();
        let expert_down_out = ctx.buffers.expert_down_out();
        // ⚠ `logits` aliased as shared-gate scratch — concurrent users
        // MUST offset past `shared_expert_intermediate_size * 2`
        // (decode_b.rs:197 uses .offset(65536)). See bug 2 in memory
        // `project_batch_decode_corruption.md` (2026-05-10).
        let shared_gate_scratch = ctx.buffers.logits();
        let shared_up_scratch = ctx.buffers.ssm_qkvz();
        let shared_out = ctx.buffers.attn_output();

        if let (Some(gp), Some(up), Some(dp), Some(shared)) = (
            self.bf16_gate_weight_ptrs,
            self.bf16_up_weight_ptrs,
            self.bf16_down_weight_ptrs,
            self.bf16_shared_expert,
        ) {
            // BF16 path: FP8-dequant-on-load. Eliminates the per-layer 0.989
            // FP8 cosine ceiling by serving experts as BF16 end-to-end.
            prof!("exp_gate_up_bf16", {
                ops::moe_expert_gate_up_shared_bf16(
                    ctx.gpu,
                    self.moe_expert_gate_up_shared_bf16_k,
                    expert_input,
                    gp,
                    expert_gate_out,
                    up,
                    expert_up_out,
                    indices_dev,
                    shared.gate_proj.weight,
                    shared_gate_scratch,
                    shared.up_proj.weight,
                    shared_up_scratch,
                    inter,
                    h,
                    top_k,
                    stream,
                )
            })?;
            if single_seq_decode {
                self.apply_expert_lora_decode_gateup(
                    expert_gate_out,
                    expert_up_out,
                    expert_input,
                    indices_dev,
                    top_k,
                    top_k,
                    DevicePtr::NULL,
                    ctx,
                    stream,
                )?;
            }
            prof!("exp_silu_down_bf16", {
                ops::moe_expert_silu_down_shared_bf16(
                    ctx.gpu,
                    self.moe_expert_silu_down_shared_bf16_k,
                    expert_gate_out,
                    expert_up_out,
                    dp,
                    expert_down_out,
                    indices_dev,
                    shared_gate_scratch,
                    shared_up_scratch,
                    shared.down_proj.weight,
                    shared_out,
                    h,
                    inter,
                    top_k,
                    stream,
                )
            })?;
        } else if let (Some(gp), Some(up), Some(dp), Some(sh)) = (
            &self.fp8_gate_weight_ptrs,
            &self.fp8_up_weight_ptrs,
            &self.fp8_down_weight_ptrs,
            &self.fp8_shared_expert,
        ) {
            // FP8 path: fused expert gate+up with FP8 weight/scale pointer tables
            prof!("exp_gate_up_fp8", {
                ops::moe_expert_gate_up_shared_fp8(
                    ctx.gpu,
                    self.moe_expert_gate_up_shared_fp8,
                    expert_input,
                    gp.weight_ptrs,
                    gp.scale_ptrs,
                    expert_gate_out,
                    up.weight_ptrs,
                    up.scale_ptrs,
                    expert_up_out,
                    indices_dev,
                    &sh.gate_proj,
                    shared_gate_scratch,
                    &sh.up_proj,
                    shared_up_scratch,
                    inter,
                    h,
                    top_k,
                    stream,
                )
            })?;

            if single_seq_decode {
                self.apply_expert_lora_decode_gateup(
                    expert_gate_out,
                    expert_up_out,
                    expert_input,
                    indices_dev,
                    top_k,
                    top_k,
                    DevicePtr::NULL,
                    ctx,
                    stream,
                )?;
            }
            // FP8 path: fused silu+down
            prof!("exp_silu_down_fp8", {
                ops::moe_expert_silu_down_shared_fp8(
                    ctx.gpu,
                    self.moe_expert_silu_down_shared_fp8,
                    expert_gate_out,
                    expert_up_out,
                    dp.weight_ptrs,
                    dp.scale_ptrs,
                    expert_down_out,
                    indices_dev,
                    shared_gate_scratch,
                    shared_up_scratch,
                    &sh.down_proj,
                    shared_out,
                    h,
                    inter,
                    top_k,
                    stream,
                )
            })?;
        } else if self.use_t_layout_for_decode() {
            prof!("exp_unified_t", {
                self.dispatch_unified_t_decode(
                    ctx,
                    expert_input,
                    expert_gate_out,
                    expert_up_out,
                    expert_down_out,
                    shared_gate_scratch,
                    shared_up_scratch,
                    shared_out,
                    indices_dev,
                    h,
                    inter,
                    top_k,
                    single_seq_decode,
                    stream,
                )
            })?;
        } else {
            // NVFP4 path: fused routed+shared gate+up
            prof!("exp_gate_up", {
                ops::moe_expert_gate_up_shared(
                    ctx.gpu,
                    self.moe_expert_gate_up_shared,
                    expert_input,
                    self.gate_ptrs.packed_ptrs,
                    self.gate_ptrs.scale_ptrs,
                    self.gate_ptrs.scale2_vals,
                    expert_gate_out,
                    self.up_ptrs.packed_ptrs,
                    self.up_ptrs.scale_ptrs,
                    self.up_ptrs.scale2_vals,
                    expert_up_out,
                    indices_dev,
                    &self.weights.shared_expert.gate_proj,
                    shared_gate_scratch,
                    &self.weights.shared_expert.up_proj,
                    shared_up_scratch,
                    inter,
                    h,
                    top_k,
                    stream,
                )
            })?;

            if single_seq_decode {
                self.apply_expert_lora_decode_gateup(
                    expert_gate_out,
                    expert_up_out,
                    expert_input,
                    indices_dev,
                    top_k,
                    top_k,
                    DevicePtr::NULL,
                    ctx,
                    stream,
                )?;
            }

            if tracing::enabled!(tracing::Level::DEBUG) && !ctx.graph_capture {
                ctx.gpu.synchronize(stream)?;
                // Dump gate/up outputs for expert slot 0
                let mut gate_buf = vec![0u8; 16];
                ctx.gpu.copy_d2h(expert_gate_out, &mut gate_buf)?;
                let gate_vals: Vec<f32> = (0..8)
                    .map(|i| {
                        let bits = u16::from_le_bytes([gate_buf[i * 2], gate_buf[i * 2 + 1]]);
                        f32::from_bits((bits as u32) << 16)
                    })
                    .collect();
                tracing::info!("  MoE gate_out[slot0,0..8]: {:?}", gate_vals);
                let mut up_buf = vec![0u8; 16];
                ctx.gpu.copy_d2h(expert_up_out, &mut up_buf)?;
                let up_vals: Vec<f32> = (0..8)
                    .map(|i| {
                        let bits = u16::from_le_bytes([up_buf[i * 2], up_buf[i * 2 + 1]]);
                        f32::from_bits((bits as u32) << 16)
                    })
                    .collect();
                tracing::info!("  MoE up_out[slot0,0..8]: {:?}", up_vals);
                // Shared expert gate/up scratch outputs
                let mut sg_buf = vec![0u8; 16];
                ctx.gpu.copy_d2h(shared_gate_scratch, &mut sg_buf)?;
                let sg_vals: Vec<f32> = (0..8)
                    .map(|i| {
                        let bits = u16::from_le_bytes([sg_buf[i * 2], sg_buf[i * 2 + 1]]);
                        f32::from_bits((bits as u32) << 16)
                    })
                    .collect();
                tracing::info!("  MoE shared_gate_scratch[0..8]: {:?}", sg_vals);
                let mut su_buf = vec![0u8; 16];
                ctx.gpu.copy_d2h(shared_up_scratch, &mut su_buf)?;
                let su_vals: Vec<f32> = (0..8)
                    .map(|i| {
                        let bits = u16::from_le_bytes([su_buf[i * 2], su_buf[i * 2 + 1]]);
                        f32::from_bits((bits as u32) << 16)
                    })
                    .collect();
                tracing::info!("  MoE shared_up_scratch[0..8]: {:?}", su_vals);
            }

            // NVFP4 path: fused routed+shared silu+down
            prof!("exp_silu_down", {
                ops::moe_expert_silu_down_shared(
                    ctx.gpu,
                    self.moe_expert_silu_down_shared,
                    expert_gate_out,
                    expert_up_out,
                    self.down_ptrs.packed_ptrs,
                    self.down_ptrs.scale_ptrs,
                    self.down_ptrs.scale2_vals,
                    expert_down_out,
                    indices_dev,
                    shared_gate_scratch,
                    shared_up_scratch,
                    &self.weights.shared_expert.down_proj,
                    shared_out,
                    h,
                    inter,
                    top_k,
                    stream,
                )
            })?;
        }

        // SOLID Incr-4 decode expert down-fold: land the routed-expert down_proj
        // LoRA delta into `expert_down_out` (slot-major [top_k, hidden]) IN PLACE,
        // recomputing `x = silu(gate)*up` from the still-materialized
        // `expert_gate_out`/`expert_up_out`. Must run BEFORE `moe_weighted_sum_blend`
        // (so the router weight scales base+delta) AND before the EP zero-temp
        // memset below (which reuses `expert_gate_out` as scratch). NULL
        // row_adapter: a genuine single-token decode is one homogeneous request —
        // `moe_route_gate` (Fold/Skip/Refuse) is the per-request opt-out. No-op
        // when no MoE LoRA / no expert adapter is installed (base byte-identical).
        if single_seq_decode {
            self.apply_expert_lora_decode_down(
                expert_gate_out,
                expert_up_out,
                expert_down_out,
                indices_dev,
                top_k,
                top_k,
                DevicePtr::NULL,
                ctx,
                stream,
            )?;
        }

        if self.has_mixed_bf16_shared_expert() {
            self.run_bf16_shared_expert(
                input,
                1,
                h,
                shared_inter,
                shared_gate_scratch,
                shared_up_scratch,
                shared_out,
                ctx,
                stream,
            )?;
        }

        if tracing::enabled!(tracing::Level::DEBUG) && !ctx.graph_capture {
            ctx.gpu.synchronize(stream)?;
            // Dump down outputs for expert slot 0
            let mut down_buf = vec![0u8; 16];
            ctx.gpu.copy_d2h(expert_down_out, &mut down_buf)?;
            let down_vals: Vec<f32> = (0..8)
                .map(|i| {
                    let bits = u16::from_le_bytes([down_buf[i * 2], down_buf[i * 2 + 1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect();
            tracing::info!("  MoE down_out[slot0,0..8]: {:?}", down_vals);
            // Shared out
            let mut sh_buf = vec![0u8; 16];
            ctx.gpu.copy_d2h(shared_out, &mut sh_buf)?;
            let sh_vals: Vec<f32> = (0..8)
                .map(|i| {
                    let bits = u16::from_le_bytes([sh_buf[i * 2], sh_buf[i * 2 + 1]]);
                    f32::from_bits((bits as u32) << 16)
                })
                .collect();
            tracing::info!("  MoE shared_out[0..8]: {:?}", sh_vals);
        }

        // Fused wsum+blend+gate: routed expert weighted sum + sigmoid(gate)*shared
        // Gate scalar GEMV is computed inline by each block (redundant but negligible).
        //
        // EP fix: for EP>1, the shared expert is computed identically on all ranks.
        // If we include it in the output before all-reduce, it gets summed world_size
        // times. Solution: pass NULL shared_out for EP, all-reduce the routed sum,
        // then add shared_out once after all-reduce.
        let output = ctx.buffers.moe_output();
        let is_ep = ctx.comm.is_some() && ctx.config.ep_world_size > 1;
        let shared_for_blend = if is_ep && !shared_out.is_null() {
            // EP: exclude shared expert from blend (will add after all-reduce).
            // Zero a temp buffer to pass as shared_out (kernel reads it even with NULL gate).
            let zero_buf = ctx.buffers.expert_gate_out(); // temp buffer, will be zeroed
            ctx.gpu.memset_async(zero_buf, 0, h as usize * 2, stream)?;
            zero_buf
        } else {
            shared_out
        };
        prof!("wsum_blend", {
            ops::moe_weighted_sum_blend(
                ctx.gpu,
                self.moe_weighted_sum_blend,
                output,
                expert_down_out,
                weights_dev,
                shared_for_blend,
                input,
                self.weights.shared_expert_gate.weight,
                h,
                top_k,
                h,
                stream,
            )
        })?;

        // EP all-reduce: sum partial expert outputs across ranks.
        // Each rank only computed its local experts (remote → zero), so
        // SUM gives the correct global result.
        if let Some(comm) = ctx.comm
            && ctx.config.ep_world_size > 1
        {
            if ctx.graph_capture {
                comm.all_reduce(output.0, h as usize * 2)?;
            } else {
                comm.all_reduce_async(output.0, h as usize * 2, stream)?;
            }
            // Now add shared expert contribution ONCE (after all-reduce).
            // Must apply the sigmoid gate: output += sigmoid(dot(input, gate_w)) * shared_out.
            // Using moe_batched_blend with num_tokens=1 computes the gate and blends correctly.
            // BUG #41 fix: previous code used residual_add (ignoring the gate), producing
            // wrong output that compounded across 48 layers into gibberish.
            if !shared_out.is_null() {
                if self.weights.shared_expert_gate.weight.0 == 0 {
                    // No gate weight (e.g., Mistral): shared expert always at full strength.
                    ops::residual_add(ctx.gpu, self.residual_add, output, shared_out, h, stream)?;
                } else {
                    // Gated shared expert (e.g., Qwen3.5): apply sigmoid gate.
                    ops::moe_batched_blend(
                        ctx.gpu,
                        self.moe_batched_blend,
                        output,
                        shared_out,
                        input,
                        self.weights.shared_expert_gate.weight,
                        h,
                        1,
                        stream,
                    )?;
                }
            }
        }

        if tracing::enabled!(tracing::Level::DEBUG) && !ctx.graph_capture {
            ctx.gpu.synchronize(stream)?;
            let mut buf = vec![0u8; 8];
            ctx.gpu.copy_d2h(output, &mut buf)?;
            let vals: Vec<f32> = (0..4)
                .map(|i| {
                    let lo = buf[i * 2];
                    let hi = buf[i * 2 + 1];
                    f32::from_bits(((lo as u32) | ((hi as u32) << 8)) << 16)
                })
                .collect();
            tracing::info!("  MoE output: {:?}", vals);
        }

        Ok(output)
    }
}
