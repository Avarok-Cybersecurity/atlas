// SPDX-License-Identifier: AGPL-3.0-only

//! The GDN block body — steps 2-10 of the SSM prefill.
//!
//! Split out of `trait_prefill.rs` because TWO entry paths need it and
//! neither is a wrapper around the other.
//!
//! `prefill_inner` fuses its residual bookkeeping into its norms:
//!
//! ```text
//! rms_norm_residual(hidden, input_norm) -> normed, residual      # step 1
//!   ... THIS FILE ...                   -> out_proj_buf          # steps 2-10
//! residual_add_rms_norm(hidden, out_proj_buf, post_attn_norm)    # step 11
//! ffn.forward_prefill(norm_output)                               # step 12
//! residual_add(hidden, moe_output)                               # step 13
//! ```
//!
//! Under an mHC highway (Qwen3.8-Flash-Next) the highway IS the residual and
//! the block output must reach it through `hc_post`, so steps 1, 11 and 13
//! double-count. `prefill_inner_hc` therefore replaces them wholesale rather
//! than wrapping them.
//!
//! What makes the split cheap is that the block itself is residual-free: it
//! reads `normed`, writes `out_proj_buf`, and touches neither `hidden` nor
//! `residual`. This file is that body moved VERBATIM — no logic changed, so
//! a GDN regression on any existing model would be a transcription error and
//! nothing else.

// Same glob the sibling prefill files use — this body was moved verbatim out
// of `trait_prefill.rs` and resolves the same names it always did.
use super::*;

impl Qwen3SsmLayer {
    /// Steps 2-10: QKVZ projection, conv1d, gates, the delta-rule recurrence,
    /// the gated norm, and `out_proj`. Returns the buffer holding
    /// `out_proj`'s output.
    ///
    /// `ssm_layer_idx` is passed in rather than re-fetched: it comes from a
    /// global call counter that must be bumped exactly once per layer per
    /// prefill, and both entry paths bump it before calling here.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn prefill_block(
        &self,
        normed: DevicePtr,
        num_tokens: usize,
        state: &mut dyn LayerState,
        ssm_layer_idx: usize,
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<DevicePtr> {
        let h = ctx.config.hidden_size;
        let eps = ctx.config.rms_norm_eps as f32;
        let k = num_tokens as u32;
        let bf16 = 2usize;
        #[allow(unused_variables)]
        let fp32 = 4usize;

        let ssm_state = state
            .as_any_mut()
            .downcast_mut::<SsmLayerState>()
            .ok_or_else(|| anyhow::anyhow!("Expected SsmLayerState"))?;

        let nk = ctx.config.linear_num_key_heads;
        let kd = ctx.config.linear_key_head_dim;
        let nv = ctx.config.linear_num_value_heads;
        let vd = ctx.config.linear_value_head_dim;
        let vpg = nv / nk;
        let key_dim = nk * kd;
        let value_dim = nv * vd;
        #[allow(unused_variables)]
        let conv_dim = key_dim * 2 + value_dim;
        let d_conv = ctx.config.linear_conv_kernel_dim;
        let qkvz_size = ctx.config.ssm_qkvz_size();

        macro_rules! prof {
            ($label:expr, $t0:expr) => {
                if ctx.profile {
                    if let Some(t0) = $t0 {
                        ctx.gpu.synchronize(stream)?;
                        let elapsed = t0.elapsed().as_micros();
                        tracing::info!("  SSM prefill [{}] N={}: {}\u{b5}s", $label, k, elapsed);
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

        // ── 2+3. QKVZ GEMM (+ deinterleave if needed) ──
        // Dispatch hoisted to trait_prefill_proj.rs to keep this file under
        // the 500 LoC cap; behavior identical.
        let deinterleaved = ctx.buffers.ssm_deinterleaved();
        self.prefill_qkvz_proj(
            normed,
            deinterleaved,
            k,
            qkvz_size,
            h,
            nk,
            kd,
            vpg,
            vd,
            ctx,
            stream,
        )?;
        // ATLAS_GDN_DUMP hook #0c: post-qkvz GEMM (deinterleaved input
        // to conv1d). qkvz_size = key_dim*2 + value_dim*2 = 12288 for A3B
        // (Q+K+V+Z, head-major within each segment). Compare against HF's
        // in_proj_qkv output (only 8192 — Q+K+V; HF has separate in_proj_z).
        super::debug::maybe_dump_gdn_buf(
            ctx.gpu,
            deinterleaved,
            (num_tokens - 1) * qkvz_size * bf16,
            qkvz_size,
            ssm_layer_idx,
            "post_qkvz",
            &super::debug::DUMP_GDN,
            stream,
        )?;

        // Bisect taps. `L00_block_out` diverges at cosine 0.845 while hc_pre's
        // outputs are bit-exact, so the fault is inside this block. These
        // split it: the projection is a pure GEMM (cheap to reproduce), the
        // pre-out_proj buffer is everything after it.
        crate::layers::ple::dump::tap_bf16(
            ctx.gpu,
            deinterleaved,
            ssm_layer_idx,
            "qkvz_preconv",
            num_tokens * qkvz_size,
            stream,
        );

        prof!("qkvz_gemm", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ── 4+5. Fused BA GEMM + GDN gates (token-parallel) ──
        // Replaces dense_gemm([M,K]×[N,K]) + compute_gdn_gates.
        // Vectorized uint4 loads, warp shuffle reduction, inline sigmoid/exp.
        // gate_out layout: [gate(nv), beta(nv)] per token, gate_stride = 2*nv FP32.
        let ba_size = ctx.config.ssm_ba_size(); // 64
        let gates_buf = ctx.buffers.ssm_gates();
        let gate_stride = nv * 2; // FP32 elements per token
        ops::dense_gemm_ba_gates_prefill(
            ctx.gpu,
            self.ba_gates_prefill_k,
            normed,
            &self.ssm.in_proj_ba,
            self.ssm.a_log.weight,
            self.ssm.dt_bias.weight,
            gates_buf,
            k,
            ba_size as u32,
            h as u32,
            h as u32,
            gate_stride as u32,
            nv as u32,
            vpg as u32,
            stream,
        )?;
        // Bisect tap: the gates as the recurrence will read them,
        // [g(nv), beta(nv)] FP32 per token. Everything upstream of the
        // recurrence except these is already verified, so this is the last
        // split: gates wrong => the BA GEMM/transforms; gates right => the
        // recurrence kernel itself.
        crate::layers::ple::dump::tap_f32(
            ctx.gpu,
            gates_buf,
            ssm_layer_idx,
            "gates",
            num_tokens * gate_stride,
            stream,
        );
        prof!("ba+gates", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ── 6. Batched conv1d for all N tokens (sequential per-channel in registers) ──
        // Reuse ssm_qkvz buffer for conv output (safe: deinterleave is done)
        let conv_out_buf = ctx.buffers.ssm_qkvz();
        let gdn_out_buf = ctx.buffers.attn_output();

        // Input: deinterleaved [N, qkvz_size], output: conv_out [N, conv_dim]
        // Conv1d processes QKV channels (first conv_dim of each token's qkvz_size)
        // MID-CHUNK tail capture: reserve THIS SSM layer's per-pass ordinal
        // once (shared by the conv + recurrence splits below). `None` unless
        // ATLAS_SSM_TAIL_MIDCHUNK is active for a pass that spans `tb`.
        let midcap_idx = ctx.midchunk_capture.as_ref().map(|c| {
            c.ssm_layer_counter
                .fetch_add(1, std::sync::atomic::Ordering::Relaxed)
        });
        // Conv1d — optionally split at cap_local, capturing conv_state @ tb.
        self.conv1d_prefill_capture(
            ctx,
            ssm_state.conv_state,
            deinterleaved,
            conv_out_buf,
            conv_dim,
            d_conv,
            k,
            qkvz_size,
            midcap_idx,
            stream,
        )?;
        // Bisect tap. The projection matches (cos 0.999998) and
        // `pre_out_proj` does not (cos 0.801), so the fault is conv / gates /
        // recurrence / gated-norm. The gates are COMPUTED above but not
        // CONSUMED until the recurrence, so this tap isolates the conv alone.
        crate::layers::ple::dump::tap_bf16(
            ctx.gpu,
            conv_out_buf,
            ssm_layer_idx,
            "post_conv",
            num_tokens * conv_dim,
            stream,
        );

        // ATLAS_GDN_DUMP hook #1: post-conv1d (post-silu, applied inside
        // the kernel). Last-token slice, flat [conv_dim] bf16. Layer
        // index from SSM_LAYER_CALL_COUNTER; latched by per-layer
        // AtomicBool so each (layer_idx, stage) dumps at most once.
        super::debug::maybe_dump_gdn_buf(
            ctx.gpu,
            conv_out_buf,
            (num_tokens - 1) * conv_dim * bf16,
            conv_dim,
            ssm_layer_idx,
            "conv",
            &super::debug::DUMP_CONV,
            stream,
        )?;
        prof!("conv1d", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ── 7. Batched L2 norm on Q,K for all N tokens ──
        // Q,K are the first 2*key_dim elements of each token's conv_out.
        // Stride between tokens in conv_out = conv_dim.
        ops::l2_norm(
            ctx.gpu,
            self.l2_norm_k,
            conv_out_buf,
            (nk * 2) as u32,
            kd as u32,
            1e-6,
            k,
            conv_dim as u32,
            stream,
        )?;
        // ATLAS_GDN_DUMP hook #2: post-L2 norm on q,k (v unchanged).
        // Same buffer/shape as the conv dump — l2_norm operates in
        // place on the q,k segments of conv_out_buf.
        super::debug::maybe_dump_gdn_buf(
            ctx.gpu,
            conv_out_buf,
            (num_tokens - 1) * conv_dim * bf16,
            conv_dim,
            ssm_layer_idx,
            "l2",
            &super::debug::DUMP_L2,
            stream,
        )?;
        prof!("l2_norm", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ── 8. GDN prefill via WY4-persistent kernel ──
        // Processes 4 tokens per iteration with WY algebraic correction, keeping
        // H state in shared memory for the entire sequence. 4× fewer sequential
        // state multiplications vs single-token kernel, preventing precision
        // drift at long context (28K+). Falls back to single-token persistent,
        // then split4 for unsupported configurations.
        let q_ptr = conv_out_buf;
        let k_ptr = conv_out_buf.offset(key_dim * bf16);
        let v_ptr = conv_out_buf.offset(key_dim * 2 * bf16);

        // Recurrence kernel dispatch hoisted to trait_prefill_recur.rs to
        // keep this file under the 500 LoC cap; behavior identical.
        self.prefill_gdn_recurrence_staged(
            ssm_state,
            q_ptr,
            k_ptr,
            v_ptr,
            gates_buf,
            gdn_out_buf,
            k,
            nk,
            nv,
            kd,
            vd,
            conv_dim,
            midcap_idx,
            ctx,
            stream,
        )?;

        // Bisect tap: the RAW recurrence output, before the gated norm.
        //
        // At token 0 the recurrent state is zero, so the raw output must be
        // beta * (q . k) * v — EXACTLY parallel to that head's v — in any
        // correct implementation. The post-norm taps show per-head cosines of
        // 0.63-0.98 against the reference, which parallel vectors cannot
        // produce (RMS-norm of parallel vectors matches to +-1). Either the
        // state is NOT zero at prefill start, or the recurrence reads
        // something it should not. This tap decides which stage lies.
        crate::layers::ple::dump::tap_bf16(
            ctx.gpu,
            gdn_out_buf,
            ssm_layer_idx,
            "raw_recur",
            num_tokens * value_dim,
            stream,
        );

        // ATLAS_GDN_DUMP hook #3: post-GDN recurrence (pre-gnorm,
        // value-space). gdn_out_buf is [num_tokens, value_dim] bf16
        // row-major; dump the last token's value_dim slice.
        super::debug::maybe_dump_gdn_buf(
            ctx.gpu,
            gdn_out_buf,
            (num_tokens - 1) * value_dim * bf16,
            value_dim,
            ssm_layer_idx,
            "gdn",
            &super::debug::DUMP_GDN,
            stream,
        )?;
        prof!("gdn_prefill", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        // ── 9. Gated RMS norm (batched: all tokens × heads in one launch) ──
        let normed_out_buf = conv_out_buf;
        let z_base = deinterleaved.offset((key_dim * 2 + value_dim) * bf16);
        ops::gated_rms_norm_prefill(
            ctx.gpu,
            self.gated_rms_norm_prefill_k,
            gdn_out_buf,
            z_base,
            &self.ssm.norm,
            normed_out_buf,
            nv as u32,
            vd as u32,
            eps,
            k,
            value_dim as u32,
            qkvz_size as u32,
            stream,
        )?;
        // ATLAS_GDN_DUMP hook #4: post-gated-RMSNorm. Downstream
        // `prefill_out_proj_dispatch` (line ~411) consumes this buffer
        // as `[num_tokens, value_dim]`, so the row stride is value_dim
        // (= nv*vd = 4096 for A3B). normed_out_buf aliases conv_out_buf
        // (in-place reuse — conv_out is dead by this point).
        super::debug::maybe_dump_gdn_buf(
            ctx.gpu,
            normed_out_buf,
            (num_tokens - 1) * value_dim * bf16,
            value_dim,
            ssm_layer_idx,
            "gnorm",
            &super::debug::DUMP_GNORM,
            stream,
        )?;
        prof!("gated_rms_norm", t0);
        t0 = if ctx.profile {
            ctx.gpu.synchronize(stream)?;
            Some(std::time::Instant::now())
        } else {
            None
        };

        crate::layers::ple::dump::tap_bf16(
            ctx.gpu,
            normed_out_buf,
            ssm_layer_idx,
            "pre_out_proj",
            num_tokens * value_dim,
            stream,
        );

        // ── 10. Output projection GEMM: [N, 4096] × [4096, 2048] → [N, 2048] ──
        let out_proj_buf = ctx.buffers.moe_output();
        self.prefill_out_proj_dispatch(ctx, normed_out_buf, out_proj_buf, k, h, value_dim, stream)?;
        // GDN HeadParallel: reduce the row-parallel partial out_proj across TP
        // ranks (num_tokens × h BF16) before the residual add. No-op at tp=1.
        self.ssm_tp_all_reduce(out_proj_buf, normed_out_buf, num_tokens, ctx, stream)?;
        // ATLAS_GDN_DUMP hook: SSM out_proj output — drift attribution.
        super::debug::maybe_dump_gdn_buf(
            ctx.gpu,
            out_proj_buf,
            (num_tokens - 1) * h * bf16,
            h,
            ssm_layer_idx,
            "out_proj",
            &super::debug::DUMP_GDN,
            stream,
        )?;

        prof!("out_proj", t0);
        Ok(out_proj_buf)
    }
}
