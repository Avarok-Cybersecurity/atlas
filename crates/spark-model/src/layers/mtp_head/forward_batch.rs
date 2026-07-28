// SPDX-License-Identifier: AGPL-3.0-only

//! Batched cross-sequence MTP propose: one drafter forward for n sequences.
//!
//! The measured C=4 gate shortfall (batched-MTP fixer round 1): with the
//! verify forward batched, the per-sequence PROPOSE loop dominated the
//! remaining step — 12 drafter `forward_one` calls x ~5 ms (the BF16 drafter
//! reads ~850 MB of weights per forward) = ~62 ms of a ~180 ms step at C=4.
//! Drafts chain autoregressively WITHIN a sequence but are independent
//! ACROSS sequences, so each draft position batches n sequences into M=n-row
//! GEMMs that read every weight ONCE.
//!
//! Microbenchmarked at M=4 on the real 27B drafter shapes (GB10):
//! `dense_gemm_bf16_pipelined` 5.1 ms vs the 4x `dense_gemv_bf16` loop
//! 14.4 ms per draft position (scalar `dense_gemm_bf16`: 8.1 ms). The small
//! K/V projections (N = nkv*hd) stay on the per-row GEMV (44 us vs 194 us
//! pipelined at N=1024 — the 128-wide tile under-fills the grid). The LM
//! head batches through the existing `w4a16_gemv_batch4` kernel.
//!
//! Everything non-weight-bearing (deinterleave, Q/K norms, RoPE, KV write,
//! paged attention, sigmoid gate) LOOPS per row with `forward_one`'s exact
//! kernels and arguments — per-sequence math identical, only base addresses
//! move. Scope mirrors the drafter-prefill v1 contract: BF16 head
//! (`--mtp-quantization bf16`) with BF16 drafter KV; anything else returns
//! "unsupported" and the caller falls back to the per-seq `propose` loop.
//!
//! Reachability: only from the scheduler's batched K=4 verify step
//! (`ATLAS_MTP_MAX_SEQS > 1`); C=1 and the default cap never enter.

use anyhow::{Result, ensure};
use spark_runtime::gpu::DevicePtr;

use super::{MtpHead, MtpProposerState, MtpQuantization, ProjectionWeight};
use crate::layer::ForwardContext;
use crate::layers::ops;
use crate::weight_map::DenseWeight;

/// Per-sequence drafter attention metadata stride within scratch. Layout per
/// sequence i at `scratch + META_BASE + i*META_STRIDE` mirrors `forward_one`:
/// [0..4) position u32 | [8..16) slot i64 | [16..20) seq_len i32 |
/// [256..) block table i32[]. 2048 bytes caps the block table at 448 entries
/// (>= 4096-token drafter at block size 16 needs 257).
const META_BASE: usize = 49152;
const META_STRIDE: usize = 2048;

impl MtpHead {
    /// Whether the batched cross-sequence propose can run for `n` sequences.
    /// BF16-everything scope + both batch kernels resolved (`try_kernel`
    /// returns handle 0 silently — gate on it, never assume).
    pub(crate) fn can_propose_batch(&self, n: usize) -> bool {
        let bf16_proj = |p: &ProjectionWeight| matches!(p, ProjectionWeight::Bf16(_));
        (2..=4).contains(&n)
            && matches!(self.quant, MtpQuantization::Bf16)
            && self.kv_bf16
            && bf16_proj(&self.fc)
            && bf16_proj(&self.q_proj)
            && bf16_proj(&self.k_proj)
            && bf16_proj(&self.v_proj)
            && bf16_proj(&self.o_proj)
            && self
                .dense_ffn_generic
                .as_ref()
                .is_some_and(|(g, u, d)| bf16_proj(g) && bf16_proj(u) && bf16_proj(d))
            && self.dense_gemm_pipelined_k.0 != 0
            && self.w4a16_gemv_batch4_k.0 != 0
            && self.dense_gemv_k.is_some()
            && self.deinterleave_qg_k.is_some()
            && self.moe_silu_mul_k.is_some()
    }

    /// M=n-row BF16 GEMM dispatch: pipelined tensor-core GEMM for the large
    /// weight-bearing shapes (reads B once for all rows), per-row GEMV for
    /// small N where the 128-wide tile under-fills the grid (measured).
    fn gemm_rows(
        &self,
        gpu: &dyn spark_runtime::gpu::GpuBackend,
        input: DevicePtr,
        w: &DenseWeight,
        output: DevicePtr,
        m: usize,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        if n >= 4096 && (k & 7) == 0 {
            ops::dense_gemm_bf16_pipelined(
                gpu,
                self.dense_gemm_pipelined_k,
                input,
                w,
                output,
                m as u32,
                n,
                k,
                stream,
            )
        } else {
            let gemv_k = self.dense_gemv_k.unwrap();
            for r in 0..m {
                ops::dense_gemv(
                    gpu,
                    gemv_k,
                    input.offset(r * k as usize * 2),
                    w,
                    output.offset(r * n as usize * 2),
                    n,
                    k,
                    stream,
                )?;
            }
            Ok(())
        }
    }

    /// One draft position for n sequences: the M=n-row sibling of
    /// `forward_one`. Row i consumes `(tokens[i], hiddens[i])` at sequence
    /// position `positions[i]`, appends one drafter KV row per sequence, and
    /// returns the n argmax draft ids via ONE small D2H (the per-seq path
    /// pays one 4-byte sync D2H per sequence per draft).
    ///
    /// Buffer layout (row i at offset i*width, matching `forward_one`'s
    /// buffer choices): embeds ssm_qkvz [n,h] | normed_embed
    /// ssm_deinterleaved [n,h] | normed_hidden ssm_gates [n,h] | concat
    /// ssm_ba [n,2h] | hidden hidden_states [n,h] | residual residual [n,h]
    /// | q/k/v qkv_output [n,qg]+[n,kv]+[n,kv] | attn attn_output [n,q_dim]
    /// | ffn expert_gate/up_out [n,inter] -> moe_output [n,h] | logits
    /// [n,vocab].
    #[allow(clippy::too_many_arguments)]
    fn forward_batch_position(
        &self,
        tokens: &[u32],
        hiddens: &[DevicePtr],
        positions: &[usize],
        states: &mut [&mut MtpProposerState],
        ctx: &ForwardContext,
        stream: u64,
        out_ids: &mut [u32],
    ) -> Result<()> {
        let n = tokens.len();
        let h = ctx.config.hidden_size;
        let nq = ctx.config.num_attention_heads as u32;
        let nkv = ctx.config.num_key_value_heads as u32;
        let hd = ctx.config.head_dim as u32;
        let eps = ctx.config.rms_norm_eps as f32;
        let bf16 = 2usize;
        let gpu = ctx.gpu;

        let q_dim = (nq * hd) as usize;
        let qg_dim = q_dim * 2;
        let kv_dim = (nkv * hd) as usize;

        // 1. Embed the n tokens (row i at embeds + i*h).
        let embeds = ctx.buffers.ssm_qkvz();
        for (i, &t) in tokens.iter().enumerate() {
            let src = self.embed_tokens.weight.offset(t as usize * h * bf16);
            gpu.copy_d2d_async(src, embeds.offset(i * h * bf16), h * bf16, stream)?;
        }

        // 2. Pre-fc norms. Embeds are contiguous (one n-row call); target
        // hiddens may be non-contiguous across sequences (stash rows vs the
        // drafter's own hidden rows) -> per-row calls into a compact buffer.
        let normed_embed = ctx.buffers.ssm_deinterleaved();
        ops::rms_norm(
            gpu,
            self.rms_norm_k,
            embeds,
            &self.pre_fc_norm_embedding,
            normed_embed,
            n as u32,
            h as u32,
            eps,
            stream,
        )?;
        let normed_hidden = ctx.buffers.ssm_gates();
        for (i, &hp) in hiddens.iter().enumerate() {
            ops::rms_norm(
                gpu,
                self.rms_norm_k,
                hp,
                &self.pre_fc_norm_hidden,
                normed_hidden.offset(i * h * bf16),
                1,
                h as u32,
                eps,
                stream,
            )?;
        }

        // 3. Per-row concat [normed_embed_i | normed_hidden_i] -> [n, 2h].
        let concat = ctx.buffers.ssm_ba();
        for i in 0..n {
            ops::bf16_concat(
                gpu,
                self.bf16_concat_k,
                normed_embed.offset(i * h * bf16),
                normed_hidden.offset(i * h * bf16),
                concat.offset(i * 2 * h * bf16),
                h as u32,
                stream,
            )?;
        }

        // 4. fc: [n, 2h] -> [n, h]; copy to residual stream.
        let hidden = ctx.buffers.hidden_states();
        let (fc_w, q_w, k_w, v_w, o_w) = match (
            &self.fc,
            &self.q_proj,
            &self.k_proj,
            &self.v_proj,
            &self.o_proj,
        ) {
            (
                ProjectionWeight::Bf16(fc),
                ProjectionWeight::Bf16(q),
                ProjectionWeight::Bf16(k),
                ProjectionWeight::Bf16(v),
                ProjectionWeight::Bf16(o),
            ) => (fc, q, k, v, o),
            _ => anyhow::bail!("propose_batch: non-BF16 projections (can_propose_batch lied)"),
        };
        self.gemm_rows(
            gpu,
            concat,
            fc_w,
            hidden,
            n,
            h as u32,
            (2 * h) as u32,
            stream,
        )?;
        let residual = ctx.buffers.residual();
        gpu.copy_d2d_async(hidden, residual, n * h * bf16, stream)?;

        // 5. Input layernorm [n, h].
        let normed = ctx.buffers.norm_output();
        ops::rms_norm(
            gpu,
            self.rms_norm_k,
            hidden,
            &self.input_layernorm,
            normed,
            n as u32,
            h as u32,
            eps,
            stream,
        )?;

        // 6. Q+Gate / K / V projections. Row layout: q [n, qg] at qkv_output,
        // k [n, kv] after it, v [n, kv] after that (fits: n*(qg+2kv) =
        // n*qkv_dim rows of the arena).
        let q_out = ctx.buffers.qkv_output();
        let k_out = q_out.offset(n * qg_dim * bf16);
        let v_out = k_out.offset(n * kv_dim * bf16);
        self.gemm_rows(gpu, normed, q_w, q_out, n, qg_dim as u32, h as u32, stream)?;
        self.gemm_rows(gpu, normed, k_w, k_out, n, kv_dim as u32, h as u32, stream)?;
        self.gemm_rows(gpu, normed, v_w, v_out, n, kv_dim as u32, h as u32, stream)?;

        // Per-row Q/Gate deinterleave + Q/K norms (forward_one's exact calls).
        let deint_k = self.deinterleave_qg_k.unwrap();
        for i in 0..n {
            ops::deinterleave_qg(
                gpu,
                deint_k,
                q_out.offset(i * qg_dim * bf16),
                1,
                nq,
                hd,
                nq * hd * 2,
                stream,
            )?;
        }
        // Q norm MUST loop per sequence: each sequence's row is a strided
        // [q(q_dim) | gate(q_dim)] block at stride qg_dim, so a single packed
        // n*nq-row launch would (a) RMS-norm seq0's sigmoid GATE with q_norm
        // weights and (b) leave the later sequences' Q heads un-normalized —
        // the measured batched accept-p1 divergence (0.59-0.71 vs single-seq
        // 0.70-0.82). nq rows over exactly the q_dim span per seq is
        // forward_one's exact call. K below is packed [n, kv_dim] and safe.
        for i in 0..n {
            let q_row = q_out.offset(i * qg_dim * bf16);
            ops::rms_norm(
                gpu,
                self.rms_norm_k,
                q_row,
                &self.q_norm,
                q_row,
                nq,
                hd,
                eps,
                stream,
            )?;
        }
        ops::rms_norm(
            gpu,
            self.rms_norm_k,
            k_out,
            &self.k_norm,
            k_out,
            n as u32 * nkv,
            hd,
            eps,
            stream,
        )?;

        // 7. Per-sequence attention metadata + RoPE + KV write + paged attn.
        let mut kv_cache = self.kv_cache.lock();
        let bs = kv_cache.block_size();
        let scratch = ctx.buffers.scratch();
        let attn_out = ctx.buffers.attn_output();
        let inv_sqrt_d = 1.0f32 / (hd as f32).sqrt();
        let kv_stride = nkv * hd;
        for i in 0..n {
            let state = &mut *states[i];
            let blocks_needed = (state.seq_len / bs) + 1;
            while state.block_table.len() < blocks_needed {
                state.block_table.push(kv_cache.alloc_block()?);
            }
            let bt_len = state.block_table.len() * 4;
            ensure!(
                256 + bt_len <= META_STRIDE,
                "propose_batch: drafter block table {} exceeds meta stride",
                bt_len
            );
            let meta_base = scratch.offset(META_BASE + i * META_STRIDE);
            let block_idx = state.block_table[state.seq_len / bs];
            let global_slot = (block_idx as i64) * (bs as i64) + ((state.seq_len % bs) as i64);
            let mut meta_buf = vec![0u8; 256 + bt_len];
            meta_buf[0..4].copy_from_slice(&(positions[i] as u32).to_le_bytes());
            meta_buf[8..16].copy_from_slice(&global_slot.to_le_bytes());
            meta_buf[16..20].copy_from_slice(&((state.seq_len + 1) as i32).to_le_bytes());
            let bt_bytes: &[u8] = unsafe {
                std::slice::from_raw_parts(state.block_table.as_ptr() as *const u8, bt_len)
            };
            meta_buf[256..256 + bt_len].copy_from_slice(bt_bytes);
            gpu.copy_h2d_async(&meta_buf, meta_base, stream)?;

            let q_row = q_out.offset(i * qg_dim * bf16);
            let k_row = k_out.offset(i * kv_dim * bf16);
            let v_row = v_out.offset(i * kv_dim * bf16);
            ops::rope(
                gpu,
                self.rope_k,
                q_row,
                k_row,
                meta_base,
                1,
                nq,
                nkv,
                hd,
                ctx.config.rotary_dim() as u32,
                ctx.config.rope_theta as f32,
                stream,
            )?;
            ops::reshape_and_cache(
                gpu,
                self.reshape_cache_k,
                k_row,
                v_row,
                kv_cache.k_pool_ptr(self.attn_layer_idx),
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                meta_base.offset(8),
                1,
                nkv,
                hd,
                bs as u32,
                kv_stride,
                kv_stride,
                kv_cache.cache_stride() as u64,
                stream,
            )?;
            ops::paged_decode_attn_bf16(
                gpu,
                self.paged_decode_k,
                q_row,
                kv_cache.k_pool_ptr(self.attn_layer_idx),
                kv_cache.v_pool_ptr(self.attn_layer_idx),
                attn_out.offset(i * q_dim * bf16),
                meta_base.offset(256),
                meta_base.offset(16),
                state.block_table.len() as u32,
                1,
                nq,
                nkv,
                hd,
                bs as u32,
                inv_sqrt_d,
                nq * hd,
                0,
                stream,
            )?;
            // Sigmoid gate: attn_i *= sigmoid(gate_i).
            ops::sigmoid_gate_mul(
                gpu,
                self.sigmoid_gate_mul_k,
                attn_out.offset(i * q_dim * bf16),
                q_row.offset(q_dim * bf16),
                attn_out.offset(i * q_dim * bf16),
                nq * hd,
                stream,
            )?;
        }
        drop(kv_cache);

        // 8. O projection [n, q_dim] -> [n, h]; residual + post-attn norm.
        let o_out = ctx.buffers.norm_output();
        self.gemm_rows(gpu, attn_out, o_w, o_out, n, h as u32, q_dim as u32, stream)?;
        let normed2 = ctx.buffers.norm_output();
        ops::residual_add_rms_norm(
            gpu,
            self.residual_add_rms_norm_k,
            hidden,
            o_out,
            &self.post_attn_layernorm,
            normed2,
            residual,
            n as u32,
            h as u32,
            eps,
            stream,
        )?;

        // 9. Dense FFN [n, h] -> [n, h] (dense_ffn_forward_generic, M=n).
        let inter = if ctx.config.intermediate_size > 0 {
            ctx.config.intermediate_size as u32
        } else {
            ctx.config.moe_intermediate_size as u32
        };
        let (gate_w, up_w, down_w) = match self.dense_ffn_generic.as_ref() {
            Some((
                ProjectionWeight::Bf16(g),
                ProjectionWeight::Bf16(u),
                ProjectionWeight::Bf16(d),
            )) => (g, u, d),
            _ => anyhow::bail!("propose_batch: non-BF16 FFN (can_propose_batch lied)"),
        };
        let gate_out = ctx.buffers.expert_gate_out();
        let up_out = ctx.buffers.expert_up_out();
        self.gemm_rows(gpu, normed2, gate_w, gate_out, n, inter, h as u32, stream)?;
        self.gemm_rows(gpu, normed2, up_w, up_out, n, inter, h as u32, stream)?;
        ops::moe_silu_mul(
            gpu,
            self.moe_silu_mul_k.unwrap(),
            gate_out,
            up_out,
            gate_out,
            n as u32 * inter,
            stream,
        )?;
        let ffn_out = ctx.buffers.moe_output();
        self.gemm_rows(gpu, gate_out, down_w, ffn_out, n, h as u32, inter, stream)?;
        ops::residual_add(
            gpu,
            self.residual_add_k,
            hidden,
            ffn_out,
            n as u32 * h as u32,
            stream,
        )?;

        // 10. Final norm [n, h] + batched LM head + per-row argmax.
        let final_normed = ctx.buffers.norm_output();
        ops::rms_norm(
            gpu,
            self.rms_norm_k,
            hidden,
            &self.norm,
            final_normed,
            n as u32,
            h as u32,
            eps,
            stream,
        )?;
        let v = if self.mtp_vocab_size > 0 {
            self.mtp_vocab_size.min(ctx.config.vocab_size as u32)
        } else {
            ctx.config.vocab_size as u32
        };
        let logits = ctx.buffers.logits();
        ops::w4a16_gemv_batchm(
            gpu,
            self.w4a16_gemv_batch4_k,
            final_normed,
            &self.lm_head_nvfp4,
            logits,
            n as u32,
            v,
            h as u32,
            stream,
        )?;
        if self.argmax_batch_k.0 != 0 {
            ops::argmax_bf16_batch(gpu, self.argmax_batch_k, logits, scratch, v, n as u32, v, stream)?;
        } else {
            for i in 0..n {
                ops::argmax_bf16(
                    gpu,
                    self.argmax_k,
                    logits.offset(i * v as usize * bf16),
                    scratch.offset(i * 4),
                    v,
                    stream,
                )?;
            }
        }

        // 11. One sync D2H for the n ids (the per-seq path pays n of these).
        let mut buf = vec![0u8; n * 4];
        gpu.copy_d2h(scratch, &mut buf)?;
        for (i, id) in out_ids.iter_mut().enumerate() {
            *id = u32::from_le_bytes([buf[i * 4], buf[i * 4 + 1], buf[i * 4 + 2], buf[i * 4 + 3]]);
        }

        // 12. Bookkeeping (forward_one's tail, per row).
        for (i, state) in states.iter_mut().enumerate() {
            state.seq_len += 1;
            state.last_pair_key = Some(positions[i].saturating_sub(1));
        }
        Ok(())
    }

    /// Batched propose driver: `num_drafts` chained positions, each one
    /// M=n-row forward. Draft 0 consumes the caller's per-sequence target
    /// hiddens; draft j>0 consumes the drafter's own hidden row i (written
    /// by position j-1's fc — read [step 2] strictly before the next fc
    /// overwrites it [step 4], same stream). Sets `last_num_drafted`
    /// incrementally so a mid-chain error trims exactly the rows written.
    pub(crate) fn propose_batch_impl(
        &self,
        last_tokens: &[u32],
        target_hiddens: &[DevicePtr],
        positions: &[usize],
        num_drafts: usize,
        states: &mut [&mut MtpProposerState],
        ctx: &ForwardContext,
        stream: u64,
    ) -> Result<Vec<Vec<u32>>> {
        let n = last_tokens.len();
        ensure!(
            n == target_hiddens.len() && n == positions.len() && n == states.len(),
            "propose_batch: length mismatch"
        );
        static LOGGED: std::sync::Once = std::sync::Once::new();
        LOGGED.call_once(|| {
            tracing::info!(
                "MTP propose_batch active: n={n} pipelined_gemm={:#x} lm_head_batch4={:#x}",
                self.dense_gemm_pipelined_k.0,
                self.w4a16_gemv_batch4_k.0
            );
        });
        // Parity with propose(): reset chain confidence (unused here — the
        // batched path is gated to draft_conf_tau() == 0).
        self.last_conf_bits
            .store(1.0f32.to_bits(), std::sync::atomic::Ordering::Relaxed);

        let h = ctx.config.hidden_size;
        let mut cur_tokens = last_tokens.to_vec();
        let mut cur_positions = positions.to_vec();
        let mut ids = vec![0u32; n];
        let mut all: Vec<Vec<u32>> = vec![Vec::with_capacity(num_drafts); n];
        for j in 0..num_drafts {
            let hiddens_j: Vec<DevicePtr> = if j == 0 {
                target_hiddens.to_vec()
            } else {
                (0..n)
                    .map(|i| ctx.buffers.hidden_states().offset(i * h * 2))
                    .collect()
            };
            self.forward_batch_position(
                &cur_tokens,
                &hiddens_j,
                &cur_positions,
                states,
                ctx,
                stream,
                &mut ids,
            )?;
            for i in 0..n {
                all[i].push(ids[i]);
                cur_positions[i] += 1;
            }
            cur_tokens.copy_from_slice(&ids);
            for state in states.iter_mut() {
                state.last_num_drafted = j + 1;
            }
        }
        Ok(all)
    }
}
