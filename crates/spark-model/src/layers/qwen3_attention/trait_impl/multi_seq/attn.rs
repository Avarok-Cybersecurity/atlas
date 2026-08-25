// SPDX-License-Identifier: AGPL-3.0-only

//! Phases 3-6: per-sequence RoPE, KV-cache write, batched paged
//! attention, gate multiply + O projection.

use anyhow::Result;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::kv_cache::PagedKvCache;

use super::ctx::MultiSeqCtx;
use crate::layer::AttnMetadataDev;
use crate::layers::ops;
use crate::layers::qwen3_attention::Qwen3AttentionLayer;

/// One batched `reshape_and_cache` launch for all N sequences instead of one
/// launch per sequence. Kill switch: `ATLAS_NO_ATTN_BATCH_CACHE_WRITE=1`.
fn batch_cache_write_enabled() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| {
        std::env::var("ATLAS_NO_ATTN_BATCH_CACHE_WRITE")
            .ok()
            .as_deref()
            != Some("1")
    })
}

impl Qwen3AttentionLayer {
    /// Phase 3: per-token RoPE (each sequence has its own position).
    pub(super) fn ms_phase_rope(&self, c: &MultiSeqCtx<'_>, meta: AttnMetadataDev) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            nq,
            nkv,
            hd,
            bf16,
            q_proj_bytes,
            per_seq_qkv,
            qkv_buf,
            ..
        } = *c;
        // ONE launch for all n sequences when the strided kernel is present. The
        // packed `rope` derives row addresses from num_*_heads*head_dim, but these
        // rows sit `per_seq_qkv` apart inside the interleaved [Q|K|V|gate] block,
        // so the per-sequence loop below was calling it n times with seq_len=1 —
        // 258 launches/step at 4.6 us = 1.18 ms across the 16 attention layers.
        // Bit-identical: same math and ordering, only the row address differs.
        // Kill switch: ATLAS_NO_ROPE_STRIDED=1.
        fn rope_strided_enabled() -> bool {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| std::env::var("ATLAS_NO_ROPE_STRIDED").ok().as_deref() != Some("1"))
        }
        // YARN-scaled layers (yarn_inv_freq set -- Laguna full-attention) must
        // take the per-seq loop below: rope_strided computes plain-theta rope
        // and has no yarn table argument, so a K-row verify through it rotates
        // Q/K differently than the serial decode path. Bit-parity for
        // non-yarn layers is unchanged.
        if n > 1
            && self.rope_strided_k.0 != 0
            && rope_strided_enabled()
            && self.yarn_inv_freq.is_null()
        {
            let stride_e = (per_seq_qkv / bf16) as u32;
            return ops::rope_strided(
                fwd.gpu,
                self.rope_strided_k,
                qkv_buf,
                qkv_buf.offset(q_proj_bytes),
                meta.positions,
                n as u32,
                nq,
                nkv,
                hd,
                self.rotary_dim_override
                    .unwrap_or(fwd.config.rotary_dim() as u32),
                self.rope_theta_override
                    .unwrap_or(fwd.config.rope_theta as f32),
                stride_e,
                stride_e,
                stream,
            );
        }
        for i in 0..n {
            let q_out_i = qkv_buf.offset(i * per_seq_qkv);
            let k_out_i = q_out_i.offset(q_proj_bytes);
            let pos_i = meta.positions.offset(i * 4); // u32 per position
            if self.yarn_inv_freq.is_null() {
                ops::rope(
                    fwd.gpu,
                    self.rope_k,
                    q_out_i,
                    k_out_i,
                    pos_i,
                    1,
                    nq,
                    nkv,
                    hd,
                    self.rotary_dim_override
                        .unwrap_or(fwd.config.rotary_dim() as u32),
                    self.rope_theta_override
                        .unwrap_or(fwd.config.rope_theta as f32),
                    stream,
                )?;
            } else {
                ops::rope_yarn_scaled(
                    fwd.gpu,
                    self.rope_yarn_scaled_k,
                    q_out_i,
                    k_out_i,
                    pos_i,
                    1,
                    nq,
                    nkv,
                    hd,
                    self.rotary_dim_override
                        .unwrap_or(fwd.config.rotary_dim() as u32),
                    self.yarn_inv_freq,
                    self.yarn_attention_factor,
                    stream,
                )?;
            }
        }
        Ok(())
    }

    /// Phase 4: per-token KV cache write.
    pub(super) fn ms_phase_cache_write(
        &self,
        c: &MultiSeqCtx<'_>,
        kv_cache: &mut PagedKvCache,
        meta: AttnMetadataDev,
    ) -> Result<()> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            nkv,
            hd,
            bs,
            bf16,
            q_proj_bytes,
            per_seq_qkv,
            qkv_buf,
            ..
        } = *c;
        let kv_stride = nkv * hd;
        // The reshape_and_cache kernels already take `num_tokens` plus explicit
        // key/value row strides ("row stride may differ", reshape_and_cache.cu),
        // and their grid is [num_tokens,1,1] — so all N sequences go in ONE
        // launch. `slot` and `positions` are already contiguous per-sequence
        // arrays. Each sequence's K row sits `per_seq_qkv` bytes after the last,
        // so the row stride is that gap in ELEMENTS.
        //
        // `ATLAS_NO_ATTN_BATCH_CACHE_WRITE=1` restores the per-sequence loop.
        let k_out_0 = qkv_buf.offset(q_proj_bytes);
        let v_out_0 = k_out_0.offset((nkv * hd) as usize * bf16);
        if n > 1 && batch_cache_write_enabled() && per_seq_qkv.is_multiple_of(bf16) {
            let row_stride = (per_seq_qkv / bf16) as u32;
            return self.write_kv_cache(
                fwd.gpu,
                k_out_0,
                v_out_0,
                kv_cache,
                meta.slot,
                n as u32,
                nkv,
                hd,
                bs,
                row_stride,
                row_stride,
                stream,
                fwd.graph_capture,
            );
        }
        for i in 0..n {
            let q_out_i = qkv_buf.offset(i * per_seq_qkv);
            let k_out_i = q_out_i.offset(q_proj_bytes);
            let v_out_i = k_out_i.offset((nkv * hd) as usize * bf16);
            let slot_i = meta.slot.offset(i * 8); // i64 per slot
            self.write_kv_cache(
                fwd.gpu,
                k_out_i,
                v_out_i,
                kv_cache,
                slot_i,
                1,
                nkv,
                hd,
                bs,
                kv_stride,
                kv_stride,
                stream,
                fwd.graph_capture,
            )?;
        }
        Ok(())
    }

    /// Phase 5: build contiguous Q buffer + run BATCHED paged decode.
    /// Returns the attn_out buffer pointer for downstream phases.
    pub(super) fn ms_phase_paged_decode(
        &self,
        c: &MultiSeqCtx<'_>,
        kv_cache: &mut PagedKvCache,
        meta: AttnMetadataDev,
    ) -> Result<DevicePtr> {
        let MultiSeqCtx {
            fwd,
            n,
            stream,
            nq,
            nkv,
            hd,
            bs,
            bf16,
            q_dim,
            per_seq_qkv,
            qkv_buf,
            ..
        } = *c;
        // TurboQuant WHT bookends (mirrors decode/attention_forward.rs).
        // The cache holds WHT(K)/WHT(V) for turbo dtypes: rotate the batched
        // Q rows before the paged decode and rotate the output back after —
        // without these the multi-seq batched decode scores raw Q against
        // rotated K and returns output in the rotated-V basis.
        // Hoisted above the Q staging below: those rotations mutate the staged
        // buffer IN PLACE, so they are exactly what makes the copy unskippable.
        let (wht_k_dtype, wht_v_dtype) = self.kv_dtype.kv_pair();
        let k_is_turbo = wht_k_dtype.is_wht_rotated();
        let v_is_turbo = wht_v_dtype.is_wht_rotated();

        // ── Q for the batched paged decode ────────────────────────────────
        // `run_paged_decode` already takes an explicit `q_stride` and the kernel
        // indexes `Q + seq_idx*q_stride` (paged_decode_attn.cu:96, splitk twin
        // :364), so when nothing rewrites Q we can point it straight at the
        // interleaved [Q|K|V|gate] block and read in place — the rows are simply
        // `per_seq_qkv` apart instead of packed.
        //
        // That removes 16 layers x 16 seqs = 256 D2D copies of 12288 B per step
        // (0.19 ms of GPU copy plus 0.23 ms of host issue measured by nsys), and
        // 256 nodes from the captured graph. Bit-identical: same values, same
        // kernel, only the addressing changes.
        //
        // NOT skippable under TurboQuant: the innerQ/WHT bookends below rotate
        // the staged buffer in place, and `qkv_buf` must not be mutated.
        // Kill switch: ATLAS_NO_ATTN_Q_INPLACE=1.
        fn q_inplace_enabled() -> bool {
            static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
            *ON.get_or_init(|| {
                std::env::var("ATLAS_NO_ATTN_Q_INPLACE").ok().as_deref() != Some("1")
            })
        }
        let q_inplace =
            !k_is_turbo && !v_is_turbo && q_inplace_enabled() && per_seq_qkv.is_multiple_of(bf16);
        let q_contiguous = if q_inplace {
            qkv_buf
        } else {
            let staged = fwd.buffers.ssm_qkvz();
            for i in 0..n {
                let q_out_i = qkv_buf.offset(i * per_seq_qkv);
                fwd.gpu.copy_d2d_async(
                    q_out_i,
                    staged.offset(i * q_dim as usize * bf16),
                    q_dim as usize * bf16,
                    stream,
                )?;
            }
            staged
        };
        let q_stride = if q_inplace {
            (per_seq_qkv / bf16) as u32
        } else {
            nq * hd
        };
        let attn_out = fwd.buffers.attn_output();
        let inv_sqrt_d = self.effective_attn_scale(hd);
        let weight_pre_rotated = std::env::var("TQ_PLUS_WEIGHT_ROTATION")
            .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
            .unwrap_or(false);
        let wht_runtime_active = !weight_pre_rotated && (hd == 128 || hd == 256 || hd == 512);
        if k_is_turbo && self.innerq_apply_q_k.0 != 0 && hd == 128 {
            use spark_runtime::kernel_args::KernelLaunch;
            KernelLaunch::new(fwd.gpu, self.innerq_apply_q_k)
                .grid([n as u32 * nq, 1, 1])
                .block([32, 1, 1])
                .arg_ptr(q_contiguous)
                .arg_u32(hd)
                .launch(stream)?;
        }
        if k_is_turbo && wht_runtime_active && self.wht_bf16_k.0 != 0 {
            use spark_runtime::kernel_args::KernelLaunch;
            KernelLaunch::new(fwd.gpu, self.wht_bf16_k)
                .grid([n as u32 * nq, 1, 1]) // one warp per (seq, q_head)
                .block([32, 1, 1])
                .arg_ptr(q_contiguous)
                .arg_u32(hd)
                .launch(stream)?;
        }
        self.run_paged_decode(
            fwd.gpu,
            q_contiguous,
            kv_cache,
            attn_out,
            meta.block_table,
            meta.seq_len,
            meta.max_blocks_per_seq,
            n as u32,
            nq,
            nkv,
            hd,
            bs,
            inv_sqrt_d,
            q_stride,
            fwd.buffers.splitk_workspace(),
            fwd.levers.max_decode_seqs,
            stream,
        )?;
        if v_is_turbo && wht_runtime_active && self.wht_bf16_k_inv.0 != 0 {
            use spark_runtime::kernel_args::KernelLaunch;
            KernelLaunch::new(fwd.gpu, self.wht_bf16_k_inv)
                .grid([n as u32 * nq, 1, 1])
                .block([32, 1, 1])
                .arg_ptr(attn_out)
                .arg_u32(hd)
                .launch(stream)?;
        }
        Ok(attn_out)
    }
}

mod o_proj;
