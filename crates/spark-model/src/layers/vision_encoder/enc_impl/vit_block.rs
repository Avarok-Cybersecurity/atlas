// SPDX-License-Identifier: AGPL-3.0-only

//! Single ViT block (norm → QKV → RoPE attention → proj → +residual →
//! norm → fc1 → GELU → fc2 → +residual).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::super::{ViTBlock, VisionEncoder};

impl VisionEncoder {
    /// ViT GEMM with bias: C[m,n] = A[m,k] @ B[n,k]^T + bias[n] (BF16).
    /// Prefers the tensor-core `dense_gemm_bf16_pipelined` (~40× the scalar
    /// `vision_gemm_bias` on the ViT's large-M shapes) + a fused bias-add; falls
    /// back to the scalar fused kernel if either handle is unavailable. The ViT
    /// GEMMs dominate image prefill (~5s/image on the scalar path).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn vit_gemm_bias(
        &self,
        gpu: &dyn GpuBackend,
        a: DevicePtr,
        b: DevicePtr,
        bias: DevicePtr,
        c: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        if self.k_gemm_pipelined.0 != 0 && self.k_add_bias.0 != 0 {
            KernelLaunch::new(gpu, self.k_gemm_pipelined)
                .grid([div_ceil(n, 128), div_ceil(m, 128), 1])
                .block([256, 1, 1])
                .arg_ptr(a)
                .arg_ptr(b)
                .arg_ptr(c)
                .arg_u32(m)
                .arg_u32(n)
                .arg_u32(k)
                .launch(stream)?;
            KernelLaunch::new(gpu, self.k_add_bias)
                .grid([div_ceil(m * n, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(c)
                .arg_ptr(bias)
                .arg_u32(m)
                .arg_u32(n)
                .launch(stream)
        } else {
            KernelLaunch::new(gpu, self.k_gemm)
                .grid([div_ceil(n, 32), div_ceil(m, 32), 1])
                .block([32, 32, 1])
                .arg_ptr(a)
                .arg_ptr(b)
                .arg_ptr(bias)
                .arg_ptr(c)
                .arg_u32(m)
                .arg_u32(n)
                .arg_u32(k)
                .launch(stream)
        }
    }

    /// GEMM-based SDPA for one image's [seq, 3*H*D] QKV slice → O[seq, H*D].
    /// Fast replacement for the warp-per-query `vision_attention_rope`. Once per
    /// call: rope + deinterleave + V-transpose (all heads). Then per head:
    ///   GEMM1 raw QKᵀ (f32 out) → row softmax (scale folded) → GEMM2 P·V →
    ///   scatter into the interleaved O head slot.
    /// `qkv/o/cos/sin` are per-image base pointers (caller already offset them).
    /// All launches share `stream`, so each head's GEMM1→softmax→GEMM2→scatter is
    /// ordered before head h+1 reuses buf_scores/buf_probs/buf_o_stage — do NOT
    /// split heads across streams without per-head score buffers. seq ≤ 1024
    /// (buf_scores/probs are [1024,1024]).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn vit_attention_gemm(
        &self,
        gpu: &dyn GpuBackend,
        qkv: DevicePtr, // [seq, 3*H*D]
        o: DevicePtr,   // [seq, H*D]
        cos: DevicePtr, // [seq, D]
        sin: DevicePtr, // [seq, D]
        seq: u32,
        stream: u64,
    ) -> Result<()> {
        debug_assert!(
            seq <= 1024,
            "ViT SDPA seq {seq} exceeds buf_scores cap (1024)"
        );
        let h_n = self.num_heads as u32;
        let d = self.head_dim as u32; // 72
        let hd = self.hidden_size as u32; // H*D = 1152

        // (1) rope + deinterleave + V-transpose → buf_qr/buf_kr/buf_vt (all heads)
        KernelLaunch::new(gpu, self.k_rope_deint)
            .grid([div_ceil(seq * d, 256), h_n, 1])
            .block([256, 1, 1])
            .arg_ptr(qkv)
            .arg_ptr(self.buf_qr)
            .arg_ptr(self.buf_kr)
            .arg_ptr(self.buf_vt)
            .arg_ptr(cos)
            .arg_ptr(sin)
            .arg_u32(seq)
            .arg_u32(h_n)
            .arg_u32(d)
            .launch(stream)?;

        let qk_head = (seq * d) as usize; // Qr/Kr head stride (seq*D elems)
        let v_head = (d * seq) as usize; // Vt head stride (D*seq elems)
        for head in 0..self.num_heads {
            let qr_h = self.buf_qr.offset(head * qk_head * 2); // ×2 bytes (bf16)
            let kr_h = self.buf_kr.offset(head * qk_head * 2);
            let vt_h = self.buf_vt.offset(head * v_head * 2);
            let o_h = o.offset(head * self.head_dim * 2); // O[seq,H*D] head slot

            // (2) GEMM1: S[seq,seq] = Qr_h[seq,D] @ Kr_h[seq,D]ᵀ (raw, f32 out).
            //     f32out is TILE=16: block (16,16), grid (ceil(N/16),ceil(M/16)).
            KernelLaunch::new(gpu, self.k_gemm_f32)
                .grid([div_ceil(seq, 16), div_ceil(seq, 16), 1])
                .block([16, 16, 1])
                .arg_ptr(qr_h)
                .arg_ptr(kr_h)
                .arg_ptr(self.buf_scores)
                .arg_u32(seq) // M
                .arg_u32(seq) // N
                .arg_u32(d) // K = 72
                .launch(stream)?;

            // (3) row softmax (scale folded) → buf_probs[seq,seq] bf16
            KernelLaunch::new(gpu, self.k_softmax)
                .grid([seq, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(self.buf_scores)
                .arg_ptr(self.buf_probs)
                .arg_u32(seq)
                .arg_u32(d)
                .launch(stream)?;

            // (4) GEMM2: O_stage[seq,D] = P[seq,seq] @ Vt_h[D,seq]ᵀ = P·V.
            //     pipelined grid is (ceil(N/128),ceil(M/128)): N=d→1 tile, M=seq.
            KernelLaunch::new(gpu, self.k_gemm_pipelined)
                .grid([div_ceil(d, 128), div_ceil(seq, 128), 1])
                .block([256, 1, 1])
                .arg_ptr(self.buf_probs)
                .arg_ptr(vt_h)
                .arg_ptr(self.buf_o_stage)
                .arg_u32(seq) // M
                .arg_u32(d) // N
                .arg_u32(seq) // K
                .launch(stream)?;

            // (5) scatter contiguous O_stage[seq,D] → interleaved o head slot
            KernelLaunch::new(gpu, self.k_scatter_head)
                .grid([div_ceil(seq * d, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(self.buf_o_stage)
                .arg_ptr(o_h)
                .arg_u32(seq)
                .arg_u32(d)
                .arg_u32(hd) // dst row stride = H*D
                .launch(stream)?;
        }
        Ok(())
    }

    /// Run one ViT block (in-place on buf_h1; buf_h2 and buf_wide are scratch).
    pub(super) fn vit_block(
        &self,
        blk: &ViTBlock,
        p: usize,
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let h = self.hidden_size as u32;
        let p32 = p as u32;
        let qkv_n = (3 * self.num_heads * self.head_dim) as u32; // 3456
        let inter = self.intermediate_size as u32; // 4304
        let n_h = p * self.hidden_size;
        // Attention-kernel shared memory: scores[p] + q_rope[head_dim].
        let sm_bytes = (p + self.head_dim) * std::mem::size_of::<f32>();

        // --- Attention sub-block ---
        // 1. save residual
        KernelLaunch::new(gpu, self.k_copy)
            .grid([div_ceil(n_h as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.buf_h1)
            .arg_ptr(self.buf_h2)
            .arg_u32(n_h as u32)
            .launch(stream)?;
        // 2. norm1 in-place
        KernelLaunch::new(gpu, self.k_norm)
            .grid([p32, 1, 1])
            .block([h.min(1024), 1, 1])
            .arg_ptr(self.buf_h1)
            .arg_ptr(blk.norm1_w)
            .arg_ptr(blk.norm1_b)
            .arg_u32(p32)
            .arg_u32(h)
            .arg_f32(1e-6)
            .launch(stream)?;
        // 3. QKV GEMM → buf_wide
        self.vit_gemm_bias(
            gpu,
            self.buf_h1,
            blk.qkv_w,
            blk.qkv_b,
            self.buf_wide,
            p32,
            qkv_n,
            h,
            stream,
        )?;
        // 4. Attention. GEMM-based SDPA by default; ATLAS_VISION_ATTN_LEGACY=1
        //    restores the warp-per-query kernel for A/B / fallback. Also auto-
        //    falls back when the GEMM-ViT kernels aren't in this model's vision
        //    tree (null handle — qwen3-vl-30b / gemma-4 ship only the legacy
        //    `vision_attention_rope`); without this they'd launch a null kernel.
        if std::env::var("ATLAS_VISION_ATTN_LEGACY").is_ok() || self.k_rope_deint.0 == 0 {
            KernelLaunch::new(gpu, self.k_attn)
                .grid([p32, self.num_heads as u32, 1])
                .block([32, 1, 1])
                .shared_mem(sm_bytes as u32)
                .arg_ptr(self.buf_wide)
                .arg_ptr(self.buf_h1)
                .arg_ptr(self.buf_rope_cos)
                .arg_ptr(self.buf_rope_sin)
                .arg_u32(p32)
                .arg_u32(self.num_heads as u32)
                .arg_u32(self.head_dim as u32)
                .launch(stream)?;
        } else {
            self.vit_attention_gemm(
                gpu,
                self.buf_wide,
                self.buf_h1,
                self.buf_rope_cos,
                self.buf_rope_sin,
                p32,
                stream,
            )?;
        }
        // 5. proj GEMM → buf_wide (reuse)
        self.vit_gemm_bias(
            gpu,
            self.buf_h1,
            blk.proj_w,
            blk.proj_b,
            self.buf_wide,
            p32,
            h,
            h,
            stream,
        )?;
        // 6. residual add: buf_wide += buf_h2
        KernelLaunch::new(gpu, self.k_add)
            .grid([div_ceil(n_h as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.buf_wide)
            .arg_ptr(self.buf_h2)
            .arg_u32(n_h as u32)
            .launch(stream)?;
        // 7. copy post-attn back to buf_h1
        KernelLaunch::new(gpu, self.k_copy)
            .grid([div_ceil(n_h as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.buf_wide)
            .arg_ptr(self.buf_h1)
            .arg_u32(n_h as u32)
            .launch(stream)?;

        // --- FFN sub-block ---
        // 8. save residual
        KernelLaunch::new(gpu, self.k_copy)
            .grid([div_ceil(n_h as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.buf_h1)
            .arg_ptr(self.buf_h2)
            .arg_u32(n_h as u32)
            .launch(stream)?;
        // 9. norm2 in-place
        KernelLaunch::new(gpu, self.k_norm)
            .grid([p32, 1, 1])
            .block([h.min(1024), 1, 1])
            .arg_ptr(self.buf_h1)
            .arg_ptr(blk.norm2_w)
            .arg_ptr(blk.norm2_b)
            .arg_u32(p32)
            .arg_u32(h)
            .arg_f32(1e-6)
            .launch(stream)?;
        // 10. fc1 GEMM → buf_wide
        self.vit_gemm_bias(
            gpu,
            self.buf_h1,
            blk.fc1_w,
            blk.fc1_b,
            self.buf_wide,
            p32,
            inter,
            h,
            stream,
        )?;
        // 11. GELU in-place on buf_wide
        KernelLaunch::new(gpu, self.k_gelu)
            .grid([div_ceil(p32 * inter, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.buf_wide)
            .arg_u32(p32 * inter)
            .launch(stream)?;
        // 12. fc2 GEMM → buf_h1 (overwrites normed hidden, OK — normed already consumed by fc1)
        self.vit_gemm_bias(
            gpu,
            self.buf_wide,
            blk.fc2_w,
            blk.fc2_b,
            self.buf_h1,
            p32,
            h,
            inter,
            stream,
        )?;
        // 13. residual add: buf_h1 += buf_h2
        KernelLaunch::new(gpu, self.k_add)
            .grid([div_ceil(n_h as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.buf_h1)
            .arg_ptr(self.buf_h2)
            .arg_u32(n_h as u32)
            .launch(stream)
    }

    /// Batched ViT block over N images packed at `p_off[i]` (rows), `p_total`
    /// total rows. Identical kernel sequence to `vit_block` except (a) all
    /// element/GEMM counts use `p_total`, and (b) the attention kernel loops
    /// per image over its disjoint `[p_off[i], p_off[i]+p_i[i])` slice so SDPA
    /// never crosses image boundaries. For N=1 (p_off=[0], p_total=p_i[0]) the
    /// emitted kernel stream is identical to `vit_block`.
    pub(super) fn vit_block_batched(
        &self,
        blk: &ViTBlock,
        p_total: usize,
        p_i: &[usize],
        p_off: &[usize],
        gpu: &dyn GpuBackend,
        stream: u64,
    ) -> Result<()> {
        let h = self.hidden_size as u32;
        let pt = p_total as u32;
        let qkv_n = (3 * self.num_heads * self.head_dim) as u32; // 3456
        let inter = self.intermediate_size as u32; // 4304
        let n_h = p_total * self.hidden_size;

        // --- Attention sub-block ---
        // 1. save residual
        KernelLaunch::new(gpu, self.k_copy)
            .grid([div_ceil(n_h as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.buf_h1)
            .arg_ptr(self.buf_h2)
            .arg_u32(n_h as u32)
            .launch(stream)?;
        // 2. norm1 in-place (per-row, M-agnostic)
        KernelLaunch::new(gpu, self.k_norm)
            .grid([pt, 1, 1])
            .block([h.min(1024), 1, 1])
            .arg_ptr(self.buf_h1)
            .arg_ptr(blk.norm1_w)
            .arg_ptr(blk.norm1_b)
            .arg_u32(pt)
            .arg_u32(h)
            .arg_f32(1e-6)
            .launch(stream)?;
        // 3. QKV GEMM over M=p_total → buf_wide
        self.vit_gemm_bias(
            gpu,
            self.buf_h1,
            blk.qkv_w,
            blk.qkv_b,
            self.buf_wide,
            pt,
            qkv_n,
            h,
            stream,
        )?;
        // 4. Attention PER IMAGE over its disjoint slice. buf_wide (QKV) is
        //    read-only here; each image writes a disjoint buf_h1 row range.
        // ATLAS_VISION_NOATTN: skip the attention loop (WRONG output) to measure
        // its share of block time vs the batched GEMMs. Diagnostic only.
        let skip_attn = std::env::var("ATLAS_VISION_NOATTN").is_ok();
        // Legacy when explicitly requested OR when the GEMM-ViT kernels are
        // absent from this model's vision tree (null handle — qwen3-vl-30b /
        // gemma-4 ship only `vision_attention_rope`).
        let legacy_attn =
            std::env::var("ATLAS_VISION_ATTN_LEGACY").is_ok() || self.k_rope_deint.0 == 0;
        for (i, &p) in p_i.iter().enumerate() {
            if skip_attn {
                break;
            }
            let p32 = p as u32;
            let qkv = self.buf_wide.offset(p_off[i] * qkv_n as usize * 2);
            let o = self.buf_h1.offset(p_off[i] * self.hidden_size * 2);
            let cos = self.buf_rope_cos.offset(p_off[i] * self.head_dim * 2);
            let sin = self.buf_rope_sin.offset(p_off[i] * self.head_dim * 2);
            if legacy_attn {
                let sm_bytes = ((p + self.head_dim) * std::mem::size_of::<f32>()) as u32;
                KernelLaunch::new(gpu, self.k_attn)
                    .grid([p32, self.num_heads as u32, 1])
                    .block([32, 1, 1])
                    .shared_mem(sm_bytes)
                    .arg_ptr(qkv)
                    .arg_ptr(o)
                    .arg_ptr(cos)
                    .arg_ptr(sin)
                    .arg_u32(p32)
                    .arg_u32(self.num_heads as u32)
                    .arg_u32(self.head_dim as u32)
                    .launch(stream)?;
            } else {
                self.vit_attention_gemm(gpu, qkv, o, cos, sin, p32, stream)?;
            }
        }
        // 5. proj GEMM over M=p_total → buf_wide (reuse)
        self.vit_gemm_bias(
            gpu,
            self.buf_h1,
            blk.proj_w,
            blk.proj_b,
            self.buf_wide,
            pt,
            h,
            h,
            stream,
        )?;
        // 6. residual add: buf_wide += buf_h2
        KernelLaunch::new(gpu, self.k_add)
            .grid([div_ceil(n_h as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.buf_wide)
            .arg_ptr(self.buf_h2)
            .arg_u32(n_h as u32)
            .launch(stream)?;
        // 7. copy post-attn back to buf_h1
        KernelLaunch::new(gpu, self.k_copy)
            .grid([div_ceil(n_h as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.buf_wide)
            .arg_ptr(self.buf_h1)
            .arg_u32(n_h as u32)
            .launch(stream)?;

        // --- FFN sub-block ---
        // 8. save residual
        KernelLaunch::new(gpu, self.k_copy)
            .grid([div_ceil(n_h as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.buf_h1)
            .arg_ptr(self.buf_h2)
            .arg_u32(n_h as u32)
            .launch(stream)?;
        // 9. norm2 in-place
        KernelLaunch::new(gpu, self.k_norm)
            .grid([pt, 1, 1])
            .block([h.min(1024), 1, 1])
            .arg_ptr(self.buf_h1)
            .arg_ptr(blk.norm2_w)
            .arg_ptr(blk.norm2_b)
            .arg_u32(pt)
            .arg_u32(h)
            .arg_f32(1e-6)
            .launch(stream)?;
        // 10. fc1 GEMM → buf_wide
        self.vit_gemm_bias(
            gpu,
            self.buf_h1,
            blk.fc1_w,
            blk.fc1_b,
            self.buf_wide,
            pt,
            inter,
            h,
            stream,
        )?;
        // 11. GELU in-place on buf_wide
        KernelLaunch::new(gpu, self.k_gelu)
            .grid([div_ceil(pt * inter, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.buf_wide)
            .arg_u32(pt * inter)
            .launch(stream)?;
        // 12. fc2 GEMM → buf_h1
        self.vit_gemm_bias(
            gpu,
            self.buf_wide,
            blk.fc2_w,
            blk.fc2_b,
            self.buf_h1,
            pt,
            h,
            inter,
            stream,
        )?;
        // 13. residual add: buf_h1 += buf_h2
        KernelLaunch::new(gpu, self.k_add)
            .grid([div_ceil(n_h as u32, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.buf_h1)
            .arg_ptr(self.buf_h2)
            .arg_u32(n_h as u32)
            .launch(stream)
    }
}
