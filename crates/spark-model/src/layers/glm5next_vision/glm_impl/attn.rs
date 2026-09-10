// SPDX-License-Identifier: AGPL-3.0-only

//! The tower's two GEMM primitives: a bias-fused projection and the GEMM-based
//! non-causal SDPA over one image's patches.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::super::Glm5NextVisionEncoder;

impl Glm5NextVisionEncoder {
    /// `C[m,n] = A[m,k] @ B[n,k]^T + bias[n]`, all BF16.
    ///
    /// Body copied from `enc_impl::vit_block::vit_gemm_bias`, including its
    /// tensor-core-first decision (`dense_gemm_bf16_pipelined` measured ~40x
    /// the scalar `vision_gemm_bias` on the ViT's large-M shapes). The handles
    /// are hard-required here rather than soft, so the fallback arm only fires
    /// if a future kernel tree drops one.
    ///
    /// `B` is the checkpoint's `nn.Linear` weight verbatim: HF stores
    /// `[out, in]`, which IS `B[n, k]`. Only `downsample` needs a permutation
    /// (see `init::permute_downsample_weight`).
    ///
    /// Four of GLM's projections carry no bias; they pass
    /// `scratch().buf_zero_bias` rather than taking a second code path.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn glm_gemm_bias(
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

    /// Full non-causal SDPA for one image's `[seq, 3*H*D]` QKV slice → `O[seq, H*D]`.
    ///
    /// Once per call: rope + deinterleave + V-transpose across all heads. Then
    /// per head: raw QKᵀ (f32 out) → row softmax with the scale folded in →
    /// P·V → scatter into the interleaved O head slot.
    ///
    /// `vit_rope_deinterleave` applies NEOX rotate-half — `q[d]*cos -
    /// q[d+half]*sin` for `d < half`, `q[d]*cos + q[d-half]*sin` above
    /// (`vision_encoder.cu:290-335`) — which is exactly GLM's `(d, d+32)`
    /// pairing. "Deinterleave" names the QKV stride split, not a GPT-J
    /// conversion. Reused verbatim; there is no GLM-specific rope kernel.
    ///
    /// All launches share `stream`, so each head's chain is ordered before head
    /// h+1 reuses `buf_scores`/`buf_probs`/`buf_o_stage` — do NOT split heads
    /// across streams without per-head score buffers.
    ///
    /// The Qwen twin carries a `debug_assert!(seq <= 1024)` and a matching doc
    /// claim. Both are stale: `buf_scores`/`buf_probs` are sized to `p_max`
    /// (init.rs), so the real bound is `p_max` and it is enforced upstream by
    /// `build_rope_cossin_into`. Not copied.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn glm_attention_gemm(
        &self,
        gpu: &dyn GpuBackend,
        qkv: DevicePtr, // [seq, 3*H*D]
        o: DevicePtr,   // [seq, H*D]
        cos: DevicePtr, // [seq, D]
        sin: DevicePtr, // [seq, D]
        seq: u32,
        stream: u64,
    ) -> Result<()> {
        let h_n = self.num_heads as u32;
        let d = self.head_dim as u32; // 64
        let hd = self.hidden_size as u32; // H*D = 1024

        KernelLaunch::new(gpu, self.k_rope_deint)
            .grid([div_ceil(seq * d, 256), h_n, 1])
            .block([256, 1, 1])
            .arg_ptr(qkv)
            .arg_ptr(self.scratch().buf_qr)
            .arg_ptr(self.scratch().buf_kr)
            .arg_ptr(self.scratch().buf_vt)
            .arg_ptr(cos)
            .arg_ptr(sin)
            .arg_u32(seq)
            .arg_u32(h_n)
            .arg_u32(d)
            .launch(stream)?;

        let qk_head = (seq * d) as usize;
        let v_head = (d * seq) as usize;
        for head in 0..self.num_heads {
            let qr_h = self.scratch().buf_qr.offset(head * qk_head * 2);
            let kr_h = self.scratch().buf_kr.offset(head * qk_head * 2);
            let vt_h = self.scratch().buf_vt.offset(head * v_head * 2);
            let o_h = o.offset(head * self.head_dim * 2);

            // S[seq,seq] = Qr_h[seq,D] @ Kr_h[seq,D]^T, raw, f32 out. TILE=16.
            KernelLaunch::new(gpu, self.k_gemm_f32)
                .grid([div_ceil(seq, 16), div_ceil(seq, 16), 1])
                .block([16, 16, 1])
                .arg_ptr(qr_h)
                .arg_ptr(kr_h)
                .arg_ptr(self.scratch().buf_scores)
                .arg_u32(seq)
                .arg_u32(seq)
                .arg_u32(d)
                .launch(stream)?;

            KernelLaunch::new(gpu, self.k_softmax)
                .grid([seq, 1, 1])
                .block([256, 1, 1])
                .arg_ptr(self.scratch().buf_scores)
                .arg_ptr(self.scratch().buf_probs)
                .arg_u32(seq)
                .arg_u32(d)
                .launch(stream)?;

            KernelLaunch::new(gpu, self.k_gemm_pipelined)
                .grid([div_ceil(d, 128), div_ceil(seq, 128), 1])
                .block([256, 1, 1])
                .arg_ptr(self.scratch().buf_probs)
                .arg_ptr(vt_h)
                .arg_ptr(self.scratch().buf_o_stage)
                .arg_u32(seq)
                .arg_u32(d)
                .arg_u32(seq)
                .launch(stream)?;

            KernelLaunch::new(gpu, self.k_scatter_head)
                .grid([div_ceil(seq * d, 256), 1, 1])
                .block([256, 1, 1])
                .arg_ptr(self.scratch().buf_o_stage)
                .arg_ptr(o_h)
                .arg_u32(seq)
                .arg_u32(d)
                .arg_u32(hd)
                .launch(stream)?;
        }
        Ok(())
    }
}
