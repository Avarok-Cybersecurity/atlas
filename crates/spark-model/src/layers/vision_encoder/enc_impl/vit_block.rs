// SPDX-License-Identifier: AGPL-3.0-only

//! Single ViT block (norm → QKV → RoPE attention → proj → +residual →
//! norm → fc1 → GELU → fc2 → +residual).

use anyhow::Result;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

use super::super::{ViTBlock, VisionEncoder};

impl VisionEncoder {
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
        KernelLaunch::new(gpu, self.k_gemm)
            .grid([div_ceil(qkv_n, 32), div_ceil(p32, 32), 1])
            .block([32, 32, 1])
            .arg_ptr(self.buf_h1)
            .arg_ptr(blk.qkv_w)
            .arg_ptr(blk.qkv_b)
            .arg_ptr(self.buf_wide)
            .arg_u32(p32)
            .arg_u32(qkv_n)
            .arg_u32(h)
            .launch(stream)?;
        // 4. Attention with 2D rotary pos emb applied inline to Q/K
        //    (blockDim=32 for correct warp reduction; rope buffers already
        //    uploaded once per image by `build_rope_cossin`).
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
        // 5. proj GEMM → buf_wide (reuse)
        KernelLaunch::new(gpu, self.k_gemm)
            .grid([div_ceil(h, 32), div_ceil(p32, 32), 1])
            .block([32, 32, 1])
            .arg_ptr(self.buf_h1)
            .arg_ptr(blk.proj_w)
            .arg_ptr(blk.proj_b)
            .arg_ptr(self.buf_wide)
            .arg_u32(p32)
            .arg_u32(h)
            .arg_u32(h)
            .launch(stream)?;
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
        KernelLaunch::new(gpu, self.k_gemm)
            .grid([div_ceil(inter, 32), div_ceil(p32, 32), 1])
            .block([32, 32, 1])
            .arg_ptr(self.buf_h1)
            .arg_ptr(blk.fc1_w)
            .arg_ptr(blk.fc1_b)
            .arg_ptr(self.buf_wide)
            .arg_u32(p32)
            .arg_u32(inter)
            .arg_u32(h)
            .launch(stream)?;
        // 11. GELU in-place on buf_wide
        KernelLaunch::new(gpu, self.k_gelu)
            .grid([div_ceil(p32 * inter, 256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(self.buf_wide)
            .arg_u32(p32 * inter)
            .launch(stream)?;
        // 12. fc2 GEMM → buf_h1 (overwrites normed hidden, OK — normed already consumed by fc1)
        KernelLaunch::new(gpu, self.k_gemm)
            .grid([div_ceil(h, 32), div_ceil(p32, 32), 1])
            .block([32, 32, 1])
            .arg_ptr(self.buf_wide)
            .arg_ptr(blk.fc2_w)
            .arg_ptr(blk.fc2_b)
            .arg_ptr(self.buf_h1)
            .arg_u32(p32)
            .arg_u32(h)
            .arg_u32(inter)
            .launch(stream)?;
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
