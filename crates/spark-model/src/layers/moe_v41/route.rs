// SPDX-License-Identifier: AGPL-3.0-only
// provenance-id: 526f6e616c6420522e205374657369616b

//! `MoeV41::route`: the router logits on the GPU, the top-k selection on the
//! CPU.

use anyhow::{Result, ensure};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::KernelLaunch;

use super::{MoeV41, MoeV41LayerWeights, route_from_logits};

impl MoeV41 {
    /// Router logits on the GPU (f32-accumulated), the selection on the CPU.
    pub fn route(
        &self,
        gpu: &dyn GpuBackend,
        w: &MoeV41LayerWeights,
        x: DevicePtr,
        m: usize,
        stream: u64,
    ) -> Result<(Vec<f32>, Vec<usize>)> {
        self.route_launch(gpu, w, x, m, stream)?;
        self.route_select(gpu, w, m, stream)
    }

    /// The router GEMM/GEMV into `self.logits`: device work only (the half a
    /// CUDA graph can hold).
    pub fn route_launch(
        &self,
        gpu: &dyn GpuBackend,
        w: &MoeV41LayerWeights,
        x: DevicePtr,
        m: usize,
        stream: u64,
    ) -> Result<()> {
        let c = &self.cfg;
        ensure!(
            m >= 1 && m <= c.max_tokens,
            "moe_v41: {m} tokens outside 1..={}",
            c.max_tokens
        );
        // the gate at decode: one thread per logit in strict k order (the same
        // numbers as the tiled kernel, one pass over the gate rows)
        // Decode: the staged GEMV (two gate rows a block through shared
        // memory, the strict-order chain of `router_gemv` over them, same
        // bits). `ATLAS_DS41_ROUTER_STAGED=0` keeps the direct GEMV.
        let (kernel, grid, block, smem) = if m <= 8 {
            if router_staged() {
                (
                    self.k.router_gemv_staged,
                    [(c.n_routed as u32).div_ceil(2), m as u32, 1],
                    [256, 1, 1],
                    3 * c.dim as u32 * 2,
                )
            } else {
                (
                    self.k.router_gemv,
                    [(c.n_routed as u32).div_ceil(64), m as u32, 1],
                    [64, 1, 1],
                    0,
                )
            }
        } else {
            (
                self.k.gemm_f32out,
                [(c.n_routed as u32).div_ceil(16), (m as u32).div_ceil(16), 1],
                [16, 16, 1],
                0,
            )
        };
        KernelLaunch::new(gpu, kernel)
            .grid(grid)
            .block(block)
            .shared_mem(smem)
            .arg_ptr(x)
            .arg_ptr(w.gate_w)
            .arg_ptr(self.logits)
            .arg_u32(m as u32)
            .arg_u32(c.n_routed as u32)
            .arg_u32(c.dim as u32)
            .launch(stream)
    }

    /// The selection: drain the stream, download the logits `route_launch`
    /// wrote, pick the top-k on the CPU as the reference does.
    pub fn route_select(
        &self,
        gpu: &dyn GpuBackend,
        w: &MoeV41LayerWeights,
        m: usize,
        stream: u64,
    ) -> Result<(Vec<f32>, Vec<usize>)> {
        let c = &self.cfg;
        // One stream-ordered read-back (enqueue + one wait on this stream)
        // instead of a stream sync followed by a blocking copy, which the
        // CUDA backend runs as a second async copy + sync on its own stream.
        let mut bytes = vec![0u8; m * c.n_routed * 4];
        gpu.copy_d2h_on_stream(self.logits, &mut bytes, stream)?;
        let logits: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        ensure!(
            w.gate_bias.len() == c.n_routed,
            "gate bias has {} entries for {} experts",
            w.gate_bias.len(),
            c.n_routed
        );
        Ok(route_from_logits(&logits, m, &w.gate_bias, c))
    }
}

impl MoeV41 {
    /// Diagnostics: the router of this layer on `x` (one token), the `k`
    /// best experts by `score + bias` in the reference's order. Drains the
    /// stream; clobbers `self.logits`.
    pub fn route_predict(
        &self,
        gpu: &dyn GpuBackend,
        w: &MoeV41LayerWeights,
        x: DevicePtr,
        k: usize,
        stream: u64,
    ) -> Result<Vec<usize>> {
        let c = &self.cfg;
        self.route_launch(gpu, w, x, 1, stream)?;
        let mut bytes = vec![0u8; c.n_routed * 4];
        gpu.copy_d2h_on_stream(self.logits, &mut bytes, stream)?;
        let logits: Vec<f32> = bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect();
        let mut wide = c.clone();
        wide.topk = k.min(c.n_routed);
        Ok(route_from_logits(&logits, 1, &w.gate_bias, &wide).1)
    }
}

fn router_staged() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| !std::env::var("ATLAS_DS41_ROUTER_STAGED").is_ok_and(|v| v == "0"))
}
