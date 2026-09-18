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
        let c = &self.cfg;
        ensure!(
            m >= 1 && m <= c.max_tokens,
            "moe_v41: {m} tokens outside 1..={}",
            c.max_tokens
        );
        // the gate at decode: one thread per logit in strict k order (the same
        // numbers as the tiled kernel, one pass over the gate rows)
        let (kernel, grid, block) = if m <= 8 {
            (
                self.k.router_gemv,
                [(c.n_routed as u32).div_ceil(64), m as u32, 1],
                [64, 1, 1],
            )
        } else {
            (
                self.k.gemm_f32out,
                [(c.n_routed as u32).div_ceil(16), (m as u32).div_ceil(16), 1],
                [16, 16, 1],
            )
        };
        KernelLaunch::new(gpu, kernel)
            .grid(grid)
            .block(block)
            .arg_ptr(x)
            .arg_ptr(w.gate_w)
            .arg_ptr(self.logits)
            .arg_u32(m as u32)
            .arg_u32(c.n_routed as u32)
            .arg_u32(c.dim as u32)
            .launch(stream)?;
        gpu.synchronize(stream)?;
        let mut bytes = vec![0u8; m * c.n_routed * 4];
        gpu.copy_d2h(self.logits, &mut bytes)?;
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
