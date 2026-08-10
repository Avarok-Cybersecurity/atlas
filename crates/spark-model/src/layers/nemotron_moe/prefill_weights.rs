// SPDX-License-Identifier: AGPL-3.0-only

//! `NemotronMoeLayer` prefill-side weight preparation: the transposed-NVFP4 /
//! pre-dequantized-FP8 / FP4 expert-weight copies and the dense-GEMM prefill
//! dispatcher. Split from `nemotron_moe.rs` (500-LoC cap).

use anyhow::Result;
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use super::NemotronMoeLayer;
use super::build_ptr_table_from_weights;
use crate::layers::ops;
use crate::weight_map::DenseWeight;

impl NemotronMoeLayer {
    /// Dense BF16 GEMM for the prefill path.
    ///
    /// Prefers the pipelined tensor-core kernel and falls back to the scalar
    /// `dense_gemm_bf16` only if it is not compiled for this target. Single
    /// source of truth for the three dense GEMMs of a LatentMoE layer (gate,
    /// fc1_latent, fc2_latent).
    #[allow(clippy::too_many_arguments)]
    pub(super) fn dense_gemm_prefill(
        &self,
        gpu: &dyn GpuBackend,
        input: DevicePtr,
        weight: &DenseWeight,
        output: DevicePtr,
        m: u32,
        n: u32,
        k: u32,
        stream: u64,
    ) -> Result<()> {
        if self.dense_gemm_pipelined_k.0 != 0 {
            ops::dense_gemm_bf16_pipelined(
                gpu,
                self.dense_gemm_pipelined_k,
                input,
                weight,
                output,
                m,
                n,
                k,
                stream,
            )
        } else {
            ops::dense_gemm(
                gpu,
                self.dense_gemm_k,
                input,
                weight,
                output,
                m,
                n,
                k,
                stream,
            )
        }
    }

    /// Transpose expert weights for fast grouped GEMM prefill.
    /// Called from weight loader after construction. Skips expert transposition
    /// when memory is tight (Super 120B: 128 experts × 40 layers would OOM).
    pub fn prepare_prefill_weights(&mut self, gpu: &dyn GpuBackend, config: &ModelConfig) {
        let h = config.hidden_size;
        let inter = self.moe_inter;
        let shared_inter = config.shared_expert_intermediate_size;

        // Only transpose routed experts for small models (Nano 30B: 23 MoE layers × 128 experts).
        // Super 120B has 40 MoE layers × 128 experts = 5120 matrices — too much memory.
        // The sorted grouped GEMM still works with non-transposed weights via the base kernel.
        if self.moe_latent_size == 0 {
            let expert_k = h;
            let mut up_t = Vec::new();
            let mut down_t = Vec::new();
            for expert in &self.weights.experts {
                if let Ok(ut) = expert.up_proj.transpose_for_gemm(gpu, inter, expert_k) {
                    up_t.push(ut);
                }
                if let Ok(dt) = expert.down_proj.transpose_for_gemm(gpu, expert_k, inter) {
                    down_t.push(dt);
                }
            }
            if up_t.len() == self.weights.experts.len()
                && let Ok(ptrs) = build_ptr_table_from_weights(&up_t, gpu)
            {
                self.up_ptrs_t = Some(ptrs);
            }
            if down_t.len() == self.weights.experts.len()
                && let Ok(ptrs) = build_ptr_table_from_weights(&down_t, gpu)
            {
                self.down_ptrs_t = Some(ptrs);
            }
        }

        // Transpose the shared expert weights unconditionally.
        //
        // This is only TWO matrices per layer (shared_up, shared_down), unlike the
        // routed experts above (512 per layer). It was previously gated behind the
        // same `moe_latent_size == 0` memory guard, which lumped a ~cheap transpose
        // in with the expensive one and left every LatentMoE layer on the base
        // `w4a16_gemm`. On Puzzle the shared-expert GEMMs were a large slice of
        // prefill; the transposed copies unlock `w4a16_gemm_t` (FP8 MMA, N128/K32,
        // cp.async) for them. `.ok()` keeps the base GEMM as the fallback.
        //
        // Same idea as the SSM projections: pre-dequantize to FP8 E4M3 once at load so
        // prefill runs `fp8_gemm_t_m128_mfast` (no dequant phase, M on the fast grid
        // axis). But unlike the SSM ones this is OPT-IN, because it is a real trade:
        //
        //   off (default) : 1k TTFT 490 ms, decode 33.3 tok/s  <- decode at baseline
        //   on            : 1k TTFT 450 ms, decode 32.6 tok/s  <- -2.4% decode
        //
        // The ~2.1 GB of extra resident weights (shared_up/down + fc1/fc2) costs ~2%
        // of decode, which is memory-bandwidth-bound on this box. Same-binary A/B,
        // 10 runs each, verified on a cold server (not thermal). The SSM copies
        // (4.3 GB, allocated during the load itself) cost nothing measurable, so the
        // two are gated separately. Set ATLAS_SHARED_FP8_PREFILL=1 to take the trade.
        let fp8_prefill = std::env::var("ATLAS_SHARED_FP8_PREFILL").is_ok();
        if fp8_prefill
            && self.fp8_gemm_m128_k.0 != 0
            && let Ok(pdq_k) = gpu.kernel("w4a16", "predequant_nvfp4_to_fp8")
        {
            self.shared_up_pd_fp8 = self
                .weights
                .shared_up
                .predequant_to_fp8(gpu, pdq_k, shared_inter, h, 0)
                .ok();
            self.shared_down_pd_fp8 = self
                .weights
                .shared_down
                .predequant_to_fp8(gpu, pdq_k, h, shared_inter, 0)
                .ok();
        }
        if self.shared_up_pd_fp8.is_none() || self.shared_down_pd_fp8.is_none() {
            self.shared_up_t = self
                .weights
                .shared_up
                .transpose_for_gemm(gpu, shared_inter, h)
                .ok();
            self.shared_down_t = self
                .weights
                .shared_down
                .transpose_for_gemm(gpu, h, shared_inter)
                .ok();
        }

        // fc1/fc2 latent are BF16 dense and were the only prefill GEMMs still on
        // dense_gemm_bf16_pipelined. Converting them to FP8 E4M3 both halves their
        // bytes and moves them onto fp8_gemm_t_m128_mfast.
        let lat = self.moe_latent_size;
        if lat > 0
            && fp8_prefill
            && self.fp8_gemm_m128_k.0 != 0
            && let Ok(b2f) = gpu.kernel("w4a16", "bf16_to_fp8")
        {
            let conv = |w: &DenseWeight, n: usize, k: usize| -> Option<DevicePtr> {
                let dst = gpu.alloc(n * k).ok()?;
                crate::layers::ops::bf16_to_fp8(gpu, b2f, w.weight, dst, (n * k) as u32, 0).ok()?;
                gpu.synchronize(0).ok()?;
                Some(dst)
            };
            self.fc1_pd_fp8 = self
                .weights
                .fc1_latent_proj
                .as_ref()
                .and_then(|w| conv(w, lat, h));
            self.fc2_pd_fp8 = self
                .weights
                .fc2_latent_proj
                .as_ref()
                .and_then(|w| conv(w, h, lat));
        }
    }
}
