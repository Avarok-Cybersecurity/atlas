// SPDX-License-Identifier: AGPL-3.0-only

//! Setters + transposes + transpose_for_prefill_unified_inner.

use super::*;

impl MoeLayer {
    /// Pick the FP8 grouped GEMM kernel handle — v2 (coalesced thread remap)
    /// when ATLAS_FP8_MOE_COALESCED=1 and the v2 kernel is linked into the
    /// image; v1 otherwise. v1 is the validated default path so any image
    /// that predates the v2 kernel silently stays on the correct behaviour.
    #[inline]
    pub(super) fn fp8_grouped_kernel(&self) -> KernelHandle {
        if self.fp8_moe_coalesced_enabled && self.moe_fp8_grouped_gemm_v2_k.0 != 0 {
            self.moe_fp8_grouped_gemm_v2_k
        } else {
            self.moe_fp8_grouped_gemm_k
        }
    }

    /// Transpose MoE weights for coalesced prefill GEMM reads.
    ///
    /// Transposes per-expert routed weights [N, K/2] → [K/2, N] to enable
    /// the cp.async pipelined FP8-MMA K64 kernels. This doubles expert
    /// memory (~17 GB for 35B, ~30 GB for 122B) but eliminates the
    /// catastrophic uncoalesced B reads in the fallback grouped GEMM,
    /// cutting MoE prefill time by ~2x.
    /// Set pre-expert norm (Gemma-4 26B: pre_feedforward_layernorm_2).
    /// Applied to input AFTER routing but BEFORE expert dispatch.
    pub fn set_pre_expert_norm(&mut self, norm: crate::weight_map::DenseWeight) {
        self.pre_expert_norm = Some(norm);
    }

    /// Set GeGLU activation for MoE experts (Gemma-4 26B).
    /// Replaces SiLU with GELU in the sorted/unfused path and forces decode
    /// to use the sorted path (avoiding fused SiLU kernels).
    pub fn set_gelu_activation(&mut self, gpu: &dyn GpuBackend) -> Result<()> {
        self.moe_act_mul = gpu.kernel("gelu", "gelu_mul")?;
        self.gelu_activation = true;
        Ok(())
    }

    pub fn transpose_for_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        self.transpose_for_prefill_impl(gpu, config, true)
    }

    /// Transpose only the gate+up routed weights, leaving the down projection
    /// in its original layout. Cuts the transpose memory cost from ~3×
    /// (gate+up+down) to ~2× per expert. Used by MiniMax M2.7-NVFP4 EP=2
    /// when the full transpose doesn't fit but gate+up does — the fused
    /// `moe_w4a16_fused_gate_up_k64_n128` kernel still runs (capturing the
    /// dominant gate+up bandwidth savings), while down stays on the
    /// uncoalesced grouped-GEMM path.
    pub fn transpose_gate_up_for_prefill(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        self.transpose_for_prefill_impl(gpu, config, false)
    }

    pub(super) fn transpose_for_prefill_impl(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
        include_down: bool,
    ) -> Result<()> {
        let h = config.hidden_size;
        let inter = config.moe_intermediate_size;
        let shared_inter = config.shared_expert_intermediate_size;

        // Transpose per-expert routed weights for coalesced prefill GEMM reads.
        let num_experts = self.weights.experts.len();
        let mut gate_t = Vec::with_capacity(num_experts);
        let mut up_t = Vec::with_capacity(num_experts);
        let mut down_t = Vec::with_capacity(num_experts);

        for expert in &self.weights.experts {
            if expert.gate_proj.is_null() {
                gate_t.push(QuantizedWeight::null());
                up_t.push(QuantizedWeight::null());
                if include_down {
                    down_t.push(QuantizedWeight::null());
                }
            } else {
                gate_t.push(expert.gate_proj.transpose_for_gemm(gpu, inter, h)?);
                up_t.push(expert.up_proj.transpose_for_gemm(gpu, inter, h)?);
                if include_down {
                    down_t.push(expert.down_proj.transpose_for_gemm(gpu, h, inter)?);
                }
            }
        }

        self.gate_ptrs_t = Some(build_ptr_table_from_qw(&gate_t, gpu)?);
        self.up_ptrs_t = Some(build_ptr_table_from_qw(&up_t, gpu)?);
        if include_down {
            self.down_ptrs_t = Some(build_ptr_table_from_qw(&down_t, gpu)?);
        }

        // Transpose shared expert weights (tiny: ~5 MB per layer).
        if !self.weights.shared_expert.gate_proj.is_null() && shared_inter > 0 {
            self.shared_gate_t = Some(self.weights.shared_expert.gate_proj.transpose_for_gemm(
                gpu,
                shared_inter,
                h,
            )?);
            self.shared_up_t = Some(self.weights.shared_expert.up_proj.transpose_for_gemm(
                gpu,
                shared_inter,
                h,
            )?);
            if include_down {
                self.shared_down_t =
                    Some(self.weights.shared_expert.down_proj.transpose_for_gemm(
                        gpu,
                        h,
                        shared_inter,
                    )?);
            }
        }

        Ok(())
    }

    /// Phase 8a unified-layout transpose pass: build persistent transposed
    /// gate/up/down for all experts, freeing the untransposed copies between
    /// phases so the entire pass fits in tight memory budgets that the
    /// non-unified `transpose_for_prefill_impl(true)` would reject.
    ///
    /// Phased flow (memory math for MiniMax M2.7-NVFP4 EP=2 ≈ 47 GB free):
    ///   A. Transpose gate+up               (allocs +39 GB; free ≈ 8 GB)
    ///   B. Free gate+up untransposed       (frees 39 GB; free ≈ 47 GB)
    ///   C. Transpose down                  (allocs +20 GB; free ≈ 27 GB)
    ///   D. Free down untransposed          (frees 20 GB; free ≈ 47 GB)
    ///
    /// Net memory: same as starting point, but layout is now unified
    /// (transposed-only) — the `[N, K/2]` decode kernels can no longer
    /// run; dispatch must use the `_t` decode kernels (which do).
    ///
    /// Caller responsibilities:
    ///   1. Set `ATLAS_UNIFIED_MOE_LAYOUT=1` so `MoeLayer::use_t_layout_for_decode()`
    ///      returns true at dispatch time.
    ///   2. Call this method INSTEAD of `transpose_for_prefill` /
    ///      `transpose_gate_up_for_prefill`.
    pub fn transpose_for_prefill_unified(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        self.transpose_for_prefill_unified_inner(gpu, config, false)
    }

    /// Hybrid-layout transpose pass — analogue of `transpose_for_prefill_unified`
    /// that **keeps** the untransposed originals so decode + MTP verify dispatch
    /// can continue using the warp-reduction kernels. Allocates ~58 GB
    /// transposed alongside the existing ~58 GB originals on MiniMax M2.7-NVFP4
    /// EP=2; fits in 122 GB GB10 with KV-cache headroom up to ~32K context.
    /// Caller is responsible for memory-fit gating (factory checks free memory
    /// before invoking this).
    pub fn transpose_for_prefill_hybrid(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
    ) -> Result<()> {
        self.transpose_for_prefill_unified_inner(gpu, config, true)
    }

    /// Phased build of the transposed weight set. When `keep_originals` is true
    /// (hybrid-layout mode), Phase B and Phase D frees are skipped so decode
    /// paths still find the untransposed weights. When false (unified-layout
    /// mode), the originals are freed between phases — current Phase 8a
    /// behavior.
    pub(super) fn transpose_for_prefill_unified_inner(
        &mut self,
        gpu: &dyn GpuBackend,
        config: &atlas_core::config::ModelConfig,
        keep_originals: bool,
    ) -> Result<()> {
        let h = config.hidden_size;
        let inter = config.moe_intermediate_size;
        let shared_inter = config.shared_expert_intermediate_size;
        let num_experts = self.weights.experts.len();

        // ── Phase A: transpose gate+up routed experts ──
        let mut gate_t = Vec::with_capacity(num_experts);
        let mut up_t = Vec::with_capacity(num_experts);
        for expert in &self.weights.experts {
            if expert.gate_proj.is_null() {
                gate_t.push(QuantizedWeight::null());
                up_t.push(QuantizedWeight::null());
            } else {
                gate_t.push(expert.gate_proj.transpose_for_gemm(gpu, inter, h)?);
                up_t.push(expert.up_proj.transpose_for_gemm(gpu, inter, h)?);
            }
        }
        self.gate_ptrs_t = Some(build_ptr_table_from_qw(&gate_t, gpu)?);
        self.up_ptrs_t = Some(build_ptr_table_from_qw(&up_t, gpu)?);
        // Shared expert (tiny, do unconditionally — fits regardless).
        if !self.weights.shared_expert.gate_proj.is_null() && shared_inter > 0 {
            self.shared_gate_t = Some(self.weights.shared_expert.gate_proj.transpose_for_gemm(
                gpu,
                shared_inter,
                h,
            )?);
            self.shared_up_t = Some(self.weights.shared_expert.up_proj.transpose_for_gemm(
                gpu,
                shared_inter,
                h,
            )?);
        }

        if !keep_originals {
            // ── Phase B: free gate+up untransposed ──
            // The previous gate_ptrs / up_ptrs device-side pointer tables now
            // contain stale addresses, but the unified dispatch never reads
            // them (gated by `use_t_layout_for_decode()`).
            for expert in &mut self.weights.experts {
                if !expert.gate_proj.weight.is_null() {
                    gpu.free(expert.gate_proj.weight)?;
                    gpu.free(expert.gate_proj.weight_scale)?;
                    expert.gate_proj.weight = DevicePtr::NULL;
                    expert.gate_proj.weight_scale = DevicePtr::NULL;
                }
                if !expert.up_proj.weight.is_null() {
                    gpu.free(expert.up_proj.weight)?;
                    gpu.free(expert.up_proj.weight_scale)?;
                    expert.up_proj.weight = DevicePtr::NULL;
                    expert.up_proj.weight_scale = DevicePtr::NULL;
                }
            }
            if !self.weights.shared_expert.gate_proj.weight.is_null() && shared_inter > 0 {
                gpu.free(self.weights.shared_expert.gate_proj.weight)?;
                gpu.free(self.weights.shared_expert.gate_proj.weight_scale)?;
                self.weights.shared_expert.gate_proj.weight = DevicePtr::NULL;
                self.weights.shared_expert.gate_proj.weight_scale = DevicePtr::NULL;
                gpu.free(self.weights.shared_expert.up_proj.weight)?;
                gpu.free(self.weights.shared_expert.up_proj.weight_scale)?;
                self.weights.shared_expert.up_proj.weight = DevicePtr::NULL;
                self.weights.shared_expert.up_proj.weight_scale = DevicePtr::NULL;
            }
        }

        // ── Phase C: transpose down routed experts ──
        let mut down_t = Vec::with_capacity(num_experts);
        for expert in &self.weights.experts {
            if expert.down_proj.is_null() {
                down_t.push(QuantizedWeight::null());
            } else {
                down_t.push(expert.down_proj.transpose_for_gemm(gpu, h, inter)?);
            }
        }
        self.down_ptrs_t = Some(build_ptr_table_from_qw(&down_t, gpu)?);
        if !self.weights.shared_expert.down_proj.is_null() && shared_inter > 0 {
            self.shared_down_t = Some(self.weights.shared_expert.down_proj.transpose_for_gemm(
                gpu,
                h,
                shared_inter,
            )?);
        }

        if !keep_originals {
            // ── Phase D: free down untransposed ──
            for expert in &mut self.weights.experts {
                if !expert.down_proj.weight.is_null() {
                    gpu.free(expert.down_proj.weight)?;
                    gpu.free(expert.down_proj.weight_scale)?;
                    expert.down_proj.weight = DevicePtr::NULL;
                    expert.down_proj.weight_scale = DevicePtr::NULL;
                }
            }
            if !self.weights.shared_expert.down_proj.weight.is_null() && shared_inter > 0 {
                gpu.free(self.weights.shared_expert.down_proj.weight)?;
                gpu.free(self.weights.shared_expert.down_proj.weight_scale)?;
                self.weights.shared_expert.down_proj.weight = DevicePtr::NULL;
                self.weights.shared_expert.down_proj.weight_scale = DevicePtr::NULL;
            }
        }

        Ok(())
    }
}
