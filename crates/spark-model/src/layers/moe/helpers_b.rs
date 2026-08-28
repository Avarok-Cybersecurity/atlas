// SPDX-License-Identifier: AGPL-3.0-only

//! set_down_transpose_scratch.

use super::*;

impl MoeLayer {
    /// Wire a shared per-prefill down_proj scratch + transposed pointer table.
    ///
    /// Called by the factory after the persistent MoE transpose pass falls
    /// back to gate+up only. The scratch and pointer tables are shared
    /// across all MoE layers — one allocation reused layer-by-layer during
    /// the sequential forward. The same `scale2_vals` buffer is reused
    /// from the existing untransposed `down_ptrs` (transpose preserves
    /// per-tensor scales).
    pub fn set_down_transpose_scratch(
        &mut self,
        scratch_packed: DevicePtr,
        scratch_scale: DevicePtr,
        packed_ptrs_t: DevicePtr,
        scale_ptrs_t: DevicePtr,
    ) {
        self.down_t_scratch_packed = Some(scratch_packed);
        self.down_t_scratch_scale = Some(scratch_scale);
        self.down_ptrs_t = Some(ExpertPtrTable {
            packed_ptrs: packed_ptrs_t,
            scale_ptrs: scale_ptrs_t,
            scale2_vals: self.down_ptrs.scale2_vals,
        });
    }

    /// Run the batched transpose kernel to populate `down_t_scratch_*` from
    /// the untransposed `down_ptrs` source. Must be called once at the
    /// start of every layer's prefill, before the silu_down GEMM. No-op
    /// when scratch isn't wired (decode-only / persistent-full-transpose
    /// paths).
    pub(crate) fn transpose_down_into_scratch(
        &self,
        ctx: &crate::layer::ForwardContext,
        stream: u64,
    ) -> Result<()> {
        let Some(dpt) = self.down_ptrs_t.as_ref() else {
            return Ok(());
        };
        // Only run the transpose when scratch is wired (vs persistent
        // transpose_for_prefill_impl path which sets down_ptrs_t to its
        // own allocations and leaves scratch fields None).
        if self.down_t_scratch_packed.is_none() {
            return Ok(());
        }
        let num_experts = ctx.config.num_experts as u32;
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.moe_intermediate_size as u32;
        // Packed: [N=hidden, K/2=inter/2] → [K/2, N] per expert.
        crate::layers::ops::moe_transpose_u8_batched(
            ctx.gpu,
            self.moe_transpose_u8_batched_k,
            self.down_ptrs.packed_ptrs,
            dpt.packed_ptrs,
            h,
            inter / 2,
            num_experts,
            stream,
        )?;
        // Scale: [N, K/GROUP_SIZE=inter/16] → [K/16, N] per expert.
        crate::layers::ops::moe_transpose_u8_batched(
            ctx.gpu,
            self.moe_transpose_u8_batched_k,
            self.down_ptrs.scale_ptrs,
            dpt.scale_ptrs,
            h,
            inter / 16,
            num_experts,
            stream,
        )?;
        Ok(())
    }

    /// **NOT CURRENTLY WIRED IN** — this helper attempted to overlap the
    /// lazy down_proj transpose with the TP attention allreduce by
    /// kicking it off on `prefill_stream` right after attention. It
    /// regressed cold TTFT by ~30 % on GB10 — both when scheduled
    /// against compute-bound MoE GEMMs AND when scheduled against the
    /// (RDMA-dominated) TP allreduce window. Either GB10's SM scheduling
    /// has hidden contention costs across streams, or the per-call
    /// event-sync overhead exceeds the ~4 ms transpose savings.
    ///
    /// Kept in source for future reference — future work that figures
    /// out the GB10 stream-scheduling pattern can re-wire it from
    /// `qwen3_attention::trait_impl::prefill` after attention but before
    /// the TP allreduce, then have silu_down stall via
    /// `lazy_transpose_done_event()`.
    #[allow(dead_code)]
    pub(crate) fn kick_off_lazy_transpose(
        &self,
        ctx: &crate::layer::ForwardContext,
        compute_stream: u64,
    ) -> Result<()> {
        let Some(dpt) = self.down_ptrs_t.as_ref() else {
            return Ok(());
        };
        if self.down_t_scratch_packed.is_none() {
            return Ok(());
        }
        // prefill_stream waits for compute_stream's "attention done" point.
        ctx.gpu.record_event(self.event_a, compute_stream)?;
        ctx.gpu
            .stream_wait_event(self.prefill_stream, self.event_a)?;

        let num_experts = ctx.config.num_experts as u32;
        let h = ctx.config.hidden_size as u32;
        let inter = ctx.config.moe_intermediate_size as u32;
        crate::layers::ops::moe_transpose_u8_batched(
            ctx.gpu,
            self.moe_transpose_u8_batched_k,
            self.down_ptrs.packed_ptrs,
            dpt.packed_ptrs,
            h,
            inter / 2,
            num_experts,
            self.prefill_stream,
        )?;
        crate::layers::ops::moe_transpose_u8_batched(
            ctx.gpu,
            self.moe_transpose_u8_batched_k,
            self.down_ptrs.scale_ptrs,
            dpt.scale_ptrs,
            h,
            inter / 16,
            num_experts,
            self.prefill_stream,
        )?;
        // Record transpose-done event on the secondary stream. The
        // silu_down call site stalls compute_stream on this event before
        // reading from scratch.
        ctx.gpu.record_event(self.event_b, self.prefill_stream)?;
        Ok(())
    }

    /// Companion to the (currently-unwired) `kick_off_lazy_transpose`
    /// — silu_down would call this to know whether to stall on the
    /// secondary-stream event.
    #[allow(dead_code)]
    pub(crate) fn has_overlapped_transpose(&self) -> bool {
        self.down_t_scratch_packed.is_some()
    }

    /// Companion to `kick_off_lazy_transpose`.
    #[allow(dead_code)]
    pub(crate) fn lazy_transpose_done_event(&self) -> u64 {
        self.event_b
    }

    /// True when prefill dispatch (forward_batched) should route to
    /// `_t` transposed-layout kernels.
    ///
    /// Fires for both unified mode (Phase 8a — originals freed) and hybrid
    /// mode (Block C Path 2 — originals retained alongside transposed).
    /// Both build the same persistent `*_ptrs_t` device-side pointer tables.
    ///
    /// Requires:
    /// 1. `ATLAS_UNIFIED_MOE_LAYOUT=1` OR `ATLAS_HYBRID_MOE_LAYOUT=1`
    ///    (read at construction).
    /// 2. Persistent transposed pointer tables for all three projections.
    /// 3. NOT the lazy-scratch path — scratch-backed `down_ptrs_t` only
    ///    holds one layer at a time, so multi-layer dispatch would read
    ///    stale data. Persistent transpose pass must have populated down_t.
    #[inline]
    pub(crate) fn use_t_layout_for_prefill(&self) -> bool {
        (self.unified_layout || self.hybrid_layout)
            && self.gate_ptrs_t.is_some()
            && self.up_ptrs_t.is_some()
            && self.down_ptrs_t.is_some()
            && self.down_t_scratch_packed.is_none()
    }

    /// True when decode dispatch (forward, forward_k2, forward_k3) should
    /// route to `_t` transposed-layout kernels.
    ///
    /// Only fires in unified mode — hybrid mode keeps the originals so
    /// decode + MTP verify (small N, warp-reduction wins) can preserve
    /// the ~35 tok/s throughput that pure unified layout regresses by 15 %.
    #[inline]
    /// Eligible to route DECODE through the grouped read-once GEMM
    /// (forward_prefill). Native-NVFP4 routed only: bf16/fp8-dequant gate ptrs
    /// absent, no DeepSeek-V4 hash routing, single-GPU (no EP). Deliberately
    /// ALLOWS the mixed BF16 shared expert (forward_prefill handles it as a
    /// separate batched pass), unlike forward_token_major_decode which bails.
    pub(crate) fn grouped_decode_ok(&self) -> bool {
        self.bf16_gate_weight_ptrs.is_none()
            && self.fp8_gate_weight_ptrs.is_none()
            && self.tid2eid_dev.is_none()
    }

    pub(crate) fn use_t_layout_for_decode(&self) -> bool {
        self.unified_layout
            && !self.hybrid_layout
            && self.gate_ptrs_t.is_some()
            && self.up_ptrs_t.is_some()
            && self.down_ptrs_t.is_some()
            && self.down_t_scratch_packed.is_none()
    }
}

impl super::MoeLayer {
    /// The routed grouped-GEMM kernel: the bit-exact wider-K twin when it
    /// resolved (ATLAS_MOE_GROUPED_K32=1 and the target ships it), else the
    /// original. Both take the same grid/block, so the launcher is shared.
    pub(super) fn grouped_gemm_kernel(&self) -> spark_runtime::gpu::KernelHandle {
        if self.moe_grouped_gemm_k32.0 != 0 {
            self.moe_grouped_gemm_k32
        } else {
            self.moe_grouped_gemm
        }
    }
}

impl super::MoeLayer {
    /// Launch the routed grouped GEMM, choosing the widest tiling this build
    /// resolved. M_TILE=256 re-reads the expert weights ONCE instead of three
    /// times at ~160 rows/expert, so `max_m_tiles` must be recomputed against
    /// 256 — the same adjustment the m128 path makes with `div_ceil(2)`.
    /// All arms are bit-exact with each other.
    ///
    /// ⚠ BOTH WIDE ARMS MEASURED AS NON-WINS (qwen4_exp, GB10, 2026-08-27) and
    /// are default-OFF. Controlled four-arm sweep at 28K prefill, same box,
    /// back-to-back: baseline 272 → +QSA-TC 286 → +k32 289 → +m256 290 tok/s.
    /// The k32/m256 deltas (+1.0% / +1.4% over the QSA-TC arm) are inside
    /// run-to-run noise; the entire +5.1% came from the QSA tensor-core
    /// scorer, NOT from this GEMM.
    ///
    /// nsys proves the null is real rather than a silent `try_kernel`
    /// fallback: `..._m256` appears in the trace with 540 calls / 38.1% of GPU
    /// time, and averages 31.30 ms/call against the base kernel's 26.17 ms —
    /// i.e. ~20% SLOWER per call at matched shapes. A DRAM-bound standalone
    /// harness predicted 1.43x for it. Do not re-enable either arm on
    /// microbenchmark evidence; re-measure end-to-end first.
    #[allow(clippy::too_many_arguments)]
    pub(super) fn launch_grouped_gemm(
        &self,
        gpu: &dyn spark_runtime::gpu::GpuBackend,
        a: spark_runtime::gpu::DevicePtr,
        packed_ptrs: spark_runtime::gpu::DevicePtr,
        scale_ptrs: spark_runtime::gpu::DevicePtr,
        scale2_vals: spark_runtime::gpu::DevicePtr,
        c: spark_runtime::gpu::DevicePtr,
        expert_offsets: spark_runtime::gpu::DevicePtr,
        sorted_token_ids: spark_runtime::gpu::DevicePtr,
        num_experts: u32,
        n_out: u32,
        k: u32,
        max_m_tiles: u32,
        stream: u64,
    ) -> anyhow::Result<()> {
        if self.moe_grouped_gemm_m256.0 != 0 {
            return crate::layers::ops::moe_w4a16_grouped_gemm_ptrtable_m256(
                gpu,
                self.moe_grouped_gemm_m256,
                a,
                packed_ptrs,
                scale_ptrs,
                scale2_vals,
                c,
                expert_offsets,
                sorted_token_ids,
                num_experts,
                n_out,
                k,
                max_m_tiles.div_ceil(4).max(1),
                stream,
            );
        }
        crate::layers::ops::moe_w4a16_grouped_gemm_ptrtable(
            gpu,
            self.grouped_gemm_kernel(),
            a,
            packed_ptrs,
            scale_ptrs,
            scale2_vals,
            c,
            expert_offsets,
            sorted_token_ids,
            num_experts,
            n_out,
            k,
            max_m_tiles,
            stream,
        )
    }
}
