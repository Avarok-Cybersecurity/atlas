// SPDX-License-Identifier: AGPL-3.0-only

//! How an expert's weights are addressed on the device.
//!
//! Split from `moe/mod.rs` on the 500-line cap. These four types are the
//! layer's vocabulary rather than its behaviour: three descriptions of where
//! expert weights live, and the enum that lets the forward path pick a fused
//! kernel by matching on the quantisation it actually landed in — instead of
//! inferring it, which is how a shared expert exempted from quantisation ends
//! up silently read as though it were not.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

use crate::weight_map::DenseWeight;

/// Device-side pointer table for one projection across all experts.
///
/// Enables GPU-side expert dispatch: the batched GEMV kernel reads
/// expert_id from device memory, then indexes these tables to find
/// the correct weight pointers — no CPU involvement needed.
pub(crate) struct ExpertPtrTable {
    /// `[num_experts]` u64 device pointers to each expert's B_packed.
    pub(crate) packed_ptrs: DevicePtr,
    /// `[num_experts]` u64 device pointers to each expert's B_scale.
    pub(crate) scale_ptrs: DevicePtr,
    /// `[num_experts]` f32 per-expert scale2 values.
    pub(crate) scale2_vals: DevicePtr,
}

/// Device-side pointer table for FP8 expert dispatch (one projection).
///
/// FP8 experts use 2 pointer arrays (weight + block_scale) instead of
/// NVFP4's 3 (packed + scale + scale2). The fused FP8 MoE kernel indexes
/// these tables by expert_id to load the correct FP8 weight matrix.
pub(crate) struct Fp8ExpertPtrTable {
    /// `[num_experts]` u64 device pointers to each expert's FP8 weight.
    pub(crate) weight_ptrs: DevicePtr,
    /// `[num_experts]` u64 device pointers to each expert's block scales.
    pub(crate) scale_ptrs: DevicePtr,
}

/// Checkpoint-native BF16 weights for a shared expert.
///
/// This is intentionally independent of routed-expert precision. Models such
/// as Laguna ship NVFP4 routed experts but explicitly exempt the shared expert
/// from quantization, so coupling these pointers to the all-BF16 routed path
/// silently changes model numerics.
#[derive(Debug, Clone, Copy)]
pub(crate) struct Bf16SharedExpert {
    // `pub(super)`, not private: these were reachable from every sibling while
    // the type lived in `mod.rs`, and the split must not narrow that.
    pub(super) gate_proj: DenseWeight,
    pub(super) up_proj: DenseWeight,
    pub(super) down_proj: DenseWeight,
}

impl Bf16SharedExpert {
    pub(super) fn new(
        gate_proj: DenseWeight,
        up_proj: DenseWeight,
        down_proj: DenseWeight,
    ) -> Result<Self> {
        anyhow::ensure!(
            !gate_proj.weight.is_null() && !up_proj.weight.is_null() && !down_proj.weight.is_null(),
            "BF16 shared expert requires non-null gate/up/down weights"
        );
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }
}

/// Device-side pointer table for native EXL3 (QTIP trellis) expert dispatch
/// — one projection across the EP-LOCAL experts (`ATLAS_EXL3_NATIVE_MOE=1`).
///
/// DENSE over the local range: entry `i` addresses global expert
/// `local_start + i`, and there are NO null entries. This is load-bearing
/// for correctness, not a compaction nicety: the mgemm grouped weighted
/// reduction skips a slot only when its `B_indices` value is NEGATIVE — a
/// NULL pointer-table entry reachable through a valid index skips compute
/// (`B == nullptr` guard) but still SUMS the slot's stale fp32 C scratch
/// into the token output (vendored `exl3_gemm_kernel.cuh`, reduction
/// epilogue). Remote experts under EP must therefore be encoded as `-1`
/// slot indices over this dense local table — see
/// [`exl3_expert_slot_index`] for the canonical mapping.
///
/// One table = one `(k_bits, cb)` kernel template; the builder
/// (`ptr_table_build.rs::build_exl3_ptr_table`) enforces uniformity.
#[allow(dead_code)] // consumed by the native MoE dispatch (forward arm)
#[derive(Debug)]
pub(crate) struct Exl3ExpertPtrTable {
    /// `[num_local]` u64 device pointers to each local expert's `.trellis`.
    pub(crate) trellis_ptrs: DevicePtr,
    /// `[num_local]` u64 device pointers to each local expert's `.suh`.
    pub(crate) suh_ptrs: DevicePtr,
    /// `[num_local]` u64 device pointers to each local expert's `.svh`.
    pub(crate) svh_ptrs: DevicePtr,
    /// `[num_local]` host copies of the same `[trellis, suh, svh]` raw
    /// device addresses the arrays above hold — the prefill OVERFLOW path
    /// (an expert routed > 128 rows in one batch) launches per-expert
    /// `exl3_gemm`s on that expert's trellis and needs its pointers without
    /// a D2H of the table.
    pub(crate) host_ptrs: Vec<[u64; 3]>,
    /// Local expert count (== the EP-local range width).
    pub(crate) num_local: usize,
    /// Global id of local expert 0 (`config.local_expert_range().0`).
    pub(crate) local_start: usize,
    /// Trellis bits/weight — every entry shares it (one mgemm template).
    pub(crate) k_bits: u32,
    /// Kernel codebook index: 1 = MCG, 2 = MUL1 (cb0 has no instances).
    pub(crate) cb: u32,
    /// Projection geometry (gate/up: `[hidden -> inter]`, down: the
    /// transpose). Every entry shares it.
    pub(crate) in_dim: usize,
    pub(crate) out_dim: usize,
}

impl Exl3ExpertPtrTable {
    /// Free the three device pointer arrays (NOT the expert tensors they
    /// point into — those live in the adopted WeightStore and are freed by
    /// its release, weights-last). Without an explicit caller this is
    /// reclaimed by `sweep_unreleased` at teardown, the same backstop the
    /// NVFP4 `ExpertPtrTable`s use.
    #[allow(dead_code)]
    pub(crate) fn release(&self, gpu: &dyn GpuBackend) -> Result<()> {
        gpu.free(self.trellis_ptrs)?;
        gpu.free(self.suh_ptrs)?;
        gpu.free(self.svh_ptrs)?;
        Ok(())
    }
}

/// Canonical GLOBAL-expert-id -> `b_indices` slot value mapping for the
/// dense local table: the local table index for a local expert, `-1` for a
/// remote one. The device-side routing kernel that stages `b_indices` from
/// the top-k output MUST implement exactly this mapping (this host helper is
/// the tested definition); the `-1` is what makes the mgemm reduction skip
/// remote slots (see [`Exl3ExpertPtrTable`]). Never use mgemm's
/// `min_index`/`max_index` filtering instead — it caps `bszm` at 128 and
/// routes through module-scope `__device__` globals.
#[allow(dead_code)] // referenced by the native MoE dispatch (forward arm)
pub(crate) fn exl3_expert_slot_index(
    global_id: usize,
    local_start: usize,
    num_local: usize,
) -> i64 {
    if global_id >= local_start && global_id < local_start + num_local {
        (global_id - local_start) as i64
    } else {
        -1
    }
}

/// Slot-batch cap for the native EXL3 MoE mgemm scratch, in TOKENS: launches
/// are chunked to `EXL3_MOE_SLOT_BATCH_TOKENS * top_k` (token,expert) slots
/// so the fp16/fp32 slabs stay ~138 MB (qwen4_exp shapes at 5,120 slots)
/// instead of the ~1.05 GB a full 4096-token prefill chunk would need.
/// Groups stay intact: `bszm` must be a multiple of `num_tokens`, and a
/// whole token's `top_k` slots always land in the same batch.
pub(crate) const EXL3_MOE_SLOT_BATCH_TOKENS: usize = 512;

/// Default token-batch cap of the PREFILL tier (`ATLAS_EXL3_MOE_PREFILL_
/// BATCH_TOKENS` overrides): one fused `exl3_moe` launch + at most one
/// host-sync per batch. 4096 covers the canonical prefill chunk in one
/// batch; larger prefill chunks are sorted and served per batch slice.
pub(crate) const EXL3_MOE_PREFILL_BATCH_TOKENS_DEFAULT: usize = 4096;

/// Row chunk of the prefill overflow (count > 128) dense GEMMs — fixed slab
/// rows, so an arbitrarily hot expert never scales the scratch.
pub(crate) const EXL3_MOE_OVERFLOW_CHUNK_ROWS: usize = 1024;

/// Construction-time launch state for native EXL3 expert mgemm calls —
/// ONE per model, shared by every MoE layer (all MoE mgemm launches are
/// serialized on the primary stream / the stacked prefill phase, so one
/// locks buffer and one slab set suffice; see the scope design's
/// concurrency section). Nothing on the hot path may alloc or sync (901
/// playbook); everything here is allocated once at load, before the KV
/// budget is computed, so it is inside the util pledge.
///
/// The `Exl3LmHead` owns its own locks buffer, so a co-dispatched head
/// projection can never race these MoE launches on the locks.
#[allow(dead_code)] // consumed by the native MoE dispatch (forward arm)
#[derive(Debug)]
pub(crate) struct Exl3MoeState {
    /// Per-model cooperative-launch locks (`ops::EXL3_LOCKS_BYTES`, zeroed
    /// once; kernels self-reset). Launch on the primary stream only; a new
    /// CONCURRENT caller needs its own locks buffer.
    pub(crate) locks: DevicePtr,
    /// fp16 activation ingress `[s_cap, hidden]` — one row per
    /// (token,expert) slot (activations replicated top_k-wide; `bszm_in=1`
    /// broadcast only covers the single-token case).
    pub(crate) a_f16: DevicePtr,
    /// fp16 `A_had` rotation scratch `[s_cap, hidden]`, SEPARATE from
    /// `a_f16`: the A_had-aliases-A sanction is upstream-verified for the
    /// single-matrix gemm only, not for mgemm's per-slot slabs. Also covers
    /// the down call's `[s_cap, inter]` need (inter < hidden).
    pub(crate) a_had_f16: DevicePtr,
    /// C slab for the gate mgemm, `[s_cap, inter]`, allocated at 4
    /// bytes/element. The decode pipeline launches gate/up with f16 C
    /// (upstream's tier writes fp16 gate/up, fp32 only for down) and uses the
    /// first half; the f32-sized allocation is deliberate headroom so an
    /// f32-C variant needs no resize (`Exl3MoeScratch::c_gate_f16` views it).
    pub(crate) c_gate_f32: DevicePtr,
    /// Same as `c_gate_f32`, for the up mgemm.
    pub(crate) c_up_f32: DevicePtr,
    /// fp32 C for the down mgemm, `[s_cap, hidden]` — full slot width so it
    /// doubles as the grouped-reduction scratch (the kernel writes per-slot
    /// partials before reducing into rows `0..num_tokens`).
    pub(crate) c_down_f32: DevicePtr,
    /// fp16 `silu(gate) * up` ingress for the down call, `[s_cap, inter]`.
    pub(crate) inter_f16: DevicePtr,
    /// `[s_cap]` i64 per-slot local-expert indices (`-1` = remote), the
    /// mgemm `b_indices` argument — see [`exl3_expert_slot_index`].
    pub(crate) b_indices: DevicePtr,
    /// Per-slot routing weights for the mgemm `b_weights` grouped-reduction
    /// argument. The kernel reads HALF values (upstream's fp16 routing
    /// weights); `exl3_moe_stage_routing` writes f16 into this slab, which
    /// is allocated 4 bytes/slot (2x headroom — an f32 variant needs no
    /// resize).
    pub(crate) b_weights: DevicePtr,
    /// Slot capacity: `EXL3_MOE_SLOT_BATCH_TOKENS * top_k`.
    pub(crate) s_cap: usize,
    pub(crate) hidden: usize,
    pub(crate) inter: usize,
    pub(crate) top_k: usize,
    /// Resolved once at construction (the GpuBackend trait forbids
    /// per-launch queries).
    pub(crate) sm_count: u32,

    // ── PREFILL tier (fused `exl3_moe` kernel + overflow path) ──
    /// f16 `[pf_t_cap, hidden]` RAW activation ingress (`hidden_state`).
    pub(crate) pf_hidden_f16: DevicePtr,
    /// f32 `[pf_t_cap, hidden]` routed accumulator (`output_state`).
    pub(crate) pf_out_f32: DevicePtr,
    /// f16 `[pf_concurrency, 128, hidden]` x2 staging slabs (no zero-init).
    pub(crate) pf_temp_state_g: DevicePtr,
    pub(crate) pf_temp_state_u: DevicePtr,
    /// f16 `[pf_concurrency, 128, inter]` x2 intermediate slabs.
    pub(crate) pf_temp_inter_g: DevicePtr,
    pub(crate) pf_temp_inter_u: DevicePtr,
    /// i64 / f16 `[pf_t_cap * top_k]` LOCAL-sorted slot maps.
    pub(crate) pf_token_sorted: DevicePtr,
    pub(crate) pf_weight_sorted: DevicePtr,
    /// i64 `[pf_e_cap + 1]` per-local-expert bincount + EP sentinel.
    pub(crate) pf_expert_count: DevicePtr,
    /// Overflow-path chunk slabs `[EXL3_MOE_OVERFLOW_CHUNK_ROWS, _]`: f16
    /// gathered A, f16 `A_had` (dedicated — gate and up share the A), f16
    /// gate / up C, f32 down C.
    pub(crate) pf_ov_a: DevicePtr,
    pub(crate) pf_ov_a_had: DevicePtr,
    pub(crate) pf_ov_gate: DevicePtr,
    pub(crate) pf_ov_up: DevicePtr,
    pub(crate) pf_ov_down: DevicePtr,
    /// Token-batch capacity of the prefill tier (callers slice above it).
    pub(crate) pf_t_cap: usize,
    /// Local-expert capacity of `pf_expert_count` (= num_experts at build).
    pub(crate) pf_e_cap: usize,
    /// Fused-kernel temp-slab count C (`sm_count / 8`, clamped to 1..=64).
    pub(crate) pf_concurrency: usize,

    // ── Runtime enforcement of the sharing contract ──
    /// Set for the duration of one layer's dispatch section (host side). The
    /// slabs, `b_indices` and the locks buffer are shared by every layer and
    /// both tiers, and the fused/cooperative kernels need full SM
    /// co-residency — a second dispatcher overlapping this one (another
    /// thread, a side-forward) would corrupt state or deadlock, so it is
    /// refused loudly instead.
    pub(crate) in_flight: std::sync::atomic::AtomicBool,
    /// The stream the most recent dispatch section launched on (0 = none
    /// yet). Atlas runs prefill and decode on DIFFERENT streams that may
    /// overlap on the device, so a section on a new stream first waits on
    /// [`Self::fence`] — the device-side end marker of the previous section —
    /// before touching the shared state. Same-stream sections are already
    /// stream-ordered and skip the wait.
    pub(crate) dispatch_stream: std::sync::atomic::AtomicU64,
    /// CUDA event recorded on the dispatching stream at the end of every
    /// section (RAII, in the guard's drop). GPU-side only — the host never
    /// blocks on it.
    pub(crate) fence: u64,
}

/// RAII token for one dispatch section — see [`Exl3MoeState::dispatch_guard`].
/// Dropping it records the fence on the section's stream and releases the
/// host-side claim.
pub(crate) struct Exl3MoeDispatchGuard<'a> {
    st: &'a Exl3MoeState,
    gpu: &'a dyn GpuBackend,
    stream: u64,
}

impl Drop for Exl3MoeDispatchGuard<'_> {
    fn drop(&mut self) {
        if let Err(e) = self.gpu.record_event(self.st.fence, self.stream) {
            // Cannot propagate from Drop; the next cross-stream section would
            // then wait on a stale fence, so make the failure visible.
            tracing::error!(
                "EXL3 native MoE: fence record failed on stream {:#x}: {e}",
                self.stream
            );
        }
        self.st
            .in_flight
            .store(false, std::sync::atomic::Ordering::Release);
    }
}

impl Exl3MoeState {
    /// Claim the shared state for one layer's dispatch section on `stream`.
    /// Refuses (does not wait) if another section is in flight on the host —
    /// a contract breach, never a normal condition. A stream change is
    /// normal (prefill vs decode streams) and is made safe by ordering this
    /// stream behind the previous section's fence on the device.
    pub(crate) fn dispatch_guard<'a>(
        &'a self,
        gpu: &'a dyn GpuBackend,
        stream: u64,
    ) -> Result<Exl3MoeDispatchGuard<'a>> {
        use std::sync::atomic::Ordering;
        anyhow::ensure!(
            self.in_flight
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok(),
            "EXL3 native MoE: a second dispatch overlapped one in flight — the \
             shared slabs/locks admit ONE dispatcher at a time (a concurrent \
             caller needs its own Exl3MoeState)"
        );
        let guard = Exl3MoeDispatchGuard {
            st: self,
            gpu,
            stream,
        };
        let prev = self.dispatch_stream.swap(stream, Ordering::AcqRel);
        if prev != 0 && prev != stream {
            gpu.stream_wait_event(stream, self.fence)?;
        }
        Ok(guard)
    }
}

impl Exl3MoeState {
    /// Assemble the ops-level prefill scratch view over this state's slabs.
    pub(crate) fn prefill_scratch(&self) -> crate::layers::ops::Exl3MoePrefillScratch {
        crate::layers::ops::Exl3MoePrefillScratch {
            hidden_f16: self.pf_hidden_f16,
            out_f32: self.pf_out_f32,
            temp_state_g: self.pf_temp_state_g,
            temp_state_u: self.pf_temp_state_u,
            temp_inter_g: self.pf_temp_inter_g,
            temp_inter_u: self.pf_temp_inter_u,
            token_sorted: self.pf_token_sorted,
            weight_sorted: self.pf_weight_sorted,
            expert_count: self.pf_expert_count,
            ov_a_f16: self.pf_ov_a,
            ov_a_had_f16: self.pf_ov_a_had,
            ov_gate_f16: self.pf_ov_gate,
            ov_up_f16: self.pf_ov_up,
            ov_down_f32: self.pf_ov_down,
            t_cap: self.pf_t_cap,
            e_cap: self.pf_e_cap,
            concurrency: self.pf_concurrency,
            ov_chunk: EXL3_MOE_OVERFLOW_CHUNK_ROWS,
        }
    }
}

impl Exl3MoeState {
    /// Free the locks + slabs. Without an explicit caller this is reclaimed
    /// by `sweep_unreleased` at teardown (documented backstop).
    #[allow(dead_code)]
    pub(crate) fn release(&self, gpu: &dyn GpuBackend) -> Result<()> {
        for p in [
            self.locks,
            self.a_f16,
            self.a_had_f16,
            self.c_gate_f32,
            self.c_up_f32,
            self.c_down_f32,
            self.inter_f16,
            self.b_indices,
            self.b_weights,
            self.pf_hidden_f16,
            self.pf_out_f32,
            self.pf_temp_state_g,
            self.pf_temp_state_u,
            self.pf_temp_inter_g,
            self.pf_temp_inter_u,
            self.pf_token_sorted,
            self.pf_weight_sorted,
            self.pf_expert_count,
            self.pf_ov_a,
            self.pf_ov_a_had,
            self.pf_ov_gate,
            self.pf_ov_up,
            self.pf_ov_down,
        ] {
            gpu.free(p)?;
        }
        gpu.destroy_event(self.fence)?;
        Ok(())
    }
}

/// Unified expert pointer table for any quantization format.
///
/// Replaces the separate `ExpertPtrTable` (NVFP4) and `Fp8ExpertPtrTable` (FP8)
/// with a single enum. The MoE forward path matches on this to select the
/// correct fused kernel (moe_shared_expert_fused vs moe_shared_expert_fused_fp8).
#[allow(dead_code)]
pub(crate) enum ExpertPtrSet {
    /// NVFP4: 3 pointer arrays (packed_ptrs, scale_ptrs, per-expert scale2 f32).
    Nvfp4 {
        packed_ptrs: DevicePtr,
        scale_ptrs: DevicePtr,
        scale2_vals: DevicePtr,
    },
    /// FP8: 2 pointer arrays (weight_ptrs, block_scale_ptrs).
    Fp8 {
        weight_ptrs: DevicePtr,
        scale_ptrs: DevicePtr,
    },
}
