// SPDX-License-Identifier: AGPL-3.0-only

//! DETERMINISTIC epilogue of the native EXL3 MoE prefill tier.
//!
//! ## The defect this replaces
//!
//! Upstream's fused `exl3_moe` kernel ends every expert with an fp32
//! `atomicAdd` of that expert's weighted output row into the token's ONE
//! shared accumulator row (`hadamard_inner.cuh::had_hf_r_128_d_inner`), and
//! experts are handed to the ~6 concurrent expert-groups by a DYNAMIC ticket
//! draw (`exl3_moe_kernel.cuh`, `sched[2 + group_idx] = num_groups +
//! atomicAdd(&sched[0], 1)`). fp32 addition is not associative, so the order
//! the contributions land in — and therefore the bits of every prefill hidden
//! state — changes run to run. Measured on qwen4_exp at temp 0 with the
//! prefix cache off: 7 of 8 identical 89-token prompts produced DIFFERENT
//! prompt logprob vectors, and 2 of 6 identical 250-token completions
//! differed in text. The overflow (`count > 128`) tier's
//! `exl3_moe_scatter_add_f32` has the same defect on the same buffer, and it
//! is the tier that fires on long prefills.
//!
//! ## The contract, mirrored from DECODE
//!
//! The decode tier is bit-deterministic by construction: its mgemm reduces a
//! token's expert slots in a fixed `j = 0..stride-1` loop over per-slot fp32
//! partials. This module gives prefill the same shape:
//!
//! 1. every expert PLAIN-STORES its weighted row into its OWN sorted slot of
//!    a `[t_cap * top_k, hidden]` fp32 slab — one writer per slot row, no
//!    atomics, no ordering to get wrong (fused kernel: the `output_slots`
//!    argument; overflow tier: [`exl3_moe_store_slots_f32`]);
//! 2. [`exl3_moe_reduce_slots_f32`] then sums each token's `top_k` slots in
//!    ASCENDING FLAT-SLOT ORDER into the fp32 accumulator.
//!
//! Same addends, same fp32 arithmetic, ONE order. The serialization
//! experiment that proved the diagnosis (`num_groups = 1`) is NOT this: it
//! cost +88% TTFT because it gave up expert concurrency, whereas this keeps
//! the dynamic scheduler untouched and pays only one extra pass over the
//! routed rows.
//!
//! ## Why no zero-init
//!
//! Every LOCAL slot is written exactly once per call — by the fused kernel
//! (`0 < count <= 128`) or by the overflow tier (`count > 128`); an expert
//! with `count == 0` owns no slot. EP-remote slots land in the sentinel tail
//! and are never written, and [`slot_of_flat`] is exactly the predicate that
//! skips them. So the slab needs no memset, and the accumulator's memset is
//! skipped too because the reduce overwrites every element of it.

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::kernel_args::{KernelLaunch, div_ceil};

/// The resolved decision. `None` until the command line publishes one; the
/// fallback is the SAFE arm (deterministic), never the environment — house
/// rule, and a nondeterministic prefill is not something a stray variable
/// should be able to turn on.
static DET_PREFILL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();

/// Publish `--deterministic-moe-prefill`. Returns the value IN FORCE, which
/// differs from `enabled` when something already resolved the cell (then the
/// command line did NOT take effect and the caller warns — the
/// `gdn_flags::set_from_cli` / `set_prefill_varlen_from_cli` precedent).
pub fn set_exl3_det_moe_prefill_from_cli(enabled: bool) -> bool {
    let _ = DET_PREFILL.set(enabled);
    *DET_PREFILL.get().expect("just set")
}

/// Whether the native EXL3 MoE PREFILL tier runs its deterministic epilogue.
///
/// DEFAULT ON. `--deterministic-moe-prefill false` is the kill switch: it
/// restores upstream's atomicAdd epilogue (and skips the slot slab entirely,
/// so the memory comes back too) for A/B work against the numbers this
/// module changes. Read ONCE, when the MoE state sizes its slabs at load —
/// nothing on the hot path consults it, the presence of the slab does.
pub fn exl3_det_moe_prefill_enabled() -> bool {
    *DET_PREFILL.get_or_init(|| true)
}

/// LOCAL-sorted slot of a flat `(token, k)` slot, or `None` when the slot is
/// EP-REMOTE.
///
/// THE host-side definition of the rotation `exl3_moe_stage_sorted` applies
/// when it lays the sorted slots out, and the reduce kernel applies again to
/// find them: sorted positions `[lo, hi)` are exactly the EP-local slots and
/// are rotated to the front, every remote position lands in the tail at
/// `>= hi - lo`. Non-EP degenerates to the identity (`lo = 0`, `hi = s`), and
/// then nothing is ever remote.
///
/// Pure and tested here so the three spellings of one rule (staging kernel,
/// reduce kernel, this) cannot drift apart silently: a wrong mapping would
/// sum somebody else's expert row into a token, which is a plausible-looking
/// wrong answer rather than a crash.
pub fn slot_of_flat(perm_pos: usize, lo: usize, hi: usize) -> Option<usize> {
    let nloc = hi - lo;
    let dst = if perm_pos >= lo && perm_pos < hi {
        perm_pos - lo
    } else if perm_pos < lo {
        perm_pos + nloc
    } else {
        perm_pos
    };
    (dst < nloc).then_some(dst)
}

/// Overflow-tier deterministic epilogue: write `m` weighted rows to their own
/// slots (`out_slots` already offset to the chunk's slot base) instead of
/// atomically accumulating them into the shared per-token rows.
pub fn exl3_moe_store_slots_f32(
    gpu: &dyn GpuBackend,
    down_f32: DevicePtr,
    weight_sorted_base: DevicePtr,
    out_slots: DevicePtr,
    m: usize,
    hidden: usize,
    stream: u64,
) -> Result<()> {
    let h = gpu.kernel("exl3_matmul", "exl3_moe_store_slots_f32")?;
    let total = m * hidden;
    let grid = div_ceil(total as u32, 256).clamp(1, 4096);
    KernelLaunch::new(gpu, h)
        .grid([grid, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(down_f32)
        .arg_ptr(weight_sorted_base)
        .arg_ptr(out_slots)
        .arg_u64(hidden as u64)
        .arg_u64(total as u64)
        .launch(stream)
}

/// Step 6 of `exl3_moe_prefill_routed` on the DETERMINISTIC arm; a no-op on
/// the atomic arm, where the epilogue has already accumulated into `out_f32`.
/// Branching here rather than at the call site keeps the two arms' contract
/// ("`out_f32` holds the weighted routed sums") in one place.
#[allow(clippy::too_many_arguments)]
pub(super) fn reduce_slots_if_deterministic(
    gpu: &dyn GpuBackend,
    scratch: &super::Exl3MoePrefillScratch,
    token_to_perm: DevicePtr,
    expert_offsets: DevicePtr,
    local_start: usize,
    num_local: usize,
    top_k: usize,
    t: usize,
    hidden: usize,
    stream: u64,
) -> Result<()> {
    let Some(slots) = scratch.slot_f32 else {
        return Ok(());
    };
    exl3_moe_reduce_slots_f32(
        gpu,
        slots,
        token_to_perm,
        expert_offsets,
        scratch.out_f32,
        local_start,
        num_local,
        top_k,
        t,
        hidden,
        stream,
    )
}

/// Reduce each token's `top_k` per-slot rows into the fp32 accumulator in
/// fixed flat-slot order. Writes every element of `out[0 .. t*hidden]`, so
/// the accumulator needs no memset on this arm.
#[allow(clippy::too_many_arguments)]
pub fn exl3_moe_reduce_slots_f32(
    gpu: &dyn GpuBackend,
    slots: DevicePtr,
    token_to_perm: DevicePtr,
    expert_offsets: DevicePtr,
    out_f32: DevicePtr,
    local_start: usize,
    num_local: usize,
    top_k: usize,
    t: usize,
    hidden: usize,
    stream: u64,
) -> Result<()> {
    let h = gpu.kernel("exl3_matmul", "exl3_moe_reduce_slots_f32")?;
    let total = t * hidden;
    let grid = div_ceil(total as u32, 256).clamp(1, 4096);
    KernelLaunch::new(gpu, h)
        .grid([grid, 1, 1])
        .block([256, 1, 1])
        .arg_ptr(slots)
        .arg_ptr(token_to_perm)
        .arg_ptr(expert_offsets)
        .arg_ptr(out_f32)
        .arg_i32(local_start as i32)
        .arg_i32(num_local as i32)
        .arg_i32(top_k as i32)
        .arg_u64(hidden as u64)
        .arg_u64(total as u64)
        .launch(stream)
}

#[cfg(test)]
mod tests {
    use super::{exl3_det_moe_prefill_enabled, slot_of_flat};

    /// Non-EP (`lo = 0`, `hi = s`): the mapping is the identity and NOTHING
    /// is remote — a dropped slot here would silently lose one expert's
    /// contribution from a token.
    #[test]
    fn non_ep_is_the_identity_and_never_remote() {
        let s = 40;
        for p in 0..s {
            assert_eq!(slot_of_flat(p, 0, s), Some(p), "p={p}");
        }
    }

    /// EP: the local run `[lo, hi)` rotates to the front IN ORDER, and every
    /// remote position (before or after the run) is reported remote. The
    /// local images must be exactly `0..nloc` with no repeats — a collision
    /// would make two experts share one slot row and one of them would be
    /// overwritten rather than summed.
    #[test]
    fn ep_rotation_is_a_bijection_onto_the_local_prefix() {
        let (s, lo, hi) = (40usize, 12usize, 28usize);
        let nloc = hi - lo;
        let mut seen = vec![false; nloc];
        for p in 0..s {
            match slot_of_flat(p, lo, hi) {
                Some(d) => {
                    assert!(p >= lo && p < hi, "remote p={p} mapped local");
                    assert_eq!(d, p - lo);
                    assert!(!seen[d], "slot {d} claimed twice");
                    seen[d] = true;
                }
                None => assert!(p < lo || p >= hi, "local p={p} mapped remote"),
            }
        }
        assert!(seen.into_iter().all(|v| v), "local prefix not covered");
    }

    /// All-remote EP shard (`lo == hi`): every slot is remote, so the reduce
    /// contributes nothing and the token row stays an exact 0.0 — the EP
    /// partial-sum convention the prefill pipeline documents.
    #[test]
    fn an_empty_local_range_makes_every_slot_remote() {
        for p in 0..16 {
            assert_eq!(slot_of_flat(p, 7, 7), None, "p={p}");
        }
    }

    /// The DEFAULT is deterministic. Nothing in the environment can flip it;
    /// only `--deterministic-moe-prefill false` can, and only before anything
    /// has read the cell.
    #[test]
    fn determinism_is_the_default() {
        assert!(exl3_det_moe_prefill_enabled());
    }
}
