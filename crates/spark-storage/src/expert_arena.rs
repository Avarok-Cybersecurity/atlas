// SPDX-License-Identifier: AGPL-3.0-only
//
// The UMA zero-copy expert arena — the one genuinely-new primitive of the
// streaming-experts feature.
//
// A pinned LPDDR buffer (`cuMemAllocHost`) organized as a ring of `num_slabs`
// slabs, each holding `slots_per_slab` fixed-stride expert records. On GB10 the
// pinned allocation is GPU-addressable at the *same* virtual address (Gate
// 0(b)), so a record read straight into a slot — by O_DIRECT NVMe today, by
// one-sided RDMA_READ later — is immediately consumable by the fused MoE
// kernels with no `cuMemcpyHtoD` bounce: the expert pointer table is simply
// patched to point at the slot's device VA.
//
// This module owns the memory + geometry only. Who *fills* a slot (NVMe vs
// RDMA) is an `ExpertTier` concern; the arena is registrable as an RDMA MR
// (Stage 4) exactly because it is ordinary page-locked host memory.

use anyhow::{Context, Result, bail};
use std::ffi::c_void;

use crate::cuda_min::PinnedBuffer;

/// A ring of pinned per-layer slabs. Slot geometry is one expert record
/// (`record_stride`, a 4 KiB multiple so O_DIRECT / RDMA landings are aligned).
pub struct ExpertArena {
    pinned: PinnedBuffer,
    /// Device VA of the arena base. On GB10 this equals the host VA.
    dev_base: u64,
    num_slabs: u32,
    slots_per_slab: u32,
    record_stride: usize,
}

impl ExpertArena {
    /// Allocate `num_slabs * slots_per_slab * record_stride` bytes of pinned
    /// LPDDR and confirm the GB10 same-VA property (fail loudly otherwise — the
    /// zero-copy patch would be silently wrong on a host without it).
    pub fn new(num_slabs: u32, slots_per_slab: u32, record_stride: usize) -> Result<Self> {
        if num_slabs == 0 || slots_per_slab == 0 || record_stride == 0 {
            bail!("ExpertArena: zero geometry ({num_slabs},{slots_per_slab},{record_stride})");
        }
        if !record_stride.is_multiple_of(4096) {
            bail!("ExpertArena: record_stride {record_stride} must be a 4 KiB multiple (O_DIRECT)");
        }
        let total = (num_slabs as usize)
            .checked_mul(slots_per_slab as usize)
            .and_then(|v| v.checked_mul(record_stride))
            .context("ExpertArena: size overflow")?;
        let pinned = PinnedBuffer::new(total)?;
        let dev_base = pinned.device_ptr()?;
        let host_base = pinned.ptr as u64;
        if dev_base != host_base {
            // Not fatal to correctness on a mapped device, but it means the
            // ptr-table patch must use dev_base (not the host VA) and the
            // zero-copy assumption behind our bandwidth model is weaker. On
            // GB10 they are equal; assert so a non-UMA host is caught early.
            bail!(
                "ExpertArena: pinned host VA {host_base:#x} != device VA {dev_base:#x} \
                 — host is not unified-addressing (UMA zero-copy unavailable)"
            );
        }
        Ok(Self {
            pinned,
            dev_base,
            num_slabs,
            slots_per_slab,
            record_stride,
        })
    }

    pub fn num_slabs(&self) -> u32 {
        self.num_slabs
    }
    pub fn slots_per_slab(&self) -> u32 {
        self.slots_per_slab
    }
    pub fn record_stride(&self) -> usize {
        self.record_stride
    }

    fn linear_slot(&self, slab: u32, slot: u32) -> Result<usize> {
        if slab >= self.num_slabs || slot >= self.slots_per_slab {
            bail!(
                "ExpertArena: slot ({slab},{slot}) out of range ({},{})",
                self.num_slabs,
                self.slots_per_slab
            );
        }
        Ok((slab as usize) * (self.slots_per_slab as usize) + (slot as usize))
    }

    /// Host pointer of a slot — the O_DIRECT / RDMA landing target.
    pub fn slot_host_ptr(&self, slab: u32, slot: u32) -> Result<*mut u8> {
        let i = self.linear_slot(slab, slot)?;
        // SAFETY: i < num_slabs*slots_per_slab, so the offset is within the
        // single pinned allocation.
        Ok(unsafe { (self.pinned.ptr as *mut u8).add(i * self.record_stride) })
    }

    /// Device VA of a slot — what the expert pointer table is patched to.
    pub fn slot_dev_va(&self, slab: u32, slot: u32) -> Result<u64> {
        let i = self.linear_slot(slab, slot)?;
        Ok(self.dev_base + (i as u64) * (self.record_stride as u64))
    }

    /// Raw pinned base as a `*mut c_void` (for future `ibv_reg_mr`).
    pub fn base_ptr(&self) -> *mut c_void {
        self.pinned.ptr
    }
    pub fn total_bytes(&self) -> usize {
        self.pinned.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Pure-geometry checks that don't need a GPU are impossible here (the ctor
    // allocates pinned CUDA memory), so these are GPU-gated.
    #[test]
    #[ignore = "requires GPU"]
    fn arena_slots_are_strided_and_same_va() {
        let _ctx = crate::cuda_min::CudaCtx::new(0).unwrap();
        let stride = 8192; // 2x4KiB
        let arena = ExpertArena::new(2, 3, stride).unwrap();
        assert_eq!(arena.total_bytes(), 2 * 3 * stride);
        // Slot device VAs are contiguous and stride-spaced.
        let a = arena.slot_dev_va(0, 0).unwrap();
        let b = arena.slot_dev_va(0, 1).unwrap();
        let c = arena.slot_dev_va(1, 0).unwrap();
        assert_eq!(b - a, stride as u64);
        assert_eq!(c - a, 3 * stride as u64);
        // Same-VA property already asserted in the ctor; re-confirm host==dev.
        assert_eq!(
            arena.slot_host_ptr(1, 2).unwrap() as u64,
            arena.slot_dev_va(1, 2).unwrap()
        );
        // Out-of-range is an error, not a panic.
        assert!(arena.slot_dev_va(2, 0).is_err());
    }
}
