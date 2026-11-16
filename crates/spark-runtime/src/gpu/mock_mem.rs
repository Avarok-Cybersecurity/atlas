// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: the mock backend's simulated device memory lookups and the
//! device-to-device byte mover. A child of `mock`, so the allocation table stays
//! private to the mock; moved out of `mock.rs` unchanged when the port sprint's
//! extension overrides would have taken that file past the 500-line cap.

use std::collections::HashMap;

use anyhow::Result;

use super::{MockAlloc, MockGpuBackend};
use crate::gpu::DevicePtr;

impl MockGpuBackend {
    /// `bytes` from `src` to `dst` inside the simulated device memory.
    ///
    /// Real byte movement, not a no-op: a D2D that silently succeeds without
    /// moving anything lets a test "pass" while asserting the destination is
    /// still zero — the exact shape of a rollback bug this backend exists to
    /// catch. Source is staged through a temporary so `src` and `dst` may sit
    /// in the same allocation (the borrow checker would otherwise reject it,
    /// and the real `cudaMemcpyAsync` accepts it for non-overlapping ranges).
    pub(super) fn blit(&self, src: DevicePtr, dst: DevicePtr, bytes: usize) -> Result<()> {
        if bytes == 0 {
            return Ok(());
        }
        let mut allocs = self.allocs.lock();
        let staged = {
            let (offset, alloc) = find_alloc(&allocs, src)
                .ok_or_else(|| anyhow::anyhow!("copy_d2d: src {src} not allocated"))?;
            if offset + bytes > alloc.bytes {
                anyhow::bail!("copy_d2d: src {src} + {bytes} overruns its allocation");
            }
            alloc.data[offset..offset + bytes].to_vec()
        };
        let (offset, alloc) = find_alloc_mut(&mut allocs, dst)
            .ok_or_else(|| anyhow::anyhow!("copy_d2d: dst {dst} not allocated"))?;
        if offset + bytes > alloc.bytes {
            anyhow::bail!("copy_d2d: dst {dst} + {bytes} overruns its allocation");
        }
        alloc.data[offset..offset + bytes].copy_from_slice(&staged);
        Ok(())
    }
}

/// Find the allocation containing `ptr` (supports offset pointers).
pub(super) fn find_alloc(
    allocs: &HashMap<u64, MockAlloc>,
    ptr: DevicePtr,
) -> Option<(usize, &MockAlloc)> {
    for (&base, alloc) in allocs.iter() {
        if ptr.0 >= base && ptr.0 < base + alloc.bytes as u64 {
            return Some(((ptr.0 - base) as usize, alloc));
        }
    }
    None
}

/// Mutable version of find_alloc.
pub(super) fn find_alloc_mut(
    allocs: &mut HashMap<u64, MockAlloc>,
    ptr: DevicePtr,
) -> Option<(usize, &mut MockAlloc)> {
    for (&base, alloc) in allocs.iter_mut() {
        if ptr.0 >= base && ptr.0 < base + alloc.bytes as u64 {
            return Some(((ptr.0 - base) as usize, alloc));
        }
    }
    None
}
