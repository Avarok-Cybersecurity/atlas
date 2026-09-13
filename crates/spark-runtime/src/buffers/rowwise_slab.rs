// SPDX-License-Identifier: AGPL-3.0-only

//! The row-wise FP8 GDN prefill BF16-weight slab: its size accessor and the
//! bump carve the prefill arms take their per-layer slices from. Split from
//! `accessors.rs` (≤500 LoC cap) rather than wedged into it, because the carve
//! is a tiny allocator and not a getter.
//!
//! **WHY IT EXISTS (#917 H100 receipt, 2026-09-11).** The `ATLAS_FP8_ROWWISE`
//! GDN arms used to get their BF16 weight from a lazy `gpu.alloc` memoised by
//! weight pointer — `167772160` B for the fused `[QKV|Z]` weight per layer,
//! with no `BufferSizes` entry, so `--gpu-memory-utilization` could not see it
//! and a 28-token prefill died at layer 36 with `cuMemAlloc_v2 failed:
//! status 2`. The bytes are the same; the difference is that they are now ONE
//! ledgered allocation the preflight fitter prices.

use super::BufferArena;
use crate::gpu::DevicePtr;

impl BufferArena {
    /// Allocated byte size of the row-wise GDN prefill BF16-weight slab.
    /// 0 when `ATLAS_FP8_ROWWISE` was not armed at boot.
    pub fn ssm_rowwise_w_bf16_bytes(&self) -> usize {
        self.sizes.ssm_rowwise_w_bf16
    }

    /// Carve the next `bytes` of the row-wise GDN prefill BF16-weight slab.
    ///
    /// Bump-only and never returned: each GDN layer takes its `in_proj_qkvz`
    /// and `out_proj` slices on its FIRST prefill and holds them for the life
    /// of the arena, because a dequanted weight is as immutable as the weight
    /// it came from. Sized in `sizes_rowwise::ssm_rowwise_w_bf16_bytes` for
    /// exactly `num_ssm_layers` of those pairs, so exhaustion means the sizing
    /// and the callers disagree — a bug, reported as one rather than papered
    /// over with a fresh allocation (that is the #917 defect this replaces).
    ///
    /// `Relaxed` is enough: the scheduler drives one forward at a time (the
    /// same single-threaded invariant `cublaslt::Ctx` documents), so this is a
    /// counter that happens to be atomic rather than a contended one.
    pub fn take_ssm_rowwise_w_bf16(&self, bytes: usize) -> anyhow::Result<DevicePtr> {
        use std::sync::atomic::Ordering;
        if self.ssm_rowwise_w_bf16 == DevicePtr::NULL {
            anyhow::bail!(
                "row-wise GDN prefill BF16-weight slab is absent: the arena was sized without                  ATLAS_FP8_ROWWISE=1 but a row-wise prefill arm asked for {bytes} B. Both the                  loader's weight install and this ledger entry read the same lever, so they                  cannot legitimately disagree"
            );
        }
        let base = self
            .ssm_rowwise_w_bf16_used
            .fetch_add(bytes, Ordering::Relaxed);
        let end = base + bytes;
        if end > self.sizes.ssm_rowwise_w_bf16 {
            anyhow::bail!(
                "row-wise GDN prefill BF16-weight slab exhausted: wanted {bytes} B at offset                  {base}, slab is {} B. `sizes_rowwise::ssm_rowwise_w_bf16_bytes` sizes it for                  num_ssm_layers x (in_proj_qkvz + out_proj)",
                self.sizes.ssm_rowwise_w_bf16
            );
        }
        Ok(self.ssm_rowwise_w_bf16.offset(base))
    }
}
