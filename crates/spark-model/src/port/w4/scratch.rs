// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: how much scratch the [`W4Linear`] calls of one layer need
//! ([`ScratchSpec`]), and the shared device regions that serve them.
//!
//! # One set of regions per backend
//!
//! The regions live in the backend's [`OpCache`](spark_runtime::op_cache::OpCache)
//! under the `port.w4.*` tags: grow-only, released with the backend. Every W4
//! consumer (FFN, GDN, attention, LM head) on that backend shares them, which
//! is sound for the reason the `BufferArena` is: the layers run one after
//! another on the model's stream, so one call's use of the scratch is ordered
//! before the next call's. A per-layer copy would cost the 4B 36x the memory
//! for no gain.
//!
//! # Reserved at load
//!
//! [`acquire`] is called once per layer at load (the dense FFN does it from
//! `finalize_q4k_load`, which the loader already runs before the KV cache is
//! sized), so the regions exist before any CUDA graph capture. The caller
//! keeps the returned [`W4Scratch`]: a later, larger reservation by another
//! layer allocates a new region and leaves the earlier one alive (the cache
//! never frees inside a model's life), so a kept view stays valid and big
//! enough for its own layer.
//!
//! # Sizes for the 4B (hidden 2560, intermediate 9216, affine4 g64)
//!
//! `xq` = 2048 rows x 9216 = 18.9 MB, `xs` = 2048 x 9216 / 16 x 4 = 4.7 MB,
//! `bf16` (the decline fallback, one matrix at a time) = 9216 x 2560 x 2 =
//! 47.2 MB, tickets 64 KiB, plus the largest `scratch_bytes` the weights
//! report. Reserved once per backend, not per layer.

use super::{W4Linear, W4Scratch, select};
use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};

/// `u32` split-K ticket counters in [`W4Scratch::tickets`].
pub const TICKET_SLOTS: usize = 16 * 1024;
/// Bytes of the ticket region.
pub const TICKET_BYTES: usize = TICKET_SLOTS * 4;
/// Rows of one prefill chunk the `xq`/`xs` regions are sized for (raised per
/// weight by [`select::prefill_chunk_cap`] when its decode window is wide).
pub const PREFILL_ROWS: usize = 2048;
/// Inputs per `f32` of [`W4Scratch::xs`].
pub const XS_GROUP: usize = 16;
/// Every region is a multiple of this (and at least this), so the cache is
/// never asked for zero bytes.
pub const ALIGN: usize = 256;

const TAG_WS: &str = "port.w4.ws";
const TAG_TICKETS: &str = "port.w4.tickets";
const TAG_XQ: &str = "port.w4.xq";
const TAG_XS: &str = "port.w4.xs";
const TAG_BF16: &str = "port.w4.bf16";
const ONCE_TICKETS_ZEROED: &str = "port.w4.tickets.zeroed";

/// Bytes of int8 activations for `rows` rows of `k` inputs.
pub fn xq_bytes(rows: usize, k: usize) -> usize {
    rows * k
}

/// Bytes of per-16-input `f32` activation sums or scales for `rows` rows.
pub fn xs_bytes(rows: usize, k: usize) -> usize {
    rows * k.div_ceil(XS_GROUP) * 4
}

fn align(bytes: usize) -> usize {
    bytes.max(1).div_ceil(ALIGN) * ALIGN
}

/// The scratch one or more weights need, field by field.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct ScratchSpec {
    pub ws_bytes: usize,
    pub xq_bytes: usize,
    pub xs_bytes: usize,
    /// The BF16 `[n, k]` dequant target of the decline fallback.
    pub bf16_bytes: usize,
}

impl ScratchSpec {
    /// One weight: its decode window and its largest prefill chunk.
    pub fn for_linear(w: &dyn W4Linear) -> Self {
        let max_rows = w.max_decode_rows();
        let rows = select::prefill_chunk_cap(max_rows).max(max_rows);
        Self {
            ws_bytes: w.scratch_bytes(max_rows).max(w.scratch_bytes(rows)),
            xq_bytes: xq_bytes(rows, w.k()),
            xs_bytes: xs_bytes(rows, w.k()),
            bf16_bytes: w.n() * w.k() * 2,
        }
    }

    /// Field-wise maximum: the scratch that serves both.
    pub fn union(self, o: Self) -> Self {
        Self {
            ws_bytes: self.ws_bytes.max(o.ws_bytes),
            xq_bytes: self.xq_bytes.max(o.xq_bytes),
            xs_bytes: self.xs_bytes.max(o.xs_bytes),
            bf16_bytes: self.bf16_bytes.max(o.bf16_bytes),
        }
    }

    /// The union over several weights.
    pub fn for_all(ws: &[&dyn W4Linear]) -> Self {
        ws.iter()
            .fold(Self::default(), |acc, w| acc.union(Self::for_linear(*w)))
    }

    /// Device bytes [`acquire`] reserves for this spec, tickets included
    /// (an upper bound on what it adds: the regions are shared).
    pub fn device_bytes(&self) -> usize {
        align(self.ws_bytes)
            + TICKET_BYTES
            + align(self.xq_bytes)
            + align(self.xs_bytes)
            + align(self.bf16_bytes)
    }
}

/// Reserve (or find) the shared regions for `spec` and return the view a
/// [`W4Linear`] call takes. Allocates only when a region is missing or too
/// small, so call it at load, outside graph capture. The ticket region is
/// zeroed once per backend, on `stream`, the first time it is created; its
/// size is fixed, so it is never re-created.
pub fn acquire(gpu: &dyn GpuBackend, spec: &ScratchSpec, stream: u64) -> Result<W4Scratch> {
    let cache = gpu.op_cache();
    let ws_bytes = align(spec.ws_bytes);
    let ws = cache.scratch(gpu, TAG_WS, ws_bytes)?;
    let tickets = cache.scratch(gpu, TAG_TICKETS, TICKET_BYTES)?;
    if cache.once(ONCE_TICKETS_ZEROED) {
        gpu.memset_async(tickets, 0, TICKET_BYTES, stream)?;
    }
    let xq = cache.scratch(gpu, TAG_XQ, align(spec.xq_bytes))?;
    let xs = cache.scratch(gpu, TAG_XS, align(spec.xs_bytes))?;
    // The fallback target is reserved now too, so the KV cache never claims
    // the memory a declined prefill needs later.
    bf16_dequant(gpu, spec)?;
    Ok(W4Scratch {
        ws,
        ws_bytes,
        tickets,
        xq,
        xs,
    })
}

/// The shared BF16 dequant target of the decline fallback, at least
/// `spec.bf16_bytes`. Used one matrix at a time on the model's stream.
pub fn bf16_dequant(gpu: &dyn GpuBackend, spec: &ScratchSpec) -> Result<DevicePtr> {
    gpu.op_cache()
        .scratch(gpu, TAG_BF16, align(spec.bf16_bytes))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::port::w4::fake::{FakeW4, log};
    use spark_runtime::gpu::mock::MockGpuBackend;

    #[test]
    fn the_4b_ffn_spec_matches_the_documented_sizes() {
        let l = log();
        let gate = FakeW4::new("gate", 9216, 2560, 16, &l);
        let down = FakeW4::new("down", 2560, 9216, 16, &l);
        let s = ScratchSpec::for_all(&[&gate, &gate, &down]);
        assert_eq!(s.xq_bytes, 2048 * 9216);
        assert_eq!(s.xs_bytes, 2048 * (9216 / 16) * 4);
        assert_eq!(s.bf16_bytes, 9216 * 2560 * 2);
        assert_eq!(
            s.ws_bytes,
            gate.scratch_bytes(2048).max(down.scratch_bytes(2048))
        );
        assert!(s.device_bytes() >= s.xq_bytes + s.xs_bytes + s.bf16_bytes + TICKET_BYTES);
    }

    #[test]
    fn union_is_field_wise() {
        let a = ScratchSpec {
            ws_bytes: 1,
            xq_bytes: 9,
            xs_bytes: 3,
            bf16_bytes: 4,
        };
        let b = ScratchSpec {
            ws_bytes: 5,
            xq_bytes: 2,
            xs_bytes: 7,
            bf16_bytes: 0,
        };
        let u = a.union(b);
        assert_eq!(
            (u.ws_bytes, u.xq_bytes, u.xs_bytes, u.bf16_bytes),
            (5, 9, 7, 4)
        );
        assert_eq!(ScratchSpec::for_all(&[]), ScratchSpec::default());
    }

    #[test]
    fn a_wide_decode_window_widens_the_chunk_rows() {
        let l = log();
        let w = FakeW4::new("w", 64, 1024, 1000, &l);
        let s = ScratchSpec::for_linear(&w);
        assert_eq!(s.xq_bytes, select::prefill_chunk_cap(1000) * 1024);
    }

    #[test]
    fn acquire_reuses_regions_and_zeroes_tickets_once() {
        let gpu = MockGpuBackend::new();
        let spec = ScratchSpec {
            ws_bytes: 1000,
            xq_bytes: 4096,
            xs_bytes: 512,
            bf16_bytes: 2048,
        };
        let a = acquire(&gpu, &spec, 7).unwrap();
        assert_eq!(a.ws_bytes, 1024, "aligned up");
        assert!(gpu.read_alloc(a.tickets).unwrap().iter().all(|&b| b == 0));
        let allocs = gpu.alloc_count();
        let b = acquire(&gpu, &spec, 7).unwrap();
        assert_eq!(a, b, "a second reservation of the same spec is a cache hit");
        assert_eq!(gpu.alloc_count(), allocs, "and allocates nothing");
        // A larger spec grows only the regions it outgrows.
        let bigger = ScratchSpec {
            ws_bytes: 5000,
            ..spec
        };
        let c = acquire(&gpu, &bigger, 7).unwrap();
        assert_ne!(c.ws, a.ws);
        assert_eq!((c.tickets, c.xq, c.xs), (a.tickets, a.xq, a.xs));
        assert_eq!(gpu.alloc_count(), allocs + 1);
        // The earlier view is still a live allocation (never freed in-life).
        assert!(gpu.read_alloc(a.ws).is_some());
    }

    #[test]
    fn a_zero_spec_still_gets_real_regions() {
        let gpu = MockGpuBackend::new();
        let s = acquire(&gpu, &ScratchSpec::default(), 0).unwrap();
        assert!(!s.ws.is_null() && !s.xq.is_null() && !s.xs.is_null() && !s.tickets.is_null());
        assert!(
            !bf16_dequant(&gpu, &ScratchSpec::default())
                .unwrap()
                .is_null()
        );
    }
}
