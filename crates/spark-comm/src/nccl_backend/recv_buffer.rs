// SPDX-License-Identifier: AGPL-3.0-only

//! Capacity invariant for the 2-rank send/recv all-reduce receive buffer.
//!
//! At `world_size == 2` the all-reduce is not `ncclAllReduce` (which reduces
//! in-place and needs no scratch): it is a paired `ncclSend`/`ncclRecv` into a
//! persistent receive buffer, followed by a local add. That buffer is the only
//! place in the collective path where a payload is written into an allocation
//! it did not come from — so it is the only place that can overrun.
//!
//! The invariant this module exists to hold:
//!
//! > **`payload_bytes <= recv_capacity_bytes`, always.**
//!
//! and the capacity is **derived from the configured maximum transfer**, never
//! assumed. A fixed constant here previously coexisted with a unit test that
//! modelled a 4096-token prefill chunk while the shipped default was 8192; the
//! test passed and bounded nothing. *A constant plus a stale assumption is not
//! a guard.*

use anyhow::{Context, Result};

/// Element width, in bytes, of the dtype the 2-rank send/recv all-reduce moves.
///
/// The `ncclSend`/`ncclRecv` pair is typed `NcclDataType::Bfloat16` while the
/// collective API takes a **byte** count, so this is the single source of truth
/// for converting between the two. The receive-buffer capacity is derived from
/// the same constant deliberately: a buffer sized for one dtype and an element
/// count computed for another is exactly the drift this module prevents.
pub const ALL_REDUCE_DTYPE_BYTES: usize = 2;

/// Bytes required for the 2-rank all-reduce receive buffer.
///
/// The largest payload any caller can hand a collective is one full arena
/// buffer — `max_batch_tokens × hidden_size × dtype` — which is exactly how
/// `moe_output` is sized (`spark-runtime/src/buffers/sizes.rs`). The
/// tensor-parallel attention and SSM reduces produce the same
/// `[num_tokens, hidden_size]` BF16 shape, and `num_tokens` is capped by
/// `max_batch_tokens`, so this bound covers **every** caller of
/// `all_reduce` / `all_reduce_async`.
///
/// Arithmetic is **checked**: a configuration whose buffer does not fit in a
/// `usize` is rejected at startup rather than wrapping into a small allocation,
/// which is how a sizing bug becomes an out-of-bounds write.
pub fn required_recv_bytes(
    max_batch_tokens: usize,
    hidden_size: usize,
    dtype_bytes: usize,
) -> Result<usize> {
    max_batch_tokens
        .checked_mul(hidden_size)
        .and_then(|elems| elems.checked_mul(dtype_bytes))
        .with_context(|| {
            format!(
                "receive-buffer size overflows usize: \
                 max_batch_tokens={max_batch_tokens} × hidden_size={hidden_size} \
                 × dtype_bytes={dtype_bytes}"
            )
        })
}

/// Enforce the capacity invariant before any NCCL call or kernel launch.
///
/// A free function, not a method, so the guard itself is unit-testable without
/// a live communicator or a GPU: the tests below exercise **this code**, not a
/// re-statement of it.
///
/// In a correctly-configured serve this never fires — [`required_recv_bytes`]
/// already sized the buffer for the worst case. It is retained as defense in
/// depth, because it is the only thing standing between a future caller with a
/// larger payload and a silent out-of-bounds write, and the cost of being wrong
/// is not a bad number: it is corrupted device memory that looks plausible.
pub(crate) fn ensure_payload_fits(
    bytes: usize,
    capacity: usize,
    rank: usize,
    world_size: usize,
) -> Result<()> {
    if bytes > capacity {
        anyhow::bail!(
            "2-rank all-reduce payload exceeds receive-buffer capacity: \
             requested {bytes} bytes ({} elements × {ALL_REDUCE_DTYPE_BYTES} B/elem), \
             capacity {capacity} bytes, rank {rank}, world_size {world_size}. \
             The receive buffer is sized from the configured maximum transfer \
             (max_batch_tokens × hidden_size × {ALL_REDUCE_DTYPE_BYTES} B); a larger \
             payload would write past the allocation. Refusing to send.",
            bytes / ALL_REDUCE_DTYPE_BYTES,
        );
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    const BF16: usize = 2;
    const FP32: usize = 4;

    /// The old fixed buffer, for reference in the boundary tests below.
    const OLD_FIXED_BUFFER: usize = 64 * 1024 * 1024;

    /// The dtype the send/recv path actually moves. If this changes, the
    /// element count and the derived capacity must change together — that
    /// coupling is the entire point of the constant.
    #[test]
    fn all_reduce_dtype_width_is_bf16() {
        assert_eq!(ALL_REDUCE_DTYPE_BYTES, BF16);
    }

    /// The exact boundary the shipped DS4F default sits on:
    /// 8192 × 4096 × 2 = 67,108,864 B — the old fixed buffer, to the byte,
    /// with zero headroom. Must not regress.
    #[test]
    fn exact_boundary_bf16() {
        let need = required_recv_bytes(8192, 4096, BF16).unwrap();
        assert_eq!(need, 67_108_864);
        assert_eq!(
            need, OLD_FIXED_BUFFER,
            "DS4F default sat exactly on the cap"
        );
        // Capacity exactly equal to request: PASS.
        assert!(ensure_payload_fits(need, need, 0, 2).is_ok());
    }

    /// One token past the old boundary. Never silently accepted against 64 MiB:
    /// either the capacity covers it, or it is rejected before communication.
    #[test]
    fn over_boundary_bf16() {
        let need = required_recv_bytes(8193, 4096, BF16).unwrap();
        assert_eq!(need, 67_117_056);
        assert!(need > OLD_FIXED_BUFFER);
        assert!(ensure_payload_fits(need, need, 0, 2).is_ok());
        assert!(ensure_payload_fits(need, OLD_FIXED_BUFFER, 0, 2).is_err());
    }

    /// `hidden_size = 6144` at the DEFAULT 8192-token chunk — the configuration
    /// that overran the fixed buffer by 33,554,432 B at default flags on any
    /// 2-rank serve.
    #[test]
    fn wide_model_bf16() {
        let need = required_recv_bytes(8192, 6144, BF16).unwrap();
        assert_eq!(need, 100_663_296);
        assert_eq!(
            need - OLD_FIXED_BUFFER,
            33_554_432,
            "the overrun this fixes"
        );
        assert!(ensure_payload_fits(need, need, 0, 2).is_ok());
        assert!(ensure_payload_fits(need, OLD_FIXED_BUFFER, 0, 2).is_err());
    }

    /// FP32 sizing is representable. No FP32 collective is enabled by this
    /// change; the point is that the arithmetic is dtype-parameterised, so
    /// widening the reduce dtype resizes the buffer instead of overrunning it.
    #[test]
    fn fp32_is_representable() {
        let need = required_recv_bytes(8192, 4096, FP32).unwrap();
        assert_eq!(need, 134_217_728);
        assert_eq!(need, 2 * required_recv_bytes(8192, 4096, BF16).unwrap());
        assert_eq!(
            need,
            2 * OLD_FIXED_BUFFER,
            "FP32 would have been a 2× overrun"
        );
        assert!(ensure_payload_fits(need, need, 0, 2).is_ok());
    }

    /// Decode B=1 (`h × 2`) shares the buffer with prefill and is trivially
    /// inside any prefill-sized capacity.
    #[test]
    fn decode_b1_bf16() {
        let cap = required_recv_bytes(8192, 4096, BF16).unwrap();
        let decode = 4096 * BF16;
        assert_eq!(decode, 8_192);
        assert!(ensure_payload_fits(decode, cap, 0, 2).is_ok());
    }

    /// Zero elements: a clean no-op, not an error and not a kernel launch.
    #[test]
    fn zero_elements() {
        let cap = required_recv_bytes(8192, 4096, BF16).unwrap();
        assert_eq!(required_recv_bytes(0, 4096, BF16).unwrap(), 0);
        assert!(ensure_payload_fits(0, cap, 0, 2).is_ok());
        assert!(ensure_payload_fits(0, 0, 0, 2).is_ok());
    }

    /// Integer overflow yields a clean error — no allocation, no communication.
    /// It must not wrap into a small allocation.
    #[test]
    fn overflow_is_rejected() {
        assert!(required_recv_bytes(usize::MAX, 4096, BF16).is_err());
        assert!(required_recv_bytes(usize::MAX, 1, 2).is_err());
        assert!(required_recv_bytes(usize::MAX / 2 + 1, 2, 1).is_err());
        // A wrapping multiply would have produced 0 here — which would have
        // allocated nothing and then been "big enough" for every payload.
        assert!(required_recv_bytes(1 << 62, 4, 1).is_err());
    }

    /// Capacity one byte below the request is a hard failure, and the error
    /// carries enough to diagnose the misconfiguration.
    #[test]
    fn one_byte_short_is_rejected() {
        let need = required_recv_bytes(8192, 4096, BF16).unwrap();
        let err = ensure_payload_fits(need, need - 1, 1, 2)
            .unwrap_err()
            .to_string();
        assert!(
            err.contains(&need.to_string()),
            "must report requested bytes: {err}"
        );
        assert!(
            err.contains(&(need - 1).to_string()),
            "must report capacity: {err}"
        );
        assert!(err.contains("world_size 2"), "must report the path: {err}");
    }
}
