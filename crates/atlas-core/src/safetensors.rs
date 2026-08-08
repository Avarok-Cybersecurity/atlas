// SPDX-License-Identifier: AGPL-3.0-only

//! The one place that turns a safetensors `data_offsets` pair into a byte span.
//!
//! A checkpoint is third-party data: Atlas loads it by URL, so every number in
//! the header is attacker-controlled until it has been checked. The header
//! declares each tensor as `"data_offsets": [start, end]` relative to the data
//! section, and the naive `end - start` is a `u64` subtraction that WRAPS on a
//! crafted or truncated file — a reversed pair yields a length near `u64::MAX`,
//! which downstream becomes an allocation size, a `pread` window, or an RDMA
//! `len` published to a peer.
//!
//! Two loaders parse the same header format for different transports
//! (`spark_runtime::fast_weights` reads it with O_DIRECT, and
//! `spark_storage::weight_peer` republishes it as an RDMA manifest). They live
//! in crates that cannot depend on each other, so the *rule* lives here and
//! both call it, rather than each keeping its own copy of the arithmetic.

use anyhow::{Context, Result, bail};
use serde_json::Value;

/// A validated tensor byte span: an absolute file offset and a length that is
/// known to fit inside the file it came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TensorSpan {
    /// Absolute byte offset in the shard file where the tensor starts.
    pub abs_offset: u64,
    /// Tensor byte length.
    pub len: u64,
}

/// Validate one tensor's `data_offsets` against the file that declared it.
///
/// `offsets` is the raw `data_offsets` JSON value, `data_start` is
/// `8 + header_size` (where the data section begins), and `file_len` is the
/// shard's size on disk.
///
/// Rejects, in this order: a pair that isn't two integers, a reversed pair
/// (`end < start`, the underflow), and a span that runs past end-of-file.
/// Callers get a named error instead of a wrapped length.
pub fn tensor_span(
    tensor: &str,
    offsets: &Value,
    data_start: u64,
    file_len: u64,
) -> Result<TensorSpan> {
    let arr = offsets
        .as_array()
        .with_context(|| format!("tensor {tensor}: data_offsets is not an array"))?;
    if arr.len() != 2 {
        bail!(
            "tensor {tensor}: data_offsets has {} entries, expected 2",
            arr.len()
        );
    }
    let rel_start = arr[0]
        .as_u64()
        .with_context(|| format!("tensor {tensor}: bad data_offsets[0]"))?;
    let rel_end = arr[1]
        .as_u64()
        .with_context(|| format!("tensor {tensor}: bad data_offsets[1]"))?;
    // The underflow. `rel_end - rel_start` wraps to ~u64::MAX on a reversed
    // pair; in release builds that is silent.
    if rel_end < rel_start {
        bail!("tensor {tensor}: data_offsets [{rel_start}, {rel_end}] end precedes start");
    }
    let len = rel_end - rel_start;
    // `data_start` is bounded by the 64 MiB header cap the callers enforce, but
    // `rel_start` is not, so the sum still needs checking.
    let abs_offset = data_start
        .checked_add(rel_start)
        .with_context(|| format!("tensor {tensor}: data_offsets[0] {rel_start} overflows"))?;
    let abs_end = abs_offset
        .checked_add(len)
        .with_context(|| format!("tensor {tensor}: span {abs_offset}+{len} overflows"))?;
    if abs_end > file_len {
        bail!(
            "tensor {tensor}: spans bytes {abs_offset}..{abs_end} but the shard is only \
             {file_len} bytes (truncated or corrupt checkpoint)"
        );
    }
    Ok(TensorSpan { abs_offset, len })
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    /// A well-formed pair keeps its exact span, converted to absolute.
    #[test]
    fn accepts_well_formed_span() {
        let s = tensor_span("w", &json!([0, 128]), 72, 4096).unwrap();
        assert_eq!(
            s,
            TensorSpan {
                abs_offset: 72,
                len: 128
            }
        );
        let s = tensor_span("w", &json!([128, 256]), 72, 4096).unwrap();
        assert_eq!(
            s,
            TensorSpan {
                abs_offset: 200,
                len: 128
            }
        );
    }

    /// A zero-length tensor is legal safetensors and must not be rejected.
    #[test]
    fn accepts_empty_tensor() {
        let s = tensor_span("w", &json!([64, 64]), 8, 4096).unwrap();
        assert_eq!(s.len, 0);
    }

    /// A tensor ending exactly at EOF is the last tensor of every real shard.
    #[test]
    fn accepts_span_ending_exactly_at_eof() {
        let s = tensor_span("w", &json!([0, 4024]), 72, 4096).unwrap();
        assert_eq!(s.len, 4024);
    }

    /// THE BUG: `rel_end - rel_start` on a reversed pair wraps to ~u64::MAX in
    /// release builds. Reject it instead.
    #[test]
    fn rejects_reversed_offsets_instead_of_underflowing() {
        let err = tensor_span("evil", &json!([4096, 0]), 8, 65536).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("evil"), "{msg}");
        assert!(msg.contains("end precedes start"), "{msg}");
        // Sanity-check that the unchecked form really does wrap, so this test
        // is guarding a live hazard and not a hypothetical one.
        assert_eq!(0u64.wrapping_sub(4096), u64::MAX - 4095);
    }

    /// A truncated shard: the header still advertises the full tensor.
    #[test]
    fn rejects_span_past_end_of_file() {
        let err = tensor_span("w", &json!([0, 1_000_000]), 72, 4096).unwrap_err();
        assert!(err.to_string().contains("truncated or corrupt"), "{err}");
    }

    /// A huge `rel_start` must not wrap when added to `data_start`.
    #[test]
    fn rejects_offset_that_overflows_u64() {
        let err = tensor_span("w", &json!([u64::MAX, u64::MAX]), 72, 4096).unwrap_err();
        assert!(err.to_string().contains("overflows"), "{err}");
    }

    /// Malformed `data_offsets` shapes are named, not indexed into blindly.
    #[test]
    fn rejects_malformed_offsets() {
        assert!(tensor_span("w", &json!("nope"), 8, 4096).is_err());
        assert!(tensor_span("w", &json!([0]), 8, 4096).is_err());
        assert!(tensor_span("w", &json!([0, 1, 2]), 8, 4096).is_err());
        assert!(tensor_span("w", &json!([-1, 16]), 8, 4096).is_err());
        assert!(tensor_span("w", &json!([0.5, 16]), 8, 4096).is_err());
    }
}
