// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the packed-weight linear contract of the port work
//! (`port/consumer-opt`): [`W4Format`], [`W4Scratch`], the
//! [`W4Linear`] trait, [`W4Box`] and [`W4Opts`].
//!
//! # Who is on which side
//!
//! * Implementers (each in its own module): `port::affine` (affine 4-bit, and
//!   the other widths through `port::affine::widths`), `port::marlin`,
//!   `port::gguf_q4` (IQ4_XS, Q4_K, Q6_K), `port::exl3`, and the NVFP4 wrapper
//!   [`nvfp4`] over the existing `QuantizedWeight` + `w4a16_gemv` family.
//! * Consumers: the dense FFN (`layers/dense_ffn/w4.rs`), the GDN projections
//!   (`layers/qwen3_ssm/w4_proj.rs`), the attention projections
//!   (`layers/qwen3_attention/w4_proj.rs`), the loader routes and the packed
//!   LM head.
//!
//! # The contract, stated once
//!
//! * Shapes: a weight is `[n, k]` (n outputs, k inputs). `x` is `[rows, k]`
//!   BF16 row-major, `out` is `[rows, n]` BF16 row-major, both contiguous.
//! * [`W4Linear::decode`] takes `rows` in `1..=max_decode_rows()` and must be
//!   ROW INVARIANT: the bits of output row `r` depend only on input row `r`
//!   and the weight, never on `rows`. That is what makes an MTP verify row
//!   (M = k) equal the decode of the same token (M = 1), so speculative
//!   decoding at temperature 0 stays exact.
//! * [`W4Linear::prefill`] takes `m > max_decode_rows()`. `Ok(false)` means
//!   declined: nothing useful was written and the caller dequantizes with
//!   [`W4Linear::dequant_bf16`] and runs its BF16 GEMM. Prefill bits need not
//!   equal decode bits.
//! * [`W4Linear::decode_pair`] is gate+up (or q/k/v) sharing `x`; its result
//!   must be byte-equal to two [`W4Linear::decode`] calls.
//! * [`W4Linear::dequant_bf16`] writes the whole `[n, k]` weight as BF16
//!   row-major (the `DenseWeight` layout) into `out`, which holds at least
//!   `n * k * 2` bytes. It is both the decline fallback and the oracle.
//! * Every call is stream-ordered on `stream` and must be capturable into a
//!   CUDA graph when `rows <= max_decode_rows()` (no host sync, no
//!   allocation, no host read of device data inside decode).
//! * The scratch rules are in [`W4Scratch`] and [`scratch`].
//!
//! Methods with default bodies were added beyond the 3.7.2 text only where
//! they cannot break an implementer ([`W4Linear::as_any`]).

use anyhow::Result;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use std::sync::atomic::{AtomicU64, Ordering};

/// The NVFP4 `W4Linear` wrapper (written by lane L23b).
pub mod nvfp4;
/// Scratch sizing and the shared, backend-scoped scratch regions.
pub mod scratch;
/// Pure dispatch decisions (decode vs prefill, prefill chunks, shape checks).
pub mod select;

/// The packed format behind a [`W4Linear`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum W4Format {
    /// MLX-style affine: `w = scale * q + bias` per `group` inputs, `bits` wide.
    Affine { bits: u8, group: u16 },
    /// NVFP4 (E2M1 codes, E4M3 per-16 scales, an F32 tensor scale).
    Nvfp4,
    /// GPTQ (`zero_point: false`) or AWQ (`zero_point: true`) int4 in the
    /// Marlin layout.
    MarlinInt4 { group: u16, zero_point: bool },
    /// GGUF IQ4_XS, kept packed.
    GgufIq4Xs,
    /// GGUF Q4_K, kept packed.
    GgufQ4K,
    /// GGUF Q6_K, kept packed (the 27B LM head).
    GgufQ6K,
    /// EXL3 trellis, `bpw_x100` bits per weight times 100 (e.g. 350).
    Exl3 { bpw_x100: u16 },
}

impl std::fmt::Display for W4Format {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match *self {
            Self::Affine { bits, group } => write!(f, "affine{bits}/g{group}"),
            Self::Nvfp4 => f.write_str("nvfp4"),
            Self::MarlinInt4 { group, zero_point } => {
                let kind = if zero_point { "awq" } else { "gptq" };
                write!(f, "marlin-int4-{kind}/g{group}")
            }
            Self::GgufIq4Xs => f.write_str("gguf-iq4_xs"),
            Self::GgufQ4K => f.write_str("gguf-q4_k"),
            Self::GgufQ6K => f.write_str("gguf-q6_k"),
            Self::Exl3 { bpw_x100 } => {
                write!(f, "exl3/{}.{:02}bpw", bpw_x100 / 100, bpw_x100 % 100)
            }
        }
    }
}

/// Device scratch handed to every [`W4Linear`] call.
///
/// One set of regions is shared by every layer on a backend (the calls run
/// one after another on the model's stream, the same rule the `BufferArena`
/// relies on), reserved at load by [`scratch::acquire`] before the KV cache
/// is sized and before any graph capture. Rules for implementers:
///
/// * `ws`: `ws_bytes` uninitialized bytes, at least the
///   [`W4Linear::scratch_bytes`] the implementation reported for the row
///   count of the call. Free for the duration of one call.
/// * `tickets`: [`scratch::TICKET_SLOTS`] `u32` counters, all zero on entry,
///   and the implementation must leave them zero on exit (the self-resetting
///   split-K ticket of TensorFold's `tf_lane_gemv_split`). A kernel needing
///   more counters keeps them inside `ws` and zeroes them itself.
/// * `xq`: at least [`scratch::xq_bytes`]`(rows, k)` bytes (int8 activations).
/// * `xs`: at least [`scratch::xs_bytes`]`(rows, k)` bytes, one `f32` per 16
///   inputs per row (group sums and activation scales for any group >= 16).
///
/// `rows` there is at most `max(max_decode_rows(), select::prefill_chunk_cap(..))`:
/// the dense FFN splits a long prefill into chunks of at most that many rows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct W4Scratch {
    pub ws: DevicePtr,
    pub ws_bytes: usize,
    pub tickets: DevicePtr,
    pub xq: DevicePtr,
    pub xs: DevicePtr,
}

/// A packed-weight linear layer `out = x * W^T`. See the module docs for the
/// full contract.
pub trait W4Linear: Send + Sync {
    fn format(&self) -> W4Format;
    /// Output features.
    fn n(&self) -> usize;
    /// Input features.
    fn k(&self) -> usize;
    /// Resident device bytes of the packed weight (codes, scales, biases).
    fn device_bytes(&self) -> usize;
    /// Largest `rows` [`W4Linear::decode`] serves, e.g. 16 (affine), 64
    /// (Marlin), 8 (GGUF mmvq).
    fn max_decode_rows(&self) -> usize;
    /// Bytes of `ws` a call with at most `max_rows` rows needs.
    fn scratch_bytes(&self, max_rows: usize) -> usize;
    /// x: [rows, k] BF16 row-major, out: [rows, n] BF16. rows in 1..=max_decode_rows(). Row r's bits must not depend on rows.
    fn decode(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        rows: usize,
        out: DevicePtr,
        s: &W4Scratch,
        stream: u64,
    ) -> Result<()>;
    /// m > max_decode_rows(). Ok(false) = declined (caller dequantizes to BF16 scratch and runs cuBLASLt).
    fn prefill(
        &self,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        m: usize,
        out: DevicePtr,
        s: &W4Scratch,
        stream: u64,
    ) -> Result<bool>;
    /// Optional fused pair (gate+up sharing x and its group sums); default calls decode twice.
    ///
    /// `self` writes `out_a`, `other` writes `out_b`. An override must be
    /// byte-equal to the default and must fall back to it when `other` is
    /// not a weight it can fuse with (see [`W4Linear::as_any`]).
    #[allow(clippy::too_many_arguments)]
    fn decode_pair(
        &self,
        other: &dyn W4Linear,
        gpu: &dyn GpuBackend,
        x: DevicePtr,
        rows: usize,
        out_a: DevicePtr,
        out_b: DevicePtr,
        s: &W4Scratch,
        stream: u64,
    ) -> Result<()> {
        self.decode(gpu, x, rows, out_a, s, stream)?;
        other.decode(gpu, x, rows, out_b, s, stream)
    }
    /// Write the whole weight as BF16 `[n, k]` row-major into `out`.
    fn dequant_bf16(&self, gpu: &dyn GpuBackend, out: DevicePtr, stream: u64) -> Result<()>; // fallback and oracle
    /// The concrete type, so a [`W4Linear::decode_pair`] override can
    /// recognize a partner of its own kind (`other.as_any()?.downcast_ref`).
    /// Implementers that fuse return `Some(self)`; the default opts out.
    fn as_any(&self) -> Option<&dyn std::any::Any> {
        None
    }
}

pub type W4Box = std::sync::Arc<dyn W4Linear>;

/// Load-time quantizer options (`--w4-bits`, `--w4-group`, `--w4-clip-search`,
/// the `affine4_prefill_int8` row).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct W4Opts {
    pub bits: u8,
    pub group: u16,
    pub clip_search: u8,
    pub int8_prefill: bool,
}

impl Default for W4Opts {
    /// 4 bits, groups of 64, no clip search, BF16 prefill: the baseline of
    /// every route row.
    fn default() -> Self {
        Self {
            bits: 4,
            group: 64,
            clip_search: 0,
            int8_prefill: false,
        }
    }
}

/// Process-wide count of declined [`W4Linear::prefill`] calls that took the
/// BF16 dequant fallback (any consumer). A metric source for `/metrics`;
/// lane L23b renders the series.
static PREFILL_DECLINES: AtomicU64 = AtomicU64::new(0);

/// Format and shape only (no device read), so `Result<W4Box>` and friends can be
/// unwrapped and compared in error paths and tests.
impl std::fmt::Debug for dyn W4Linear {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("W4Linear")
            .field("format", &self.format())
            .field("n", &self.n())
            .field("k", &self.k())
            .finish()
    }
}

/// Declined prefill calls so far (monotonic).
pub fn prefill_declines() -> u64 {
    PREFILL_DECLINES.load(Ordering::Relaxed)
}

/// Record one declined prefill call. Consumers call this exactly once per
/// `Ok(false)` they turn into the BF16 fallback.
pub fn note_prefill_decline() {
    PREFILL_DECLINES.fetch_add(1, Ordering::Relaxed);
}

/// A recording, optionally computing, `W4Linear` for consumer tests.
#[cfg(test)]
pub(crate) mod fake;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn format_labels_are_stable() {
        assert_eq!(
            W4Format::Affine { bits: 4, group: 64 }.to_string(),
            "affine4/g64"
        );
        assert_eq!(W4Format::Nvfp4.to_string(), "nvfp4");
        let gptq = W4Format::MarlinInt4 {
            group: 128,
            zero_point: false,
        };
        assert_eq!(gptq.to_string(), "marlin-int4-gptq/g128");
        let awq = W4Format::MarlinInt4 {
            group: 128,
            zero_point: true,
        };
        assert_eq!(awq.to_string(), "marlin-int4-awq/g128");
        assert_eq!(W4Format::GgufIq4Xs.to_string(), "gguf-iq4_xs");
        assert_eq!(W4Format::GgufQ4K.to_string(), "gguf-q4_k");
        assert_eq!(W4Format::GgufQ6K.to_string(), "gguf-q6_k");
        assert_eq!(W4Format::Exl3 { bpw_x100: 350 }.to_string(), "exl3/3.50bpw");
        assert_eq!(W4Format::Exl3 { bpw_x100: 305 }.to_string(), "exl3/3.05bpw");
    }

    #[test]
    fn opts_default_is_the_row_baseline() {
        let o = W4Opts::default();
        assert_eq!(
            (o.bits, o.group, o.clip_search, o.int8_prefill),
            (4, 64, 0, false)
        );
    }

    /// `(name, x, rows, out)` per decode.
    type Seen =
        std::sync::Arc<parking_lot::Mutex<Vec<(&'static str, DevicePtr, usize, DevicePtr)>>>;

    /// Records decodes and overrides nothing, so `decode_pair` and `as_any`
    /// run the trait's default bodies.
    struct Plain {
        name: &'static str,
        seen: Seen,
    }

    impl W4Linear for Plain {
        fn format(&self) -> W4Format {
            W4Format::Nvfp4
        }
        fn n(&self) -> usize {
            8
        }
        fn k(&self) -> usize {
            16
        }
        fn device_bytes(&self) -> usize {
            64
        }
        fn max_decode_rows(&self) -> usize {
            4
        }
        fn scratch_bytes(&self, _max_rows: usize) -> usize {
            0
        }
        fn decode(
            &self,
            _gpu: &dyn GpuBackend,
            x: DevicePtr,
            rows: usize,
            out: DevicePtr,
            _s: &W4Scratch,
            _stream: u64,
        ) -> Result<()> {
            self.seen.lock().push((self.name, x, rows, out));
            Ok(())
        }
        fn prefill(
            &self,
            _gpu: &dyn GpuBackend,
            _x: DevicePtr,
            _m: usize,
            _out: DevicePtr,
            _s: &W4Scratch,
            _stream: u64,
        ) -> Result<bool> {
            Ok(false)
        }
        fn dequant_bf16(&self, _gpu: &dyn GpuBackend, _out: DevicePtr, _stream: u64) -> Result<()> {
            Ok(())
        }
    }

    #[test]
    fn the_default_pair_is_two_decodes_in_order() {
        let gpu = spark_runtime::gpu::mock::MockGpuBackend::new();
        let seen = std::sync::Arc::new(parking_lot::Mutex::new(Vec::new()));
        let a = Plain {
            name: "a",
            seen: seen.clone(),
        };
        let b = Plain {
            name: "b",
            seen: seen.clone(),
        };
        let s = W4Scratch {
            ws: DevicePtr::NULL,
            ws_bytes: 0,
            tickets: DevicePtr::NULL,
            xq: DevicePtr::NULL,
            xs: DevicePtr::NULL,
        };
        let (x, oa, ob) = (DevicePtr(0x100), DevicePtr(0x200), DevicePtr(0x300));
        a.decode_pair(&b, &gpu, x, 3, oa, ob, &s, 7).unwrap();
        assert_eq!(*seen.lock(), vec![("a", x, 3, oa), ("b", x, 3, ob)]);
        assert!(a.as_any().is_none(), "the default opts out of fusion");
    }

    #[test]
    fn declines_are_counted() {
        let before = prefill_declines();
        note_prefill_decline();
        assert!(prefill_declines() > before);
    }
}
