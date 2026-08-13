// SPDX-License-Identifier: AGPL-3.0-only

//! Metal-build stub of the cuda-only `flashinfer` module.
//!
//! The real [`crate::flashinfer`] module (FlashInfer ragged BF16 prefill) is
//! gated behind `feature = "cuda"`; it links the FlashInfer AOT artifacts that
//! do not exist on macOS. spark-model names these entry points unconditionally,
//! so the metal build (`cargo check --features metal`, cuda off) needs them to
//! resolve. `available()` reports `false`, so the (guarded) prefill call is
//! never reached — its body is `unreachable!`.

use anyhow::Result;

/// FlashInfer is never available in a non-cuda build.
pub fn available() -> bool {
    false
}

#[allow(clippy::too_many_arguments)]
pub fn ragged_prefill_bf16_hd256(
    _q: u64,
    _k: u64,
    _v: u64,
    _o: u64,
    _qo_indptr_h: &[i32],
    _kv_indptr_h: &[i32],
    _qo_indptr_d: u64,
    _kv_indptr_d: u64,
    _batch: u32,
    _total_qo_rows: u32,
    _total_kv_rows: u32,
    _num_qo_heads: u32,
    _num_kv_heads: u32,
    _head_dim: u32,
    _sm_scale: f32,
    _causal: bool,
    _stream: u64,
) -> Result<()> {
    unreachable!("flashinfer::ragged_prefill_bf16_hd256 is cuda-only (not built for metal)")
}

#[allow(clippy::too_many_arguments)]
pub fn ragged_prefill_bf16_hd128(
    _q: u64,
    _k: u64,
    _v: u64,
    _o: u64,
    _qo_indptr_h: &[i32],
    _kv_indptr_h: &[i32],
    _qo_indptr_d: u64,
    _kv_indptr_d: u64,
    _batch: u32,
    _total_qo_rows: u32,
    _total_kv_rows: u32,
    _num_qo_heads: u32,
    _num_kv_heads: u32,
    _head_dim: u32,
    _sm_scale: f32,
    _causal: bool,
    _sliding_window: Option<u32>,
    _stream: u64,
) -> Result<()> {
    unreachable!("flashinfer::ragged_prefill_bf16_hd128 is cuda-only (not built for metal)")
}
