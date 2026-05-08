// SPDX-License-Identifier: AGPL-3.0-only
//! Vendor-agnostic per-layer forward orchestration.
//!
//! This module hosts the per-layer kernel-launch sequences that any
//! end-to-end inference driver (CUDA, Metal, AMD ROCm, …) needs to call.
//! It depends only on the `GpuBackend` trait + kernel-name strings + a
//! `QuantWeights` trait that lets concrete weight types plug in without
//! the forward path knowing which quantisation format they speak.
//!
//! Goal: when a future hardware-target driver lands, its `examples/{vendor}_qwen35_inference`
//! shrinks to tokenizer + weight load + token loop — the per-layer
//! orchestration is *here*, written once.
//!
//! Out of scope: the production decode forward in
//! `crate::model::trait_impl::decode_a` is heavily coupled to CUDA-graph
//! capture / NCCL / paged-KV machinery; this module is intentionally
//! simpler so a single-token decoder example can call it directly.

pub mod quant_weights;
pub mod qwen3_5;

pub use quant_weights::QuantWeights;
