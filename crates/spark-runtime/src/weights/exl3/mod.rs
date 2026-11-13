// SPDX-License-Identifier: AGPL-3.0-only
// After ExLlamaV3 (turboderp-org/exllamav3), MIT, Copyright (c) 2025 Turboderp, via TensorFold
// zig/src/core/exl3_format.zig, exl3_load.zig, exl3_layout.zig, exl3_prepare.zig (Apache-2.0,
// ashhart/TensorFold, commit 3885dce; Copyright 2026 TensorFold contributors). Changes: translated
// from Zig to Rust and split by responsibility into the child modules below. See EXL3-ATTRIBUTION.md.

//! Responsibility: the EXL3 weight reader (port lane L20a): admit and load the trellis-quantized
//! linear layers of an ExLlamaV3 checkpoint into host tensors, plus the FP64 reference decode that
//! the EXL3 kernels are checked against.
//!
//! EXL3 is ExLlamaV3's QTIP-style trellis quantization: each linear layer `[K, N]` is a group of
//! tensors under one prefix (`trellis` int16 `[K/16, N/16, 16 * bits]`, fp16 `suh` / `svh` or packed
//! sign words `su` / `sv`, an optional fp16 `bias`, and a marker naming the codebook), decoded as
//! `W = diag(suh) @ H_K @ W_q @ H_N @ diag(svh)` with 128-point Hadamard rotations. Widths run from 1
//! to 8 bits in half-bit steps (half steps with the `mul1` codebook only), mixed per tensor.
//!
//! WHO CALLS THIS. Nothing on any target unless lane L20b's row `weights_exl3` is on (false on every
//! target): spark-model's `port::exl3::from_exl3_tensors(gpu, &Exl3Tensor, stream)` uploads an
//! [`Exl3Tensor`] for the Ada EXL3 kernels, and the packed-install routes of lane L23b walk an
//! [`Exl3Scan`] (`groups()` for the quantized layers, `plain_tensors()` for everything the family's
//! ordinary loader reads). The reader has no side effects: a scan reads safetensors headers only, and
//! a load reads one group's tensors with positioned reads.
//!
//! Modules:
//! - [`fp16`]: exact fp16 conversions (the codebooks round once from float64).
//! - [`format`]: widths, codebooks, the tile bitstream, placement, unpack, packed signs.
//! - [`dequant`]: Hadamard rotations, full and block dequantization, the FP64 matvec (the oracle).
//! - [`layout`]: group geometry, tensor-parallel parts, expert layouts, the strips order.
//! - [`shards`]: header-only safetensors shard reading.
//! - [`load`]: group admission and refusal reasons, the scan, loading a group.
//! - [`config`]: `config.json`'s EXL3 block.
//! - [`tensor`]: [`Exl3Tensor`], the host tensor lane L20b consumes.

pub mod config;
pub mod dequant;
pub mod format;
pub mod fp16;
pub mod layout;
pub mod load;
pub mod shards;
pub mod tensor;

pub use config::{Exl3Config, config_from_json, is_exl3_checkpoint, read_config};
pub use format::{Codebook, Width};
pub use layout::Split;
pub use load::{Exl3Scan, GroupInfo, Refusal, Summary};
pub use shards::ShardSet;
pub use tensor::{Exl3Tensor, ScaleForm};

#[cfg(test)]
mod tests;

#[cfg(test)]
mod fixture;

#[cfg(test)]
mod layout_tests;

#[cfg(test)]
mod load_tests;

#[cfg(test)]
mod oracle_tests;
