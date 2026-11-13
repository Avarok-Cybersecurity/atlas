// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the root of the port sprint's (`port/consumer-opt`) new
//! top-level spark-model modules.
//!
//! Declared from `lib.rs` rather than `layers/mod.rs`, which is at the
//! 500-line cap. Each lane's tree lives under `port/<name>/` and is declared
//! here; lines are inserted immediately BEFORE the marker below by the
//! coordinator's fragment apply pass.

/// The affine 4-bit weight route: MLX-affine weights quantized from BF16 at load, served on the Ada
/// lane GEMV (lanes L17a, L17b, L17c).
pub mod affine;
/// The Marlin W4A16 route: GPTQ/AWQ int4 checkpoints and opt-in round-to-nearest int4 from BF16,
/// repacked on the host and served on vLLM's Marlin GEMM (lane L18b; kernels by lane L18a).
pub mod marlin;
/// Per-model state the port lanes add (`ModelPortState`), one field per lane.
pub mod state;
/// The packed-weight linear contract (`W4Linear`, master plan 3.7.2), its
/// shared scratch and dispatch rules. Lane L21; `w4/nvfp4.rs` is L23b's.
pub mod w4;
// @port:mods
