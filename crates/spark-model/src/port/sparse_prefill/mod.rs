// SPDX-License-Identifier: AGPL-3.0-only
// Algorithm after XAttention (antidiagonal block scoring for block-sparse prefill), described in its paper;
// independent implementation, no source copied (the reference repository has no detected license).

//! Responsibility: block-sparse PREFILL attention, XAttention-style (lane L35): the row and knob
//! resolution, the module map, and the one entry the attention layer calls.
//!
//! WHAT. For a long prompt, each chunk's full-attention layer scores every (query block, key block) pair
//! of 128 tokens with a strided antidiagonal sum, keeps the key blocks that hold `threshold` of each
//! query block's softmax mass (plus the sink block and the local diagonal band), and runs flash
//! attention over the kept blocks only (`kernels/ada/common/prefill_block_sparse_ada.cu`). Decode is
//! untouched; so is every prompt shorter than `--prefill-sparse-min-tokens`.
//!
//! APPROXIMATE, OFF EVERYWHERE. The `[defaults] prefill_sparse_attn` row is `false` on every target and
//! never defaults on. With it off, [`config`] reports disabled and the layer's hook
//! returns before reading anything else, so the dense path is byte-identical to the base. With it on,
//! the arm runs only where the module is in the image (Ada); elsewhere one log line says it is absent
//! and the dense kernel runs. A threshold of 1.0 means "keep everything" and takes the DENSE kernel, so
//! `AVAROK_PREFILL_SPARSE_ATTN=1 --prefill-sparse-threshold 1.0` is the serve-level identity gate.
//!
//! COVERAGE. The single-stream paged prefill (`qwen3_attention/prefill/paged_attn.rs`, every chunk
//! after the first) with BF16 or FP8 E4M3 KV, head_dim 128 or 256, no sliding window, outside CUDA
//! graph capture, and with no sparse-attention seam selector registered (L29's prefill View would
//! already have compacted the extent; two approximations are never stacked). Chunk 0 (the contiguous
//! cache-skip path) and co-dispatched batched prefill keep the dense kernels; the gate is the attended
//! length (`kv_len`, the prompt prefix through this chunk).
//!
//! Resolution of the knobs: the `spark serve` flag (published by `port_apply` through [`apply_flags`]),
//! else the environment variable, else the default below. [`validate_flags`] refuses an out-of-range
//! value at startup.

pub mod config;
pub mod launch;
pub mod metrics;
pub mod reference;
pub mod select;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod tests_reference;

pub use config::{Config, Knobs};
pub use launch::{Call, Decline, Geometry, KvKind, run};

/// Tokens per selection block, for queries and keys (the kernel's `psa::BLOCK`).
pub const BLOCK: u32 = 128;
/// Query rows per attention CTA (the kernel's `psa::TILE_Q`).
pub const TILE_Q: u32 = 64;
/// Antidiagonal strides the kernel is compiled for.
pub const STRIDES: [u32; 3] = [4, 8, 16];
/// Head widths the kernel is compiled for.
pub const HEAD_DIMS: [u32; 2] = [128, 256];

/// `--prefill-sparse-min-tokens` default: the attended length at which a chunk goes sparse.
pub const DEFAULT_MIN_TOKENS: u32 = 32_768;
/// `--prefill-sparse-threshold` default: the softmax mass each query block keeps.
pub const DEFAULT_THRESHOLD: f32 = 0.9;
/// `--prefill-sparse-stride` default: the antidiagonal stride.
pub const DEFAULT_STRIDE: u32 = 8;

pub const ENV_MIN_TOKENS: &str = "AVAROK_PREFILL_SPARSE_MIN_TOKENS";
pub const ENV_THRESHOLD: &str = "AVAROK_PREFILL_SPARSE_THRESHOLD";
pub const ENV_STRIDE: &str = "AVAROK_PREFILL_SPARSE_STRIDE";

/// Whether the `prefill_sparse_attn` row resolved on (env `AVAROK_PREFILL_SPARSE_ATTN`, else the target).
pub fn row_on() -> bool {
    crate::layers::ops::port_rows::prefill_sparse_attn::row()
}

/// The resolved configuration, sealed on first read (after `port_apply::apply` published the flags).
pub fn config() -> &'static Config {
    config::resolved()
}

/// `port_apply::validate` entry: refuse a flag or environment value that cannot be served.
pub fn validate_flags(
    min_tokens: Option<u32>,
    threshold: Option<f32>,
    stride: Option<u32>,
) -> anyhow::Result<()> {
    let knobs = Knobs {
        min_tokens,
        threshold,
        stride,
    };
    config::resolve(false, &knobs, |name| std::env::var(name).ok()).map(|_| ())
}

/// `port_apply::apply` entry: publish the flags, then log the boot line when the row is on.
pub fn apply_flags(min_tokens: Option<u32>, threshold: Option<f32>, stride: Option<u32>) {
    let knobs = Knobs {
        min_tokens,
        threshold,
        stride,
    };
    let any_flag = knobs.any();
    config::set_cli(knobs);
    let c = config();
    if c.enabled {
        tracing::warn!("{}", c.boot_line());
    } else if any_flag {
        tracing::info!(
            "--prefill-sparse-* flags given but the prefill_sparse_attn row is off \
             (AVAROK_PREFILL_SPARSE_ATTN unset or 0): the flags change nothing"
        );
    }
}
