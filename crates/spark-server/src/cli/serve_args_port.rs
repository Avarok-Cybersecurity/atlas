// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: the port sprint's (`port/consumer-opt`) `spark serve` flags,
//! flattened into `ServeArgs` as its `port_sprint` field (`ServeArgs::port` is
//! the HTTP listen port).
//!
//! Its own struct so the lanes' flags never touch `serve_args.rs` (allow-listed
//! over the cap). Every flag is an `Option` or has a default that changes
//! nothing: an absent flag publishes nothing, so the row's environment
//! variable and then the target's declaration decide, exactly as without the
//! flag. Each flag is published by one line in `main_modules/port_apply.rs`.
//! Lines are inserted immediately BEFORE the marker below by the coordinator's
//! fragment apply pass.

/// The port lanes' serve flags.
#[derive(clap::Args, Debug, Clone, Default, PartialEq)]
pub struct PortArgs {
    /// Affine W4 routes (lane L23a): code width of the load-time BF16 -> affine
    /// quantizer, 2, 3, 4, 5, 6 or 8 (default 4, or `AVAROK_W4_BITS`). Widths
    /// other than 4 need `--w4-group 64`. Changes nothing unless an affine
    /// route row is on (`AVAROK_DENSE_AFFINE4_FROM_BF16`,
    /// `AVAROK_GDN_AFFINE4_FROM_BF16`, `AVAROK_ATTN_AFFINE4_FROM_BF16`).
    #[arg(long = "w4-bits", value_name = "BITS")]
    pub w4_bits: Option<u8>,
    /// Affine W4 routes: inputs per scale and bias, 32, 64 or 128 at 4 bits,
    /// 64 at the other widths (default 64, or `AVAROK_W4_GROUP`;
    /// `AVAROK_W4_GDN_GROUP` sets the GDN projections' group alone).
    #[arg(long = "w4-group", value_name = "N")]
    pub w4_group: Option<u16>,
    /// Affine W4 routes: shrink candidates the quantizer tries per group,
    /// keeping the least squared error (0 = plain MLX min-max, the default,
    /// or `AVAROK_W4_CLIP_SEARCH`; at most 10 at 4 bits, 8 otherwise). Load
    /// time only.
    #[arg(long = "w4-clip-search", value_name = "N")]
    pub w4_clip_search: Option<u8>,
    /// Packed LM head (lane L24): keep a GGUF checkpoint's Q6_K output head as
    /// the file's raw super-blocks and serve it with the Ada keep-packed GEMV
    /// (`port::packed_head`; 1.04 GB instead of 2.54 GB read per token on
    /// Qwen3.8-27B). Needs the keep-packed GGUF route (`AVAROK_GGUF_NATIVE_IQ4`)
    /// and `--lm-head-dtype bf16`; refused with `--speculative`, whose drafter
    /// reads the head as BF16. Off: the head is expanded as before.
    #[arg(long = "lm-head-packed")]
    pub lm_head_packed: bool,
    /// Pinned host pool (lane L25): MiB page-locked per `cuMemHostRegister`
    /// call, 1 to 4096 (default 64). Each pool chunk is pinned by its own call,
    /// on a background thread after the first chunks; a run never spans two
    /// chunks, so this is also the largest run, and the batched KV swap's
    /// staging (two halves) never exceeds one chunk.
    #[arg(long = "pinned-pool-chunk-mb", value_name = "MIB")]
    pub pinned_pool_chunk_mb: Option<u32>,
    // @port:fields
}
