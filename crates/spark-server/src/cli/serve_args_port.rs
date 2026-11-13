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
    /// MTP draft-head vocabulary (`mtp_vocab_rank` row): `id` holds the first
    /// `--mtp-vocab` token ids, `freq` the most frequent tokens of the shipped
    /// ranking (Qwen3.5 tokenizer only; logs and keeps `id` otherwise). Absent:
    /// `AVAROK_MTP_VOCAB_RANK`, then the target's declaration.
    #[arg(long = "mtp-vocab-rank", value_name = "RANK", value_parser = ["id", "freq"])]
    pub mtp_vocab_rank: Option<String>,
    /// Copy drafter for speculative rounds (lane L13): `off`, `chain` (fill
    /// the MTP verify chain from an earlier copy of the context), `rounds`
    /// (verify a confident copy of up to `--copy-draft-max-len` tokens alone,
    /// single sequence) or `auto` (both). Absent: the `copy_draft` row
    /// (`AVAROK_COPY_DRAFT`, else the target's declaration, `off`). Outputs
    /// stay the target's own: every copied token is verified.
    #[arg(long = "copy-draft", value_name = "MODE")]
    pub copy_draft: Option<String>,
    /// Copy drafter: tokens a copy source must agree on before it may fill the
    /// MTP chain (2 to 64; default 8, or `AVAROK_COPY_DRAFT_MIN_MATCH`).
    #[arg(long = "copy-draft-min-match", value_name = "N")]
    pub copy_draft_min_match: Option<u32>,
    /// Copy drafter: the longest lookup round, in tokens (1 to 15; default 15,
    /// or `AVAROK_COPY_DRAFT_MAX_LEN`).
    #[arg(long = "copy-draft-max-len", value_name = "N")]
    pub copy_draft_max_len: Option<u32>,
    /// Copy drafter: the factor by which a copy chain's expected accepted
    /// tokens must beat the MTP chain's (default 1.3, or
    /// `AVAROK_COPY_DRAFT_ADVANTAGE`).
    #[arg(long = "copy-draft-advantage", value_name = "X")]
    pub copy_draft_advantage: Option<f32>,
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
    /// Host-RAM KV tier budget in GiB (lane L26b; 0 or absent = off). The tier serves only
    /// with the `kv_host_tier` row on (`AVAROK_KV_HOST_TIER`) and the prefix cache on; the
    /// budget must leave 8 GiB of host RAM.
    #[arg(long = "kv-host-tier-gb", value_name = "GIB")]
    pub kv_host_tier_gb: Option<u32>,
    /// Host KV tier: the shortest restore, in tokens (default 256).
    #[arg(long = "kv-tier-min-tokens", value_name = "N")]
    pub kv_tier_min_tokens: Option<u32>,
    /// Host KV tier: D2H write-back bytes one scheduler step may start, in MiB (default 32).
    #[arg(long = "kv-tier-writeback-mb-per-step", value_name = "MIB")]
    pub kv_tier_writeback_mb_per_step: Option<u32>,
    /// Host KV tier eviction policy: `lru` (default), `arc` or `heat`.
    #[arg(long = "kv-tier-policy", value_name = "POLICY")]
    pub kv_tier_policy: Option<String>,
    /// Host KV tier store mode: `eager` (default: copy newly cached blocks at once) or `lazy`
    /// (copy them only when they approach GPU eviction).
    #[arg(long = "kv-tier-store", value_name = "MODE")]
    pub kv_tier_store: Option<String>,
    /// Disk KV tier (lane L27, row `kv_tier_disk`, `AVAROK_KV_TIER_DISK`):
    /// directory where host-tier blocks evicted from host RAM are kept, one
    /// record file each, and adopted again by the next start of the same
    /// model. Needs `--kv-tier-disk-gb` and the host KV tier; refused under
    /// `--hermetic`. One server per directory. Ignored while the row is off.
    #[arg(long = "kv-tier-dir", value_name = "DIR")]
    pub kv_tier_dir: Option<std::path::PathBuf>,
    /// Disk KV tier: byte budget in GiB (least recently used records are
    /// removed first). Ignored while the `kv_tier_disk` row is off.
    #[arg(long = "kv-tier-disk-gb", value_name = "GIB")]
    pub kv_tier_disk_gb: Option<u64>,
    // @port:fields
}
