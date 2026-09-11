// SPDX-License-Identifier: AGPL-3.0-only

//! The SERVING defaults baked from the compiled target's
//! `kernels/<hw>/HARDWARE.toml` `[defaults]` table.
//!
//! # Why this is code and not a launch script
//!
//! Maintainer review, 2026-09-11 (tbraun96), on the H100 integration branch:
//!
//! > There is no arch separation at all. H100 builds compile GB10's kernel
//! > tree. Every Hopper/GB10 divergence is expressed as an env lever set by an
//! > H100 recipe living outside this repo — not as arch-selected code. "No
//! > interference" rests on discipline rather than structure.
//!
//! Each field below was a line in that external recipe. Baking them from the
//! target's own HARDWARE.toml makes the recipe STRUCTURAL: `build.rs` reads
//! exactly one `kernels/<hw>` tree, so a binary built for GB10 cannot carry
//! Hopper's numbers, and an H100 serve with an empty environment reproduces
//! the measured configuration without anyone remembering a prefix.
//!
//! # The rule every consumer follows
//!
//! **Baked default first, environment second.** The environment is an
//! EXPLICIT operator override, not the source of truth, and `spark-server`
//! logs one `target defaults (<hw>): …` line naming every resolved value and
//! which of them came from the environment. See
//! `spark_model::layers::ops::target_defaults` for the resolvers and the
//! override grammar.
//!
//! # Adding a lever
//!
//! Four places, one commit: the field here, the parse arm in
//! `build_defaults.rs`, the resolver in `spark-model`, and the row in each
//! `kernels/<hw>/HARDWARE.toml` that differs from
//! [`build_defaults::baseline`](../../build_defaults.rs). `parse_defaults`
//! panics on an unknown `[defaults]` key, so a table that names a lever the
//! code does not have fails the build instead of reading as agreement.

/// One compiled target's serving defaults.
///
/// `Copy` and entirely `'static` — it is a `const` emitted by `build.rs` into
/// `OUT_DIR/target_defaults.rs` and `include!`d by `lib.rs`, so
/// [`crate::TARGET_DEFAULTS`] is resolved at compile time with no I/O.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TargetDefaults {
    /// `kernels/<hw>` this binary's kernels were compiled from — `gb10`,
    /// `hopper`, `b200`, …. Empty only when a build read no HARDWARE.toml at
    /// all; consumers print it verbatim and must not branch on it (branching
    /// on the name would re-create the per-arch `if` this table replaces).
    pub hw: &'static str,
    /// `ATLAS_CUBLAS_GEMM` grammar — which projection FAMILIES take a cuBLASLt
    /// arm, as a comma-separated subset of `all|ffn|attn|ssm|head|off`. A
    /// STRING, not a bitfield, because the grammar (and its "unknown token
    /// never widens the set" rule) lives in
    /// `spark_model::layers::ops::parse_cublas_scope` and atlas-kernels is
    /// below spark-model in the dependency graph.
    pub cublas_gemm_scope: &'static str,
    /// `w8a16_gemv_batch16` serves the 5..=32-row native-FP8 dense-FFN decode
    /// widths (`dense_ffn_batch16_decode.rs`).
    pub ffn_batch16_tier: bool,
    /// `w8a16_gemm_m16` serves those same widths on TENSOR CORES instead
    /// (`dense_ffn_m16_tc.rs`). Reassociates the K reduction.
    pub ffn_m16_tc: bool,
    /// `w8a16_gemm_m16{,_strided}` serves the decode QKV and o_proj tiers.
    pub attn_m16_tc: bool,
    /// `w8a16_gemv_batch16_ncol{2,4}{,_strided}` serve the decode attention
    /// projections (`qwen3_attention/attn_ncol_gemv.rs`).
    pub attn_ncol_gemv: bool,
    /// `dense_gemm_m16_bf16` serves the BF16 decode head at 5..=16 rows
    /// (`model/trait_impl/lm_head_batched.rs`).
    pub lm_head_m16_tc: bool,
    /// Upper edge of the BF16 decode head's batched-GEMV band. Clamped by the
    /// resolver to the kernel's compile-time row bound; it is a BAND, not a
    /// switch, so there is no "off".
    pub lm_head_batchm_max: u32,
    /// One strided recurrent launch per batch on the GDN decode path
    /// (`layers/qwen3_ssm/gdn_flags.rs`).
    pub ssm_batched_recurrent: bool,
    /// Split SiLU+down on the decode path (`ModelLevers::decode_split_silu`).
    pub decode_split_silu: bool,
    /// `auto` — size the decode-rollback ring from free memory at preflight
    /// (#915) — or a decimal depth `0..=DECODE_ROLLBACK_RING_SLOTS`.
    pub ssm_decode_ring_slots: &'static str,
}
