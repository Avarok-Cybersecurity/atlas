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
//! # Adding a lever — four places, one commit
//!
//! A lever row is: the field here, the parse arm in `build_defaults.rs`, the
//! resolver field in `spark_model::layers::ops::target_defaults`, and the row
//! in EVERY `kernels/<hw>/HARDWARE.toml` that has a `[defaults]` table — plus
//! the boot line and a test, which the resolver and
//! `tests/target_defaults.rs` already force. All in ONE commit.
//!
//! ★ THE CONTRACT IS ALSO A SCOPE RULE. A row belongs in the commit that
//! lands its CONSUMER, not in the commit that builds this table. A row whose
//! dispatch site does not exist yet is a declaration nothing reads: it cannot
//! be graded, an operator who sets its variable gets silence, and the `(env)`
//! tag in the boot line would report a decision that changes no code. So a
//! kernel PR adds its own row here, in `build_defaults.rs`, in the resolver
//! and in all three tables, together with the arm that reads it.
//!
//! `parse_defaults` panics on an unknown `[defaults]` key, so a table that
//! names a lever the code does not have fails the build instead of reading as
//! agreement — which is what makes "one commit" enforceable rather than
//! merely asked for.

/// One compiled target's serving defaults.
///
/// `Copy` and entirely `'static` — it is a `const` emitted by `build.rs` into
/// `OUT_DIR/target_ptx.rs` and `include!`d by `lib.rs`, so
/// [`crate::TARGET_DEFAULTS`] is resolved at compile time with no I/O.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TargetDefaults {
    /// `kernels/<hw>` this binary's kernels were compiled from — `gb10`,
    /// `hopper`, `b200`, …. Empty only when a build read no HARDWARE.toml at
    /// all; consumers print it verbatim and must not branch on it (branching
    /// on the name would re-create the per-arch `if` this table replaces).
    pub hw: &'static str,
    /// Upper edge of the BF16 decode head's batched-GEMV band
    /// (`model/trait_impl/lm_head_batched.rs`). Clamped by the resolver to the
    /// kernel's compile-time row bound; it is a BAND, not a switch, so there
    /// is no "off".
    pub lm_head_batchm_max: u32,
    /// One strided recurrent launch per batch on the GDN decode path
    /// (`layers/qwen3_ssm/gdn_flags.rs`).
    ///
    /// TRUE on hopper: +6% on the serve, and md5-identical output to the
    /// per-sequence launches. It was `ATLAS_SSM_BATCHED_RECURRENT=1` in an
    /// H100 launch script outside this repository, which is the arrangement
    /// the 2026-09-11 review called discipline rather than structure.
    pub ssm_batched_recurrent: bool,
    /// Split SiLU+down on the decode path (`ModelLevers::decode_split_silu`).
    pub decode_split_silu: bool,
    /// The `w8a16_gemm_m16` tensor-core tier on the DENSE-FFN decode arm
    /// (`layers/dense_ffn_m16_tc.rs`), rungs 2-3 of the `w8_gemm!` ladder.
    ///
    /// FALSE on hopper, and it is the one row in this table whose receipt is a
    /// LOSS. H100 round 6, serve J against serve I on the same binary: C=16
    /// aggregate 228.02 -> 216.27 (-5.2%) on the short shape and 177.60 ->
    /// 171.62 (-3.4%) on the long one, TPOT +5.7% / +4.3%, against deltas
    /// 30-95x the rep-to-rep spread. The kernel is 3.4-3.7x faster than the
    /// tier it replaces in the microtest and still costs the serve, because it
    /// dispatches by ROW COUNT and so catches a chunked prefill's tail chunk.
    /// The ATTENTION half of the same kernel wins (`attn_m16_tc`), which is why
    /// this is two rows and not one.
    pub ffn_m16_tc: bool,
    /// The `w8a16_gemm_m16{,_strided}` tensor-core tiers on the decode Q/K/V
    /// and o_proj projections (`layers/qwen3_attention/`).
    ///
    /// TRUE on hopper: H100 round 9 cell W against cell U, C=16 aggregate
    /// 235.47 -> 247.85 (+5.26%) and TPOT 53.42 -> 50.01 ms (-6.38%) against a
    /// 0.15% rep spread; long shape +4.12% / -5.40%. C=1 is 17.87 vs 17.89 ms,
    /// a measured null, which is correct by construction — the tier is
    /// restricted to 5..16 rows. Same kernel family as [`Self::ffn_m16_tc`],
    /// opposite verdict, which is why they are two rows.
    pub attn_m16_tc: bool,
}
