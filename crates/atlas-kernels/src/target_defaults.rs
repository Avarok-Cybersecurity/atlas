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
    /// `gated_delta_rule_chunk_delta_h_tcfuse_x2` serves the GDN chunked
    /// PREFILL state spine on tensor cores (`layers/ops/ssm_gdn_a3.rs`).
    ///
    /// FALSE on every target here, deliberately. The kernel lives in
    /// `kernels/hopper/common/gated_delta_rule_chunk_tc.cu` — developed and
    /// validated on GB10, arch-neutral, but declared in the Hopper tree because
    /// rule S1 refuses new cross-hardware symlinks — and the arm reassociates
    /// the k-reduction into the MMA tree, so promotion needs the ssm-poisoning
    /// tripwire rather than a cosine (#928).
    /// It is here so the probe that loads it is GATED on the same bit that
    /// launches it, like every other kernel in this table.
    pub gdn_prefill_tc: bool,
    /// `dense_gemm_ba_gates_prefill_hopper` serves the SSM BA projection +
    /// GDN gate transforms with ONE CTA per token
    /// (`layers/ops/ssm_ba_gates_hopper.rs`), in place of its gb10 parent's
    /// `ceil(N/4)` CTAs per token.
    ///
    /// TRUE on hopper, false elsewhere. The twin is BIT-IDENTICAL to the
    /// parent by construction — same lane-strided K sweep, same butterfly,
    /// same cross-warp order — so the row is purely a speed claim, and the
    /// claim is about ISSUED work: at N=96 the parent re-reads and re-converts
    /// each token's whole `K=5120` activation row 96 times, once per BA output
    /// (nsys round 13: 26 881.8 us = 5.85% of a 4593-token H100 prefill, at
    /// 88 GB/s of compulsory traffic — 2.6% of HBM, so not a bandwidth bound).
    /// The twin reads it 12 times and issues ~1.8x fewer instructions for the
    /// same bits (SASS, sm_90a). `kernels/gb10`
    /// and `kernels/b200` do not carry the source, so the row is INERT there
    /// and declared only because the lever list is one list (#928).
    /// `ATLAS_SSM_BA_GATES_HOPPER=0` is the A/B. Numbers:
    /// `SSM-BA-GATES-ATTRIBUTION.md`.
    pub ssm_ba_gates_hopper: bool,
    /// `per_token_group_quant_fp8_hopper` serves the per-token FP8 activation
    /// quantizer with 16 threads per 128-element K-group and **8 groups per
    /// CTA** (`layers/ops/fp8_act_quant.rs`), in place of its gb10 parent's
    /// one CTA per group.
    ///
    /// TRUE on hopper, false elsewhere. The twin is BIT-IDENTICAL to the
    /// parent — same `amax / 448.0f`, same `1e-12f` floor, same per-element
    /// `div.rn.f32`, same saturating E4M3 convert; only the reduction tree
    /// moves — so the row is purely a speed claim, and it is a claim with a
    /// WIDTH. `native_fp8_act_quant_hopper_microtest`, 1xH100 80GB HBM3,
    /// round 16: **3.30-3.59x at M in {1168, 4576}** (63.7-68.4% of HBM
    /// against the parent's 18.6-19.1%) and **0.76x-0.95x at M in {16, 17, 25}
    /// for K in {5120, 6144}** — 8 groups per CTA is 8x fewer CTAs, and at
    /// those M the parent's grid is already under one wave on 132 SMs.
    ///
    /// So the row arms a kernel that is also behind a CTA-count floor
    /// (`layers/ops/fp8_act_quant_floor.rs`): the twin takes a launch only
    /// when its own grid clears `2 * sm_count` CTAs. `kernels/gb10` and
    /// `kernels/b200` do not carry the source, so the row is INERT there and
    /// declared only because the lever list is one list.
    /// `ATLAS_FP8_ACT_QUANT_HOPPER=0` is the A/B. Numbers:
    /// `FP8-ACT-QUANT-ATTRIBUTION.md`.
    pub fp8_act_quant_hopper: bool,
    /// Split SiLU+down on the decode path (`ModelLevers::decode_split_silu`).
    pub decode_split_silu: bool,
    /// How the paged-decode attention path picks its KV split count (#928):
    /// `legacy` (the pre-#928 rule), `auto` (fill this target's SMs at the
    /// single-stream shape) or a pinned decimal count. Parsed by
    /// [`crate::attn_splitk::parse`], which owns the grammar and the clamps.
    ///
    /// A STRING rather than a number: the policy is a small grammar, and the
    /// target declares WHICH RULE it wants rather than a count that would
    /// silently be wrong on the next card.
    pub attn_decode_splitk: &'static str,
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
    /// The `dense_gemm_m16_bf16` tensor-core arm on the BF16 decode head
    /// (`model/trait_impl/lm_head_batched.rs`), for 5..16 rows.
    ///
    /// TRUE on hopper: H100 round 9 cell Y against cell U, C=16 aggregate
    /// 235.47 -> 245.10 (+4.09%), TPOT 53.42 -> 50.79 ms (-4.92%). The head is
    /// ~7% of the step (round 12 nsys: 943.8 us, 4.31%), so this is most of
    /// what is there to win at that site.
    ///
    /// 🔴 It REASSOCIATES the K reduction against `dense_gemv_bf16`, and at the
    /// LM head that is a token-visible seam rather than a rounding detail — a
    /// near-tie argmax can flip. It is ON here on a measured serve receipt, not
    /// on a microtest.
    pub lm_head_m16_tc: bool,
    /// `w8a16_gemv_batch16_ncol{2,4}` on the decode attention projections
    /// (`layers/qwen3_attention/attn_ncol_gemv.rs`).
    ///
    /// FALSE everywhere, INCLUDING hopper, and the reason is stated rather
    /// than implied: there is no serving A/B for it on any target. The
    /// microtest exists; the receipt does not. The row is here so the kernel
    /// has a declared way to be turned on for the measurement that would earn
    /// it, not because it has been shown to pay.
    pub attn_ncol_gemv: bool,
    /// One fused `[gate | up]` cuBLASLt W8A8 GEMM at `N = 2 * intermediate` on
    /// the 5..=16-row decode band, instead of two at `N = intermediate`
    /// (#927). TRUE only on hopper: the arm's strided-SiLU consumer
    /// (`silu_mul_strided.cu`) is a Hopper-owned source, so the row is inert
    /// on a target whose tree does not carry it.
    pub ffn_gateup_fused: bool,
}
