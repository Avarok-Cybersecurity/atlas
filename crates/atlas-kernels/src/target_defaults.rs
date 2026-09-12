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
    /// widths (`dense_ffn_batch16_decode.rs`). FALSE on every target: on H100
    /// the cuBLASLt FFN arm owns those same widths and beat it (-5.4%
    /// aggregate, +50 ms TTFT), and on GB10 the kernel is not in the set at
    /// all. `ATLAS_FFN_BATCH16=1` is the opt-in.
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
    /// `gated_delta_rule_decode_f32{,_strided}_hopper` serve the GDN decode
    /// recurrence (`layers/ops/ssm_gdn_hopper.rs`), in place of their gb10
    /// parents.
    ///
    /// FALSE on every target. The twins are BIT-IDENTICAL to their parents on
    /// all 12 microtest legs, so this row is purely a speed claim, and on H100
    /// the claim is negative: 0.83x at contiguous n=1, +6.8% per C=1 step in
    /// nsys, -0.4% on the serve A/B (round 12, `GDN-DECODE-ATTRIBUTION.md`).
    /// GB10 does not compile the kernel at all; B200 does (its `common/` tree
    /// symlinks Hopper's) but has no receipt. `ATLAS_GDN_DECODE_HOPPER=1` is
    /// the positive lever, and the legacy `ATLAS_NO_GDN_HOPPER=1` kill switch
    /// still outranks it.
    pub gdn_decode_hopper: bool,
    /// `gated_delta_rule_decode_f32_strided_hopper_smem` serves the BATCHED
    /// GDN decode recurrence (`layers/ops/ssm_gdn_strided_hopper.rs`), in
    /// place of its gb10 parent, at n >= 4.
    ///
    /// TRUE on hopper, false elsewhere. A DIFFERENT lever from
    /// `gdn_decode_hopper` above and a different kernel: that one
    /// re-partitions state COLUMNS to fill a 132-SM device at n=1 and lost;
    /// this one keeps the parent's partition exactly — it has to, because the
    /// `kd` reduction is a serial f32 chain — and reads the state ONCE for 96
    /// of its 128 rows instead of twice. nsys round 13 cell V prices the
    /// parent at 2 748.9 us = 13.82% of a 19.887 ms n=16 step, 57.27 us per
    /// launch, issuing 150.99 MB where 100.66 MB is compulsory
    /// (`GDN-DECODE-ATTRIBUTION.md`, "Round 17").
    ///
    /// On without a serving receipt because it cannot change a bit of output
    /// (`native_gdn_decode_hopper_microtest` asserts byte equality, not a
    /// tolerance) and because its occupancy is the parent's — six resident
    /// CTAs per SM at 80 registers and 37 904 B of smem, against the 5.82 the
    /// n=16 grid supplies. `ATLAS_GDN_DECODE_STRIDED_HOPPER=0` is the
    /// one-variable A/B; `ATLAS_NO_GDN_HOPPER=1` outranks it, the same kill
    /// switch that outranks `gdn_decode_hopper`.
    pub gdn_decode_strided_hopper: bool,
    /// `gated_delta_rule_chunk_delta_h_tcfuse_x2` serves the GDN chunked
    /// PREFILL state spine on tensor cores (`layers/ops/ssm_gdn_a3.rs`).
    ///
    /// FALSE on every target, deliberately. The kernel is shared — it lives in
    /// `kernels/gb10/common/gated_delta_rule_chunk_tc.cu` and is validated on
    /// GB10 — and the arm reassociates the k-reduction into the MMA tree, so
    /// promotion needs the ssm-poisoning tripwire rather than a cosine (#928).
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
    /// The dense FFN's gate and up projections run as ONE block-scaled FP8
    /// GEMM at `N = 2 * intermediate` on the 5..=16-row decode band, instead of
    /// two at `N = intermediate` (`layers/dense_ffn_gateup_fused.rs`).
    ///
    /// TRUE on hopper, false elsewhere. The two GEMMs read the same weight
    /// bytes either way, so this is not a traffic claim — it is a per-launch
    /// one. nsys, 1xH100 80GB HBM3, Qwen/Qwen3.8-27B-FP8 @ `3717cb05e`, round
    /// 13 cell V (`h100-r13-attribution.md` SS C.2-C.4): at n=16 the pair is
    /// 128 graph nodes, 5 730.5 us = 44.77 us/node for 89.1 MB of weights =
    /// **1 991 GB/s, 59.4% of HBM**, while `down` (same bytes, one launch,
    /// K=17408 N=5120) reaches 71.4% and SSM `in_proj_qkvz` (N=16384) 73.2%.
    /// One launch of twice the N halves the per-launch fixed cost and doubles
    /// the tile count per wave; at an 80% target that is **1 476 us of a
    /// 19.887 ms step (7.4%)**.
    ///
    /// gb10 and b200 declare FALSE — no receipt on either, and b200 must not
    /// inherit a Hopper recipe by resemblance. `ATLAS_FFN_GATEUP_FUSED=0` is
    /// the A/B. Numbers: `FFN-GATEUP-FUSION-ATTRIBUTION.md`.
    pub ffn_gateup_fused: bool,
    /// The attention Q/K/V decode projections run as ONE block-scaled FP8
    /// cuBLASLt GEMM at `N = q_proj_dim + 2*kv_dim` on the 5..=16-row decode
    /// band, instead of three — `q_proj` at N=12288 plus `k_proj` and
    /// `v_proj` at N=1024 each
    /// (`qwen3_attention/trait_impl/multi_seq/qkv_fused.rs`).
    ///
    /// TRUE on hopper, false elsewhere. Same shape of claim as
    /// [`Self::ffn_gateup_fused`]: the weight bytes are read once either way,
    /// so this is a per-LAUNCH row. nsys `--cuda-graph-trace=node`, 1xH100
    /// 80GB HBM3, Qwen/Qwen3.8-27B-FP8 @ `3717cb05e`, round 13 cell V
    /// (`h100-r13-attribution.md` SS C.2-C.4): at n=16 `q_proj` is 16 graph
    /// nodes, 460.0 us = 28.75 us/node = 2 189 GB/s (65.3% of HBM), while
    /// `k_proj` + `v_proj` are 32 nodes, 510.9 us = 15.97 us/node for a
    /// **10.5 MB** weight read = **328 GB/s, 9.8% of HBM**. A 10 MB GEMM
    /// cannot amortise a launch; appended onto `q_proj` it is 8 more 128-wide
    /// N tiles on a wave that is already running. Rank 5 of the round-13
    /// decode table: **428 us of a 19.887 ms step (2.2%)**.
    ///
    /// gb10 and b200 declare FALSE — no receipt on either, and both declare
    /// `cublas_gemm_scope` without the attention arm this row changes.
    /// `ATLAS_ATTN_QKV_FUSED=0` is the A/B. Numbers:
    /// `ATTN-QKV-FUSION-ATTRIBUTION.md`.
    pub attn_qkv_fused: bool,
    /// Split SiLU+down on the decode path (`ModelLevers::decode_split_silu`).
    pub decode_split_silu: bool,
    /// `auto` — size the decode-rollback ring from free memory at preflight
    /// (#915) — or a decimal depth `0..=DECODE_ROLLBACK_RING_SLOTS`.
    pub ssm_decode_ring_slots: &'static str,
    /// Upper `M` for the W8A8 block-scaled dense-FFN prefill on a WIDENING
    /// projection (`n > k`: gate/up). `u32::MAX` = no cap, the baseline.
    ///
    /// W8A8 feeds the FP8 tensor cores instead of dequantizing into a BF16
    /// MMA, and on H100 that is 2.0-3.1x at every M measured — so Hopper
    /// declares nothing here and keeps the baseline. On sm_121 it is not: W8A8
    /// throughput is FLAT at ~14 TFLOP/s from M=128 to M=2048 while W8A16
    /// climbs to ~26 and stays there. A kernel whose throughput does not move
    /// with M is not compute-bound — it is pinned by the per-token activation
    /// quantization and its FP32 scale epilogue, which W8A16 never pays. So
    /// W8A8 wins only while the GEMM is small enough that the quantization is
    /// not the bill, and where that stops is a property of the ARCH.
    ///
    /// Two rows and not one because the crossover is shape-dependent: measured
    /// 2026-09-11 on spark-256a at the real Qwen3.8-27B dims, gate/up
    /// (N=17408, K=5120) crosses at M~64-128 and down (N=5120, K=17408) at
    /// M~384-512.
    pub w8a8_prefill_max_m_widening: u32,
    /// Upper `M` for the same path on a NARROWING projection (`n <= k`: down).
    /// See [`Self::w8a8_prefill_max_m_widening`].
    pub w8a8_prefill_max_m_narrowing: u32,
    /// How the paged-decode attention path picks its KV split count (#928):
    /// `legacy` (the pre-#928 rule), `auto` (fill this target's SMs at the
    /// single-stream shape) or a pinned decimal count. Parsed by
    /// [`crate::attn_splitk::parse`], which owns the grammar and the clamps.
    ///
    /// A STRING for the reason `cublas_gemm_scope` is one: the policy is a
    /// small grammar, not a bool, and the target declares WHICH RULE it wants
    /// rather than a number that would silently be wrong on the next card.
    pub attn_decode_splitk: &'static str,
}
