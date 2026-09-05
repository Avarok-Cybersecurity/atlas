// SPDX-License-Identifier: AGPL-3.0-only

//! GDN / SSM decode-path flags, resolved ONCE from the serve command line.
//!
//! These three select KERNELS on the GDN decode path, and they are coupled:
//! the FP16 h-state twins only exist on the fused-norm arm, so `h_f16` without
//! `fused_norm` reaches an FP32-only kernel that would read the FP16 pool as
//! FP32 — plausible numbers, silent garbage. That coupling is checked at serve
//! time by `spark-server`'s arg validation, not discovered at the first decode
//! step.
//!
//! ## Why these are set, not read
//!
//! They were three independent `std::env::var` reads scattered across six call
//! sites, each with its own convention (`ATLAS_SSM_H_FP16` presence-gated —
//! where `=0` meant ON — and the other two `== "1"`). That is how the same
//! flag came to be decoded two different ways in one binary. They are now ONE
//! cell, written once from [`set_from_cli`] before any model is built.
//!
//! The environment variables remain honoured when the setter never runs (a
//! test, a microbenchmark example, an older script), so nothing that worked
//! before stops working; the CLI wins when both are present.
//!
//! Follow-up: this is process-scoped, so a hot-swap to a model with a
//! different recipe keeps the first model's kernel selection. The proper home
//! is `ModelLevers`, which is carried per model — deferred because the h-state
//! dtype is read from `SsmLayerState` construction sites that have no
//! `ForwardContext`.

/// The resolved flags. `None` until `set_from_cli` or the first env fallback.
static FLAGS: std::sync::OnceLock<GdnFlags> = std::sync::OnceLock::new();

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GdnFlags {
    /// `--ssm-h-dtype f16`: store the GDN decode h-state as FP16.
    pub h_f16: bool,
    /// Stage 3 of the f16 h-state: additionally SIZE the h pools at 2 bytes
    /// per element. Must imply `h_f16` (a narrow pool holding FP32 would be
    /// an OOB write, not a mode). NOT serveable yet and therefore has NO
    /// CLI surface — the CLI mapping always publishes `false`, and
    /// `ssm_h_fp16_preconditions` refuses it besides (defense in depth) —
    /// but the sizing plumbing keys off THIS field so the pool, preflight
    /// and every byte-copier already agree on the storage width when
    /// prefill narrowing lands.
    pub h_f16_pool: bool,
    /// `--gdn-fused-norm`: fused GDN output-norm decode kernel.
    pub fused_norm: bool,
    /// `--ssm-batched-recurrent`: one strided recurrent launch per batch.
    pub batched_recurrent: bool,
    /// `--exact-verify`: run the sequential-decode-EXACT per-token MTP-verify
    /// chain (issue #435 route (a)) instead of the default WY-chunkwise /
    /// fused BF16-conv arms. OPT-IN, default OFF; the measured decode-step
    /// cost (~+22-36% at the n=8/16/32 verify rungs) is why.
    ///
    /// SCOPE: this makes the GDN/SSM verify chain exact. It does NOT deliver
    /// end-to-end spec-on == spec-off, because every FFN and attention
    /// projection dispatches on ROW COUNT (verify K=4 takes
    /// `w4a16_gemv_batch4`, decode takes `w4a16_gemv`) and those separate
    /// implementations round differently — ~5e-5 of lanes by 1 ULP, on every
    /// shape measured (#459). Closing that needs single-row routing for the
    /// whole verify forward, which is future work.
    pub exact_verify: bool,
}

impl GdnFlags {
    /// Whether the MTP-verify pass must run the sequential-decode-exact
    /// conv+GDN chain (issue #435 route (a)). Default FALSE: exact verify is
    /// opt-in via `--exact-verify`, so with default settings spec-on output
    /// is NOT bitwise-equal to spec-off (the #435 divergence ships).
    ///
    /// Pure so it is testable without touching the process-global flags cell.
    /// `h_f16` forces non-exact even when requested, because an FP16 h-state
    /// is a whole-chain numerics change that is not bit-comparable to the
    /// FP32 reference in the first place, and the exact arm's kernels are
    /// FP32 readers (reading the FP16 pool through them would be silent
    /// garbage, not an error). CLI validation additionally REJECTS the
    /// explicit pair, so this clause is defense in depth, not the interface.
    pub fn verify_exact_active(self) -> bool {
        self.exact_verify && !self.h_f16
    }
    /// The legacy environment reading, used when the CLI never set anything.
    ///
    /// `ATLAS_SSM_H_FP16` stays PRESENCE-gated here on purpose: that is how
    /// every script and ledger in the campaign wrote it, and silently changing
    /// `=0` from ON to OFF would retroactively re-label measurements. New
    /// configuration should use `--ssm-h-dtype`.
    fn from_env() -> Self {
        Self {
            h_f16: std::env::var("ATLAS_SSM_H_FP16").is_ok(),
            // No environment fallback on purpose (house rule: no new env
            // knobs) — stage 3 has no CLI surface either until prefill
            // narrowing lands; only unit tests exercise the sizing.
            h_f16_pool: false,
            fused_norm: std::env::var("ATLAS_GDN_FUSED_NORM").as_deref() == Ok("1"),
            batched_recurrent: std::env::var("ATLAS_SSM_BATCHED_RECURRENT").as_deref() == Ok("1"),
            // No legacy environment variable on purpose (house rule: CLI flags
            // or defaults, no new env knobs). Default = the legacy WY arms;
            // exact verify is CLI-opt-in only (`--exact-verify`).
            exact_verify: false,
        }
    }
}

/// Publish the command line's resolution. Call once, before the model builds.
///
/// Returns the value in force, which is the argument unless something already
/// read a flag (in which case the read wins and the caller should say so
/// rather than pretend the setting took).
pub fn set_from_cli(flags: GdnFlags) -> GdnFlags {
    let _ = FLAGS.set(flags);
    *FLAGS.get().expect("just set")
}

/// The resolved flags, falling back to the environment on first touch.
pub fn flags() -> GdnFlags {
    *FLAGS.get_or_init(GdnFlags::from_env)
}

/// Widest chain-verify K with an FP16 h-state twin
/// (`gated_delta_rule_wy{5..16}_f16`).
///
/// The SSOT for "can this verify width run under the f16 pool". K=17 — the
/// DFlash arm at gamma 16 — has no twin, and the FP32 wy17 kernel over an
/// FP16 h-state emits fluent garbage rather than faulting, so the CLI
/// validator and the serve preflight both gate on this instead of a literal.
///
/// Expressed as K, not gamma: the DFlash verify width is gamma + 1, and
/// conflating the two is how the width check came to admit gamma 16 (K=17,
/// no twin) while its message claimed to cover "widths 5..16".
pub const MAX_F16_TWIN_K: usize = 16;

/// The largest `--dflash-gamma` whose verify width still has an FP16 twin.
pub const MAX_F16_TWIN_DFLASH_GAMMA: usize = MAX_F16_TWIN_K - 1;

/// The served DFlash gamma for a drafter of this trained block size, when
/// no `--dflash-gamma` was given.
///
/// THE SSOT, and it must stay that way: the drafter head resolves its gamma
/// through this, and so does every preflight that sizes a pool or a reserve
/// from a peeked `dflash_config.block_size`. Two spellings of this rule is
/// how the SSM MTP intermediates came to be reserved for K=9 while verify
/// asked for K=10, a hard error at the first verify step, mid-graph-capture.
///
/// `block + 2`, because these drafters chain PAST their trained block:
/// measured on Qwen3.8-27B DFlash2 (block 8), gamma 8 runs 7 drafts, one
/// short, while gamma 10 holds full-block 9/9 accepts and is the fastest
/// measured serve (63.0 vs 56.2 tok/s on GB10, 2026-08-29).
///
/// Clamped to `MAX_F16_TWIN_DFLASH_GAMMA` so a block-16-class drafter lands
/// on 15, the widest verify width with kernel coverage under both h-state
/// dtypes, instead of gamma 18 / K=19, which no wyN kernel serves.
pub const fn default_dflash_gamma(trained_block_size: usize) -> usize {
    let bumped = trained_block_size + 2;
    if bumped > MAX_F16_TWIN_DFLASH_GAMMA {
        MAX_F16_TWIN_DFLASH_GAMMA
    } else {
        bumped
    }
}

/// `--ssm-h-dtype f16` (legacy `ATLAS_SSM_H_FP16`).
pub fn ssm_h_fp16_enabled() -> bool {
    flags().h_f16
}

/// Stage 3 of the f16 h-state: h pools SIZED at 2 bytes/element
/// (`--ssm-h-dtype f16-pool`). Implies [`ssm_h_fp16_enabled`] — a narrow
/// pool holding FP32 would be an OOB write, not a mode — which
/// [`ssm_h_dtype_bits`] guarantees at the one place the value is decoded.
pub fn ssm_h_f16_pool_enabled() -> bool {
    flags().h_f16_pool
}

/// SSOT decode of `--ssm-h-dtype` into the two h-state bits it publishes:
/// `(h_f16, h_f16_pool)`.
///
/// Both the CLI validator (which rejects the pairs the mode cannot serve)
/// and `publish_kernel_flags` (which publishes the cell the kernels
/// dispatch on) go through THIS, so a validator that accepted one reading
/// while the kernels took another is not expressible. Anything that is not
/// exactly `f16` or `f16-pool` — including `f32` and an absent flag — is
/// FP32; `check_enum` has already rejected unknown spellings by the time
/// this runs, and defaulting an unknown one to FP32 here is the safe arm
/// besides.
pub fn ssm_h_dtype_bits(dtype: Option<&str>) -> (bool, bool) {
    match dtype {
        Some("f16") => (true, false),
        // f16-pool is f16 PLUS the narrow pool: never one without the other.
        Some("f16-pool") => (true, true),
        _ => (false, false),
    }
}

/// `--gdn-fused-norm` (legacy `ATLAS_GDN_FUSED_NORM=1`).
pub fn gdn_fused_norm_enabled() -> bool {
    flags().fused_norm
}

/// `--ssm-batched-recurrent` (legacy `ATLAS_SSM_BATCHED_RECURRENT=1`).
pub fn ssm_batched_recurrent_enabled() -> bool {
    flags().batched_recurrent
}

/// `--exact-verify` given (and h-state is FP32): the MTP-verify pass runs
/// the sequential-decode-exact chain. FALSE by default — without the flag the
/// verify pass runs the WY/chunkwise arms and #435's spec-on/spec-off output
/// divergence remains. See [`GdnFlags::verify_exact_active`].
pub fn verify_exact_enabled() -> bool {
    flags().verify_exact_active()
}

/// Does THIS pass have to run its K verify rows as K sequential DECODE rows —
/// the same kernels, at the same launch geometry, that `decode()` would have
/// run — instead of the row-count-shaped batched arms?
///
/// PURE (SBIO): every input is a parameter, so the decision is decidable
/// without a GPU, a layer, or a process-global read. Every site that consumes
/// it MUST read the same predicate: the conv+GDN arm writes the block's final
/// normed rows itself, so the phase-8 norm has to skip on exactly the same
/// answer or the rows are normalised twice.
///
/// * `exact_verify` — the global `--exact-verify` opt-in, which applies to
///   every verify body (DFlash, the batched multi-seq verify, this one).
/// * `pass_exact_replay` — `ForwardContext::gdn_exact_replay`, THIS pass's own
///   "reproduce the token-sequential recurrence bitwise" contract. The mHC MTP
///   verify (`model/trait_impl/verify_hc.rs`) is the only `decode_batched`
///   caller that sets it; every other one passes `false`, so this widens
///   nothing else.
/// * `lever` — the kill switch (`ATLAS_NO_VERIFY_ROW_EXACT`), so the row-shaped
///   arms stay measurable against the batched ones.
/// * `h_f16` — an FP16 h-state pool. The exact arm's kernels are FP32 readers;
///   reading an FP16 pool through them is silent garbage, not an error. Same
///   clause, same reason, as [`GdnFlags::verify_exact_active`].
pub const fn verify_row_exact_required(
    exact_verify: bool,
    pass_exact_replay: bool,
    lever: bool,
    h_f16: bool,
) -> bool {
    (exact_verify || (pass_exact_replay && lever)) && !h_f16
}

/// Kill switch for the pass-scoped row-exact verify arms
/// (`ATLAS_NO_VERIFY_ROW_EXACT`). PRESENCE check per the house convention
/// (`=0` is NOT "off"), read once per process. `--exact-verify` is a separate,
/// wider opt-in and is unaffected.
fn row_exact_lever() -> bool {
    static ON: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *ON.get_or_init(|| std::env::var_os("ATLAS_NO_VERIFY_ROW_EXACT").is_none())
}

/// [`verify_row_exact_required`] resolved against the process flags, for a pass
/// whose `ForwardContext::gdn_exact_replay` is `pass_exact_replay`.
pub fn verify_row_exact_for_pass(pass_exact_replay: bool) -> bool {
    let f = flags();
    verify_row_exact_required(f.exact_verify, pass_exact_replay, row_exact_lever(), f.h_f16)
}

/// Which stage of the row-exact chain a caller is asking about.
///
/// The chain is four independent legs, and each costs differently: the two
/// `hc_pre` collapses (K cuBLASLt GEMM triples instead of one), the GDN
/// projections + BA gates (K weight passes instead of one), the conv+GDN
/// recurrence (the exact per-token chain instead of the WY arms) and the MoE
/// (K single-row expert passes instead of the fused K=2 one). Naming them
/// separately is what makes "which leg buys the bit-equality, and what does it
/// cost" a measurement rather than an argument — each has its own PRESENCE
/// kill switch, and `ATLAS_NO_VERIFY_ROW_EXACT` still disarms all four.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum RowExactLeg {
    /// The two mHC `hc_pre` sites (`ATLAS_NO_VERIFY_ROW_HC`).
    HcPre,
    /// GDN QKVZ / out_proj / BA gates. The ONLY leg that is default-OFF, and
    /// the only one that was MEASURED not to matter: `exl3_gemv` does select
    /// its kernel instance by row count (`_m0_` at m == 1, `_m1_` at 2..=8),
    /// but the two agree bit-for-bit on row 0 — the 40-token probe scored
    /// 38/38 equal verify rows with this leg disarmed and 0/38 with either of
    /// the other three disarmed. Arm it with `ATLAS_VERIFY_ROW_PROJ=1` if a
    /// checkpoint ever contradicts that; it costs a second pass over the GDN
    /// in_proj + out_proj trellises (~29 MB/layer) per extra row.
    Proj,
    /// The conv + GDN recurrence and its norm (`ATLAS_NO_VERIFY_ROW_GDN`).
    ConvGdn,
    /// The MoE / FFN (`ATLAS_NO_VERIFY_ROW_FFN`).
    Ffn,
}

impl RowExactLeg {
    /// The leg's env knob, and whether that knob DISARMS a default-on leg
    /// (`true`) or ARMS a default-off one (`false`). PRESENCE-checked either
    /// way, per the house convention (`=0` is NOT "off").
    const fn env(self) -> (&'static str, bool) {
        match self {
            Self::HcPre => ("ATLAS_NO_VERIFY_ROW_HC", true),
            Self::Proj => ("ATLAS_VERIFY_ROW_PROJ", false),
            Self::ConvGdn => ("ATLAS_NO_VERIFY_ROW_GDN", true),
            Self::Ffn => ("ATLAS_NO_VERIFY_ROW_FFN", true),
        }
    }
}

/// Is `leg` of the row-exact chain armed for a pass with this
/// `gdn_exact_replay`? The master predicate AND the leg's own kill switch.
/// Reads are cached per leg, so this is safe inside a layer loop.
pub fn verify_row_exact_leg(pass_exact_replay: bool, leg: RowExactLeg) -> bool {
    static LEGS: std::sync::OnceLock<[bool; 4]> = std::sync::OnceLock::new();
    let on = LEGS.get_or_init(|| {
        [
            RowExactLeg::HcPre,
            RowExactLeg::Proj,
            RowExactLeg::ConvGdn,
            RowExactLeg::Ffn,
        ]
        .map(|l| {
            let (name, kill) = l.env();
            std::env::var_os(name).is_some() != kill
        })
    });
    let idx = match leg {
        RowExactLeg::HcPre => 0,
        RowExactLeg::Proj => 1,
        RowExactLeg::ConvGdn => 2,
        RowExactLeg::Ffn => 3,
    };
    verify_row_exact_for_pass(pass_exact_replay) && on[idx]
}

/// Batch width at which the multi-seq decode projections switch to the
/// 128-row M-tile. `None` (kill switch `ATLAS_NO_SSM_M128`, PRESENCE check —
/// `=0` is NOT "off") keeps the 64-row twin at every width.
///
/// 65 is the DERIVED crossover, not a tuned constant: `ceil(m/64) >
/// ceil(m/128)` first holds at m=65, so m<=64 gains no weight-read reduction
/// from the wider tile and would only pad MMA rows. Identical rule to the
/// dense-FFN prefill macro's `m <= 64` small-M arm.
pub(crate) fn ssm_m128_min_m() -> Option<u32> {
    static M: std::sync::OnceLock<Option<u32>> = std::sync::OnceLock::new();
    *M.get_or_init(|| {
        if std::env::var("ATLAS_NO_SSM_M128").is_ok() {
            None
        } else {
            Some(65)
        }
    })
}

#[cfg(test)]
mod tests {
    use super::{GdnFlags, ssm_h_dtype_bits, verify_row_exact_required};

    const BASE: GdnFlags = GdnFlags {
        h_f16: false,
        h_f16_pool: false,
        fused_norm: false,
        batched_recurrent: false,
        exact_verify: false,
    };


    // ── The pass-scoped row-exact contract (2026-09-03) ──
    //
    // The mHC MTP verify re-processes an ALREADY-COMMITTED token as its row 0,
    // so that row's logits must equal the serial `decode()` that produced it.
    // It cannot: `decode_batched_block` dispatches every stage on ROW COUNT —
    // `exl3_gemv` picks its `_m0_` instance at m == 1 and `_m1_` at 2..=8, the
    // conv+GDN block takes the WY/BF16-conv arms, the MoE takes `forward_k2` —
    // and those are different compiled bodies, not different spellings of one.
    // Measured on qwen3.8-flash-next (native EXL3, gamma=1): row-0 logits
    // matched serial decode 0/N with the batched arms.
    //
    // `ForwardContext::gdn_exact_replay` is that pass's declaration, and these
    // pin that it — alone, with NO CLI flag — selects the row-exact chain.

    /// POSITIVE, and the leg that FAILS without the fix: a pass that declares
    /// `gdn_exact_replay` gets the row-exact chain even though `--exact-verify`
    /// was never given. The old predicate for the same three sites was
    /// `verify_exact_active()`, which is false here (asserted alongside, so
    /// this test states the behaviour CHANGE and not merely a new true).
    #[test]
    fn a_pass_declaring_exact_replay_is_row_exact_without_the_cli_flag() {
        assert!(
            verify_row_exact_required(false, true, true, false),
            "the verify pass's own gdn_exact_replay contract must select the \
             row-exact chain"
        );
        assert!(
            !BASE.verify_exact_active(),
            "and it must do so WITHOUT --exact-verify — that is the change"
        );
    }

    /// NEGATIVE: every other `decode_batched` caller passes
    /// `gdn_exact_replay: false` (DFlash, verify_c2/e, the batched multi-seq
    /// verify, plain decode). They keep the batched arms — this widens nothing.
    #[test]
    fn passes_that_do_not_declare_exact_replay_keep_the_batched_arms() {
        assert!(!verify_row_exact_required(false, false, true, false));
    }

    /// NEGATIVE: the kill switch (`ATLAS_NO_VERIFY_ROW_EXACT`) puts the
    /// declaring pass back on the batched arms, so the two are A/B-able.
    #[test]
    fn the_kill_switch_restores_the_batched_arms() {
        assert!(!verify_row_exact_required(false, true, false, false));
    }

    /// POSITIVE: `--exact-verify` is the WIDER opt-in and is untouched by the
    /// kill switch — it still selects the exact chain for every verify body.
    #[test]
    fn the_cli_opt_in_is_independent_of_the_pass_scoped_lever() {
        assert!(verify_row_exact_required(true, false, false, false));
    }

    /// NEGATIVE: an FP16 h-state pool forces non-exact from EITHER source. The
    /// exact arm's kernels are FP32 readers; reading the narrow pool through
    /// them is silent garbage, not an error. Same clause as
    /// `verify_exact_active`, restated here because this predicate has a
    /// second way in.
    #[test]
    fn an_f16_h_pool_refuses_the_exact_chain_from_either_source() {
        assert!(!verify_row_exact_required(true, false, true, true));
        assert!(!verify_row_exact_required(false, true, true, true));
    }

    /// POSITIVE (the default): with no flags the verify pass runs the legacy
    /// WY/chunkwise arms, NOT the exact chain. Exact verify became OPT-IN
    /// (every surveyed production engine ships exactness opt-in; its measured
    /// decode-step cost here is ~+22-36%), so the #435 divergence is the
    /// documented default behaviour — this test pins that polarity.
    #[test]
    fn legacy_wy_verify_is_the_default() {
        assert!(
            !BASE.verify_exact_active(),
            "default must be the legacy WY arms — exact verify is opt-in"
        );
        // Orthogonal flags do not sneak exact mode on.
        assert!(
            !GdnFlags {
                fused_norm: true,
                batched_recurrent: true,
                ..BASE
            }
            .verify_exact_active()
        );
    }

    /// POSITIVE (the opt-in): `--exact-verify` selects the exact chain, alone
    /// and beside the orthogonal GDN flags.
    #[test]
    fn exact_verify_flag_selects_the_exact_chain() {
        assert!(
            GdnFlags {
                exact_verify: true,
                ..BASE
            }
            .verify_exact_active()
        );
        assert!(
            GdnFlags {
                exact_verify: true,
                fused_norm: true,
                batched_recurrent: true,
                ..BASE
            }
            .verify_exact_active()
        );
    }

    /// The environment fallback can NEVER turn exact verify on: there is no
    /// `ATLAS_*` variable for it on purpose (house rule: no new env knobs),
    /// so a serve that skips `set_from_cli` still defaults to the WY arms.
    /// Deterministic despite reading the process environment, because only
    /// the `exact_verify` field is asserted and no variable feeds it.
    #[test]
    fn env_fallback_never_enables_exact_verify() {
        assert!(!GdnFlags::from_env().exact_verify);
        // Same rule for the stage-3 pool sizing: no env variable feeds it.
        // `--ssm-h-dtype f16-pool` is the ONLY way to publish it, so a
        // legacy `ATLAS_SSM_H_FP16=1` script keeps the FP32-sized pool.
        assert!(!GdnFlags::from_env().h_f16_pool);
    }

    /// A narrow pool holding FP32 is an out-of-bounds write, not a mode, so
    /// `h_f16_pool` without `h_f16` must not be expressible from any input.
    /// This is the ONE decode both the validator and the publisher use, so
    /// pinning it here pins it for both.
    #[test]
    fn the_pool_bit_is_never_set_without_the_dtype_bit() {
        for (spelling, expected) in [
            (None, (false, false)),
            (Some("f32"), (false, false)),
            (Some("f16"), (true, false)),
            (Some("f16-pool"), (true, true)),
            (Some(""), (false, false)),
            (Some("F16-POOL"), (false, false)),
            (Some("f16 "), (false, false)),
        ] {
            assert_eq!(ssm_h_dtype_bits(spelling), expected, "{spelling:?}");
        }
    }

    /// NEGATIVE: an FP16 h-state forces non-exact EVEN WHEN exact was
    /// requested — the exact arm's FP32 kernels must never read the FP16
    /// pool. (CLI validation rejects the explicit pair; this is the
    /// defense-in-depth layer beneath it.)
    #[test]
    fn h_f16_forces_non_exact_even_when_requested() {
        assert!(
            !GdnFlags {
                exact_verify: true,
                h_f16: true,
                ..BASE
            }
            .verify_exact_active()
        );
        assert!(
            !GdnFlags {
                h_f16: true,
                ..BASE
            }
            .verify_exact_active()
        );
    }
}
