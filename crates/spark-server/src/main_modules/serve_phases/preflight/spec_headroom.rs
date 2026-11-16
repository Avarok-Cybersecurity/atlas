// SPDX-License-Identifier: AGPL-3.0-only

//! The CUDA headroom term of the inference reserve, and how a speculative
//! serve prices it.
//!
//! A sibling of `preflight.rs` (which sits near the 500-line cap), following
//! the `per_sequence_state.rs` precedent.
//!
//! # What the flat 4 GiB was paying for
//!
//! Every serve keeps [`BASE_HEADROOM`] (512 MiB) beside its pools for CUDA
//! context growth and per-launch workspaces. A speculative serve kept 4 GiB,
//! because some of what a drafter allocates is allocated AFTER the KV pool is
//! sized and no other term charged it (`glm5next_mtp_head.rs`, ANOMALIES A59:
//! GLM-5.3's prompt-hidden capture reached 4.0 GiB at `--max-seq-len 524288`).
//!
//! For the Qwen-shaped MTP head (`MtpWeights` + `MtpHead`, built by
//! `impl_a1_init::build_mtp_proposer`) every large allocation is now charged
//! somewhere else:
//!
//! | allocation | charged by |
//! |---|---|
//! | SSM verify intermediates and checkpoints | `ssm_pool_reserve_bytes` (this reserve) |
//! | drafter propose KV pool | `factory::build`'s MTP propose-pool reserve |
//! | draft-only NVFP4 LM head, MTP weights | built before the KV sizing, so in its ledger |
//! | prompt-hidden capture (`impl_a1.rs`) | NOT charged: [`capture_bytes`] here |
//! | head scratch (`mtp_head/new.rs`) | NOT charged: inside [`MTP_MARGIN`] |
//!
//! The head scratch is its batched-prefill buffers (`PREFILL_CHUNK` = 512
//! rows x (7 x hidden + 2 x kv heads x head dim + q heads x head dim) BF16,
//! "~50 MB" at the 27B by the head's own comment, 23.6 MB in the 4B's boot
//! ledger) and 32 propose-metadata slabs. [`MTP_MARGIN`] is 512 MiB, the same
//! allowance [`BASE_HEADROOM`] keeps for everything uncharged on a plain
//! serve, so a Qwen MTP serve keeps 1 GiB plus its capture where it kept
//! 4 GiB: on Qwen3.5-4B at 8,192 tokens, 1,064 MiB instead of 4,096 MiB.
//!
//! # DFlash on a Qwen-shaped target (ada only)
//!
//! `--dflash` with `z-lab/Qwen3.5-4B-DFlash` is priced too, by the same
//! method: charge what nothing else charges, keep the flat 4 GiB as the
//! ceiling. Where the drafter's allocations are charged:
//!
//! | allocation | charged by |
//! |---|---|
//! | drafter weights (1.27 GB BF16) | loaded before the KV sizing, so in `used_so_far` |
//! | drafter KV, fused KV, prompt-hidden capture, FP8 mirrors, 300 MB scratch | `factory::build`'s `dflash_reserve` |
//! | SSM verify pools at K = gamma + 1 | `ssm_pool_reserve_bytes` (this reserve) |
//! | the pools' dummy slot (a base blob plus one verify slot) | NOT charged: [`dflash_pool_dummy_bytes`] here |
//! | per-sequence ctx accumulators (taps x hidden x window, lazily per request) | NOT charged: [`dflash_ctx_accumulator_bytes`] here |
//! | head scratch beyond the 300 MB estimate (the 4B drafter's wider FFN) | NOT charged: inside [`DFLASH_MARGIN`] |
//!
//! The sum is clamped to the flat 4 GiB, so pricing can only lower the
//! reserve, never raise it past what every target has always kept. At the
//! default `--max-batch-size 8` and 8,192 tokens the accumulators alone are
//! 2.5 GiB and the clamp binds (no saving); at `--max-batch-size 1` the 4B
//! reserves about 2.7 GiB instead of 4. UNMEASURED: the receipt is a boot
//! ledger whose outside-the-ledger bytes stay under this term at C=1 and C=8
//! (a GPU script outside this repository), and `AVAROK_SPEC_HEADROOM_FULL` restores the
//! flat reserve.
//!
//! # What keeps the flat 4 GiB
//!
//! `--self-speculative` and `--ngram-speculative` (CUDA-graphed verify), every
//! drafter that is not the Qwen-shaped head (GLM-5.3, DeepSeek-V4, MiniMax,
//! ...), DFlash on a non-Qwen family, any serve with `AVAROK_SPEC_HEADROOM_FULL`
//! in the environment (presence; the escape hatch back to the old reserve),
//! and every target that does not declare the `memory_fit_consumer` row (all
//! but ada), so GB10, Hopper, B200 and B300 reserve exactly what they did.

use avarok_core::config::ModelConfig;

use crate::cli;

const MIB: usize = 1024 * 1024;

/// What every serve keeps for uncharged CUDA allocations.
pub(super) const BASE_HEADROOM: usize = 512 * MIB;

/// The pre-sizing speculative headroom, kept for every drafter not priced
/// here.
pub(super) const FLAT_SPEC_HEADROOM: usize = 4096 * MIB;

/// Beside the capture, what a Qwen MTP head may allocate uncharged.
pub(super) const MTP_MARGIN: usize = 512 * MIB;

/// Beside the pool dummy and the ctx accumulators, what a DFlash head may
/// allocate uncharged: scratch the 4B drafter's 9,216-wide FFN and 8 taps make
/// larger than `factory::build`'s 300 MB estimate (about 400 MB by the head's
/// own sizing comment scaled to 2560 x 9216), plus launch workspaces.
pub(super) const DFLASH_MARGIN: usize = 1024 * MIB;

/// Target-layer taps the per-sequence ctx accumulator holds per position. The
/// widest published Qwen-shaped drafter taps 8 (Qwen3.5-4B-DFlash); the
/// Qwen3.6 drafters tap 5. The preflight cannot read the drafter's config (it
/// may be a remote id), so it charges the widest.
pub(super) const DFLASH_TAP_CEILING: usize = 8;

/// The head's default cap on the ctx accumulator window, in tokens
/// (`dflash_head::dflash_ctx_cap`, `AVAROK_DFLASH_CTX_CAP`, default 16,384).
pub(super) const DFLASH_CTX_CAP: usize = 16 * 1024;

/// Draft count the verify pools are sized for when neither `--dflash-gamma`
/// nor a peeked config states one: `resolved_dflash_gamma`'s ceiling, the same
/// unknown-gamma width `SsmStatePool` allocates, so a miss over-reserves.
pub(super) const DFLASH_POOL_DRAFTS_FALLBACK: usize = 16;

/// The headroom term and a one-line reason for the boot log.
pub(super) fn cuda_headroom(args: &cli::ServeArgs, config: &ModelConfig) -> (usize, String) {
    let spec_on =
        args.speculative || args.self_speculative || args.ngram_speculative || args.dflash;
    let full = std::env::var_os("AVAROK_SPEC_HEADROOM_FULL").is_some();
    let consumer_fit = spark_model::layers::ops::target_defaults::resolved()
        .memory_fit_consumer
        .value;
    cuda_headroom_for(args, config, spec_on, full, consumer_fit)
}

/// [`cuda_headroom`] with its environment inputs passed in. `consumer_fit` is
/// the `memory_fit_consumer` target row: only a target that declares it (ada)
/// prices the Qwen MTP head; every other target keeps the flat 4 GiB.
pub(super) fn cuda_headroom_for(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    spec_on: bool,
    full: bool,
    consumer_fit: bool,
) -> (usize, String) {
    if !spec_on {
        return (BASE_HEADROOM, "no speculative mode".to_string());
    }
    // DFlash on a Qwen-shaped target, ada only (`consumer_fit`): priced, not
    // flat. Every other combination falls through to the old rules below.
    let qwen_dflash_only = args.dflash
        && !args.speculative
        && !args.self_speculative
        && !args.ngram_speculative
        && qwen_mtp_family(config);
    if qwen_dflash_only && consumer_fit && !full {
        return dflash_headroom(args, config);
    }
    let qwen_mtp_only = args.speculative
        && !args.dflash
        && !args.self_speculative
        && !args.ngram_speculative
        && qwen_mtp_family(config);
    if !qwen_mtp_only || full || !consumer_fit {
        let why = if full {
            "AVAROK_SPEC_HEADROOM_FULL"
        } else if !consumer_fit {
            "flat, memory_fit_consumer off for this target"
        } else {
            "drafter not priced, flat"
        };
        return (FLAT_SPEC_HEADROOM, why.to_string());
    }
    let capture = capture_bytes(args, config);
    let total = BASE_HEADROOM + capture + MTP_MARGIN;
    (
        total,
        format!(
            "Qwen MTP head: base {} + prompt capture {} + head margin {} MB",
            BASE_HEADROOM / MIB,
            capture / MIB,
            MTP_MARGIN / MIB,
        ),
    )
}

/// Draft count the DFlash verify pools are sized for: the explicit
/// `--dflash-gamma`, else the unknown-gamma ceiling.
fn dflash_pool_drafts(args: &cli::ServeArgs) -> usize {
    args.dflash_gamma
        .unwrap_or(DFLASH_POOL_DRAFTS_FALLBACK)
        .max(1)
}

/// What `SsmStatePool::new` allocates beyond `ssm_pool_reserve_bytes`: the
/// reserved dummy slot of the base pool (one blob) and of the verify pools
/// (one full-width verify slot). `preflight_reserve` has never counted them
/// ("the CUDA headroom term absorbs it", `ssm_reserve.rs`), so a priced
/// headroom must. One slot's worth of the SSOT reserve is exactly that pair.
/// FP32-width h states and snapshot rollback: the larger of each option.
pub(super) fn dflash_pool_dummy_bytes(args: &cli::ServeArgs, config: &ModelConfig) -> usize {
    spark_model::ssm_reserve::ssm_pool_reserve_bytes(
        1,
        config.num_ssm_layers() * config.ssm_h_state_bytes(),
        config.num_ssm_layers() * config.ssm_conv_state_bytes(),
        true,
        dflash_pool_drafts(args),
        1,
        true,
        false,
        spark_model::ssm_reserve::SsmRollbackMode::Snapshot,
    )
}

/// The drafter's per-sequence ctx accumulators at full batch:
/// `max_batch_size x window x taps x hidden x 2` BF16, with the window the
/// smaller of `--max-seq-len` and the head's ctx cap. Allocated lazily as
/// requests arrive (`alloc_state_windowed`), long after any pool is sized.
pub(super) fn dflash_ctx_accumulator_bytes(args: &cli::ServeArgs, config: &ModelConfig) -> usize {
    dflash_ctx_accumulator_bytes_with_cap(args, config, ctx_window_cap())
}

/// The window cap the accumulators are priced at: the head's compiled-in
/// default, or with `AVAROK_DFLASH_FIT` on the head's own
/// `AVAROK_DFLASH_CTX_CAP` (`0` = uncapped), so the reserve prices the window
/// the head will really allocate.
fn ctx_window_cap() -> usize {
    if super::dflash_fit::enabled() {
        super::dflash_fit::head_ctx_cap()
    } else {
        DFLASH_CTX_CAP
    }
}

/// [`dflash_ctx_accumulator_bytes`] at an explicit window cap (a CPU test
/// drives the opt-in's pricing without the process environment).
pub(super) fn dflash_ctx_accumulator_bytes_with_cap(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    cap: usize,
) -> usize {
    let window = args.max_seq_len.min(cap);
    args.max_batch_size
        .max(1)
        .saturating_mul(window)
        .saturating_mul(DFLASH_TAP_CEILING)
        .saturating_mul(config.hidden_size)
        .saturating_mul(2)
}

/// The priced DFlash headroom and its boot-log reason. Clamped to the flat
/// 4 GiB: pricing never reserves more than every target always has.
fn dflash_headroom(args: &cli::ServeArgs, config: &ModelConfig) -> (usize, String) {
    dflash_headroom_with_cap(args, config, ctx_window_cap())
}

/// [`dflash_headroom`] at an explicit accumulator window cap.
pub(super) fn dflash_headroom_with_cap(
    args: &cli::ServeArgs,
    config: &ModelConfig,
    cap: usize,
) -> (usize, String) {
    let dummy = dflash_pool_dummy_bytes(args, config);
    let acc = dflash_ctx_accumulator_bytes_with_cap(args, config, cap);
    let sum = BASE_HEADROOM
        .saturating_add(dummy)
        .saturating_add(acc)
        .saturating_add(DFLASH_MARGIN);
    let bytes = sum.min(FLAT_SPEC_HEADROOM);
    let why = format!(
        "DFlash drafter: base {} + pool dummy slot {} + ctx accumulators {} + margin {} MB{}",
        BASE_HEADROOM / MIB,
        dummy / MIB,
        acc / MIB,
        DFLASH_MARGIN / MIB,
        if sum > FLAT_SPEC_HEADROOM {
            " (clamped to the flat 4096 MB)"
        } else {
            ""
        },
    );
    (bytes, why)
}

/// The MTP drafter's prompt-hidden capture, `rows x hidden` BF16, allocated
/// after the KV sizing (`impl_a1.rs`, "MTP drafter context: allocating").
/// Its rows are `min(prefill_hidden_rows(max_seq_len), max_seq_len)`, so
/// `max_seq_len` is the bound.
pub(super) fn capture_bytes(args: &cli::ServeArgs, config: &ModelConfig) -> usize {
    args.max_seq_len
        .saturating_mul(config.hidden_size)
        .saturating_mul(2)
}

/// The families whose `--speculative` drafter is the Qwen-shaped `MtpHead`
/// (`load_mtp_weights` -> `MtpWeights`). Named, not inferred: a family that
/// drafts another way keeps the flat headroom until it is priced.
fn qwen_mtp_family(config: &ModelConfig) -> bool {
    let family = config.model_type.to_lowercase().replace(['-', '.'], "_");
    matches!(
        family.as_str(),
        "qwen3_5_moe" | "qwen3_5" | "qwen35_moe" | "qwen35"
    )
}

#[cfg(test)]
#[path = "spec_headroom_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "spec_headroom_dflash_tests.rs"]
mod dflash_tests;
