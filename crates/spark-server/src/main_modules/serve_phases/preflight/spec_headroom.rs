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
//! # What keeps the flat 4 GiB
//!
//! DFlash (its drafter loads after this preflight and sizes its own state),
//! `--self-speculative` and `--ngram-speculative` (CUDA-graphed verify), every
//! drafter that is not the Qwen-shaped head (GLM-5.3, DeepSeek-V4, MiniMax,
//! ...), any serve with `AVAROK_SPEC_HEADROOM_FULL` in the environment
//! (presence; the escape hatch back to the old reserve), and every target
//! that does not declare the `memory_fit_consumer` row (all but ada), so
//! GB10, Hopper, B200 and B300 reserve exactly what they did.

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
