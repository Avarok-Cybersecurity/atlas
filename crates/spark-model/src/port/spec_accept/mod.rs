// SPDX-License-Identifier: AGPL-3.0-only
//! Responsibility: device-side speculative acceptance (lane L12, row `mtp_device_accept`): the request
//! the scheduler hands the model, the per-model state behind it, and this module's tree.
//!
//! The MTP verify path's slow tail copies every verify row to the host and runs the pre-sample
//! pipeline there (`spark-server` `verify_pipeline_helper`). For a row the scheduler's planner proves
//! eligible (no grammar, no logit_bias, no LZ/DRY, no stateful thinking stage), it instead describes
//! the pipeline's effect as a sparse edit list ([`EditDesc`]) and asks the model, through
//! `ModelPortExt::spec_accept`, to pick on the device: `kernels/ada/common/spec_accept_sample_ada.cu`.
//!
//! * Greedy rows: bit-identical to the host pick (same edits in the same f32 operations, the same
//!   first-index-wins argmax).
//! * Sampled rows: distribution-exact: accept a one-hot draft with probability p(d),
//!   else sample the residual; with no draft, sample p. The RNG is keyed by (request seed or a fresh
//!   key, absolute position, purpose), so it differs from the host sampler's `StdRng` stream.
//!
//! [`reference`] is the CPU model of the kernel (the oracle every test and the GPU example compare
//! against); [`blob`] is the one encoding both read; [`launch`] runs the kernel; [`metrics`] holds the
//! `atlas_mtp_device_accept_*` series.

use std::sync::{Mutex, OnceLock};

use spark_runtime::gpu::{DevicePtr, KernelHandle};

pub mod blob;
pub mod launch;
pub mod metrics;
pub mod reference;
#[cfg(test)]
mod reference_tests;
#[cfg(test)]
mod reference_tests_b;

/// Kernel module (the `.cu` stem) and entry point.
pub const MODULE: &str = "spec_accept_sample_ada";
/// The one entry point of [`MODULE`].
pub const ENTRY: &str = "spec_accept_rows_ada";
/// Verify positions per request (the drafts region of the blob is this long).
pub const MAX_POS: usize = 32;
/// Regime 0 (the state at step start) and regime 1 (after a `</think>` pick inside thinking).
pub const MAX_REGIMES: usize = 2;
/// Truncated supports are built from at most this many candidates (the kernel's `kCandCap`).
pub const CAND_CAP: usize = 64;
/// Bias entries one edited token may carry, applied in order.
pub const MAX_BIAS: usize = 2;
/// Edit descriptors per regime the blob buffer holds.
pub const MAX_EDITS: usize = 2048;
/// "No draft at this position": sample p directly.
pub const NO_DRAFT: u32 = u32::MAX;
/// Threads per CTA (the kernel's `kBlock`).
pub const BLOCK: u32 = 1024;
/// Dynamic plus static shared memory per CTA, without the opt-in attribute.
pub const SMEM_LIMIT: usize = 48 * 1024;
/// Upper bound of the kernel's static `__shared__` arrays (about 2.6 KiB today).
pub const STATIC_SMEM_RESERVE: usize = 6 * 1024;

/// Per-job status bits the kernel (and [`reference`]) return beside the pick.
pub mod status {
    /// The processed row held a NaN or +inf (information only; greedy stays exact).
    pub const NONFINITE: u32 = 1;
    /// A sampled job found no finite logit: the host must pick.
    pub const EMPTY: u32 = 2;
    /// A sampled job's support does not fit [`super::CAND_CAP`] candidates: the host must pick.
    pub const SUPPORT: u32 = 4;
    /// The job's draft was accepted (information only).
    pub const ACCEPTED: u32 = 8;
    /// Either bit means the job's pick is not valid.
    pub const FALLBACK: u32 = EMPTY | SUPPORT;
}

/// The sampler parameters of a sampled request, as the host verify path resolves them.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct SampleParams {
    /// Strictly positive and finite.
    pub temperature: f32,
    /// 0 = no top-k.
    pub top_k: u32,
    /// 1.0 = no nucleus.
    pub top_p: f32,
    /// 0.0 = no min-p.
    pub min_p: f32,
}

/// Greedy (the penalty-aware argmax) or sampled.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AcceptMode {
    /// First-index-wins argmax of the processed row.
    Greedy,
    /// Sample (or accept a draft from) the processed row's distribution.
    Sample(SampleParams),
}

/// One token the host pre-sample pipeline touches, and how, in its order of application.
#[derive(Clone, Copy, Debug, PartialEq, Default)]
pub struct EditDesc {
    /// Token id, below the vocabulary size; ids are unique and ascending within a regime.
    pub token: u32,
    /// Set to -inf at every position (the think and tool masks).
    pub mask: bool,
    /// Set to -inf at positions below the request's `eos_until` (the `min_tokens` EOS mask).
    pub eos: bool,
    /// `x - 12.0` (the tool-opener suppression outside thinking), when not masked.
    pub sub12: bool,
    /// Repetition-penalty applications (`x > 0 ? x / r : x * r`), one per window occurrence.
    pub rep_count: u32,
    /// `x - delta` (presence and frequency, `freq * count as f32 + pres`).
    pub delta: Option<f32>,
    /// Additive biases, applied in order; the first `n_bias` are live.
    pub bias: [f32; MAX_BIAS],
    /// Live entries of `bias`.
    pub n_bias: u8,
}

/// One sequence's verify span, ready for the device.
#[derive(Clone, Copy, Debug)]
pub struct AcceptRequest<'a> {
    /// First logits row of the span: `[n_pos, vocab]` BF16, row stride `vocab`.
    pub logits: DevicePtr,
    /// Vocabulary size (the row length).
    pub vocab: usize,
    /// Verify positions, 1..=[`MAX_POS`].
    pub n_pos: usize,
    /// One draft per position ([`NO_DRAFT`] where none), or empty for "no drafts".
    pub drafts: &'a [u32],
    /// Greedy or sampled.
    pub mode: AcceptMode,
    /// RNG key: the request seed, or a fresh key per call.
    pub rng_key: u64,
    /// Absolute output position of row 0 (`output_tokens.len()`).
    pub pos_base: u64,
    /// Rows below this index mask the EOS-flagged edits.
    pub eos_until: u32,
    /// The repetition penalty the `rep_count` edits divide or multiply by.
    pub rep_penalty: f32,
    /// One edit list per regime (1 or 2).
    pub regimes: &'a [Vec<EditDesc>],
}

/// One job's answer: the picked token and its [`status`] bits.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub struct JobResult {
    /// Picked token id (meaningless when `status & status::FALLBACK != 0`).
    pub pick: u32,
    /// [`status`] bits.
    pub status: u32,
}

/// Words of the edit bitmap the kernel keeps in shared memory.
pub fn bitmap_words(vocab: usize) -> usize {
    vocab.div_ceil(32)
}

/// Dynamic shared memory of one CTA: the bitmap plus the largest regime's ids and values.
pub fn dyn_smem_bytes(vocab: usize, max_regime_edits: usize) -> usize {
    bitmap_words(vocab) * 4 + max_regime_edits * 8
}

/// The most edits one regime may carry for this vocabulary (0 when the bitmap alone does not fit).
pub fn max_edits(vocab: usize) -> usize {
    let budget = SMEM_LIMIT - STATIC_SMEM_RESERVE;
    (budget.saturating_sub(bitmap_words(vocab) * 4) / 8).min(MAX_EDITS)
}

/// Reject a request the kernel cannot serve. The scheduler's planner never builds one; this is the
/// launch-side guard, so a planner bug is an error, not an out-of-bounds read.
pub fn validate(req: &AcceptRequest<'_>) -> Result<(), String> {
    if req.vocab == 0 || req.vocab >= (1usize << 31) {
        return Err(format!("vocab {} out of range", req.vocab));
    }
    if req.n_pos == 0 || req.n_pos > MAX_POS {
        return Err(format!("{} positions (1..={MAX_POS})", req.n_pos));
    }
    if req.regimes.is_empty() || req.regimes.len() > MAX_REGIMES {
        return Err(format!("{} regimes (1..={MAX_REGIMES})", req.regimes.len()));
    }
    if !req.drafts.is_empty() && req.drafts.len() != req.n_pos {
        return Err(format!(
            "{} drafts for {} positions",
            req.drafts.len(),
            req.n_pos
        ));
    }
    if req
        .drafts
        .iter()
        .any(|&d| d != NO_DRAFT && d as usize >= req.vocab)
    {
        return Err("draft id outside the vocabulary".to_string());
    }
    if let AcceptMode::Sample(p) = req.mode {
        let sane = p.temperature.is_finite()
            && p.temperature > 0.0
            && p.top_p.is_finite()
            && p.min_p.is_finite();
        if !sane {
            return Err(format!("sampler parameters {p:?}"));
        }
    }
    let cap = max_edits(req.vocab);
    for (r, edits) in req.regimes.iter().enumerate() {
        if edits.len() > cap {
            return Err(format!("regime {r}: {} edits (cap {cap})", edits.len()));
        }
        let mut prev: Option<u32> = None;
        for e in edits {
            if e.token as usize >= req.vocab {
                return Err(format!("regime {r}: edit id {} >= vocab", e.token));
            }
            if prev.is_some_and(|p| p >= e.token) {
                return Err(format!("regime {r}: edit ids not strictly ascending"));
            }
            if e.n_bias as usize > MAX_BIAS {
                return Err(format!("regime {r}: {} biases on one id", e.n_bias));
            }
            prev = Some(e.token);
        }
    }
    Ok(())
}

/// Per-model state: the lazily resolved kernel and the device buffers of the request and results.
///
/// The buffers are allocated on first use and live as long as the model (about 100 KiB); a model
/// that never takes the device path allocates nothing.
#[derive(Default)]
pub struct SpecAcceptState {
    pub(crate) kernel: OnceLock<KernelHandle>,
    pub(crate) bufs: Mutex<Option<launch::Buffers>>,
}
