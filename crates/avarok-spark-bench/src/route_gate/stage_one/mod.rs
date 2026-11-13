// SPDX-License-Identifier: AGPL-3.0-only
// Portions derived from rayline-ai/vllm-arc (vllm_arc/arc_heads.py and vllm_arc/heads.py, commit 4ee6a9f), Copyright
// 2026 Atlas Futures, Inc., licensed under the Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0).
// Translated from Python/torch (via NumPy float64) to Rust f64. Changes: only the branches that the shipped release
// tensors select are kept (no PCA components, no context or warm-cache features, no usage floor head, no shrink, no
// per-member orderer, no quantile_direct orderer, no GPU); torch.searchsorted/gather/einsum became plain loops; the
// loader reads the safetensors file with this crate's own reader and never imports torch or vllm_arc; a tensor that
// would select an unported branch makes the loader refuse instead of guessing; shapes are checked up front. By Atlas
// contributors. The upstream NOTICE text is reproduced in NOTICE-rayline.txt and RAYLINE-ROUTING-ATTRIBUTION.md.
//! Responsibility: the Stage-1 routing head of the ARC-1.0 release in float64 (offline B0 study, CPU only).
//!
//! What it computes, for readout rows `x[n, 1024]` and a regime name per row:
//!
//! ```text
//! z      = (x - mean) * scale
//! eta    = logit_bias[m, c] + regime[m, r] - z . difficulty[m] + sum_r (z . v[m, r]) u[m, c, r]
//!          + <relu(z w1[m] + c1[m]) w2[m] + c2[m], embed[m, c]>
//! p      = sigmoid(clip(eta, -40, 40)), NaN where p_ok is 0                          // [M, n, C]
//! usage  = clip(exp(z . w_usage[m, c, k] + b_usage[m, c, k]) * smear - 1, low, high)  // [M, n, C, 4], NaN if not ok
//! usage  = usage * order_ratio(...), then the per-(candidate, leg) recalibration       // the "orderer"
//! out    = Platt per ensemble member, mean over the finite members (point row), draws appended
//! ```
//!
//! M = 5 members, C = 56 candidates, 4 usage legs (uncached input, cache read, cache write, output). There is no
//! upstream oracle for this port in the study: the torch reference is not run. The formula tests pin each stage
//! against NumPy's own `quantile`/`interp` definitions and against hand-computed values. The only external check
//! available offline is the sanity of the resulting probabilities, so treat the raw-p arms of the AUROC study as
//! unverified until a one-time reference dump exists.

mod blocks;
mod forward;
mod load;

pub use blocks::{
    ETA_CLAMP, FLOOR_TOKENS, MIN_POOL, PLATT_EPS, Side, batched_searchsorted, interp_row,
    linear_quantile_row, order_ratio_one, platt, search_sorted,
};
pub use load::{CONFIG, MANIFEST, WEIGHTS, check_release, release_sha256};

/// Exactly the tensors this release ships (read from the safetensors header of arc-1.0-20261111). `auxiliary` is
/// carried by the file but ignored by the reference loader, so it is ignored here too.
pub const SUPPORTED: &[&str] = &[
    "b_usage",
    "c1",
    "c2",
    "cal_a",
    "cal_b",
    "cal_platt",
    "difficulty",
    "embed",
    "ens_draws",
    "ens_owner",
    "ens_point",
    "high",
    "logit_bias",
    "low",
    "mean",
    "ord_alpha",
    "ord_axes",
    "ord_b",
    "ord_b1",
    "ord_center",
    "ord_e1",
    "ord_level",
    "ord_mean",
    "ord_pool_glm",
    "ord_pool_n",
    "ord_pool_net",
    "ord_recal_c",
    "ord_recal_n",
    "ord_recal_x",
    "ord_seen",
    "ord_spread",
    "ord_w1",
    "ord_w2",
    "p_ok",
    "regime",
    "scale",
    "smear",
    "u",
    "usage_ok",
    "v",
    "w1",
    "w2",
    "w_usage",
];

/// Tensors of branches the reference module has and this release does not use, with the branch they would select.
pub const UNPORTED: &[(&str, &str)] = &[
    ("components", "PCA basis (z = centered @ components.T)"),
    ("transform_weight", "input transform"),
    ("transform_bias", "input transform"),
    ("shrink_ref", "unmeasured-candidate shrink"),
    ("shrink_strength", "unmeasured-candidate shrink"),
    ("ctx_mean", "request context features"),
    ("ctx_scale", "request context features"),
    ("w_ctx", "request context features"),
    ("w_frac", "context window term"),
    ("inv_window", "context window term"),
    ("w_cache", "warm-cache features"),
    ("cache_model", "warm-cache features"),
    ("floor_ratio", "usage floor head"),
    ("floor_head_ok", "usage floor head"),
    ("floor_b", "usage floor head"),
    ("floor_rq", "usage floor head"),
    ("floor_calls", "usage floor head"),
    ("ord_fit_member", "per-member orderer"),
    ("ord_w2q", "quantile_direct orderer"),
    ("ord_bq", "quantile_direct orderer"),
    ("ord_w2m", "per-candidate orderer head"),
    ("ord_w", "linear orderer"),
    ("ord_direct", "direct orderer"),
    ("ord_bound", "bounded orderer"),
];

/// One candidate of the release's `config.json` (`arc.stage_one.candidates`).
#[derive(Clone, Debug, PartialEq)]
pub struct Candidate {
    pub model: String,
    pub effort: Option<String>,
}

impl Candidate {
    /// A candidate with no effort level.
    pub fn named(model: &str) -> Self {
        Self {
            model: model.to_string(),
            effort: None,
        }
    }

    /// `model`, or `model@effort` when an effort is set.
    pub fn display_name(&self) -> String {
        match self.effort.as_deref() {
            Some(e) if !e.is_empty() => format!("{}@{e}", self.model),
            _ => self.model.clone(),
        }
    }

    pub(crate) fn from_json(v: &serde_json::Value) -> Self {
        let model = v.get("model").map_or_else(
            || "None".to_string(),
            |m| m.as_str().unwrap_or("None").to_string(),
        );
        let effort = v.get("effort").and_then(|e| e.as_str()).map(str::to_string);
        Self { model, effort }
    }
}

/// Sizes of every axis of the head, read from the tensor shapes.
#[derive(Clone, Copy, Debug)]
pub struct Dims {
    /// Ensemble members.
    pub m: usize,
    /// Readout width.
    pub d: usize,
    /// Candidates.
    pub c: usize,
    /// Usage legs.
    pub l: usize,
    /// Hidden width of the MLP term.
    pub h: usize,
    /// Embedding width of the MLP term.
    pub e: usize,
    /// Regimes.
    pub reg: usize,
    /// Rank of the `v`/`u` term.
    pub r: usize,
    /// Orderer projection width.
    pub dd: usize,
    /// Orderer hidden width.
    pub ho: usize,
    /// Orderer pool length (padded).
    pub p: usize,
    /// Orderer recalibration knots (padded).
    pub k: usize,
    /// Ensemble draws.
    pub draws: usize,
    /// Ensemble point members.
    pub points: usize,
}

/// The tensors of the head, flattened row-major in the layout the shapes in `blocks` and `forward` document.
pub(crate) struct Stage1Tensors {
    pub mean: Vec<f64>,
    pub scale: Vec<f64>,
    pub difficulty: Vec<f64>,
    pub logit_bias: Vec<f64>,
    pub regime: Vec<f64>,
    pub v: Vec<f64>,
    pub u: Vec<f64>,
    pub w1: Vec<f64>,
    pub c1: Vec<f64>,
    pub w2: Vec<f64>,
    pub c2: Vec<f64>,
    pub embed: Vec<f64>,
    pub w_usage: Vec<f64>,
    pub b_usage: Vec<f64>,
    pub smear: Vec<f64>,
    pub low: Vec<f64>,
    pub high: Vec<f64>,
    pub usage_ok: Vec<bool>,
    pub p_ok: Vec<bool>,
    pub ord_mean: Vec<f64>,
    pub ord_axes: Vec<f64>,
    pub ord_center: Vec<f64>,
    pub ord_spread: Vec<f64>,
    pub ord_w1: Vec<f64>,
    pub ord_e1: Vec<f64>,
    pub ord_b1: Vec<f64>,
    pub ord_w2: Vec<f64>,
    pub ord_level: Vec<f64>,
    pub ord_b: Vec<f64>,
    pub ord_seen: Vec<bool>,
    pub ord_pool_glm: Vec<f64>,
    pub ord_pool_net: Vec<f64>,
    pub ord_pool_n: Vec<usize>,
    pub ord_recal_x: Vec<f64>,
    pub ord_recal_c: Vec<f64>,
    pub ord_recal_n: Vec<usize>,
    pub ord_alpha: Vec<f64>,
    pub cal_a: Vec<f64>,
    pub cal_b: Vec<f64>,
    pub cal_platt: Vec<bool>,
    pub ens_point: Vec<usize>,
    pub ens_draws: Vec<usize>,
    pub ens_owner: Vec<usize>,
}

/// The Stage-1 head. [`StageOne::forward`] returns `(p[1 + D, n, C], usage[1 + D, n, C, 4])`; row 0 is the point.
pub struct StageOne {
    pub dims: Dims,
    pub candidates: Vec<Candidate>,
    pub regimes: Vec<String>,
    pub(crate) t: Stage1Tensors,
}

/// Output of [`StageOne::forward`].
pub struct StageOut {
    /// Rows of the output: 1 point row plus the draws.
    pub rows: usize,
    /// Input rows.
    pub n: usize,
    pub c: usize,
    pub l: usize,
    /// `[rows][n][C]`.
    pub p: Vec<f64>,
    /// `[rows][n][C][L]`.
    pub usage: Vec<f64>,
}
