// SPDX-License-Identifier: AGPL-3.0-only
// Portions derived from the ARC-1.0 release files (config.json signal head metadata and the model card) of
// rayline-ai/ARC-1.0, Copyright 2026 Atlas Futures, Inc., licensed under the Apache License 2.0
// (https://www.apache.org/licenses/LICENSE-2.0). The head structure (standardize, Linear 1024 to 256, activation,
// Linear 256 to C, Platt) is the one the release tensors and the model card describe; no reference code exists for it
// upstream, so this is a clean Rust implementation. By Atlas contributors. See RAYLINE-ROUTING-ATTRIBUTION.md.
//! Responsibility: the float64 signal heads (PII payload, topic domain, topic operation) of the ARC-1.0 release.
//!
//! Tensor layout, read from the safetensors header of arc-1.0-20261111 (all under `signals.<head>.`):
//!
//! ```text
//! mean[1024], scale[1024]               (F32)   standardization
//! net.0.weight[256, 1024], net.0.bias[256]      first Linear
//! net.2.weight[C, 256], net.2.bias[C]           second Linear
//! platt[C, 2]                           (F64)   per-class Platt map
//! ```
//!
//! NOT documented anywhere upstream, and therefore chosen by agreement with `goldens.signals.jsonl` (see
//! `goldens.rs`): the activation between the Linears, whether `scale` multiplies or divides, which platt column is
//! the slope, and which readout row feeds the heads. The defaults are only the first guess of the search.

use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use serde_json::Value;

use super::error::{GResult, value_err};
use super::mat::{Mat, dot, sigmoid};
use super::stage_one::platt;
use super::tensorio::{Tensor, TensorFile};

pub const HEADS: [&str; 3] = ["pii.payload", "topic.domain", "topic.operation"];

/// `erf(x)` to about 1e-16: the positive-term series `2/sqrt(pi) e^{-x^2} sum 2^n x^{2n+1} / (2n+1)!!`
/// (no cancellation), exact enough that `erf(6)` rounds to 1.
pub fn erf(x: f64) -> f64 {
    let a = x.abs();
    if a >= 6.0 {
        return x.signum();
    }
    let mut term = a;
    let mut sum = a;
    let mut n = 0u32;
    while term > sum * 1e-18 && n < 400 {
        n += 1;
        term *= 2.0 * a * a / f64::from(2 * n + 1);
        sum += term;
    }
    let v = 2.0 / std::f64::consts::PI.sqrt() * (-a * a).exp() * sum;
    x.signum() * v.min(1.0)
}

/// The activation between the two Linears.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Activation {
    Relu,
    Gelu,
    GeluTanh,
    Silu,
    Tanh,
}

impl Activation {
    pub const ALL: [Activation; 5] = [
        Self::Relu,
        Self::Gelu,
        Self::GeluTanh,
        Self::Silu,
        Self::Tanh,
    ];

    pub fn name(self) -> &'static str {
        match self {
            Self::Relu => "relu",
            Self::Gelu => "gelu",
            Self::GeluTanh => "gelu_tanh",
            Self::Silu => "silu",
            Self::Tanh => "tanh",
        }
    }

    pub fn apply(self, x: f64) -> f64 {
        match self {
            Self::Relu => x.max(0.0),
            Self::Gelu => 0.5 * x * (1.0 + erf(x / std::f64::consts::SQRT_2)),
            Self::GeluTanh => {
                0.5 * x
                    * (1.0
                        + ((2.0 / std::f64::consts::PI).sqrt() * (x + 0.044715 * x * x * x)).tanh())
            }
            Self::Silu => x * sigmoid(x),
            Self::Tanh => x.tanh(),
        }
    }
}

/// Whether `scale` multiplies or divides the centered input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleMode {
    Mul,
    Div,
}

impl ScaleMode {
    pub const ALL: [ScaleMode; 2] = [Self::Mul, Self::Div];

    pub fn name(self) -> &'static str {
        match self {
            Self::Mul => "mul",
            Self::Div => "div",
        }
    }
}

/// `platt[:, 0]` is the slope `a` (and `[:, 1]` the offset `b`), or the reverse.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlattOrder {
    Ab,
    Ba,
}

impl PlattOrder {
    pub const ALL: [PlattOrder; 2] = [Self::Ab, Self::Ba];

    pub fn name(self) -> &'static str {
        match self {
            Self::Ab => "ab",
            Self::Ba => "ba",
        }
    }
}

/// One variant of the undocumented choices.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Variant {
    pub activation: Activation,
    pub scale_mode: ScaleMode,
    pub platt_order: PlattOrder,
}

impl Default for Variant {
    fn default() -> Self {
        Self {
            activation: Activation::Relu,
            scale_mode: ScaleMode::Mul,
            platt_order: PlattOrder::Ab,
        }
    }
}

/// Every `(activation, scale_mode, platt_order)` variant the oracle search tries, activation outermost.
pub fn grid() -> Vec<Variant> {
    let mut out = Vec::new();
    for &activation in &Activation::ALL {
        for &scale_mode in &ScaleMode::ALL {
            for &platt_order in &PlattOrder::ALL {
                out.push(Variant {
                    activation,
                    scale_mode,
                    platt_order,
                });
            }
        }
    }
    out
}

/// Thresholds for [`SignalHead::present`]: one for every class, or one per class name.
pub enum Thresholds<'a> {
    Scalar(f64),
    PerClass(&'a HashMap<String, f64>),
}

/// One signal head. [`SignalHead::probs`] gives `[n, C]` calibrated probabilities.
pub struct SignalHead {
    pub name: String,
    pub classes: Vec<String>,
    pub config: Value,
    mean: Vec<f64>,
    scale: Vec<f64>,
    w0: Vec<f64>,
    b0: Vec<f64>,
    w2: Vec<f64>,
    b2: Vec<f64>,
    platt: Vec<f64>,
    width: usize,
    hidden: usize,
}

impl SignalHead {
    /// Builds a head from its tensors (names without the `signals.<head>.` prefix) and class list.
    pub fn new(
        name: &str,
        tensors: &BTreeMap<String, Tensor>,
        classes: Vec<String>,
        config: Option<Value>,
    ) -> GResult<Self> {
        let need = [
            "mean",
            "scale",
            "net.0.weight",
            "net.0.bias",
            "net.2.weight",
            "net.2.bias",
            "platt",
        ];
        let missing: Vec<&str> = need
            .iter()
            .copied()
            .filter(|k| !tensors.contains_key(*k))
            .collect();
        if !missing.is_empty() {
            return value_err(format!("signals.{name}: tensors missing: {missing:?}"));
        }
        let c = classes.len();
        let (w0, w2, pl, mean) = (
            &tensors["net.0.weight"],
            &tensors["net.2.weight"],
            &tensors["platt"],
            &tensors["mean"],
        );
        if w0.shape.len() != 2 || w2.shape.len() != 2 || mean.shape.len() != 1 {
            return value_err(format!("signals.{name}: unexpected tensor ranks"));
        }
        if w2.shape[0] != c || pl.shape != [c, 2] {
            return value_err(format!(
                "signals.{name}: {c} classes but net.2.weight {:?}, platt {:?}",
                w2.shape, pl.shape
            ));
        }
        if w0.shape[1] != mean.shape[0] || w2.shape[1] != w0.shape[0] {
            return value_err(format!("signals.{name}: layer shapes do not chain"));
        }
        Ok(Self {
            name: name.to_string(),
            classes,
            config: config.unwrap_or(Value::Null),
            mean: mean.data.clone(),
            scale: tensors["scale"].data.clone(),
            width: w0.shape[1],
            hidden: w0.shape[0],
            w0: w0.data.clone(),
            b0: tensors["net.0.bias"].data.clone(),
            w2: w2.data.clone(),
            b2: tensors["net.2.bias"].data.clone(),
            platt: pl.data.clone(),
        })
    }

    /// Root mean square of the standardized readout row `i` (about 1 when the readouts look like the head's training
    /// data), and the correlation of the raw row with the head's `mean` vector (a coarse check of the encoder).
    pub fn standardized_stats(&self, x: &Mat, i: usize, scale_mode: ScaleMode) -> (f64, f64) {
        let row = x.row(i);
        let z: Vec<f64> = (0..self.width)
            .map(|k| match scale_mode {
                ScaleMode::Mul => (row[k] - self.mean[k]) * self.scale[k],
                ScaleMode::Div => (row[k] - self.mean[k]) / self.scale[k],
            })
            .collect();
        let rms = (z.iter().map(|v| v * v).sum::<f64>() / z.len() as f64).sqrt();
        let (mx, my) = (
            row.iter().sum::<f64>() / row.len() as f64,
            self.mean.iter().sum::<f64>() / self.mean.len() as f64,
        );
        let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
        for (r, m) in row.iter().zip(&self.mean) {
            let (a, b) = (r - mx, m - my);
            sxy += a * b;
            sxx += a * a;
            syy += b * b;
        }
        (rms, sxy / (sxx * syy).sqrt())
    }

    /// Pre-sigmoid logits `[n, C]`.
    pub fn logits(&self, x: &Mat, activation: Activation, scale_mode: ScaleMode) -> GResult<Mat> {
        if x.cols != self.width {
            return value_err(format!(
                "signals.{}: readout width {} but the head takes {}",
                self.name, x.cols, self.width
            ));
        }
        let c = self.classes.len();
        let mut out = Mat::zeros(x.rows, c);
        for i in 0..x.rows {
            let z: Vec<f64> = (0..self.width)
                .map(|k| match scale_mode {
                    ScaleMode::Mul => (x.at(i, k) - self.mean[k]) * self.scale[k],
                    ScaleMode::Div => (x.at(i, k) - self.mean[k]) / self.scale[k],
                })
                .collect();
            let h: Vec<f64> = (0..self.hidden)
                .map(|j| {
                    activation
                        .apply(dot(&z, &self.w0[j * self.width..(j + 1) * self.width]) + self.b0[j])
                })
                .collect();
            for ci in 0..c {
                out.data[i * c + ci] =
                    dot(&h, &self.w2[ci * self.hidden..(ci + 1) * self.hidden]) + self.b2[ci];
            }
        }
        Ok(out)
    }

    /// Calibrated probabilities `[n, C]`.
    pub fn probs(&self, x: &Mat, v: Variant) -> GResult<Mat> {
        let mut out = self.logits(x, v.activation, v.scale_mode)?;
        let c = self.classes.len();
        for i in 0..out.rows {
            for ci in 0..c {
                let (p0, p1) = (self.platt[ci * 2], self.platt[ci * 2 + 1]);
                let (a, b) = if v.platt_order == PlattOrder::Ab {
                    (p0, p1)
                } else {
                    (p1, p0)
                };
                out.data[i * c + ci] = platt(sigmoid(out.data[i * c + ci]), true, a, b);
            }
        }
        Ok(out)
    }

    /// Classes whose probability reaches their threshold.
    pub fn present(&self, probs_row: &[f64], thresholds: &Thresholds<'_>) -> Vec<String> {
        let mut out = Vec::new();
        for (cls, &p) in self.classes.iter().zip(probs_row) {
            let th = match thresholds {
                Thresholds::Scalar(t) => Some(*t),
                Thresholds::PerClass(m) => m.get(cls).copied(),
            };
            if th.is_some_and(|t| p >= t) {
                out.push(cls.clone());
            }
        }
        out
    }
}

/// All three heads from `model.safetensors`, with class lists from `config["arc"]["signals"]["heads"]`.
pub fn load_signal_heads(
    release_dir: impl AsRef<Path>,
    config: &Value,
) -> GResult<Vec<(String, SignalHead)>> {
    let file = TensorFile::open(release_dir.as_ref().join("model.safetensors"))?;
    let mut heads = Vec::new();
    for name in HEADS {
        let meta = &config["arc"]["signals"]["heads"][name];
        let classes: Vec<String> = meta["classes"]
            .as_array()
            .map(|a| {
                a.iter()
                    .map(|c| c.as_str().unwrap_or_default().to_string())
                    .collect()
            })
            .unwrap_or_default();
        let tensors = file.read_prefix(&format!("signals.{name}."), true)?;
        heads.push((
            name.to_string(),
            SignalHead::new(name, &tensors, classes, Some(meta.clone()))?,
        ));
    }
    Ok(heads)
}

/// The golden probability vector of one head for a case, in `classes` order; `None` when the case has no scope.
pub fn expected_probs(golden_head: &Value, classes: &[String]) -> Option<Vec<f64>> {
    let scope = golden_head.get("scopes")?.get("conversation")?;
    let probs = scope.get("probabilities")?;
    if probs.is_null() {
        return None;
    }
    classes
        .iter()
        .map(|c| probs.get(c).and_then(Value::as_f64))
        .collect()
}
