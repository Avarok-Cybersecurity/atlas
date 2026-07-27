// SPDX-License-Identifier: AGPL-3.0-only

//! PEFT LoRA adapter support for the NLLB / M2M-100 CPU runtime.
//!
//! A LoRA adapter is a low-rank residual `ΔW = scale · B · A` applied on top of
//! a base `nn.Linear` weight WITHOUT merging into it (runtime delta, exactly
//! like the GPU engine's `spark-model::lora`). For a projection
//! `y = x·Wᵀ + b`, the adapted output is
//!
//! ```text
//! y = x·Wᵀ + b + scale · (x·Aᵀ)·Bᵀ
//! ```
//!
//! with `A: [r, in]`, `B: [out, r]`, and `scale = alpha/r` (or `alpha/√r` under
//! rsLoRA). Both extra matmuls reuse [`ops::linear`], so the delta is just two
//! GEMVs and a scaled add — cheap relative to the base projection.
//!
//! Adapter format is standard HuggingFace PEFT: `adapter_config.json` (rank,
//! alpha, `target_modules`) + `adapter_model.safetensors` whose keys look like
//! `base_model.model.<module>.lora_A.weight` / `...lora_B.weight`, where
//! `<module>` is the base weight path this runtime already fetches by name
//! (e.g. `model.encoder.layers.0.self_attn.q_proj`).

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use half::bf16;
use safetensors::SafeTensors;
use safetensors::tensor::Dtype;
use serde::Deserialize;

use crate::ops;

/// One adapted module's low-rank pair.
struct LoraPair {
    a: Vec<f32>, // [r, in]  (row-major, PEFT `lora_A.weight`)
    b: Vec<f32>, // [out, r] (row-major, PEFT `lora_B.weight`)
    r: usize,
    in_dim: usize,
    out_dim: usize,
}

/// A loaded PEFT adapter: the scaling and the per-module low-rank pairs, keyed
/// by base-module path (the same string the model fetches weights under, minus
/// the trailing `.weight`).
pub struct LoraSet {
    scale: f32,
    pairs: HashMap<String, LoraPair>,
}

#[derive(Deserialize)]
struct PeftConfig {
    r: usize,
    lora_alpha: f32,
    #[serde(default)]
    use_rslora: bool,
    #[serde(default)]
    target_modules: Vec<String>,
}

impl LoraSet {
    /// Load a PEFT adapter directory (`adapter_config.json` +
    /// `adapter_model.safetensors`). Fails fast on a missing/one-sided pair or
    /// an A/B rank mismatch — never a silent skip.
    pub fn load_dir(dir: &Path) -> Result<Self> {
        let cfg_path = dir.join("adapter_config.json");
        let cfg: PeftConfig = serde_json::from_str(
            &std::fs::read_to_string(&cfg_path)
                .with_context(|| format!("reading {}", cfg_path.display()))?,
        )
        .context("parsing adapter_config.json")?;
        if cfg.r == 0 {
            bail!("adapter_config.json: r must be > 0");
        }
        let scale = if cfg.use_rslora {
            cfg.lora_alpha / (cfg.r as f32).sqrt()
        } else {
            cfg.lora_alpha / cfg.r as f32
        };

        let wt_path = dir.join("adapter_model.safetensors");
        let bytes =
            std::fs::read(&wt_path).with_context(|| format!("reading {}", wt_path.display()))?;
        let st = SafeTensors::deserialize(&bytes)
            .with_context(|| format!("deserializing {}", wt_path.display()))?;

        // Gather A and B tensors per module path, then pair them.
        let mut a_map: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
        let mut b_map: HashMap<String, (Vec<usize>, Vec<f32>)> = HashMap::new();
        for name in st.names() {
            let (module, is_a) = if let Some(m) = strip_lora_key(name, ".lora_A.weight") {
                (m, true)
            } else if let Some(m) = strip_lora_key(name, ".lora_B.weight") {
                (m, false)
            } else {
                continue; // non-LoRA tensor (e.g. embeddings) — ignore
            };
            let view = st.tensor(name)?;
            let data = to_f32(view.dtype(), view.data())
                .with_context(|| format!("unsupported dtype for {name}"))?;
            let entry = (view.shape().to_vec(), data);
            if is_a {
                a_map.insert(module, entry);
            } else {
                b_map.insert(module, entry);
            }
        }

        let mut pairs = HashMap::new();
        for (module, (a_shape, a)) in a_map {
            let (b_shape, b) = b_map.remove(&module).ok_or_else(|| {
                anyhow::anyhow!("adapter has lora_A but no lora_B for module '{module}'")
            })?;
            // A: [r, in], B: [out, r].
            if a_shape.len() != 2 || b_shape.len() != 2 {
                bail!("module '{module}': lora_A/lora_B must be 2-D");
            }
            let (r, in_dim) = (a_shape[0], a_shape[1]);
            let (out_dim, r_b) = (b_shape[0], b_shape[1]);
            if r != r_b {
                bail!("module '{module}': lora_A rank {r} != lora_B rank {r_b}");
            }
            pairs.insert(
                module,
                LoraPair {
                    a,
                    b,
                    r,
                    in_dim,
                    out_dim,
                },
            );
        }
        if let Some((leftover, _)) = b_map.into_iter().next() {
            bail!("adapter has lora_B but no lora_A for module '{leftover}'");
        }
        if pairs.is_empty() {
            bail!(
                "adapter '{}' contained no lora_A/lora_B tensors (target_modules={:?})",
                dir.display(),
                cfg.target_modules
            );
        }
        Ok(Self { scale, pairs })
    }

    /// Number of adapted modules (for reporting/tests).
    pub fn adapted_modules(&self) -> usize {
        self.pairs.len()
    }

    /// The LoRA residual for base-module `module` over input `x` (`[rows, in]`),
    /// or `None` if this module is not adapted. Result is `[rows, out]` and is
    /// already scaled — the caller adds it onto the base projection output.
    pub fn delta(&self, module: &str, x: &[f32], rows: usize) -> Option<Vec<f32>> {
        let p = self.pairs.get(module)?;
        debug_assert_eq!(x.len(), rows * p.in_dim);
        // xa = x · Aᵀ : [rows, r]  (A is [r, in], i.e. an nn.Linear [out=r, in]).
        let xa = ops::linear(x, rows, p.in_dim, &p.a, p.r, None);
        // delta = xa · Bᵀ : [rows, out]  (B is [out, r]).
        let mut d = ops::linear(&xa, rows, p.r, &p.b, p.out_dim, None);
        for v in d.iter_mut() {
            *v *= self.scale;
        }
        Some(d)
    }
}

/// Strip PEFT wrapping to recover the base-module path a runtime weight lives
/// under. `name` like `base_model.model.model.encoder.layers.0.self_attn\
/// .q_proj.lora_A.weight` with `suffix = ".lora_A.weight"` yields
/// `model.encoder.layers.0.self_attn.q_proj`. Returns `None` if `suffix` is
/// absent.
fn strip_lora_key(name: &str, suffix: &str) -> Option<String> {
    let base = name.strip_suffix(suffix)?;
    // PEFT wraps the base model as `base_model.model.<module>`; strip that
    // prefix if present so the remainder matches the runtime's weight names.
    let base = base.strip_prefix("base_model.model.").unwrap_or(base);
    Some(base.to_string())
}

/// Decode safetensors bytes to `f32` (F32 / BF16 / F16 adapters).
fn to_f32(dtype: Dtype, bytes: &[u8]) -> Result<Vec<f32>> {
    match dtype {
        Dtype::F32 => Ok(bytes
            .chunks_exact(4)
            .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
            .collect()),
        Dtype::BF16 => Ok(bytes
            .chunks_exact(2)
            .map(|b| bf16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect()),
        Dtype::F16 => Ok(bytes
            .chunks_exact(2)
            .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
            .collect()),
        other => bail!("unsupported adapter dtype {other:?}"),
    }
}

#[cfg(test)]
#[path = "lora_tests.rs"]
mod tests;
