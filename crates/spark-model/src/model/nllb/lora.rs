// SPDX-License-Identifier: AGPL-3.0-only

//! PEFT LoRA for the served NLLB model. Reuses the existing GPU apply kernel
//! ([`apply_lora_delta`]) + [`LoraKernels`] + [`LoraPair`]; only the NLLB key
//! classifier and per-projection hook are new (the decoder-only pool/routing +
//! `classify_key` reject `nllb`/`encoder_attn`, so they can't be reused).
//!
//! `y = x·Wᵀ + b + scale·(x·Aᵀ)·Bᵀ` is applied after every adapted projection,
//! keyed by the base weight path (e.g. `model.decoder.layers.0.encoder_attn\
//! .q_proj`) — the same string the runtime already passes to `linear`/`linear1`.
//! The model owns the adapter's A/B device memory (the `WeightStore` field) so
//! the `LoraPair` pointers stay valid for the model's lifetime.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, anyhow, bail};
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::WeightStore;

use crate::layers::ops::lora_delta::{LoraKernels, LoraPair, apply_lora_delta};
use crate::weight_map::DenseWeight;

/// Minimal PEFT config (r + alpha + rsLoRA) — enough for the LoRA scaling. The
/// on-disk `target_modules` list is not validated here (the safetensors keys
/// are authoritative, and NLLB's `out_proj`/`fc1`/`fc2` targets are rejected by
/// the decoder-only PEFT parser).
#[derive(serde::Deserialize)]
struct PeftCfg {
    r: usize,
    lora_alpha: f64,
    #[serde(default)]
    use_rslora: bool,
}

pub(super) struct NllbLora {
    /// Owns the adapter A/B device buffers referenced by every `LoraPair`.
    _store: WeightStore,
    kernels: LoraKernels,
    /// Keyed by base-module path (weight name minus `.weight`).
    pairs: HashMap<String, LoraPair>,
    xa: DevicePtr,
    delta: DevicePtr,
    max_rows: usize,
}

impl NllbLora {
    /// Load a PEFT adapter directory (`adapter_config.json` +
    /// `adapter_model.safetensors`) for NLLB. `max_rows` bounds the per-call row
    /// count (encoder source length for prefill; 1 for decode).
    pub(super) fn load(dir: &Path, gpu: &dyn GpuBackend, max_rows: usize) -> Result<Self> {
        let raw = std::fs::read_to_string(dir.join("adapter_config.json"))
            .with_context(|| format!("reading {}/adapter_config.json", dir.display()))?;
        // Minimal PEFT parse: NLLB targets `out_proj`/`fc1`/`fc2`, which the
        // decoder-only `parse_peft_adapter_config` rejects as GDN projections.
        // The safetensors A/B keys are the authority on which modules exist, so
        // we only need r + alpha (+ rsLoRA) for the scaling.
        let peft: PeftCfg =
            serde_json::from_str(&raw).context("parsing NLLB adapter_config.json")?;
        if peft.r == 0 {
            bail!("NLLB adapter: r must be > 0");
        }
        let scale = if peft.use_rslora {
            peft.lora_alpha as f32 / (peft.r as f32).sqrt()
        } else {
            peft.lora_alpha as f32 / peft.r as f32
        };
        let store = spark_runtime::weights::adapter::load_adapter_safetensors(dir, gpu, 0)
            .context("loading NLLB adapter safetensors")?;

        // Gather A/B tensors (shape + device ptr) per base-module path.
        let mut a_map: HashMap<String, (Vec<usize>, DevicePtr)> = HashMap::new();
        let mut b_map: HashMap<String, (Vec<usize>, DevicePtr)> = HashMap::new();
        for name in store.names() {
            let t = store.get(name)?;
            if let Some(m) = strip_lora_key(name, ".lora_A.weight") {
                a_map.insert(m, (t.shape.clone(), t.ptr));
            } else if let Some(m) = strip_lora_key(name, ".lora_B.weight") {
                b_map.insert(m, (t.shape.clone(), t.ptr));
            }
        }

        let mut pairs = HashMap::new();
        let (mut max_rank, mut max_n_out) = (0u32, 0usize);
        for (module, (a_shape, a_ptr)) in a_map {
            let (b_shape, b_ptr) = b_map
                .remove(&module)
                .ok_or_else(|| anyhow!("NLLB adapter: lora_A without lora_B for '{module}'"))?;
            if a_shape.len() != 2 || b_shape.len() != 2 {
                bail!("NLLB adapter: '{module}' A/B must be 2-D");
            }
            // A: [rank, k_in], B: [n_out, rank].
            let (rank, k_in) = (a_shape[0] as u32, a_shape[1] as u32);
            let n_out = b_shape[0] as u32;
            if b_shape[1] as u32 != rank {
                bail!(
                    "NLLB adapter: '{module}' rank mismatch A={rank} B={}",
                    b_shape[1]
                );
            }
            max_rank = max_rank.max(rank);
            max_n_out = max_n_out.max(n_out as usize);
            pairs.insert(
                module,
                LoraPair {
                    a: DenseWeight { weight: a_ptr },
                    b: DenseWeight { weight: b_ptr },
                    rank,
                    k_in,
                    n_out,
                    scale,
                    // Single adapter → no rank padding; kernels run at true rank.
                    max_rank: rank,
                },
            );
        }
        if let Some((leftover, _)) = b_map.into_iter().next() {
            bail!("NLLB adapter: lora_B without lora_A for '{leftover}'");
        }
        if pairs.is_empty() {
            bail!(
                "NLLB adapter '{}' contained no lora_A/lora_B tensors",
                dir.display()
            );
        }

        let kernels = LoraKernels::new(gpu)?;
        let xa = gpu.alloc(max_rows * max_rank as usize * 2)?;
        let delta = gpu.alloc(max_rows * max_n_out * 2)?;
        tracing::info!(
            "NLLB LoRA loaded: {} modules, r_max={max_rank}, scale={scale:.4} from {}",
            pairs.len(),
            dir.display()
        );
        Ok(Self {
            _store: store,
            kernels,
            pairs,
            xa,
            delta,
            max_rows,
        })
    }

    /// Apply the LoRA residual for base-module `prefix` in place onto
    /// `base_out` (`[m, n_out]` bf16), reading input `x` (`[m, k_in]` bf16).
    /// No-op if this module is not adapted.
    pub(super) fn apply(
        &self,
        gpu: &dyn GpuBackend,
        prefix: &str,
        x: DevicePtr,
        base_out: DevicePtr,
        m: u32,
        stream: u64,
    ) -> Result<()> {
        if let Some(pair) = self.pairs.get(prefix) {
            if m as usize > self.max_rows {
                bail!("nllb lora: m={m} exceeds scratch rows {}", self.max_rows);
            }
            apply_lora_delta(
                gpu,
                &self.kernels,
                pair,
                x,
                base_out,
                m,
                self.xa,
                self.delta,
                stream,
            )?;
        }
        Ok(())
    }
}

/// Recover the base-module path from a PEFT key: strip a trailing `suffix` and
/// a leading `base_model.model.` wrapper. `None` if `suffix` is absent.
fn strip_lora_key(name: &str, suffix: &str) -> Option<String> {
    let base = name.strip_suffix(suffix)?;
    Some(
        base.strip_prefix("base_model.model.")
            .unwrap_or(base)
            .to_string(),
    )
}
