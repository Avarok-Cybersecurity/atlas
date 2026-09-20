// SPDX-License-Identifier: AGPL-3.0-only
//! Checkpoint header inventory and TP admission. No GPU allocation here.
use crate::weights::WeightDtype;
use anyhow::{Context, Result, ensure};
use avarok_core::config::ModelConfig;
use avarok_core::kimi_k3::tp::{TpBytePlan, plan_tensor_bytes};
use avarok_core::kimi_k3::{K3Graph, MixerKind};
use std::collections::{BTreeSet, HashSet};
use std::path::{Path, PathBuf};

pub(super) struct TensorPlan {
    pub name: String,
    pub source_shape: Vec<usize>,
    pub source_dtype: safetensors::Dtype,
    pub dtype: WeightDtype,
    pub plan: TpBytePlan,
}
pub(super) struct ShardPlan {
    pub path: PathBuf,
    pub tensors: Vec<TensorPlan>,
}

fn text_tensor(name: &str) -> bool {
    let n = name.strip_prefix("language_model.").unwrap_or(name);
    n.starts_with("model.layers.")
        || n.starts_with("model.embed_tokens.")
        || n.starts_with("model.norm.")
        || n.starts_with("model.output_attn_res_")
        || n.starts_with("lm_head.")
}

pub(super) fn scan(dir: &Path, config: &ModelConfig) -> Result<Vec<ShardPlan>> {
    let root = dir.canonicalize()?;
    ensure!(
        !root.join("extra_weights.safetensors").exists(),
        "K3 rank loader does not support extra/MTP weight grafts"
    );
    let index = super::index::read(&root)?;
    let names: BTreeSet<String> = if let Some(index) = &index {
        index.values().cloned().collect()
    } else if root.join("model.safetensors").is_file() {
        BTreeSet::from(["model.safetensors".into()])
    } else {
        std::fs::read_dir(&root)?
            .map(|r| Ok(r?.file_name().to_string_lossy().into_owned()))
            .collect::<Result<Vec<String>>>()?
            .into_iter()
            .filter(|n| n.starts_with("model-") && n.ends_with(".safetensors"))
            .collect()
    };
    ensure!(!names.is_empty(), "K3 checkpoint has no safetensors files");
    let graph = K3Graph::from_config(config);
    let mut result = Vec::new();
    let mut seen = HashSet::new();
    let mut packed = Vec::new();
    let mut scales = Vec::new();
    let mut skipped = 0usize;
    for file_name in names {
        ensure!(
            Path::new(&file_name).components().count() == 1 && file_name.ends_with(".safetensors"),
            "invalid K3 shard filename"
        );
        let path = root.join(&file_name).canonicalize()?;
        ensure!(path.starts_with(&root), "K3 shard escapes checkpoint root");
        let file = std::fs::File::open(&path)?;
        let mmap = unsafe { memmap2::MmapOptions::new().map(&file)? };
        let tensors = safetensors::SafeTensors::deserialize(&mmap)?;
        let mut plans = Vec::new();
        let mut tensor_names = tensors.names();
        tensor_names.sort();
        for name in tensor_names {
            if !text_tensor(name) {
                skipped += 1;
                continue;
            }
            if let Some(index) = &index {
                ensure!(
                    index.get(name).is_some_and(|s| s == &file_name),
                    "K3 tensor {name} missing/mismatched index entry"
                );
            }
            ensure!(seen.insert(name.to_string()), "duplicate K3 tensor {name}");
            let view = tensors.tensor(name)?;
            let valid_dtype = if name.ends_with(".weight_packed") {
                view.dtype() == safetensors::Dtype::U8
            } else if name.ends_with(".weight_scale") {
                matches!(
                    view.dtype(),
                    safetensors::Dtype::U8 | safetensors::Dtype::F8_E8M0
                )
            } else {
                matches!(
                    view.dtype(),
                    safetensors::Dtype::BF16 | safetensors::Dtype::F32
                )
            };
            ensure!(
                valid_dtype,
                "{name}: storage dtype {:?} does not match K3 tensor role",
                view.dtype()
            );
            let dtype = WeightDtype::from_safetensors(view.dtype())?;
            let mixer = if let Some((_, tail)) = name.split_once("model.layers.") {
                let layer: usize = tail
                    .split('.')
                    .next()
                    .context("missing layer index")?
                    .parse()?;
                graph
                    .layers
                    .get(layer)
                    .context("K3 tensor layer index outside configuration")?
                    .mixer
            } else {
                MixerKind::Kda
            };
            if avarok_core::kimi_k3::tp::tensor_plan(name, mixer, config).0
                == avarok_core::kimi_k3::tp::TpAxis::Replicated
            {
                avarok_core::kimi_k3::weights_geometry::validate_replicated_shape(
                    name,
                    view.shape(),
                    config,
                )?;
            }
            let plan = plan_tensor_bytes(name, view.shape(), dtype.byte_size(), mixer, config)?;
            // Padding guards run during preflight, before the first GPU allocation.
            plan.validate_source(view.data())?;
            if let Some(base) = name.strip_suffix(".weight_packed") {
                packed.push(base.to_string());
            }
            if let Some(base) = name.strip_suffix(".weight_scale") {
                scales.push(base.to_string());
            }
            plans.push(TensorPlan {
                name: name.to_string(),
                source_shape: view.shape().to_vec(),
                source_dtype: view.dtype(),
                dtype,
                plan,
            });
        }
        result.push(ShardPlan {
            path,
            tensors: plans,
        });
    }
    if let Some(index) = index {
        for name in index.keys().filter(|n| text_tensor(n)) {
            ensure!(
                seen.contains(name),
                "K3 indexed tensor {name} missing in checkpoint"
            );
        }
    }
    for required in avarok_core::kimi_k3::weights::required_names(config) {
        let exists = seen.contains(&required)
            || (required.ends_with("lm_head.weight") && seen.contains("lm_head.weight"))
            || (required.contains(".block_sparse_moe.experts.")
                && required.ends_with(".weight")
                && seen.contains(&format!("{}_packed", required))
                && seen.contains(&format!("{}_scale", required)));
        ensure!(
            exists,
            "K3 missing required tensor {required} before GPU allocation"
        );
    }
    for base in packed {
        ensure!(
            seen.contains(&format!("{base}.weight_scale")),
            "K3 packed tensor {base} missing scale pair"
        );
    }
    for base in scales {
        ensure!(
            seen.contains(&format!("{base}.weight_packed")),
            "K3 scale {base} missing packed pair"
        );
    }
    ensure!(
        !seen.is_empty(),
        "K3 checkpoint has no supported text tensors"
    );
    tracing::info!(
        event = "k3_weight_inventory",
        loaded_tensors = seen.len(),
        skipped_tensors = skipped,
        "K3 text inventory validated; unsupported modalities/MTP excluded"
    );
    Ok(result)
}
