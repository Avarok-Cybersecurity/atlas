// SPDX-License-Identifier: AGPL-3.0-only

//! Safetensors weight store for the NLLB CPU runtime.
//!
//! NLLB-200 ships as PyTorch `.bin` (pickle) which Atlas cannot read; this
//! store consumes the safetensors conversion (e.g.
//! `MonumentalSystems/nllb-200-3.3B`). All tensors are held as `f32` on the
//! host — the reference runtime is fp32.

use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result, bail};
use half::bf16;
use safetensors::SafeTensors;
use safetensors::tensor::Dtype;
use serde::Deserialize;

/// A dense host tensor, row-major, stored as `f32`.
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn rows(&self) -> usize {
        self.shape[0]
    }
    pub fn cols(&self) -> usize {
        self.shape[1]
    }
}

/// Name → tensor map loaded from one or more safetensors shards.
pub struct WeightStore {
    tensors: HashMap<String, Tensor>,
}

#[derive(Deserialize)]
struct StIndex {
    weight_map: HashMap<String, String>,
}

impl WeightStore {
    /// Load a model directory containing either `model.safetensors` or a
    /// sharded `model.safetensors.index.json` + shard files.
    pub fn load_dir(dir: &Path) -> Result<Self> {
        let index_path = dir.join("model.safetensors.index.json");
        let single = dir.join("model.safetensors");

        let shard_files: Vec<String> = if index_path.exists() {
            let idx: StIndex = serde_json::from_str(
                &std::fs::read_to_string(&index_path)
                    .with_context(|| format!("reading {}", index_path.display()))?,
            )
            .context("parsing safetensors index")?;
            let mut files: Vec<String> = idx.weight_map.values().cloned().collect();
            files.sort();
            files.dedup();
            files
        } else if single.exists() {
            vec!["model.safetensors".to_string()]
        } else {
            bail!(
                "no safetensors found in {} (expected model.safetensors or \
                 model.safetensors.index.json)",
                dir.display()
            );
        };

        let mut tensors = HashMap::new();
        for shard in &shard_files {
            let path = dir.join(shard);
            let bytes = std::fs::read(&path)
                .with_context(|| format!("reading shard {}", path.display()))?;
            let st = SafeTensors::deserialize(&bytes)
                .with_context(|| format!("deserializing {}", path.display()))?;
            for name in st.names() {
                let view = st.tensor(name)?;
                let data = to_f32(view.dtype(), view.data()).with_context(|| {
                    format!("unsupported dtype {:?} for tensor {name}", view.dtype())
                })?;
                tensors.insert(
                    name.to_string(),
                    Tensor {
                        shape: view.shape().to_vec(),
                        data,
                    },
                );
            }
        }
        Ok(Self { tensors })
    }

    /// Fetch a tensor by exact name (fail-fast if absent).
    pub fn get(&self, name: &str) -> Result<&Tensor> {
        self.tensors
            .get(name)
            .with_context(|| format!("weight tensor '{name}' not found"))
    }

    /// Fetch the first tensor that exists from a list of candidate names.
    /// Used for tied weights (NLLB stores only `model.shared.weight`).
    pub fn get_any(&self, names: &[&str]) -> Result<&Tensor> {
        for n in names {
            if let Some(t) = self.tensors.get(*n) {
                return Ok(t);
            }
        }
        bail!("none of the candidate weights {:?} were found", names)
    }
}

/// Decode raw safetensors bytes into `f32`, supporting F32 and BF16 checkpoints.
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
        other => bail!("unsupported safetensors dtype {other:?}"),
    }
}
