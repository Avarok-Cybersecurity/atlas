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

    /// Load weights from an NLLB GGUF file (arch `nllb`, F16/F32).
    ///
    /// The GGUF carries only the weights: its `enc/dec.blk.N.*` tensor names are
    /// remapped to the HuggingFace M2M-100 keys this runtime expects (see
    /// [`map_gguf_name`]). The learned `position_embd.weight` buffer is skipped —
    /// the model regenerates sinusoidal positions. Config + tokenizer must come
    /// from the safetensors sidecar (the GGUF does not carry the HF tokenizer).
    pub fn load_gguf(path: &Path) -> Result<Self> {
        let raw = crate::gguf::read_gguf_f32(path)
            .with_context(|| format!("reading GGUF {}", path.display()))?;
        let mut tensors = HashMap::new();
        for t in raw {
            let Some(hf) = map_gguf_name(&t.name) else {
                continue; // position_embd + any auxiliary tensors
            };
            // GGUF dims are ggml order (fastest-first); torch shape is reversed.
            let mut shape = t.dims.clone();
            shape.reverse();
            tensors.insert(
                hf,
                Tensor {
                    shape,
                    data: t.data,
                },
            );
        }
        if tensors.is_empty() {
            bail!("no recognised NLLB tensors in GGUF {}", path.display());
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

/// Map a GGUF (`arch=nllb`) tensor name to the HuggingFace M2M-100 key this
/// runtime loads. Returns `None` for tensors with no HF counterpart (the learned
/// `position_embd.weight`, which the model replaces with generated sinusoids).
///
/// GGUF → HF, per encoder/decoder block `N`:
/// - `attn_{q,k,v,o}`      → `self_attn.{q,k,v,out}_proj`
/// - `attn_norm`           → `self_attn_layer_norm`
/// - `cross_attn_{q,k,v,o}`→ `encoder_attn.{q,k,v,out}_proj` (decoder only)
/// - `cross_attn_norm`     → `encoder_attn_layer_norm`      (decoder only)
/// - `ffn_up` / `ffn_down` → `fc1` / `fc2`
/// - `ffn_norm`            → `final_layer_norm`
/// - `enc/dec.output_norm` → `model.{encoder,decoder}.layer_norm`
/// - `token_embd`          → `model.shared` (tied embedding)
pub fn map_gguf_name(name: &str) -> Option<String> {
    // `enc.blk.N.<rest>` / `dec.blk.N.<rest>`
    let block = |side: &str, name: &str| -> Option<String> {
        let rest = name.strip_prefix(&format!("{side}.blk."))?;
        let (n, tail) = rest.split_once('.')?;
        let (module, suffix) = tail.rsplit_once('.')?; // strip .weight/.bias
        let hf_side = if side == "enc" { "encoder" } else { "decoder" };
        let hf_module = match module {
            "attn_q" => "self_attn.q_proj",
            "attn_k" => "self_attn.k_proj",
            "attn_v" => "self_attn.v_proj",
            "attn_o" => "self_attn.out_proj",
            "attn_norm" => "self_attn_layer_norm",
            "cross_attn_q" => "encoder_attn.q_proj",
            "cross_attn_k" => "encoder_attn.k_proj",
            "cross_attn_v" => "encoder_attn.v_proj",
            "cross_attn_o" => "encoder_attn.out_proj",
            "cross_attn_norm" => "encoder_attn_layer_norm",
            "ffn_up" => "fc1",
            "ffn_down" => "fc2",
            "ffn_norm" => "final_layer_norm",
            _ => return None,
        };
        Some(format!("model.{hf_side}.layers.{n}.{hf_module}.{suffix}"))
    };

    match name {
        "token_embd.weight" => Some("model.shared.weight".to_string()),
        "enc.output_norm.weight" => Some("model.encoder.layer_norm.weight".to_string()),
        "enc.output_norm.bias" => Some("model.encoder.layer_norm.bias".to_string()),
        "dec.output_norm.weight" => Some("model.decoder.layer_norm.weight".to_string()),
        "dec.output_norm.bias" => Some("model.decoder.layer_norm.bias".to_string()),
        n if n.starts_with("enc.blk.") => block("enc", n),
        n if n.starts_with("dec.blk.") => block("dec", n),
        _ => None, // position_embd.weight and anything else
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

#[cfg(test)]
mod name_map_tests {
    use super::map_gguf_name;

    #[test]
    fn maps_encoder_decoder_and_special() {
        let cases = [
            ("token_embd.weight", Some("model.shared.weight")),
            (
                "enc.blk.0.attn_q.weight",
                Some("model.encoder.layers.0.self_attn.q_proj.weight"),
            ),
            (
                "enc.blk.5.attn_o.bias",
                Some("model.encoder.layers.5.self_attn.out_proj.bias"),
            ),
            (
                "enc.blk.3.ffn_up.weight",
                Some("model.encoder.layers.3.fc1.weight"),
            ),
            (
                "enc.blk.3.ffn_down.weight",
                Some("model.encoder.layers.3.fc2.weight"),
            ),
            (
                "enc.blk.3.ffn_norm.bias",
                Some("model.encoder.layers.3.final_layer_norm.bias"),
            ),
            (
                "enc.blk.7.attn_norm.weight",
                Some("model.encoder.layers.7.self_attn_layer_norm.weight"),
            ),
            (
                "enc.output_norm.weight",
                Some("model.encoder.layer_norm.weight"),
            ),
            (
                "dec.blk.23.cross_attn_v.weight",
                Some("model.decoder.layers.23.encoder_attn.v_proj.weight"),
            ),
            (
                "dec.blk.1.cross_attn_norm.bias",
                Some("model.decoder.layers.1.encoder_attn_layer_norm.bias"),
            ),
            (
                "dec.blk.1.attn_k.weight",
                Some("model.decoder.layers.1.self_attn.k_proj.weight"),
            ),
            (
                "dec.output_norm.bias",
                Some("model.decoder.layer_norm.bias"),
            ),
            ("position_embd.weight", None),
            ("something.unknown", None),
        ];
        for (gguf, hf) in cases {
            assert_eq!(map_gguf_name(gguf).as_deref(), hf, "mapping {gguf}");
        }
    }
}
