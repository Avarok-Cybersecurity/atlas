// SPDX-License-Identifier: AGPL-3.0-only

//! Load unpacked BF16/F32 K3 safetensors into [`K3CpuModel`].
//! Packed MXFP4 (`weight_packed`) is refused unless `K3_ALLOW_MXFP4=1`, which
//! unpacks via the DSV4 E8M0 host path (`crate::mxfp4_e8m0`).

use std::collections::HashMap;
use std::fs::File;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use memmap2::Mmap;
use safetensors::SafeTensors;
use safetensors::tensor::Dtype;

use super::cpu_bind::{assemble, inventory, kda_from, mla_from, moe_from, text_key};
use super::cpu_weights::K3CpuModel;
use crate::config::parse_config;

impl K3CpuModel {
    /// Bind a HuggingFace twin dir (`config.json` + unpacked `.safetensors`).
    pub fn from_pretrained(dir: &Path) -> Result<Self> {
        if !dir.exists() {
            bail!("K3 twin dir {} does not exist", dir.display());
        }
        let cfg_path = dir.join("config.json");
        let raw = std::fs::read_to_string(&cfg_path)
            .with_context(|| format!("read {}", cfg_path.display()))?;
        let c = parse_config(&raw).context("parse K3 config.json")?;
        let graph = super::layer::K3Graph::from_config(&c);
        let kda = kda_from(&c);
        let mla = mla_from(&c);
        let moe = moe_from(&c);
        let mut store = load_dir(dir)?;
        let mut got = HashMap::new();
        for (name, n) in inventory(&c, &graph, &kda, &mla, &moe) {
            got.insert(name.clone(), take_len(&mut store, &name, n)?);
        }
        if c.tie_word_embeddings {
            let embed_k = text_key(&c.weight_prefix, "model.embed_tokens.weight");
            let head_k = text_key(&c.weight_prefix, "lm_head.weight");
            if !got.contains_key(&head_k) {
                let embed = got
                    .get(&embed_k)
                    .with_context(|| format!("tied lm_head needs {embed_k}"))?
                    .clone();
                got.insert(head_k, embed);
            }
        }
        let mut unused: Vec<String> = store
            .keys()
            .filter(|k| is_text_weight(k))
            .cloned()
            .collect();
        if !unused.is_empty() {
            unused.sort();
            eprintln!(
                "K3CpuModel::from_pretrained unmapped tensors: {}",
                unused.join(", ")
            );
        }
        assemble(&c, graph, kda, mla, moe, &mut got)
    }
}

/// C1–C6 spark2 runs. Missing `K3_TWIN` skips. A present dir that fails to load panics.
#[cfg(test)]
pub(super) fn twin_from_env() -> Option<K3CpuModel> {
    match std::env::var("K3_TWIN") {
        Ok(p) => {
            let path = Path::new(&p);
            if !path.exists() {
                eprintln!("skip twin: K3_TWIN={p} does not exist");
                return None;
            }
            Some(
                K3CpuModel::from_pretrained(path)
                    .unwrap_or_else(|e| panic!("K3_TWIN={p} load failed: {e:#}")),
            )
        }
        _ => None,
    }
}

fn is_vision(name: &str) -> bool {
    name.starts_with("vision_tower.") || name.starts_with("mm_projector.")
}

fn is_text_weight(name: &str) -> bool {
    !is_vision(name)
        && (name.starts_with("language_model.")
            || name.starts_with("model.")
            || name.starts_with("lm_head."))
}

fn take_len(store: &mut HashMap<String, Vec<f32>>, key: &str, n: usize) -> Result<Vec<f32>> {
    let v = take_tensor(store, key)?;
    if v.len() != n {
        bail!("{key}: {} elems, expected {n}", v.len());
    }
    Ok(v)
}

fn take_tensor(store: &mut HashMap<String, Vec<f32>>, key: &str) -> Result<Vec<f32>> {
    if let Some(v) = store.remove(key) {
        return Ok(v);
    }
    if let Some(rest) = key.strip_prefix("language_model.")
        && let Some(v) = store.remove(rest)
    {
        return Ok(v);
    }
    if key.ends_with("lm_head.weight")
        && let Some(v) = store.remove("lm_head.weight")
    {
        return Ok(v);
    }
    bail!("tensor {key} not found")
}

fn load_dir(dir: &Path) -> Result<HashMap<String, Vec<f32>>> {
    let mut out = HashMap::new();
    let mut packed = super::mxfp4::PackedSink::default();
    for path in shard_files(dir)? {
        ingest_shard(&path, &mut out, &mut packed)?;
    }
    packed.finish(&mut out)?;
    if out.is_empty() {
        bail!("no language-model tensors in {}", dir.display());
    }
    Ok(out)
}

fn shard_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let index = dir.join("model.safetensors.index.json");
    if index.exists() {
        let raw =
            std::fs::read_to_string(&index).with_context(|| format!("read {}", index.display()))?;
        let v: serde_json::Value = serde_json::from_str(&raw).context("safetensors index JSON")?;
        let map = v
            .get("weight_map")
            .and_then(|m| m.as_object())
            .context("index missing weight_map")?;
        let mut files: Vec<String> = map
            .values()
            .filter_map(|x| x.as_str().map(str::to_string))
            .collect();
        files.sort();
        files.dedup();
        return Ok(files.into_iter().map(|f| dir.join(f)).collect());
    }
    let single = dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }
    let mut found = Vec::new();
    for ent in std::fs::read_dir(dir).with_context(|| format!("read_dir {}", dir.display()))? {
        let p = ent?.path();
        if p.extension().and_then(|e| e.to_str()) == Some("safetensors") {
            found.push(p);
        }
    }
    found.sort();
    if found.is_empty() {
        bail!(
            "no safetensors in {} (expected model.safetensors or index.json)",
            dir.display()
        );
    }
    Ok(found)
}

fn ingest_shard(
    path: &Path,
    out: &mut HashMap<String, Vec<f32>>,
    packed: &mut super::mxfp4::PackedSink,
) -> Result<()> {
    let file = File::open(path).with_context(|| format!("open {}", path.display()))?;
    // SAFETY: read-only mapping of an immutable checkpoint shard.
    let mmap = unsafe { Mmap::map(&file) }.with_context(|| format!("mmap {}", path.display()))?;
    let st = SafeTensors::deserialize(&mmap)
        .with_context(|| format!("safetensors {}", path.display()))?;
    for name in st.names() {
        if is_vision(name) {
            continue;
        }
        let t = st.tensor(name)?;
        if packed.take(name, t.dtype(), t.shape(), t.data())? {
            continue;
        }
        let data = to_f32(name, t.dtype(), t.data())?;
        if out.insert(name.to_string(), data).is_some() {
            bail!("duplicate tensor {name} in {}", path.display());
        }
    }
    Ok(())
}

fn to_f32(name: &str, dtype: Dtype, data: &[u8]) -> Result<Vec<f32>> {
    match dtype {
        Dtype::F32 => {
            if !data.len().is_multiple_of(4) {
                bail!(
                    "tensor {name}: F32 byte length {} not multiple of 4",
                    data.len()
                );
            }
            Ok(data
                .chunks_exact(4)
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect())
        }
        Dtype::BF16 => {
            if !data.len().is_multiple_of(2) {
                bail!(
                    "tensor {name}: BF16 byte length {} not multiple of 2",
                    data.len()
                );
            }
            Ok(data
                .chunks_exact(2)
                .map(|b| {
                    let u = u16::from_le_bytes([b[0], b[1]]);
                    f32::from_bits((u as u32) << 16)
                })
                .collect())
        }
        other => bail!("tensor {name}: unsupported dtype {other:?} (want BF16 or F32)"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kimi_k3::cpu_weights::{Ablation, MixerW};
    use crate::kimi_k3::greedy::greedy_decode;
    use crate::kimi_k3::layer::K3Graph;
    use safetensors::tensor::{TensorView, serialize};
    use std::env;

    const TINY_CFG: &str = r#"{
      "model_type": "kimi_k3",
      "text_config": {
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
        "attn_res_block_size": 2,
        "first_k_dense_replace": 1,
        "head_dim": 2,
        "hidden_act": "situ",
        "hidden_size": 4,
        "intermediate_size": 4,
        "kv_lora_rank": 2,
        "latent_moe_use_norm": true,
        "linear_attn_config": {
          "full_attn_layers": [2],
          "head_dim": 2,
          "kda_layers": [1],
          "num_heads": 1,
          "short_conv_kernel_size": 4,
          "use_full_rank_gate": true
        },
        "mla_use_nope": true,
        "mla_use_output_gate": true,
        "model_type": "kimi_linear",
        "moe_intermediate_size": 4,
        "moe_renormalize": true,
        "moe_router_activation_func": "sigmoid",
        "num_attention_heads": 1,
        "num_experts": 2,
        "num_experts_per_token": 1,
        "num_hidden_layers": 2,
        "num_shared_experts": 1,
        "q_lora_rank": 4,
        "qk_nope_head_dim": 2,
        "qk_rope_head_dim": 2,
        "rms_norm_eps": 1e-5,
        "rope_theta": 10000.0,
        "routed_expert_hidden_size": 4,
        "topk_method": "noaux_tc",
        "v_head_dim": 2,
        "vocab_size": 8
      }
    }"#;

    fn write_f32_st(dir: &Path, tensors: &[(String, usize)]) {
        let backing: Vec<(String, Vec<u8>)> = tensors
            .iter()
            .map(|(n, len)| {
                let bytes: Vec<u8> = (0..*len)
                    .flat_map(|i| (0.01 * (i as f32 + 1.0)).to_le_bytes())
                    .collect();
                (n.clone(), bytes)
            })
            .collect();
        let views: Vec<(String, TensorView)> = tensors
            .iter()
            .zip(backing.iter())
            .map(|((n, len), (_, b))| {
                (
                    n.clone(),
                    TensorView::new(Dtype::F32, vec![*len], b).expect(n),
                )
            })
            .collect();
        let bytes = serialize(views, None).expect("serialize");
        std::fs::write(dir.join("model.safetensors"), bytes).unwrap();
    }

    fn scratch_dir(tag: &str) -> PathBuf {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let dir = env::temp_dir().join(format!("k3-{tag}-{}-{nanos}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn tiny_dir() -> PathBuf {
        let dir = scratch_dir("cpu-load");
        std::fs::write(dir.join("config.json"), TINY_CFG).unwrap();
        let c = parse_config(TINY_CFG).unwrap();
        let graph = K3Graph::from_config(&c);
        let inv = inventory(&c, &graph, &kda_from(&c), &mla_from(&c), &moe_from(&c));
        write_f32_st(&dir, &inv);
        dir
    }

    #[test]
    fn from_pretrained_missing_dir_errors() {
        let err = K3CpuModel::from_pretrained(Path::new("/no/such/k3-twin"))
            .unwrap_err()
            .to_string();
        assert!(err.contains("does not exist"), "{err}");
    }

    #[test]
    fn from_pretrained_tiny_safetensors() {
        let dir = tiny_dir();
        let m = K3CpuModel::from_pretrained(&dir).expect("load tiny");
        assert_eq!(m.layers.len(), 2);
        assert_eq!(m.vocab, 8);
        assert_eq!(m.graph.hidden, 4);
        assert!(matches!(m.layers[0].mixer, MixerW::Kda(_)));
        assert!(matches!(m.layers[1].mixer, MixerW::Mla(_)));
        let out = greedy_decode(&m, &[1, 2], 3, Ablation::default());
        assert_eq!(out.len(), 5);
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn from_pretrained_refuses_weight_packed() {
        let dir = scratch_dir("packed");
        std::fs::write(dir.join("config.json"), TINY_CFG).unwrap();
        let data = [0u8; 4];
        let view = TensorView::new(Dtype::U8, vec![4], &data).unwrap();
        let bytes = serialize(
            [(
                "language_model.model.layers.1.block_sparse_moe.experts.0.w1.weight_packed"
                    .to_string(),
                view,
            )],
            None,
        )
        .unwrap();
        std::fs::write(dir.join("model.safetensors"), bytes).unwrap();
        let err = K3CpuModel::from_pretrained(&dir).unwrap_err().to_string();
        assert!(err.contains("S5 MXFP4 not this slice"), "{err}");
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn bf16_converts_one() {
        let bits = 0x3f80u16;
        let v = to_f32("t", Dtype::BF16, &bits.to_le_bytes()).unwrap();
        assert_eq!(v, vec![1.0]);
    }
}
