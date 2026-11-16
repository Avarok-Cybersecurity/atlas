// SPDX-License-Identifier: AGPL-3.0-only
// Portions follow the ARC-1.0 release documentation (weight names of the Qwen3.5 text stack under
// `model.language_model.`), Copyright 2026 Atlas Futures, Inc., licensed under the Apache License 2.0. By Atlas
// contributors. See RAYLINE-ROUTING-ATTRIBUTION.md.
//! Responsibility: build an [`Encoder`] from a release directory (or the stock checkpoint): parse `text_config`, read
//! the `arc.depth` layers (or a given depth), the embedding table and the final norm from `model.language_model.*`
//! (BF16 weights stay BF16), check every shape, and apply the tied LM head from the embedding table.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use rayon::prelude::*;

use super::ops::dot_f32;
use super::{
    AttnWeights, Encoder, EncoderOptions, GdnWeights, Layer, LayerType, Lin, Mixer, TextConfig,
};
use crate::route_gate::error::{GResult, GateError, value_err};
use crate::route_gate::tensorio::TensorFile;

const PREFIX: &str = "model.language_model.";

/// Where the embedding table lives: owned BF16 bits (tests) or the memory-mapped release file.
pub enum EmbedSource {
    Owned(Vec<u16>),
    File { file: Arc<TensorFile>, name: String },
}

impl EmbedSource {
    /// The embedding rows of `ids`, `[ids.len(), hidden]`.
    pub fn rows(&self, ids: &[u32], hidden: usize, vocab: usize) -> GResult<Vec<f32>> {
        let mut out = Vec::with_capacity(ids.len() * hidden);
        for &id in ids {
            let id = id as usize;
            if id >= vocab {
                return value_err(format!("token id {id} is outside the vocabulary ({vocab})"));
            }
            match self {
                Self::Owned(bits) => {
                    out.extend(
                        bits[id * hidden..(id + 1) * hidden]
                            .iter()
                            .map(|&b| f32::from_bits(u32::from(b) << 16)),
                    );
                }
                Self::File { file, name } => {
                    let bytes = file.raw(name)?;
                    out.extend(
                        bytes[id * hidden * 2..(id + 1) * hidden * 2]
                            .chunks_exact(2)
                            .map(|c| {
                                f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16)
                            }),
                    );
                }
            }
        }
        Ok(out)
    }

    /// Widens row `r` of the table into `out` (`bytes` is the mapped table of a `File` source).
    fn widen_row(&self, bytes: Option<&[u8]>, r: usize, out: &mut [f32]) {
        let h = out.len();
        match (self, bytes) {
            (Self::Owned(bits), _) => {
                for (d, &b) in out.iter_mut().zip(&bits[r * h..(r + 1) * h]) {
                    *d = f32::from_bits(u32::from(b) << 16);
                }
            }
            (Self::File { .. }, Some(b)) => {
                for (d, c) in out
                    .iter_mut()
                    .zip(b[r * h * 2..(r + 1) * h * 2].chunks_exact(2))
                {
                    *d = f32::from_bits(u32::from(u16::from_le_bytes([c[0], c[1]])) << 16);
                }
            }
            (Self::File { .. }, None) => out.fill(0.0),
        }
    }

    /// The tied LM head: `x` is `[t, hidden]` (final-normed hidden states), the result `[t, vocab]` f32 logits
    /// `x . E^T` with the BF16 embedding rows widened exactly.
    pub fn logits(&self, x: &[f32], t: usize, hidden: usize, vocab: usize) -> GResult<Vec<f32>> {
        if x.len() != t * hidden || t == 0 {
            return value_err(format!("logits: input of {} for t={t}", x.len()));
        }
        let bytes = match self {
            Self::File { file, name } => {
                let b = file.raw(name)?;
                if b.len() < vocab * hidden * 2 {
                    return value_err("logits: embedding table is smaller than vocab x hidden");
                }
                Some(b)
            }
            Self::Owned(bits) => {
                if bits.len() < vocab * hidden {
                    return value_err("logits: embedding table is smaller than vocab x hidden");
                }
                None
            }
        };
        let mut cols = vec![0f32; vocab * t];
        cols.par_chunks_mut(t).enumerate().for_each_init(
            || vec![0f32; hidden],
            |row, (r, col)| {
                self.widen_row(bytes, r, row);
                for (tt, c) in col.iter_mut().enumerate() {
                    *c = dot_f32(row, &x[tt * hidden..(tt + 1) * hidden]);
                }
            },
        );
        let mut out = vec![0f32; t * vocab];
        for (r, col) in cols.chunks(t).enumerate() {
            for (tt, v) in col.iter().enumerate() {
                out[tt * vocab + r] = *v;
            }
        }
        Ok(out)
    }
}

fn check_shape(file: &TensorFile, name: &str, want: &[usize]) -> GResult<()> {
    let got = file.shape(name)?;
    if got != want {
        return value_err(format!("{name}: shape {got:?}, expected {want:?}"));
    }
    Ok(())
}

fn floats(file: &TensorFile, name: &str, want: &[usize]) -> GResult<Vec<f32>> {
    check_shape(file, name, want)?;
    Ok(file.read_f32(name)?.1)
}

fn lin(file: &TensorFile, name: &str, out: usize, inp: usize) -> GResult<Lin> {
    check_shape(file, name, &[out, inp])?;
    let (_, bits) = file.read_bf16_bits(name)?;
    Ok(Lin::from_bits(out, inp, bits))
}

/// The weights file of a model directory: `model.safetensors`, else the only `model.safetensors-*-of-00001` shard (the
/// layout of the stock Hugging Face checkpoint).
pub fn weights_file(dir: &Path) -> GResult<PathBuf> {
    let single = dir.join("model.safetensors");
    if single.is_file() {
        return Ok(single);
    }
    let mut shards: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.file_name()
                .and_then(|n| n.to_str())
                .is_some_and(|n| n.starts_with("model.safetensors-") && n.ends_with(".safetensors"))
        })
        .collect();
    shards.sort();
    match shards.as_slice() {
        [one] => Ok(one.clone()),
        [] => value_err(format!("{}: no model.safetensors", dir.display())),
        _ => value_err(format!(
            "{}: {} weight shards; only single-file checkpoints are supported",
            dir.display(),
            shards.len()
        )),
    }
}

impl Encoder {
    /// Loads the stack from `release_dir` (`config.json` and `model.safetensors`), `arc.depth` layers deep (every layer
    /// of the text config when the config has no `arc` block).
    pub fn load(release_dir: impl AsRef<Path>, opts: EncoderOptions) -> GResult<Self> {
        Self::load_depth(release_dir, None, opts)
    }

    /// Like [`Encoder::load`], with the depth given explicitly (`None`: `arc.depth`, else every layer). The full-LM
    /// oracle loads the stock checkpoint this way with all of its layers.
    pub fn load_depth(
        release_dir: impl AsRef<Path>,
        depth: Option<usize>,
        opts: EncoderOptions,
    ) -> GResult<Self> {
        let dir = release_dir.as_ref();
        let cfg_json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        let text = cfg_json.get("text_config").unwrap_or(&cfg_json);
        let cfg = TextConfig::from_json(text)?;
        let depth = depth.unwrap_or_else(|| {
            cfg_json["arc"]["depth"]
                .as_u64()
                .map_or(cfg.layer_types.len(), |d| d as usize)
        });
        if depth > cfg.layer_types.len() {
            return value_err(format!(
                "depth {depth} exceeds the {} layers of the text config",
                cfg.layer_types.len()
            ));
        }
        let file = Arc::new(TensorFile::open(weights_file(dir)?)?);
        let n = |s: &str| format!("{PREFIX}{s}");
        let embed_name = n("embed_tokens.weight");
        check_shape(&file, &embed_name, &[cfg.vocab, cfg.hidden])?;
        if file.dtype(&embed_name)? != "BF16" {
            return Err(GateError::Value(format!("{embed_name}: expected BF16")));
        }
        let h = cfg.hidden;
        let (key_dim, value_dim) = (
            cfg.lin_k_heads * cfg.lin_k_dim,
            cfg.lin_v_heads * cfg.lin_v_dim,
        );
        let conv_dim = 2 * key_dim + value_dim;
        let mut layers = Vec::with_capacity(depth);
        for (i, ty) in cfg.layer_types.iter().take(depth).enumerate() {
            let p = |s: &str| n(&format!("layers.{i}.{s}"));
            let mixer = match ty {
                LayerType::Linear => Mixer::Gdn(GdnWeights {
                    in_qkv: lin(&file, &p("linear_attn.in_proj_qkv.weight"), conv_dim, h)?,
                    in_z: lin(&file, &p("linear_attn.in_proj_z.weight"), value_dim, h)?,
                    in_b: lin(
                        &file,
                        &p("linear_attn.in_proj_b.weight"),
                        cfg.lin_v_heads,
                        h,
                    )?,
                    in_a: lin(
                        &file,
                        &p("linear_attn.in_proj_a.weight"),
                        cfg.lin_v_heads,
                        h,
                    )?,
                    conv: floats(
                        &file,
                        &p("linear_attn.conv1d.weight"),
                        &[conv_dim, 1, cfg.conv_kernel],
                    )?,
                    a_log: floats(&file, &p("linear_attn.A_log"), &[cfg.lin_v_heads])?,
                    dt_bias: floats(&file, &p("linear_attn.dt_bias"), &[cfg.lin_v_heads])?,
                    norm: floats(&file, &p("linear_attn.norm.weight"), &[cfg.lin_v_dim])?,
                    out: lin(&file, &p("linear_attn.out_proj.weight"), h, value_dim)?,
                }),
                LayerType::Full => {
                    let q_out = cfg.heads * cfg.head_dim * if cfg.attn_gate { 2 } else { 1 };
                    Mixer::Attn(AttnWeights {
                        q: lin(&file, &p("self_attn.q_proj.weight"), q_out, h)?,
                        k: lin(
                            &file,
                            &p("self_attn.k_proj.weight"),
                            cfg.kv_heads * cfg.head_dim,
                            h,
                        )?,
                        v: lin(
                            &file,
                            &p("self_attn.v_proj.weight"),
                            cfg.kv_heads * cfg.head_dim,
                            h,
                        )?,
                        o: lin(
                            &file,
                            &p("self_attn.o_proj.weight"),
                            h,
                            cfg.heads * cfg.head_dim,
                        )?,
                        q_norm: floats(&file, &p("self_attn.q_norm.weight"), &[cfg.head_dim])?,
                        k_norm: floats(&file, &p("self_attn.k_norm.weight"), &[cfg.head_dim])?,
                    })
                }
            };
            layers.push(Layer {
                input_norm: floats(&file, &p("input_layernorm.weight"), &[h])?,
                post_norm: floats(&file, &p("post_attention_layernorm.weight"), &[h])?,
                mixer,
                gate: lin(&file, &p("mlp.gate_proj.weight"), cfg.inter, h)?,
                up: lin(&file, &p("mlp.up_proj.weight"), cfg.inter, h)?,
                down: lin(&file, &p("mlp.down_proj.weight"), h, cfg.inter)?,
            });
        }
        let final_norm = floats(&file, &n("norm.weight"), &[h])?;
        Ok(Self {
            cfg,
            opts,
            embed: EmbedSource::File {
                file,
                name: embed_name,
            },
            layers,
            final_norm,
        })
    }
}
