// SPDX-License-Identifier: AGPL-3.0-only
// Portions follow the ARC-1.0 release documentation (weight names of the Qwen3.5 text stack under
// `model.language_model.`), Copyright 2026 Atlas Futures, Inc., licensed under the Apache License 2.0. By Atlas
// contributors. See RAYLINE-ROUTING-ATTRIBUTION.md.
//! Responsibility: build an [`Encoder`] from a release directory: parse `text_config`, read exactly the 12 layers, the
//! embedding table and the final norm from `model.language_model.*` (BF16 weights stay BF16), and check every shape.

use std::path::Path;
use std::sync::Arc;

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

impl Encoder {
    /// Loads the stack from `release_dir` (`config.json` and `model.safetensors`).
    pub fn load(release_dir: impl AsRef<Path>, opts: EncoderOptions) -> GResult<Self> {
        let dir = release_dir.as_ref();
        let cfg_json: serde_json::Value =
            serde_json::from_slice(&std::fs::read(dir.join("config.json"))?)?;
        let text = cfg_json.get("text_config").unwrap_or(&cfg_json);
        let cfg = TextConfig::from_json(text)?;
        let depth = cfg_json["arc"]["depth"]
            .as_u64()
            .map_or(cfg.layer_types.len(), |d| d as usize);
        if depth > cfg.layer_types.len() {
            return value_err(format!(
                "arc.depth {depth} exceeds the {} layers of the text config",
                cfg.layer_types.len()
            ));
        }
        let file = Arc::new(TensorFile::open(dir.join("model.safetensors"))?);
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
