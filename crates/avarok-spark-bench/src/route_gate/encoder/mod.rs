// SPDX-License-Identifier: AGPL-3.0-only
// Portions derived from the ARC-1.0 release documentation and from rayline-ai/vllm-arc (examples/readouts.py, which
// shows how the readouts are taken: normed layer-12 output gathered at block offsets), Copyright 2026 Atlas Futures,
// Inc., licensed under the Apache License 2.0 (https://www.apache.org/licenses/LICENSE-2.0). Reimplemented as a plain
// CPU f32 forward in Rust; no torch, no transformers, no vLLM, no plugin, no fork. By Atlas contributors. See
// RAYLINE-ROUTING-ATTRIBUTION.md.
//! Responsibility: the truncated text stack of the ARC-1.0 encoder (the base model's first 12 layers, `qwen3_5_text`)
//! as a CPU forward that returns the final-RMSNorm hidden state of every token.
//!
//! What a readout is: the final RMSNorm of the base model applied to the output of layer 12 at the last token of each
//! role block (80-routing.md section 3.1). Only `model.language_model.*` tensors are read; the vision tower and every
//! head tensor are never touched. Weights stay BF16 in memory; activations are f32, with an optional `bf16` mode that
//! rounds at the operator boundaries a bf16 torch model would.

mod attn;
mod gdn;
mod load;
mod ops;
mod rope;

pub use attn::{AttnWeights, attn_forward, causal_attention};
pub use gdn::{GdnWeights, delta_rule_head, gdn_forward};
pub use load::EmbedSource;
pub use ops::{
    Lin, add, dot_f32, l2_normalize, maybe_round, rms_norm_1p, rms_norm_gated, round_bf16, silu,
    softplus,
};
pub use rope::{
    RopeConfig, RopeTables, RopeVariant, inv_freq_for, mrope_angles, plain_inv_freq, yarn_inv_freq,
    yarn_mscale,
};

use crate::route_gate::error::{GResult, value_err};

/// Which token mixer a layer uses.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerType {
    Linear,
    Full,
}

/// The text config fields the forward needs (`text_config` of the release `config.json`).
#[derive(Clone, Debug)]
pub struct TextConfig {
    pub hidden: usize,
    pub layer_types: Vec<LayerType>,
    pub inter: usize,
    pub eps: f32,
    pub vocab: usize,
    pub lin_k_heads: usize,
    pub lin_v_heads: usize,
    pub lin_k_dim: usize,
    pub lin_v_dim: usize,
    pub conv_kernel: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub attn_gate: bool,
    pub rope: RopeConfig,
}

impl TextConfig {
    /// Parses a `text_config` object.
    pub fn from_json(v: &serde_json::Value) -> GResult<Self> {
        let usz = |k: &str| -> GResult<usize> {
            v.get(k)
                .and_then(|x| x.as_u64())
                .map(|x| x as usize)
                .map_or_else(|| value_err(format!("text_config.{k} missing")), Ok)
        };
        let types = v
            .get("layer_types")
            .and_then(|x| x.as_array())
            .map_or_else(|| value_err("text_config.layer_types missing"), Ok)?
            .iter()
            .map(|t| match t.as_str() {
                Some("linear_attention") => Ok(LayerType::Linear),
                Some("full_attention") => Ok(LayerType::Full),
                other => value_err(format!("unknown layer type {other:?}")),
            })
            .collect::<GResult<Vec<_>>>()?;
        let rp = v
            .get("rope_parameters")
            .or_else(|| v.get("rope_scaling"))
            .cloned()
            .unwrap_or(serde_json::Value::Null);
        let f =
            |o: &serde_json::Value, k: &str, d: f64| o.get(k).and_then(|x| x.as_f64()).unwrap_or(d);
        let max_position = f(v, "max_position_embeddings", 262_144.0);
        let sections: Vec<usize> = rp
            .get("mrope_section")
            .and_then(|x| x.as_array())
            .map(|a| {
                a.iter()
                    .filter_map(|n| n.as_u64())
                    .map(|n| n as usize)
                    .collect()
            })
            .unwrap_or_else(|| vec![11, 11, 10]);
        if sections.len() != 3 {
            return value_err("mrope_section needs three entries");
        }
        let rope = RopeConfig {
            theta: rp
                .get("rope_theta")
                .and_then(|x| x.as_f64())
                .unwrap_or_else(|| f(v, "rope_theta", 1e7)),
            partial_rotary_factor: rp
                .get("partial_rotary_factor")
                .and_then(|x| x.as_f64())
                .unwrap_or_else(|| f(v, "partial_rotary_factor", 0.25)),
            factor: f(&rp, "factor", 1.0),
            original_max_position: f(&rp, "original_max_position_embeddings", max_position),
            max_position,
            mrope_section: [sections[0], sections[1], sections[2]],
        };
        Ok(Self {
            hidden: usz("hidden_size")?,
            inter: usz("intermediate_size")?,
            eps: f(v, "rms_norm_eps", 1e-6) as f32,
            vocab: usz("vocab_size")?,
            lin_k_heads: usz("linear_num_key_heads")?,
            lin_v_heads: usz("linear_num_value_heads")?,
            lin_k_dim: usz("linear_key_head_dim")?,
            lin_v_dim: usz("linear_value_head_dim")?,
            conv_kernel: usz("linear_conv_kernel_dim")?,
            heads: usz("num_attention_heads")?,
            kv_heads: usz("num_key_value_heads")?,
            head_dim: usz("head_dim")?,
            attn_gate: v
                .get("attn_output_gate")
                .and_then(|x| x.as_bool())
                .unwrap_or(false),
            layer_types: types,
            rope,
        })
    }
}

/// How the encoder computes (the goldens oracle searches these).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct EncoderOptions {
    /// Round at operator boundaries like a bf16 torch model (weights are BF16 either way).
    pub bf16: bool,
    pub rope: RopeVariant,
}

impl Default for EncoderOptions {
    fn default() -> Self {
        Self {
            bf16: false,
            rope: RopeVariant::Yarn,
        }
    }
}

/// One decoder layer: norms, a token mixer and the SwiGLU MLP.
pub struct Layer {
    pub input_norm: Vec<f32>,
    pub post_norm: Vec<f32>,
    pub mixer: Mixer,
    pub gate: Lin,
    pub up: Lin,
    pub down: Lin,
}

/// The token mixer of a layer.
pub enum Mixer {
    Gdn(GdnWeights),
    Attn(AttnWeights),
}

/// The truncated text stack.
pub struct Encoder {
    pub cfg: TextConfig,
    pub opts: EncoderOptions,
    pub embed: EmbedSource,
    pub layers: Vec<Layer>,
    pub final_norm: Vec<f32>,
}

impl Encoder {
    /// Final-RMSNorm hidden states `[t, hidden]` of the stack for `ids` (batch 1, no padding, no cache).
    pub fn hidden(&self, ids: &[u32]) -> GResult<Vec<f32>> {
        let c = &self.cfg;
        let (t, bf16) = (ids.len(), self.opts.bf16);
        if t == 0 {
            return value_err("hidden: no tokens");
        }
        let mut h = self.embed.rows(ids, c.hidden, c.vocab)?;
        let needs_rope = self
            .layers
            .iter()
            .any(|l| matches!(l.mixer, Mixer::Attn(_)));
        let rope = needs_rope.then(|| RopeTables::for_text(&c.rope, c.head_dim, t, self.opts.rope));
        for layer in &self.layers {
            let x = rms_norm_1p(&h, &layer.input_norm, c.eps, bf16);
            let mixed = match &layer.mixer {
                Mixer::Gdn(w) => gdn_forward(c, w, &x, t, bf16),
                Mixer::Attn(w) => match &rope {
                    Some(r) => attn_forward(c, w, &x, t, r, bf16),
                    None => return value_err("rope tables missing"),
                },
            };
            h = add(&h, &mixed, bf16);
            let y = rms_norm_1p(&h, &layer.post_norm, c.eps, bf16);
            let g = layer.gate.forward(&y, t, bf16);
            let u = layer.up.forward(&y, t, bf16);
            let act = maybe_round(g.iter().map(|&v| silu(v)).collect(), bf16);
            let prod = maybe_round(act.iter().zip(&u).map(|(a, b)| a * b).collect(), bf16);
            let m = layer.down.forward(&prod, t, bf16);
            h = add(&h, &m, bf16);
        }
        Ok(rms_norm_1p(&h, &self.final_norm, c.eps, bf16))
    }
}
