// SPDX-License-Identifier: AGPL-3.0-only
// Portions translated from HuggingFace transformers (models/lfm3, Apache-2.0, Copyright 2026 The HuggingFace Team) and vLLM (vllm/model_executor/models/lfm3.py, Apache-2.0, Copyright contributors to the vLLM project). Changes: ported from Python/Triton to Rust/CUDA for Atlas. Tensor names: the safetensors header of LiquidAI/LFM3-1.7B (286 tensors) and llama.cpp conversion/lfm3.py (MIT) as a cross-check.

//! Responsibility: the LFM3 tensor-name table and the shape/dtype check that
//! makes a wrong checkpoint fail loudly at load, before any layer is built.
//!
//! One list of `(name, shape, role)` per layer kind, derived from the model
//! geometry; [`problems`] walks it against a lookup (pure, so a CPU test
//! drives it with a fake) and [`validate`] adapts the real [`WeightStore`].
//!
//! Names (`N` = layer index), shapes for LFM3-1.7B:
//!
//! | tensor | shape |
//! |---|---|
//! | `model.embed_tokens.weight` | `[128000, 2048]` |
//! | `model.embedding_norm.weight` (the FINAL norm) | `[2048]` |
//! | `model.layers.N.{operator,post_operator,ffn,post_ffn}_norm.weight` | `[2048]` |
//! | `model.layers.N.feed_forward.w1` (gate) / `w3` (up) | `[7680, 2048]` |
//! | `model.layers.N.feed_forward.w2` (down) | `[2048, 7680]` |
//! | conv: `conv.in_proj` | `[6144, 2048]` (B, C, x) |
//! | conv: `conv.conv` | `[2048, 1, 7]` (depthwise; `[2048, 7]` also accepted) |
//! | conv: `conv.out_proj` | `[2048, 2048]` |
//! | attn: `self_attn.q_proj` / `o_proj` | `[2048, 2048]` |
//! | attn: `self_attn.k_proj` / `v_proj` | `[512, 2048]` |
//! | attn: `self_attn.q_norm` / `k_norm` | `[128]` |
//! | attn: `self_attn.gate_proj` (per-head sigmoid gate) | `[16, 2048]` |
//!
//! `lm_head.weight` is absent on a tied checkpoint.

use anyhow::{Result, bail};
use avarok_core::config::{LayerType, ModelConfig};
use spark_runtime::weights::{WeightDtype, WeightStore};

pub const EMBED: &str = "model.embed_tokens.weight";
/// LFM3's final norm is `model.embedding_norm`, not `model.norm`.
pub const FINAL_NORM: &str = "model.embedding_norm.weight";
pub const LM_HEAD: &str = "lm_head.weight";

/// What dtypes a tensor may carry.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// A projection: BF16, or FP8 E4M3 with a block scale beside it.
    Proj,
    /// A norm weight, the conv taps or the head gate: BF16 (FP32 is converted
    /// on load by `dense_auto`).
    Dense,
    /// The embedding table: BF16 only (it is gathered raw and doubles as the
    /// tied head).
    Embed,
}

/// One expected tensor.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Expect {
    pub name: String,
    pub shape: Vec<usize>,
    /// A second accepted shape (the conv taps ship as `[C, 1, 7]` but are
    /// consumed flat as `[C, 7]`).
    pub alt_shape: Option<Vec<usize>>,
    pub role: Role,
}

impl Expect {
    fn proj(name: String, n: usize, k: usize) -> Self {
        Self {
            name,
            shape: vec![n, k],
            alt_shape: None,
            role: Role::Proj,
        }
    }
    fn dense(name: String, shape: Vec<usize>) -> Self {
        Self {
            name,
            shape,
            alt_shape: None,
            role: Role::Dense,
        }
    }
}

/// The model geometry every expected shape derives from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Geometry {
    pub hidden: usize,
    pub inter: usize,
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub vocab: usize,
    pub conv_width: usize,
}

impl Geometry {
    pub fn of(config: &ModelConfig) -> Self {
        Self {
            hidden: config.hidden_size,
            inter: config.intermediate_size,
            heads: config.num_attention_heads,
            kv_heads: config.num_key_value_heads,
            head_dim: config.head_dim,
            vocab: config.vocab_size,
            conv_width: config.linear_conv_kernel_dim,
        }
    }

    /// Width of the query projection (`heads * head_dim`).
    pub fn q_width(&self) -> usize {
        self.heads * self.head_dim
    }

    /// Width of the key and value projections (`kv_heads * head_dim`).
    pub fn kv_width(&self) -> usize {
        self.kv_heads * self.head_dim
    }

    /// The conv `in_proj` output: B, C and x concatenated.
    pub fn bcx_width(&self) -> usize {
        3 * self.hidden
    }
}

/// The tensors every layer carries: four norms and the SwiGLU FFN.
fn common(g: &Geometry, lp: &str) -> Vec<Expect> {
    let mut v = Vec::new();
    for norm in [
        "operator_norm",
        "post_operator_norm",
        "ffn_norm",
        "post_ffn_norm",
    ] {
        v.push(Expect::dense(format!("{lp}.{norm}.weight"), vec![g.hidden]));
    }
    v.push(Expect::proj(
        format!("{lp}.feed_forward.w1.weight"),
        g.inter,
        g.hidden,
    ));
    v.push(Expect::proj(
        format!("{lp}.feed_forward.w3.weight"),
        g.inter,
        g.hidden,
    ));
    v.push(Expect::proj(
        format!("{lp}.feed_forward.w2.weight"),
        g.hidden,
        g.inter,
    ));
    v
}

/// Every tensor layer `lp` of kind `kind` must carry.
pub fn layer_expectations(g: &Geometry, lp: &str, kind: LayerType) -> Vec<Expect> {
    let mut v = common(g, lp);
    match kind {
        LayerType::LinearAttention => {
            v.push(Expect::proj(
                format!("{lp}.conv.in_proj.weight"),
                g.bcx_width(),
                g.hidden,
            ));
            v.push(Expect {
                name: format!("{lp}.conv.conv.weight"),
                shape: vec![g.hidden, 1, g.conv_width],
                alt_shape: Some(vec![g.hidden, g.conv_width]),
                role: Role::Dense,
            });
            v.push(Expect::proj(
                format!("{lp}.conv.out_proj.weight"),
                g.hidden,
                g.hidden,
            ));
        }
        _ => {
            let p = format!("{lp}.self_attn");
            v.push(Expect::proj(
                format!("{p}.q_proj.weight"),
                g.q_width(),
                g.hidden,
            ));
            v.push(Expect::proj(
                format!("{p}.k_proj.weight"),
                g.kv_width(),
                g.hidden,
            ));
            v.push(Expect::proj(
                format!("{p}.v_proj.weight"),
                g.kv_width(),
                g.hidden,
            ));
            v.push(Expect::proj(
                format!("{p}.o_proj.weight"),
                g.hidden,
                g.q_width(),
            ));
            v.push(Expect::dense(
                format!("{p}.q_norm.weight"),
                vec![g.head_dim],
            ));
            v.push(Expect::dense(
                format!("{p}.k_norm.weight"),
                vec![g.head_dim],
            ));
            v.push(Expect::dense(
                format!("{p}.gate_proj.weight"),
                vec![g.heads, g.hidden],
            ));
        }
    }
    v
}

/// The model-level tensors (the head is checked separately: it is optional
/// on a tied checkpoint).
pub fn global_expectations(g: &Geometry) -> Vec<Expect> {
    vec![
        Expect {
            name: EMBED.to_string(),
            shape: vec![g.vocab, g.hidden],
            alt_shape: None,
            role: Role::Embed,
        },
        Expect::dense(FINAL_NORM.to_string(), vec![g.hidden]),
    ]
}

/// What a lookup returns for a tensor name.
pub type Found = Option<(Vec<usize>, WeightDtype)>;

fn dtype_ok(role: Role, dtype: WeightDtype) -> bool {
    match role {
        Role::Proj => matches!(dtype, WeightDtype::BF16 | WeightDtype::FP8E4M3),
        Role::Dense => matches!(dtype, WeightDtype::BF16 | WeightDtype::FP32),
        Role::Embed => dtype == WeightDtype::BF16,
    }
}

/// Every discrepancy between the checkpoint and the model geometry, one
/// sentence each (empty = consistent). Pure over `lookup`.
pub fn problems(config: &ModelConfig, lookup: &dyn Fn(&str) -> Found) -> Vec<String> {
    let g = Geometry::of(config);
    let mut expects = global_expectations(&g);
    for i in 0..config.num_hidden_layers {
        expects.extend(layer_expectations(
            &g,
            &config.layer_prefix(i),
            config.layer_type(i),
        ));
    }
    let mut out = Vec::new();
    for e in &expects {
        let Some((shape, dtype)) = lookup(&e.name) else {
            out.push(format!("{} is missing", e.name));
            continue;
        };
        let shape_ok = shape == e.shape || e.alt_shape.as_ref() == Some(&shape);
        if !shape_ok {
            out.push(format!(
                "{} has shape {shape:?}, expected {:?}",
                e.name, e.shape
            ));
        }
        if !dtype_ok(e.role, dtype) {
            out.push(format!("{} has unsupported dtype {dtype:?}", e.name));
        }
    }
    if let Some((shape, dtype)) = lookup(LM_HEAD) {
        if shape != [g.vocab, g.hidden] {
            out.push(format!(
                "{LM_HEAD} has shape {shape:?}, expected [{}, {}]",
                g.vocab, g.hidden
            ));
        }
        if !matches!(dtype, WeightDtype::BF16 | WeightDtype::FP8E4M3) {
            out.push(format!("{LM_HEAD} has unsupported dtype {dtype:?}"));
        }
    } else if !config.tie_word_embeddings {
        out.push(format!(
            "{LM_HEAD} is missing and tie_word_embeddings is false"
        ));
    }
    out
}

/// Fail with every discrepancy listed (at most 12 lines, then a count).
pub fn validate(store: &WeightStore, config: &ModelConfig) -> Result<()> {
    let found = problems(config, &|name: &str| {
        store.get(name).ok().map(|t| (t.shape.clone(), t.dtype))
    });
    if found.is_empty() {
        return Ok(());
    }
    let shown = found
        .iter()
        .take(12)
        .cloned()
        .collect::<Vec<_>>()
        .join("; ");
    let more = found.len().saturating_sub(12);
    bail!(
        "lfm3 checkpoint does not match its config ({} problem(s)): {shown}{}",
        found.len(),
        if more > 0 {
            format!("; and {more} more")
        } else {
            String::new()
        }
    );
}
