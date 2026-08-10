// SPDX-License-Identifier: AGPL-3.0-only

//! GGUF tensor-name → Atlas HF tensor-name translation.
//!
//! GGUF names decoder weights as `blk.N.<sub>` plus a handful of top-level
//! tensors (`token_embd`, `output_norm`, `output`). Atlas per-arch loaders ask
//! the [`crate::weights::WeightStore`] for HuggingFace names
//! (`model.layers.N.self_attn.q_proj.weight`, …). This module is the pure,
//! side-effect-free bridge between the two. It emits standard HF names for the
//! `weight_prefix = "model"` convention (see `ModelConfig::layer_prefix`).
//!
//! Expert-stacked GGUF tensors (`blk.N.ffn_{gate,up,down}_exps.weight`) are a
//! single `[n_expert, …]` tensor that Atlas expects as `num_experts` separate
//! `…experts.{E}.*` tensors. A 1:1 name map cannot express that fan-out, so
//! those names resolve to [`GgufName::ExpertStack`] and the loader is
//! responsible for slicing + naming each expert. Everything else resolves to
//! [`GgufName::Direct`] (a single HF name) or [`GgufName::Drop`] (metadata-only
//! tensors like `rope_freqs.weight` that carry no learnable weight).

/// Result of translating one GGUF tensor name.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum GgufName {
    /// A single HF tensor name to store the (dequantized) weight under.
    Direct(String),
    /// A stacked MoE expert tensor `blk.N.ffn_<which>_exps.weight`. The loader
    /// must split the leading `[n_expert]` dimension and emit, per expert `E`,
    /// `model.layers.N.mlp.experts.{E}.{proj}_proj.weight`.
    ExpertStack {
        layer: usize,
        /// `"gate"`, `"up"`, or `"down"` — the projection to suffix as
        /// `{proj}_proj`.
        proj: &'static str,
    },
    /// Tensor carries no learnable weight for Atlas (e.g. precomputed rope
    /// frequencies); the loader should skip it.
    Drop,
}

/// The HF weight prefix Atlas defaults to when `weight_prefix` is empty
/// (`ModelConfig::layer_prefix` → `model.layers.N`). Kept as a constant so the
/// non-layer names below stay in sync with the per-layer names.
const HF_PREFIX: &str = "model";

/// Translate a GGUF tensor name to its Atlas HF equivalent for architecture
/// `arch` (the value of GGUF metadata key `general.architecture`, already
/// lower-cased by the caller — e.g. `"llama"`, `"qwen2"`, `"qwen3"`,
/// `"gemma2"`). Returns `None` for names this translator does not recognize, so
/// the loader can surface an explicit error rather than silently mis-storing a
/// tensor.
pub fn translate(gguf_name: &str, arch: &str) -> Option<GgufName> {
    // Per-arch override hook. Only `default` is populated today; add arms here
    // as GGUF variants diverge (e.g. a future arch that renames `ffn_norm`).
    if let Some(overridden) = arch_override(gguf_name, arch) {
        return Some(overridden);
    }
    translate_default(gguf_name)
}

/// Architecture-specific overrides. Return `Some(_)` to short-circuit
/// [`translate_default`], `None` to fall through so [`translate_default`] runs.
fn arch_override(gguf_name: &str, arch: &str) -> Option<GgufName> {
    match arch {
        // Qwen3.5/3.6 GDN-hybrid (`general.architecture = qwen35`). llama.cpp
        // emits the linear-attention (Gated DeltaNet) projections under `ssm_*`
        // / `attn_{qkv,gate}` names and the second RMSNorm as
        // `post_attention_norm`; everything else (attn_q/k/v/output, q/k norms,
        // ffn_*, top-level tensors) matches the default translation.
        "qwen35" | "qwen3_5" => translate_qwen35_layer(gguf_name),
        // NLLB-200 / M2M-100 encoder-decoder translation family. Names diverge
        // wholesale from the decoder-only default (two stacks `enc/dec.blk.N.*`,
        // tied `token_embd` → `model.shared`, cross-attention), so this arch is
        // fully handled here and never falls through to `translate_default`.
        "nllb" | "m2m_100" => translate_nllb(gguf_name),
        // mmproj vision tower (`general.architecture = clip`,
        // `clip.projector_type = qwen3vl_merger`). Names live in a disjoint
        // namespace (`v.*` / `mm.*`) from the text backbone, so this never
        // collides with the qwen35 text arm — the two GGUFs open separately.
        "clip" => translate_clip(gguf_name),
        _ => None,
    }
}

/// NLLB-200 / M2M-100 encoder-decoder name remaps (`general.architecture =
/// nllb` / `m2m_100`). llama.cpp names the two stacks `enc.blk.N.*` /
/// `dec.blk.N.*` plus per-stack `{enc,dec}.output_norm`; the GPU `model/nllb`
/// runtime (`crate`-external `spark_model::model::nllb`) indexes weights by
/// HuggingFace M2M-100 keys (`model.{encoder,decoder}.layers.N.*`,
/// `model.shared.weight`). Both F16 (projections/embedding) and F32
/// (norms/biases) GGUF tensors are dequantized uniformly to BF16 by the loader,
/// which is exactly the dtype `model/nllb` requires — so no per-tensor dtype
/// handling is needed here.
///
/// GGUF → HF, per encoder/decoder block `N`:
///   * `attn_{q,k,v,o}`       → `self_attn.{q,k,v,out}_proj`
///   * `attn_norm`            → `self_attn_layer_norm`
///   * `cross_attn_{q,k,v,o}` → `encoder_attn.{q,k,v,out}_proj`  (decoder only)
///   * `cross_attn_norm`      → `encoder_attn_layer_norm`        (decoder only)
///   * `ffn_up` / `ffn_down`  → `fc1` / `fc2`
///   * `ffn_norm`             → `final_layer_norm`
///
/// Top-level: `token_embd` → `model.shared` (tied embedding),
/// `{enc,dec}.output_norm` → `model.{encoder,decoder}.layer_norm`. The learned
/// `position_embd.weight` resolves to [`GgufName::Drop`] — NLLB regenerates
/// sinusoidal positions at runtime.
fn translate_nllb(gguf_name: &str) -> Option<GgufName> {
    // ── Top-level (non-block) tensors ──
    let top = match gguf_name {
        "token_embd.weight" => Some("model.shared.weight"),
        "position_embd.weight" => return Some(GgufName::Drop),
        "enc.output_norm.weight" => Some("model.encoder.layer_norm.weight"),
        "enc.output_norm.bias" => Some("model.encoder.layer_norm.bias"),
        "dec.output_norm.weight" => Some("model.decoder.layer_norm.weight"),
        "dec.output_norm.bias" => Some("model.decoder.layer_norm.bias"),
        _ => None,
    };
    if let Some(hf) = top {
        return Some(GgufName::Direct(hf.to_string()));
    }

    // ── Per-block tensors: `enc.blk.N.<sub>` / `dec.blk.N.<sub>` ──
    let (side, rest) = if let Some(r) = gguf_name.strip_prefix("enc.blk.") {
        ("encoder", r)
    } else if let Some(r) = gguf_name.strip_prefix("dec.blk.") {
        ("decoder", r)
    } else {
        return None;
    };
    let (n_str, tail) = rest.split_once('.')?;
    let layer: usize = n_str.parse().ok()?;
    // Split off the `.weight` / `.bias` extension (every NLLB projection, norm
    // and FFN weight carries a bias).
    let (module, ext) = match tail.rsplit_once('.') {
        Some((module, ext @ ("weight" | "bias"))) => (module, ext),
        _ => return None,
    };
    let hf_module = match module {
        // Self-attention (both stacks).
        "attn_q" => "self_attn.q_proj",
        "attn_k" => "self_attn.k_proj",
        "attn_v" => "self_attn.v_proj",
        "attn_o" => "self_attn.out_proj",
        "attn_norm" => "self_attn_layer_norm",
        // Cross-attention (decoder only; encoder never carries these names).
        "cross_attn_q" => "encoder_attn.q_proj",
        "cross_attn_k" => "encoder_attn.k_proj",
        "cross_attn_v" => "encoder_attn.v_proj",
        "cross_attn_o" => "encoder_attn.out_proj",
        "cross_attn_norm" => "encoder_attn_layer_norm",
        // Feed-forward.
        "ffn_up" => "fc1",
        "ffn_down" => "fc2",
        "ffn_norm" => "final_layer_norm",
        _ => return None,
    };
    Some(GgufName::Direct(format!(
        "model.{side}.layers.{layer}.{hf_module}.{ext}"
    )))
}

/// HF prefix Atlas's Qwen3.6 ViT tower loads its tensors under. See
/// `Qwen35WeightLoader::load_vision_encoder`, which probes
/// `model.visual.patch_embed.proj.weight`. The mmproj GGUF path always produces
/// the flat form, so we emit `model.visual.*`.
const VISION_PREFIX: &str = "model.visual";

/// Translate an mmproj (`general.architecture = clip`) tensor name to the
/// `model.visual.*` HF name Atlas's Qwen3.6 vision encoder expects.
///
/// llama.cpp's `clip` writer (projector `qwen3vl_merger`) names the tower:
///   * per-block  `v.blk.N.{attn_qkv,attn_out,ffn_up,ffn_down,ln1,ln2}.{w,b}`
///   * patch conv `v.patch_embd.{weight,weight.1,bias}`
///   * pos table  `v.position_embd.weight`
///   * post-norm  `v.post_ln.{weight,bias}`      (→ the final merger's norm)
///   * projector  `mm.0.{w,b}` (fc1), `mm.2.{w,b}` (fc2); `mm.1` = GELU, no tensor
///
/// The consumer (`Qwen35WeightLoader::load_vision_encoder`) reads:
/// `blocks.N.{norm1,attn.qkv,attn.proj,norm2,mlp.linear_fc1,mlp.linear_fc2}`,
/// `patch_embed.proj`, `pos_embed.weight`, and a single `merger.{norm,
/// linear_fc1,linear_fc2}`.
///
/// NOTE — the patch-embed WEIGHT (`v.patch_embd.weight{,.1}`) is a temporal-split
/// Conv3d that a 1:1 map cannot fuse; those frames are intercepted by the loader
/// (`value_transform::vision_patch_frame` → `patch_embed_concat`) BEFORE this
/// translator runs, so they are (correctly) left unmatched here. Only the
/// patch-embed BIAS maps 1:1.
fn translate_clip(gguf_name: &str) -> Option<GgufName> {
    // ── Per-block: `v.blk.N.<stem>.<ext>` ──
    if let Some(rest) = gguf_name.strip_prefix("v.blk.") {
        let (n_str, sub) = rest.split_once('.')?;
        let layer: usize = n_str.parse().ok()?;
        let (stem, ext) = match sub.rsplit_once('.') {
            Some((stem, ext @ ("weight" | "bias"))) => (stem, ext),
            _ => return None,
        };
        let hf_sub = match stem {
            "ln1" => "norm1",
            "ln2" => "norm2",
            "attn_qkv" => "attn.qkv",
            "attn_out" => "attn.proj",
            "ffn_up" => "mlp.linear_fc1",
            "ffn_down" => "mlp.linear_fc2",
            _ => return None,
        };
        return Some(GgufName::Direct(format!(
            "{VISION_PREFIX}.blocks.{layer}.{hf_sub}.{ext}"
        )));
    }

    // ── Top-level (non-block) tensors ──
    let hf = match gguf_name {
        // Projector MLP: mm.0 = fc1, mm.2 = fc2 (mm.1 is the GELU, no tensor).
        "mm.0.weight" => format!("{VISION_PREFIX}.merger.linear_fc1.weight"),
        "mm.0.bias" => format!("{VISION_PREFIX}.merger.linear_fc1.bias"),
        "mm.2.weight" => format!("{VISION_PREFIX}.merger.linear_fc2.weight"),
        "mm.2.bias" => format!("{VISION_PREFIX}.merger.linear_fc2.bias"),
        // Post-encoder LayerNorm feeds the final merger → merger.norm.
        "v.post_ln.weight" => format!("{VISION_PREFIX}.merger.norm.weight"),
        "v.post_ln.bias" => format!("{VISION_PREFIX}.merger.norm.bias"),
        // Learned absolute position table `[2304, 1152]` (interpolated/image).
        "v.position_embd.weight" => format!("{VISION_PREFIX}.pos_embed.weight"),
        // Patch-embed Conv3d bias maps 1:1 (the weight frames are loader-fused).
        "v.patch_embd.bias" => format!("{VISION_PREFIX}.patch_embed.proj.bias"),
        _ => return None,
    };
    Some(GgufName::Direct(hf))
}

/// Qwen3.5/3.6-specific per-layer name remaps. Returns `None` for names the
/// default translator already handles (so they fall through), and for anything
/// unrecognized.
///
/// Layer roles are positional (every `full_attention_interval`-th layer is full
/// attention, the rest are GDN/linear-attention), but the tensor NAMES are
/// unambiguous per role: full-attention layers carry `attn_q/k/v/output`, GDN
/// layers carry `attn_qkv` (fused Q|K|V), `attn_gate` (the Z gate) and the
/// `ssm_*` family. So a pure name map suffices — no layer-index arithmetic.
///
/// GDN → Atlas HF mapping (`model.layers.N.linear_attn.*`, consumed by
/// `Qwen35DenseWeightLoader`'s `LinearAttention` arm):
///   * `attn_qkv`   → `in_proj_qkv`   (fused Q|K|V, rows = ssm_qkv_size)
///   * `attn_gate`  → `in_proj_z`     (Z gate, rows = ssm_z_size)
///   * `ssm_alpha`  → `in_proj_a`     ([num_v_heads, hidden])
///   * `ssm_beta`   → `in_proj_b`     ([num_v_heads, hidden])
///   * `ssm_conv1d` → `conv1d`
///   * `ssm_a`      → `A_log`         (bare — no `.weight`; F32 decay, per v-head)
///   * `ssm_dt.bias`→ `dt_bias`       (bare — loader keys it without `.bias`)
///   * `ssm_norm`   → `norm`
///   * `ssm_out`    → `out_proj`
fn translate_qwen35_layer(gguf_name: &str) -> Option<GgufName> {
    let rest = gguf_name.strip_prefix("blk.")?;
    let (n_str, sub) = rest.split_once('.')?;
    let layer: usize = n_str.parse().ok()?;
    let la = |suffix: &str| format!("{HF_PREFIX}.layers.{layer}.linear_attn.{suffix}");

    let hf = match sub {
        // Second RMSNorm (both layer roles). GGUF names it `post_attention_norm`
        // rather than the default translator's `ffn_norm`.
        "post_attention_norm.weight" => {
            format!("{HF_PREFIX}.layers.{layer}.post_attention_layernorm.weight")
        }
        // GDN / linear-attention (Gated DeltaNet) projections.
        "attn_qkv.weight" => la("in_proj_qkv.weight"),
        "attn_gate.weight" => la("in_proj_z.weight"),
        "ssm_alpha.weight" => la("in_proj_a.weight"),
        "ssm_beta.weight" => la("in_proj_b.weight"),
        "ssm_conv1d.weight" => la("conv1d.weight"),
        "ssm_norm.weight" => la("norm.weight"),
        "ssm_out.weight" => la("out_proj.weight"),
        // Bare (extension-less) SSM gate params — the loader keys A_log / dt_bias
        // without a `.weight` / `.bias` suffix, and loads them as F32.
        "ssm_a" => la("A_log"),
        "ssm_dt.bias" => la("dt_bias"),
        // Everything else (attn_norm, attn_q/k/v/output, attn_q/k_norm, ffn_*)
        // is handled by the default translator.
        _ => return None,
    };
    Some(GgufName::Direct(hf))
}

/// The default (llama/qwen2/qwen3/gemma-family) name translation.
fn translate_default(gguf_name: &str) -> Option<GgufName> {
    // ── Top-level (non-layer) tensors ──
    match gguf_name {
        "token_embd.weight" => {
            return Some(GgufName::Direct(format!("{HF_PREFIX}.embed_tokens.weight")));
        }
        "output_norm.weight" => {
            return Some(GgufName::Direct(format!("{HF_PREFIX}.norm.weight")));
        }
        // Untied LM head. (Tied models omit this tensor and reuse token_embd.)
        "output.weight" => return Some(GgufName::Direct("lm_head.weight".to_string())),
        // Precomputed rope frequency table — Atlas builds rope itself.
        "rope_freqs.weight" => return Some(GgufName::Drop),
        _ => {}
    }

    // ── Per-layer tensors: `blk.<N>.<sub>` ──
    let rest = gguf_name.strip_prefix("blk.")?;
    let (n_str, sub) = rest.split_once('.')?;
    let layer: usize = n_str.parse().ok()?;

    translate_layer_sub(layer, sub)
}

/// Translate the portion of a per-layer name after `blk.N.`.
fn translate_layer_sub(layer: usize, sub: &str) -> Option<GgufName> {
    // Stacked MoE experts fan out to many HF tensors — handled by the loader.
    match sub {
        "ffn_gate_exps.weight" => {
            return Some(GgufName::ExpertStack {
                layer,
                proj: "gate",
            });
        }
        "ffn_up_exps.weight" => return Some(GgufName::ExpertStack { layer, proj: "up" }),
        "ffn_down_exps.weight" => {
            return Some(GgufName::ExpertStack {
                layer,
                proj: "down",
            });
        }
        _ => {}
    }

    // Split off the `.weight` / `.bias` extension so biases (qwen2 QKV) map for
    // free alongside their weight.
    let (stem, ext) = match sub.rsplit_once('.') {
        Some((stem, ext @ ("weight" | "bias"))) => (stem, ext),
        // Unknown extension → unrecognized; let the caller error explicitly.
        _ => return None,
    };

    // Map the GGUF sub-stem to the HF sub-path (without extension).
    let hf_sub = match stem {
        // Norms
        "attn_norm" => "input_layernorm",
        "ffn_norm" => "post_attention_layernorm",
        // Attention projections
        "attn_q" => "self_attn.q_proj",
        "attn_k" => "self_attn.k_proj",
        "attn_v" => "self_attn.v_proj",
        "attn_output" => "self_attn.o_proj",
        // Qwen3 per-head QK RMSNorm
        "attn_q_norm" => "self_attn.q_norm",
        "attn_k_norm" => "self_attn.k_norm",
        // Dense MLP
        "ffn_gate" => "mlp.gate_proj",
        "ffn_up" => "mlp.up_proj",
        "ffn_down" => "mlp.down_proj",
        // MoE router
        "ffn_gate_inp" => "mlp.gate",
        _ => return None,
    };

    Some(GgufName::Direct(format!(
        "{HF_PREFIX}.layers.{layer}.{hf_sub}.{ext}"
    )))
}

/// True if a mapped HF tensor name is a "big" dense projection that the native
/// keep-packed Q2_0 decode path (`ATLAS_GGUF_NATIVE_Q2=1`) can serve without
/// dequantizing — i.e. its weight stays a raw `block_q2_0` buffer in VRAM.
///
/// Scoped for Tier-1 (decode) to the dense **FFN** projections only
/// (`gate_proj` / `up_proj` / `down_proj`), the bulk of a dense checkpoint's
/// weight. These have no load-time value transform and a matching consumer in
/// `DenseFfnLayer` (native `q2_0_gemv`). Deliberately EXCLUDED for now:
///   * attention `q/k/v/o_proj` and `lm_head` — keep-packable (transform-free)
///     but their layer-side consumers are not yet wired; they stay on the
///     BF16→NVFP4 path. Extending this filter requires adding those setters.
///   * all GDN / linear-attention projections (`in_proj_*`, `out_proj`,
///     `conv1d`, …) — these carry value-head REORDER transforms
///     (`value_transform::needs` is true), and a column reorder splits Q2_0
///     blocks, so they can never be kept packed. The caller ANDs this with
///     `!value_transform::needs(hf)` as a belt-and-suspenders guard.
///
/// Pure + name-only, so it is unit-testable without a GPU or a real GGUF.
pub fn is_keep_packed_proj(hf: &str) -> bool {
    hf.ends_with(".mlp.gate_proj.weight")
        || hf.ends_with(".mlp.up_proj.weight")
        || hf.ends_with(".mlp.down_proj.weight")
        // Tier-1c: FULL-ATTENTION q/k/v/o projections. These are standard
        // attention (NO GDN value-head reorder — `value_transform::needs` is
        // false for them), so they keep-pack DIRECTLY like the FFN. The GDN
        // `linear_attn.in_proj_*` still need a packed row-permute and are
        // handled by `value_transform::packed_reorder_rows`, NOT this filter.
        || hf.ends_with(".self_attn.q_proj.weight")
        || hf.ends_with(".self_attn.k_proj.weight")
        || hf.ends_with(".self_attn.v_proj.weight")
        || hf.ends_with(".self_attn.o_proj.weight")
}

/// Expand an [`GgufName::ExpertStack`] into the concrete per-expert HF name for
/// expert `e`. The loader calls this while slicing the stacked tensor so the
/// naming convention lives next to the translation table it mirrors.
pub fn expert_name(layer: usize, proj: &str, e: usize) -> String {
    format!("{HF_PREFIX}.layers.{layer}.mlp.experts.{e}.{proj}_proj.weight")
}

#[cfg(test)]
mod tests;
