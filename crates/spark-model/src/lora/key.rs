// SPDX-License-Identifier: AGPL-3.0-only

//! LoRA key classification + adapter identity: the FNV-1a `adapter_id_hash`
//! cache key and the `classify_key` PEFT-key → (layer, module, A|B) decoder
//! (every unsupported shape a NAMED hard rejection). Split out of the former
//! monolithic `lora/mod.rs` (SDD seam: KEY CLASSIFICATION) — visibility
//! unchanged.

use anyhow::{Result, anyhow, bail};
use atlas_core::config::{LayerType, ModelConfig};

use super::*;

/// Stable u64 identity for an adapter, derived from its human NAME (never the
/// runtime pool slot index, which is reused across swap/rotation). Task #24:
/// this is the cache-identity key that keeps the KV/prefix cache adapter-correct
/// so a request reuses ONLY blocks computed under the same adapter.
///
/// FNV-1a over the name bytes. `0` is the RESERVED base/no-adapter sentinel, so
/// a real name that would hash to 0 is bumped to 1 — a real adapter never aliases
/// base. Two different names never collide (modulo the 64-bit hash); the SAME
/// adapter re-staged into a different pool slot keeps its name, hence its id.
///
/// Task #25 (slot generation): `generation` folds into the identity ONLY when it
/// is non-zero, so `generation == 0` returns byte-identically to the pre-#25
/// value (first-load ids and the base sentinel are unchanged — the #24 base
/// byte-identity pins hold). A re-staged slot bumps its generation, changing the
/// id so a later request under the SAME name misses the stale prior-generation
/// prefix/KV. The `if h == 0 { 1 }` base-reserve is re-applied AFTER the fold so
/// no (name, generation) pair can alias the base sentinel.
pub fn adapter_id_hash(name: &str, generation: u64) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325; // FNV-1a basis
    for &b in name.as_bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3); // FNV-1a prime
    }
    // gen 0 = strict no-op → byte-identical to the pre-#25 name-only hash.
    if generation != 0 {
        for &b in generation.to_le_bytes().iter() {
            h ^= b as u64;
            h = h.wrapping_mul(0x100000001b3);
        }
    }
    if h == 0 { 1 } else { h }
}

/// PEFT key → (layer, module, A|B). Every unsupported shape is a NAMED
/// hard rejection — never a skip. Prefix-agnostic on purpose: the Holo
/// base checkpoint keys are `model.language_model.layers.{i}.*`
/// (weight_prefix auto-detected server-side), but a PEFT trainer wrapping
/// the text trunk emits `model.layers.{i}.*`; both carry the layer index
/// right after ".layers.".
pub fn classify_key(key: &str, cfg: &ModelConfig) -> Result<(usize, LoraModule, AdapterAb)> {
    let stripped = key.strip_prefix("base_model.model.").ok_or_else(|| {
        anyhow!("REJECT[not-peft-key]: '{key}' lacks the 'base_model.model.' PEFT prefix")
    })?;
    if stripped.contains("lora_embedding_") {
        bail!("REJECT[embedding-lora]: '{key}' — embed_tokens/lm_head LoRA is out of v0 scope");
    }
    let (module_path, ab) = if let Some(p) = stripped.strip_suffix(".lora_A.weight") {
        (p, AdapterAb::A)
    } else if let Some(p) = stripped.strip_suffix(".lora_B.weight") {
        (p, AdapterAb::B)
    } else {
        bail!(
            "REJECT[unrecognized-tensor]: '{key}' is not a lora_A/lora_B weight \
             (modules_to_save exports and old '.lora_A.<adapter>.weight' layouts \
             are not supported in v0)"
        );
    };
    let (_prefix, rest) = module_path.split_once(".layers.").ok_or_else(|| {
        anyhow!("REJECT[non-layer-module]: '{key}' targets '{module_path}' outside the layer stack")
    })?;
    let (idx_str, tail) = rest
        .split_once('.')
        .ok_or_else(|| anyhow!("REJECT[malformed-key]: '{key}'"))?;
    let layer_idx: usize = idx_str
        .parse()
        .map_err(|_| anyhow!("REJECT[malformed-layer-index]: '{key}'"))?;
    if layer_idx >= cfg.num_hidden_layers {
        bail!(
            "REJECT[layer-out-of-range]: '{key}' targets layer {layer_idx} \
             (model has {})",
            cfg.num_hidden_layers
        );
    }
    let module = match tail {
        "self_attn.q_proj" => LoraModule::QProj,
        "self_attn.k_proj" => LoraModule::KProj,
        "self_attn.v_proj" => LoraModule::VProj,
        "self_attn.o_proj" => LoraModule::OProj,
        "mlp.gate_proj" => LoraModule::GateProj,
        "mlp.up_proj" => LoraModule::UpProj,
        "mlp.down_proj" => LoraModule::DownProj,
        t if t.starts_with("linear_attn.") => bail!(
            "REJECT[gdn-target]: '{key}' — GDN/linear-attention projections \
             (in_proj_qkv / in_proj_z / in_proj_a / in_proj_b / out_proj) are \
             rejected until an exact-replay parity harness exists"
        ),
        other => bail!("REJECT[unsupported-module]: '{key}' targets '{other}'"),
    };
    match cfg.layer_type(layer_idx) {
        LayerType::FullAttention => {}
        lt => bail!(
            "REJECT[non-full-attention-layer]: '{key}' targets layer {layer_idx} \
             ({lt:?}); v0 applies LoRA only on the full-attention layers \
             {:?}. NOTE: dense mlp.* exists on the GDN layers too — train with \
             layers_to_transform=[3,7,11,15,19,23] to produce a loadable adapter",
            full_attention_layers(cfg)
        ),
    }
    Ok((layer_idx, module, ab))
}

#[cfg(test)]
#[path = "key_tests.rs"]
mod tests;
