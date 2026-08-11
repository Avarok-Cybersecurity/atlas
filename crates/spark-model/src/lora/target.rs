// SPDX-License-Identifier: AGPL-3.0-only

//! Feature-1 (MoE expert + router LoRA) target surface.
//!
//! [`classify_key`](super::classify_key) historically decoded a PEFT tensor
//! name to `(layer, LoraModule, A|B)` where [`LoraModule`] is the flat 7-variant
//! dense-projection enum (q/k/v/o + gate/up/down). MoE routed-expert and router
//! deltas need a third dimension the dense enum cannot carry: the EXPERT INDEX
//! (`mlp.experts.{N}.{gate,up,down}_proj`) and the router (`mlp.gate`, distinct
//! from the dense `mlp.gate_proj`). [`LoraTarget`] wraps the dense enum and adds
//! those two variants so the classifier return keeps `LoraModule` intact for the
//! attention/dense path (zero blast-radius on the S-LoRA BGMV route tables) while
//! routing expert/router deltas into their own sparse per-layer storage.
//!
//! CORRECTNESS-FIRST (Feature-1 phase 1): the internal representation is always
//! PER-EXPERT ([`ExpertLoraLayer`], a sparse `BTreeMap<(expert, proj), LoraPair>`),
//! applied via the existing `apply_lora_delta` fold (no new CUDA kernel). Fused
//! on-disk import (`target_parameters` / fused `experts.gate_up_proj`) and the
//! grouped-BGMV / fused-epilogue kernels are explicit follow-ups (phases 2/3).

use std::collections::BTreeMap;

use atlas_core::config::ModelConfig;

use crate::layers::ops::lora_delta::LoraPair;

use super::LoraModule;

/// Which routed-expert projection a delta targets. Ordered so
/// `BTreeMap<(u16, ExpertProj), _>` has a stable, testable key order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ExpertProj {
    Gate,
    Up,
    Down,
}

impl ExpertProj {
    /// PEFT suffix (the leaf `.`-segment of the on-disk tensor name).
    pub fn peft_name(&self) -> &'static str {
        match self {
            Self::Gate => "gate_proj",
            Self::Up => "up_proj",
            Self::Down => "down_proj",
        }
    }

    /// (out_dim, in_dim) of the base routed-expert projection on `layer`.
    ///
    /// Uses the PER-LAYER routed intermediate ([`ModelConfig::moe_intermediate_size_for`],
    /// e.g. 512 on Holo-3.1-35B-A3B) — NEVER the dense `intermediate_size`
    /// (5120), which belongs to the standalone SwiGLU FFN. gate/up map hidden→
    /// inter, down maps inter→hidden.
    pub fn dims(&self, cfg: &ModelConfig, layer: usize) -> (usize, usize) {
        let h = cfg.hidden_size;
        let inter = cfg.moe_intermediate_size_for(layer);
        match self {
            Self::Gate | Self::Up => (inter, h),
            Self::Down => (h, inter),
        }
    }
}

/// (out_dim, in_dim) of the base router (`mlp.gate`) projection: `[num_experts,
/// hidden]`. A router LoRA perturbs the pre-selection routing logits.
pub fn router_dims(cfg: &ModelConfig) -> (usize, usize) {
    (cfg.num_experts, cfg.hidden_size)
}

/// The decoded target of one PEFT LoRA tensor. `Attn` keeps the existing dense
/// [`LoraModule`] path byte-identical; `Router` / `Expert` are the Feature-1
/// additions routed into per-layer [`ExpertLoraLayer`] / router storage.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoraTarget {
    /// One of the 7 dense projections (q/k/v/o/gate/up/down_proj).
    Attn(LoraModule),
    /// The MoE router `mlp.gate` (`[num_experts, hidden]`).
    Router,
    /// One routed expert's projection (`mlp.experts.{n}.{proj}`).
    Expert { n: u16, proj: ExpertProj },
}

/// One MoE layer's routed-expert LoRA coverage: a SPARSE map keyed by
/// `(expert_index, projection)`. Real adapters adapt a subset of the (up to 512)
/// experts, so a dense `Vec<[LoraPair; 3]>` sized `num_experts` would waste
/// storage and force a 512-wide static walk; the map only holds the packed pairs.
/// `LoraPair` is `Copy`, so the whole struct is cheap to `Clone` on install.
#[derive(Clone, Default)]
pub struct ExpertLoraLayer {
    pub pairs: BTreeMap<(u16, ExpertProj), LoraPair>,
}

impl ExpertLoraLayer {
    /// This layer's `(expert, proj)` pair, if adapted.
    pub fn pair(&self, expert: u16, proj: ExpertProj) -> Option<&LoraPair> {
        self.pairs.get(&(expert, proj))
    }

    /// Sorted, deduped list of the expert indices this layer adapts — the
    /// SSOT for "which experts get a delta applied" in the forward side-path.
    pub fn adapted_experts(&self) -> Vec<u16> {
        let mut v: Vec<u16> = self.pairs.keys().map(|(e, _)| *e).collect();
        v.dedup();
        v
    }

    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }
}

/// Pure padded-byte estimator for the (separate) expert/router pool, used by the
/// VRAM preflight and pinned by a golden unit test — mirrors
/// `super::pool_slot_bytes` but over the audited routed-expert + router key
/// set (real adapters target a SUBSET, so this is sized from the audit, never
/// from `num_experts × num_layers` maxima). Per (layer, expert, proj) and per
/// router layer: `(stride·in + out·stride)·2` BF16 bytes, where `stride` is
/// the DERIVED uint4-aligned `expert_pack::packed_stride` of
/// `max_rank` — the same derivation the pack loop uses (SSOT), so sizing and
/// packing agree byte-for-byte even at a non-multiple-of-8 rank cap.
pub fn expert_router_bytes(
    cfg: &ModelConfig,
    expert_keys: &[(usize, ExpertProj)],
    router_layers: &[usize],
    max_rank: usize,
) -> usize {
    let stride = super::expert_pack::packed_stride(max_rank);
    let per = |out: usize, inp: usize| (stride * inp + out * stride) * 2;
    let experts: usize = expert_keys
        .iter()
        .map(|(layer, proj)| {
            let (out, inp) = proj.dims(cfg, *layer);
            per(out, inp)
        })
        .sum();
    let routers: usize = router_layers
        .iter()
        .map(|_| {
            let (out, inp) = router_dims(cfg);
            per(out, inp)
        })
        .sum();
    experts + routers
}

#[cfg(test)]
#[path = "target_tests.rs"]
mod tests;
