// SPDX-License-Identifier: AGPL-3.0-only

#![allow(clippy::doc_lazy_continuation)]
#![allow(clippy::doc_overindented_list_items)]

//! Per-layer + per-tensor precision overrides (C.3, 2026-04-25).
//!
//! Reference: NVIDIA Transformer Engine 2.14 + EAQuant (arXiv:2506.13329)
//! + community 2025 mixed-precision recipes. In MoE models, the
//! quantization-sensitivity hierarchy holds across re-tested
//! benchmarks:
//!
//!   1. **Router** (gate weights): hidden × num_experts, tiny in
//!      memory, but routing accuracy collapses fast under quant.
//!      Keep BF16 wherever feasible.
//!   2. **LM head**: hidden × vocab, large but determines output
//!      logit fidelity. BF16 closes the dominant chunk of
//!      perplexity gap.
//!   3. **First 1-2 transformer blocks** + **last 2-3 blocks**:
//!      the embedding-adjacent layers carry sink-token outliers and
//!      the output-adjacent layers shape final logits. Keep at FP8
//!      (one tier above the bulk).
//!   4. **Bulk MoE experts**: NVFP4 / FP8 — the model has the most
//!      slack here.
//!
//! ## Scope
//!
//! This module ships:
//!   - [`Role`] — semantic tag for each tensor the loader wants to
//!     classify (router, lm_head, attention, expert, etc.).
//!   - [`Dtype`] — target precision values the schedule emits.
//!   - [`PrecisionSchedule`] — the per-(layer, role) → dtype
//!     decision table, built from `[precision]` in MODEL.toml.
//!
//! The loader consults `schedule.dtype_for(layer_idx, role)` at
//! tensor-load time and chooses the matching path. When the
//! schedule is in its `default()` state (no `[precision]` block in
//! MODEL.toml), every lookup returns `Dtype::Inherit` — meaning
//! "use whatever the existing per-checkpoint logic decides." This
//! keeps the pre-2026-04-25 behaviour bit-exact.

use std::collections::BTreeSet;

/// Semantic role of a tensor, used for precision lookups. The set is
/// closed and minimal — adding a new role requires extending the
/// `Dtype::for_role` match.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Role {
    /// MoE router gate (hidden × num_experts).
    Router,
    /// Final unembedding (hidden × vocab).
    LmHead,
    /// Token embedding (vocab × hidden).
    Embedding,
    /// Attention Q/K/V/O projection.
    Attention,
    /// Expert FFN (gate / up / down per expert).
    Expert,
    /// Shared-expert FFN, when present (DeepSeek-V3 / Qwen3.5 style).
    SharedExpert,
    /// Layer norm scales (RMSNorm `weight`).
    Norm,
}

impl Role {
    pub fn name(&self) -> &'static str {
        match self {
            Role::Router => "router",
            Role::LmHead => "lm_head",
            Role::Embedding => "embedding",
            Role::Attention => "attention",
            Role::Expert => "expert",
            Role::SharedExpert => "shared_expert",
            Role::Norm => "norm",
        }
    }

    /// Parse a role tag from MODEL.toml. Returns `None` for unknown
    /// names so the operator gets a load-time warning rather than a
    /// silent miss.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "router" => Some(Role::Router),
            "lm_head" => Some(Role::LmHead),
            "embedding" => Some(Role::Embedding),
            "attention" => Some(Role::Attention),
            "expert" => Some(Role::Expert),
            "shared_expert" => Some(Role::SharedExpert),
            "norm" => Some(Role::Norm),
            _ => None,
        }
    }
}

/// Target precision for a tensor. `Inherit` means "let the existing
/// per-checkpoint logic decide" (preserves pre-C.3 behaviour); the
/// other variants are hard requests.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    /// Honour the existing checkpoint-format detection and CLI flag.
    /// Equivalent to "no override" — the loader falls through.
    Inherit,
    Bf16,
    Fp8,
    Nvfp4,
}

impl Dtype {
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "inherit" => Some(Dtype::Inherit),
            "bf16" => Some(Dtype::Bf16),
            "fp8" => Some(Dtype::Fp8),
            "nvfp4" => Some(Dtype::Nvfp4),
            _ => None,
        }
    }

    /// True iff the loader should override the inherited path.
    pub fn is_override(&self) -> bool {
        !matches!(self, Dtype::Inherit)
    }
}

/// Per-(layer, role) precision schedule built from MODEL.toml's
/// `[precision]` block. Lookups are O(1) for the common tables;
/// per-layer overrides hit a small sorted set.
#[derive(Debug, Clone)]
pub struct PrecisionSchedule {
    /// Default for any (layer, role) not specifically overridden.
    /// Typically `Dtype::Inherit` so the existing path runs unchanged.
    default: Dtype,
    /// Role-specific defaults. `router_dtype = "bf16"` populates
    /// `roles[Role::Router]`. Lookups fall back from per-layer
    /// override → per-role default → global default.
    router_dtype: Dtype,
    lm_head_dtype: Dtype,
    embedding_dtype: Dtype,
    attention_dtype: Dtype,
    expert_dtype: Dtype,
    shared_expert_dtype: Dtype,
    norm_dtype: Dtype,
    /// Layer indices marked "sensitive" — typically the first 1-2
    /// and last 2-3 transformer blocks. Tensors of role
    /// `Attention`/`Expert` in these layers get `sensitive_dtype`.
    sensitive_layers: BTreeSet<u16>,
    sensitive_dtype: Dtype,
}

impl Default for PrecisionSchedule {
    /// Empty schedule — every lookup returns `Inherit`. Bit-exact
    /// equivalent to the pre-C.3 behaviour. MODEL.toml omits the
    /// `[precision]` block to opt into this default.
    fn default() -> Self {
        Self {
            default: Dtype::Inherit,
            router_dtype: Dtype::Inherit,
            lm_head_dtype: Dtype::Inherit,
            embedding_dtype: Dtype::Inherit,
            attention_dtype: Dtype::Inherit,
            expert_dtype: Dtype::Inherit,
            shared_expert_dtype: Dtype::Inherit,
            norm_dtype: Dtype::Inherit,
            sensitive_layers: BTreeSet::new(),
            sensitive_dtype: Dtype::Inherit,
        }
    }
}

impl PrecisionSchedule {
    /// Build from the four documented MODEL.toml fields:
    ///   - `router_dtype`: dtype for the MoE gate
    ///   - `lm_head_dtype`: dtype for the final unembedding
    ///   - `sensitive_block_dtype` + `sensitive_block_indices`: the
    ///     "extra precision" tier for the first/last few blocks
    ///   - `default_dtype`: bulk fallback (typically Inherit)
    ///
    /// For now, the simpler `[precision]` schema only exposes these
    /// four; per-tensor / per-layer YAML can extend later.
    pub fn build(
        router_dtype: Dtype,
        lm_head_dtype: Dtype,
        sensitive_block_indices: &[u16],
        sensitive_block_dtype: Dtype,
        default_dtype: Dtype,
    ) -> Self {
        Self {
            default: default_dtype,
            router_dtype,
            lm_head_dtype,
            embedding_dtype: Dtype::Inherit,
            attention_dtype: Dtype::Inherit,
            expert_dtype: Dtype::Inherit,
            shared_expert_dtype: Dtype::Inherit,
            norm_dtype: Dtype::Inherit,
            sensitive_layers: sensitive_block_indices.iter().copied().collect(),
            sensitive_dtype: sensitive_block_dtype,
        }
    }

    /// Resolve the target dtype for a tensor. `layer_idx = None` is
    /// used for non-layer tensors (embedding, lm_head, final norm).
    /// Lookup order:
    ///   1. Sensitive-layer override (only for Attention/Expert)
    ///   2. Per-role default
    ///   3. Global default
    pub fn dtype_for(&self, layer_idx: Option<u16>, role: Role) -> Dtype {
        // Sensitive-layer pass: applies to weight-bearing layer
        // tensors only. Norms / embeddings / LM head are exempt
        // (they have their own role-specific overrides).
        if let Some(li) = layer_idx
            && matches!(role, Role::Attention | Role::Expert | Role::SharedExpert)
            && self.sensitive_layers.contains(&li)
            && self.sensitive_dtype.is_override()
        {
            return self.sensitive_dtype;
        }
        let role_dtype = match role {
            Role::Router => self.router_dtype,
            Role::LmHead => self.lm_head_dtype,
            Role::Embedding => self.embedding_dtype,
            Role::Attention => self.attention_dtype,
            Role::Expert => self.expert_dtype,
            Role::SharedExpert => self.shared_expert_dtype,
            Role::Norm => self.norm_dtype,
        };
        if role_dtype.is_override() {
            role_dtype
        } else {
            self.default
        }
    }

    /// True iff the schedule will produce any non-Inherit overrides.
    /// Loaders can use this to skip the per-tensor lookups entirely
    /// when no overrides are configured (default case).
    pub fn has_any_override(&self) -> bool {
        self.default.is_override()
            || self.router_dtype.is_override()
            || self.lm_head_dtype.is_override()
            || self.embedding_dtype.is_override()
            || self.attention_dtype.is_override()
            || self.expert_dtype.is_override()
            || self.shared_expert_dtype.is_override()
            || self.norm_dtype.is_override()
            || (self.sensitive_dtype.is_override() && !self.sensitive_layers.is_empty())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_schedule_is_all_inherit() {
        let s = PrecisionSchedule::default();
        assert!(!s.has_any_override());
        assert_eq!(s.dtype_for(None, Role::LmHead), Dtype::Inherit);
        assert_eq!(s.dtype_for(Some(0), Role::Router), Dtype::Inherit);
        assert_eq!(s.dtype_for(Some(38), Role::Expert), Dtype::Inherit);
    }

    #[test]
    fn role_override_wins_over_default() {
        let s = PrecisionSchedule::build(
            Dtype::Bf16,    // router
            Dtype::Bf16,    // lm head
            &[],            // no sensitive layers
            Dtype::Inherit, // sensitive dtype unused
            Dtype::Nvfp4,   // bulk default
        );
        assert!(s.has_any_override());
        assert_eq!(s.dtype_for(None, Role::Router), Dtype::Bf16);
        assert_eq!(s.dtype_for(None, Role::LmHead), Dtype::Bf16);
        assert_eq!(s.dtype_for(Some(15), Role::Expert), Dtype::Nvfp4);
    }

    #[test]
    fn sensitive_layer_overrides_role_default_for_weights() {
        let s = PrecisionSchedule::build(
            Dtype::Bf16,
            Dtype::Bf16,
            &[0, 1, 38, 39],
            Dtype::Fp8,
            Dtype::Nvfp4,
        );
        // Layer 0 expert is sensitive → FP8
        assert_eq!(s.dtype_for(Some(0), Role::Expert), Dtype::Fp8);
        // Layer 38 attention is sensitive → FP8
        assert_eq!(s.dtype_for(Some(38), Role::Attention), Dtype::Fp8);
        // Layer 5 expert is bulk → NVFP4
        assert_eq!(s.dtype_for(Some(5), Role::Expert), Dtype::Nvfp4);
    }

    #[test]
    fn sensitive_layer_does_not_override_router_or_lm_head() {
        // Router is not Attention/Expert; sensitivity table never
        // applies to it. Routing dtype is governed by router_dtype only.
        let s = PrecisionSchedule::build(Dtype::Bf16, Dtype::Bf16, &[0], Dtype::Fp8, Dtype::Nvfp4);
        assert_eq!(s.dtype_for(Some(0), Role::Router), Dtype::Bf16);
        assert_eq!(s.dtype_for(Some(0), Role::LmHead), Dtype::Bf16);
    }

    #[test]
    fn role_str_round_trips() {
        for r in [
            Role::Router,
            Role::LmHead,
            Role::Embedding,
            Role::Attention,
            Role::Expert,
            Role::SharedExpert,
            Role::Norm,
        ] {
            assert_eq!(Role::from_str(r.name()), Some(r));
        }
        assert_eq!(Role::from_str("nonsense"), None);
    }

    #[test]
    fn dtype_str_parsing() {
        assert_eq!(Dtype::from_str("bf16"), Some(Dtype::Bf16));
        assert_eq!(Dtype::from_str("fp8"), Some(Dtype::Fp8));
        assert_eq!(Dtype::from_str("nvfp4"), Some(Dtype::Nvfp4));
        assert_eq!(Dtype::from_str("inherit"), Some(Dtype::Inherit));
        assert_eq!(Dtype::from_str("bogus"), None);
        assert!(!Dtype::Inherit.is_override());
        assert!(Dtype::Bf16.is_override());
    }
}
