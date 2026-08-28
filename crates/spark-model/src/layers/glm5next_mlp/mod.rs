// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3-Flash **MLP production surface** — dense FFN and routed MoE.
//!
//! Scoped to `LibertAIDAI/GLM-5.3-Flash-NVFP4@9e0d74e3`.
//!
//! Same shape as [`crate::layers::glm5next_dsa`]: the CUDA kernels already exist and are
//! numerically proven against HF 5.16.1 on real weights — `kernels/gb10/common/glm5next_ffn.cu`
//! (clamped SwiGLU, sigmoid router top-k, routed/shared combine), gated by
//! `examples/glm5next_{ffn,moe}_microtest.rs`. What was missing, and what this module adds, is
//! the production surface: config, kernel resolution, a weight contract and a forward a real
//! layer can call, rather than an example wiring pointers by hand.
//!
//! # The stack this layer runs
//!
//! ```text
//! DENSE (layers 0..first_k_dense_replace)   x -> gate_proj/up_proj -> clamped SwiGLU -> down_proj
//!
//! ROUTED (every later layer)
//!   x ─┬─ gate.weight ──── logits(f32) ─ router_topk ─ ids[K], weights[K]
//!      ├─ experts[id]  ─── NVFP4 gate/up -> clamped SwiGLU -> NVFP4 down  (LOCAL ids only)
//!      └─ shared_experts ─ BF16 gate/up  -> clamped SwiGLU -> BF16 down
//!                                       └─> moe_combine -> partial -> all_reduce
//! ```
//!
//! # 🔴 Why one all-reduce covers BOTH EP and TP
//!
//! The campaign's topology is `world = 2, TP = 2, EP = 2` on the *same* two ranks. The routed
//! experts are EP-sharded (144/rank, remote ids contribute zero) and the dense/shared FFN is
//! TP-sharded (column-parallel gate/up, row-parallel down). Both leave a **partial sum** of the
//! same `[T, hidden]` output, and `all_reduce(SUM)` is linear, so the two partials are summed by
//! one collective at the end of the site. Reducing them separately would be two collectives for
//! the same answer.
//!
//! 🪤 That is only true because the combine happens BEFORE the reduce. Adding the shared expert
//! after an all-reduce of the routed partial — which is what `layers::moe::forward` does, for a
//! model whose shared expert is replicated rather than TP-sharded — would add a TP-partial
//! shared output exactly once and lose the other rank's half.
//!
//! # 🪤 Traps this module exists to hold
//!
//! * **The SwiGLU clamp is ASYMMETRIC**: `gate` is upper-bounded only, `up` is bounded both
//!   ways. It is also invisible on well-scaled activations — it fires on the tails. The limit
//!   comes from `ModelConfig::swiglu_limit`, which the `glm5_next` parser refuses to default.
//! * **The router's correction bias steers SELECTION ONLY.** The emitted weight is the
//!   *unbiased* sigmoid score of the chosen expert. Gathering the biased score still produces a
//!   plausible mixture.
//! * **`routed_scaling_factor` rides on the top-k weights and the shared expert is NOT scaled by
//!   it** (`apply_routed_scale_to_output = false`). Scaling the shared output is the same defect
//!   with the opposite sign.
//! * **The router is REPLICATED and must stay bit-identical across ranks.** Masked-local EP is
//!   only equivalent to dispatch when every rank agrees on the same `ids`. Sharding the gate
//!   would give each rank partial logits and a different top-k — no crash, different experts.
//! * **`num_experts` here is the FULL count (288).** The rank's local range is a separate field.
//!   Passing the local count to `glm5next_router_topk` would rank 144 experts and renormalise
//!   over the wrong denominator.

use anyhow::{Result, bail};
use atlas_core::config::{Glm5NextRouterMode, ModelConfig};
use spark_runtime::gpu::{GpuBackend, KernelHandle};

pub mod build;
pub mod forward;
pub mod weights;

pub use weights::{Glm5NextDenseMlpWeights, Glm5NextExpertWeights, Glm5NextMoeWeights};

/// Module name the GLM FFN kernels resolve from. `kernels/gb10/common/glm5next_ffn.cu` is not
/// listed in `common/KERNEL.toml`'s `[modules]`, so it takes its **file stem**.
///
/// 🪤 The other three modules below are the opposite case and are listed: `dense_gemm_bf16` maps
/// to `"gemm"` and `w4a16_gemm` maps to `"w4a16"`. Guessing the stem for those resolves to
/// nothing — the bug caught at `0846fad3`. Grep `[modules]` before writing any `resolve()`.
pub const FFN_MODULE: &str = "glm5next_ffn";
/// `[modules]`: `dense_gemm_bf16 = "gemm"`.
pub const GEMM_MODULE: &str = "gemm";
/// `[modules]`: `w4a16_gemm = "w4a16"`.
pub const W4A16_MODULE: &str = "w4a16";

/// `float best_w[16]` in `glm5next_router_topk` — the most experts it can select per token.
pub const KERNEL_MAX_TOP_K: usize = 16;

/// Every kernel a GLM MLP site launches.
///
/// Resolved with `kernel()` (not `try_kernel`): a missing entry point is a hard error, never a
/// silent fallback onto `moe_silu_mul`, whose SwiGLU does **not** clamp.
#[derive(Clone, Copy)]
pub struct Glm5NextMlpKernels {
    /// `C = A @ B^T`, BF16 out — dense FFN, shared expert.
    pub gemm: KernelHandle,
    /// Same, FP32 out. 🪤 The router GEMM MUST take this one: `glm5next_router_topk` reads
    /// `const float* logits`, while `gate.weight` is BF16 on disk.
    pub gemm_f32: KernelHandle,
    /// NVFP4 `C = A @ B^T` for the routed experts.
    pub w4a16: KernelHandle,
    /// 🪤 **Clamped** SwiGLU, asymmetric. Not `moe_silu_mul`.
    pub swiglu: KernelHandle,
    pub router: KernelHandle,
    pub combine: KernelHandle,
}

impl Glm5NextMlpKernels {
    pub fn resolve(gpu: &dyn GpuBackend) -> Result<Self> {
        Ok(Self {
            gemm: gpu.kernel(GEMM_MODULE, "dense_gemm_bf16")?,
            gemm_f32: gpu.kernel(GEMM_MODULE, "dense_gemm_bf16_f32out")?,
            w4a16: gpu.kernel(W4A16_MODULE, "w4a16_gemm")?,
            swiglu: gpu.kernel(FFN_MODULE, "glm5next_swiglu_clamp")?,
            router: gpu.kernel(FFN_MODULE, "glm5next_router_topk")?,
            combine: gpu.kernel(FFN_MODULE, "glm5next_moe_combine")?,
        })
    }
}

/// Which MLP a layer runs. Mirrors [`crate::layers::glm5next_skeleton::Mlp`]; kept separate so
/// the runtime does not depend on the skeleton's design-artifact types.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Glm5NextMlpKind {
    Dense,
    RoutedMoe,
}

/// GLM MLP geometry for one rank, read from the checkpoint config — never defaulted.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Glm5NextMlpConfig {
    pub hidden: usize,
    /// `intermediate_size` — the dense layers' FFN width, **this rank's share**.
    pub local_dense_intermediate: usize,
    /// `moe_intermediate_size` — one routed expert's width. **Never TP-sharded**: an expert is
    /// owned whole by one EP rank.
    pub moe_intermediate: usize,
    /// `n_shared_experts * moe_intermediate_size`, **this rank's share**.
    pub local_shared_intermediate: usize,
    /// FULL routed-expert count (288). Not the local count — see the module header.
    pub num_experts: usize,
    /// Experts this rank owns, as a contiguous range `[ep_rank * local, +local)`.
    pub local_experts: usize,
    pub ep_rank: usize,
    pub top_k: usize,
    /// `routed_scaling_factor`. 🪤 Applied to the top-k WEIGHTS, never to the shared expert.
    pub routed_scale: f32,
    /// `norm_topk_prob` — renormalise the top-k weights (with a `1e-20` epsilon, not `1e-6`).
    pub renormalize: bool,
    /// 🪤 Asymmetric. See the module header.
    pub swiglu_limit: f32,
    /// True reproduces vLLM's bf16 router ladder; false is HF 5.16.1's fp32 semantics and the
    /// production default. **Semantic, not precision** — the two select different experts.
    pub router_bf16_ladder: bool,
    /// TP ranks the dense/shared FFN is split over. `> 1` ⇒ the site output is a partial sum.
    pub tp_world_size: usize,
    /// EP ranks the routed experts are split over. `> 1` ⇒ the routed sum is a partial sum.
    pub ep_world_size: usize,
}

impl Glm5NextMlpConfig {
    /// `config` carries GLOBAL MoE counts; the TP division of the dense/shared widths and the EP
    /// division of the expert set are applied here, in one place.
    pub fn from_config(config: &ModelConfig) -> Result<Self> {
        let tp = config.tp_world_size.max(1);
        let ep = config.ep_world_size.max(1);
        if !config.intermediate_size.is_multiple_of(tp) {
            bail!(
                "GLM MLP: intermediate_size {} does not divide over tp_world_size {tp}",
                config.intermediate_size
            );
        }
        if !config.shared_expert_intermediate_size.is_multiple_of(tp) {
            bail!(
                "GLM MLP: shared_expert_intermediate_size {} does not divide over \
                 tp_world_size {tp}",
                config.shared_expert_intermediate_size
            );
        }
        if !config.num_experts.is_multiple_of(ep) {
            bail!(
                "GLM MLP: num_experts {} does not divide over ep_world_size {ep}; a ragged \
                 expert split would leave some ids owned by nobody",
                config.num_experts
            );
        }
        let c = Self {
            hidden: config.hidden_size,
            local_dense_intermediate: config.intermediate_size / tp,
            moe_intermediate: config.moe_intermediate_size,
            local_shared_intermediate: config.shared_expert_intermediate_size / tp,
            num_experts: config.num_experts,
            local_experts: config.num_experts / ep,
            ep_rank: config.ep_rank,
            top_k: config.num_experts_per_tok,
            routed_scale: config.routed_scaling_factor as f32,
            renormalize: config.norm_topk_prob,
            swiglu_limit: config.swiglu_limit,
            router_bf16_ladder: matches!(config.glm5next_router_mode, Glm5NextRouterMode::VllmBf16),
            tp_world_size: tp,
            ep_world_size: ep,
        };
        c.validate()?;
        Ok(c)
    }

    /// The half-open global expert-id range this rank owns.
    pub fn local_expert_range(&self) -> std::ops::Range<usize> {
        let start = self.ep_rank * self.local_experts;
        start..start + self.local_experts
    }

    /// Global expert id → local slot, or `None` when another rank owns it.
    ///
    /// 🪤 The whole EP scheme rests on this: a remote id must contribute **zero**, not be
    /// clamped into a local slot. Indexing a local array with a global id is the silent version
    /// of that mistake and yields a real expert's weights for the wrong token.
    pub fn local_slot(&self, global_id: usize) -> Option<usize> {
        let r = self.local_expert_range();
        r.contains(&global_id).then(|| global_id - r.start)
    }

    /// Whether the site output leaves this rank as a partial sum needing `all_reduce(SUM)`.
    pub fn needs_all_reduce(&self) -> bool {
        self.tp_world_size > 1 || self.ep_world_size > 1
    }

    pub fn validate(&self) -> Result<()> {
        if self.hidden == 0 {
            bail!("GLM MLP: hidden_size is 0");
        }
        if self.swiglu_limit <= 0.0 {
            bail!(
                "GLM MLP: swiglu_limit is {}. GLM-5.3 clamps its SwiGLU and the clamp is \
                 asymmetric; a zero limit is not 'no clamp', it is a gate forced to <= 0. \
                 The glm5_next parser reads the real value (10.0) and refuses to default it.",
                self.swiglu_limit
            );
        }
        if self.top_k == 0 || self.top_k > KERNEL_MAX_TOP_K {
            bail!(
                "GLM MLP: num_experts_per_tok {} is outside the {}-slot bound \
                 glm5next_router_topk keeps in registers (`float best_w[16]`)",
                self.top_k,
                KERNEL_MAX_TOP_K
            );
        }
        if self.top_k > self.num_experts {
            bail!(
                "GLM MLP: top_k {} exceeds num_experts {}",
                self.top_k,
                self.num_experts
            );
        }
        if self.moe_intermediate == 0 {
            bail!("GLM MLP: moe_intermediate_size is 0 — a routed layer would compute nothing");
        }
        if self.ep_rank >= self.ep_world_size {
            bail!(
                "GLM MLP: ep_rank {} is outside ep_world_size {}",
                self.ep_rank,
                self.ep_world_size
            );
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests;
