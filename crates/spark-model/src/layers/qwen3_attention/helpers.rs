// SPDX-License-Identifier: AGPL-3.0-only

//! `Qwen3AttentionLayer` setters and small per-layer compute helpers
//! (`apply_layer_scalar`, `effective_attn_scale`).

use super::types::{HcWeights, MlaWeights, Qwen3AttentionLayer};
use crate::layers::FfnComponent;
use crate::weight_map::DenseWeight;

/// YaRN attention-temperature factor for a single `mscale` value.
/// Matches HF `yarn_get_mscale`: `0.1 * mscale * ln(scale) + 1.0` for
/// `scale > 1`, else `1.0`.
fn yarn_get_mscale(scale: f32, mscale: f32) -> f32 {
    if scale <= 1.0 {
        1.0
    } else {
        0.1 * mscale * scale.ln() + 1.0
    }
}

/// Compute the YaRN `_mscale` ratio that DeepSeek folds into the rope
/// cos/sin: `get_mscale(factor, mscale) / get_mscale(factor, mscale_all_dim)`.
/// Returns 1.0 when YaRN is disabled (`yarn_factor <= 1`).
pub(crate) fn yarn_rope_mscale(config: &atlas_core::config::ModelConfig) -> f32 {
    let factor = config.yarn_factor;
    if factor <= 1.0 {
        return 1.0;
    }
    let num = yarn_get_mscale(factor, config.yarn_mscale);
    let den = yarn_get_mscale(factor, config.yarn_mscale_all_dim);
    num / den
}

impl Qwen3AttentionLayer {
    /// Set MLA weights for 2-step latent decode. When set, decode uses
    /// latent→norm→expand instead of single-step GEMV.
    pub fn set_mla_weights(&mut self, mla: MlaWeights) {
        self.mla = Some(mla);
    }

    /// Set per-block Manifold-Constrained Hyper-Connection weights
    /// (DeepSeek-V4). When set, the attn/ffn residual sites route through
    /// `hc_pre`/`hc_post` against the model-level `hc_streams` buffer.
    pub fn set_hc_weights(&mut self, hc: HcWeights) {
        self.hc = Some(hc);
    }

    /// Set per-layer dimension overrides for heterogeneous models (Gemma-4).
    /// Full-attention layers have different Q/KV head counts and head_dim
    /// than sliding layers.
    pub fn set_dimension_overrides(
        &mut self,
        head_dim: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
    ) {
        self.head_dim_override = Some(head_dim);
        self.num_q_heads_override = Some(num_q_heads);
        self.num_kv_heads_override = Some(num_kv_heads);
    }

    /// Set per-layer sliding-window size (Gemma-4 hybrid attention).
    /// Call with `Some(window_size)` on sliding layers, `None` on
    /// full-attention layers. Non-Gemma-4 models never call this.
    pub fn set_sliding_window(&mut self, window: Option<u32>) {
        self.sliding_window = window;
    }

    /// Set per-head attention gate weight (Step 3.7 g_proj).
    /// The weight is BF16 shape [num_q_heads, hidden_size].
    pub fn set_head_gate_weight(&mut self, w: DenseWeight) {
        self.head_gate_weight = Some(w);
    }

    /// Set per-layer RoPE overrides (theta, rotary_dim) for dual-RoPE
    /// models (Gemma-4).
    pub fn set_rope_overrides(&mut self, theta: f32, rotary_dim: u32) {
        self.rope_theta_override = Some(theta);
        self.rotary_dim_override = Some(rotary_dim);
    }

    /// Enable proportional RoPE (Gemma-4 full-attention layers). Must be
    /// called AFTER `set_rope_overrides`; the `rotary_dim` set there is
    /// reinterpreted as the number of non-zero rotation pairs.
    pub fn set_rope_proportional(&mut self, enable: bool) {
        self.rope_proportional = enable;
    }

    /// Set per-layer attention scale override. Gemma-4 uses QK-norm, so
    /// attention scale should be 1.0 (not 1/sqrt(head_dim)).
    pub fn set_attn_scale_override(&mut self, scale: f32) {
        self.attn_scale_override = Some(scale);
    }

    /// Set K=V mode (Gemma-4 full-attention layers).
    ///
    /// `v_norm_weight` is a BF16 weight buffer of size `[head_dim]`. For
    /// Gemma-4 it's ones-filled because Gemma-4's rms_norm kernel uses
    /// the absolute convention `out = x * rms * weight`, and `weight =
    /// 1.0` gives pure RMSNorm (matching HF
    /// `Gemma4RMSNorm(with_scale=False)`).
    pub fn set_k_eq_v(&mut self, v_norm_weight: DenseWeight) {
        self.k_eq_v = true;
        self.v_norm_weight = Some(v_norm_weight);
    }

    /// Install a pure-RMSNorm v_norm WITHOUT enabling K=V aliasing. Used
    /// for Gemma-4 sliding-attention layers where V_proj exists on disk
    /// but HF `Gemma4TextAttention.forward()` still applies
    /// `value_states = self.v_norm(value_states)` with
    /// `Gemma4RMSNorm(with_scale=False)` — pure `x * rms`.
    pub fn set_v_norm(&mut self, v_norm_weight: DenseWeight) {
        self.v_norm_weight = Some(v_norm_weight);
    }

    /// Install a BF16 dense fallback for the output projection. When
    /// set, decode + prefill skip the NVFP4 `attn.o_proj` path and use
    /// this BF16 dense_gemv / dense_gemm instead. Required for Gemma-4
    /// dense (Nvidia ModelOpt's official ignore list keeps ALL
    /// self_attn projections at BF16).
    pub fn set_o_dense_bf16(&mut self, o_dense: DenseWeight) {
        self.o_dense_bf16 = Some(o_dense);
    }

    /// Set post-sublayer norms (Gemma-4: 4-norm residual structure).
    pub fn set_post_sublayer_norms(
        &mut self,
        post_attn_out: DenseWeight,
        post_ffn_out: DenseWeight,
    ) {
        self.post_attn_out_norm = Some(post_attn_out);
        self.post_ffn_out_norm = Some(post_ffn_out);
    }

    /// Set per-layer scalar (Gemma-4: hidden_states *= scalar at end of
    /// layer).
    pub fn set_layer_scalar(&mut self, scalar: f32) {
        self.layer_scalar = Some(scalar);
    }

    /// Set secondary MoE FFN (Gemma-4 26B dual-FFN: dense + MoE per
    /// layer).
    pub fn set_moe_ffn(
        &mut self,
        ffn: FfnComponent,
        pre_norm: DenseWeight,
        post_norm: DenseWeight,
        post_dense_norm: DenseWeight,
    ) {
        self.moe_ffn = Some(ffn);
        self.pre_moe_norm = Some(pre_norm);
        self.post_moe_out_norm = Some(post_norm);
        self.post_dense_ffn_norm = Some(post_dense_norm);
    }

    /// Apply layer_scalar in-place: `hidden *= scalar`. Uses
    /// `bf16_scale_inplace` for the (always BF16) residual stream.
    pub(crate) fn apply_layer_scalar(
        &self,
        gpu: &dyn spark_runtime::gpu::GpuBackend,
        hidden: spark_runtime::gpu::DevicePtr,
        hidden_size: usize,
        scalar: f32,
        stream: u64,
    ) -> anyhow::Result<()> {
        use spark_runtime::kernel_args::KernelLaunch;
        let scale_k = gpu.kernel("embed_scale", "bf16_scale_inplace")?;
        let n = hidden_size as u32;
        KernelLaunch::new(gpu, scale_k)
            .grid([n.div_ceil(256), 1, 1])
            .block([256, 1, 1])
            .arg_ptr(hidden)
            .arg_u32(n)
            .arg_f32(scalar)
            .launch(stream)
    }

    /// Compute effective attention scale: override if set, else
    /// `1/sqrt(head_dim)`.
    pub(crate) fn effective_attn_scale(&self, head_dim: u32) -> f32 {
        self.attn_scale_override
            .unwrap_or_else(|| 1.0f32 / (head_dim as f32).sqrt())
    }
}

#[cfg(test)]
mod yarn_mscale_tests {
    use super::yarn_rope_mscale;
    use atlas_core::config::ModelConfig;

    // Test 1 + Test 4: with the DS4F-forced config (yarn_mscale ==
    // yarn_mscale_all_dim == 0.0, factor 16), yarn_rope_mscale returns EXACTLY
    // 1.0 — the single value fed to all nine DS4F rope call sites, removing the
    // erroneous 1.2772589 amplitude on CSA/HCA layers.
    #[test]
    fn ds4f_forced_config_yields_mscale_one() {
        let mut c = ModelConfig::qwen3_next_80b_nvfp4();
        c.yarn_factor = 16.0;
        c.yarn_mscale = 0.0;
        c.yarn_mscale_all_dim = 0.0;
        assert_eq!(yarn_rope_mscale(&c), 1.0);
    }

    // Test 5 (helper side): the helper itself is UNCHANGED. Under the generic
    // HF-DeepseekV3 default (mscale 1.0, mscale_all_dim 0.0) it still returns the
    // 1.2772589 ratio, so any legitimate YaRN-mscale caller (a different model
    // whose config sets these fields) is unaffected. Only the DS4F *config* flips
    // the result, not this function.
    #[test]
    fn generic_yarn_default_unchanged_1277() {
        let mut c = ModelConfig::qwen3_next_80b_nvfp4();
        c.yarn_factor = 16.0;
        c.yarn_mscale = 1.0;
        c.yarn_mscale_all_dim = 0.0;
        let m = yarn_rope_mscale(&c);
        assert!((m - 1.2772589).abs() < 1e-5, "expected ~1.2772589, got {m}");
    }

    // YaRN disabled (factor <= 1) short-circuits to 1.0 (unchanged behavior).
    #[test]
    fn yarn_disabled_factor_one_is_mscale_one() {
        let mut c = ModelConfig::qwen3_next_80b_nvfp4();
        c.yarn_factor = 1.0;
        c.yarn_mscale = 1.0;
        c.yarn_mscale_all_dim = 0.0;
        assert_eq!(yarn_rope_mscale(&c), 1.0);
    }
}
