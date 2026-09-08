// SPDX-License-Identifier: AGPL-3.0-only

//! GLM-5.3's MTP block — `model.language_model.layers.45` — loaded as a draft proposer body.
//!
//! 🪤 GLM does NOT use DeepSeek's `mtp.0.*` naming, so `grep mtp` over the checkpoint finds
//! nothing. The block sits one past the 45-layer text stack, and `prune_after_load`
//! deliberately keeps it: its routed experts are already resident and
//! [`super::glm5_next_load`]'s expert binder references their store pointers, so binding this
//! layer moves almost no bytes.
//!
//! # What shape it is
//!
//! A DSA mixer + a routed MoE over the same 288 experts + `shared_head.norm`, wrapped by the
//! usual MTP front end: `eh_proj(concat(enorm(embed(t)), hnorm(h)))`. It carries **no `hc_*`
//! tensors**, so unlike every text layer it is a plain pre-norm residual block —
//! [`crate::layers::glm5next_layer::Glm5NextLayer`]'s `mhc: None` path.
//!
//! # Why it takes its own KV cache
//!
//! The target's KV pool is sized to `num_attention_layers()` (11 on GLM-5.3), addressed by a
//! DSA layer's ordinal among KV-consuming layers. The drafter is a 12th consumer whose entries
//! must be trimmable independently of the target's, so it gets a one-layer pool of its own and
//! `attn_layer_idx = 0` within it.

use anyhow::{Context, Result};
use atlas_core::config::ModelConfig;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::WeightStore;

use crate::layers::glm5next_dsa::build::build_dsa_weights;
use crate::layers::glm5next_dsa::layer::{
    Glm5NextDsaLayer, Glm5NextDsaLayerKernels, Glm5NextDsaWorkspace,
};
use crate::layers::glm5next_dsa::{Glm5NextDsaConfig, Glm5NextDsaKernels};
use crate::layers::glm5next_layer::{Glm5NextLayer, Glm5NextMixer, Glm5NextMlpSite};
use crate::layers::glm5next_mlp::{Glm5NextMlpConfig, Glm5NextMlpKernels, build as mlp_build};
use crate::weight_map::DenseWeight;

/// The MTP block plus the four tensors that sit around it.
pub struct Glm5NextMtpModule {
    /// `layers.45` as a `mhc: None` layer — DSA mixer, routed MoE, plain residual.
    pub layer: Glm5NextLayer,
    /// `[hidden, 2 * hidden]` BF16: the concat projection.
    pub eh_proj: DenseWeight,
    /// RMSNorm weights for the embedding half and the target-hidden half of the concat.
    pub enorm: DevicePtr,
    pub hnorm: DevicePtr,
    /// `shared_head.norm` — the final norm before the SHARED `lm_head`.
    pub final_norm: DevicePtr,
}

/// Build the MTP block, or `None` when the checkpoint has no `layers.{num_hidden_layers}`.
///
/// Runs on EVERY rank, unlike DeepSeek-V4's rank-0-local drafter: GLM's MTP MoE is the same
/// 288-expert layout as the text stack and is EP-sharded, so both ranks hold a half and the
/// block's own all-reduce assembles it. A rank-0-only drafter would silently drop half the
/// routed sum.
pub fn load_glm5next_mtp_module(
    store: &WeightStore,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
) -> Result<Option<Glm5NextMtpModule>> {
    let idx = config.num_hidden_layers;
    let prefix = format!("model.language_model.layers.{idx}.");
    if !store.names().any(|n| n.starts_with(&prefix)) {
        return Ok(None);
    }
    let src = super::glm5_next_load::layer_source(gpu, store, idx)
        .with_context(|| format!("glm5_next MTP: collecting layer {idx}"))?;
    let load = |n: &str| src.f32(n);

    let dsa_cfg = Glm5NextDsaConfig::from_config(config)?;
    let mlp_cfg = Glm5NextMlpConfig::from_config(config)?;
    let dsa_plan = crate::layers::glm5next_dsa::tp::DsaTpPlan::new(
        config.tp_rank,
        config.tp_world_size.max(1),
        &dsa_cfg,
    )?;
    let dsa_layer_kernels = Glm5NextDsaLayerKernels::resolve(gpu)?;
    let dsa_kernels = Glm5NextDsaKernels::resolve(gpu)?;
    let mlp_kernels = Glm5NextMlpKernels::resolve(gpu)?;

    let mixer = Glm5NextMixer::Dsa(Box::new(Glm5NextDsaLayer {
        persist_bt: std::env::var("ATLAS_GLM_DSA_ALLOC_PER_STEP").as_deref() != Ok("1"),
        cfg: dsa_cfg,
        weights: build_dsa_weights(gpu, &dsa_cfg, &dsa_plan, &load)?,
        kernels: dsa_layer_kernels,
        select_kernels: dsa_kernels,
        decode_kernel: crate::layers::glm5next_dsa::attend::Glm5NextDsaDecodeKernel::resolve(gpu)?,
        // The drafter proposes ONE token at a time, so a single-row workspace is the whole
        // requirement; a verify never runs this layer.
        workspace: Glm5NextDsaWorkspace::new(gpu, &dsa_cfg, 1)?,
        layer_idx: idx,
        // Sole consumer of its own one-layer KV pool.
        attn_layer_idx: 0,
        rms_eps: config.rms_norm_eps as f32,
        kv_scale: 1.0,
    }));

    let expert = |id: usize| super::glm5_next_load::bind_expert_at(gpu, store, idx, id);
    // The draft block's routed experts are packed too on an EXL3 checkpoint.
    // The earlier assumption here was that `scope: glm53_routed_experts_only`
    // meant the MAIN layers only and the draft block stayed NVFP4 — measured
    // false on GLM-5.3-Flash-EXL3-K2, whose `.trellis` tensors span layers
    // 3..=45 with 864 of them (288 experts x 3 projections) on layer 45 alone.
    //
    // Without this arm `--speculative` died on
    // `layers.45.mlp.experts.144.gate_proj.weight` — the BF16 name at rank 1's
    // EP-local start — and speculative decoding was DISABLED while the server
    // came up serial. An MTP benchmark would then have measured the serial path
    // and reported it as MTP.
    //
    // Probe an expert THIS rank owns: under EP the store is sharded and expert
    // 0 lives only on rank 0 (see bf9e8b23d).
    let q = |leaf: &str| super::glm5_next_load::qualify_at(idx, leaf);
    let local = mlp_cfg.local_expert_range();
    let exl3 = if super::glm5_next_exl3::layer_is_exl3(store, &q, local.start) {
        Some(super::glm5_next_exl3::bind_experts_exl3(
            gpu,
            store,
            &q,
            mlp_cfg.num_experts,
            (local.start, local.end),
            (mlp_cfg.hidden, mlp_cfg.moe_intermediate, mlp_cfg.top_k),
        )?)
    } else {
        None
    };
    let mlp = Glm5NextMlpSite::Moe(Box::new(mlp_build::build_moe(
        gpu,
        &mlp_cfg,
        config.tp_rank,
        config.shared_expert_intermediate_size,
        &load,
        &expert,
        exl3,
    )?));

    let up =
        |n: &str| -> Result<DevicePtr> { super::glm5_next_load::upload_bf16(gpu, &src.f32(n)?) };
    Ok(Some(Glm5NextMtpModule {
        layer: Glm5NextLayer {
            layer_idx: idx,
            mixer,
            mlp,
            mlp_cfg,
            mlp_kernels,
            mlp_ws: crate::layers::glm5next_mlp::forward::Glm5NextMlpWorkspace::new(
                gpu, &mlp_cfg, 1,
            )?,
            // 🔴 The one GLM-5.3 block with no hyper-connection. The checkpoint carries zero
            // `hc_*` tensors here, and `forward_one` takes its plain residual path.
            mhc: None,
            input_norm: up("input_layernorm.weight")?,
            post_attn_norm: up("post_attention_layernorm.weight")?,
            rms_norm_k: gpu.kernel("rms_norm_vanilla", "rms_norm_vanilla")?,
            add_k: crate::layers::try_kernel(gpu, "bf16_add", "bf16_add_inplace"),
            rms_eps: config.rms_norm_eps as f32,
            hidden: config.hidden_size,
            // Row-parallel `o_proj`, same as every DSA text layer.
            mixer_all_reduce: dsa_plan.needs_output_all_reduce(),
            // Neither: the highway does not exist here.
            is_first: false,
            is_last: false,
        },
        eh_proj: DenseWeight {
            weight: store.get(&format!("{prefix}eh_proj.weight"))?.ptr,
        },
        enorm: up("enorm.weight")?,
        hnorm: up("hnorm.weight")?,
        final_norm: up("shared_head.norm.weight")?,
    }))
}
