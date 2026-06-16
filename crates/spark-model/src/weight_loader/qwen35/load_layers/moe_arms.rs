// SPDX-License-Identifier: AGPL-3.0-only
//
// Helpers for the per-layer MoE expert loading paths of `load_layers`:
// the ATLAS_FP8_DEQUANT_MOE_TO_BF16 dequant-to-BF16 path and the native
// FP8 expert load path. Both mutate the `MoeLayer` in place.

use atlas_core::config::ModelConfig;
use spark_runtime::gpu::GpuBackend;
use spark_runtime::weights::WeightStore;

use crate::layers::MoeLayer;
use crate::weight_map::load_moe_qwen35_fp8_experts;

/// ATLAS_FP8_DEQUANT_MOE_TO_BF16: dequant FP8 experts to BF16 at load, route
/// MoE through the BF16 grouped GEMM + fused-decode kernels. Eliminates the
/// per-layer 0.989 FP8 cosine ceiling. Memory cost: ~2× expert weights vs
/// native FP8.
pub(super) fn dequant_moe_experts_to_bf16(
    layer_idx: usize,
    store: &WeightStore,
    lp: &str,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    moe_layer: &mut MoeLayer,
) {
    let i = layer_idx;
    use crate::weight_map::dequant_fp8_blockscaled_to_bf16;
    let p = format!("{lp}.mlp");
    let mut gate_bf16 = Vec::with_capacity(config.num_experts);
    let mut up_bf16 = Vec::with_capacity(config.num_experts);
    let mut down_bf16 = Vec::with_capacity(config.num_experts);
    let mut load_err: Option<anyhow::Error> = None;
    // Free FP8 source GPU memory after each successful dequant.
    // The HashMap entry retains a stale ptr; nothing else reads
    // these expert weights after dequant on the BF16 path, so
    // the orphan key is benign.
    let free_src = |prefix: &str| {
        for suffix in ["weight", "weight_scale_inv"] {
            let k = format!("{prefix}.{suffix}");
            if let Ok(w) = store.get(&k) {
                let _ = gpu.free(w.ptr);
            }
        }
    };
    for e in 0..config.num_experts {
        let ep = format!("{p}.experts.{e}");
        let gate_key = format!("{ep}.gate_proj");
        let up_key = format!("{ep}.up_proj");
        let down_key = format!("{ep}.down_proj");
        let g = dequant_fp8_blockscaled_to_bf16(store, &gate_key, gpu);
        let u = dequant_fp8_blockscaled_to_bf16(store, &up_key, gpu);
        let d = dequant_fp8_blockscaled_to_bf16(store, &down_key, gpu);
        match (g, u, d) {
            (Ok(g), Ok(u), Ok(d)) => {
                gate_bf16.push(g);
                up_bf16.push(u);
                down_bf16.push(d);
                free_src(&gate_key);
                free_src(&up_key);
                free_src(&down_key);
            }
            (g, u, d) => {
                load_err = Some(anyhow::anyhow!(
                    "Layer {i} expert {e}: BF16 dequant failed (gate_ok={}, up_ok={}, down_ok={})",
                    g.is_ok(),
                    u.is_ok(),
                    d.is_ok(),
                ));
                break;
            }
        }
    }
    // Shared expert (Qwen3.6 ships one).
    let sp = format!("{p}.shared_expert");
    let sh_gate_key = format!("{sp}.gate_proj");
    let sh_up_key = format!("{sp}.up_proj");
    let sh_down_key = format!("{sp}.down_proj");
    let sh_g = dequant_fp8_blockscaled_to_bf16(store, &sh_gate_key, gpu).ok();
    let sh_u = dequant_fp8_blockscaled_to_bf16(store, &sh_up_key, gpu).ok();
    let sh_d = dequant_fp8_blockscaled_to_bf16(store, &sh_down_key, gpu).ok();
    if sh_g.is_some() {
        free_src(&sh_gate_key);
    }
    if sh_u.is_some() {
        free_src(&sh_up_key);
    }
    if sh_d.is_some() {
        free_src(&sh_down_key);
    }
    let sh_g_ptr = sh_g
        .map(|w| w.weight)
        .unwrap_or(spark_runtime::gpu::DevicePtr::NULL);
    let sh_u_ptr = sh_u
        .map(|w| w.weight)
        .unwrap_or(spark_runtime::gpu::DevicePtr::NULL);
    let sh_d_ptr = sh_d
        .map(|w| w.weight)
        .unwrap_or(spark_runtime::gpu::DevicePtr::NULL);
    match load_err {
        Some(e) => {
            tracing::error!("Layer {i}: dequant-to-BF16 MoE load failed: {e:#}");
            tracing::warn!("Layer {i}: falling back to native FP8 MoE");
        }
        None => {
            if let Err(e) = moe_layer.set_bf16_experts(
                &gate_bf16, &up_bf16, &down_bf16, sh_g_ptr, sh_u_ptr, sh_d_ptr, gpu,
            ) {
                tracing::error!("Layer {i}: failed to build BF16 expert pointer tables: {e:#}");
            } else {
                tracing::info!(
                    "Layer {i}: MoE experts dequanted FP8→BF16 ({} routed + 1 shared)",
                    config.num_experts
                );
            }
        }
    }
}

/// Native FP8 MoE: load FP8 expert weights for decode. Returns `true` if the
/// FP8 experts were loaded (so the caller can skip further attempts).
pub(super) fn load_fp8_moe_experts(
    layer_idx: usize,
    store: &WeightStore,
    lp: &str,
    config: &ModelConfig,
    gpu: &dyn GpuBackend,
    moe_layer: &mut MoeLayer,
) -> bool {
    let i = layer_idx;
    let Ok(fp8_experts) = load_moe_qwen35_fp8_experts(store, lp, config.num_experts, gpu, config)
    else {
        return false;
    };
    let sp = format!("{lp}.mlp.shared_expert");
    use crate::weight_map::{
        Fp8ExpertWeight as FEW, Fp8Weight as FW, load_fp8_block_scaled_as_fp8weight,
    };
    use spark_runtime::gpu::DevicePtr;
    let null_fw = FW {
        weight: DevicePtr::NULL,
        row_scale: DevicePtr::NULL,
        n: 0,
        k: 0,
        // Placeholder for absent shared-expert tensor: the
        // calling site checks `weight == NULL` before
        // launching any kernel, so the tag is conventional.
        // Match the block-scaled FP8 loader the other arms
        // use so the format is consistent.
        scale_format: crate::weight_map::WeightQuantFormat::Fp8BlockScaled,
    };
    let sh_gate = load_fp8_block_scaled_as_fp8weight(store, &format!("{sp}.gate_proj"), gpu);
    let sh_up = load_fp8_block_scaled_as_fp8weight(store, &format!("{sp}.up_proj"), gpu);
    let sh_down = load_fp8_block_scaled_as_fp8weight(store, &format!("{sp}.down_proj"), gpu);
    if sh_gate.is_err() || sh_up.is_err() || sh_down.is_err() {
        tracing::warn!(
            "Layer {i}: shared expert FP8 load failed (gate={}, up={}, down={})",
            sh_gate.is_ok(),
            sh_up.is_ok(),
            sh_down.is_ok(),
        );
    }
    let shared_fp8 = FEW {
        gate_proj: sh_gate.unwrap_or(null_fw),
        up_proj: sh_up.unwrap_or(null_fw),
        down_proj: sh_down.unwrap_or(null_fw),
    };
    if let Err(e) = moe_layer.set_fp8_experts(&fp8_experts, shared_fp8, gpu) {
        tracing::error!("Layer {i}: failed to build FP8 expert pointer tables: {e:#}");
        tracing::warn!("Layer {i}: falling back to NVFP4-only decode for MoE experts");
    } else {
        tracing::info!("Layer {i}: MoE experts loaded as native FP8");
    }
    true
}
