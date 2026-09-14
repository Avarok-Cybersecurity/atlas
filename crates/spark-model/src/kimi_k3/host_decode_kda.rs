// SPDX-License-Identifier: AGPL-3.0-only

//! BoundLayer mixer CUDA flags: KDA unless `want_cuda_kda` is false; MLA
//! only when `want_cuda_mla` is true. Packed LatentMoE launches E8M0 GEMM.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use atlas_core::config::ModelConfig;
use atlas_core::kimi_k3::{K3CpuModel, MixerKind, MlpKind};
use half::bf16;
use parking_lot::Mutex;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::WeightDtype;

use super::bound::{K3BoundLayer, K3HostShared};
use super::kda_cuda::{CONV_ENTRY, MODULE, RECURRENT_ENTRY};
use super::mla_cuda::{MODULE as MLA_MODULE, ROPE_ENTRY, SDPA_ENTRY};
use super::moe_cuda::{E8M0_ENTRY, MODULE as MOE_MODULE};
use super::state::K3CpuFallbackState;
use crate::layer::{ForwardContext, MoeLoraRoute};
use crate::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};
use crate::weight_map::{DenseWeight, QuantizedWeight};

fn tiny_config(hidden: usize, eps: f32, theta: f32, inter: usize) -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.hidden_size = hidden;
    c.intermediate_size = inter;
    c.vocab_size = 32;
    c.num_experts = 1;
    c.num_hidden_layers = 8;
    c.rms_norm_eps = eps as f64;
    c.rope_theta = theta as f64;
    c.linear_num_key_heads = 1;
    c.linear_key_head_dim = 2;
    c.linear_num_value_heads = 1;
    c.linear_value_head_dim = 2;
    c.num_attention_heads = 1;
    c.head_dim = 4;
    c
}

fn dummy_qw(gpu: &MockGpuBackend) -> QuantizedWeight {
    QuantizedWeight {
        weight: gpu.alloc(8).unwrap(),
        weight_scale: gpu.alloc(8).unwrap(),
        weight_scale_2: 1.0,
        input_scale: DevicePtr::NULL,
        weight_scale_2_vec: DevicePtr::NULL,
    }
}

fn packed_for_layer(
    gpu: &MockGpuBackend,
    layer: usize,
    n_experts: usize,
) -> Vec<(String, QuantizedWeight)> {
    let mut v = Vec::new();
    for e in 0..n_experts {
        for w in ["w1", "w2", "w3"] {
            v.push((
                format!("model.layers.{layer}.block_sparse_moe.experts.{e}.{w}"),
                dummy_qw(gpu),
            ));
        }
    }
    v
}

fn run_layers(steps: &[(usize, bool, bool)]) -> (usize, Vec<(String, String)>) {
    run_layers_try(steps, None, false).unwrap()
}

fn run_layers_inner(
    steps: &[(usize, bool, bool)],
    packed_layer: Option<usize>,
) -> (usize, Vec<(String, String)>) {
    run_layers_try(steps, packed_layer, false).unwrap()
}

fn run_layers_try(
    steps: &[(usize, bool, bool)],
    packed_layer: Option<usize>,
    deny_moe: bool,
) -> anyhow::Result<(usize, Vec<(String, String)>)> {
    let gpu = MockGpuBackend::new();
    if deny_moe {
        gpu.deny_kernel(MOE_MODULE, E8M0_ENTRY);
    }
    let model = K3CpuModel::synthetic_tiny();
    let h = model.graph.hidden;
    let config = tiny_config(h, model.eps, model.rope_theta, model.dense_intermediate);
    let dummy = gpu.alloc((h * 2).max(1)).unwrap();
    let shared = Arc::new(K3HostShared {
        config: config.clone(),
        graph: model.graph.clone(),
        kda: model.kda,
        mla: model.mla,
        moe: model.moe,
        output_res_proj: DenseWeight { weight: dummy },
        output_res_norm: DenseWeight { weight: dummy },
        output_res_proj_meta: (WeightDtype::BF16, h),
        output_res_norm_meta: (WeightDtype::BF16, h),
        output_host: OnceLock::new(),
        kda_kernels: OnceLock::new(),
        mla_kernels: OnceLock::new(),
        moe_kernels: OnceLock::new(),
        attnres: Mutex::new(HashMap::new()),
    });
    let hidden = gpu.alloc(h * 2).unwrap();
    let raw: Vec<u8> = (0..h)
        .flat_map(|i| bf16::from_f32(0.1 * (i as f32 + 1.0)).to_le_bytes())
        .collect();
    gpu.copy_h2d(&raw, hidden).unwrap();
    let buffers = BufferArena::new(&config, 2, 16, 16, 2, &gpu).unwrap();
    let dispatch = GemmDispatch::defaults();
    let derived = DerivedWeights::new();
    let levers = ModelLevers::defaults();
    let stats = ModelStats::new();
    let ctx = ForwardContext {
        buffers: &buffers,
        hc_row_offset: 0,
        gpu: &gpu,
        config: &config,
        dispatch: &dispatch,
        derived: &derived,
        levers: &levers,
        stats: &stats,
        attn_metadata: None,
        profile: false,
        comm: None,
        graph_capture: false,
        decode_step: true,
        gdn_exact_replay: false,
        token_ids: None,
        host_token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
        moe_lora_route: MoeLoraRoute::Fold,
    };
    for &(layer_idx, want_cuda_kda, want_cuda_mla) in steps {
        let host = OnceLock::new();
        let _ = host.set(model.layers[layer_idx].clone());
        let layer = K3BoundLayer {
            index: layer_idx,
            spec: model.layers[layer_idx].spec,
            weights: Vec::new(),
            weight_meta: Vec::new(),
            mxfp4_experts: if packed_layer == Some(layer_idx) {
                packed_for_layer(&gpu, layer_idx, model.moe.n_routed)
            } else {
                Vec::new()
            },
            host,
            shared: shared.clone(),
        };
        let mut state = K3CpuFallbackState {
            cache: match layer.spec.mixer {
                MixerKind::Kda => atlas_core::kimi_k3::LayerCache::Kda(
                    atlas_core::kimi_k3::KdaState::new(&model.kda),
                ),
                MixerKind::Mla => {
                    atlas_core::kimi_k3::LayerCache::Mla(atlas_core::kimi_k3::MlaKv::default())
                }
            },
        };
        layer.decode_host(
            hidden,
            DevicePtr::NULL,
            &mut state,
            0,
            &ctx,
            3,
            want_cuda_kda,
            want_cuda_mla,
        )?;
    }
    Ok((gpu.launch_count(), gpu.kernel_lookups_snapshot()))
}

#[test]
fn kda_layer_cpu_escape_does_not_launch_kda_decode() {
    let model = K3CpuModel::synthetic_tiny();
    assert_eq!(model.layers[0].spec.mixer, MixerKind::Kda);
    let (n, lookups) = run_layers(&[(0, false, false)]);
    assert_eq!(n, 0, "CPU mixer must not launch CUDA KDA");
    assert!(
        lookups.iter().all(|(m, _)| m != MODULE && m != MLA_MODULE),
        "CPU path must not look up CUDA mixers: {lookups:?}"
    );
}

#[test]
fn kda_layer_cuda_flag_launches_conv_then_recurrent() {
    let (n, lookups) = run_layers(&[(0, true, false)]);
    assert_eq!(n, 2, "conv then recurrent");
    assert_eq!(
        lookups,
        vec![
            (MODULE.to_string(), CONV_ENTRY.to_string()),
            (MODULE.to_string(), RECURRENT_ENTRY.to_string()),
        ]
    );
}

#[test]
fn mla_layer_ignores_cuda_kda_flag() {
    let model = K3CpuModel::synthetic_tiny();
    assert_eq!(model.layers[3].spec.mixer, MixerKind::Mla);
    // Layer 0 seeds AttnRes; MLA still must not touch kda_decode.
    let (n, lookups) = run_layers(&[(0, false, false), (3, true, false)]);
    assert_eq!(n, 0, "MLA mixer stays on CPU when CUDA MLA flag is false");
    assert!(
        lookups.iter().all(|(m, _)| m != MODULE && m != MLA_MODULE),
        "MLA must not look up CUDA mixers: {lookups:?}"
    );
}

#[test]
fn kda_layer_ignores_cuda_mla_flag() {
    let (n, lookups) = run_layers(&[(0, false, true)]);
    assert_eq!(n, 0, "KDA mixer must ignore K3_CUDA_MLA");
    assert!(
        lookups.iter().all(|(m, _)| m != MLA_MODULE),
        "KDA must not look up {MLA_MODULE}: {lookups:?}"
    );
}

#[test]
fn mla_layer_cuda_flag_launches_rope_then_sdpa() {
    let model = K3CpuModel::synthetic_tiny();
    assert_eq!(model.layers[3].spec.mixer, MixerKind::Mla);
    let (n, lookups) = run_layers(&[(0, false, false), (3, false, true)]);
    assert_eq!(n, 2, "rope then sdpa_gate");
    assert_eq!(
        lookups,
        vec![
            (MLA_MODULE.to_string(), ROPE_ENTRY.to_string()),
            (MLA_MODULE.to_string(), SDPA_ENTRY.to_string()),
        ]
    );
}

#[test]
fn dense_layer_packed_experts_do_not_launch_gemm() {
    let model = K3CpuModel::synthetic_tiny();
    assert_eq!(model.layers[0].spec.mlp, MlpKind::Dense);
    let (n, lookups) = run_layers_inner(&[(0, false, false)], Some(0));
    assert_eq!(n, 0, "Dense MLP must ignore packed expert tables");
    assert!(
        lookups.iter().all(|(m, _)| m != MOE_MODULE),
        "Dense must not look up {MOE_MODULE}: {lookups:?}"
    );
}

#[test]
fn latent_moe_unpacked_does_not_lookup_gemm() {
    let model = K3CpuModel::synthetic_tiny();
    assert_eq!(model.layers[1].spec.mlp, MlpKind::LatentMoe);
    let (n, lookups) = run_layers(&[(0, false, false), (1, false, false)]);
    assert_eq!(n, 0, "BF16 twin LatentMoE stays host");
    assert!(
        lookups.iter().all(|(m, _)| m != MOE_MODULE),
        "unpacked MoE must not look up {MOE_MODULE}: {lookups:?}"
    );
}

#[test]
fn latent_moe_packed_launches_three_e8m0_gemms() {
    let model = K3CpuModel::synthetic_tiny();
    assert_eq!(model.layers[1].spec.mlp, MlpKind::LatentMoe);
    let (n, lookups) = run_layers_inner(&[(0, false, false), (1, false, false)], Some(1));
    assert_eq!(n, 3, "w1, w3, w2 grouped GEMM");
    assert_eq!(
        lookups,
        vec![(MOE_MODULE.to_string(), E8M0_ENTRY.to_string())]
    );
}

#[test]
fn latent_moe_packed_lookup_fail_bails_not_cpu() {
    let err = run_layers_try(&[(1, false, false)], Some(1), true)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains(E8M0_ENTRY) && err.contains("cannot silently run host F32"),
        "{err}"
    );
}
