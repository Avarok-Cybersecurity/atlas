// SPDX-License-Identifier: AGPL-3.0-only

//! BoundLayer KDA mixer: CUDA `kda_decode` unless `want_cuda_kda` is false.

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use atlas_core::config::ModelConfig;
use atlas_core::kimi_k3::{K3CpuModel, MixerKind};
use half::bf16;
use parking_lot::Mutex;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::gpu::{DevicePtr, GpuBackend};
use spark_runtime::weights::WeightDtype;

use super::bound::{K3BoundLayer, K3HostShared};
use super::kda_cuda::{CONV_ENTRY, MODULE, RECURRENT_ENTRY};
use super::state::K3CpuFallbackState;
use crate::layer::{ForwardContext, MoeLoraRoute};
use crate::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};
use crate::weight_map::DenseWeight;

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

fn run_layers(steps: &[(usize, bool)]) -> (usize, Vec<(String, String)>) {
    let gpu = MockGpuBackend::new();
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
    for &(layer_idx, want_cuda_kda) in steps {
        let host = OnceLock::new();
        let _ = host.set(model.layers[layer_idx].clone());
        let layer = K3BoundLayer {
            index: layer_idx,
            spec: model.layers[layer_idx].spec,
            weights: Vec::new(),
            weight_meta: Vec::new(),
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
        layer
            .decode_host(
                hidden,
                DevicePtr::NULL,
                &mut state,
                0,
                &ctx,
                3,
                want_cuda_kda,
            )
            .unwrap();
    }
    (gpu.launch_count(), gpu.kernel_lookups_snapshot())
}

#[test]
fn kda_layer_cpu_escape_does_not_launch_kda_decode() {
    let model = K3CpuModel::synthetic_tiny();
    assert_eq!(model.layers[0].spec.mixer, MixerKind::Kda);
    let (n, lookups) = run_layers(&[(0, false)]);
    assert_eq!(n, 0, "CPU mixer must not launch CUDA KDA");
    assert!(
        lookups.iter().all(|(m, _)| m != MODULE),
        "CPU path must not look up {MODULE}: {lookups:?}"
    );
}

#[test]
fn kda_layer_cuda_flag_launches_conv_then_recurrent() {
    let (n, lookups) = run_layers(&[(0, true)]);
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
    let (n, lookups) = run_layers(&[(0, false), (3, true)]);
    assert_eq!(n, 0, "MLA mixer stays on CPU");
    assert!(
        lookups.iter().all(|(m, _)| m != MODULE),
        "MLA must not look up {MODULE}: {lookups:?}"
    );
}
