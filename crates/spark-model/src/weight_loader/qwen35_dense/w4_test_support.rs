// SPDX-License-Identifier: AGPL-3.0-only

//! Responsibility: shared fixtures of lane L23b's 4-bit loader tests: the Qwen3.5 4B and 0.8B dense
//! geometries, mock-backend store builders, and a lever value. Test-only (`#[cfg(test)]` in the parent).

use std::collections::HashMap;

use avarok_core::config::{LayerType, ModelConfig};
use spark_runtime::gpu::GpuBackend;
use spark_runtime::gpu::mock::MockGpuBackend;
use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};

use crate::layers::ops::target_defaults::{Resolved, Source};

/// A resolved lever from the target's declaration.
pub(crate) fn lever(on: bool) -> Resolved<bool> {
    Resolved {
        value: on,
        source: Source::Target,
    }
}

fn dense_qwen35(layers: usize, hidden: usize, inter: usize) -> ModelConfig {
    let mut c = ModelConfig::qwen3_next_80b_nvfp4();
    c.model_type = "qwen3_5".to_string();
    c.num_experts = 0;
    c.num_experts_per_tok = 0;
    c.moe_intermediate_size = 0;
    c.hidden_size = hidden;
    c.intermediate_size = inter;
    c.num_hidden_layers = layers;
    c.full_attention_interval = 4;
    c.layer_types = (0..layers)
        .map(|i| {
            if (i + 1) % 4 == 0 {
                LayerType::FullAttention
            } else {
                LayerType::LinearAttention
            }
        })
        .collect();
    c.quantization_config = None;
    c.tp_world_size = 1;
    c.adapter_max_rank = 0;
    c
}

/// Qwen3.5-4B / Holo 3.1 4B: 32 layers (24 GDN, 8 attention), hidden 2560, FFN 9216, GDN 16x128 key and
/// 32x128 value heads (`[Q|K|V|Z]` 12,288 rows, value 4,096), attention 16 heads x 256 (gated q: 8,192
/// rows), 4 KV heads.
pub(crate) fn qwen35_4b() -> ModelConfig {
    let mut c = dense_qwen35(32, 2560, 9216);
    c.num_attention_heads = 16;
    c.num_key_value_heads = 4;
    c.head_dim = 256;
    c.attn_gated = true;
    c.linear_num_key_heads = 16;
    c.linear_key_head_dim = 128;
    c.linear_num_value_heads = 32;
    c.linear_value_head_dim = 128;
    c
}

/// Qwen3.5-0.8B / Holo 3.1 0.8B: 24 layers (18 GDN, 6 attention), hidden 1024, FFN 3584, GDN 16x128 key and
/// 16x128 value heads (`[Q|K|V|Z]` 8,192 rows, value 2,048), attention 8 heads x 256 (gated), 2 KV heads.
pub(crate) fn qwen35_0p8b() -> ModelConfig {
    let mut c = dense_qwen35(24, 1024, 3584);
    c.num_attention_heads = 8;
    c.num_key_value_heads = 2;
    c.head_dim = 256;
    c.attn_gated = true;
    c.linear_num_key_heads = 16;
    c.linear_key_head_dim = 128;
    c.linear_num_value_heads = 16;
    c.linear_value_head_dim = 128;
    c
}

/// One `[n, k]` tensor of `dtype` on the mock backend.
pub(crate) fn tensor(gpu: &MockGpuBackend, shape: [usize; 2], dtype: WeightDtype) -> WeightTensor {
    let per = match dtype {
        WeightDtype::BF16 => 2,
        WeightDtype::FP32 => 4,
        _ => 1,
    };
    WeightTensor {
        ptr: gpu
            .alloc((shape[0] * shape[1] * per).max(1))
            .expect("mock alloc"),
        shape: shape.to_vec(),
        dtype,
    }
}

/// The FFN of layers `0..layers` on a store: `[inter, hidden]` gate/up and `[hidden, inter]` down.
pub(crate) fn ffn_store(
    gpu: &MockGpuBackend,
    layers: usize,
    hidden: usize,
    inter: usize,
) -> WeightStore {
    let mut m = HashMap::new();
    for l in 0..layers {
        let lp = format!("model.layers.{l}");
        for (name, shape) in [
            ("gate_proj", [inter, hidden]),
            ("up_proj", [inter, hidden]),
            ("down_proj", [hidden, inter]),
        ] {
            m.insert(
                format!("{lp}.mlp.{name}.weight"),
                tensor(gpu, shape, WeightDtype::BF16),
            );
        }
    }
    WeightStore::from_map(m)
}

/// Device bytes still allocated on the mock at `ptr` (`None` once freed).
pub(crate) fn live(gpu: &MockGpuBackend, ptr: spark_runtime::gpu::DevicePtr) -> Option<usize> {
    gpu.read_alloc(ptr).map(|d| d.len())
}
