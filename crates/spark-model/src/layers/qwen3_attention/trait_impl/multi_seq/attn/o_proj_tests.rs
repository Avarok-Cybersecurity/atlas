// SPDX-License-Identifier: AGPL-3.0-only

use super::super::super::ctx::MultiSeqCtx;
use crate::layer::{ForwardContext, MoeLoraRoute};
use crate::layers::ops::{DerivedWeights, GemmDispatch, ModelLevers, ModelStats};
use crate::layers::{FfnComponent, qwen3_attention::Qwen3AttentionLayer};
use crate::weight_map::{
    AttentionWeights, DenseWeight, Fp8Weight, QuantWeight, QuantizedWeight, WeightQuantFormat,
};
use atlas_core::config::ModelConfig;
use spark_runtime::buffers::BufferArena;
use spark_runtime::gpu::mock::{MockArg, MockGpuBackend};
use spark_runtime::gpu::{GpuBackend, KernelHandle};
use spark_runtime::kv_cache::KvCacheDtype;

#[test]
fn native_fp8_attention_o_projection_batches_four_real_rows() {
    check_dispatch(4, 128, true, WeightQuantFormat::Fp8BlockScaled, true);
}

fn check_dispatch(
    rows: usize,
    width: usize,
    available: bool,
    format: WeightQuantFormat,
    batched: bool,
) {
    let gpu = MockGpuBackend::new();
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.hidden_size = width;
    config.intermediate_size = 128;
    config.num_attention_heads = 1;
    config.num_key_value_heads = 1;
    config.head_dim = width;
    config.num_experts = 1;
    config.num_experts_per_tok = 1;
    config.moe_intermediate_size = 128;
    config.vocab_size = 128;
    let buffers = BufferArena::new(&config, 8, 16, 16, 8, &gpu).unwrap();
    let dense = DenseWeight {
        weight: gpu.alloc(128 * 128 * 2).unwrap(),
    };
    let fallback = QuantizedWeight::null();
    let attn = AttentionWeights {
        q_proj: dense,
        k_proj: dense,
        v_proj: dense,
        o_proj: fallback,
        q_norm: dense,
        k_norm: dense,
        q_norm_full: None,
        k_norm_full: None,
        k_scale: 1.0,
        v_scale: 1.0,
    };
    let mut layer = Qwen3AttentionLayer::new(
        dense,
        attn,
        dense,
        FfnComponent::None,
        0,
        None,
        None,
        None,
        &gpu,
        KvCacheDtype::Bf16,
        0,
        &config,
    )
    .unwrap();
    layer.w8a16_gemv_k = KernelHandle(0xF081);
    layer.w8a16_gemv_batch4_k = KernelHandle(if available { 0xF084 } else { 0 });
    let fp8 = Fp8Weight {
        weight: gpu.alloc(128 * 128).unwrap(),
        row_scale: gpu.alloc(4).unwrap(),
        n: 128,
        k: 128,
        scale_format: format,
    };
    layer.o_weight = Some(QuantWeight::Fp8(fp8));
    let dispatch = GemmDispatch::defaults();
    let derived = DerivedWeights::new();
    let levers = ModelLevers::defaults();
    let stats = ModelStats::new();
    let fwd = ForwardContext {
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
        gdn_exact_replay: false,
        token_ids: None,
        host_token_ids: None,
        routed_lora_layers: None,
        midchunk_capture: None,
        moe_lora_route: MoeLoraRoute::Fold,
    };
    let c = MultiSeqCtx::new(
        &layer,
        &fwd,
        buffers.hidden_states(),
        buffers.residual(),
        rows,
        16,
        0,
    );
    let first = gpu.launch_count();
    let allocations = gpu.alloc_count();
    let output = layer.ms_phase_o_proj(&c, buffers.attn_output()).unwrap();
    let all = gpu.launches_snapshot();
    let launches: Vec<_> = all[first..]
        .iter()
        .filter(|l| l.args.contains(&MockArg::Buffer(fp8.weight)))
        .collect();
    let step = if batched { 4 } else { 1 };
    assert_eq!(
        launches.len(),
        rows.div_ceil(step),
        "production O-projection dispatch"
    );
    assert_eq!(
        gpu.alloc_count(),
        allocations,
        "projection must reuse existing buffers"
    );
    for (group, launch) in launches.iter().enumerate() {
        let row = group * step;
        assert_eq!(launch.func, if batched { 0xF084 } else { 0xF081 });
        assert_eq!(
            launch.args[0],
            MockArg::Buffer(buffers.attn_output().offset(row * width * 2))
        );
        assert_eq!(launch.args[1], MockArg::Buffer(fp8.weight));
        assert_eq!(launch.args[2], MockArg::Buffer(fp8.row_scale));
        assert_eq!(
            launch.args[3],
            MockArg::Buffer(output.offset(row * width * 2))
        );
        if batched {
            assert_eq!(
                launch.args[4],
                MockArg::Bytes(((rows - row).min(4) as u32).to_ne_bytes().to_vec())
            );
        }
    }
}

#[test]
fn native_fp8_attention_o_projection_chunks_preserve_offsets() {
    for rows in [2, 5, 16] {
        check_dispatch(rows, 128, true, WeightQuantFormat::Fp8BlockScaled, true);
    }
}

#[test]
fn native_fp8_attention_o_projection_retains_scalar_fallbacks() {
    check_dispatch(1, 128, true, WeightQuantFormat::Fp8BlockScaled, false);
    check_dispatch(4, 128, false, WeightQuantFormat::Fp8BlockScaled, false);
    check_dispatch(4, 64, true, WeightQuantFormat::Fp8BlockScaled, false);
    check_dispatch(4, 128, true, WeightQuantFormat::Fp8PerRow, false);
}
