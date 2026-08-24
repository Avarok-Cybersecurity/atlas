// SPDX-License-Identifier: AGPL-3.0-only

use std::collections::HashMap;

use atlas_core::config::ModelConfig;
use spark_runtime::gpu::DevicePtr;
use spark_runtime::weights::{WeightDtype, WeightStore, WeightTensor};

use super::{discover_dspark_stages, expect_tensor, native_dspark_config};

fn store(entries: &[(&str, WeightDtype, &[usize])]) -> WeightStore {
    let weights = entries
        .iter()
        .map(|(name, dtype, shape)| {
            (
                (*name).to_string(),
                WeightTensor {
                    ptr: DevicePtr::NULL,
                    shape: shape.to_vec(),
                    dtype: *dtype,
                },
            )
        })
        .collect::<HashMap<_, _>>();
    WeightStore::from_map(weights)
}

#[test]
fn stage_discovery_is_exact_and_ordered() {
    let store = store(&[
        ("mtp.2.norm.weight", WeightDtype::BF16, &[1]),
        ("mtp.0.main_norm.weight", WeightDtype::BF16, &[1]),
        ("layers.0.attn_norm.weight", WeightDtype::BF16, &[1]),
        ("mtp.1.ffn_norm.weight", WeightDtype::BF16, &[1]),
        ("xmtp.9.ffn_norm.weight", WeightDtype::BF16, &[1]),
        ("mtp.bad.ffn_norm.weight", WeightDtype::BF16, &[1]),
        ("mtp.-1.ffn_norm.weight", WeightDtype::BF16, &[1]),
    ]);
    assert_eq!(
        discover_dspark_stages(&store),
        std::collections::BTreeSet::from([0, 1, 2])
    );
}

#[test]
fn tensor_contract_rejects_e8m0_where_fp8_weight_is_required() {
    let store = store(&[(
        "mtp.0.main_proj.weight",
        WeightDtype::FP8E8M0,
        &[4096, 12288],
    )]);
    let err = expect_tensor(
        &store,
        "mtp.0.main_proj.weight",
        WeightDtype::FP8E4M3,
        &[4096, 12288],
    )
    .expect_err("wrong dtype must fail closed");
    let msg = err.to_string();
    assert!(msg.contains("mtp.0.main_proj.weight"));
    assert!(msg.contains("FP8E8M0"));
    assert!(msg.contains("expected FP8E4M3"));
}

#[test]
fn tensor_contract_rejects_wrong_scale_shape() {
    let store = store(&[(
        "mtp.0.ffn.experts.0.w1.scale",
        WeightDtype::FP8E8M0,
        &[2048, 64],
    )]);
    let err = expect_tensor(
        &store,
        "mtp.0.ffn.experts.0.w1.scale",
        WeightDtype::FP8E8M0,
        &[2048, 128],
    )
    .expect_err("wrong scale geometry must fail closed");
    let msg = err.to_string();
    assert!(msg.contains("mtp.0.ffn.experts.0.w1.scale"));
    assert!(msg.contains("[2048, 64]"));
    assert!(msg.contains("expected [2048, 128]"));
}

#[test]
fn tensor_contract_rejects_missing_required_tensor() {
    let err = expect_tensor(
        &store(&[]),
        "mtp.2.confidence_head.proj.weight",
        WeightDtype::BF16,
        &[1, 4096],
    )
    .expect_err("missing tensor must fail closed");
    assert!(
        err.to_string()
            .contains("missing required tensor `mtp.2.confidence_head.proj.weight`")
    );
}

#[test]
fn pure_tp_draft_contract_restores_full_width_and_partitions_experts() {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.model_type = "deepseek_v4".to_string();
    config.moe_intermediate_size = 1024;
    config.shared_expert_intermediate_size = 1024;
    config.tp_rank = 1;
    config.tp_world_size = 2;
    config.ep_rank = 0;
    config.ep_world_size = 1;

    let draft = native_dspark_config(&config).unwrap();
    assert_eq!(draft.moe_intermediate_size, 2048);
    assert_eq!(draft.shared_expert_intermediate_size, 2048);
    assert_eq!((draft.tp_rank, draft.tp_world_size), (1, 2));
    assert_eq!((draft.ep_rank, draft.ep_world_size), (1, 2));
    assert_eq!(draft.local_expert_range(), (256, 512));
}

#[test]
fn pure_tp_draft_contract_rejects_invalid_or_overflowing_geometry() {
    let mut config = ModelConfig::qwen3_next_80b_nvfp4();
    config.tp_world_size = 2;
    config.tp_rank = 2;
    config.ep_world_size = 1;
    let err = native_dspark_config(&config).expect_err("rank outside world must reject");
    assert!(
        err.to_string()
            .contains("TP rank 2 is outside world size 2")
    );

    config.tp_rank = 0;
    config.moe_intermediate_size = usize::MAX;
    let err = native_dspark_config(&config).expect_err("routed width overflow must reject");
    assert!(err.to_string().contains("routed width overflow"));
}
